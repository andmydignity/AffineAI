"""
Custom Triton Kernels for Local Predictive Coding (LPC)
========================================================
Implements fused Forward-Predict-Loss-Gradient operations specifically tailored
for in-place, forward-only local error learning:
1. Zero DRAM allocation for un-materialized intermediate logits [B, T, V].
   For V<=1024 falls back to torch allocating [N,V] logits; fusion only for V>1024.
2. Fused online Log-Sum-Exp (LSE) loss calculation in GPU SRAM.
3. In-place gradient projection (dH = err @ W_head) and dW_head accumulation.
4. Seamless integration with standard PyTorch autograd and forward-only execution.
Coalesced access via tl.make_block_ptr with boundary_check for tails.
"""

import warnings

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple, Optional


def _is_turing() -> bool:
    try:
        from affine_ai.kernels import _IS_TURING as _T  # type: ignore

        return bool(_T)
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return (7, 5) <= tuple(cap) < (8, 0)
    except Exception:
        pass
    return False


def _prune_turing_lpc_configs(configs, named_args, **kwargs):
    if _is_turing():
        pruned = [
            c
            for c in configs
            if c.kwargs.get("BLOCK_M", 64) <= 64
            and c.kwargs.get("BLOCK_V", 64) <= 64
            and c.kwargs.get("BLOCK_D", 64) <= 64
            and c.kwargs.get("BLOCK_N", 64) <= 64
        ]
        if pruned:
            return pruned
    return configs


def _use_torch_path(V: int) -> bool:
    return V <= 1024


# ==============================================================================
# Fused LPC Head Forward & Loss Autotuning
# ==============================================================================
lpc_fwd_configs = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_V': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_V': 128, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_V': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_V': 64, 'BLOCK_D': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 16, 'BLOCK_V': 32, 'BLOCK_D': 32}, num_warps=2, num_stages=2),
]

lpc_bwd_dh_configs = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_V': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_V': 128, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_V': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_V': 64, 'BLOCK_D': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 16, 'BLOCK_V': 32, 'BLOCK_D': 32}, num_warps=2, num_stages=2),
]

lpc_bwd_dw_configs = [
    triton.Config({'BLOCK_V': 64, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_V': 64, 'BLOCK_N': 128, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_V': 32, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_V': 32, 'BLOCK_N': 32, 'BLOCK_D': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_V': 32, 'BLOCK_N': 16, 'BLOCK_D': 32}, num_warps=2, num_stages=2),
]



@triton.autotune(
    configs=lpc_fwd_configs,
    key=['D', 'V', 'BLOCK_M'],
    prune_configs_by={'early_config_prune': _prune_turing_lpc_configs},
)
@triton.jit
def _triton_lpc_fwd_kernel(
    H_ptr, W_ptr, Targets_ptr, Loss_ptr, LSE_ptr,
    stride_hb, stride_hd,
    stride_wv, stride_wd,
    stride_tb,
    stride_lb,
    stride_lse,
    ignore_index: tl.constexpr,
    N, D: tl.constexpr, V: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N

    target = tl.load(Targets_ptr + offs_m * stride_tb, mask=mask_m, other=ignore_index)
    valid_mask = mask_m & (target != ignore_index) & (target >= 0) & (target < V)

    # acc_dtype derived from max(H,W): promote bf16/fp16 -> fp32, else fp64 if either is fp64
    acc_dtype = tl.float32 if (H_ptr.dtype.element_ty == tl.bfloat16 or W_ptr.dtype.element_ty == tl.bfloat16 or H_ptr.dtype.element_ty == tl.float16 or W_ptr.dtype.element_ty == tl.float16) else (tl.float64 if (H_ptr.dtype.element_ty == tl.float64 or W_ptr.dtype.element_ty == tl.float64) else tl.float32)

    m_i = tl.full([BLOCK_M], -1e30, dtype=acc_dtype)
    l_i = tl.zeros([BLOCK_M], dtype=acc_dtype)
    target_logit = tl.zeros([BLOCK_M], dtype=acc_dtype)

    for v_start in range(0, V, BLOCK_V):
        offs_v = v_start + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V

        acc = tl.zeros([BLOCK_M, BLOCK_V], dtype=acc_dtype)
        for d_start in range(0, D, BLOCK_D):
            # coalesced block_ptr loads where D is contiguous (stride_hd==1, stride_wd==1) with boundary_check for tails
            H_block = tl.make_block_ptr(base=H_ptr, shape=(N, D), strides=(stride_hb, stride_hd), offsets=(pid_m * BLOCK_M, d_start), block_shape=(BLOCK_M, BLOCK_D), order=(1, 0))
            h_tile = tl.load(H_block, boundary_check=(0, 1))
            W_block = tl.make_block_ptr(base=W_ptr, shape=(V, D), strides=(stride_wv, stride_wd), offsets=(v_start, d_start), block_shape=(BLOCK_V, BLOCK_D), order=(1, 0))
            w_tile = tl.load(W_block, boundary_check=(0, 1))

            acc += tl.dot(h_tile, tl.trans(w_tile), input_precision="ieee")

        acc_masked = tl.where(mask_m[:, None] & mask_v[None, :], acc, -1e30)
        chunk_max = tl.max(acc_masked, axis=1)
        m_new = tl.maximum(m_i, chunk_max)

        alpha = tl.where(m_i > -1e20, tl.exp(m_i - m_new), 0.0)
        diff = tl.where(mask_m[:, None] & mask_v[None, :], acc - m_new[:, None], -50.0)
        p = tl.where(mask_m[:, None] & mask_v[None, :], tl.exp(diff), 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        is_target = (target[:, None] == offs_v[None, :]) & valid_mask[:, None] & mask_v[None, :]
        target_logit += tl.sum(tl.where(is_target, acc, 0.0), axis=1)

        m_i = m_new
        l_i = l_new

    lse = tl.where(valid_mask, m_i + tl.log(tl.maximum(l_i, 1e-12)), 0.0)
    loss = tl.where(valid_mask, lse - target_logit, 0.0)

    tl.store(Loss_ptr + offs_m * stride_lb, loss, mask=mask_m)
    tl.store(LSE_ptr + offs_m * stride_lse, lse, mask=mask_m)


@triton.autotune(
    configs=lpc_bwd_dh_configs,
    key=['D', 'V', 'BLOCK_M'],
    prune_configs_by={'early_config_prune': _prune_turing_lpc_configs},
)
@triton.jit
def _triton_lpc_bwd_dh_kernel(
    H_ptr, W_ptr, Targets_ptr, LSE_ptr, Grad_scale_ptr, DH_ptr,
    stride_hb, stride_hd,
    stride_wv, stride_wd,
    stride_tb,
    stride_lse,
    stride_dhb, stride_dhd,
    ignore_index: tl.constexpr,
    N, D: tl.constexpr, V: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    target = tl.load(Targets_ptr + offs_m * stride_tb, mask=mask_m, other=ignore_index)
    valid_mask = mask_m & (target != ignore_index) & (target >= 0) & (target < V)
    lse = tl.load(LSE_ptr + offs_m * stride_lse, mask=mask_m, other=0.0)
    grad_scale = tl.load(Grad_scale_ptr)

    acc_dtype = tl.float32 if (H_ptr.dtype.element_ty == tl.bfloat16 or W_ptr.dtype.element_ty == tl.bfloat16 or H_ptr.dtype.element_ty == tl.float16 or W_ptr.dtype.element_ty == tl.float16) else (tl.float64 if (H_ptr.dtype.element_ty == tl.float64 or W_ptr.dtype.element_ty == tl.float64) else tl.float32)
    dh_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=acc_dtype)

    for v_start in range(0, V, BLOCK_V):
        offs_v = v_start + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V

        logits = tl.zeros([BLOCK_M, BLOCK_V], dtype=acc_dtype)
        for d_k in range(0, D, BLOCK_D):
            H_block = tl.make_block_ptr(base=H_ptr, shape=(N, D), strides=(stride_hb, stride_hd), offsets=(pid_m * BLOCK_M, d_k), block_shape=(BLOCK_M, BLOCK_D), order=(1, 0))
            h_k = tl.load(H_block, boundary_check=(0, 1))
            W_block = tl.make_block_ptr(base=W_ptr, shape=(V, D), strides=(stride_wv, stride_wd), offsets=(v_start, d_k), block_shape=(BLOCK_V, BLOCK_D), order=(1, 0))
            w_k = tl.load(W_block, boundary_check=(0, 1))
            logits += tl.dot(h_k, tl.trans(w_k), input_precision="ieee")

        diff = tl.where(valid_mask[:, None] & mask_v[None, :], logits - lse[:, None], -50.0)
        p = tl.where(valid_mask[:, None] & mask_v[None, :], tl.exp(tl.minimum(diff, 0.0)), 0.0)
        is_target = (target[:, None] == offs_v[None, :]) & valid_mask[:, None] & mask_v[None, :]
        dlogits = tl.where(valid_mask[:, None] & mask_v[None, :], p - tl.where(is_target, 1.0, 0.0), 0.0)
        scaled_fp32 = dlogits * grad_scale
        Wd_block = tl.make_block_ptr(base=W_ptr, shape=(V, D), strides=(stride_wv, stride_wd), offsets=(v_start, pid_d * BLOCK_D), block_shape=(BLOCK_V, BLOCK_D), order=(1, 0))
        w_d = tl.load(Wd_block, boundary_check=(0, 1))
        dh_acc += tl.dot(scaled_fp32, w_d.to(tl.float32), input_precision="ieee")

    DH_block = tl.make_block_ptr(base=DH_ptr, shape=(N, D), strides=(stride_dhb, stride_dhd), offsets=(pid_m * BLOCK_M, pid_d * BLOCK_D), block_shape=(BLOCK_M, BLOCK_D), order=(1, 0))
    tl.store(DH_block, dh_acc.to(H_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=lpc_bwd_dw_configs,
    key=['D', 'V', 'BLOCK_M'],
    prune_configs_by={'early_config_prune': _prune_turing_lpc_configs},
)
@triton.jit
def _triton_lpc_bwd_dw_kernel(
    H_ptr, W_ptr, Targets_ptr, LSE_ptr, DW_ptr,
    Grad_scale_ptr,
    stride_hb, stride_hd,
    stride_wv, stride_wd,
    stride_tb,
    stride_lse,
    stride_dwv, stride_dwd,
    ignore_index: tl.constexpr,
    N, D: tl.constexpr, V: tl.constexpr,
    BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_v = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    acc_dtype = tl.float32 if (H_ptr.dtype.element_ty == tl.bfloat16 or W_ptr.dtype.element_ty == tl.bfloat16 or H_ptr.dtype.element_ty == tl.float16 or W_ptr.dtype.element_ty == tl.float16) else (tl.float64 if (H_ptr.dtype.element_ty == tl.float64 or W_ptr.dtype.element_ty == tl.float64) else tl.float32)
    grad_scale = tl.load(Grad_scale_ptr)
    dw_acc = tl.zeros([BLOCK_V, BLOCK_D], dtype=acc_dtype)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        target = tl.load(Targets_ptr + offs_n * stride_tb, mask=mask_n, other=ignore_index)
        valid_mask = mask_n & (target != ignore_index) & (target >= 0) & (target < V)
        lse = tl.load(LSE_ptr + offs_n * stride_lse, mask=mask_n, other=0.0)

        logits = tl.zeros([BLOCK_V, BLOCK_N], dtype=acc_dtype)
        for d_k in range(0, D, BLOCK_D):
            Wk_block = tl.make_block_ptr(base=W_ptr, shape=(V, D), strides=(stride_wv, stride_wd), offsets=(pid_v * BLOCK_V, d_k), block_shape=(BLOCK_V, BLOCK_D), order=(1, 0))
            w_k = tl.load(Wk_block, boundary_check=(0, 1))
            Hk_block = tl.make_block_ptr(base=H_ptr, shape=(N, D), strides=(stride_hb, stride_hd), offsets=(n_start, d_k), block_shape=(BLOCK_N, BLOCK_D), order=(1, 0))
            h_k = tl.load(Hk_block, boundary_check=(0, 1))
            logits += tl.dot(w_k, tl.trans(h_k), input_precision="ieee")

        diff = tl.where(mask_v[:, None] & valid_mask[None, :], logits - lse[None, :], -50.0)
        p = tl.where(mask_v[:, None] & valid_mask[None, :], tl.exp(tl.minimum(diff, 0.0)), 0.0)
        is_target = (offs_v[:, None] == target[None, :]) & mask_v[:, None] & valid_mask[None, :]
        dlogits = tl.where(mask_v[:, None] & valid_mask[None, :], p - tl.where(is_target, 1.0, 0.0), 0.0)
        scaled_fp32 = dlogits * grad_scale
        Hd_block = tl.make_block_ptr(base=H_ptr, shape=(N, D), strides=(stride_hb, stride_hd), offsets=(n_start, pid_d * BLOCK_D), block_shape=(BLOCK_N, BLOCK_D), order=(1, 0))
        h_d = tl.load(Hd_block, boundary_check=(0, 1))
        dw_acc += tl.dot(scaled_fp32, h_d.to(tl.float32), input_precision="ieee")

    DW_block = tl.make_block_ptr(base=DW_ptr, shape=(V, D), strides=(stride_dwv, stride_dwd), offsets=(pid_v * BLOCK_V, pid_d * BLOCK_D), block_shape=(BLOCK_V, BLOCK_D), order=(1, 0))
    tl.store(DW_block, dw_acc.to(W_ptr.dtype.element_ty), boundary_check=(0, 1))


class _TritonFusedLPCHeadFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        h: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100
    ) -> torch.Tensor:
        # Turing sm_75: 64KB SMEM => BLOCK<=64 already pruned; force fp16→fp32 accum even if bf16; bf16→fp16 fallback
        if _is_turing():
            if h.dtype == torch.bfloat16:
                warnings.warn(
                    "Turing sm_75 LPC: bf16 unsupported, forcing fp16→fp32 accum (bf16→fp16 fallback).",
                    stacklevel=3,
                )
                h = h.to(torch.float16)
            if weight.dtype == torch.bfloat16:
                warnings.warn(
                    "Turing sm_75 LPC weight: bf16 unsupported, forcing fp16.",
                    stacklevel=3,
                )
                weight = weight.to(torch.float16)
        # Keep original weight untouched; compute matmuls in promoted dtype without mutating param
        calc_dtype = torch.float32 if h.dtype in (torch.bfloat16, torch.float16) else (torch.float64 if h.dtype == torch.float64 else h.dtype)
        weight_compute = weight.to(calc_dtype) if weight.dtype != calc_dtype else weight
        h = h.contiguous()
        weight_compute = weight_compute.contiguous()
        weight_orig = weight.contiguous()
        targets = targets.contiguous()

        orig_shape = h.shape
        h_flat = h.view(-1, orig_shape[-1])
        targets_flat = targets.view(-1)

        N, D = h_flat.shape
        V = weight.shape[0]

        # For V<=1024 falls back to torch allocating [N,V] logits; fusion only for V>1024
        alloc_dtype = torch.float64 if h.dtype == torch.float64 else torch.float32
        if _use_torch_path(V):
            # Capture-safe branchless small-V path (no .item()/sync, no nested
            # autograd), mirroring triton_cross_entropy semantics.
            valid_bool = (targets_flat != ignore_index) & (targets_flat >= 0) & (targets_flat < V)
            # Use int64/float32 for n_valid sum to avoid bf16 mantissa loss
            n_valid = valid_bool.sum().to(torch.float32).clamp(min=1)
            valid = valid_bool.to(torch.float32)
            lse = torch.empty(0, dtype=alloc_dtype, device=h.device)
            ctx.save_for_backward(h_flat, weight_orig, targets_flat, lse, n_valid)
            ctx.orig_shape = orig_shape
            ctx.N = N
            ctx.D = D
            ctx.V = V
            ctx.ignore_index = ignore_index
            ctx.use_torch = True
            safe_idx = targets_flat.clamp(0, V - 1).unsqueeze(1)
            # Ensure matmul uses fp32 when bf16 for stable log_softmax
            h_for_logits = h_flat.to(torch.float32) if h.dtype in (torch.bfloat16, torch.float16) else h_flat.to(calc_dtype)
            w_for_logits = weight_compute.to(torch.float32) if h.dtype in (torch.bfloat16, torch.float16) else weight_compute
            logits = torch.matmul(h_for_logits, w_for_logits.t())
            # log_softmax in fp32 when bf16: keep logits fp32
            nll = -logits.log_softmax(dim=-1).gather(1, safe_idx).squeeze(1).to(torch.float32)
            total_loss = (nll * valid).sum() / n_valid
            return total_loss.to(h.dtype)

        losses = torch.empty(N, dtype=alloc_dtype, device=h.device)
        lse = torch.empty(N, dtype=alloc_dtype, device=h.device)

        grid = lambda META: (triton.cdiv(N, META['BLOCK_M']),)  # N varies: grid depends on N via cdiv

        _triton_lpc_fwd_kernel[grid](
            h_flat, weight_orig, targets_flat, losses, lse,
            h_flat.stride(0), h_flat.stride(1),
            weight_orig.stride(0), weight_orig.stride(1),
            targets_flat.stride(0),
            losses.stride(0),
            lse.stride(0),
            ignore_index=ignore_index,
            N=N, D=D, V=V,
        )

        valid_mask = (targets_flat != ignore_index) & (targets_flat >= 0) & (targets_flat < V)
        n_valid = valid_mask.sum().to(torch.float32).clamp(min=1)
        total_loss = losses.sum() / n_valid

        ctx.save_for_backward(h_flat, weight_orig, targets_flat, lse, n_valid)
        ctx.orig_shape = orig_shape
        ctx.N = N
        ctx.D = D
        ctx.V = V
        ctx.ignore_index = ignore_index
        ctx.use_torch = False
        return total_loss.to(h.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None, None]:
        h_flat, weight, targets_flat, lse, n_valid = ctx.saved_tensors
        N = ctx.N
        D = ctx.D
        V = ctx.V
        ignore_index = ctx.ignore_index

        if getattr(ctx, 'use_torch', False) or _use_torch_path(V):
            # Explicit softmax backward, capture-safe (plain tensor ops only).
            need_dx = ctx.needs_input_grad[0]
            need_dw = ctx.needs_input_grad[1]
            if not need_dx and not need_dw:
                return None, None, None, None
            valid_bool = ((targets_flat != ignore_index) & (targets_flat >= 0) & (targets_flat < V))
            safe_idx = targets_flat.clamp(0, V - 1).unsqueeze(1)
            # Promote bf16/fp16 to fp32 for stable log_softmax and to avoid mantissa loss
            acc_dtype = torch.float32 if h_flat.dtype in (torch.bfloat16, torch.float16) else (torch.float64 if h_flat.dtype == torch.float64 else h_flat.dtype)
            valid = valid_bool.to(torch.float32)
            # Ensure matmul uses fp32 when bf16 for stable log_softmax
            h_for_logits = h_flat.to(torch.float32) if h_flat.dtype in (torch.bfloat16, torch.float16) else h_flat.to(acc_dtype)
            w_for_logits = weight.to(torch.float32) if h_flat.dtype in (torch.bfloat16, torch.float16) else weight.to(acc_dtype)
            logits = torch.matmul(h_for_logits, w_for_logits.t())
            probs = logits.log_softmax(dim=-1).exp()
            one_hot = torch.zeros_like(probs)
            one_hot.scatter_(1, safe_idx, valid.unsqueeze(1))
            dlogits = (probs - one_hot) * valid.unsqueeze(1)
            dlogits = dlogits * (grad_output / n_valid).to(dlogits.dtype)
            dh_flat = torch.matmul(dlogits, w_for_logits).to(h_flat.dtype).view(ctx.orig_shape) if need_dx else None
            dw = torch.matmul(dlogits.t(), h_flat.to(torch.float32) if h_flat.dtype in (torch.bfloat16, torch.float16) else h_flat.to(acc_dtype)).to(weight.dtype) if need_dw else None
            return dh_flat, dw, None, None

        scale_dtype = torch.float64 if h_flat.dtype == torch.float64 else torch.float32
        grad_scale_tensor = (grad_output / n_valid).to(scale_dtype)

        dh_flat = None
        if ctx.needs_input_grad[0]:
            dh_flat = torch.empty((N, D), dtype=h_flat.dtype, device=h_flat.device)
            grid_dh = lambda META: (triton.cdiv(N, META['BLOCK_M']), triton.cdiv(D, META['BLOCK_D']))

            _triton_lpc_bwd_dh_kernel[grid_dh](
                h_flat, weight, targets_flat, lse, grad_scale_tensor, dh_flat,
                h_flat.stride(0), h_flat.stride(1),
                weight.stride(0), weight.stride(1),
                targets_flat.stride(0),
                lse.stride(0),
                dh_flat.stride(0), dh_flat.stride(1),
                ignore_index=ignore_index,
                N=N, D=D, V=V,
            )
            dh_flat = dh_flat.view(ctx.orig_shape)

        dw = None
        if ctx.needs_input_grad[1]:
            dw = torch.empty_like(weight)
            grid_dw = lambda META: (triton.cdiv(V, META['BLOCK_V']), triton.cdiv(D, META['BLOCK_D']))

            _triton_lpc_bwd_dw_kernel[grid_dw](
                h_flat, weight, targets_flat, lse, dw,
                grad_scale_tensor,
                h_flat.stride(0), h_flat.stride(1),
                weight.stride(0), weight.stride(1),
                targets_flat.stride(0),
                lse.stride(0),
                dw.stride(0), dw.stride(1),
                ignore_index=ignore_index,
                N=N, D=D, V=V,
            )

        return dh_flat, dw, None, None


def triton_fused_lpc_head(
    h: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Computes Local Predictive Coding loss and exact in-place gradients via Triton.
    Eliminates intermediate logits allocation in DRAM. For V<=1024 falls back to
    torch allocating [N,V] logits; fusion only for V>1024. Masking OOB (ignore_index,
    target <0 or >=V) is handled via valid_mask with masked stores/loads.

    Args:
        h: Local layer hidden representation of shape (B, T, D) or (N, D).
        weight: Local predictive head projection weights of shape (V, D).
        targets: Target token IDs of shape (B, T) or (N,).
        ignore_index: Index to ignore in loss calculation (default -100).

    Returns:
        Scalar local cross-entropy loss.
    """
    if not h.is_cuda:
        if _use_torch_path(weight.shape[0]):
            # Capture-safe branchless torch path via the Function (single code
            # path; also faster than the AVX C++ head for small V on CPU).
            return _TritonFusedLPCHeadFunc.apply(h, weight, targets, ignore_index)
        try:
            from affine_ai.core.cpp_ops import asdag_cpu_lpc_head
            return asdag_cpu_lpc_head(h, weight, targets, ignore_index).to(h.dtype)
        except Exception:
            orig_shape = h.shape
            h_flat = h.view(-1, orig_shape[-1]).float()
            w_flat = weight.float()
            targets_flat = targets.view(-1)
            logits = torch.matmul(h_flat, w_flat.t())
            return torch.nn.functional.cross_entropy(logits, targets_flat, ignore_index=ignore_index).to(h.dtype)

    # Single code path on CUDA: the Function dispatches internally (see note
    # in triton_cross_entropy wrapper). Keeps the captured region sync-free.
    return _TritonFusedLPCHeadFunc.apply(h, weight, targets, ignore_index)
