"""
Custom Triton Kernels for Local Predictive Coding (LPC)
========================================================
Implements fused Forward-Predict-Loss-Gradient operations specifically tailored
for in-place, forward-only local error learning:
1. Zero DRAM allocation for un-materialized intermediate logits [B, T, V].
2. Fused online Log-Sum-Exp (LSE) loss calculation in GPU SRAM.
3. In-place gradient projection (dH = err @ W_head) and dW_head accumulation.
4. Seamless integration with standard PyTorch autograd and forward-only execution.
Coalesced access via tl.make_block_ptr with boundary_check for tails.
"""

import warnings
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


_TURING_CACHE = None


def _is_turing() -> bool:
    global _TURING_CACHE
    if _TURING_CACHE is not None:
        return _TURING_CACHE
    try:
        from affine_ai.kernels import _IS_TURING as _T  # type: ignore

        _TURING_CACHE = bool(_T)
        return _TURING_CACHE
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            _TURING_CACHE = (7, 5) <= tuple(cap) < (8, 0)
            return _TURING_CACHE
    except Exception:
        pass
    _TURING_CACHE = False
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
    return False


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
    key=['D', 'V'],
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
    BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr,
    IS_TURING: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N

    target = tl.load(Targets_ptr + offs_m * stride_tb, mask=mask_m, other=ignore_index)
    valid_mask = mask_m & (target != ignore_index) & (target >= 0) & (target < V)

    acc_dtype = tl.float32 if (H_ptr.dtype.element_ty == tl.bfloat16 or W_ptr.dtype.element_ty == tl.bfloat16 or H_ptr.dtype.element_ty == tl.float16 or W_ptr.dtype.element_ty == tl.float16) else (tl.float64 if (H_ptr.dtype.element_ty == tl.float64 or W_ptr.dtype.element_ty == tl.float64) else tl.float32)

    m_i = tl.full([BLOCK_M], -1e30, dtype=acc_dtype)
    l_i = tl.zeros([BLOCK_M], dtype=acc_dtype)
    target_logit = tl.zeros([BLOCK_M], dtype=acc_dtype)

    for v_start in range(0, V, BLOCK_V):
        offs_v = v_start + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V

        acc = tl.zeros([BLOCK_M, BLOCK_V], dtype=acc_dtype)
        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            if stride_hd == 1 and stride_wd == 1:
                H_block = tl.make_block_ptr(base=H_ptr, shape=(N, D), strides=(stride_hb, stride_hd), offsets=(pid_m * BLOCK_M, d_start), block_shape=(BLOCK_M, BLOCK_D), order=(1, 0))
                h_tile = tl.load(H_block, boundary_check=(0, 1))
                W_block = tl.make_block_ptr(base=W_ptr, shape=(V, D), strides=(stride_wv, stride_wd), offsets=(v_start, d_start), block_shape=(BLOCK_V, BLOCK_D), order=(1, 0))
                w_tile = tl.load(W_block, boundary_check=(0, 1))
            else:
                h_tile = tl.load(H_ptr + offs_m[:, None] * stride_hb + offs_d[None, :] * stride_hd, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
                w_tile = tl.load(W_ptr + offs_v[:, None] * stride_wv + offs_d[None, :] * stride_wd, mask=mask_v[:, None] & mask_d[None, :], other=0.0)

            if IS_TURING:
                acc += tl.dot(h_tile.to(tl.float16), tl.trans(w_tile).to(tl.float16), allow_tf32=False)
            else:
                acc += tl.dot(h_tile, tl.trans(w_tile), allow_tf32=False)

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
    key=['D', 'V'],
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
    BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr,
    IS_TURING: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N

    target = tl.load(Targets_ptr + offs_m * stride_tb, mask=mask_m, other=ignore_index)
    valid_mask = mask_m & (target != ignore_index) & (target >= 0) & (target < V)
    lse = tl.load(LSE_ptr + offs_m * stride_lse, mask=mask_m, other=0.0)
    grad_scale = tl.load(Grad_scale_ptr)

    acc_dtype = tl.float32 if (H_ptr.dtype.element_ty == tl.bfloat16 or W_ptr.dtype.element_ty == tl.bfloat16 or H_ptr.dtype.element_ty == tl.float16 or W_ptr.dtype.element_ty == tl.float16) else (tl.float64 if (H_ptr.dtype.element_ty == tl.float64 or W_ptr.dtype.element_ty == tl.float64) else tl.float32)

    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        dh_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=acc_dtype)

        for v_start in range(0, V, BLOCK_V):
            offs_v = v_start + tl.arange(0, BLOCK_V)
            mask_v = offs_v < V

            logits = tl.zeros([BLOCK_M, BLOCK_V], dtype=acc_dtype)
            for d_k in range(0, D, BLOCK_D):
                offs_dk = d_k + tl.arange(0, BLOCK_D)
                mask_dk = offs_dk < D
                h_k = tl.load(H_ptr + offs_m[:, None] * stride_hb + offs_dk[None, :] * stride_hd, mask=mask_m[:, None] & mask_dk[None, :], other=0.0)
                w_k = tl.load(W_ptr + offs_v[:, None] * stride_wv + offs_dk[None, :] * stride_wd, mask=mask_v[:, None] & mask_dk[None, :], other=0.0)
                if IS_TURING:
                    logits += tl.dot(h_k.to(tl.float16), tl.trans(w_k).to(tl.float16), allow_tf32=False)
                else:
                    logits += tl.dot(h_k, tl.trans(w_k), allow_tf32=False)

            diff = tl.where(valid_mask[:, None] & mask_v[None, :], logits - lse[:, None], -50.0)
            p = tl.where(valid_mask[:, None] & mask_v[None, :], tl.exp(tl.minimum(diff, 0.0)), 0.0)
            is_target = (target[:, None] == offs_v[None, :]) & valid_mask[:, None] & mask_v[None, :]
            dlogits = tl.where(valid_mask[:, None] & mask_v[None, :], p - tl.where(is_target, 1.0, 0.0), 0.0)
            scaled_fp32 = dlogits * grad_scale

            w_d = tl.load(W_ptr + offs_v[:, None] * stride_wv + offs_d[None, :] * stride_wd, mask=mask_v[:, None] & mask_d[None, :], other=0.0)
            if IS_TURING:
                dh_contrib = tl.dot(scaled_fp32.to(tl.float16), w_d.to(tl.float16), allow_tf32=False)
            else:
                dh_contrib = tl.dot(scaled_fp32, w_d.to(tl.float32), allow_tf32=False)
            dh_acc += dh_contrib

        dh_ptrs = DH_ptr + offs_m[:, None] * stride_dhb + offs_d[None, :] * stride_dhd
        tl.store(dh_ptrs, dh_acc, mask=mask_m[:, None] & mask_d[None, :])


@triton.autotune(
    configs=lpc_bwd_dw_configs,
    key=['D', 'V'],
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
    stride_split,
    ignore_index: tl.constexpr,
    N, D: tl.constexpr, V: tl.constexpr,
    BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
    SPLIT_N: tl.constexpr,
    IS_TURING: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_split = tl.program_id(1)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    acc_dtype = tl.float32 if (H_ptr.dtype.element_ty == tl.bfloat16 or W_ptr.dtype.element_ty == tl.bfloat16 or H_ptr.dtype.element_ty == tl.float16 or W_ptr.dtype.element_ty == tl.float16) else (tl.float64 if (H_ptr.dtype.element_ty == tl.float64 or W_ptr.dtype.element_ty == tl.float64) else tl.float32)
    grad_scale = tl.load(Grad_scale_ptr)

    n_per_split = (N + SPLIT_N - 1) // SPLIT_N
    split_start = pid_split * n_per_split
    split_end = tl.minimum(N, (pid_split + 1) * n_per_split)

    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        dw_tile = tl.zeros([BLOCK_V, BLOCK_D], dtype=acc_dtype)

        for n_start in range(split_start, split_end, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N

            target = tl.load(Targets_ptr + offs_n * stride_tb, mask=mask_n, other=ignore_index)
            valid_mask = mask_n & (target != ignore_index) & (target >= 0) & (target < V)
            lse = tl.load(LSE_ptr + offs_n * stride_lse, mask=mask_n, other=0.0)

            logits = tl.zeros([BLOCK_V, BLOCK_N], dtype=acc_dtype)
            for d_k in range(0, D, BLOCK_D):
                offs_dk = d_k + tl.arange(0, BLOCK_D)
                mask_dk = offs_dk < D
                w_k = tl.load(W_ptr + offs_v[:, None] * stride_wv + offs_dk[None, :] * stride_wd, mask=mask_v[:, None] & mask_dk[None, :], other=0.0)
                h_k = tl.load(H_ptr + offs_n[:, None] * stride_hb + offs_dk[None, :] * stride_hd, mask=mask_n[:, None] & mask_dk[None, :], other=0.0)
                if IS_TURING:
                    logits += tl.dot(w_k.to(tl.float16), tl.trans(h_k).to(tl.float16), allow_tf32=False)
                else:
                    logits += tl.dot(w_k, tl.trans(h_k), allow_tf32=False)

            diff = tl.where(mask_v[:, None] & valid_mask[None, :], logits - lse[None, :], -50.0)
            p = tl.where(mask_v[:, None] & valid_mask[None, :], tl.exp(tl.minimum(diff, 0.0)), 0.0)
            is_target = (offs_v[:, None] == target[None, :]) & mask_v[:, None] & valid_mask[None, :]
            dlogits = tl.where(mask_v[:, None] & valid_mask[None, :], p - tl.where(is_target, 1.0, 0.0), 0.0)
            scaled_fp32 = dlogits * grad_scale

            h_d = tl.load(H_ptr + offs_n[:, None] * stride_hb + offs_d[None, :] * stride_hd, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            if IS_TURING:
                dw_contrib = tl.dot(scaled_fp32.to(tl.float16), h_d.to(tl.float16), allow_tf32=False)
            else:
                dw_contrib = tl.dot(scaled_fp32, h_d.to(tl.float32), allow_tf32=False)
            dw_tile += dw_contrib

        dw_ptrs = DW_ptr + pid_split * stride_split + offs_v[:, None] * stride_dwv + offs_d[None, :] * stride_dwd
        tl.store(dw_ptrs, dw_tile, mask=mask_v[:, None] & mask_d[None, :])


class _TritonFusedLPCHeadFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        h: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100
    ) -> torch.Tensor:
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

        h = h.contiguous()
        weight_orig = weight.contiguous()
        targets = targets.contiguous()

        orig_shape = h.shape
        h_flat = h.view(-1, orig_shape[-1])
        targets_flat = targets.view(-1)

        N, D = h_flat.shape
        V = weight.shape[0]

        alloc_dtype = torch.float64 if h.dtype == torch.float64 else torch.float32
        losses = torch.empty(N, dtype=alloc_dtype, device=h.device)
        lse = torch.empty(N, dtype=alloc_dtype, device=h.device)

        def grid(META):
            return (triton.cdiv(N, META['BLOCK_M']),)

        _triton_lpc_fwd_kernel[grid](
            h_flat, weight_orig, targets_flat, losses, lse,
            h_flat.stride(0), h_flat.stride(1),
            weight_orig.stride(0), weight_orig.stride(1),
            targets_flat.stride(0),
            losses.stride(0),
            lse.stride(0),
            ignore_index=ignore_index,
            N=N, D=D, V=V,
            IS_TURING=_is_turing(),
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
        return total_loss.to(h.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None, None]:
        h_flat, weight, targets_flat, lse, n_valid = ctx.saved_tensors
        N = ctx.N
        D = ctx.D
        V = ctx.V
        ignore_index = ctx.ignore_index

        scale_dtype = torch.float64 if h_flat.dtype == torch.float64 else torch.float32
        grad_scale_tensor = (grad_output / n_valid).to(scale_dtype)

        dh_flat = None
        if ctx.needs_input_grad[0]:
            calc_dtype = torch.float64 if h_flat.dtype == torch.float64 else torch.float32
            dh_acc = torch.zeros((N, D), dtype=calc_dtype, device=h_flat.device)
            def grid_dh(META):
                return (triton.cdiv(N, META['BLOCK_M']),)

            _triton_lpc_bwd_dh_kernel[grid_dh](
                h_flat, weight, targets_flat, lse, grad_scale_tensor, dh_acc,
                h_flat.stride(0), h_flat.stride(1),
                weight.stride(0), weight.stride(1),
                targets_flat.stride(0),
                lse.stride(0),
                dh_acc.stride(0), dh_acc.stride(1),
                ignore_index=ignore_index,
                N=N, D=D, V=V,
                IS_TURING=_is_turing(),
            )
            dh_flat = dh_acc.to(h_flat.dtype).view(ctx.orig_shape)

        dw = None
        if ctx.needs_input_grad[1]:
            calc_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
            split_n = min(16, triton.cdiv(N, 256)) if N >= 256 else 1
            if split_n > 1:
                dw_split = torch.empty((split_n, V, D), dtype=calc_dtype, device=weight.device)
                dw_target = dw_split
                stride_split = dw_split.stride(0)
            else:
                dw_acc = torch.zeros((V, D), dtype=calc_dtype, device=weight.device)
                dw_target = dw_acc
                stride_split = 0

            def grid_dw(META):
                return (triton.cdiv(V, META['BLOCK_V']), split_n)

            _triton_lpc_bwd_dw_kernel[grid_dw](
                h_flat, weight, targets_flat, lse, dw_target,
                grad_scale_tensor,
                h_flat.stride(0), h_flat.stride(1),
                weight.stride(0), weight.stride(1),
                targets_flat.stride(0),
                lse.stride(0),
                dw_target.stride(-2), dw_target.stride(-1),
                stride_split,
                ignore_index=ignore_index,
                N=N, D=D, V=V,
                SPLIT_N=split_n,
                IS_TURING=_is_turing(),
            )
            if split_n > 1:
                dw = dw_split.sum(dim=0).to(weight.dtype)
            else:
                dw = dw_acc.to(weight.dtype)

        return dh_flat, dw, None, None


def triton_fused_lpc_head(
    h: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Computes Local Predictive Coding loss and exact in-place gradients via Triton.
    Eliminates intermediate logits allocation in DRAM.

    Args:
        h: Local layer hidden representation of shape (B, T, D) or (N, D).
        weight: Local predictive head projection weights of shape (V, D).
        targets: Target token IDs of shape (B, T) or (N,).
        ignore_index: Index to ignore in loss calculation (default -100).

    Returns:
        Scalar local cross-entropy loss.
    """
    if not h.is_cuda:
        try:
            from affine_ai.core.cpp_ops import asdag_cpu_lpc_head
            return asdag_cpu_lpc_head(h, weight, targets, ignore_index).to(h.dtype)
        except Exception:
            pass
        orig_shape = h.shape
        h_flat = h.view(-1, orig_shape[-1]).float()
        w_flat = weight.float()
        targets_flat = targets.view(-1)
        logits = torch.matmul(h_flat, w_flat.t())
        return torch.nn.functional.cross_entropy(logits, targets_flat, ignore_index=ignore_index).to(h.dtype)

    return _TritonFusedLPCHeadFunc.apply(h, weight, targets, ignore_index)
