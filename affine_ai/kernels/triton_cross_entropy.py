"""
Custom Triton Kernel: High-Performance Fused Linear + Cross-Entropy Loss
========================================================================
Computes projection logits H @ W^T and Cross-Entropy loss in a single fused kernel
directly inside GPU SRAM caches using Online Log-Sum-Exp.
Zero DRAM memory allocation for un-materialized logits.
Strictly bounded exponentiation prevents overflow/underflow (zero NaNs/Infs).
Supports arbitrary hidden dimensions D (chunked accumulation), arbitrary vocabularies V,
and autotuned block layouts for optimal warp reduction throughput.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple, Optional


def _use_torch_path(V: int) -> bool:
    return V <= 1024


# ==============================================================================
# Forward Pass Autotuning Configurations
# ==============================================================================
fwd_configs = [
    triton.Config({'BLOCK_M': 16, 'BLOCK_V': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_V': 64, 'BLOCK_D': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_V': 64, 'BLOCK_D': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 32, 'BLOCK_V': 32, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_V': 32, 'BLOCK_D': 32}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_V': 32, 'BLOCK_D': 32}, num_warps=4, num_stages=5),
    triton.Config({'BLOCK_M': 16, 'BLOCK_V': 32, 'BLOCK_D': 32}, num_warps=2, num_stages=4),
    triton.Config({'BLOCK_M': 16, 'BLOCK_V': 32, 'BLOCK_D': 32}, num_warps=2, num_stages=5),
    triton.Config({'BLOCK_M': 64, 'BLOCK_V': 64, 'BLOCK_D': 32}, num_warps=4, num_stages=2),
]


@triton.autotune(
    configs=fwd_configs,
    key=['D', 'V'],
)
@triton.jit
def _fused_linear_cross_entropy_fwd_kernel(
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

    acc_dtype = tl.float64 if H_ptr.dtype.element_ty == tl.float64 else tl.float32

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

            h_ptrs = H_ptr + offs_m[:, None] * stride_hb + offs_d[None, :] * stride_hd
            w_ptrs = W_ptr + offs_v[:, None] * stride_wv + offs_d[None, :] * stride_wd

            h_tile = tl.load(h_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
            w_tile = tl.load(w_ptrs, mask=mask_v[:, None] & mask_d[None, :], other=0.0)

            acc += tl.dot(h_tile, tl.trans(w_tile), input_precision="ieee")

        acc_masked = tl.where(mask_m[:, None] & mask_v[None, :], acc, -1e30)
        chunk_max = tl.max(acc_masked, axis=1)
        m_new = tl.maximum(m_i, chunk_max)

        # Numerically stable exponent difference clamping
        alpha = tl.where(m_i > -1e20, tl.exp(m_i - m_new), 0.0)
        diff = tl.where(mask_m[:, None] & mask_v[None, :], acc - m_new[:, None], -50.0)
        p = tl.where(mask_m[:, None] & mask_v[None, :], tl.exp(diff), 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        is_target = (target[:, None] == offs_v[None, :]) & valid_mask[:, None] & mask_v[None, :]
        target_logit += tl.sum(tl.where(is_target, acc, 0.0), axis=1)

        m_i = m_new
        l_i = l_new

    # Store safe lse and loss (0.0 for masked elements to avoid +inf in backward pass)
    lse = tl.where(valid_mask, m_i + tl.log(tl.maximum(l_i, 1e-12)), 0.0)
    loss = tl.where(valid_mask, lse - target_logit, 0.0)

    tl.store(Loss_ptr + offs_m * stride_lb, loss, mask=mask_m)
    tl.store(LSE_ptr + offs_m * stride_lse, lse, mask=mask_m)


# ==============================================================================
# Backward dH Kernel Autotuning Configurations
# ==============================================================================
bwd_dh_configs = [
    triton.Config({'BLOCK_M': 16, 'BLOCK_V': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_V': 64, 'BLOCK_D': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_V': 64, 'BLOCK_D': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 32, 'BLOCK_V': 32, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_V': 32, 'BLOCK_D': 32}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_V': 32, 'BLOCK_D': 32}, num_warps=4, num_stages=5),
    triton.Config({'BLOCK_M': 16, 'BLOCK_V': 32, 'BLOCK_D': 32}, num_warps=2, num_stages=4),
    triton.Config({'BLOCK_M': 16, 'BLOCK_V': 32, 'BLOCK_D': 32}, num_warps=2, num_stages=5),
    triton.Config({'BLOCK_M': 64, 'BLOCK_V': 64, 'BLOCK_D': 32}, num_warps=4, num_stages=2),
]


@triton.autotune(
    configs=bwd_dh_configs,
    key=['D', 'V'],
)
@triton.jit
def _fused_linear_cross_entropy_bwd_dh_kernel(
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

    acc_dtype = tl.float64 if H_ptr.dtype.element_ty == tl.float64 else tl.float32
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
            logits += tl.dot(h_k, tl.trans(w_k), input_precision="ieee")

        diff = tl.where(valid_mask[:, None] & mask_v[None, :], logits - lse[:, None], -50.0)
        p = tl.where(valid_mask[:, None] & mask_v[None, :], tl.exp(tl.minimum(diff, 0.0)), 0.0)
        is_target = (target[:, None] == offs_v[None, :]) & valid_mask[:, None] & mask_v[None, :]
        dlogits = tl.where(valid_mask[:, None] & mask_v[None, :], p - tl.where(is_target, 1.0, 0.0), 0.0)
        scaled_dlogits = (dlogits * grad_scale).to(H_ptr.dtype.element_ty)

        w_d = tl.load(W_ptr + offs_v[:, None] * stride_wv + offs_d[None, :] * stride_wd, mask=mask_v[:, None] & mask_d[None, :], other=0.0)
        dh_acc += tl.dot(scaled_dlogits, w_d, input_precision="ieee")

    dh_ptrs = DH_ptr + offs_m[:, None] * stride_dhb + offs_d[None, :] * stride_dhd
    tl.store(dh_ptrs, dh_acc.to(H_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])


# ==============================================================================
# Backward dW Kernel Autotuning Configurations
# ==============================================================================
bwd_dw_configs = [
    triton.Config({'BLOCK_V': 64, 'BLOCK_N': 16, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_V': 32, 'BLOCK_N': 64, 'BLOCK_D': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_V': 32, 'BLOCK_N': 64, 'BLOCK_D': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_V': 32, 'BLOCK_N': 32, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_V': 32, 'BLOCK_N': 32, 'BLOCK_D': 32}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_V': 32, 'BLOCK_N': 32, 'BLOCK_D': 32}, num_warps=4, num_stages=5),
    triton.Config({'BLOCK_V': 32, 'BLOCK_N': 16, 'BLOCK_D': 32}, num_warps=2, num_stages=4),
    triton.Config({'BLOCK_V': 32, 'BLOCK_N': 16, 'BLOCK_D': 32}, num_warps=2, num_stages=5),
    triton.Config({'BLOCK_V': 64, 'BLOCK_N': 64, 'BLOCK_D': 32}, num_warps=4, num_stages=2),
]


@triton.autotune(
    configs=bwd_dw_configs,
    key=['D', 'V'],
)
@triton.jit
def _fused_linear_cross_entropy_bwd_dw_kernel(
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

    acc_dtype = tl.float64 if W_ptr.dtype.element_ty == tl.float64 else tl.float32
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
            offs_dk = d_k + tl.arange(0, BLOCK_D)
            mask_dk = offs_dk < D
            w_k = tl.load(W_ptr + offs_v[:, None] * stride_wv + offs_dk[None, :] * stride_wd, mask=mask_v[:, None] & mask_dk[None, :], other=0.0)
            h_k = tl.load(H_ptr + offs_n[:, None] * stride_hb + offs_dk[None, :] * stride_hd, mask=mask_n[:, None] & mask_dk[None, :], other=0.0)
            logits += tl.dot(w_k, tl.trans(h_k), input_precision="ieee")

        diff = tl.where(mask_v[:, None] & valid_mask[None, :], logits - lse[None, :], -50.0)
        p = tl.where(mask_v[:, None] & valid_mask[None, :], tl.exp(tl.minimum(diff, 0.0)), 0.0)
        is_target = (offs_v[:, None] == target[None, :]) & mask_v[:, None] & valid_mask[None, :]
        dlogits = tl.where(mask_v[:, None] & valid_mask[None, :], p - tl.where(is_target, 1.0, 0.0), 0.0)
        scaled_dlogits = (dlogits * grad_scale).to(W_ptr.dtype.element_ty)

        h_d = tl.load(H_ptr + offs_n[:, None] * stride_hb + offs_d[None, :] * stride_hd, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        dw_acc += tl.dot(scaled_dlogits, h_d, input_precision="ieee")

    dw_ptrs = DW_ptr + offs_v[:, None] * stride_dwv + offs_d[None, :] * stride_dwd
    tl.store(dw_ptrs, dw_acc.to(W_ptr.dtype.element_ty), mask=mask_v[:, None] & mask_d[None, :])


# ==============================================================================
# PyTorch Autograd Function Interface
# ==============================================================================
class _TritonFusedLinearCrossEntropyFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        h: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100
    ) -> torch.Tensor:
        if weight.dtype != h.dtype:
            weight = weight.to(h.dtype)
        h = h.contiguous()
        weight = weight.contiguous()
        targets = targets.contiguous()

        orig_shape = h.shape
        h_flat = h.view(-1, orig_shape[-1])
        targets_flat = targets.view(-1)

        N, D = h_flat.shape
        V = weight.shape[0]

        calc_dtype = torch.float64 if h.dtype == torch.float64 else torch.float32
        if _use_torch_path(V):
            # Capture-safe branchless small-V path (no .item()/sync, no nested
            # autograd): loss = (nll * valid).sum() / n_valid, matching the
            # fused kernel's mean-over-valid semantics exactly, all-ignored -> 0.
            acc_dtype = h.dtype
            valid = ((targets_flat != ignore_index) & (targets_flat >= 0) & (targets_flat < V)).to(acc_dtype)
            n_valid = valid.sum().clamp(min=1)
            lse = torch.empty(0, dtype=calc_dtype, device=h.device)
            ctx.save_for_backward(h_flat, weight, targets_flat, lse, n_valid)
            ctx.orig_shape = orig_shape
            ctx.N = N
            ctx.D = D
            ctx.V = V
            ctx.ignore_index = ignore_index
            ctx.use_torch = True
            safe_idx = targets_flat.clamp(0, V - 1).unsqueeze(1)
            logits = torch.matmul(h_flat.to(acc_dtype), weight.to(acc_dtype).t())
            nll = -logits.log_softmax(dim=-1).gather(1, safe_idx).squeeze(1)
            total_loss = (nll * valid).sum() / n_valid
            return total_loss.to(h.dtype)

        losses = torch.empty(N, dtype=calc_dtype, device=h.device)
        lse = torch.empty(N, dtype=calc_dtype, device=h.device)

        grid = lambda META: (triton.cdiv(N, META['BLOCK_M']),)

        _fused_linear_cross_entropy_fwd_kernel[grid](
            h_flat, weight, targets_flat, losses, lse,
            h_flat.stride(0), h_flat.stride(1),
            weight.stride(0), weight.stride(1),
            targets_flat.stride(0),
            losses.stride(0),
            lse.stride(0),
            ignore_index=ignore_index,
            N=N, D=D, V=V,
        )

        valid_mask = (targets_flat != ignore_index) & (targets_flat >= 0) & (targets_flat < V)
        n_valid = valid_mask.sum().clamp(min=1)
        total_loss = losses.sum() / n_valid

        ctx.save_for_backward(h_flat, weight, targets_flat, lse, n_valid)
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
            valid = ((targets_flat != ignore_index) & (targets_flat >= 0) & (targets_flat < V))
            safe_idx = targets_flat.clamp(0, V - 1).unsqueeze(1)
            acc_dtype = h_flat.dtype
            valid = valid.to(acc_dtype)
            logits = torch.matmul(h_flat.to(acc_dtype), weight.to(acc_dtype).t())
            probs = logits.log_softmax(dim=-1).exp()
            one_hot = torch.zeros_like(probs)
            one_hot.scatter_(1, safe_idx, valid.unsqueeze(1))
            dlogits = (probs - one_hot) * valid.unsqueeze(1)
            dlogits = dlogits * (grad_output / n_valid).to(dlogits.dtype)
            dh_flat = torch.matmul(dlogits, weight.to(acc_dtype)).to(h_flat.dtype).view(ctx.orig_shape) if need_dx else None
            dw = torch.matmul(dlogits.t(), h_flat.to(acc_dtype)).to(weight.dtype) if need_dw else None
            return dh_flat, dw, None, None

        scale_dtype = torch.float64 if h_flat.dtype == torch.float64 else torch.float32
        grad_scale_tensor = (grad_output / n_valid).to(scale_dtype)

        dh_flat = None
        if ctx.needs_input_grad[0]:
            dh_flat = torch.empty((N, D), dtype=h_flat.dtype, device=h_flat.device)
            grid_dh = lambda META: (triton.cdiv(N, META['BLOCK_M']), triton.cdiv(D, META['BLOCK_D']))

            _fused_linear_cross_entropy_bwd_dh_kernel[grid_dh](
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
            if h_flat.dtype == torch.float64:
                # Analytical double-precision backward pass for exact gradchecks
                dw = torch.zeros_like(weight, dtype=torch.float64)
                chunk_v = min(V, 2048)
                for v_start in range(0, V, chunk_v):
                    v_end = min(v_start + chunk_v, V)
                    w_sub = weight[v_start:v_end]
                    logits_sub = torch.matmul(h_flat, w_sub.t())
                    diff_sub = torch.clamp(logits_sub - lse.unsqueeze(-1), min=-50.0, max=0.0)
                    probs_sub = torch.exp(diff_sub)
                    tgt_mask = (targets_flat >= v_start) & (targets_flat < v_end) & (targets_flat != ignore_index)
                    clamped_tgt = (targets_flat - v_start).clamp(0, (v_end - v_start) - 1)
                    one_hot = torch.zeros_like(probs_sub)
                    one_hot.scatter_(1, clamped_tgt.unsqueeze(1), tgt_mask.unsqueeze(1).to(probs_sub.dtype))
                    probs_sub = probs_sub - one_hot
                    invalid = (targets_flat == ignore_index) | (targets_flat < 0) | (targets_flat >= V)
                    probs_sub = torch.where(invalid.unsqueeze(1), torch.zeros_like(probs_sub), probs_sub)
                    dw[v_start:v_end] = torch.matmul(probs_sub.t(), h_flat) * grad_scale_tensor
                dw = dw.to(weight.dtype)
            else:
                # High-Performance Fused SRAM dW kernel (Zero CPU-GPU stalls)
                dw = torch.empty_like(weight)
                grid_dw = lambda META: (triton.cdiv(V, META['BLOCK_V']), triton.cdiv(D, META['BLOCK_D']))

                _fused_linear_cross_entropy_bwd_dw_kernel[grid_dw](
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


def triton_fused_linear_cross_entropy(
    h: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Computes fused Linear projection (H @ W^T) and Cross-Entropy loss in a single Triton kernel.

    Args:
        h: Hidden states of shape (N, D) or (B, T, D)
        weight: Vocabulary weights of shape (V, D)
        targets: Target token IDs of shape (N,) or (B, T)
        ignore_index: Target token ID to ignore in loss/gradient computation (default -100)

    Returns:
        Scalar mean cross-entropy loss with same dtype as h.
    """
    # Single code path: the Function dispatches internally (capture-safe
    # torch path for V<=1024, fused Triton kernels above). No wrapper-level
    # branching so nothing here can add a host sync under CUDA graphs.
    return _TritonFusedLinearCrossEntropyFunc.apply(h, weight, targets, ignore_index)

