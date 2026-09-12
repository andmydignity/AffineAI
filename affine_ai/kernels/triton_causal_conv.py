"""
Triton Causal Depthwise Conv1D (CUDA).
======================================
High-performance MatMul-free depthwise 1D causal convolution kernel.
Operates directly on contiguous [B, T, D] tensors in registers, completely
eliminating transpose(1, 2) operations and DRAM padding allocations.
Provides forward pass and exact autograd backward for dx, dw, and dbias.
CPU tensors gracefully fall back to PyTorch F.conv1d with causal padding.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None  # type: ignore
    tl = None  # type: ignore


_CAUSAL_CONV_CONFIGS = [
    triton.Config({"BLOCK_T": 32, "BLOCK_D": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_T": 64, "BLOCK_D": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_T": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_T": 128, "BLOCK_D": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_T": 128, "BLOCK_D": 64}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=_CAUSAL_CONV_CONFIGS, key=["T", "D"])
@triton.jit
def _causal_conv1d_fwd_kernel(
    X, W, Bias, Out,
    stride_xb, stride_xt, stride_xd,
    stride_outb, stride_outt, stride_outd,
    B, T, D,
    HAS_BIAS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
    if HAS_BIAS:
        bias = tl.load(Bias + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        acc += bias[None, :]

    for k in range(K):
        # Causal delay: token at offs_t needs input from offs_t - (K - 1 - k)
        t_in = offs_t - (K - 1 - k)
        mask = (t_in >= 0) & (offs_t < T)
        t_in_clamped = tl.maximum(t_in, 0)
        x_ptrs = X + pid_b * stride_xb + t_in_clamped[:, None] * stride_xt + offs_d[None, :] * stride_xd
        w_ptrs = W + offs_d * K + k
        x_val = tl.load(x_ptrs, mask=mask[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        w_val = tl.load(w_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        acc += x_val * w_val[None, :]

    mask_out = (offs_t[:, None] < T) & mask_d[None, :]
    out_ptrs = Out + pid_b * stride_outb + offs_t[:, None] * stride_outt + offs_d[None, :] * stride_outd
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=mask_out)


@triton.autotune(configs=_CAUSAL_CONV_CONFIGS, key=["T", "D"])
@triton.jit
def _causal_conv1d_bwd_dx_kernel(
    dY, W, dX,
    stride_dyb, stride_dyt, stride_dyd,
    stride_dxb, stride_dxt, stride_dxd,
    B, T, D,
    K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

    for k in range(K):
        # Anti-causal future gradient: output at offs_t + (K - 1 - k) was influenced by input at offs_t
        t_out = offs_t + (K - 1 - k)
        mask = (t_out < T) & (offs_t < T)
        t_out_clamped = tl.minimum(t_out, T - 1)
        dy_ptrs = dY + pid_b * stride_dyb + t_out_clamped[:, None] * stride_dyt + offs_d[None, :] * stride_dyd
        w_ptrs = W + offs_d * K + k
        dy_val = tl.load(dy_ptrs, mask=mask[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        w_val = tl.load(w_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        acc += dy_val * w_val[None, :]

    mask_dx = (offs_t[:, None] < T) & mask_d[None, :]
    dx_ptrs = dX + pid_b * stride_dxb + offs_t[:, None] * stride_dxt + offs_d[None, :] * stride_dxd
    tl.store(dx_ptrs, acc.to(dX.dtype.element_ty), mask=mask_dx)


@triton.autotune(configs=_CAUSAL_CONV_CONFIGS, key=["T", "D"], reset_to_zero=["dW"])
@triton.jit
def _causal_conv1d_bwd_dw_fused_kernel(
    dY, X, dW,
    stride_dyb, stride_dyt, stride_dyd,
    stride_xb, stride_xt, stride_xd,
    stride_dwd, stride_dwk,
    B, T, D,
    K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    mask_t = offs_t < T

    dy_ptrs = dY + pid_b * stride_dyb + offs_t[:, None] * stride_dyt + offs_d[None, :] * stride_dyd
    dy = tl.load(dy_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    for k in range(K):
        t_in = offs_t - (K - 1 - k)
        mask_in = (t_in >= 0) & mask_t
        t_in_clamped = tl.maximum(t_in, 0)
        x_ptrs = X + pid_b * stride_xb + t_in_clamped[:, None] * stride_xt + offs_d[None, :] * stride_xd
        x = tl.load(x_ptrs, mask=mask_in[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        dw_k = tl.sum(dy * x, axis=0)
        dw_ptrs = dW + offs_d * stride_dwd + k * stride_dwk
        tl.atomic_add(dw_ptrs, dw_k, mask=mask_d)


class TritonCausalConv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not x.is_cuda or triton is None:
            # Fallback for CPU
            K = w.shape[-1]
            x_pad = F.pad(x.transpose(1, 2), (K - 1, 0))
            w_2d = w.unsqueeze(1) if w.ndim == 2 else w
            out = F.conv1d(x_pad, w_2d, bias, groups=x.shape[-1]).transpose(1, 2)
            ctx.save_for_backward(x, w)
            ctx.has_bias = bias is not None
            ctx.K = K
            ctx.is_cpu = True
            return out

        B, T, D = x.shape
        K = w.shape[-1]
        w_flat = w.reshape(D, K)
        out = torch.empty_like(x)
        grid = lambda META: (triton.cdiv(T, META["BLOCK_T"]), triton.cdiv(D, META["BLOCK_D"]), B)
        has_bias = bias is not None

        _causal_conv1d_fwd_kernel[grid](
            x, w_flat, bias if has_bias else x, out,
            x.stride(0), x.stride(1), x.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            B, T, D,
            HAS_BIAS=has_bias, K=K,
        )

        ctx.save_for_backward(x, w_flat)
        ctx.has_bias = has_bias
        ctx.K = K
        ctx.w_shape = w.shape
        ctx.is_cpu = False
        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        x, w = ctx.saved_tensors
        B, T, D = x.shape
        K = ctx.K

        if getattr(ctx, "is_cpu", False):
            dx = torch.empty_like(x) if ctx.needs_input_grad[0] else None
            dw = torch.zeros_like(w) if ctx.needs_input_grad[1] else None
            dbias = dy.sum(dim=(0, 1)).to(w.dtype) if (ctx.has_bias and ctx.needs_input_grad[2]) else None
            if dx is not None or dw is not None:
                x_pad = F.pad(x.transpose(1, 2), (K - 1, 0))
                w_2d = w.unsqueeze(1) if w.ndim == 2 else w
                with torch.enable_grad():
                    x_p = x_pad.detach().requires_grad_(dx is not None)
                    w_p = w_2d.detach().requires_grad_(dw is not None)
                    y = F.conv1d(x_p, w_p, None, groups=D).transpose(1, 2)
                    grads = torch.autograd.grad(y, [t for t in (x_p, w_p) if t.requires_grad], dy)
                idx = 0
                if dx is not None:
                    g_pad = grads[idx].transpose(1, 2)
                    dx.copy_(g_pad[:, K - 1 :])
                    idx += 1
                if dw is not None:
                    dw.copy_(grads[idx].view_as(w))
            return dx, dw, dbias

        dx = torch.empty_like(x) if ctx.needs_input_grad[0] else None
        dbias = dy.sum(dim=(0, 1)).to(w.dtype) if (ctx.has_bias and ctx.needs_input_grad[2]) else None
        dw = None

        if dx is not None:
            grid_dx = lambda META: (triton.cdiv(T, META["BLOCK_T"]), triton.cdiv(D, META["BLOCK_D"]), B)
            _causal_conv1d_bwd_dx_kernel[grid_dx](
                dy, w, dx,
                dy.stride(0), dy.stride(1), dy.stride(2),
                dx.stride(0), dx.stride(1), dx.stride(2),
                B, T, D,
                K=K,
            )

        if ctx.needs_input_grad[1]:
            # Accumulate weight gradients in FP32 to avoid catastrophic precision loss in atomicAdd
            dw_fp32 = torch.zeros(D, K, device=w.device, dtype=torch.float32)
            grid_dw = lambda META: (triton.cdiv(D, META["BLOCK_D"]), triton.cdiv(T, META["BLOCK_T"]), B)
            _causal_conv1d_bwd_dw_fused_kernel[grid_dw](
                dy, x, dw_fp32,
                dy.stride(0), dy.stride(1), dy.stride(2),
                x.stride(0), x.stride(1), x.stride(2),
                dw_fp32.stride(0), dw_fp32.stride(1),
                B, T, D,
                K=K,
            )
            dw = dw_fp32.to(w.dtype).view(ctx.w_shape)

        return dx, dw, dbias

        return dx, dw, dbias


def triton_causal_conv1d(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies 1D causal depthwise convolution over [B, T, D] with weight [D, K] or [D, 1, K].
    Uses Triton on CUDA, PyTorch on CPU.
    """
    w_2d = w.squeeze(1) if w.ndim == 3 else w
    return TritonCausalConv1dFunction.apply(x, w_2d, bias)
