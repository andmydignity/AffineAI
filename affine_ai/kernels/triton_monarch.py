"""
Custom Triton Kernel: In-SRAM Fused Monarch Permutation Chain
============================================================
Fuses multi-stage diagonal scaling and permutation indexing directly
in GPU SRAM registers with analytical transposed backward pass.
Eliminates intermediate VRAM writes across all Monarch projection stages.
"""

import math
from typing import Tuple, Optional
import torch
import triton
import triton.language as tl


def precompute_monarch_composed_single(diagonals: torch.Tensor, perms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes composed 1D monomial scale W[d] and gather index P[d]
    across S stages for single Monarch permutation chain.
    """
    num_stages, D = diagonals.shape
    idx = torch.arange(D, device=perms.device, dtype=torch.long)
    W = diagonals[num_stages - 1].clone()
    for s in range(num_stages - 2, -1, -1):
        idx = perms[s][idx].long()
        W = W * diagonals[s][idx]
    P = idx.to(torch.int32)
    return W.contiguous(), P.contiguous()


def precompute_monarch_composed_fused(diagonals: torch.Tensor, perms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes composed 1D monomial scale W[m, d] and gather index P[d]
    across S stages for M branches.
    """
    num_branches, num_stages, D = diagonals.shape
    idx = torch.arange(D, device=perms.device, dtype=torch.long)
    W = diagonals[:, num_stages - 1].clone()  # [M, D]
    for s in range(num_stages - 2, -1, -1):
        idx = perms[s][idx].long()
        W = W * diagonals[:, s, idx]
    P = idx.to(torch.int32)
    return W.contiguous(), P.contiguous()


@triton.jit
def _monarch_chain_fwd_kernel(
    X, W, P, Bias, Y,
    stride_xm, stride_xd,
    stride_ym, stride_yd,
    N, D,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = offs_m < N
    mask_d = offs_d < D

    # P/W/B reused across N tiles -> cache-friendly; gather X via permuted index is streaming.
    # Use eviction_policy to keep P/W in cache while evicting streaming gather loads.
    p = tl.load(P + offs_d, mask=mask_d, other=0, eviction_policy="evict_first")
    w = tl.load(W + offs_d, mask=mask_d, other=0.0, eviction_policy="evict_first")
    b = tl.load(Bias + offs_d, mask=mask_d, other=0.0, eviction_policy="evict_first")

    xv = tl.load(
        X + offs_m[:, None] * stride_xm + p[None, :] * stride_xd,
        mask=mask_m[:, None] & mask_d[None, :], other=0.0,
        eviction_policy="evict_last",
    )
    y = b[None, :] + w[None, :] * xv
    tl.store(
        Y + offs_m[:, None] * stride_ym + offs_d[None, :] * stride_yd,
        y, mask=mask_m[:, None] & mask_d[None, :],
    )


@triton.jit
def _fused_monarch_chain_fwd_kernel(
    X, W, P, Bias, Y,
    stride_xm, stride_xd,
    stride_wm, stride_wd,
    stride_bm, stride_bd,
    stride_ym, stride_yn, stride_yd,
    N, D,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    br = tl.program_id(2)  # native 3D grid: branch index directly!

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = offs_m < N
    mask_d = offs_d < D

    p = tl.load(P + offs_d, mask=mask_d, other=0, eviction_policy="evict_first")
    w = tl.load(W + br * stride_wm + offs_d * stride_wd, mask=mask_d, other=0.0, eviction_policy="evict_first")
    b = tl.load(Bias + br * stride_bm + offs_d * stride_bd, mask=mask_d, other=0.0, eviction_policy="evict_first")

    xv = tl.load(
        X + offs_m[:, None] * stride_xm + p[None, :] * stride_xd,
        mask=mask_m[:, None] & mask_d[None, :], other=0.0,
        eviction_policy="evict_last",
    )
    y = b[None, :] + w[None, :] * xv
    tl.store(
        Y + br * stride_ym + offs_m[:, None] * stride_yn + offs_d[None, :] * stride_yd,
        y, mask=mask_m[:, None] & mask_d[None, :],
    )


def triton_monarch_chain_fwd(x: torch.Tensor, diagonals: torch.Tensor, perms: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, D = x.shape
    if not x.is_cuda or not torch.cuda.is_available():
        h = x * diagonals[0]
        for s in range(diagonals.shape[0] - 1):
            h = h[:, perms[s].long()] * diagonals[s + 1]
        return h + bias

    W, P = precompute_monarch_composed_single(diagonals, perms)
    out = torch.empty((N, D), device=x.device, dtype=x.dtype)
    BLOCK_M, BLOCK_D = 64, 64
    grid = ((N + BLOCK_M - 1) // BLOCK_M, (D + BLOCK_D - 1) // BLOCK_D)
    _monarch_chain_fwd_kernel[grid](
        x, W, P, bias, out,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        N, D,
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D, num_warps=4,
    )
    return out


def triton_fused_monarch_chain_fwd(x: torch.Tensor, diagonals: torch.Tensor, perms: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, D = x.shape
    M = diagonals.shape[0]
    if not x.is_cuda or not torch.cuda.is_available():
        h = x.unsqueeze(0) * diagonals[:, 0].unsqueeze(1)
        for s in range(diagonals.shape[1] - 1):
            h = h[:, :, perms[s].long()] * diagonals[:, s + 1].unsqueeze(1)
        return h + bias.unsqueeze(1)

    W, P = precompute_monarch_composed_fused(diagonals, perms)
    out = torch.empty((M, N, D), device=x.device, dtype=x.dtype)
    BLOCK_M, BLOCK_D = 64, 64
    grid = ((N + BLOCK_M - 1) // BLOCK_M, (D + BLOCK_D - 1) // BLOCK_D, M)
    _fused_monarch_chain_fwd_kernel[grid](
        x, W, P, bias, out,
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1),
        bias.stride(0), bias.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        N, D,
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D, num_warps=4,
    )
    return out


class TritonMonarchChainFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        diagonals: torch.Tensor,
        perms: torch.Tensor,
        inv_perms: torch.Tensor,
        bias: torch.Tensor
    ) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1]).to(diagonals.dtype)
        num_stages = diagonals.shape[0]

        out = triton_monarch_chain_fwd(x_flat, diagonals, perms, bias)

        ctx.save_for_backward(x_flat, diagonals, perms, inv_perms)
        ctx.num_stages = num_stages
        ctx.orig_shape = orig_shape
        ctx.orig_dtype = x.dtype
        return out.to(x.dtype).reshape(*orig_shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_flat, diagonals, perms, inv_perms = ctx.saved_tensors
        num_stages = ctx.num_stages

        h_list = []
        if num_stages > 1:
            h_list.append(x_flat * diagonals[0])
            for s in range(num_stages - 2):
                h_next = h_list[-1][:, perms[s]] * diagonals[s + 1]
                h_list.append(h_next)

        go_flat = grad_out.reshape(-1, grad_out.shape[-1]).to(diagonals.dtype)
        g_bias = go_flat.sum(0)
        g_diagonals = torch.empty_like(diagonals)
        gh = go_flat

        for s in range(num_stages - 1, 0, -1):
            h_perm = h_list[s - 1][:, perms[s - 1]]
            g_diagonals[s] = (gh * h_perm).sum(0)
            gh = (gh * diagonals[s])[:, inv_perms[s - 1]]

        g_diagonals[0] = (gh * x_flat).sum(0)
        gx = (gh * diagonals[0]).to(ctx.orig_dtype)
        return gx.reshape(*ctx.orig_shape), g_diagonals, None, None, g_bias


class TritonFusedMonarchChainFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        diagonals: torch.Tensor,
        perms: torch.Tensor,
        inv_perms: torch.Tensor,
        bias: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1]).to(diagonals.dtype)
        num_branches = diagonals.shape[0]
        num_stages = diagonals.shape[1]

        out = triton_fused_monarch_chain_fwd(x_flat, diagonals, perms, bias) # [M, N, dim]

        ctx.save_for_backward(x_flat, diagonals, perms, inv_perms)
        ctx.num_branches = num_branches
        ctx.num_stages = num_stages
        ctx.orig_shape = orig_shape
        ctx.orig_dtype = x.dtype
        return tuple(out[m].to(x.dtype).reshape(*orig_shape) for m in range(num_branches))

    @staticmethod
    def backward(ctx, *grad_outs):
        x_flat, diagonals, perms, inv_perms = ctx.saved_tensors
        num_stages = ctx.num_stages

        h_list = []
        if num_stages > 1:
            h_list.append(x_flat.unsqueeze(0) * diagonals[:, 0].unsqueeze(1))
            for s in range(num_stages - 2):
                h_next = h_list[-1][:, :, perms[s]] * diagonals[:, s + 1].unsqueeze(1)
                h_list.append(h_next)

        g_stack = torch.stack([g.reshape(-1, g.shape[-1]).to(diagonals.dtype) for g in grad_outs], dim=0) # [M, N, dim]
        g_bias = g_stack.sum(1)
        g_diagonals = torch.zeros_like(diagonals)
        gh = g_stack

        for s in range(num_stages - 1, 0, -1):
            h_perm = h_list[s - 1][:, :, perms[s - 1]]
            g_diagonals[:, s] = (gh * h_perm).sum(1)
            gh = (gh * diagonals[:, s].unsqueeze(1))[:, :, inv_perms[s - 1]]

        g_diagonals[:, 0] = (gh * x_flat.unsqueeze(0)).sum(1)
        gx = (gh * diagonals[:, 0].unsqueeze(1)).sum(0).to(ctx.orig_dtype)
        return gx.reshape(*ctx.orig_shape), g_diagonals, None, None, g_bias


def triton_monarch_chain(
    x: torch.Tensor,
    diagonals: torch.Tensor,
    perms: torch.Tensor,
    inv_perms: torch.Tensor,
    bias: torch.Tensor
) -> torch.Tensor:
    return TritonMonarchChainFunction.apply(x, diagonals, perms, inv_perms, bias)


def triton_fused_monarch_chain(
    x: torch.Tensor,
    diagonals: torch.Tensor,
    perms: torch.Tensor,
    inv_perms: torch.Tensor,
    bias: torch.Tensor
) -> Tuple[torch.Tensor, ...]:
    return TritonFusedMonarchChainFunction.apply(x, diagonals, perms, inv_perms, bias)
