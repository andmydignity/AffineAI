"""Triton Gated Linear Associative (GLA) Sequence Mixer (CUDA).

Includes native 2D/3D fused kernels for GLA decay and linear attention,
re-exporting Monarch permutation chain kernels from triton_monarch.
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Re-export Monarch permutation chain functions and kernels (Issue 17)
from affine_ai.kernels.triton_monarch import (
    _monarch_chain_fwd_kernel,
    _fused_monarch_chain_fwd_kernel,
    triton_monarch_chain_fwd,
    triton_fused_monarch_chain_fwd,
    TritonMonarchChainFunction,
    TritonFusedMonarchChainFunction,
    triton_monarch_chain,
    triton_fused_monarch_chain,
)



@triton.jit
def _gla_decay_kernel(
    Cum, Decay,
    stride_cb, stride_ch, stride_ct,
    stride_db, stride_dh, stride_di, stride_dj,
    B, H, T,
    BLOCK: tl.constexpr = 32,
    BLOCK_J: tl.constexpr = 32,
):
    if BLOCK <= 64:
        # Native 3D grid: (cdiv(T, BLOCK), cdiv(T, BLOCK_J), B * H) (Issue 21)
        tile_i = tl.program_id(0)
        tile_j = tl.program_id(1)
        bh = tl.program_id(2)
        b = bh // H
        h = bh % H

        offs_i = tile_i * BLOCK + tl.arange(0, BLOCK)
        offs_j = tile_j * BLOCK_J + tl.arange(0, BLOCK_J)

        mask_i = offs_i < T
        mask_j = offs_j < T

        # Block causal skipping: explicitly store 0.0 for anti-causal blocks to allow torch.empty
        if tile_j > tile_i:
            val = tl.zeros((BLOCK, BLOCK_J), dtype=tl.float32)
            out_ptr = Decay + b * stride_db + h * stride_dh + offs_i[:, None] * stride_di + offs_j[None, :] * stride_dj
            tl.store(out_ptr, val, mask=mask_i[:, None] & mask_j[None, :])
            return

        cum_base = Cum + b * stride_cb + h * stride_ch
        ci = tl.load(cum_base + offs_i * stride_ct, mask=mask_i, other=0.0)
        cj = tl.load(cum_base + offs_j * stride_ct, mask=mask_j, other=0.0)

        diff = ci[:, None] - cj[None, :]
        diff = tl.minimum(diff, 0.0)
        diff = tl.maximum(diff, -30.0)  # Underflow protection for FP16 (Issue 16)

        causal_mask = offs_i[:, None] >= offs_j[None, :]
        valid_mask = mask_i[:, None] & mask_j[None, :] & causal_mask

        val = tl.exp(diff)
        val = tl.where(valid_mask, val, 0.0)

        out_ptr = Decay + b * stride_db + h * stride_dh + offs_i[:, None] * stride_di + offs_j[None, :] * stride_dj
        tl.store(out_ptr, val, mask=mask_i[:, None] & mask_j[None, :])
    else:
        # 1D grid backward compatibility
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        total = B * H * T * T
        mask = offs < total
        tmp = offs
        j = tmp % T
        tmp = tmp // T
        i = tmp % T
        tmp = tmp // T
        h = tmp % H
        b = tmp // H
        ci = tl.load(
            Cum + b * stride_cb + h * stride_ch + i * stride_ct,
            mask=mask, other=0.0,
        )
        cj = tl.load(
            Cum + b * stride_cb + h * stride_ch + j * stride_ct,
            mask=mask, other=0.0,
        )
        diff = ci - cj
        diff = tl.minimum(diff, 0.0)
        diff = tl.maximum(diff, -30.0)
        m = j <= i
        val = tl.exp(diff)
        val = tl.where(m & mask, val, 0.0)
        out_ptr = Decay + b * stride_db + h * stride_dh + i * stride_di + j * stride_dj
        tl.store(out_ptr, val, mask=mask)


def triton_gla_decay_fwd(cum_log_gam: torch.Tensor) -> torch.Tensor:
    B, H, T = cum_log_gam.shape
    out = torch.empty((B, H, T, T), device=cum_log_gam.device, dtype=torch.float32)
    BLOCK = 64
    grid = ((T + BLOCK - 1) // BLOCK, (T + BLOCK - 1) // BLOCK, B * H)
    _gla_decay_kernel[grid](
        cum_log_gam, out,
        cum_log_gam.stride(0), cum_log_gam.stride(1), cum_log_gam.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, T, BLOCK=BLOCK, BLOCK_J=BLOCK, num_warps=4)
    return out


class TritonGLADecayFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gamma: torch.Tensor) -> torch.Tensor:
        # Clamp max=1.0 on gamma to prevent positive log growth in BF16 (Issue 15)
        # Clamp min=1e-5 to prevent log underflow (Issue 16)
        log_gam = torch.log(gamma.float().clamp(min=1e-5, max=1.0))
        cum = torch.cumsum(log_gam, dim=-1)
        if gamma.is_cuda and torch.cuda.is_available():
            out = triton_gla_decay_fwd(cum.contiguous())
        else:
            T = cum.shape[-1]
            decay_diff = (cum.unsqueeze(-1) - cum.unsqueeze(-2)).clamp(min=-30.0, max=0.0)
            mask = torch.tril(torch.ones(T, T, device=cum.device, dtype=torch.bool))
            out = torch.where(mask, torch.exp(decay_diff), torch.zeros_like(decay_diff))
        ctx.save_for_backward(gamma, out)
        return out.to(gamma.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not ctx.needs_input_grad[0]:
            return None
        gamma, out = ctx.saved_tensors
        # Explicit analytical backward adjoint (Issue 20)
        M = grad_output.to(out.dtype) * out
        g_c = M.sum(dim=-1) - M.sum(dim=-2)
        g_log_gam = g_c.flip(-1).cumsum(-1).flip(-1)
        g_gam = g_log_gam / gamma.float().clamp(min=1e-5)
        mask = (gamma >= 1e-5) & (gamma <= 1.0)
        g_gam = torch.where(mask, g_gam, torch.zeros_like(g_gam))
        return g_gam.to(gamma.dtype)


def triton_gla_decay(gamma: torch.Tensor) -> torch.Tensor:
    return TritonGLADecayFunction.apply(gamma)


def triton_gla_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gamma: torch.Tensor,
    chunk_size: int = 64
) -> torch.Tensor:
    """
    Gated Linear Attention avoiding [B, H, T, T] dense tensor materialization (Issue 18).
    Strictly O(T) linear / chunked memory complexity to prevent VRAM OOM on large T.
    - Intra-chunk computation on chunks of size C <= chunk_size.
    - Inter-chunk state accumulation in [B, H, D, D] state matrices.
    - Guarded against FP16 subnormal underflow (<6e-5) and BF16 log growth (Issues 15 & 16).
    """
    B, H, T, D = q.shape
    dtype = q.dtype
    eps = 1e-4 if dtype == torch.float16 else 1e-5

    if T <= chunk_size:
        log_gam = torch.log(gamma.float().clamp(min=1e-5, max=1.0))
        cum_log_gam = torch.cumsum(log_gam, dim=-1)
        decay_diff = (cum_log_gam.unsqueeze(-1) - cum_log_gam.unsqueeze(-2)).clamp(min=-30.0, max=0.0)
        causal_mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
        decay_mat = torch.where(causal_mask, torch.exp(decay_diff), torch.zeros_like(decay_diff)).to(dtype)
        scores = torch.matmul(q, k.transpose(-1, -2)) * decay_mat
        num = torch.matmul(scores, v)
        den = scores.sum(dim=-1, keepdim=True).clamp(min=eps)
        return num / den

    # Strictly chunked associative scan: O(T) memory
    pad_len = (chunk_size - (T % chunk_size)) % chunk_size
    if pad_len > 0:
        q_pad = F.pad(q, (0, 0, 0, pad_len))
        k_pad = F.pad(k, (0, 0, 0, pad_len))
        v_pad = F.pad(v, (0, 0, 0, pad_len))
        gamma_pad = F.pad(gamma, (0, pad_len), value=1.0)
    else:
        q_pad, k_pad, v_pad, gamma_pad = q, k, v, gamma

    from affine_ai.core.associative import FusedGLAAnalyticalCUDA
    out_pad = FusedGLAAnalyticalCUDA.apply(q_pad, k_pad, v_pad, gamma_pad, chunk_size)
    if pad_len > 0:
        return out_pad[:, :, :T, :]
    return out_pad
