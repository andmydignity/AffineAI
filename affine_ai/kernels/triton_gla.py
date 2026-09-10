"""Triton Gated Linear Associative (GLA) Sequence Mixer (CUDA).

Includes native 2D/3D fused kernels for GLA decay and linear attention,
re-exporting Monarch permutation chain kernels from triton_monarch.
"""

import math
import warnings
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _is_turing() -> bool:
    """Turing sm_75 detection: 64KB SMEM, FP16-only, no BF16.

    Prefer canonical ``affine_ai.kernels._IS_TURING`` when available to avoid
    redundant ``get_device_capability`` calls; fall back to direct
    capability probe ``(7,5) <= cap < (8,0)``.
    """
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

# Re-export Monarch permutation chain functions and kernels (Issue 17)
# NOTE: Circular-import risk — triton_monarch must not import from triton_gla.
# This re-export is one-way only; keep it that way.
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



_GLA_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 16, "BLOCK_J": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 32, "BLOCK_J": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 32, "BLOCK_J": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 32, "BLOCK_J": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK": 64, "BLOCK_J": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 64, "BLOCK_J": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 64, "BLOCK_J": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK": 64, "BLOCK_J": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 64, "BLOCK_J": 64}, num_warps=8, num_stages=4),
]

# Turing sm_75: 64KB SMEM cap, BLOCK <=64, num_warps 2/4 max, avoid BLOCK 128.
# Filter autotune configs on Turing to keep only BLOCK 32/64 (and 16) with warps <=4.
if _is_turing():
    _GLA_AUTOTUNE_CONFIGS = [
        c for c in _GLA_AUTOTUNE_CONFIGS
        if c.kwargs.get("BLOCK", 32) <= 64 and c.kwargs.get("BLOCK_J", 32) <= 64 and c.num_warps <= 4
    ]


@triton.autotune(configs=_GLA_AUTOTUNE_CONFIGS, key=["T", "B", "H"])
@triton.jit
def _gla_decay_kernel(
    Cum, Decay,
    stride_cb, stride_ch, stride_ct,
    stride_db, stride_dh, stride_di, stride_dj,
    B, H, T,
    BLOCK: tl.constexpr = 32,
    BLOCK_J: tl.constexpr = 32,
    CLAMP_MIN: tl.constexpr = -30.0,
):
    """
    3D-tiled GLA decay kernel: Decay[b,h,i,j] = exp(clamp(cum[i]-cum[j], CLAMP_MIN, 0)) * (i>=j).
    Grid: (cdiv(T,BLOCK), cdiv(T,BLOCK_J), B*H). Uses tl.make_block_ptr for coalesced
    Cum/Decay accesses; BLOCK/BLOCK_J <=64 required (64KB SMEM on sm_75, 99KB usable on sm_80+).
    CLAMP_MIN is tl.constexpr: -11 for fp16 (subnormal threshold), -30 for fp32/bf16.
    """
    tile_i = tl.program_id(0)
    tile_j = tl.program_id(1)
    bh = tl.program_id(2)
    b = bh // H
    h = bh % H
    offs_i = tile_i * BLOCK + tl.arange(0, BLOCK)
    offs_j = tile_j * BLOCK_J + tl.arange(0, BLOCK_J)
    mask_i = offs_i < T
    mask_j = offs_j < T
    if tile_j > tile_i:
        val = tl.zeros((BLOCK, BLOCK_J), dtype=tl.float32)
        out_block_ptr = tl.make_block_ptr(
            base=Decay + b * stride_db + h * stride_dh,
            shape=(T, T),
            strides=(stride_di, stride_dj),
            offsets=(tile_i * BLOCK, tile_j * BLOCK_J),
            block_shape=(BLOCK, BLOCK_J),
            order=(1, 0),
        )
        tl.store(out_block_ptr, val, boundary_check=(0, 1))
        return
    cum_block_ptr_i = tl.make_block_ptr(
        base=Cum + b * stride_cb + h * stride_ch,
        shape=(T,),
        strides=(stride_ct,),
        offsets=(tile_i * BLOCK,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    cum_block_ptr_j = tl.make_block_ptr(
        base=Cum + b * stride_cb + h * stride_ch,
        shape=(T,),
        strides=(stride_ct,),
        offsets=(tile_j * BLOCK_J,),
        block_shape=(BLOCK_J,),
        order=(0,),
    )
    ci = tl.load(cum_block_ptr_i, boundary_check=(0,), eviction_policy="evict_last")
    cj = tl.load(cum_block_ptr_j, boundary_check=(0,), eviction_policy="evict_last")
    diff = ci[:, None] - cj[None, :]
    diff = tl.minimum(diff, 0.0)
    diff = tl.maximum(diff, CLAMP_MIN)
    causal_mask = offs_i[:, None] >= offs_j[None, :]
    valid_mask = mask_i[:, None] & mask_j[None, :] & causal_mask
    val = tl.exp(diff)
    val = tl.where(valid_mask, val, 0.0)
    out_block_ptr = tl.make_block_ptr(
        base=Decay + b * stride_db + h * stride_dh,
        shape=(T, T),
        strides=(stride_di, stride_dj),
        offsets=(tile_i * BLOCK, tile_j * BLOCK_J),
        block_shape=(BLOCK, BLOCK_J),
        order=(1, 0),
    )
    tl.store(out_block_ptr, val, boundary_check=(0, 1))


@triton.jit
def _gla_decay_kernel_raw(
    Cum, Decay,
    stride_cb, stride_ch, stride_ct,
    stride_db, stride_dh, stride_di, stride_dj,
    B, H, T,
    BLOCK: tl.constexpr = 32,
    BLOCK_J: tl.constexpr = 32,
    CLAMP_MIN: tl.constexpr = -30.0,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = B.to(tl.int64) * H.to(tl.int64) * T.to(tl.int64) * T.to(tl.int64)
    offs_i64 = offs.to(tl.int64)
    mask = offs_i64 < total
    tmp = offs_i64
    tj = (tmp % T.to(tl.int64)).to(tl.int32)
    tmp = tmp // T.to(tl.int64)
    ti = (tmp % T.to(tl.int64)).to(tl.int32)
    tmp = tmp // T.to(tl.int64)
    th = (tmp % H.to(tl.int64)).to(tl.int32)
    tb = (tmp // H.to(tl.int64)).to(tl.int32)
    ci = tl.load(Cum + tb * stride_cb + th * stride_ch + ti * stride_ct, mask=mask, other=0.0)
    cj = tl.load(Cum + tb * stride_cb + th * stride_ch + tj * stride_ct, mask=mask, other=0.0)
    diff = ci - cj
    diff = tl.minimum(diff, 0.0)
    diff = tl.maximum(diff, CLAMP_MIN)
    m = tj <= ti
    val = tl.exp(diff)
    val = tl.where(m & mask, val, 0.0)
    out_ptr = Decay + tb * stride_db + th * stride_dh + ti * stride_di + tj * stride_dj
    tl.store(out_ptr, val, mask=mask)

_gla_decay_kernel_tuned = _gla_decay_kernel


class _GlaKernelDispatcher:
    def __getitem__(self, grid):
        def _run(*args, **kwargs):
            block = kwargs.get("BLOCK", None)
            if block is not None:
                return _gla_decay_kernel_raw[grid](*args, **kwargs)
            block_j = kwargs.get("BLOCK_J", None)
            if block_j is not None:
                return _gla_decay_kernel_raw[grid](*args, **kwargs)
            return _gla_decay_kernel_tuned[grid](*args, **kwargs)
        return _run


_gla_decay_kernel = _GlaKernelDispatcher()


def triton_gla_decay_fwd(cum_log_gam: torch.Tensor, clamp_min: float = -30.0) -> torch.Tensor:
    """
    Materializes dense decay matrix [B, H, T, T] (O(T^2) memory).
    Use only at short T (e.g. T <= 512); for long T use the chunked
    linear-attention path. Limit: T<=1024 else O(T^2) alloc explodes
    (B*H*T*T elements, e.g. B=8 H=8 T=2048 -> 2B floats ~8GB).
    """
    assert cum_log_gam.ndim == 3, f"cum_log_gam must be [B,H,T], got {cum_log_gam.shape}"
    assert cum_log_gam.is_contiguous() or cum_log_gam.stride(-1) == 1, "cum last dim should be contiguous for coalesced loads"
    B, H, T = cum_log_gam.shape
    if T > 1024:
        raise ValueError(f"T={T} exceeds 1024 limit for dense [B,H,T,T] ({B*H*T*T} floats); use chunked path")
    if B * H > 65535 and (T + 32 - 1) // 32 > 1:
        warnings.warn(f"B*H={B*H} exceeds 65535 z-grid limit with current T; grid z-dimension will be split", stacklevel=2)
    assert cum_log_gam.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert T > 0 and B > 0 and H > 0
    out = torch.empty((B, H, T, T), device=cum_log_gam.device, dtype=torch.float32)
    grid = lambda META: ((T + META["BLOCK"] - 1) // META["BLOCK"], (T + META["BLOCK_J"] - 1) // META["BLOCK_J"], B * H)
    _gla_decay_kernel[grid](
        cum_log_gam, out,
        cum_log_gam.stride(0), cum_log_gam.stride(1), cum_log_gam.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, T, CLAMP_MIN=clamp_min)
    return out


class TritonGLADecayFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gamma: torch.Tensor) -> torch.Tensor:
        # Clamp max=1.0 on gamma to prevent positive log growth in BF16 (Issue 15)
        # Clamp min=1e-5 to prevent log underflow (Issue 16)
        # Accumulation precision: cum computed in float32, final out downcast to gamma.dtype (bf16/fp16) — fp32 compute then to(dtype)
        # Turing sm_75: bf16 fallback to fp16 (acc fp32), warn once.
        if _is_turing() and gamma.dtype == torch.bfloat16:
            warnings.warn("Turing sm_75: bf16 not supported, treating as fp16 (acc fp32) in TritonGLADecayFunction", stacklevel=3)
        log_gam = torch.log(gamma.float().clamp(min=1e-5, max=1.0))
        cum = torch.cumsum(log_gam, dim=-1)  # float32 cumsum; bf16/fp16 would lose precision
        # dtype-dependent underflow clamp: fp16 subnormal floor ~ -11, fp32/bf16 ~ -30
        # Turing bf16 is treated as fp16 (clamp -11, fp32 acc).
        if _is_turing() and gamma.dtype == torch.bfloat16:
            clamp_min = -11.0
        else:
            clamp_min = -11.0 if gamma.dtype == torch.float16 else -30.0
        assert gamma.ndim == 3, f"gamma must be [B,H,T], got {gamma.shape}"
        if gamma.is_cuda and torch.cuda.is_available():
            out = triton_gla_decay_fwd(cum.contiguous(), clamp_min=clamp_min)
        else:
            T = cum.shape[-1]
            decay_diff = (cum.unsqueeze(-1) - cum.unsqueeze(-2)).clamp(min=clamp_min, max=0.0)
            mask = torch.tril(torch.ones(T, T, device=cum.device, dtype=torch.bool))
            out = torch.where(mask, torch.exp(decay_diff), torch.zeros_like(decay_diff))
        ctx.save_for_backward(gamma, out)
        # Unified dtype handling: both CUDA and CPU paths return gamma.dtype for parity
        return out.to(gamma.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not ctx.needs_input_grad[0]:
            return None
        gamma, out = ctx.saved_tensors
        # Explicit analytical backward adjoint (Issue 20)
        # g_c cumsum stability: O(T) error accumulation in float32; optional float64 for long T
        M = grad_output.to(out.dtype) * out
        g_c = M.sum(dim=-1) - M.sum(dim=-2)  # float32 reduction; for T>4k consider float64
        g_log_gam = g_c.flip(-1).cumsum(-1).flip(-1)  # cumsum error O(T); float64 alternative: g_c.double().flip(-1).cumsum(-1).flip(-1).float()
        g_gam = g_log_gam / gamma.float().clamp(min=1e-5)
        mask = (gamma >= 1e-5) & (gamma <= 1.0)
        g_gam = torch.where(mask, g_gam, torch.zeros_like(g_gam))
        # Unified dtype: return gamma.dtype
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
    Gated Linear Attention with chunked path for long sequences.

    - For T <= chunk_size: direct O(T^2) decay materialization (no chunking).
    - For T > chunk_size: delegates to FusedGLAAnalyticalCUDA which materializes
      chunk states [B, H, NC, D, D] (NC = ceil(T/chunk_size)) — NOT strictly O(T)
      VRAM; peak is B*H*NC*D*D. Chunking reduces T^2 to (T/NC)^2 per chunk plus
      state overhead.
    - Guarded against FP16 subnormal underflow (<6e-5) and BF16 log growth (Issues 15 & 16).
    - Unified dtype: returns q.dtype (gamma.dtype parity ensured).
    """
    B, H, T, D = q.shape
    dtype = q.dtype
    # Turing sm_75: bf16 -> fp16 fallback, warn
    if _is_turing() and dtype == torch.bfloat16:
        warnings.warn("Turing sm_75: bf16 not supported, treating as fp16 (acc fp32) in triton_gla_linear_attention", stacklevel=2)
        eps = 1e-4
        clamp_min = -11.0
    else:
        eps = 1e-4 if dtype == torch.float16 else 1e-5
        # dtype-dependent clamp: fp16 subnormal threshold ~ -11, fp32/bf16 ~ -30
        # Turing bf16 treated as fp16 above.
        clamp_min = -11.0 if dtype == torch.float16 else -30.0

    if T <= chunk_size:
        log_gam = torch.log(gamma.float().clamp(min=1e-5, max=1.0))
        cum_log_gam = torch.cumsum(log_gam, dim=-1)  # float32 compute then downcast
        decay_diff = (cum_log_gam.unsqueeze(-1) - cum_log_gam.unsqueeze(-2)).clamp(min=clamp_min, max=0.0)
        causal_mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
        decay_mat = torch.where(causal_mask, torch.exp(decay_diff), torch.zeros_like(decay_diff)).to(dtype)
        scores = torch.matmul(q, k.transpose(-1, -2)) * decay_mat
        num = torch.matmul(scores, v)
        den = scores.sum(dim=-1, keepdim=True).clamp(min=eps)
        return (num / den).to(gamma.dtype)

    # Chunked associative scan: delegates to FusedGLAAnalyticalCUDA; peak VRAM B*H*NC*D*D
    pad_len = (chunk_size - (T % chunk_size)) % chunk_size
    if pad_len > 0:
        q_pad = F.pad(q, (0, 0, 0, pad_len))
        k_pad = F.pad(k, (0, 0, 0, pad_len))
        v_pad = F.pad(v, (0, 0, 0, pad_len))
        gamma_pad = F.pad(gamma, (0, pad_len), value=1.0)
        # Pad value gamma=1.0 leaks dummy contribution into cum_log_gam and M_mat if
        # dummy k/v not zeroed. Ensure dummy positions are masked: zero k_pad/v_pad tail
        # and rely on NC slicing below. Coupling: FusedGLAAnalyticalCUDA must mask padded
        # cum_log tail or q_pad tail must be zero to avoid dummy state pollution.
        # Zero dummy k/v to prevent dummy keys polluting M_mat.
        q_pad[:, :, T:, :].zero_()
        k_pad[:, :, T:, :].zero_()
        v_pad[:, :, T:, :].zero_()
        # Also zero cum tail effect: gamma_pad tail already 1.0 so log=0, but M_mat tail
        # stays clean only because k/v tail zeroed above. Document coupling.
    else:
        q_pad, k_pad, v_pad, gamma_pad = q, k, v, gamma

    from affine_ai.core.associative import FusedGLAAnalyticalCUDA
    out_pad = FusedGLAAnalyticalCUDA.apply(q_pad, k_pad, v_pad, gamma_pad, chunk_size)
    if pad_len > 0:
        # Slice NC correctly: remove padded time steps from output
        return out_pad[:, :, :T, :].to(gamma.dtype)
    return out_pad.to(gamma.dtype)
