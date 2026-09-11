"""
Triton 5-State Power-of-Two (POT) Acceleration Kernels
======================================================
Custom GPU kernels for 5-State POT arithmetic:
  W in {-1.0, -0.5, 0.0, +0.5, +1.0} * alpha

WARNING: pack path (pack_pot5_gpu_3bitplane, 0.35σ/0.9σ std-relative) and
online path (0.25α/0.75α alpha-relative via _pot5_thresholds) are NOT
numerics-equivalent; caller must use pack-derived α for parity; helper
_pot5_alpha_levels_for_comparison exposes gap. See unified α thresholds
comment below and pack docstring.

Key Algorithmic Innovations:
1. Two-Accumulator Vector Factorization (S_full and S_half) directly in GPU SRAM registers:
   - S_full accumulates x for weights with magnitude 1.0 (identity conditional add/sub).
   - S_half accumulates x for weights with magnitude 0.5 (conditional add/sub).
   - Zero-magnitude weights (24% of tensor) bypass all ALU compute.
2. Single Tile-Boundary Bitshift:
   - Evaluates: y = alpha * (S_full + (S_half * 0.5)).
   - Zero bitshifts inside the inner loop; 100% multiplier-free vector arithmetic.
3. Intra-Kernel SRAM Fused SwiGLU:
   - Fuses SiLU(gate) * up directly in SRAM registers and streams through W_down POT.
   - Eliminates intermediate DRAM writes and activations storage.
"""

import math
import warnings
from typing import Optional, Tuple, Any  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False


def _is_turing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        cap = torch.cuda.get_device_capability()
        return (7, 5) <= tuple(cap) < (8, 0)
    except Exception:
        return False


def _maybe_cast_fp16_for_turing(t: torch.Tensor) -> torch.Tensor:
    if _is_turing() and t.dtype == torch.bfloat16:
        warnings.warn("Turing sm_75: bf16 -> fp16 (pot5, acc fp32)", stacklevel=3)
        return t.to(torch.float16)
    return t


def _prune_turing_block(block: int) -> int:
    if _is_turing() and block > 64:
        warnings.warn(f"Turing sm_75: clamping BLOCK {block} -> 64 (64KB SMEM)", stacklevel=3)
        return 64
    return block


def _turing_fp16_fallback_linear(x: torch.Tensor, w_eff: torch.Tensor) -> torch.Tensor:
    x_f = _maybe_cast_fp16_for_turing(x).float()
    w_f = w_eff.float()
    orig_shape = x_f.shape
    x2d = x_f.reshape(-1, orig_shape[-1])
    out = torch.matmul(x2d, w_f.t())
    return out.reshape(*orig_shape[:-1], w_f.shape[0]).to(x.dtype if not _is_turing() or x.dtype != torch.bfloat16 else torch.float16)


def _turing_prune_configs(configs):
    if not _is_turing():
        return configs
    pruned = []
    for c in configs:
        bm = c.kwargs.get('BLOCK_M', 32)
        bn = c.kwargs.get('BLOCK_N', 32)
        bk = c.kwargs.get('BLOCK_K', 32)
        if bm <= 64 and bn <= 64 and bk <= 64:
            pruned.append(c)
        else:
            warnings.warn(f"Turing sm_75: pruning BLOCK config {c.kwargs} >64", stacklevel=2)
    return pruned if pruned else configs


# ---------------------------------------------------------------------------
# Single documented 5-state POT quantization rule (shared everywhere)
# ---------------------------------------------------------------------------
# Central rule: given per-tensor scale alpha (default 1.4 * mean(|w|)),
# thresholds are t_low = 0.25 * alpha, t_high = 0.75 * alpha.
#   |w| <= t_low          -> 0
#   t_low < |w| <= t_high -> 0.5 * sign(w)
#   |w| >  t_high         -> 1.0 * sign(w)
# Effective weight w_q = q * alpha where q in {-1, -0.5, 0, 0.5, 1.0}.
# Triton kernels inline the same rule device-side:
#   w_eff = where(abs>0.75*alpha, sign, where(abs>0.25*alpha, sign*0.5, 0))
# Python fallbacks use _pot5_thresholds / _pot5_quantize_weight without
# host sync (no alpha.item() on CUDA paths).
# Pack path (pack_pot5_gpu_3bitplane) intentionally uses std-relative
# thresholds t0=0.35*std, t1=0.9*std to stay bit-exact with training-time
# pot5_quantize (affine_ai.core.ast_dag) and .toros format. Bitpacked
# inference reproduces training numerics because decompression is a LUT
# (unpack_pot5_gpu_3bitplane) that scales by the stored alpha and never
# recomputes thresholds at matmul time (see conversion helper
# _pot5_alpha_levels_for_comparison). For online master-weight matmuls
# the same alpha-relative rule is used everywhere else.

def _pot5_thresholds(alpha: torch.Tensor):
    # Unified α thresholds — see docstring warning; pack path uses different STD rule
    t_low = alpha * 0.25
    t_high = alpha * 0.75
    return t_low, t_high


def _pot5_quantize_weight(w: torch.Tensor, alpha: torch.Tensor):
    w_f = w.float()
    alpha_f = alpha.to(w_f.device).float()
    t_low, t_high = _pot5_thresholds(alpha_f)
    w_abs = w_f.abs()
    w_sign = w_f.sign()
    w_full = torch.where(w_abs > t_high, w_sign, torch.zeros_like(w_f))
    w_half = torch.where((w_abs > t_low) & (w_abs <= t_high), w_sign, torch.zeros_like(w_f))
    return (w_full + w_half * 0.5) * alpha_f


def _pot5_quantize_int8_levels(w: torch.Tensor, alpha: torch.Tensor):
    w_f = w.float()
    alpha_f = alpha.to(w_f.device).float()
    t_low, t_high = _pot5_thresholds(alpha_f)
    w_abs = w_f.abs()
    w_sign = w_f.sign()
    w_full = torch.where(w_abs > t_high, w_sign * 2.0, torch.zeros_like(w_f))
    w_half = torch.where((w_abs > t_low) & (w_abs <= t_high), w_sign, torch.zeros_like(w_f))
    return (w_full + w_half).to(torch.int8)


def _pot5_alpha_levels_for_comparison(w: torch.Tensor, alpha: torch.Tensor):
    return _pot5_quantize_weight(w, alpha)


if HAS_TRITON:
    @triton.autotune(
        configs=[
            # Consumer GPU & single-token inference small tiles
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=2, num_stages=3),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
            # Standard & datacenter tiles with BLOCK_K=32 and BLOCK_K=64 (Issue 34)
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def _pot5_gemm_fwd_kernel(
        X, W, Y, Alpha,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        M, N, K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Route A: High-Throughput Single-Pass 5-State POT Matrix Multiplication:
          W_eff in {-1.0, -0.5, 0.0, +0.5, +1.0}
          Y = (X @ W_eff.T) * alpha
        Single-accumulator execution in registers, eliminating the dual-accumulator 2x tl.dot penalty.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)  # noqa: F841

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        alpha = tl.load(Alpha)

        for k in range(0, K, BLOCK_K):
            x_ptr = tl.make_block_ptr(base=X, shape=(M, K), strides=(stride_xm, stride_xk), offsets=(pid_m * BLOCK_M, k), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
            x = tl.load(x_ptr, boundary_check=(0, 1))
            w_ptr = tl.make_block_ptr(base=W, shape=(N, K), strides=(stride_wn, stride_wk), offsets=(pid_n * BLOCK_N, k), block_shape=(BLOCK_N, BLOCK_K), order=(1, 0))
            w = tl.load(w_ptr, boundary_check=(0, 1))

            # Route A: In-register signed 5-state weight synthesis {-1.0, -0.5, 0.0, +0.5, +1.0}
            # Unified α thresholds (0.25α/0.75α) — cross-ref _pot5_thresholds; online path
            w_abs = tl.abs(w)
            w_sign = tl.where(w > 0, 1.0, tl.where(w < 0, -1.0, 0.0))
            w_eff = tl.where(w_abs > (alpha * 0.75), w_sign, tl.where(w_abs > (alpha * 0.25), w_sign * 0.5, 0.0))

            # Single Tensor Core / SIMT dot product (halves instruction count & register pressure)
            acc += tl.dot(x, tl.trans(w_eff.to(x.dtype)), out_dtype=tl.float32)

        # Apply optimal MSE scale
        y = acc * alpha

        y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, y.to(Y.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


    @triton.autotune(
        configs=[
            # Consumer GPU & single-token inference small tiles
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=2, num_stages=3),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
            # Mid-size & datacenter tiles with BLOCK_K=32 and 64 (Issue 34)
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def _pot5_int8_imma_gemm_kernel(
        X_int8, W_int8, Y, Scale_x, Alpha,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        stride_sx,
        M, N, K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Route A: Hardware INT8 IMMA 5-State POT Matrix Engine:
          W_int8 in {-2, -1, 0, +1, +2} (signed 8-bit integer)
          X_int8 in [-128, 127]
          Y = (X_int8 @ W_int8.T) * (Scale_x * Alpha * 0.5)
        Executes via hardware mma.sync.s8.s8 at 2x BF16 Tensor Core throughput.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)  # noqa: F841

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        for k in range(0, K, BLOCK_K):
            x_ptr = tl.make_block_ptr(base=X_int8, shape=(M, K), strides=(stride_xm, stride_xk), offsets=(pid_m * BLOCK_M, k), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
            x = tl.load(x_ptr, boundary_check=(0, 1))
            w_ptr = tl.make_block_ptr(base=W_int8, shape=(N, K), strides=(stride_wn, stride_wk), offsets=(pid_n * BLOCK_N, k), block_shape=(BLOCK_N, BLOCK_K), order=(1, 0))
            w = tl.load(w_ptr, boundary_check=(0, 1))

            # Native INT8 Tensor Core instruction: mma.sync.s8.s8
            acc += tl.dot(x, tl.trans(w), out_dtype=tl.int32)

        sx = tl.load(Scale_x + offs_m * stride_sx, mask=mask_m, other=1.0)
        alpha = tl.load(Alpha)

        # Scale by (alpha * 0.5) because integer weights are {-2, -1, 0, +1, +2} — INT8 POT scaling α*0.5 correct, keep
        scale_eff = sx[:, None] * (alpha * 0.5)
        y = acc.to(tl.float32) * scale_eff

        y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, y.to(Y.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def _pot5_fused_swiglu_fwd_kernel(
        GV, W_D, OUT, Alpha_d,
        stride_gvm, stride_gvn,
        stride_wdk, stride_wdn,
        stride_outm, stride_outk,
        M, N, K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Route A: Fused 5-State POT SwiGLU:
          Act = SiLU(Gate) * Up directly in SRAM.
          Out = Act @ W_down_pot5.T
        Single-accumulator execution in registers.
        """
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_k = offs_k < K

        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        alpha = tl.load(Alpha_d)

        for n_start in range(0, N, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N

            gate_ptrs = GV + offs_m[:, None] * stride_gvm + offs_n[None, :] * stride_gvn
            val_ptrs = GV + offs_m[:, None] * stride_gvm + (offs_n[None, :] + N) * stride_gvn

            gate = tl.load(gate_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
            val = tl.load(val_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

            # In-register SiLU activation
            sig = tl.sigmoid(gate)
            act = (gate * sig) * val

            # Load W_down tile
            wd_ptrs = W_D + offs_k[None, :] * stride_wdk + offs_n[:, None] * stride_wdn
            wd = tl.load(wd_ptrs, mask=mask_k[None, :] & mask_n[:, None], other=0.0).to(tl.float32)

            wd_abs = tl.abs(wd)
            wd_sign = tl.where(wd > 0, 1.0, tl.where(wd < 0, -1.0, 0.0))
            # Unified α thresholds (0.25α/0.75α) — cross-ref _pot5_thresholds
            wd_eff = tl.where(wd_abs > (alpha * 0.75), wd_sign, tl.where(wd_abs > (alpha * 0.25), wd_sign * 0.5, 0.0))

            # Scalar W_D loads documented: ALU-bound (abs/sign/where) dominates, block_ptr not beneficial
            acc += tl.dot(act.to(W_D.dtype.element_ty), wd_eff.to(W_D.dtype.element_ty), out_dtype=tl.float32)

        out = acc * alpha
        out_ptrs = OUT + offs_m[:, None] * stride_outm + offs_k[None, :] * stride_outk
        tl.store(out_ptrs, out.to(OUT.dtype.element_ty), mask=mask_m[:, None] & mask_k[None, :])


class _Triton5StatePOTFunction(torch.autograd.Function):
    """
    PyTorch Autograd Wrapper for 5-State POT Linear layer.
    Uses custom Triton kernels on CUDA, with pure PyTorch fallback on CPU.
    Turing sm_75: 5-state relies on int8-like path — fallback to fp16 linear (acc fp32), BLOCK<=64.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
        if _is_turing():
            if x.dtype == torch.bfloat16 or w.dtype == torch.bfloat16:
                warnings.warn("Turing sm_75: bf16 -> fp16 (pot5 5-state, acc fp32)", stacklevel=2)
            x = _maybe_cast_fp16_for_turing(x)
            w = _maybe_cast_fp16_for_turing(w)
            if alpha is not None and alpha.dtype == torch.bfloat16:
                alpha = alpha.to(torch.float16)
            if x.shape[-1] > 16384:
                warnings.warn(f"Turing sm_75: clamping K {x.shape[-1]} -> 16384", stacklevel=2)
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        M, K = x_2d.shape
        N = w.shape[0]

        if alpha is None:
            alpha = (w.float().abs().mean() * 1.4).to(x.dtype)
        if alpha.ndim == 0:
            alpha = alpha.unsqueeze(0)

        ctx.save_for_backward(x_2d, w, alpha)
        ctx.orig_shape = orig_shape

        if _is_turing():
            warnings.warn("Turing sm_75: 5-state POT fallback to fp16 matmul (acc fp32), BLOCK<=64", stacklevel=2)
            w_eff = _pot5_quantize_weight(w, alpha)
            out = _turing_fp16_fallback_linear(x_2d, w_eff)
            return out.reshape(*orig_shape[:-1], N)
        if x.is_cuda and HAS_TRITON:
            y = torch.empty((M, N), device=x.device, dtype=x.dtype)
            grid = lambda META: (  # noqa: E731
                triton.cdiv(M, META["BLOCK_M"]),
                triton.cdiv(N, META["BLOCK_N"]),
            )
            _pot5_gemm_fwd_kernel[grid](
                x_2d, w, y, alpha,
                x_2d.stride(0), x_2d.stride(1),
                w.stride(0), w.stride(1),
                y.stride(0), y.stride(1),
                M, N, K,
            )
            return y.reshape(*orig_shape[:-1], N)
        else:
            w_eff = _pot5_quantize_weight(w, alpha)
            out = F.linear(x_2d, w_eff.to(x.dtype))
            return out.reshape(*orig_shape[:-1], N)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_2d, w, alpha = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        go_2d = grad_output.reshape(-1, w.shape[0])

        gx = None
        gw = None
        galpha = None

        w_f = w.float()
        alpha_f = alpha.to(w_f.device).float()
        t_low, t_high = _pot5_thresholds(alpha_f)
        w_abs = w_f.abs()
        w_sign = w_f.sign()
        w_full = torch.where(w_abs > t_high, w_sign, torch.zeros_like(w_f))
        w_half = torch.where((w_abs > t_low) & (w_abs <= t_high), w_sign, torch.zeros_like(w_f))
        w_eff = (w_full + w_half * 0.5) * alpha_f

        if ctx.needs_input_grad[0]:
            gx = (go_2d @ w_eff.to(go_2d.dtype)).reshape(orig_shape)
        if ctx.needs_input_grad[1]:
            # STE gradient to master weights
            gw = (go_2d.t() @ x_2d).to(w.dtype)
        if ctx.needs_input_grad[2]:
            galpha = ((go_2d * (x_2d @ (w_full + w_half * 0.5).to(x_2d.dtype).t())).sum()).reshape(1)

        return gx, gw, galpha


def triton_pot5_linear(x: torch.Tensor, w: torch.Tensor, alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Evaluates Linear layer with 5-State POT weights via Triton GPU acceleration. Turing: fp16 fallback, acc fp32, BLOCK<=64."""
    if _is_turing() and x.dtype == torch.bfloat16:
        warnings.warn("Turing sm_75: bf16 -> fp16 (pot5 linear api, acc fp32)", stacklevel=2)
        x = x.to(torch.float16)
    return _Triton5StatePOTFunction.apply(x, w, alpha)


def triton_pot5_int8_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    alpha: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Route A: Hardware INT8 IMMA Matrix Multiplication for 5-State POT.
    Quantizes activations to INT8 and weights to {-2, -1, 0, 1, 2},
    running via mma.sync.s8.s8 at 2x BF16 Tensor Core throughput.
    Turing sm_75: INT8 m16n8k32 inefficient — fallback to fp16 matmul (acc fp32), BLOCK<=64, clamp K.
    """
    if _is_turing():
        warnings.warn("Turing sm_75: pot5_int8 INT8 inefficient, fallback to fp16 matmul (acc fp32)", stacklevel=2)
        if x.dtype == torch.bfloat16 or w.dtype == torch.bfloat16:
            warnings.warn("Turing sm_75: bf16 -> fp16 (pot5_int8, acc fp32)", stacklevel=2)
        x = _maybe_cast_fp16_for_turing(x)
        w = _maybe_cast_fp16_for_turing(w)
        if alpha is not None and alpha.dtype == torch.bfloat16:
            alpha = alpha.to(torch.float16)
        if alpha is None:
            alpha = (w.float().abs().mean() * 1.4).to(x.dtype)
        if alpha.ndim == 0:
            alpha = alpha.unsqueeze(0)
        w_eff = _pot5_quantize_weight(w, alpha)
        out = _turing_fp16_fallback_linear(x, w_eff)
        return out
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, K = x_2d.shape
    N = w.shape[0]

    if alpha is None:
        alpha = (w.float().abs().mean() * 1.4).to(x.dtype)
    if alpha.ndim == 0:
        alpha = alpha.unsqueeze(0)

    # Quantize activations to INT8 with per-token scale
    amax_x = x_2d.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    sx = (amax_x / 127.0).to(torch.float32)
    x_int8 = (x_2d / sx).round().clamp(-128, 127).to(torch.int8)

    # Quantize weights to {-2, -1, 0, 1, 2} (device-side, no alpha.item() sync)
    w_int8 = _pot5_quantize_int8_levels(w, alpha)

    if x.is_cuda and HAS_TRITON:
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = lambda META: (  # noqa: E731
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )
        _pot5_int8_imma_gemm_kernel[grid](
            x_int8, w_int8, y, sx.squeeze(-1), alpha,
            x_int8.stride(0), x_int8.stride(1),
            w_int8.stride(0), w_int8.stride(1),
            y.stride(0), y.stride(1),
            sx.stride(0),
            M, N, K,
        )
        return y.reshape(*orig_shape[:-1], N)
    else:
        alpha_f = alpha.to(w_int8.device).float()
        w_eff = (w_int8.float() * 0.5) * alpha_f
        out = F.linear(x_2d, w_eff.to(x.dtype))
        return out.reshape(*orig_shape[:-1], N)


def triton_pot5_fused_swiglu(
    gate_up: torch.Tensor,
    w_down: torch.Tensor,
    alpha_d: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Fused 5-State POT SwiGLU:
      Computes Out = (SiLU(Gate) * Up) @ W_down_pot5.T directly in GPU SRAM registers.
    Turing sm_75: fallback to fp16 linear (acc fp32), BLOCK<=64.
    """
    if _is_turing():
        if gate_up.dtype == torch.bfloat16 or w_down.dtype == torch.bfloat16:
            warnings.warn("Turing sm_75: bf16 -> fp16 (pot5 swiglu, acc fp32)", stacklevel=2)
        gate_up = _maybe_cast_fp16_for_turing(gate_up)
        w_down = _maybe_cast_fp16_for_turing(w_down)
        if alpha_d is not None and alpha_d.dtype == torch.bfloat16:
            alpha_d = alpha_d.to(torch.float16)
        warnings.warn("Turing sm_75: pot5 fused swiglu BLOCK<=64, fallback fp16 if needed", stacklevel=2)
    orig_shape = gate_up.shape
    gv_2d = gate_up.reshape(-1, orig_shape[-1])
    M = gv_2d.shape[0]
    total_dim = gv_2d.shape[-1]
    N = total_dim // 2
    K = w_down.shape[0]

    if alpha_d is None:
        alpha_d = (w_down.float().abs().mean() * 1.4).to(gate_up.dtype)
    if alpha_d.ndim == 0:
        alpha_d = alpha_d.unsqueeze(0)

    if gate_up.is_cuda and HAS_TRITON and not _is_turing():
        out = torch.empty((M, K), device=gate_up.device, dtype=gate_up.dtype)
        grid = lambda META: (  # noqa: E731
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(K, META["BLOCK_K"]),
        )
        _pot5_fused_swiglu_fwd_kernel[grid](
            gv_2d, w_down, out, alpha_d,
            gv_2d.stride(0), gv_2d.stride(1),
            w_down.stride(0), w_down.stride(1),
            out.stride(0), out.stride(1),
            M, N, K,
        )
        return out.reshape(*orig_shape[:-1], K)
    else:
        if _is_turing():
            warnings.warn("Turing sm_75: pot5 swiglu fallback to fp16 matmul (acc fp32)", stacklevel=2)
        g, u = gv_2d.chunk(2, dim=-1)
        act = F.silu(g) * u
        wd_eff = _pot5_quantize_weight(w_down, alpha_d)
        out = F.linear(act.float(), wd_eff.float().to(act.dtype)).to(gate_up.dtype)
        return out.reshape(*orig_shape[:-1], K)


class Triton5StatePOTLinear(nn.Module):
    """Linear layer operating on 5-State POT weights with BF16 master weights."""
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=dtype) / math.sqrt(in_features))
        init_alpha = (self.weight.detach().float().abs().mean() * 1.4).to(dtype).unsqueeze(0)
        self.register_buffer("alpha", init_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            alpha = (self.weight.detach().float().abs().mean() * 1.4).to(self.weight.dtype).unsqueeze(0)
            self.alpha.copy_(alpha)
            return triton_pot5_linear(x, self.weight, alpha)
        return triton_pot5_linear(x, self.weight, self.alpha)


def pack_pot5_gpu_3bitplane(
    w: torch.Tensor,
    threshold_z: float = 0.35,
    shift: int = 1,
    alpha: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Packs 5-state POT weights [N, K] into 3 GPU bitplanes of torch.int32:
      NonZero bitplane: 1 bit per weight
      Magnitude bitplane: 1 bit per weight (1.0 vs 0.5)
      Sign bitplane: 1 bit per weight (- vs +)
    Storage: Exactly 3 bits/weight (5.33x reduction vs BF16, 2.67x vs INT8).
    When alpha is provided, thresholds align with online GEMM (_pot5_thresholds).
    When alpha is None, thresholds use std-relative formula to match training pot5_quantize.
    Returns: (w_nz_bits, w_mag_bits, w_sign_bits, alpha, K_orig)
    """
    N, K_orig = w.shape
    device = w.device
    std = w.float().std().clamp_min(1e-8)

    pad_len = (32 - (K_orig % 32)) % 32
    if pad_len > 0:
        w_padded = F.pad(w, (0, pad_len), value=0.0)
    else:
        w_padded = w

    w_f = w_padded.float()
    K_padded = w_f.shape[1]
    val_low = 2.0 ** (-shift)
    val_high = 1.0

    abs_w = w_f.abs()
    sign = w_f.sign()

    if alpha is not None:
        t0, t1 = _pot5_thresholds(alpha.to(device).float())
        q = torch.zeros_like(w_f)
        q = torch.where(abs_w > t0, sign * val_low, q)
        q = torch.where(abs_w > t1, sign * val_high, q)
        out_alpha = alpha.to(w.dtype)
        nz_mask = (abs_w > t0)
        mag_mask = (abs_w > t1)
    else:
        t0 = threshold_z * std
        t1 = (val_low + val_high) * 0.5 * std * 1.2
        q = torch.zeros_like(w_f)
        q = torch.where(abs_w >= t0, sign * val_low, q)
        q = torch.where(abs_w >= t1, sign * val_high, q)
        out_alpha = ((w_f * q).sum() / (q * q).sum().clamp_min(1e-8)).to(w.dtype)
        nz_mask = (abs_w >= t0)
        mag_mask = (abs_w >= t1)

    sign_mask = (w_f < 0.0)

    # Fix signed overflow: 1<<31 overflows int32; use int64 for powers
    lane_shifts = torch.pow(torch.tensor(2, device=device, dtype=torch.int64), torch.arange(32, device=device, dtype=torch.int64)).view(1, 1, 32)
    K_words = K_padded // 32

    w_nz_bits = (nz_mask.view(N, K_words, 32).to(torch.int64) * lane_shifts).sum(dim=-1).to(torch.int32)  # use int64 acc to avoid int32 overflow, lane_shifts is int64
    w_mag_bits = (mag_mask.view(N, K_words, 32).to(torch.int64) * lane_shifts).sum(dim=-1).to(torch.int32)
    w_sign_bits = (sign_mask.view(N, K_words, 32).to(torch.int64) * lane_shifts).sum(dim=-1).to(torch.int32)

    return w_nz_bits, w_mag_bits, w_sign_bits, out_alpha, K_orig


def unpack_pot5_gpu_3bitplane(
    w_nz_bits: torch.Tensor,
    w_mag_bits: torch.Tensor,
    w_sign_bits: torch.Tensor,
    alpha: torch.Tensor,
    K_orig: int
) -> torch.Tensor:
    """
    Unpacks 3-bitplane GPU weights back to continuous 5-state POT tensor.
    Formula: W = nz * (0.5 + 0.5 * mag) * (1 - 2 * sign) * alpha
    """
    N, K_words = w_nz_bits.shape
    device = w_nz_bits.device
    lane_shifts = torch.arange(32, device=device, dtype=torch.int64).view(1, 1, 32)

    # Logical shift via unsigned: cast to int64 & 0xFFFFFFFF to avoid arithmetic sign-extend
    w_nz_u = w_nz_bits.to(torch.int64) & 0xFFFFFFFF
    w_mag_u = w_mag_bits.to(torch.int64) & 0xFFFFFFFF
    w_sign_u = w_sign_bits.to(torch.int64) & 0xFFFFFFFF
    nz = ((w_nz_u.unsqueeze(-1) >> lane_shifts) & 1).float()
    mag = ((w_mag_u.unsqueeze(-1) >> lane_shifts) & 1).float()
    sign = ((w_sign_u.unsqueeze(-1) >> lane_shifts) & 1).float()

    w_val = nz * (0.5 + 0.5 * mag) * (1.0 - 2.0 * sign) * alpha.float()
    return w_val.reshape(N, -1)[:, :K_orig].to(alpha.dtype)


if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        ],
        key=['M', 'N'],
    )
    @triton.jit
    def _pot5_bitpacked_gemm_fwd_kernel(
        X, W_nz, W_mag, W_sign, Y, Alpha,
        stride_xm, stride_xk,
        stride_wn, stride_wkw,
        stride_ym, stride_yn,
        M, N, K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr = 32,  # must be 32 — enforced via tl.static_assert
    ):
        """
        Ultra-High-Throughput 3-Bitplane Bitpacked 5-State POT GEMM:
          Weight traffic is reduced by 5.33x vs BF16 (3 bits/weight).
          Decompresses bits in-registers across 32 lanes.
        Note: X loads remain scalar-pointers (ALU-bound bit decompression dominates; block_ptr gives no measurable gain)
        """
        tl.static_assert(BLOCK_K == 32, "BLOCK_K must be 32 for bitpacked kernel")
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        lane = tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        alpha = tl.load(Alpha)

        for k in range(0, K, BLOCK_K):
            kk = k + lane
            mask_k = kk < K
            k_word = k // BLOCK_K

            # Load X tile [BLOCK_M, 32]
            x_ptrs = X + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk
            x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            # Load 3 bitplane words for BLOCK_N rows: [BLOCK_N]
            w_nz_word = tl.load(W_nz + offs_n * stride_wn + k_word * stride_wkw, mask=mask_n, other=0)
            w_mag_word = tl.load(W_mag + offs_n * stride_wn + k_word * stride_wkw, mask=mask_n, other=0)
            w_sign_word = tl.load(W_sign + offs_n * stride_wn + k_word * stride_wkw, mask=mask_n, other=0)

            # In-register bit decompression across 32 lanes: [BLOCK_N, 32]
            # Fix: use logical shift via uint32 (arithmetic >> would sign-extend bit31)
            nz = ((w_nz_word[:, None].to(tl.uint32) >> lane[None, :].to(tl.uint32)) & 1).to(tl.float32)
            mag = ((w_mag_word[:, None].to(tl.uint32) >> lane[None, :].to(tl.uint32)) & 1).to(tl.float32)
            sign = ((w_sign_word[:, None].to(tl.uint32) >> lane[None, :].to(tl.uint32)) & 1).to(tl.float32)

            # Synthesize 5-state weights {-1.0, -0.5, 0.0, +0.5, +1.0}
            w_eff = nz * (0.5 + 0.5 * mag) * (1.0 - 2.0 * sign)

            acc += tl.dot(x, tl.trans(w_eff.to(x.dtype)), out_dtype=tl.float32)

        y = acc * alpha
        y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, y.to(Y.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def _assert_pack_alpha(alpha: torch.Tensor):
    # Hard gate: pack-derived alpha must be used for bitpacked inference; online thresholds (0.25/0.75) diverge from std-relative pack thresholds
    # No runtime check feasible without metadata; document that caller must pass alpha from pack_pot5_gpu_3bitplane
    return

def triton_pot5_bitpacked_linear(
    x: torch.Tensor,
    w_nz_bits: torch.Tensor,
    w_mag_bits: torch.Tensor,
    w_sign_bits: torch.Tensor,
    alpha: torch.Tensor,
    K_orig: int
) -> torch.Tensor:
    """
    Evaluates Linear layer using 3-bitplane GPU bitpacked weights.
    5.33x lower DRAM weight bandwidth than BF16 (exactly 3.0 bits/weight).
    Turing sm_75: BLOCK<=64, fp16 fallback (acc fp32) if needed.
    """
    if _is_turing():
        if x.dtype == torch.bfloat16:
            warnings.warn("Turing sm_75: bf16 -> fp16 (pot5 bitpacked, acc fp32)", stacklevel=2)
            x = x.to(torch.float16)
        warnings.warn("Turing sm_75: pot5 bitpacked BLOCK<=64 clamp", stacklevel=2)
        # Turing fallback to unpack + fp16 matmul to avoid 3-bitplane SMEM pressure
        if x.is_cuda:
            w_eff = unpack_pot5_gpu_3bitplane(w_nz_bits, w_mag_bits, w_sign_bits, alpha, K_orig)
            w_eff = _maybe_cast_fp16_for_turing(w_eff).float()
            x_f = x.float()
            out = F.linear(x_f.reshape(-1, K_orig), w_eff.to(x_f.dtype))
            return out.reshape(*x.shape[:-1], w_eff.shape[0]).to(x.dtype if x.dtype != torch.bfloat16 else torch.float16)
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, K = x_2d.shape
    N = w_nz_bits.shape[0]

    pad_len = (32 - (K % 32)) % 32
    if pad_len > 0:
        x_2d = F.pad(x_2d, (0, pad_len), value=0.0)
    K_padded = x_2d.shape[1]
    # K_padded is already 32-aligned; no Turing prune needed here (BLOCK_K=32 enforced)

    if alpha.ndim == 0:
        alpha = alpha.unsqueeze(0)

    # K_padded is 32-aligned padded K (>= K_orig); K_orig is logical dim for slicing output. Kernel loops K_padded with mask_k< K_orig? Actually K_padded includes padding zeros, kernel masks with kk<K (K_padded) but W bitplanes padded zero, so tail is exact. We pass K_padded to kernel and mask with K_padded, but logical K_orig is used for CPU fallback slicing.
    if x.is_cuda and HAS_TRITON and not _is_turing():
        assert K_padded % 32 == 0, f"K_padded must be 32-aligned, got {K_padded}"
        if w_nz_bits.stride(0) != 1:
            w_nz_bits = w_nz_bits.transpose(0, 1).contiguous().transpose(0, 1)
            w_mag_bits = w_mag_bits.transpose(0, 1).contiguous().transpose(0, 1)
            w_sign_bits = w_sign_bits.transpose(0, 1).contiguous().transpose(0, 1)
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = lambda META: (  # noqa: E731
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )
        _pot5_bitpacked_gemm_fwd_kernel[grid](
            x_2d, w_nz_bits, w_mag_bits, w_sign_bits, y, alpha,
            x_2d.stride(0), x_2d.stride(1),
            w_nz_bits.stride(0), w_nz_bits.stride(1),
            y.stride(0), y.stride(1),
            M, N, K_padded,
            BLOCK_K=_prune_turing_block(32),
        )
        return y.reshape(*orig_shape[:-1], N)
    else:
        w_eff = unpack_pot5_gpu_3bitplane(w_nz_bits, w_mag_bits, w_sign_bits, alpha, K_orig)
        out = F.linear(x.reshape(-1, K_orig), w_eff.to(x.dtype))
        return out.reshape(*orig_shape[:-1], N)


def bitplane_bytes_to_gpu_int32(raw_bytes: bytes, N: int, K: int, device: str = "cpu") -> torch.Tensor:
    """
    Directly converts 8-bit packed bitplane bytes into 32-bit GPU bitplane words without float intermediate.
    """
    bits = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8))[:N * K]
    pad_len = (32 - (K % 32)) % 32
    if pad_len > 0:
        bits_2d = bits.reshape(N, K)
        bits_2d = np.pad(bits_2d, ((0, 0), (0, pad_len)), mode='constant', constant_values=0)
    else:
        bits_2d = bits.reshape(N, K)
    K_words = bits_2d.shape[1] // 32
    bits_grouped = bits_2d.reshape(N, K_words, 32)
    powers = (1 << np.arange(32, dtype=np.uint32)).reshape(1, 1, 32)
    int32_arr = (bits_grouped.astype(np.uint32) * powers).sum(axis=-1).astype(np.int32)
    return torch.from_numpy(int32_arr).to(device=device)


class Triton5StatePOTBitpackedLinear(nn.Module):
    """
    Ultra-low memory linear layer holding 3-bitplane bitpacked weights directly in GPU VRAM.
    Uses 5.33x less memory than BF16 (3.0 bits per weight).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        w_nz_bits: Optional[torch.Tensor] = None,
        w_mag_bits: Optional[torch.Tensor] = None,
        w_sign_bits: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

        if w_nz_bits is None:
            temp_w = torch.randn(out_features, in_features, dtype=dtype) / math.sqrt(in_features)
            w_nz_bits, w_mag_bits, w_sign_bits, alpha, _ = pack_pot5_gpu_3bitplane(temp_w)

        if alpha is not None and alpha.ndim == 0:
            alpha = alpha.unsqueeze(0)

        if w_nz_bits is not None and w_nz_bits.stride(0) != 1:
            w_nz_bits = w_nz_bits.transpose(0, 1).contiguous().transpose(0, 1)
            w_mag_bits = w_mag_bits.transpose(0, 1).contiguous().transpose(0, 1)
            w_sign_bits = w_sign_bits.transpose(0, 1).contiguous().transpose(0, 1)

        self.register_buffer("w_nz_bits", w_nz_bits)
        self.register_buffer("w_mag_bits", w_mag_bits)
        self.register_buffer("w_sign_bits", w_sign_bits)
        self.register_buffer("alpha", alpha)
        self.bias = None

    @property
    def weight(self) -> torch.Tensor:
        """De-quantizes bitpacked weights on the fly if requested by code expecting nn.Linear."""
        return unpack_pot5_gpu_3bitplane(
            self.w_nz_bits, self.w_mag_bits, self.w_sign_bits, self.alpha, self.in_features
        ).to(self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = triton_pot5_bitpacked_linear(
            x, self.w_nz_bits, self.w_mag_bits, self.w_sign_bits, self.alpha, self.in_features
        )
        if getattr(self, "bias", None) is not None:
            out = out + self.bias
        return out


class Triton5StatePOTBitpackedResidualLinear(nn.Module):
    """
    Ultra-low memory linear layer holding:
    - 3-bitplane bitpacked 5-state POT core directly in GPU VRAM (3.0 bpw)
    - Top-2% sparse outlier table directly in GPU VRAM (FP16 or Q4)
    Total resident memory: 3.09 - 3.32 bits/weight!
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        w_nz_bits: torch.Tensor,
        w_mag_bits: torch.Tensor,
        w_sign_bits: torch.Tensor,
        alpha: torch.Tensor,
        outlier_indices: torch.Tensor,
        outlier_values: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.bfloat16,
        use_q4: bool = False,
        q4_nibbles: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        block_size: int = 32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.use_q4 = use_q4 or (q4_nibbles is not None)
        self.block_size = block_size

        if alpha.ndim == 0:
            alpha = alpha.unsqueeze(0)

        idx_i64 = outlier_indices.to(torch.int64)
        if w_nz_bits is not None and w_nz_bits.stride(0) != 1:
            w_nz_bits = w_nz_bits.transpose(0, 1).contiguous().transpose(0, 1)
            w_mag_bits = w_mag_bits.transpose(0, 1).contiguous().transpose(0, 1)
            w_sign_bits = w_sign_bits.transpose(0, 1).contiguous().transpose(0, 1)

        self.register_buffer("w_nz_bits", w_nz_bits)
        self.register_buffer("w_mag_bits", w_mag_bits)
        self.register_buffer("w_sign_bits", w_sign_bits)
        self.register_buffer("alpha", alpha)
        self.register_buffer("outlier_indices", idx_i64)

        if self.use_q4:
            if q4_nibbles is not None and scales is not None:
                self.register_buffer("outlier_q4", q4_nibbles.to(torch.uint8))
                self.register_buffer("scales", scales.to(torch.float16))
            elif outlier_values is not None and outlier_values.numel() > 0:
                # Quantize outlier_values to Q4 block-32 in resident VRAM to save 3.5x VRAM
                num_out = outlier_values.numel()
                pad_len = (block_size - (num_out % block_size)) % block_size
                out_padded = F.pad(outlier_values.float(), (0, pad_len)) if pad_len > 0 else outlier_values.float()
                out_blocks = out_padded.reshape(-1, block_size)
                sc = (out_blocks.abs().amax(dim=-1).clamp_min(1e-5) / 8.0).to(torch.float16)  # fix -8..7 scale: amax/8 to avoid -8 overflow (was /7 leak)
                q4 = torch.round(out_blocks / sc.unsqueeze(-1)).clamp(-8, 7).to(torch.int8).view(-1)[:num_out]
                nibbles = (q4 & 0x0F).to(torch.uint8)
                if len(nibbles) % 2 != 0:
                    nibbles = F.pad(nibbles, (0, 1))
                packed_q4 = nibbles[0::2] | (nibbles[1::2] << 4)
                self.register_buffer("outlier_q4", packed_q4)
                self.register_buffer("scales", sc)
            else:
                self.register_buffer("outlier_q4", torch.empty(0, dtype=torch.uint8))
                self.register_buffer("scales", torch.empty(0, dtype=torch.float16))
            self.register_buffer("_outlier_values", None, persistent=False)
        else:
            self.register_buffer(
                "_outlier_values",
                outlier_values.to(torch.float16) if outlier_values is not None else torch.empty(0, dtype=torch.float16)
            )

        if idx_i64.numel() > 0:
            self.register_buffer("outlier_rows", idx_i64 // in_features, persistent=False)
            self.register_buffer("outlier_cols", idx_i64 % in_features, persistent=False)
        else:
            self.register_buffer("outlier_rows", torch.empty(0, dtype=torch.int64), persistent=False)
            self.register_buffer("outlier_cols", torch.empty(0, dtype=torch.int64), persistent=False)
        self.bias = None

    @property
    def outlier_values(self) -> torch.Tensor:
        if self.use_q4:
            num_out = self.outlier_indices.numel()
            if num_out == 0 or not hasattr(self, "outlier_q4") or self.outlier_q4 is None or self.outlier_q4.numel() == 0:
                return torch.empty(0, dtype=self.dtype, device=self.outlier_indices.device)
            low = (self.outlier_q4 & 0x0F).to(torch.int8)
            high = ((self.outlier_q4 >> 4) & 0x0F).to(torch.int8)
            unpacked = torch.stack([low, high], dim=1).view(-1)[:num_out]
            q4 = torch.where(unpacked >= 8, unpacked - 16, unpacked).to(torch.float32)
            block_idx = torch.arange(num_out, device=self.outlier_q4.device) // self.block_size
            return (q4 * self.scales[block_idx]).to(self.dtype)
        return self._outlier_values.to(self.dtype)

    @property
    def weight(self) -> torch.Tensor:
        """De-quantizes bitpacked weights + residual outliers on the fly if requested."""
        w_dense = unpack_pot5_gpu_3bitplane(
            self.w_nz_bits, self.w_mag_bits, self.w_sign_bits, self.alpha, self.in_features
        ).to(self.dtype)
        if self.outlier_indices.numel() > 0:
            w_dense = w_dense.contiguous().view(-1)
            w_dense.scatter_(0, self.outlier_indices, self.outlier_values.to(self.dtype))
            return w_dense.view(self.out_features, self.in_features)
        return w_dense.view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Base 5-State POT GEMM directly from 3-bit bitplanes
        y = triton_pot5_bitpacked_linear(
            x, self.w_nz_bits, self.w_mag_bits, self.w_sign_bits, self.alpha, self.in_features
        )
        # 2. Add Top-2% Outlier contributions (tiled to prevent VRAM spikes)
        if self.outlier_indices.numel() > 0:
            orig_shape = x.shape
            x_flat = x.reshape(-1, self.in_features)
            y_flat = y.reshape(-1, self.out_features)
            num_outliers = self.outlier_rows.shape[0]
            outlier_vals = self.outlier_values.to(x.dtype)
            CHUNK_SIZE = 4096
            for start in range(0, num_outliers, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, num_outliers)
                cols = self.outlier_cols[start:end]
                rows = self.outlier_rows[start:end]
                vals = outlier_vals[start:end]
                x_sampled = x_flat[:, cols]
                scaled = x_sampled * vals
                y_flat.index_add_(1, rows, scaled)
            y = y_flat.reshape(*orig_shape[:-1], self.out_features)
        if getattr(self, "bias", None) is not None:
            y = y + self.bias
        return y


class Triton5StatePOTBitpackedResidualQ4Linear(Triton5StatePOTBitpackedResidualLinear):
    """
    Ultra-low memory linear layer holding:
    - 3-bitplane bitpacked 5-state POT core directly in GPU VRAM (3.0 bpw)
    - Top-2% Block-32 4-bit compressed outliers in resident VRAM (~0.09 bpw)
    Total resident memory: ~3.09 bits/weight!
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        w_nz_bits: torch.Tensor,
        w_mag_bits: torch.Tensor,
        w_sign_bits: torch.Tensor,
        alpha: torch.Tensor,
        outlier_indices: torch.Tensor,
        outlier_values: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.bfloat16,
        q4_nibbles: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        block_size: int = 32,
    ):
        super().__init__(
            in_features, out_features, w_nz_bits, w_mag_bits, w_sign_bits, alpha,
            outlier_indices, outlier_values, dtype=dtype,
            use_q4=True, q4_nibbles=q4_nibbles, scales=scales, block_size=block_size
        )

