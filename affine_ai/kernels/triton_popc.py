"""
Custom Triton Kernel: 1-Bit Hardware POPC (Population Count) Binary Router
==========================================================================
Executes single-cycle binary dot products using INT32 hardware POPC (popc.b32) ALUs.
Replaces 32 floating-point FMAs with 1 XOR and 1 POPC instruction per 32 dimensions.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


@triton.jit
def _pack_sign_bits_kernel(
    X_ptr, Out_bits_ptr,
    stride_xm, stride_xd,
    stride_om, stride_ok,
    M, D, K_words,
    BLOCK_M: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    offs_b = tl.arange(0, 32)
    cols = pid_k * 32 + offs_b
    mask = mask_m[:, None] & (cols[None, :] < D)

    val = tl.load(X_ptr + offs_m[:, None] * stride_xm + cols[None, :] * stride_xd, mask=mask, other=-1.0)
    is_pos = (val >= 0.0).to(tl.int32)
    bits = is_pos << offs_b[None, :]
    acc = tl.sum(bits, axis=1)

    tl.store(Out_bits_ptr + offs_m * stride_om + pid_k * stride_ok, acc, mask=mask_m)


@triton.jit
def _popc_dot_kernel(
    X_bits_ptr, W_bits_ptr, Out_ptr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    M, N,
    K_WORDS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    rem = D % 32

    for k in range(K_WORDS):
        x = tl.load(X_bits_ptr + offs_m[:, None] * stride_xm + k * stride_xk, mask=mask_m[:, None], other=0)
        w = tl.load(W_bits_ptr + offs_n[None, :] * stride_wn + k * stride_wk, mask=mask_n[None, :], other=0)
        diff = x ^ w
        if (k == K_WORDS - 1) and (rem != 0):
            mask = (1 << rem) - 1
            diff = diff & mask
            pop = tl.inline_asm_elementwise(
                "popc.b32 $0, $1;",
                "=r,r",
                [diff],
                dtype=tl.int32,
                is_pure=True,
                pack=1
            )
            sim = rem - 2 * pop
        else:
            pop = tl.inline_asm_elementwise(
                "popc.b32 $0, $1;",
                "=r,r",
                [diff],
                dtype=tl.int32,
                is_pure=True,
                pack=1
            )
            sim = 32 - 2 * pop
        acc += sim

    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_pack_sign_bits(x: torch.Tensor) -> torch.Tensor:
    """
    Packs sign of floating-point tensor x into 32-bit integer bitmasks.
    Returns: (M, ceil(D / 32)) int32 tensor where bit_i = (x_i >= 0).
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    x_flat = x.reshape(-1, D).contiguous()
    M = x_flat.shape[0]
    K_words = triton.cdiv(D, 32)

    out_bits = torch.empty((M, K_words), dtype=torch.int32, device=x.device)
    grid = (triton.cdiv(M, 64), K_words)

    _pack_sign_bits_kernel[grid](
        x_flat, out_bits,
        x_flat.stride(0), x_flat.stride(1),
        out_bits.stride(0), out_bits.stride(1),
        M, D, K_words,
        BLOCK_M=64
    )
    return out_bits.reshape(*orig_shape[:-1], K_words)


def triton_popc_sign_similarity(
    x_bits: torch.Tensor,
    w_bits: torch.Tensor,
    scale: Optional[float] = None,
    D: Optional[int] = None,
    dim: Optional[int] = None
) -> torch.Tensor:
    """
    Computes exact inner product between sign-quantized activations and hyperplanes:
        dot(x_sign, w_sign) = D - 2 * popc(x_bits ^ w_bits)
    via single-cycle INT32 POPC hardware instructions.

    Args:
        x_bits: Packed input signs of shape (M, K_words) or (B, T, K_words).
        w_bits: Packed hyperplane signs of shape (N, K_words).
        scale: Optional scaling factor to convert integer dot products to floats.
        D: Optional original unpadded feature dimension (if not divisible by 32).
        dim: Alias for D.
    """
    orig_shape = x_bits.shape
    K_words = x_bits.shape[-1]
    actual_D = D if D is not None else (dim if dim is not None else K_words * 32)
    x_flat = x_bits.reshape(-1, K_words).contiguous()
    w_flat = w_bits.contiguous()

    M = x_flat.shape[0]
    N = w_flat.shape[0]

    out = torch.empty((M, N), dtype=torch.int32, device=x_bits.device)
    BM, BN = 32, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    _popc_dot_kernel[grid](
        x_flat, w_flat, out,
        x_flat.stride(0), x_flat.stride(1),
        w_flat.stride(0), w_flat.stride(1),
        out.stride(0), out.stride(1),
        M, N,
        K_WORDS=K_words,
        D=actual_D,
        BLOCK_M=BM, BLOCK_N=BN
    )

    out_reshaped = out.reshape(*orig_shape[:-1], N)
    if scale is not None:
        return out_reshaped.float() * scale
    return out_reshaped
