
"""
Custom Triton Kernel: 1-Bit Binary Router (SWAR POPC)
======================================================
Executes binary dot products via SWAR emulated popcount (single-cycle PTX
popc.b32 would be faster, used for portability). Replaces 32 floating-point
FMAs with 1 XOR and 1 POPC per 32 dimensions.
"""

import torch
from typing import Tuple, Optional  # noqa: F401

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False

if HAS_TRITON:
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
        is_pos = (val >= 0.0).to(tl.uint32)
        bits = is_pos << offs_b[None, :]
        acc = tl.sum(bits, axis=1)
        tl.store(Out_bits_ptr + offs_m * stride_om + pid_k * stride_ok, acc.to(tl.int32), mask=mask_m)

    @triton.jit
    def _popcount32_swar(v):
        v = v - ((v >> 1) & 0x55555555)
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
        v = (v + (v >> 4)) & 0x0F0F0F0F
        v = v + (v >> 8)
        v = v + (v >> 16)
        v = v & 0x3F
        return v

    @triton.jit
    def _popc_dot_kernel_popc(
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
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.uint32)
        rem = D % 32
        for k in range(K_WORDS):
            x = tl.load(X_bits_ptr + offs_m[:, None] * stride_xm + k * stride_xk, mask=mask_m[:, None], other=0).to(tl.uint32)
            w = tl.load(W_bits_ptr + offs_n[None, :] * stride_wn + k * stride_wk, mask=mask_n[None, :], other=0).to(tl.uint32)
            diff = x ^ w
            tail = (k == K_WORDS - 1) and (rem != 0)
            if tail:
                mask = (tl.full((), 1, dtype=tl.uint32) << rem) - 1
                diff = diff & mask
            pop = tl.inline_asm_elementwise("popc.b32 $0, $1;", "=r,r", [diff], dtype=tl.int32, is_pure=True, pack=1).to(tl.uint32)
            sim = tl.where(tail, rem - 2 * pop, 32 - 2 * pop)
            acc += sim.to(tl.uint32)
        tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc.to(tl.int32), mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _popc_dot_kernel_swar(
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
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.uint32)
        rem = D % 32
        for k in range(K_WORDS):
            x = tl.load(X_bits_ptr + offs_m[:, None] * stride_xm + k * stride_xk, mask=mask_m[:, None], other=0).to(tl.uint32)
            w = tl.load(W_bits_ptr + offs_n[None, :] * stride_wn + k * stride_wk, mask=mask_n[None, :], other=0).to(tl.uint32)
            diff = x ^ w
            tail = (k == K_WORDS - 1) and (rem != 0)
            if tail:
                mask = (tl.full((), 1, dtype=tl.uint32) << rem) - 1
                diff = diff & mask
            v = diff
            v = v - ((v >> 1) & 0x55555555)
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
            v = (v + (v >> 4)) & 0x0F0F0F0F
            v = v + (v >> 8)
            v = v + (v >> 16)
            pop = v & 0x3F
            sim = tl.where(tail, rem - 2 * pop, 32 - 2 * pop)
            acc += sim.to(tl.uint32)
        tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc.to(tl.int32), mask=mask_m[:, None] & mask_n[None, :])

    def _get_popc_dot_configs():
        return [
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        ]
    _HAS_POPC_ASM = True
    try:
        _ = tl.inline_asm_elementwise
    except Exception:
        _HAS_POPC_ASM = False
    _popc_dot_kernel_popc = triton.autotune(configs=_get_popc_dot_configs(), key=['M', 'N', 'K_WORDS'])(_popc_dot_kernel_popc)
    _popc_dot_kernel_swar = triton.autotune(configs=_get_popc_dot_configs(), key=['M', 'N', 'K_WORDS'])(_popc_dot_kernel_swar)
    _popc_dot_kernel = _popc_dot_kernel_popc if _HAS_POPC_ASM else _popc_dot_kernel_swar
else:
    _pack_sign_bits_kernel = None
    _popc_dot_kernel = None
    _popc_dot_kernel_popc = None
    _popc_dot_kernel_swar = None
    def _get_popc_dot_configs():
        return []
    _HAS_POPC_ASM = False

def triton_pack_sign_bits(x: torch.Tensor) -> torch.Tensor:
    if not HAS_TRITON or not x.is_cuda:
        orig_shape = x.shape
        D = orig_shape[-1]
        x_flat = x.reshape(-1, D)
        K_words = (D + 31) // 32
        out = torch.zeros((x_flat.shape[0], K_words), dtype=torch.int32, device=x.device)
        for k in range(K_words):
            cols = torch.arange(k * 32, min((k + 1) * 32, D), device=x.device)
            vals = x_flat[:, cols]
            is_pos = (vals >= 0).to(torch.int32)
            bits = is_pos << torch.arange(len(cols), device=x.device, dtype=torch.int32)
            acc = bits.sum(dim=-1, dtype=torch.int32)
            out[:, k] = acc
        return out.reshape(*orig_shape[:-1], K_words)
    orig_shape = x.shape
    D = orig_shape[-1]
    x_flat = x.reshape(-1, D).contiguous()
    M = x_flat.shape[0]
    K_words = triton.cdiv(D, 32)
    out_bits = torch.empty((M, K_words), dtype=torch.int32, device=x.device)
    BM = 128 if M > 2048 else 64
    grid = (triton.cdiv(M, BM), K_words)
    _pack_sign_bits_kernel[grid](
        x_flat, out_bits,
        x_flat.stride(0), x_flat.stride(1),
        out_bits.stride(0), out_bits.stride(1),
        M, D, K_words,
        BLOCK_M=BM
    )
    return out_bits.reshape(*orig_shape[:-1], K_words)

def triton_popc_sign_similarity(
    x_bits: torch.Tensor,
    w_bits: torch.Tensor,
    scale: Optional[float] = None,
    D: Optional[int] = None,
    dim: Optional[int] = None
) -> torch.Tensor:
    if not HAS_TRITON or not x_bits.is_cuda:
        orig_shape = x_bits.shape
        K_words = x_bits.shape[-1]
        actual_D = D if D is not None else (dim if dim is not None else K_words * 32)
        x_flat = x_bits.reshape(-1, K_words)
        w_flat = w_bits.contiguous()
        M = x_flat.shape[0]
        N = w_flat.shape[0]
        out = torch.empty((M, N), dtype=torch.int32, device=x_bits.device)
        for i in range(M):
            for j in range(N):
                rem = actual_D % 32
                total_pop = 0
                for k in range(K_words):
                    v = int(x_flat[i][k].item()) ^ int(w_flat[j][k].item())
                    if k == K_words - 1 and rem != 0:
                        mask = (1 << rem) - 1 if rem < 31 else 0x7FFFFFFF
                        v = (v & 0xFFFFFFFF) & mask
                    total_pop += bin(v & 0xFFFFFFFF).count('1')
                sim = actual_D - 2 * total_pop
                out[i, j] = sim
        out_reshaped = out.reshape(*orig_shape[:-1], N)
        if scale is not None:
            return out_reshaped.float() * scale
        return out_reshaped
    orig_shape = x_bits.shape
    K_words = x_bits.shape[-1]
    actual_D = D if D is not None else (dim if dim is not None else K_words * 32)
    x_flat = x_bits.reshape(-1, K_words).contiguous()
    w_flat = w_bits.contiguous()
    M = x_flat.shape[0]
    N = w_flat.shape[0]
    out = torch.empty((M, N), dtype=torch.int32, device=x_bits.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))  # noqa: E731
    kernel = _popc_dot_kernel_popc if _HAS_POPC_ASM else _popc_dot_kernel_swar
    kernel[grid](
        x_flat, w_flat, out,
        x_flat.stride(0), x_flat.stride(1),
        w_flat.stride(0), w_flat.stride(1),
        out.stride(0), out.stride(1),
        M, N,
        K_WORDS=K_words,
        D=actual_D,
    )
    out_reshaped = out.reshape(*orig_shape[:-1], N)
    if scale is not None:
        return out_reshaped.float() * scale
    return out_reshaped
