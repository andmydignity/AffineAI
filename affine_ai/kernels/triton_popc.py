
"""
Custom Triton Kernel: 1-Bit Binary Router (SWAR POPC)
======================================================
Executes binary dot products via SWAR emulated popcount (single-cycle PTX
popc.b32 would be faster, used for portability). Replaces 32 floating-point
FMAs with 1 XOR and 1 POPC per 32 dimensions.
"""

import math
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
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        mask_k = offs_k < K_words

        offs_b = tl.arange(0, 32)
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.int32)
        for ik in range(BLOCK_K):
            k = pid_k * BLOCK_K + ik
            cols = k * 32 + offs_b
            mask = mask_m[:, None] & (cols[None, :] < D)
            val = tl.load(X_ptr + offs_m[:, None] * stride_xm + cols[None, :] * stride_xd, mask=mask, other=-1.0)
            is_pos = (val >= 0.0).to(tl.uint32)
            bits = is_pos << offs_b[None, :]
            acc_k = tl.sum(bits, axis=1).to(tl.int32)
            acc = tl.where(tl.arange(0, BLOCK_K)[None, :] == ik, acc_k[:, None], acc)

        # Coalesced global stores along words (stride_ok == 1)
        out_ptrs = Out_bits_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_k[None, :])

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
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.uint32)
        rem = D % 32
        for k_base in range(0, K_WORDS, BLOCK_K):
            for ik in range(BLOCK_K):
                k = k_base + ik
                if k < K_WORDS:
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
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.uint32)
        rem = D % 32
        for k_base in range(0, K_WORDS, BLOCK_K):
            for ik in range(BLOCK_K):
                k = k_base + ik
                if k < K_WORDS:
                    x = tl.load(X_bits_ptr + offs_m[:, None] * stride_xm + k * stride_xk, mask=mask_m[:, None], other=0).to(tl.uint32)
                    w = tl.load(W_bits_ptr + offs_n[None, :] * stride_wn + k * stride_wk, mask=mask_n[None, :], other=0).to(tl.uint32)
                    diff = x ^ w
                    tail = (k == K_WORDS - 1) and (rem != 0)
                    if tail:
                        mask = (tl.full((), 1, dtype=tl.uint32) << rem) - 1
                        diff = diff & mask
                    pop = _popcount32_swar(diff)
                    sim = tl.where(tail, rem - 2 * pop, 32 - 2 * pop)
                    acc += sim.to(tl.uint32)
        tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc.to(tl.int32), mask=mask_m[:, None] & mask_n[None, :])

    def _get_popc_dot_configs():
        return [
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 4}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 4}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 4}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 4}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 4}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 4}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 4}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 4}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 4}, num_warps=8, num_stages=2),
        ]
    _IS_GENUINE_CUDA = bool(torch.cuda.is_available() and getattr(torch.version, "cuda", None) is not None)
    _HAS_POPC_ASM = _IS_GENUINE_CUDA and hasattr(tl, "inline_asm_elementwise")
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
    BK = 4
    BM = 128 if M > 2048 else 64
    grid = (triton.cdiv(M, BM), triton.cdiv(K_words, BK))
    _pack_sign_bits_kernel[grid](
        x_flat, out_bits,
        x_flat.stride(0), x_flat.stride(1),
        out_bits.stride(0), out_bits.stride(1),
        M, D, K_words,
        BLOCK_M=BM,
        BLOCK_K=BK,
    )
    return out_bits.reshape(*orig_shape[:-1], K_words)

def _unpack_sign_bits_to_sign(bits: torch.Tensor, D: int) -> torch.Tensor:
    K_words = bits.shape[-1]
    bits_flat = bits.reshape(-1, K_words)
    M = bits_flat.shape[0]
    offs_b = torch.arange(32, device=bits.device, dtype=torch.int32)
    expanded = (bits_flat.unsqueeze(-1) >> offs_b) & 1
    unpacked = expanded.reshape(M, K_words * 32)[:, :D]
    sign = torch.where(unpacked == 1, 1.0, -1.0)
    return sign.reshape(*bits.shape[:-1], D)


class TritonPopcSignSimilarityFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        scale: Optional[float] = None,
        D: Optional[int] = None,
        dim: Optional[int] = None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        actual_D = D if D is not None else (dim if dim is not None else orig_shape[-1])

        if x.is_floating_point():
            x_sign = torch.where(x >= 0, torch.tensor(1.0, dtype=x.dtype, device=x.device), torch.tensor(-1.0, dtype=x.dtype, device=x.device))
            x_bits = triton_pack_sign_bits(x)
        else:
            x_bits = x
            x_sign = _unpack_sign_bits_to_sign(x, actual_D).to(torch.float32)

        if isinstance(w, torch.Tensor) and w.is_floating_point():
            w_sign = torch.where(w >= 0, torch.tensor(1.0, dtype=w.dtype, device=w.device), torch.tensor(-1.0, dtype=w.dtype, device=w.device))
            w_bits = triton_pack_sign_bits(w)
        else:
            w_bits = w
            w_sign = _unpack_sign_bits_to_sign(w, actual_D).to(torch.float32)

        sim = triton_popc_sign_similarity(x_bits, w_bits, scale=None, D=actual_D)

        eff_scale = scale if scale is not None else math.sqrt(2.0 / (math.pi * actual_D))
        ctx.eff_scale = eff_scale
        ctx.actual_D = actual_D
        ctx.orig_shape = orig_shape
        ctx.save_for_backward(x_sign, w_sign)

        out = sim.to(x.dtype if x.is_floating_point() else torch.float32)
        if scale is not None:
            out = out * scale
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_sign, w_sign = ctx.saved_tensors
        eff_scale = ctx.eff_scale
        actual_D = ctx.actual_D

        grad_x = None
        grad_w = None

        if ctx.needs_input_grad[0]:
            gx = torch.matmul(grad_output, w_sign.to(grad_output.dtype)) * eff_scale
            grad_x = gx.to(grad_output.dtype)

        if ctx.needs_input_grad[1]:
            go_flat = grad_output.reshape(-1, grad_output.shape[-1])
            x_sign_flat = x_sign.reshape(-1, actual_D)
            gw = torch.matmul(go_flat.transpose(0, 1), x_sign_flat.to(grad_output.dtype)) * eff_scale
            grad_w = gw.to(grad_output.dtype)

        return grad_x, grad_w, None, None, None


def triton_differentiable_popc_similarity(
    x: torch.Tensor,
    w: torch.Tensor,
    scale: Optional[float] = None,
    D: Optional[int] = None,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Differentiable POPC binary similarity with Straight-Through Estimator (STE).

    Forward computes exact binary dot product between sign(x) and sign(w) (scaled if scale is provided).
    Backward computes STE gradient scaled by sqrt(2 / (pi * D)) or user-provided scale.
    """
    return TritonPopcSignSimilarityFunction.apply(x, w, scale, D, dim)


def triton_popc_sign_similarity(
    x_bits: torch.Tensor,
    w_bits: torch.Tensor,
    scale: Optional[float] = None,
    D: Optional[int] = None,
    dim: Optional[int] = None
) -> torch.Tensor:
    if x_bits.is_floating_point() or (isinstance(w_bits, torch.Tensor) and w_bits.is_floating_point()):
        return TritonPopcSignSimilarityFunction.apply(x_bits, w_bits, scale, D, dim)

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

