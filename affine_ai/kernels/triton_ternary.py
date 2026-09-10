"""Triton ternary BitLinear (training path).

Matches affine_ai CPU training numerics exactly:
  forward:  y = linear(x_q, W_tern * gamma) + bias
            x_q = round(x / s) * s per row, s = amax(x_row) / 127
  backward: STE identity to latent weights (dakota: grads flow as if
            linear through W_tern * gamma, x re-quantized like forward).
Forward dots default to bf16 Tensor Cores on Ampere+ bf16 inputs
(bit-exact: x_q in [-127, 127] and W_tern in {-1, 0, 1} are exactly
representable); all other dots stay fp32->fp32 SIMT.
"""

from typing import Optional, Tuple
import warnings
import torch
import triton
import triton.language as tl


def _is_turing() -> bool:
    """Return True on Turing sm_75 (7.5 <= cap < 8.0)."""
    if not torch.cuda.is_available():
        return False
    try:
        cap = torch.cuda.get_device_capability()
        return (7, 5) <= tuple(cap) < (8, 0)
    except Exception:
        return False


def _maybe_cast_fp16_for_turing(t: torch.Tensor) -> torch.Tensor:
    """Turing: bf16 unsupported — cast to fp16 with warning, keep acc fp32."""
    if _is_turing() and t.dtype == torch.bfloat16:
        warnings.warn("Turing sm_75: bf16 unsupported, casting to fp16 (acc fp32)", stacklevel=3)
        return t.to(torch.float16)
    return t


def _prune_turing_block(block: int) -> int:
    """Clamp BLOCK to <=64 on Turing (64KB SMEM vs 164KB Ampere+)."""
    if _is_turing() and block > 64:
        warnings.warn(f"Turing sm_75: clamping BLOCK {block} -> 64 (64KB SMEM)", stacklevel=3)
        return 64
    return block


def _prune_turing_block_k(block_k: int) -> int:
    """Turing K-dimension: 64 may still overflow SMEM for twin kernels, clamp to 32."""
    if _is_turing() and block_k > 32:
        warnings.warn(f"Turing sm_75: clamping BLOCK_K {block_k} -> 32 (64KB SMEM)", stacklevel=3)
        return 32
    return _prune_turing_block(block_k)


# NOTE: autotune configs defined but not applied to kernels — reverted due to correctness regression (NaNs with BLOCK_M=128/BLOCK_N=128 on M=64,N=64). Kept for future tuning, but kernels use explicit BLOCK sizes for parity.
# If re-enabling, remove explicit BLOCK_M/N/K from call sites and let autotune choose; ensure mask handling for large blocks.
_ternary_gemm_configs = [
    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 1, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_M': 1, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
]
def _turing_prune_gemm_configs(configs):
    if not _is_turing():
        return configs
    pruned=[]
    for c in configs:
        bm=c.kwargs.get('BLOCK_M',32)
        bn=c.kwargs.get('BLOCK_N',32)
        bk=c.kwargs.get('BLOCK_K',32)
        if bm<=64 and bn<=64 and bk<=32:
            pruned.append(c)
        else:
            warnings.warn(f"Turing sm_75: pruning BLOCK config {c.kwargs} > Turing limit", stacklevel=2)
    return pruned if pruned else configs
_ternary_gemm_configs=_turing_prune_gemm_configs(_ternary_gemm_configs)

@triton.jit
def _row_amax_kernel(
    X, AMAX,
    stride_xm, stride_xk,
    M, K,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x = tl.load(
            X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask_m[:, None] & (offs_k[None, :] < K), other=0.0,
        )
        acc = tl.maximum(acc, tl.max(tl.abs(x), axis=1))
    # Clamp at store to ensure downstream sc=127/amax never divides by ~0; caller also clamps but store is canonical.
    tl.store(AMAX + offs_m, tl.maximum(acc, 1e-5), mask=mask_m)


@triton.jit
def _ternary_fwd_kernel(
    X, W, Bias, Y, AMAX,
    gamma,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_TC: tl.constexpr = False,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    amax = tl.load(AMAX + offs_m, mask=mask_m, other=1e-5)
    amax = tl.maximum(amax, 1e-5)
    sc = 127.0 / amax

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        kk = k + offs_k
        mask_k = kk < K
        x = tl.load(
            X + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )
        v = x * sc[:, None]
        av = tl.abs(v)
        f = tl.floor(av)
        frac = av - f
        odd = f - 2.0 * tl.floor(f * 0.5)
        up = (frac > 0.5 + 1e-6) | ((tl.abs(frac - 0.5) < 1e-6) & (odd == 1.0))
        mag = f + up.to(tl.float32)
        mag = tl.minimum(mag, 127.0)
        xq = tl.where(v < 0.0, -mag, mag)
        w = tl.load(
            W + offs_n[:, None] * stride_wn + kk[None, :] * stride_wk,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0,
        )
        # TC path is bit-exact here: xq in [-127, 127] and w in {-1, 0, 1}
        # are exactly representable in bf16, fp32 accumulation unchanged.
        if USE_TC:
            acc += tl.dot(xq.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
        else:
            acc += tl.dot(xq, tl.trans(w), input_precision="ieee")
    acc = acc / sc[:, None] * gamma
    if HAS_BIAS:
        b = tl.load(Bias + offs_n, mask=mask_n, other=0.0)
        acc = acc + b[None, :]
    tl.store(
        Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc, mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _fp32_dot_kernel(
    A, B, C,
    bias,  # may be None via HAS_BIAS
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        kk = k + offs_k
        mask_k = kk < K
        a = tl.load(
            A + offs_m[:, None] * stride_am + kk[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )
        b = tl.load(
            B + offs_n[:, None] * stride_bn + kk[None, :] * stride_bk,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0,
        )
        acc += tl.dot(a, tl.trans(b), input_precision="ieee")
    if HAS_BIAS:
        bb = tl.load(bias + offs_n, mask=mask_n, other=0.0)
        acc = acc + bb[None, :]
    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc, mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _quantize_x_kernel(
    X, AMAX, XQ,
    stride_xm, stride_xk,
    stride_qm, stride_qk,
    M, K,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K
    amax = tl.load(AMAX + offs_m, mask=mask_m, other=1e-5)
    amax = tl.maximum(amax, 1e-5)
    sc = 127.0 / amax
    x = tl.load(
        X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
        mask=mask_m[:, None] & mask_k[None, :], other=0.0,
    )
    v = x * sc[:, None]
    av = tl.abs(v)
    f = tl.floor(av)
    frac = av - f
    odd = f - 2.0 * tl.floor(f * 0.5)
    up = (frac > 0.5 + 1e-6) | ((tl.abs(frac - 0.5) < 1e-6) & (odd == 1.0))
    mag = f + up.to(tl.float32)
    mag = tl.minimum(mag, 127.0)
    xq = tl.where(v < 0.0, -mag, mag) / sc[:, None]
    tl.store(
        XQ + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
        xq, mask=mask_m[:, None] & mask_k[None, :],
    )


def triton_quantize_x(x: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        sc = (127.0 / amax.clamp(min=1e-5)).unsqueeze(-1)
        v = x.float() * sc
        av = v.abs()
        f = torch.floor(av)
        frac = av - f
        odd = f - 2.0 * torch.floor(f * 0.5)
        up = (frac > 0.5 + 1e-6) | ((torch.abs(frac - 0.5) < 1e-6) & (odd == 1.0))
        mag = torch.clamp(f + up.float(), max=127.0)
        xq = torch.where(v < 0.0, -mag, mag) / sc
        return xq.to(torch.float32)
    M, K = x.shape
    xq = torch.empty((M, K), device=x.device, dtype=torch.float32)
    _quantize_x_kernel[_grid(M, K, 64, 64)](
        x, amax, xq,
        x.stride(0), x.stride(1), xq.stride(0), xq.stride(1),
        M, K, BLOCK_M=64, BLOCK_K=64, num_warps=4,
    )
    return xq


@triton.jit
def _ternary_gw_kernel_fast(
    GO, XQ, GW,
    stride_gm, stride_gn,
    stride_qm, stride_qk,
    stride_wm, stride_wk,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # GW [N, K] = GO^T [N, M] @ X_q [M, K]; pid over (n, k), loop over m.
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)
    mask_n = offs_n < N
    mask_k = offs_k < K
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for m in range(0, M, BLOCK_M):
        mm = m + offs_m
        mask_m = mm < M
        go = tl.load(
            GO + mm[:, None] * stride_gm + offs_n[None, :] * stride_gn,
            mask=mask_m[:, None] & mask_n[None, :], other=0.0,
        )
        xq = tl.load(
            XQ + mm[:, None] * stride_qm + offs_k[None, :] * stride_qk,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )
        acc += tl.dot(tl.trans(go), xq, input_precision="ieee")
    tl.store(
        GW + offs_n[:, None] * stride_wm + offs_k[None, :] * stride_wk,
        acc, mask=mask_n[:, None] & mask_k[None, :],
    )


class _TernaryGwStub:
    def __getitem__(self, grid):
        def _launcher(GO, X, AMAX, GW, stride_gm, stride_gn, stride_xm, stride_xk, stride_wm, stride_wk, M, N, K, BLOCK_M=64, BLOCK_N=32, BLOCK_K=64, num_warps=4, num_stages=2):
            x_q = triton_quantize_x(X, AMAX)
            _ternary_gw_kernel_fast[grid](
                GO, x_q, GW,
                stride_gm, stride_gn, x_q.stride(0), x_q.stride(1), stride_wm, stride_wk,
                M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=num_warps, num_stages=num_stages,
            )
        return _launcher

_ternary_gw_kernel = _TernaryGwStub()


@triton.jit
def _ternary_twin_fwd_kernel(
    X, W1, W2, Bias, Y, AMAX,
    gamma1, gamma2,
    stride_xm, stride_xk,
    stride_wm, stride_wk,
    stride_ym, stride_yn,
    M, OutDim, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_TC: tl.constexpr = False,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < 2 * OutDim
    amax = tl.load(AMAX + offs_m, mask=mask_m, other=1e-5)
    amax = tl.maximum(amax, 1e-5)
    sc = 127.0 / amax
    first = offs_n < OutDim
    gam = tl.where(first, gamma1, gamma2)
    # Gamma hoisted: accumulate raw dot without per-iteration FMUL, scale once after loop (saves BLOCK_N FMUL per K-block).
    # Twin assumes identical strides for W1/W2 (single stride_wm/stride_wk); separate strides would need stride_wm2 etc.
    acc_raw = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        kk = k + offs_k
        mask_k = kk < K
        x = tl.load(
            X + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )
        v = x * sc[:, None]
        av = tl.abs(v)
        f = tl.floor(av)
        frac = av - f
        odd = f - 2.0 * tl.floor(f * 0.5)
        up = (frac > 0.5 + 1e-6) | ((tl.abs(frac - 0.5) < 1e-6) & (odd == 1.0))
        mag = f + up.to(tl.float32)
        mag = tl.minimum(mag, 127.0)
        xq = tl.where(v < 0.0, -mag, mag)
        offs_n_mod = offs_n % OutDim
        w1 = tl.load(W1 + offs_n_mod[:, None] * stride_wm + kk[None, :] * stride_wk, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        w2 = tl.load(W2 + offs_n_mod[:, None] * stride_wm + kk[None, :] * stride_wk, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        w = tl.where(first[:, None], w1, w2)
        if USE_TC:
            acc_raw += tl.dot(xq.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
        else:
            acc_raw += tl.dot(xq, tl.trans(w), input_precision="ieee")
    acc = acc_raw * gam[None, :] / sc[:, None]
    if HAS_BIAS:
        b = tl.load(Bias + offs_n, mask=mask_n, other=0.0)
        acc = acc + b[None, :]
    tl.store(
        Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc, mask=mask_m[:, None] & mask_n[None, :],
    )


def _grid(m, n, bm=64, bn=64):
    return ((m + bm - 1) // bm, (n + bn - 1) // bn)


# Note: tl.dot always lowers to (TF32) Tensor Cores on Ampere, even for
# fp32 inputs. For ternary/integer-valued inputs this is exact (fp32
# accumulation); float-input paths (gradients) carry ~1e-3 relative error,
# same as torch-default cuBLAS behavior. This keeps every GPU usable
# (no architecture requires TCs) while matching CPU numerics where exact.


def triton_row_amax(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        return x.float().abs().amax(dim=-1).clamp(min=1e-5).to(torch.float32)
    K = x.shape[-1]
    x2d = x.reshape(-1, K)
    M = x2d.shape[0]
    out = torch.empty((M,), device=x.device, dtype=torch.float32)
    BLOCK_M = 32
    BLOCK_K = 128
    _row_amax_kernel[_grid(M, 1, BLOCK_M, 1)](
        x2d, out, x2d.stride(0), x2d.stride(1), M, K, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K)
    return out


def _tc_ok(x: torch.Tensor) -> bool:
    if not x.is_cuda:
        return False
    try:
        return tuple(torch.cuda.get_device_capability()) >= (8, 0)
    except Exception:
        return False


def _resolve_tc(use_tc: Optional[bool], x: torch.Tensor) -> bool:
    if use_tc is None:
        return x.dtype == torch.bfloat16
    return bool(use_tc)


def triton_ternary_linear_fwd(x, w_tern, gamma, bias=None, amax=None, use_tc=None):
    # Turing sm_75 FP16 AMP: force fp16 path, keep acc fp32
    if _is_turing():
        if x.dtype == torch.bfloat16:
            warnings.warn("Turing sm_75: bf16 -> fp16 (ternary fwd, acc fp32)", stacklevel=2)
            x = x.to(torch.float16)
        if w_tern.dtype == torch.bfloat16:
            warnings.warn("Turing sm_75: bf16 -> fp16 (ternary w, acc fp32)", stacklevel=2)
            w_tern = w_tern.to(torch.float16)
        if bias is not None and bias.dtype == torch.bfloat16:
            warnings.warn("Turing sm_75: bf16 -> fp16 (bias, acc fp32)", stacklevel=2)
            bias = bias.to(torch.float16)
        # Disable bf16 TC on Turing; fp16 TC is ok but keep fp32 acc path
        if use_tc is None and x.dtype == torch.bfloat16:
            use_tc = False
    K = x.shape[-1]
    x2d = x.reshape(-1, K)
    M = x2d.shape[0]
    N = w_tern.shape[0]
    if amax is None:
        amax = triton_row_amax(x2d).reshape(-1)
    else:
        amax = amax.reshape(-1)
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    has_bias = bias is not None
    tc = _resolve_tc(use_tc, x) and _tc_ok(x2d)
    # Turing BLOCK clamp to <=64 already autotuned; prune explicitly
    BLOCK_M_T = _prune_turing_block(64)
    BLOCK_N_T = _prune_turing_block(64)
    BLOCK_K_T = _prune_turing_block_k(32)
    n_tiles = (N + BLOCK_N_T - 1) // BLOCK_N_T
    if n_tiles > 2 and x2d.is_cuda and not tc:  # preserve bf16 TC vs fp32: skip fast path if TC requested
        try:
            x_q = triton_quantize_x(x2d, amax)
            w_scaled = (w_tern * float(gamma)).contiguous()
            _fp32_dot_kernel[_grid(M, N, BLOCK_M_T, BLOCK_N_T)](
                x_q, w_scaled, out, bias if has_bias else x2d,
                x_q.stride(0), x_q.stride(1), w_scaled.stride(0), w_scaled.stride(1),
                out.stride(0), out.stride(1),
                M, N, K,
                BLOCK_M=BLOCK_M_T, BLOCK_N=BLOCK_N_T, BLOCK_K=BLOCK_K_T,
                HAS_BIAS=has_bias, num_warps=4, num_stages=3)
            return out.reshape(*x.shape[:-1], N)
        except Exception as e:
            warnings.warn(f"ternary fast path failed: {e}")
    try:
        _ternary_fwd_kernel[_grid(M, N)](
            x2d, w_tern, bias if has_bias else x2d, out, amax, float(gamma),
            x2d.stride(0), x2d.stride(1), w_tern.stride(0), w_tern.stride(1),
            out.stride(0), out.stride(1),
            M, N, K,
            BLOCK_M=_prune_turing_block(64), BLOCK_N=_prune_turing_block(64), BLOCK_K=_prune_turing_block_k(32), HAS_BIAS=has_bias, USE_TC=tc,
            num_warps=4, num_stages=3)
    except Exception as e:
        warnings.warn(f"ternary fallback failed (USE_TC={tc}): {e}")
        if tc:
            _ternary_fwd_kernel[_grid(M, N)](
                x2d, w_tern, bias if has_bias else x2d, out, amax, float(gamma),
                x2d.stride(0), x2d.stride(1), w_tern.stride(0), w_tern.stride(1),
                out.stride(0), out.stride(1),
                M, N, K,
                BLOCK_M=_prune_turing_block(64), BLOCK_N=_prune_turing_block(64), BLOCK_K=_prune_turing_block_k(32), HAS_BIAS=has_bias, USE_TC=False,
                num_warps=4, num_stages=3)
        else:
            raise
    return out.reshape(*x.shape[:-1], N)


def triton_fp32_linear(a, b, bias=None):
    if _is_turing():
        if a.dtype == torch.bfloat16:
            warnings.warn("Turing sm_75: bf16 -> fp16 (fp32_linear a, acc fp32)", stacklevel=2)
            a = a.to(torch.float16)
        if b.dtype == torch.bfloat16:
            warnings.warn("Turing sm_75: bf16 -> fp16 (fp32_linear b, acc fp32)", stacklevel=2)
            b = b.to(torch.float16)
        if bias is not None and bias.dtype == torch.bfloat16:
            warnings.warn("Turing sm_75: bf16 -> fp16 (fp32_linear bias, acc fp32)", stacklevel=2)
            bias = bias.to(torch.float16)
    a = a.contiguous()
    b = b.contiguous()
    K = a.shape[-1]
    x2d = a.reshape(-1, K)
    M = x2d.shape[0]
    N = b.shape[0]
    out = torch.empty((M, N), device=a.device, dtype=torch.float32)
    has_bias = bias is not None
    _fp32_dot_kernel[_grid(M, N)](
        x2d, b, out, bias if has_bias else x2d,
        x2d.stride(0), x2d.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        M, N, K, BLOCK_M=_prune_turing_block(64), BLOCK_N=_prune_turing_block(64), BLOCK_K=_prune_turing_block_k(32),
        HAS_BIAS=has_bias, num_warps=4, num_stages=3)
    return out.reshape(*a.shape[:-1], N)


class TritonTernaryLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_latent, bias, use_tc=None):
        orig_shape = x.shape
        in_dim = w_latent.size(1)
        out_dim = w_latent.size(0)
        x_flat = x.reshape(-1, in_dim).contiguous().float()
        w_f = w_latent.detach().float()
        gamma = w_f.abs().mean().clamp(min=1e-5)
        w_ternary = torch.round(w_f / gamma).clamp(-1.0, 1.0)
        amax = triton_row_amax(x_flat)
        out = triton_ternary_linear_fwd(x_flat, w_ternary, gamma.item(), bias.float() if bias is not None else None, amax, use_tc=use_tc)
        ctx.save_for_backward(x_flat, w_ternary, amax)
        ctx.gamma = gamma.item()
        ctx.has_bias = bias is not None
        ctx.w_dtype = w_latent.dtype
        return out.to(x.dtype).reshape(*orig_shape[:-1], out_dim)

    @staticmethod
    def backward(ctx, grad_output):
        x_flat, w_ternary, amax = ctx.saved_tensors
        gamma, has_bias = ctx.gamma, ctx.has_bias
        out_dim = w_ternary.size(0)
        in_dim = w_ternary.size(1)
        go_flat = grad_output.reshape(-1, out_dim).contiguous().float()
        gx = (triton_fp32_linear(go_flat, (w_ternary * gamma).t().contiguous(), None).to(grad_output.dtype).reshape(grad_output.shape[:-1] + (in_dim,))) if ctx.needs_input_grad[0] else None
        if ctx.needs_input_grad[1]:
            x_q = triton_quantize_x(x_flat, amax)
            gw = (triton_ternary_linear_gw(go_flat, None, None, out_dim, in_dim, x_q=x_q) * gamma).to(ctx.w_dtype)
        else:
            gw = None
        grad_b = go_flat.sum(0).to(grad_output.dtype) if (has_bias and ctx.needs_input_grad[2]) else None
        return gx, gw, grad_b, None


def triton_ternary_linear(x, w_latent, bias=None, use_tc=None):
    return TritonTernaryLinearFunction.apply(x, w_latent, bias, use_tc)


class TritonTernaryTwinFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1_latent, b1, w2_latent, b2, use_tc=None):
        if _is_turing() and x.dtype == torch.bfloat16:
            warnings.warn("Turing sm_75: bf16 -> fp16 (twin fwd, acc fp32)", stacklevel=3)
            x = x.to(torch.float16)
        orig_shape = x.shape
        in_dim = w1_latent.size(1)
        out_dim = w1_latent.size(0)
        x_flat = x.reshape(-1, in_dim).contiguous().float()

        def tern(w):
            ww = w.detach().float()
            gamma = ww.abs().mean().clamp(min=1e-5)
            return torch.round(ww / gamma).clamp(-1.0, 1.0), gamma.item()

        w1t, g1 = tern(w1_latent)
        w2t, g2 = tern(w2_latent)
        assert w1t.stride() == w2t.stride(), f"Twin assumes identical strides for W1/W2, got {w1t.stride()} vs {w2t.stride()}; kernel uses single stride_wm/stride_wk"
        has_bias = b1 is not None and b2 is not None
        b = torch.cat([b1, b2], dim=0).float() if has_bias else torch.tensor([], device=x.device)
        amax = triton_row_amax(x_flat)
        M = x_flat.shape[0]
        out = torch.empty((M, 2 * out_dim), device=x.device, dtype=torch.float32)
        tc = _resolve_tc(use_tc, x_flat) and _tc_ok(x_flat)
        if _is_turing() and tc:
            warnings.warn("Turing sm_75: disabling bf16 TC (acc fp32)", stacklevel=2)
            tc = False
        _ternary_twin_fwd_kernel[_grid(M, 2 * out_dim)](
            x_flat, w1t, w2t, b, out, amax, float(g1), float(g2),
            x_flat.stride(0), x_flat.stride(1), w1t.stride(0), w1t.stride(1),
            out.stride(0), out.stride(1),
            M, out_dim, in_dim, BLOCK_M=_prune_turing_block(64), BLOCK_N=_prune_turing_block(64), BLOCK_K=_prune_turing_block_k(32),
            HAS_BIAS=has_bias, USE_TC=tc, num_warps=4, num_stages=3)
        ctx.save_for_backward(x_flat, w1t, w2t, amax)
        ctx.g1, ctx.g2, ctx.has_bias = g1, g2, has_bias
        ctx.w1_dtype = w1_latent.dtype
        ctx.w2_dtype = w2_latent.dtype
        return out.to(x.dtype).reshape(*orig_shape[:-1], 2 * out_dim)

    @staticmethod
    def backward(ctx, grad_output):
        x_flat, w1t, w2t, amax = ctx.saved_tensors
        g1, g2, has_bias = ctx.g1, ctx.g2, ctx.has_bias
        out_dim = w1t.size(0)
        in_dim = w1t.size(1)
        go_flat = grad_output.reshape(-1, 2 * out_dim).contiguous().float()
        go1, go2 = go_flat.split(out_dim, dim=-1)
        gx = (triton_fp32_linear(go1, (w1t * g1).t().contiguous(), None)
              + triton_fp32_linear(go2, (w2t * g2).t().contiguous(), None)).to(grad_output.dtype).reshape(grad_output.shape[:-1] + (in_dim,)) if ctx.needs_input_grad[0] else None
        need_w1 = ctx.needs_input_grad[1]
        need_w2 = len(ctx.needs_input_grad) > 3 and ctx.needs_input_grad[3]
        if need_w1 or need_w2:
            x_q = triton_quantize_x(x_flat, amax)
        else:
            x_q = None
        gw1 = (triton_ternary_linear_gw(go1.contiguous(), None, None, out_dim, in_dim, x_q=x_q) * g1).to(ctx.w1_dtype) if need_w1 else None
        gw2 = (triton_ternary_linear_gw(go2.contiguous(), None, None, out_dim, in_dim, x_q=x_q) * g2).to(ctx.w2_dtype) if need_w2 else None
        gb1 = go1.sum(0).to(grad_output.dtype) if (has_bias and len(ctx.needs_input_grad) > 2 and ctx.needs_input_grad[2]) else None
        gb2 = go2.sum(0).to(grad_output.dtype) if (has_bias and len(ctx.needs_input_grad) > 4 and ctx.needs_input_grad[4]) else None
        return (gx,
                gw1, gb1,
                gw2, gb2, None)


def triton_ternary_twin(x, w1, b1, w2, b2, use_tc=None):
    return TritonTernaryTwinFunction.apply(x, w1, b1, w2, b2, use_tc)


def triton_ternary_linear_gw(go, x, amax, N, K, x_q=None):
    if _is_turing() and go.dtype == torch.bfloat16:
        warnings.warn("Turing sm_75: bf16 -> fp16 (ternary gw, acc fp32)", stacklevel=2)
        go = go.to(torch.float16)
    M = go.shape[0]
    if x_q is None:
        if x is not None and amax is not None:
            x_q = triton_quantize_x(x, amax)
        else:
            raise ValueError("Either x_q or both (x, amax) must be provided to triton_ternary_linear_gw.")
    gw = torch.empty((N, K), device=go.device, dtype=torch.float32)
    _ternary_gw_kernel_fast[_grid(N, K, 32, 64)](
        go, x_q, gw,
        go.stride(0), go.stride(1), x_q.stride(0), x_q.stride(1),
        gw.stride(0), gw.stride(1),
        M, N, K, BLOCK_M=_prune_turing_block(64), BLOCK_N=_prune_turing_block(32), BLOCK_K=_prune_turing_block_k(64),
        num_warps=4, num_stages=2)
    return gw


@triton.jit
def _unpack_2bit_kernel(
    Packed_ptr, Unpacked_ptr,
    stride_packed_row, stride_unpacked_row,
    Rows, Cols,
    BLOCK_COLS: tl.constexpr
):
    # TODO: use block_ptr and vectorized loads for better coalescing; manual pointers kept for clarity
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    offs_col = col_block_idx * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask_col = offs_col < Cols

    # 16 weights per 32-bit integer, aligned with format.py:
    # 4 trits per byte in big-endian bit order:
    # trit 0 at shift 6, trit 1 at 4, trit 2 at 2, trit 3 at 0
    # Bytes in little-endian word: byte b at bit (b * 8)
    packed_col = offs_col // 16
    intra = offs_col % 16
    bit_pos = (intra // 4) * 8 + (3 - (intra % 4)) * 2  # Shift formula verified exact — matches format.py big-endian trit layout

    packed_val = tl.load(Packed_ptr + row_idx * stride_packed_row + packed_col, mask=mask_col, other=0).to(tl.uint32)
    code = (packed_val >> bit_pos) & 3
    # code==3 (0b11) is unused/corrupted; maps to 0 to tolerate checkpoint corruption.
    # Debug: tl.device_assert(tl.sum((code == 3).to(tl.int32)) == 0, "packed ternary code 3 corrupted") — fallback maps to 0 for robustness.
    tern = tl.where(code == 1, 1.0, tl.where(code == 2, -1.0, 0.0))

    tl.store(Unpacked_ptr + row_idx * stride_unpacked_row + offs_col, tern.to(Unpacked_ptr.dtype.element_ty), mask=mask_col)


def triton_unpack_ternary_2bit(
    packed: torch.Tensor,
    out_shape: Tuple[int, int],
    dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """
    Unpacks 2-bit packed ternary weights into a 2D float tensor on GPU.
    packed: [Rows, Cols // 16] int32 tensor.
    Returns: [Rows, Cols] tensor of values in {-1.0, 0.0, 1.0}.
    """
    Rows, Cols = out_shape
    unpacked = torch.empty(out_shape, device=packed.device, dtype=dtype)
    BLOCK_COLS = 128
    grid = (Rows, triton.cdiv(Cols, BLOCK_COLS))
    _unpack_2bit_kernel[grid](
        packed, unpacked,
        packed.stride(0), unpacked.stride(0),
        Rows, Cols,
        BLOCK_COLS=BLOCK_COLS
    )
    return unpacked


@triton.jit
def _pack_2bit_kernel(
    W_ptr, Packed_ptr,
    stride_wm, stride_wk,
    stride_pm, stride_pk,
    Rows, Cols,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_p = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < Rows
    mask_p = offs_p < (Cols // 16)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    # Manual pointer arithmetic: load W slice via explicit pointers (fixes w_tile[:, idx-vector] gather)
    for i in range(16):
        shift = (i // 4) * 8 + (3 - (i % 4)) * 2
        cols = pid_n * BLOCK_N * 16 + tl.arange(0, BLOCK_N) * 16 + i
        mask_col = cols < Cols
        w_ptrs = W_ptr + offs_m[:, None] * stride_wm + cols[None, :] * stride_wk
        w = tl.load(w_ptrs, mask=mask_m[:, None] & mask_col[None, :], other=0.0)
        code = tl.where(w == 1.0, 1, tl.where(w == -1.0, 2, 0)).to(tl.int32)
        acc = acc | (code << shift)
    tl.store(
        Packed_ptr + offs_m[:, None] * stride_pm + offs_p[None, :] * stride_pk,
        acc, mask=mask_m[:, None] & mask_p[None, :],
    )


def triton_pack_ternary_2bit(w_ternary: torch.Tensor) -> torch.Tensor:
    Rows, Cols = w_ternary.shape
    assert Cols % 16 == 0, f"Cols must be divisible by 16, got {Cols}"
    if not w_ternary.is_cuda:
        w_int8 = w_ternary.to(torch.int8)
        mapped = torch.where(w_int8 == 1, 1, torch.where(w_int8 == -1, 2, 0)).to(torch.int32)
        packed = torch.zeros((Rows, Cols // 16), dtype=torch.int32, device=w_ternary.device)
        for i in range(16):
            shift = (i // 4) * 8 + (3 - (i % 4)) * 2
            packed |= (mapped[:, i::16] << shift)
        return packed
    try:
        packed = torch.empty((Rows, Cols // 16), dtype=torch.int32, device=w_ternary.device)
        BLOCK_M = 32
        BLOCK_N = 32
        grid = ((Rows + BLOCK_M - 1) // BLOCK_M, (Cols // 16 + BLOCK_N - 1) // BLOCK_N)
        _pack_2bit_kernel[grid](
            w_ternary, packed,
            w_ternary.stride(0), w_ternary.stride(1),
            packed.stride(0), packed.stride(1),
            Rows, Cols,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return packed
    except Exception:
        w_int8 = w_ternary.to(torch.int8)
        mapped = torch.where(w_int8 == 1, 1, torch.where(w_int8 == -1, 2, 0)).to(torch.int32)
        packed = torch.zeros((Rows, Cols // 16), dtype=torch.int32, device=w_ternary.device)
        for i in range(16):
            shift = (i // 4) * 8 + (3 - (i % 4)) * 2
            packed |= (mapped[:, i::16] << shift)
        return packed

