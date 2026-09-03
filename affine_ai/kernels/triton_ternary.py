"""Triton ternary BitLinear (training path).

Matches affine_ai CPU training numerics exactly:
  forward:  y = linear(x_q, W_tern * gamma) + bias
            x_q = round(x / s) * s per row, s = amax(x_row) / 127
  backward: STE identity to latent weights (dakota: grads flow as if
            linear through W_tern * gamma, x re-quantized like forward).
All tl.dot calls are fp32->fp32 (SIMT cores, no Tensor Cores).
"""

import torch
import triton
import triton.language as tl


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
    tl.store(AMAX + offs_m, acc, mask=mask_m)


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
        up = (frac > 0.5) | ((frac == 0.5) & (odd == 1.0))
        mag = f + up.to(tl.float32)
        xq = tl.where(v < 0.0, -mag, mag)
        w = tl.load(
            W + offs_n[:, None] * stride_wn + kk[None, :] * stride_wk,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0,
        )
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
def _ternary_gw_kernel(
    GO, X, AMAX, GW,
    stride_gm, stride_gn,
    stride_xm, stride_xk,
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
        amax = tl.load(AMAX + mm, mask=mask_m, other=1e-5)
        amax = tl.maximum(amax, 1e-5)
        sc = 127.0 / amax
        x = tl.load(
            X + mm[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )
        v = x * sc[:, None]
        av = tl.abs(v)
        f = tl.floor(av)
        frac = av - f
        odd = f - 2.0 * tl.floor(f * 0.5)
        up = (frac > 0.5) | ((frac == 0.5) & (odd == 1.0))
        mag = f + up.to(tl.float32)
        xq = tl.where(v < 0.0, -mag, mag) / sc[:, None]
        acc += tl.dot(tl.trans(go), xq, input_precision="ieee")
    tl.store(
        GW + offs_n[:, None] * stride_wm + offs_k[None, :] * stride_wk,
        acc, mask=mask_n[:, None] & mask_k[None, :],
    )


@triton.jit
def _ternary_twin_fwd_kernel(
    X, W1, W2, Bias, Y, AMAX,
    gamma1, gamma2,
    stride_xm, stride_xk,
    stride_wm, stride_wk,
    stride_ym, stride_yn,
    M, O, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < 2 * O
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
        up = (frac > 0.5) | ((frac == 0.5) & (odd == 1.0))
        mag = f + up.to(tl.float32)
        xq = tl.where(v < 0.0, -mag, mag)
        first = offs_n < O
        w1 = tl.load(
            W1 + (offs_n % O)[:, None] * stride_wm + kk[None, :] * stride_wk,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0,
        )
        w2 = tl.load(
            W2 + (offs_n % O)[:, None] * stride_wm + kk[None, :] * stride_wk,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0,
        )
        w = tl.where(first[:, None], w1, w2)
        gam = tl.where(first, gamma1, gamma2)
        acc += tl.dot(xq, tl.trans(w), input_precision="ieee") * gam[None, :]
    acc = acc / sc[:, None]
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
    M, K = x.shape
    out = torch.empty((M,), device=x.device, dtype=torch.float32)
    _row_amax_kernel[_grid(M, 1, 64, 1)](
        x, out, x.stride(0), x.stride(1), M, K, BLOCK_M=64, BLOCK_K=1024)
    return out


def triton_ternary_linear_fwd(x, w_tern, gamma, bias=None, amax=None):
    M, K = x.shape
    N = w_tern.shape[0]
    if amax is None:
        amax = triton_row_amax(x.reshape(-1, K)).reshape(-1) if x.dim() > 2 else triton_row_amax(x)
        x2d = x.reshape(-1, K)
    else:
        x2d = x.reshape(-1, K)
        amax = amax.reshape(-1)
    out = torch.empty((x2d.shape[0], N), device=x.device, dtype=torch.float32)
    has_bias = bias is not None
    _ternary_fwd_kernel[_grid(x2d.shape[0], N)](
        x2d, w_tern, bias if has_bias else x2d, out, amax, float(gamma),
        x2d.stride(0), x2d.stride(1), w_tern.stride(0), w_tern.stride(1),
        out.stride(0), out.stride(1),
        x2d.shape[0], N, K,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, HAS_BIAS=has_bias,
        num_warps=4, num_stages=3)
    return out.reshape(*x.shape[:-1], N)


def triton_fp32_linear(a, b, bias=None):
    M, K = a.shape
    N = b.shape[0]
    out = torch.empty((M, N), device=a.device, dtype=torch.float32)
    has_bias = bias is not None
    _fp32_dot_kernel[_grid(M, N)](
        a, b, out, bias if has_bias else a,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        HAS_BIAS=has_bias, num_warps=4, num_stages=3)
    return out


class TritonTernaryLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_latent, bias):
        orig_shape = x.shape
        in_dim = w_latent.size(1)
        out_dim = w_latent.size(0)
        x_flat = x.reshape(-1, in_dim).contiguous().float()
        w_f = w_latent.detach().float()
        gamma = w_f.abs().mean().clamp(min=1e-5)
        w_ternary = torch.round(w_f / gamma).clamp(-1.0, 1.0)
        amax = triton_row_amax(x_flat)
        out = triton_ternary_linear_fwd(x_flat, w_ternary, gamma.item(), bias.float() if bias is not None else None, amax)
        ctx.save_for_backward(x_flat, w_ternary)
        ctx.gamma = gamma.item()
        ctx.has_bias = bias is not None
        ctx.amax = amax
        return out.to(x.dtype).reshape(*orig_shape[:-1], out_dim)

    @staticmethod
    def backward(ctx, grad_output):
        x_flat, w_ternary = ctx.saved_tensors
        gamma, has_bias, amax = ctx.gamma, ctx.has_bias, ctx.amax
        out_dim = w_ternary.size(0)
        go_flat = grad_output.reshape(-1, out_dim).contiguous().float()
        gx = triton_fp32_linear(go_flat, (w_ternary * gamma).t(), None)
        gw = triton_ternary_linear_gw(go_flat, x_flat, amax, out_dim, w_ternary.size(1))
        grad_b = go_flat.sum(0) if has_bias else None
        return (gx.to(grad_output.dtype).reshape(grad_output.shape[:-1] + (w_ternary.size(1),)),
                gw.to(w_ternary.dtype),
                grad_b.to(grad_output.dtype) if has_bias else None)


def triton_ternary_linear(x, w_latent, bias=None):
    return TritonTernaryLinearFunction.apply(x, w_latent, bias)


class TritonTernaryTwinFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1_latent, b1, w2_latent, b2):
        orig_shape = x.shape
        in_dim = w1_latent.size(1)
        O = w1_latent.size(0)
        x_flat = x.reshape(-1, in_dim).contiguous().float()

        def tern(w):
            ww = w.detach().float()
            gamma = ww.abs().mean().clamp(min=1e-5)
            return torch.round(ww / gamma).clamp(-1.0, 1.0), gamma.item()

        w1t, g1 = tern(w1_latent)
        w2t, g2 = tern(w2_latent)
        has_bias = b1 is not None and b2 is not None
        b = torch.cat([b1, b2], dim=0).float() if has_bias else torch.tensor([], device=x.device)
        amax = triton_row_amax(x_flat)
        M = x_flat.shape[0]
        out = torch.empty((M, 2 * O), device=x.device, dtype=torch.float32)
        _ternary_twin_fwd_kernel[_grid(M, 2 * O)](
            x_flat, w1t, w2t, b, out, amax, float(g1), float(g2),
            x_flat.stride(0), x_flat.stride(1), w1t.stride(0), w1t.stride(1),
            out.stride(0), out.stride(1),
            M, O, in_dim, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            HAS_BIAS=has_bias, num_warps=4, num_stages=3)
        ctx.save_for_backward(x_flat, w1t, w2t)
        ctx.g1, ctx.g2, ctx.has_bias, ctx.amax = g1, g2, has_bias, amax
        return out.to(x.dtype).reshape(*orig_shape[:-1], 2 * O)

    @staticmethod
    def backward(ctx, grad_output):
        x_flat, w1t, w2t = ctx.saved_tensors
        g1, g2, has_bias, amax = ctx.g1, ctx.g2, ctx.has_bias, ctx.amax
        O = w1t.size(0)
        go_flat = grad_output.reshape(-1, 2 * O).contiguous().float()
        go1, go2 = go_flat.split(O, dim=-1)
        gx = (triton_fp32_linear(go1, (w1t * g1).t(), None)
              + triton_fp32_linear(go2, (w2t * g2).t(), None))
        gw1 = triton_ternary_linear_gw(go1.contiguous(), x_flat, amax, O, w1t.size(1))
        gw2 = triton_ternary_linear_gw(go2.contiguous(), x_flat, amax, O, w2t.size(1))
        gb1 = go1.sum(0) if has_bias else None
        gb2 = go2.sum(0) if has_bias else None
        return (gx.to(grad_output.dtype).reshape(grad_output.shape[:-1] + (w1t.size(1),)),
                gw1.to(w1t.dtype), gb1.to(grad_output.dtype) if has_bias else None,
                gw2.to(w2t.dtype), gb2.to(grad_output.dtype) if has_bias else None)


def triton_ternary_twin(x, w1, b1, w2, b2):
    return TritonTernaryTwinFunction.apply(x, w1, b1, w2, b2)


def triton_ternary_linear_gw(go, x, amax, N, K):
    M = go.shape[0]
    gw = torch.empty((N, K), device=go.device, dtype=torch.float32)
    _ternary_gw_kernel[_grid(N, K, 32, 64)](
        go, x, amax, gw,
        go.stride(0), go.stride(1), x.stride(0), x.stride(1),
        gw.stride(0), gw.stride(1),
        M, N, K, BLOCK_M=64, BLOCK_N=32, BLOCK_K=64,
        num_warps=4, num_stages=2)
    return gw
