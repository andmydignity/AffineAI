"""
Custom Triton Kernel: In-SRAM Fused Permutation Projection
==========================================================
Fuses multi-permutation index gathers, ternary/float weight accumulation,
and bias addition directly in GPU SRAM registers.
Computes:
  out[m, n, d] = sum_{p=0}^{P-1} (x[n, perms[p, d]] * w[m, p, d]) + bias[m, d]

Eliminates all high-bandwidth intermediate VRAM gathers [N, P, D] and provides
analytical transposed backward kernels directly in registers.
"""

from typing import Tuple, Optional, Union
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_perm_proj_fwd_kernel(
    X, W, Perms, Biases, Out,
    stride_xn, stride_xd,
    stride_wm, stride_wp, stride_wd,
    stride_pp, stride_pd,
    stride_bm, stride_bd,
    stride_om, stride_on, stride_od,
    N, D, P, M,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_n = offs_n < N
    mask_d = offs_d < D

    acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    if HAS_BIAS:
        bias_val = tl.load(Biases + pid_m * stride_bm + offs_d * stride_bd, mask=mask_d, other=0.0)
    else:
        bias_val = 0.0

    for p in range(P):
        p_idx = tl.load(Perms + p * stride_pp + offs_d * stride_pd, mask=mask_d, other=0)
        x_ptrs = X + offs_n[:, None] * stride_xn + p_idx[None, :] * stride_xd
        w_ptrs = W + pid_m * stride_wm + p * stride_wp + offs_d[None, :] * stride_wd

        x_val = tl.load(x_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        w_val = tl.load(w_ptrs, mask=mask_d[None, :], other=0.0)
        acc += x_val.to(tl.float32) * w_val.to(tl.float32)

    acc += bias_val.to(tl.float32) if HAS_BIAS else 0.0
    out_ptrs = Out + pid_m * stride_om + offs_n[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=mask_n[:, None] & mask_d[None, :])


@triton.jit
def _fused_perm_proj_bwd_gx_kernel(
    GradOut, W, InvPerms, GX,
    stride_gom, stride_gon, stride_god,
    stride_wm, stride_wp, stride_wd,
    stride_ipp, stride_ipd,
    stride_gxn, stride_gxd,
    N, D, P, M,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_n = offs_n < N
    mask_d = offs_d < D

    acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for p in range(P):
        ip_idx = tl.load(InvPerms + p * stride_ipp + offs_d * stride_ipd, mask=mask_d, other=0)
        for m in range(M):
            go_ptrs = GradOut + m * stride_gom + offs_n[:, None] * stride_gon + ip_idx[None, :] * stride_god
            w_ptrs = W + m * stride_wm + p * stride_wp + ip_idx[None, :] * stride_wd
            go_val = tl.load(go_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            w_val = tl.load(w_ptrs, mask=mask_d[None, :], other=0.0)
            acc += go_val.to(tl.float32) * w_val.to(tl.float32)

    gx_ptrs = GX + offs_n[:, None] * stride_gxn + offs_d[None, :] * stride_gxd
    tl.store(gx_ptrs, acc.to(GX.dtype.element_ty), mask=mask_n[:, None] & mask_d[None, :])


@triton.jit
def _fused_perm_proj_bwd_gw_kernel(
    GradOut, X, Perms, GW,
    stride_gom, stride_gon, stride_god,
    stride_xn, stride_xd,
    stride_pp, stride_pd,
    stride_gwm, stride_gwp, stride_gwd,
    N, D,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    p_idx = tl.load(Perms + pid_p * stride_pp + offs_d * stride_pd, mask=mask_d, other=0)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        go_ptrs = GradOut + pid_m * stride_gom + offs_n[:, None] * stride_gon + offs_d[None, :] * stride_god
        x_ptrs = X + offs_n[:, None] * stride_xn + p_idx[None, :] * stride_xd
        go_val = tl.load(go_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        x_val = tl.load(x_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        acc += tl.sum(go_val.to(tl.float32) * x_val.to(tl.float32), axis=0)

    gw_ptrs = GW + pid_m * stride_gwm + pid_p * stride_gwp + offs_d * stride_gwd
    tl.store(gw_ptrs, acc.to(GW.dtype.element_ty), mask=mask_d)


class _TritonFusedPermProjFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        perms: torch.Tensor,
        inv_perms: torch.Tensor,
        biases: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1]).contiguous()
        N, D = x_flat.shape
        M, P, Dw = w.shape
        assert D == Dw, f"Dimension mismatch: x has dim {D}, w has dim {Dw}"

        w_c = w.contiguous()
        perms_c = perms.to(dtype=torch.long).contiguous()
        inv_perms_c = inv_perms.to(dtype=torch.long).contiguous()
        has_bias = biases is not None
        biases_c = biases.contiguous() if has_bias else torch.empty(0, device=x.device, dtype=x.dtype)

        out = torch.empty((M, N, D), device=x.device, dtype=x.dtype)

        BN, BD = min(64, triton.next_power_of_2(N)), min(64, triton.next_power_of_2(D))
        grid = (triton.cdiv(N, BN), triton.cdiv(D, BD), M)

        _fused_perm_proj_fwd_kernel[grid](
            x_flat, w_c, perms_c, biases_c, out,
            x_flat.stride(0), x_flat.stride(1),
            w_c.stride(0), w_c.stride(1), w_c.stride(2),
            perms_c.stride(0), perms_c.stride(1),
            biases_c.stride(0) if has_bias else 0, biases_c.stride(1) if has_bias else 0,
            out.stride(0), out.stride(1), out.stride(2),
            N, D, P, M,
            HAS_BIAS=has_bias,
            BLOCK_N=BN, BLOCK_D=BD,
        )

        ctx.save_for_backward(x_flat, w_c, perms_c, inv_perms_c, biases_c if has_bias else None)
        ctx.has_bias = has_bias
        ctx.orig_shape = orig_shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_flat, w_c, perms_c, inv_perms_c, biases_c = ctx.saved_tensors
        has_bias = ctx.has_bias
        go_flat = grad_out.reshape(w_c.shape[0], -1, w_c.shape[2]).contiguous()
        M, N, D = go_flat.shape
        P = perms_c.shape[0]

        gx = torch.empty_like(x_flat)
        BN, BD = min(64, triton.next_power_of_2(N)), min(64, triton.next_power_of_2(D))
        grid_gx = (triton.cdiv(N, BN), triton.cdiv(D, BD))

        _fused_perm_proj_bwd_gx_kernel[grid_gx](
            go_flat, w_c, inv_perms_c, gx,
            go_flat.stride(0), go_flat.stride(1), go_flat.stride(2),
            w_c.stride(0), w_c.stride(1), w_c.stride(2),
            inv_perms_c.stride(0), inv_perms_c.stride(1),
            gx.stride(0), gx.stride(1),
            N, D, P, M,
            BLOCK_N=BN, BLOCK_D=BD,
        )
        gx = gx.reshape(ctx.orig_shape)

        gw = torch.empty_like(w_c)
        grid_gw = (M, P, triton.cdiv(D, BD))
        _fused_perm_proj_bwd_gw_kernel[grid_gw](
            go_flat, x_flat, perms_c, gw,
            go_flat.stride(0), go_flat.stride(1), go_flat.stride(2),
            x_flat.stride(0), x_flat.stride(1),
            perms_c.stride(0), perms_c.stride(1),
            gw.stride(0), gw.stride(1), gw.stride(2),
            N, D,
            BLOCK_N=BN, BLOCK_D=BD,
        )

        gb = go_flat.sum(dim=1) if has_bias else None

        return gx, gw, None, None, gb


def triton_fused_perm_proj(
    x: torch.Tensor,
    w: torch.Tensor,
    perms: torch.Tensor,
    inv_perms: torch.Tensor,
    biases: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    High-Performance Fused Permutation Projection on CUDA via Triton.
    Args:
        x: Input tensor (*, D)
        w: Weight tensor (M, P, D)
        perms: Permutation table (P, D)
        inv_perms: Inverse permutation table (P, D)
        biases: Optional bias tensor (M, D)
    Returns:
        Tensor of shape (M, *, D)
    """
    return _TritonFusedPermProjFunc.apply(x, w, perms, inv_perms, biases)
