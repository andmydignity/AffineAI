"""
Custom Triton Kernel: In-SRAM Fused Permutation Projection
==========================================================
Fuses multi-permutation index gathers, ternary/float weight accumulation,
and bias addition directly in GPU SRAM registers.
Computes:
  out[m, n, d] = sum_{p=0}^{P-1} (x[n, perms[p, d]] * w[m, p, d]) + bias[m, d]

Eliminates all high-bandwidth intermediate VRAM gathers [N, P, D] and provides
analytical transposed backward kernels directly in registers.
Coalesced Y store via tl.make_block_ptr with boundary_check; W loads vectorized
contiguous in D; gather X via perm remains manual (random access).
"""

from typing import Optional
import torch
try:
    import triton
    import triton.language as tl
except Exception:  # CPU-only
    triton = None  # type: ignore
    tl = None  # type: ignore


_TURING_CACHE: Optional[bool] = None


def _is_turing() -> bool:
    global _TURING_CACHE
    if _TURING_CACHE is not None:
        return _TURING_CACHE
    try:
        from affine_ai.kernels import _IS_TURING as _T  # type: ignore

        _TURING_CACHE = bool(_T)
        return _TURING_CACHE
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            _TURING_CACHE = (7, 5) <= tuple(cap) < (8, 0)
            return _TURING_CACHE
    except Exception:
        pass
    _TURING_CACHE = False
    return False


if triton is not None:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 32, "BLOCK_D": 32}, num_warps=2),
            triton.Config({"BLOCK_N": 32, "BLOCK_D": 64}, num_warps=4),
            triton.Config({"BLOCK_N": 64, "BLOCK_D": 32}, num_warps=4),
            triton.Config({"BLOCK_N": 64, "BLOCK_D": 64}, num_warps=4),
            triton.Config({"BLOCK_N": 16, "BLOCK_D": 64}, num_warps=2),
            triton.Config({"BLOCK_N": 64, "BLOCK_D": 128}, num_warps=8),
        ],
        key=["N", "D"],
    )
    @triton.jit
    def _fused_perm_proj_fwd_kernel(
        X, W, Perms, Biases, Out,
        stride_xn, stride_xd,
        stride_wm, stride_wp, stride_wd,
        stride_pp, stride_pd,
        stride_bm, stride_bd,
        stride_om, stride_on, stride_od,
        N, D: tl.constexpr, P: tl.constexpr,
        M: tl.constexpr,
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

        for p in range(P):
            p_idx = tl.load(Perms + p * stride_pp + offs_d * stride_pd, mask=mask_d, other=0)
            x_ptrs = X + offs_n[:, None] * stride_xn + p_idx[None, :] * stride_xd
            x_val = tl.load(x_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            w = tl.load(W + pid_m * stride_wm + p * stride_wp + offs_d * stride_wd, mask=mask_d, other=0.0).to(tl.float32)
            acc = acc + x_val * w[None, :]

        if HAS_BIAS:
            bias_val = tl.load(Biases + pid_m * stride_bm + offs_d * stride_bd, mask=mask_d, other=0.0).to(tl.float32)
            acc = acc + bias_val[None, :]

        Out_block = tl.make_block_ptr(
            base=Out + pid_m * stride_om,
            shape=(N, D),
            strides=(stride_on, stride_od),
            offsets=(pid_n * BLOCK_N, pid_d * BLOCK_D),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0)
        )
        tl.store(Out_block, acc.to(Out.dtype.element_ty), boundary_check=(0, 1))
else:
    _fused_perm_proj_fwd_kernel = None  # type: ignore


if triton is not None:
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
        go_base_n = offs_n[:, None] * stride_gon

        for p in range(P):
            ip_idx = tl.load(InvPerms + p * stride_ipp + offs_d * stride_ipd, mask=mask_d, other=0)
            for m in range(M):
                go_ptrs = GradOut + m * stride_gom + go_base_n + ip_idx[None, :] * stride_god
                w_val = tl.load(W + m * stride_wm + p * stride_wp + ip_idx * stride_wd, mask=mask_d, other=0.0)
                go_val = tl.load(go_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
                acc += go_val.to(tl.float32) * w_val[None, :].to(tl.float32)

        gx_ptrs = GX + offs_n[:, None] * stride_gxn + offs_d[None, :] * stride_gxd
        tl.store(gx_ptrs, acc.to(GX.dtype.element_ty), mask=mask_n[:, None] & mask_d[None, :])


    @triton.jit
    def _fused_perm_proj_bwd_gw_kernel(
        GradOut, X, Perms, GW,
        stride_gom, stride_gon, stride_god,
        stride_xn, stride_xd,
        stride_pp, stride_pd,
        stride_gwm, stride_gwp, stride_gwd,
        stride_gws,
        N, D, P: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_mp = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_k = tl.program_id(2)

        pid_m = pid_mp // P
        pid_p = pid_mp % P

        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        p_idx = tl.load(Perms + pid_p * stride_pp + offs_d * stride_pd, mask=mask_d, other=0)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        n_per_split = tl.cdiv(N, NUM_SPLITS)
        n_start_split = pid_k * n_per_split
        n_end_split = tl.minimum(n_start_split + n_per_split, N)

        for n_start in range(n_start_split, n_end_split, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_end_split
            go_ptrs = GradOut + pid_m * stride_gom + offs_n[:, None] * stride_gon + offs_d[None, :] * stride_god
            x_ptrs = X + offs_n[:, None] * stride_xn + p_idx[None, :] * stride_xd
            go_val = tl.load(go_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            x_val = tl.load(x_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            acc += tl.sum(go_val.to(tl.float32) * x_val.to(tl.float32), axis=0)

        gw_ptrs = GW + pid_k * stride_gws + pid_m * stride_gwm + pid_p * stride_gwp + offs_d * stride_gwd
        tl.store(gw_ptrs, acc.to(GW.dtype.element_ty), mask=mask_d)
else:
    _fused_perm_proj_bwd_gx_kernel = None  # type: ignore
    _fused_perm_proj_bwd_gw_kernel = None  # type: ignore


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
        if D != Dw:
            raise ValueError(f"Dimension mismatch: x has dim {D}, w has dim {Dw}")
        def _check_perms(t: torch.Tensor, name: str):
            if getattr(t, "_perm_validated_dim", None) == D:
                return
            if not t.is_cuda:
                if (t >= D).any() or (t < 0).any():
                    raise ValueError(f"{name} OOB: D={D}")
            elif not torch.cuda.is_current_stream_capturing():
                if ((t < 0) | (t >= D)).any().item():
                    raise ValueError(f"{name} OOB on CUDA: D={D}")
            try:
                t._perm_validated_dim = D
            except Exception:
                pass

        _check_perms(perms, "perms")
        _check_perms(inv_perms, "inv_perms")

        w_c = w.contiguous()
        if perms.dtype == torch.int32 and perms.is_contiguous():
            perms_c = perms
        else:
            perms_c = perms.to(dtype=torch.int32).contiguous()
        if inv_perms.dtype == torch.int32 and inv_perms.is_contiguous():
            inv_perms_c = inv_perms
        else:
            inv_perms_c = inv_perms.to(dtype=torch.int32).contiguous()
        has_bias = biases is not None
        biases_c = biases.contiguous() if has_bias else torch.empty(0, device=x.device, dtype=x.dtype)

        if triton is None or not x.is_cuda or _fused_perm_proj_fwd_kernel is None:
            x_g = torch.gather(
                x_flat.unsqueeze(1).expand(-1, P, -1), dim=-1,
                index=perms_c.to(torch.long).unsqueeze(0).expand(N, -1, -1)
            )
            out = (x_g.unsqueeze(0) * w_c.unsqueeze(1)).sum(dim=2)
            if has_bias:
                out = out + biases_c.unsqueeze(1)
            ctx.save_for_backward(x_flat, w_c, perms_c, inv_perms_c)
            ctx.has_bias = has_bias
            ctx.biases_c = biases_c if has_bias else None
            ctx.orig_shape = orig_shape
            ctx._used_triton = False
            return out

        out = torch.empty((M, N, D), device=x.device, dtype=x.dtype)

        grid = lambda META: (
            triton.cdiv(N, META["BLOCK_N"]),
            triton.cdiv(D, META["BLOCK_D"]),
            M,
        )  # noqa: E731
        _fused_perm_proj_fwd_kernel[grid](
            x_flat, w_c, perms_c, biases_c, out,
            x_flat.stride(0), x_flat.stride(1),
            w_c.stride(0), w_c.stride(1), w_c.stride(2),
            perms_c.stride(0), perms_c.stride(1),
            biases_c.stride(0) if has_bias else 0, biases_c.stride(1) if has_bias else 0,
            out.stride(0), out.stride(1), out.stride(2),
            N, D, P, M,
            HAS_BIAS=has_bias,
        )

        ctx.save_for_backward(x_flat, w_c, perms_c, inv_perms_c)
        ctx.has_bias = has_bias
        ctx.biases_c = biases_c if has_bias else None
        ctx.orig_shape = orig_shape
        ctx._used_triton = True
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_flat, w_c, perms_c, inv_perms_c = ctx.saved_tensors
        _biases_c = ctx.biases_c  # noqa: F841
        has_bias = ctx.has_bias
        go_flat = grad_out.reshape(w_c.shape[0], -1, w_c.shape[2]).contiguous()
        M, N, D = go_flat.shape
        P = perms_c.shape[0]

        if not getattr(ctx, "_used_triton", True) or triton is None or _fused_perm_proj_bwd_gx_kernel is None:
            x_g = torch.gather(
                x_flat.unsqueeze(1).expand(-1, P, -1), dim=-1,
                index=perms_c.to(torch.long).unsqueeze(0).expand(N, -1, -1)
            )
            gw = (go_flat.unsqueeze(2) * x_g.unsqueeze(0)).sum(dim=1)
            gx = torch.zeros_like(x_flat)
            inv_long = inv_perms_c.to(torch.long)
            for p in range(P):
                gx += (go_flat * w_c[:, p, :].unsqueeze(1)).sum(dim=0).gather(1, inv_long[p].unsqueeze(0).expand(N, -1))
            gx = gx.reshape(ctx.orig_shape)
            gb = go_flat.sum(dim=1) if has_bias else None
            return gx, gw, None, None, gb

        gx = torch.empty_like(x_flat)
        _bd_cap2 = 64 if _is_turing() else 128
        _bn_cap2 = 64 if _is_turing() else 128
        if N == 1:
            BN, BD = 16, max(16, min(_bd_cap2, triton.next_power_of_2(D)))
        else:
            BN, BD = max(16, min(_bn_cap2, triton.next_power_of_2(N))), max(16, min(_bd_cap2, triton.next_power_of_2(D)))
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

        num_splits = 1 if N <= 64 else min(16, triton.cdiv(N, 64))
        if num_splits > 1:
            gw_splits = torch.empty((num_splits, M, P, D), device=x_flat.device, dtype=torch.float32)
            grid_gw = (M * P, triton.cdiv(D, BD), num_splits)
            _fused_perm_proj_bwd_gw_kernel[grid_gw](
                go_flat, x_flat, perms_c, gw_splits,
                go_flat.stride(0), go_flat.stride(1), go_flat.stride(2),
                x_flat.stride(0), x_flat.stride(1),
                perms_c.stride(0), perms_c.stride(1),
                gw_splits.stride(1), gw_splits.stride(2), gw_splits.stride(3),
                gw_splits.stride(0),
                N, D, P=P,
                NUM_SPLITS=num_splits,
                BLOCK_N=BN, BLOCK_D=BD,
            )
            gw = gw_splits.sum(dim=0).to(w_c.dtype)
        else:
            gw = torch.empty_like(w_c)
            grid_gw = (M * P, triton.cdiv(D, BD), 1)
            _fused_perm_proj_bwd_gw_kernel[grid_gw](
                go_flat, x_flat, perms_c, gw,
                go_flat.stride(0), go_flat.stride(1), go_flat.stride(2),
                x_flat.stride(0), x_flat.stride(1),
                perms_c.stride(0), perms_c.stride(1),
                gw.stride(0), gw.stride(1), gw.stride(2),
                0,
                N, D, P=P,
                NUM_SPLITS=1,
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
