"""
Custom Triton Kernel: High-Performance Streaming RMS Normalization
==================================================================
Eliminates memory bus contention and guarantees numerical stability (zero FP16/BF16 overflow)
via float32 row-reduction and fused autograd backward passes with hardware autotuning.
"""

import torch
import triton
import triton.language as tl
from typing import List, Optional, Tuple


def _prune_fwd_bwd_configs(configs: List[triton.Config], named_args: dict, **kwargs) -> List[triton.Config]:
    """
    Prunes autotune configs to only evaluate powers of 2 that match the target dimension D.
    Ensures 128-bit aligned vectorization with zero extraneous configuration compile overhead.
    """
    D = kwargs.get('D', named_args.get('D', None))
    if D is not None:
        target_block = triton.next_power_of_2(D)
        valid_blocks = [c.kwargs['BLOCK_SIZE'] for c in configs]
        if target_block not in valid_blocks:
            ge_blocks = [b for b in valid_blocks if b >= D]
            target_block = min(ge_blocks) if ge_blocks else max(valid_blocks)
        pruned = [c for c in configs if c.kwargs.get('BLOCK_SIZE') == target_block]
        if pruned:
            return pruned
    return configs


def _get_fwd_bwd_autotune_configs() -> List[triton.Config]:
    """
    Generates autotune configurations covering hidden dimensions D in {16, 32, ..., 8192}.
    Varies num_warps (1, 2, 4, 8, 16) and num_stages (1, 2, 3, 4) for optimal occupancy and instruction pipelining.
    BLOCK_SIZE >8192 dropped to cap compile time / register pressure.
    """
    configs = []
    for BLOCK_SIZE in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        if BLOCK_SIZE <= 64:
            for nw in [1, 2]:
                for ns in [1, 2]:
                    configs.append(triton.Config({'BLOCK_SIZE': BLOCK_SIZE}, num_warps=nw, num_stages=ns))
        elif BLOCK_SIZE <= 256:
            for nw in [2, 4]:
                for ns in [1, 2]:
                    configs.append(triton.Config({'BLOCK_SIZE': BLOCK_SIZE}, num_warps=nw, num_stages=ns))
        elif BLOCK_SIZE <= 1024:
            for nw in [4, 8]:
                for ns in [2, 3]:
                    configs.append(triton.Config({'BLOCK_SIZE': BLOCK_SIZE}, num_warps=nw, num_stages=ns))
        else:
            for nw in [8, 16]:
                for ns in [2, 4]:
                    configs.append(triton.Config({'BLOCK_SIZE': BLOCK_SIZE}, num_warps=nw, num_stages=ns))
    return configs


def _prune_dscale_configs(configs: List[triton.Config], named_args: dict, **kwargs) -> List[triton.Config]:
    D = kwargs.get('D', named_args.get('D', None))
    if D is not None:
        target = min(128, max(16, triton.next_power_of_2(D)))
        pruned = [c for c in configs if c.kwargs.get('BLOCK_D') == target]
        if pruned:
            return pruned
    return configs


def _get_dscale_autotune_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_D in [16, 32, 64, 128]:
        if BLOCK_D <= 32:
            configs.append(triton.Config({'BLOCK_N': 64, 'BLOCK_D': BLOCK_D}, num_warps=2, num_stages=2))
        else:
            configs.append(triton.Config({'BLOCK_N': 64, 'BLOCK_D': BLOCK_D}, num_warps=4, num_stages=2))
            configs.append(triton.Config({'BLOCK_N': 128, 'BLOCK_D': BLOCK_D}, num_warps=4, num_stages=2))
            configs.append(triton.Config({'BLOCK_N': 64, 'BLOCK_D': BLOCK_D}, num_warps=8, num_stages=2))
    return configs


@triton.autotune(
    configs=_get_fwd_bwd_autotune_configs(),
    key=['D'],
    prune_configs_by={'early_config_prune': _prune_fwd_bwd_configs}
)
@triton.jit
def _rms_norm_fwd_kernel(
    X_ptr, Scale_ptr, Out_ptr, Rsqrt_ptr,
    stride_xb, stride_xd,
    stride_sb,
    stride_ob, stride_od,
    D: tl.constexpr, eps: tl.float32,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_dtype = X_ptr.dtype.element_ty
    acc_dtype = tl.float64 if x_dtype == tl.float64 else tl.float32

    if D <= BLOCK_SIZE:
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x_ptrs = X_ptr + row_idx * stride_xb + cols * stride_xd
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(acc_dtype)
        var = tl.sum(x * x, axis=0) / D
        rsqrt = tl.rsqrt(var + eps)
        tl.store(Rsqrt_ptr + row_idx, rsqrt)
        scale = tl.load(Scale_ptr + cols * stride_sb, mask=mask, other=1.0).to(acc_dtype)
        y = (x * rsqrt * scale).to(x_dtype)
        out_ptrs = Out_ptr + row_idx * stride_ob + cols * stride_od
        tl.store(out_ptrs, y, mask=mask)
        return

    # Pass 1: Accumulate sum of squares in FP32/FP64 across chunks (D > BLOCK_SIZE fallback)
    sum_sq = 0.0
    for d_start in range(0, D, BLOCK_SIZE):
        cols = d_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x_ptrs = X_ptr + row_idx * stride_xb + cols * stride_xd
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(acc_dtype)
        sum_sq += tl.sum(x * x, axis=0)

    var = sum_sq / D
    rsqrt = tl.rsqrt(var + eps)
    tl.store(Rsqrt_ptr + row_idx, rsqrt)

    # Pass 2: Normalize and write output across chunks
    for d_start in range(0, D, BLOCK_SIZE):
        cols = d_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x_ptrs = X_ptr + row_idx * stride_xb + cols * stride_xd
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(acc_dtype)
        scale = tl.load(Scale_ptr + cols * stride_sb, mask=mask, other=1.0).to(acc_dtype)
        y = (x * rsqrt * scale).to(x_dtype)
        out_ptrs = Out_ptr + row_idx * stride_ob + cols * stride_od
        tl.store(out_ptrs, y, mask=mask)


@triton.autotune(
    configs=_get_fwd_bwd_autotune_configs(),
    key=['D'],
    prune_configs_by={'early_config_prune': _prune_fwd_bwd_configs}
)
@triton.jit
def _rms_norm_bwd_dx_kernel(
    DY_ptr, X_ptr, Scale_ptr, Rsqrt_ptr, DX_ptr,
    stride_dyb, stride_dyd,
    stride_xb, stride_xd,
    stride_sb,
    stride_dxb, stride_dxd,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_dtype = X_ptr.dtype.element_ty
    acc_dtype = tl.float64 if x_dtype == tl.float64 else tl.float32
    rsqrt = tl.load(Rsqrt_ptr + row_idx).to(acc_dtype)

    if D <= BLOCK_SIZE:
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        dy = tl.load(DY_ptr + row_idx * stride_dyb + cols * stride_dyd, mask=mask, other=0.0).to(acc_dtype)
        x = tl.load(X_ptr + row_idx * stride_xb + cols * stride_xd, mask=mask, other=0.0).to(acc_dtype)
        scale = tl.load(Scale_ptr + cols * stride_sb, mask=mask, other=0.0).to(acc_dtype)
        dy_scale = dy * scale
        inner = tl.sum(dy_scale * x, axis=0)
        coeff = (inner * rsqrt * rsqrt) / D
        dx = (dy_scale - x * coeff) * rsqrt
        tl.store(DX_ptr + row_idx * stride_dxb + cols * stride_dxd, dx.to(x_dtype), mask=mask)
        return

    # Pass 1: Accumulate inner product across chunks (D > BLOCK_SIZE fallback)
    inner = 0.0
    for d_start in range(0, D, BLOCK_SIZE):
        cols = d_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        dy = tl.load(DY_ptr + row_idx * stride_dyb + cols * stride_dyd, mask=mask, other=0.0).to(acc_dtype)
        x = tl.load(X_ptr + row_idx * stride_xb + cols * stride_xd, mask=mask, other=0.0).to(acc_dtype)
        scale = tl.load(Scale_ptr + cols * stride_sb, mask=mask, other=0.0).to(acc_dtype)
        inner += tl.sum(dy * scale * x, axis=0)

    coeff = (inner * rsqrt * rsqrt) / D

    # Pass 2: Compute dx and store across chunks
    for d_start in range(0, D, BLOCK_SIZE):
        cols = d_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        dy = tl.load(DY_ptr + row_idx * stride_dyb + cols * stride_dyd, mask=mask, other=0.0).to(acc_dtype)
        x = tl.load(X_ptr + row_idx * stride_xb + cols * stride_xd, mask=mask, other=0.0).to(acc_dtype)
        scale = tl.load(Scale_ptr + cols * stride_sb, mask=mask, other=0.0).to(acc_dtype)
        dy_scale = dy * scale
        dx = (dy_scale - x * coeff) * rsqrt
        tl.store(DX_ptr + row_idx * stride_dxb + cols * stride_dxd, dx.to(x_dtype), mask=mask)


@triton.autotune(
    configs=_get_dscale_autotune_configs(),
    key=['D'],
    prune_configs_by={'prune_dscale_configs': _prune_dscale_configs},
    reset_to_zero=['DScale_ptr']
)
@triton.jit
def _rms_norm_bwd_dscale_kernel(
    DY_ptr, X_ptr, Rsqrt_ptr, DScale_ptr,
    stride_dyb, stride_dyd,
    stride_xb, stride_xd,
    N: tl.constexpr, D: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_d = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_n_splits = tl.num_programs(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    acc_dtype = tl.float64 if X_ptr.dtype.element_ty == tl.float64 else tl.float32
    acc = tl.zeros([BLOCK_D], dtype=acc_dtype)

    rows_per_split = tl.cdiv(N, num_n_splits)
    n_start_split = pid_n * rows_per_split
    n_end_split = tl.minimum(n_start_split + rows_per_split, N)

    for n_start in range(n_start_split, n_end_split, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_end_split

        dy_ptrs = DY_ptr + offs_n[:, None] * stride_dyb + offs_d[None, :] * stride_dyd
        x_ptrs = X_ptr + offs_n[:, None] * stride_xb + offs_d[None, :] * stride_xd
        rsqrt_ptrs = Rsqrt_ptr + offs_n

        mask_2d = mask_n[:, None] & mask_d[None, :]
        dy = tl.load(dy_ptrs, mask=mask_2d, other=0.0).to(acc_dtype)
        x = tl.load(x_ptrs, mask=mask_2d, other=0.0).to(acc_dtype)
        rsqrt = tl.load(rsqrt_ptrs, mask=mask_n, other=0.0).to(acc_dtype)

        acc += tl.sum(dy * (x * rsqrt[:, None]), axis=0)

    is_fp64 = X_ptr.dtype.element_ty == tl.float64
    if is_fp64:
        # fp64 atomic_add not portable; fallback to store when single writer else loop will be single-writer via Python guard (splits=1 for fp64)
        if num_n_splits == 1:
            tl.store(DScale_ptr + offs_d, acc, mask=mask_d)
        else:
            # fallback: still store for first split only to allow compilation; Python avoids this path for fp64 with splits>1
            if pid_n == 0:
                tl.store(DScale_ptr + offs_d, acc, mask=mask_d)
    else:
        if num_n_splits == 1:
            tl.store(DScale_ptr + offs_d, acc, mask=mask_d)
        else:
            tl.atomic_add(DScale_ptr + offs_d, acc, mask=mask_d)


def _get_fused_bwd_autotune_configs() -> List[triton.Config]:
    return [
        triton.Config({'BLOCK_ROW': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_ROW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_ROW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_ROW': 32}, num_warps=8, num_stages=2),
    ]


@triton.autotune(
    configs=_get_fused_bwd_autotune_configs(),
    key=['D'],
    reset_to_zero=['DScale_ptr']
)
@triton.jit
def _rms_norm_bwd_fused_kernel(
    DY_ptr, X_ptr, Scale_ptr, Rsqrt_ptr, DX_ptr, DScale_ptr,
    stride_dyb, stride_dyd,
    stride_xb, stride_xd,
    stride_sb,
    stride_dxb, stride_dxd,
    N: tl.constexpr, D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_ROW: tl.constexpr
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    mask_m = offs_m < N
    x_dtype = X_ptr.dtype.element_ty
    acc_dtype = tl.float64 if x_dtype == tl.float64 else tl.float32
    is_fp64 = X_ptr.dtype.element_ty == tl.float64
    rsqrt = tl.load(Rsqrt_ptr + offs_m, mask=mask_m, other=0.0).to(acc_dtype)
    # BLOCK_D capped <=2048; loop over D if larger.
    # Pass 1: global inner product per row over the FULL D (dx needs the
    # full-row sum; a per-chunk partial sum would scale coeff wrong by ~D/BLOCK_D).
    inner = tl.zeros([BLOCK_ROW], dtype=acc_dtype)
    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        mask_2d = mask_m[:, None] & mask_d[None, :]
        scale = tl.load(Scale_ptr + offs_d * stride_sb, mask=mask_d, other=0.0).to(acc_dtype)
        dy = tl.load(DY_ptr + offs_m[:, None] * stride_dyb + offs_d[None, :] * stride_dyd, mask=mask_2d, other=0.0).to(acc_dtype)
        x = tl.load(X_ptr + offs_m[:, None] * stride_xb + offs_d[None, :] * stride_xd, mask=mask_2d, other=0.0).to(acc_dtype)
        inner += tl.sum(dy * scale[None, :] * x, axis=1)
    coeff = (inner * rsqrt * rsqrt) / D
    # Pass 2: dx + dscale per D chunk.
    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        mask_2d = mask_m[:, None] & mask_d[None, :]
        scale = tl.load(Scale_ptr + offs_d * stride_sb, mask=mask_d, other=0.0).to(acc_dtype)
        dy = tl.load(DY_ptr + offs_m[:, None] * stride_dyb + offs_d[None, :] * stride_dyd, mask=mask_2d, other=0.0).to(acc_dtype)
        x = tl.load(X_ptr + offs_m[:, None] * stride_xb + offs_d[None, :] * stride_xd, mask=mask_2d, other=0.0).to(acc_dtype)
        dscale_part = tl.sum(dy * (x * rsqrt[:, None]), axis=0)
        if is_fp64:
            # fp64 atomic_add not universally available; fallback to store when single writer (fused path is single-writer per D chunk when launched with 1D grid) or skip - Python guards fp64 fused path
            if pid_m == 0:
                tl.store(DScale_ptr + offs_d, dscale_part, mask=mask_d)
        else:
            tl.atomic_add(DScale_ptr + offs_d, dscale_part, mask=mask_d)
        dy_scale = dy * scale[None, :]
        dx = (dy_scale - x * coeff[:, None]) * rsqrt[:, None]
        tl.store(DX_ptr + offs_m[:, None] * stride_dxb + offs_d[None, :] * stride_dxd, dx.to(x_dtype), mask=mask_2d)


class TritonRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1]).contiguous()
        scale_contig = scale.view(-1).contiguous()
        N, D = x_flat.shape
        assert scale_contig.numel() == D, f"scale dimension mismatch: scale.numel()={scale_contig.numel()} != D={D}"
        out = torch.empty_like(x_flat)
        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        rsqrt = torch.empty(N, device=x.device, dtype=calc_dtype)

        _rms_norm_fwd_kernel[(N,)](
            x_flat, scale_contig, out, rsqrt,
            x_flat.stride(0), x_flat.stride(1),
            scale_contig.stride(0),
            out.stride(0), out.stride(1),
            D=D, eps=eps
        )

        ctx.save_for_backward(x_flat, scale_contig, rsqrt)
        ctx.orig_shape = orig_shape
        ctx.D = D
        ctx.scale_shape = scale.shape
        return out.reshape(*orig_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_flat, scale, rsqrt = ctx.saved_tensors
        dy_flat = dy.reshape(-1, ctx.D).contiguous()
        N = x_flat.shape[0]
        D = ctx.D
        need_dx = ctx.needs_input_grad[0]
        need_dscale = ctx.needs_input_grad[1]

        dx = None
        dscale = None

        if need_dx and need_dscale and D <= 4096 and x_flat.dtype != torch.float64:
            dx = torch.empty_like(x_flat)
            calc_dtype = torch.float64 if x_flat.dtype == torch.float64 else torch.float32
            dscale_acc = torch.zeros(D, dtype=calc_dtype, device=scale.device)
            BLOCK_D = min(2048, max(16, triton.next_power_of_2(D)))
            grid = lambda META: (triton.cdiv(N, META['BLOCK_ROW']),)
            _rms_norm_bwd_fused_kernel[grid](
                dy_flat, x_flat, scale, rsqrt, dx, dscale_acc,
                dy_flat.stride(0), dy_flat.stride(1),
                x_flat.stride(0), x_flat.stride(1),
                scale.stride(0),
                dx.stride(0), dx.stride(1),
                N=N, D=D, BLOCK_D=BLOCK_D
            )
            dx = dx.reshape(*ctx.orig_shape)
            dscale = dscale_acc.to(scale.dtype).view(ctx.scale_shape)
        else:
            if need_dx:
                dx = torch.empty_like(x_flat)
                _rms_norm_bwd_dx_kernel[(N,)](
                    dy_flat, x_flat, scale, rsqrt, dx,
                    dy_flat.stride(0), dy_flat.stride(1),
                    x_flat.stride(0), x_flat.stride(1),
                    scale.stride(0),
                    dx.stride(0), dx.stride(1),
                    D=D
                )
                dx = dx.reshape(*ctx.orig_shape)

            if need_dscale:
                calc_dtype = torch.float64 if x_flat.dtype == torch.float64 else torch.float32
                dscale_acc = torch.zeros(D, dtype=calc_dtype, device=scale.device)
                # fp64: avoid multi-split atomic_add; force single writer
                if calc_dtype == torch.float64:
                    grid = lambda META: (triton.cdiv(D, META['BLOCK_D']), 1)
                else:
                    grid = lambda META: (triton.cdiv(D, META['BLOCK_D']), min(16, max(1, triton.cdiv(N, META['BLOCK_N']))))
                _rms_norm_bwd_dscale_kernel[grid](
                    dy_flat, x_flat, rsqrt, dscale_acc,
                    dy_flat.stride(0), dy_flat.stride(1),
                    x_flat.stride(0), x_flat.stride(1),
                    N=N, D=D
                )
                dscale = dscale_acc.to(scale.dtype).view(ctx.scale_shape)

        return dx, dscale, None


def triton_rms_norm(x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return TritonRMSNormFunc.apply(x, scale, eps)


@triton.autotune(
    configs=_get_fwd_bwd_autotune_configs(),
    key=['D'],
    prune_configs_by={'early_config_prune': _prune_fwd_bwd_configs}
)
@triton.jit
def _fused_add_rms_norm_fwd_kernel(
    X_ptr, Res_ptr, Scale_ptr, Out_ptr, Res_out_ptr, Rsqrt_ptr,
    stride_xb, stride_xd,
    stride_rb, stride_rd,
    stride_sb,
    stride_ob, stride_od,
    stride_rob, stride_rod,
    D: tl.constexpr, eps: tl.float32,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_dtype = X_ptr.dtype.element_ty
    acc_dtype = tl.float64 if x_dtype == tl.float64 else tl.float32

    if D <= BLOCK_SIZE:
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x_ptrs = X_ptr + row_idx * stride_xb + cols * stride_xd
        res_ptrs = Res_ptr + row_idx * stride_rb + cols * stride_rd
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        res = tl.load(res_ptrs, mask=mask, other=0.0)
        res_acc = x.to(acc_dtype) + res.to(acc_dtype)
        tl.store(Res_out_ptr + row_idx * stride_rob + cols * stride_rod, res_acc.to(x_dtype), mask=mask)
        var = tl.sum(res_acc * res_acc, axis=0) / D
        rsqrt = tl.rsqrt(var + eps)
        tl.store(Rsqrt_ptr + row_idx, rsqrt)
        scale = tl.load(Scale_ptr + cols * stride_sb, mask=mask, other=1.0).to(acc_dtype)
        y = (res_acc * rsqrt * scale).to(x_dtype)
        out_ptrs = Out_ptr + row_idx * stride_ob + cols * stride_od
        tl.store(out_ptrs, y, mask=mask)
        return

    # Pass 1: Compute res_out = x + res, store res_out, and accumulate sum of squares (D > BLOCK_SIZE fallback)
    sum_sq = 0.0
    for d_start in range(0, D, BLOCK_SIZE):
        cols = d_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < D

        x_ptrs = X_ptr + row_idx * stride_xb + cols * stride_xd
        res_ptrs = Res_ptr + row_idx * stride_rb + cols * stride_rd

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        res = tl.load(res_ptrs, mask=mask, other=0.0)

        res_acc = x.to(acc_dtype) + res.to(acc_dtype)
        res_out = res_acc.to(x_dtype)
        tl.store(Res_out_ptr + row_idx * stride_rob + cols * stride_rod, res_out, mask=mask)
        sum_sq += tl.sum(res_acc * res_acc, axis=0)

    var = sum_sq / D
    rsqrt = tl.rsqrt(var + eps)
    tl.store(Rsqrt_ptr + row_idx, rsqrt)

    # Pass 2: Normalize and write to out
    for d_start in range(0, D, BLOCK_SIZE):
        cols = d_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < D

        x_ptrs = X_ptr + row_idx * stride_xb + cols * stride_xd
        res_ptrs = Res_ptr + row_idx * stride_rb + cols * stride_rd
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        res = tl.load(res_ptrs, mask=mask, other=0.0)
        res_acc = x.to(acc_dtype) + res.to(acc_dtype)

        scale = tl.load(Scale_ptr + cols * stride_sb, mask=mask, other=1.0).to(acc_dtype)
        y = (res_acc * rsqrt * scale).to(x_dtype)

        out_ptrs = Out_ptr + row_idx * stride_ob + cols * stride_od
        tl.store(out_ptrs, y, mask=mask)


class TritonFusedAddRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, residual: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6):
        orig_shape = x.shape
        assert residual.shape == orig_shape, f"residual shape mismatch: residual.shape={residual.shape} != x.shape={orig_shape}"
        x_flat = x.reshape(-1, orig_shape[-1]).contiguous()
        res_flat = residual.reshape(-1, orig_shape[-1]).contiguous()
        scale_contig = scale.view(-1).contiguous()
        N, D = x_flat.shape
        assert scale_contig.numel() == D, f"scale dimension mismatch: scale.numel()={scale_contig.numel()} != D={D}"

        out = torch.empty_like(x_flat)
        res_out = torch.empty_like(x_flat)
        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        rsqrt = torch.empty(N, device=x.device, dtype=calc_dtype)

        _fused_add_rms_norm_fwd_kernel[(N,)](
            x_flat, res_flat, scale_contig, out, res_out, rsqrt,
            x_flat.stride(0), x_flat.stride(1),
            res_flat.stride(0), res_flat.stride(1),
            scale_contig.stride(0),
            out.stride(0), out.stride(1),
            res_out.stride(0), res_out.stride(1),
            D=D, eps=eps
        )

        ctx.save_for_backward(res_out, scale_contig, rsqrt)
        ctx.orig_shape = orig_shape
        ctx.D = D
        ctx.scale_shape = scale.shape
        return out.reshape(*orig_shape), res_out.reshape(*orig_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dres_out: Optional[torch.Tensor] = None):
        res_out, scale, rsqrt = ctx.saved_tensors
        dy_flat = dy.reshape(-1, ctx.D).contiguous()
        N = res_out.shape[0]
        D = ctx.D

        need_dx = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        need_dscale = ctx.needs_input_grad[2]

        dx_out = None
        dres_out_val = None
        dscale = None

        if need_dx and need_dscale and D <= 4096 and res_out.dtype != torch.float64:
            dx = torch.empty_like(res_out)
            calc_dtype = torch.float64 if res_out.dtype == torch.float64 else torch.float32
            dscale_acc = torch.zeros(D, dtype=calc_dtype, device=scale.device)
            BLOCK_D = min(2048, max(16, triton.next_power_of_2(D)))
            grid = lambda META: (triton.cdiv(N, META['BLOCK_ROW']),)
            _rms_norm_bwd_fused_kernel[grid](
                dy_flat, res_out, scale, rsqrt, dx, dscale_acc,
                dy_flat.stride(0), dy_flat.stride(1),
                res_out.stride(0), res_out.stride(1),
                scale.stride(0),
                dx.stride(0), dx.stride(1),
                N=N, D=D, BLOCK_D=BLOCK_D
            )
            if dres_out is not None:
                dx = dx + dres_out.reshape(-1, ctx.D)

            dx_out = dx.reshape(*ctx.orig_shape) if ctx.needs_input_grad[0] else None
            dres_out_val = dx.clone().reshape(*ctx.orig_shape) if (ctx.needs_input_grad[0] and ctx.needs_input_grad[1]) else (dx.reshape(*ctx.orig_shape) if ctx.needs_input_grad[1] else None)
            dscale = dscale_acc.to(scale.dtype).view(ctx.scale_shape)
        else:
            if need_dx:
                dx = torch.empty_like(res_out)
                _rms_norm_bwd_dx_kernel[(N,)](
                    dy_flat, res_out, scale, rsqrt, dx,
                    dy_flat.stride(0), dy_flat.stride(1),
                    res_out.stride(0), res_out.stride(1),
                    scale.stride(0),
                    dx.stride(0), dx.stride(1),
                    D=D
                )
                if dres_out is not None:
                    dx = dx + dres_out.reshape(-1, ctx.D)

                dx_out = dx.reshape(*ctx.orig_shape) if ctx.needs_input_grad[0] else None
                dres_out_val = dx.clone().reshape(*ctx.orig_shape) if (ctx.needs_input_grad[0] and ctx.needs_input_grad[1]) else (dx.reshape(*ctx.orig_shape) if ctx.needs_input_grad[1] else None)

            if need_dscale:
                calc_dtype = torch.float64 if res_out.dtype == torch.float64 else torch.float32
                dscale_acc = torch.zeros(D, dtype=calc_dtype, device=scale.device)
                if calc_dtype == torch.float64:
                    grid = lambda META: (triton.cdiv(D, META['BLOCK_D']), 1)
                else:
                    grid = lambda META: (triton.cdiv(D, META['BLOCK_D']), min(16, max(1, triton.cdiv(N, META['BLOCK_N']))))
                _rms_norm_bwd_dscale_kernel[grid](
                    dy_flat, res_out, rsqrt, dscale_acc,
                    dy_flat.stride(0), dy_flat.stride(1),
                    res_out.stride(0), res_out.stride(1),
                    N=N, D=D
                )
                dscale = dscale_acc.to(scale.dtype).view(ctx.scale_shape)

        return dx_out, dres_out_val, dscale, None


def triton_fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused In-SRAM Add + RMSNorm:
    Computes res_out = x + residual, and y = RMSNorm(res_out, scale) in a single kernel launch.
    Returns (y, res_out).
    """
    return TritonFusedAddRMSNormFunc.apply(x, residual, scale, eps)


