"""
Custom Triton Kernel: High-Performance Streaming RMS Normalization
==================================================================
Eliminates memory bus contention and guarantees numerical stability (zero FP16/BF16 overflow)
via float32 row-reduction and fused autograd backward passes with hardware autotuning.
"""

import torch
import triton
import triton.language as tl
from typing import List, Optional


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
    Generates autotune configurations covering hidden dimensions D in {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}.
    Varies num_warps (1, 2, 4, 8, 16) and num_stages (1, 2, 3, 4) for optimal occupancy and instruction pipelining.
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
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    x_ptrs = X_ptr + row_idx * stride_xb + cols * stride_xd
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Convert to float32/float64 before squaring to prevent FP16/BF16 overflow (max FP16 is 65,504)
    x_dtype = x.dtype
    acc_dtype = tl.float64 if x_dtype == tl.float64 else tl.float32
    x_acc = x.to(acc_dtype)

    var = tl.sum(x_acc * x_acc, axis=0) / D
    rsqrt = tl.rsqrt(var + eps)
    tl.store(Rsqrt_ptr + row_idx, rsqrt)

    scale = tl.load(Scale_ptr + cols * stride_sb, mask=mask, other=1.0).to(acc_dtype)
    y = (x_acc * rsqrt * scale).to(x_dtype)

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
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    dy = tl.load(DY_ptr + row_idx * stride_dyb + cols * stride_dyd, mask=mask, other=0.0)
    x = tl.load(X_ptr + row_idx * stride_xb + cols * stride_xd, mask=mask, other=0.0)
    scale = tl.load(Scale_ptr + cols * stride_sb, mask=mask, other=0.0)
    rsqrt = tl.load(Rsqrt_ptr + row_idx)

    # Cast to float32/float64 for intermediate gradient math to prevent precision loss and underflow
    acc_dtype = tl.float64 if x.dtype == tl.float64 else tl.float32
    dy_acc = dy.to(acc_dtype)
    x_acc = x.to(acc_dtype)
    scale_acc = scale.to(acc_dtype)
    rsqrt_acc = rsqrt.to(acc_dtype)

    dy_scale = dy_acc * scale_acc
    inner = tl.sum(dy_scale * x_acc, axis=0)
    coeff = (inner * rsqrt_acc * rsqrt_acc) / D
    dx = (dy_scale - x_acc * coeff) * rsqrt_acc

    tl.store(DX_ptr + row_idx * stride_dxb + cols * stride_dxd, dx.to(x.dtype), mask=mask)


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

    if num_n_splits == 1:
        tl.store(DScale_ptr + offs_d, acc, mask=mask_d)
    else:
        tl.atomic_add(DScale_ptr + offs_d, acc, mask=mask_d)


class TritonRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1]).contiguous()
        scale_contig = scale.view(-1).contiguous()
        N, D = x_flat.shape
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

        dx = None
        if ctx.needs_input_grad[0]:
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

        dscale = None
        if ctx.needs_input_grad[1]:
            calc_dtype = torch.float64 if x_flat.dtype == torch.float64 else torch.float32
            dscale_acc = torch.zeros(D, dtype=calc_dtype, device=scale.device)
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
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    x_ptrs = X_ptr + row_idx * stride_xb + cols * stride_xd
    res_ptrs = Res_ptr + row_idx * stride_rb + cols * stride_rd

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    res = tl.load(res_ptrs, mask=mask, other=0.0)

    x_dtype = x.dtype
    acc_dtype = tl.float64 if x_dtype == tl.float64 else tl.float32
    res_acc = x.to(acc_dtype) + res.to(acc_dtype)
    res_out = res_acc.to(x_dtype)
    tl.store(Res_out_ptr + row_idx * stride_rob + cols * stride_rod, res_out, mask=mask)

    var = tl.sum(res_acc * res_acc, axis=0) / D
    rsqrt = tl.rsqrt(var + eps)
    tl.store(Rsqrt_ptr + row_idx, rsqrt)

    scale = tl.load(Scale_ptr + cols * stride_sb, mask=mask, other=1.0).to(acc_dtype)
    y = (res_acc * rsqrt * scale).to(x_dtype)

    out_ptrs = Out_ptr + row_idx * stride_ob + cols * stride_od
    tl.store(out_ptrs, y, mask=mask)


class TritonFusedAddRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, residual: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1]).contiguous()
        res_flat = residual.reshape(-1, orig_shape[-1]).contiguous()
        scale_contig = scale.view(-1).contiguous()
        N, D = x_flat.shape

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

        dscale = None
        if ctx.needs_input_grad[2]:
            calc_dtype = torch.float64 if res_out.dtype == torch.float64 else torch.float32
            dscale_acc = torch.zeros(D, dtype=calc_dtype, device=scale.device)
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


