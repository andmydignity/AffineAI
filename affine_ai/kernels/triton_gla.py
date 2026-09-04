"""Triton fused Monarch permutation chain (CUDA).

Replaces the torch-op reference implementations with single-launch
fused kernels. Math (single chain, S stages):
  y[b,d] = bias[d] + D[S-1][d] * D[S-2][p[S-2][d]] * ... * x[b, P(d)]
where P(d) is the composed gather index. All stages fuse into one pass:
no intermediate VRAM writes. Elementwise FMAs only (SIMT).

Backward recomputes with plain torch ops under enable_grad (same
pattern as triton_ternary / triton_tree).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _monarch_chain_fwd_kernel(
    X, Diag, Perms, Bias, Y,
    stride_xm, stride_xd,
    stride_ds, stride_dd,
    stride_ps, stride_pd,
    stride_ym, stride_yd,
    N, D, NSTAGES,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = offs_m < N
    mask_d = offs_d < D

    idx = offs_d.to(tl.int32)
    wacc = tl.full((BLOCK_M, BLOCK_D), 1.0, dtype=tl.float32)
    for s in range(NSTAGES - 1, 0, -1):
        wd = tl.load(
            Diag + s * stride_ds + idx[None, :] * stride_dd,
            mask=mask_m[:, None] & mask_d[None, :], other=1.0,
        )
        wacc = wacc * wd
        idx = tl.load(
            Perms + (s - 1) * stride_ps + idx * stride_pd,
            mask=mask_d, other=0,
        )
    w0 = tl.load(
        Diag + idx[None, :] * stride_dd,
        mask=mask_m[:, None] & mask_d[None, :], other=1.0,
    )
    wacc = wacc * w0
    xv = tl.load(
        X + offs_m[:, None] * stride_xm + idx[None, :] * stride_xd,
        mask=mask_m[:, None] & mask_d[None, :], other=0.0,
    )
    b = tl.load(Bias + offs_d, mask=mask_d, other=0.0)
    y = b[None, :] + wacc * xv
    tl.store(
        Y + offs_m[:, None] * stride_ym + offs_d[None, :] * stride_yd,
        y, mask=mask_m[:, None] & mask_d[None, :],
    )


@triton.jit
def _fused_monarch_chain_fwd_kernel(
    X, Diag, Perms, Bias, Y,
    stride_xm, stride_xd,
    stride_dm, stride_ds, stride_dd,
    stride_ps, stride_pd,
    stride_bm, stride_bd,
    stride_ym, stride_yb, stride_yd,
    N, D, NSTAGES, NBR,
    BLOCK_MB: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_mb = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_mb = pid_mb * BLOCK_MB + tl.arange(0, BLOCK_MB)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_mb = offs_mb < NBR * N
    mask_d = offs_d < D
    m_idx = offs_mb // N
    b_idx = offs_mb % N
    mask_m = m_idx < NBR
    mask_b = b_idx < N
    row_ok = mask_mb & mask_b

    idx = offs_d.to(tl.int32)
    wacc = tl.full((BLOCK_MB, BLOCK_D), 1.0, dtype=tl.float32)
    for s in range(NSTAGES - 1, 0, -1):
        wd = tl.load(
            Diag + m_idx[:, None] * stride_dm + s * stride_ds + idx[None, :] * stride_dd,
            mask=row_ok[:, None] & mask_d[None, :], other=1.0,
        )
        wacc = wacc * wd
        idx = tl.load(
            Perms + (s - 1) * stride_ps + idx * stride_pd,
            mask=mask_d, other=0,
        )
    w0 = tl.load(
        Diag + m_idx[:, None] * stride_dm + idx[None, :] * stride_dd,
        mask=row_ok[:, None] & mask_d[None, :], other=1.0,
    )
    wacc = wacc * w0
    xv = tl.load(
        X + b_idx[:, None] * stride_xm + idx[None, :] * stride_xd,
        mask=row_ok[:, None] & mask_d[None, :], other=0.0,
    )
    b = tl.load(
        Bias + m_idx[:, None] * stride_bm + offs_d[None, :] * stride_bd,
        mask=row_ok[:, None] & mask_d[None, :], other=0.0,
    )
    y = b + wacc * xv
    tl.store(
        Y + m_idx[:, None] * stride_ym + b_idx[:, None] * stride_yb + offs_d[None, :] * stride_yd,
        y, mask=row_ok[:, None] & mask_d[None, :],
    )


def _grid(m, n, bm=32, bn=32):
    return ((m + bm - 1) // bm, (n + bn - 1) // bn)


def triton_fused_monarch_chain_fwd(x, diagonals, perms, bias):
    N, D = x.shape
    M, S = diagonals.shape[0], diagonals.shape[1]
    if perms.dtype != torch.int32:
        perms = perms.to(torch.int32)
    out = torch.empty((M, N, D), device=x.device, dtype=torch.float32)
    BM = 32
    _fused_monarch_chain_fwd_kernel[_grid(M * N, D, BM, 32)](
        x, diagonals, perms, bias, out,
        x.stride(0), x.stride(1),
        diagonals.stride(0), diagonals.stride(1), diagonals.stride(2),
        perms.stride(0), perms.stride(1),
        bias.stride(0), bias.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        N, D, S, M, BLOCK_MB=BM, BLOCK_D=32, num_warps=4)
    return out


def triton_monarch_chain_fwd(x, diagonals, perms, bias):
    N, D = x.shape
    S = diagonals.shape[0]
    if perms.dtype != torch.int32:
        perms = perms.to(torch.int32)
    out = torch.empty((N, D), device=x.device, dtype=torch.float32)
    _monarch_chain_fwd_kernel[_grid(N, D, 32, 32)](
        x, diagonals, perms, bias, out,
        x.stride(0), x.stride(1),
        diagonals.stride(0), diagonals.stride(1),
        perms.stride(0), perms.stride(1),
        out.stride(0), out.stride(1),
        N, D, S, BLOCK_M=32, BLOCK_D=32, num_warps=4)
    return out


class TritonMonarchChainFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, diagonals, perms, inv_perms, bias):
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1]).to(diagonals.dtype)
        num_stages = diagonals.shape[0]
        
        h_list = [x_flat * diagonals[0]]
        for s in range(num_stages - 1):
            h_next = h_list[-1][:, perms[s]] * diagonals[s + 1]
            h_list.append(h_next)
            
        out = h_list[-1] + bias
        ctx.save_for_backward(x_flat, diagonals, perms, inv_perms, *h_list[:-1])
        ctx.num_stages = num_stages
        ctx.orig_shape = orig_shape
        ctx.orig_dtype = x.dtype
        return out.to(x.dtype).reshape(*orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        x_flat = saved[0]
        diagonals = saved[1]
        perms = saved[2]
        inv_perms = saved[3]
        num_stages = ctx.num_stages
        h_list = saved[4:]
        
        go_flat = grad_output.reshape(-1, grad_output.shape[-1]).to(diagonals.dtype)
        g_bias = go_flat.sum(0)
        g_diagonals = torch.empty_like(diagonals)
        gh = go_flat
        
        for s in range(num_stages - 1, 0, -1):
            h_perm = h_list[s - 1][:, perms[s - 1]]
            g_diagonals[s] = (gh * h_perm).sum(0)
            gh = (gh * diagonals[s])[:, inv_perms[s - 1]]
            
        g_diagonals[0] = (gh * x_flat).sum(0)
        gx = (gh * diagonals[0]).to(ctx.orig_dtype)
        return gx.reshape(*ctx.orig_shape), g_diagonals, None, None, g_bias


class TritonFusedMonarchChainFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, diagonals, perms, inv_perms, bias):
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1]).to(diagonals.dtype)
        num_branches = diagonals.shape[0]
        num_stages = diagonals.shape[1]
        
        h_list = [x_flat.unsqueeze(0) * diagonals[:, 0].unsqueeze(1)]
        for s in range(num_stages - 1):
            h_next = h_list[-1][:, :, perms[s]] * diagonals[:, s + 1].unsqueeze(1)
            h_list.append(h_next)
            
        out = h_list[-1] + bias.unsqueeze(1)
        ctx.save_for_backward(x_flat, diagonals, perms, inv_perms, *h_list[:-1])
        ctx.num_branches = num_branches
        ctx.num_stages = num_stages
        ctx.orig_shape = orig_shape
        ctx.orig_dtype = x.dtype
        return tuple(out[m].to(x.dtype).reshape(*orig_shape) for m in range(num_branches))

    @staticmethod
    def backward(ctx, *grad_outs):
        saved = ctx.saved_tensors
        x_flat = saved[0]
        diagonals = saved[1]
        perms = saved[2]
        inv_perms = saved[3]
        num_stages = ctx.num_stages
        h_list = saved[4:]
        
        g_stack = torch.stack([g.reshape(-1, g.shape[-1]).to(diagonals.dtype) for g in grad_outs], dim=0)
        g_bias = g_stack.sum(1)
        g_diagonals = torch.empty_like(diagonals)
        gh = g_stack
        
        for s in range(num_stages - 1, 0, -1):
            h_perm = h_list[s - 1][:, :, perms[s - 1]]
            g_diagonals[:, s] = (gh * h_perm).sum(1)
            gh = (gh * diagonals[:, s].unsqueeze(1))[:, :, inv_perms[s - 1]]
            
        g_diagonals[:, 0] = (gh * x_flat.unsqueeze(0)).sum(1)
        gx = (gh * diagonals[:, 0].unsqueeze(1)).sum(0).to(ctx.orig_dtype)
        return gx.reshape(*ctx.orig_shape), g_diagonals, None, None, g_bias


def triton_monarch_chain(x, diagonals, perms, inv_perms, bias):
    return TritonMonarchChainFunction.apply(x, diagonals, perms, inv_perms, bias)


def triton_fused_monarch_chain(x, diagonals, perms, inv_perms, bias):
    return TritonFusedMonarchChainFunction.apply(x, diagonals, perms, inv_perms, bias)


@triton.jit
def _gla_decay_kernel(
    Cum, Decay,
    stride_cb, stride_ch, stride_ct,
    stride_db, stride_dh, stride_di, stride_dj,
    B, H, T,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = B * H * T * T
    mask = offs < total
    tmp = offs
    j = tmp % T
    tmp = tmp // T
    i = tmp % T
    tmp = tmp // T
    h = tmp % H
    b = tmp // H
    ci = tl.load(
        Cum + b * stride_cb + h * stride_ch + i * stride_ct,
        mask=mask, other=0.0,
    )
    cj = tl.load(
        Cum + b * stride_cb + h * stride_ch + j * stride_ct,
        mask=mask, other=0.0,
    )
    diff = ci - cj
    diff = tl.minimum(diff, 0.0)
    m = j <= i
    val = tl.exp(diff)
    val = tl.where(m & mask, val, 0.0)
    out_ptr = Decay + b * stride_db + h * stride_dh + i * stride_di + j * stride_dj
    tl.store(out_ptr, val, mask=mask)


def triton_gla_decay_fwd(cum_log_gam):
    B, H, T = cum_log_gam.shape
    out = torch.empty((B, H, T, T), device=cum_log_gam.device, dtype=torch.float32)
    BLOCK = 1024
    grid = ((B * H * T * T + BLOCK - 1) // BLOCK,)
    _gla_decay_kernel[grid](
        cum_log_gam, out,
        cum_log_gam.stride(0), cum_log_gam.stride(1), cum_log_gam.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, T, BLOCK=BLOCK, num_warps=4)
    return out


class TritonGLADecayFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gamma):
        log_gam = torch.log(gamma.float().clamp(min=1e-5))
        cum = torch.cumsum(log_gam, dim=-1)
        out = triton_gla_decay_fwd(cum.contiguous())
        ctx.save_for_backward(gamma)
        return out.to(gamma.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None
        (gamma,) = ctx.saved_tensors
        with torch.enable_grad():
            gr = gamma.detach().requires_grad_(True)
            log_gam = torch.log(gr.float().clamp(min=1e-5))
            cum = torch.cumsum(log_gam, dim=-1)
            T = cum.shape[-1]
            decay_diff = (cum.unsqueeze(-1) - cum.unsqueeze(-2)).clamp(max=0.0)
            mask = torch.tril(torch.ones(T, T, device=cum.device, dtype=torch.bool))
            out = torch.where(mask, torch.exp(decay_diff), torch.zeros_like(decay_diff))
            torch.autograd.backward(out, grad_output.reshape(out.shape).float())
        return gr.grad.to(gamma.dtype) if gr.grad is not None else None


def triton_gla_decay(gamma):
    return TritonGLADecayFunction.apply(gamma)
