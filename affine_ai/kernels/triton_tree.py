"""Triton fused sparse-tree-perm forward (CUDA).

Fuses the ASTDAG perm-leaf CUDA fallback path:
  leaf_prim[b,k,d] = bias[k,d] + sum_p w_perm[k,p,d] * r_in[b, perms[k,p,d]]
  act = relu6(leaf_prim)
  out[b,d] = sum_k routing_sparse[b,k] * act[b,k,d]

...into a single kernel that evaluates ONLY the top-k active leaves per
row (the reference computes all K then masks). Elementwise FMAs only
(SIMT, no Tensor Cores). Backward recomputes with plain torch ops under
enable_grad (same pattern as the C++ sparse_tree_perm fallback).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _tree_perm_fwd_kernel(
    R, W, B, P, TI, TW, Y,
    stride_rm, stride_rd,
    stride_wk, stride_wp, stride_wd,
    stride_bk, stride_bd,
    stride_pk, stride_pp, stride_pd,
    stride_tbt, stride_tbk,
    stride_twb, stride_twk,
    stride_ym, stride_yd,
    B_ROWS, K_LEAVES, D_DIM, P_NUM, TOPK,
    BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_b = offs_b < B_ROWS
    mask_d = offs_d < D_DIM
    mask = mask_b[:, None] & mask_d[None, :]

    acc_total = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
    for tk in range(TOPK):
        ki = tl.load(TI + offs_b * stride_tbt + tk * stride_tbk, mask=mask_b, other=0)
        tw = tl.load(TW + offs_b * stride_twb + tk * stride_twk, mask=mask_b, other=0.0)
        ki = tl.where(mask_b, ki, 0)

        acc = tl.load(
            B + ki[:, None] * stride_bk + offs_d[None, :] * stride_bd,
            mask=mask, other=0.0,
        )
        for p in range(P_NUM):
            w = tl.load(
                W + ki[:, None] * stride_wk + p * stride_wp + offs_d[None, :] * stride_wd,
                mask=mask, other=0.0,
            )
            pm = tl.load(
                P + ki[:, None] * stride_pk + p * stride_pp + offs_d[None, :] * stride_pd,
                mask=mask, other=0,
            )
            xv = tl.load(
                R + offs_b[:, None] * stride_rm + pm * stride_rd,
                mask=mask, other=0.0,
            )
            acc += w * xv
        acc = tl.minimum(tl.maximum(acc, 0.0), 6.0)
        acc_total += acc * tw[:, None]

    tl.store(
        Y + offs_b[:, None] * stride_ym + offs_d[None, :] * stride_yd,
        acc_total, mask=mask,
    )


def triton_tree_perm_fwd(r_in, w_perm, bias, perms, top_idx, top_w):
    B, D = r_in.shape
    K, P, Dp = w_perm.shape
    Tk = top_idx.shape[1]
    assert Dp == D
    out = torch.zeros((B, D), device=r_in.device, dtype=torch.float32)
    BM, BD = 16, 32
    grid = ((B + BM - 1) // BM, (D + BD - 1) // BD)
    _tree_perm_fwd_kernel[grid](
        r_in, w_perm, bias, perms, top_idx, top_w, out,
        r_in.stride(0), r_in.stride(1),
        w_perm.stride(0), w_perm.stride(1), w_perm.stride(2),
        bias.stride(0), bias.stride(1),
        perms.stride(0), perms.stride(1), perms.stride(2),
        top_idx.stride(0), top_idx.stride(1),
        top_w.stride(0), top_w.stride(1),
        out.stride(0), out.stride(1),
        B, K, D, P, Tk,
        BLOCK_B=BM, BLOCK_D=BD, num_warps=4)
    return out


class TritonTreePermFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r_in, w_perm, bias, perms, top_idx, top_w):
        ctx.save_for_backward(r_in, w_perm, bias, perms, top_idx, top_w)
        r_f = r_in.reshape(-1, r_in.shape[-1]).contiguous().float()
        out = triton_tree_perm_fwd(
            r_f,
            w_perm.detach().float().contiguous(),
            bias.detach().float().contiguous(),
            perms.detach().to(torch.int32).contiguous(),
            top_idx.detach().to(torch.int32).contiguous(),
            top_w.detach().float().contiguous(),
        )
        return out.to(r_in.dtype).reshape(*r_in.shape[:-1], r_in.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        r_in, w_perm, bias, perms, top_idx, top_w = ctx.saved_tensors
        orig_shape = r_in.shape
        D = r_in.shape[-1]
        r_flat = r_in.reshape(-1, D)
        B_flat = r_flat.shape[0]
        K = w_perm.shape[0]
        P = w_perm.shape[1]
        Tk = top_idx.shape[1]

        go_flat = grad_output.reshape(B_flat, D).contiguous()

        flat_k = top_idx.reshape(-1)
        b_active = bias[top_idx]
        w_active = w_perm[top_idx]
        perms_active = perms[top_idx]

        r_exp = r_flat.unsqueeze(1).unsqueeze(2).expand(B_flat, Tk, P, D)
        x_p_active = torch.gather(r_exp, -1, perms_active.long())

        prim_active = b_active + (x_p_active * w_active).sum(dim=2)
        mask_relu = ((prim_active > 0.0) & (prim_active < 6.0)).to(go_flat.dtype)
        act_active = F.relu6(prim_active)

        gt = (act_active * go_flat.unsqueeze(1)).sum(dim=-1) if ctx.needs_input_grad[5] else None

        d_prim = (go_flat.unsqueeze(1) * top_w.unsqueeze(-1)) * mask_relu

        gb = None
        if ctx.needs_input_grad[2] and isinstance(bias, torch.Tensor):
            gb = torch.zeros(bias.shape, dtype=torch.float32, device=bias.device)
            gb.index_add_(0, flat_k, d_prim.reshape(-1, D).float())
            gb = gb.to(bias.dtype)

        gw = None
        if ctx.needs_input_grad[1]:
            gw_active = d_prim.unsqueeze(2) * x_p_active
            gw = torch.zeros(w_perm.shape, dtype=torch.float32, device=w_perm.device)
            gw.index_add_(0, flat_k, gw_active.reshape(-1, P, D).float())
            gw = gw.to(w_perm.dtype)

        gr = None
        if ctx.needs_input_grad[0]:
            dx_p_active = d_prim.unsqueeze(2) * w_active
            perms_flat_tp = perms_active.view(B_flat, -1, D)
            dx_flat_tp = dx_p_active.view(B_flat, -1, D).float()
            gr_flat = torch.zeros((B_flat, D), dtype=torch.float32, device=r_flat.device)
            for p_idx in range(Tk * P):
                gr_flat.scatter_add_(-1, perms_flat_tp[:, p_idx, :].long(), dx_flat_tp[:, p_idx, :])
            gr = gr_flat.to(r_flat.dtype).reshape(orig_shape)

        return gr, gw, gb, None, None, gt


def triton_tree_perm(r_in, w_perm, bias, perms, top_idx, top_w):
    return TritonTreePermFunction.apply(r_in, w_perm, bias, perms, top_idx, top_w)
