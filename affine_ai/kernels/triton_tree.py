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
    BLOCK_BTK: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_btk = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_btk = pid_btk * BLOCK_BTK + tl.arange(0, BLOCK_BTK)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_btk = offs_btk < B_ROWS * TOPK
    mask_d = offs_d < D_DIM

    b = offs_btk // TOPK
    tk = offs_btk % TOPK
    mask_b = b < B_ROWS

    ki = tl.load(TI + b * stride_tbt + tk * stride_tbk, mask=mask_btk, other=0)
    tw = tl.load(TW + b * stride_twb + tk * stride_twk, mask=mask_btk, other=0.0)
    ki = tl.where(mask_btk, ki, 0)

    acc = tl.load(
        B + ki[:, None] * stride_bk + offs_d[None, :] * stride_bd,
        mask=mask_btk[:, None] & mask_d[None, :], other=0.0,
    )
    for p in range(P_NUM):
        w = tl.load(
            W + ki[:, None] * stride_wk + p * stride_wp + offs_d[None, :] * stride_wd,
            mask=mask_btk[:, None] & mask_d[None, :], other=0.0,
        )
        pm = tl.load(
            P + ki[:, None] * stride_pk + p * stride_pp + offs_d[None, :] * stride_pd,
            mask=mask_btk[:, None] & mask_d[None, :], other=0,
        )
        xv = tl.load(
            R + b[:, None] * stride_rm + pm * stride_rd,
            mask=mask_btk[:, None] & mask_d[None, :], other=0.0,
        )
        acc += w * xv
    acc = tl.minimum(tl.maximum(acc, 0.0), 6.0)
    acc = acc * tw[:, None]
    tl.atomic_add(
        Y + b[:, None] * stride_ym + offs_d[None, :] * stride_yd,
        acc, mask=mask_btk[:, None] & mask_d[None, :],
    )


def triton_tree_perm_fwd(r_in, w_perm, bias, perms, top_idx, top_w):
    B, D = r_in.shape
    K, P, Dp = w_perm.shape
    Tk = top_idx.shape[1]
    assert Dp == D
    out = torch.zeros((B, D), device=r_in.device, dtype=torch.float32)
    BM, BD = 16, 32
    grid = ((B * Tk + BM - 1) // BM, (D + BD - 1) // BD)
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
        BLOCK_BTK=BM, BLOCK_D=BD, num_warps=4)
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
        B = r_in.shape[0]
        K = w_perm.shape[0]
        D = r_in.shape[-1]
        P = w_perm.shape[1]
        with torch.enable_grad():
            xr = r_in.detach().requires_grad_(r_in.requires_grad)
            wr = w_perm.detach().requires_grad_(w_perm.requires_grad)
            br = bias.detach().requires_grad_(bias.requires_grad if isinstance(bias, torch.Tensor) else False)
            tw = top_w.detach().requires_grad_(top_w.requires_grad)
            leaf_prim = br.unsqueeze(0).expand(B, K, -1).clone()
            r_exp = xr.unsqueeze(1).expand(-1, K, -1)
            for p_idx in range(P):
                p_k = perms[:, p_idx]
                x_p = torch.gather(r_exp, -1, p_k.unsqueeze(0).expand(B, -1, -1))
                leaf_prim = leaf_prim + x_p * wr[:, p_idx].unsqueeze(0)
            act = F.relu6(leaf_prim)
            ao = act[torch.arange(B, device=act.device).unsqueeze(1), top_idx]
            out = (ao * tw.unsqueeze(-1)).sum(1)
            torch.autograd.backward(out, grad_output.reshape(-1, D).to(out.dtype))
        gr = xr.grad if xr.requires_grad else None
        gw = wr.grad if wr.requires_grad else None
        gb = br.grad if isinstance(br, torch.Tensor) and br.requires_grad else None
        gt = tw.grad if tw.requires_grad else None
        return gr, gw, gb, None, None, gt


def triton_tree_perm(r_in, w_perm, bias, perms, top_idx, top_w):
    return TritonTreePermFunction.apply(r_in, w_perm, bias, perms, top_idx, top_w)
