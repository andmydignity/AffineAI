"""Triton fused hierarchical sign-router cascade + top-k (CUDA).

Mirrors HierarchicalSignRouter.route_tokens + top-k normalize:
  logits [B, I] (cuBLAS, kept outside) -> sigmoid cascade over tree
  levels -> 16 leaf probs -> normalize -> exact top-2 (lower-index-first
  ties, matching torch.topk) + weights.

Backward recomputes with plain torch ops under enable_grad (same
pattern as triton_ternary / triton_tree / triton_gla). Elementwise only
(SIMT); the matmul stays in cuBLAS.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _router_cascade_topk_kernel(
    Logits, TopIdx, TopW,
    stride_lm, stride_li,
    B, I,
    NLEAF: tl.constexpr, DEPTH: tl.constexpr, TOPK: tl.constexpr,
    MAXW: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < B

    col = tl.arange(0, MAXW)
    # Level 0: root node 0 splits into (p_left, p_right)
    z0 = tl.load(Logits + offs_m * stride_lm, mask=mask_m, other=0.0)
    pr0 = tl.sigmoid(tl.clamp(2.0 * z0, -30.0, 30.0))
    cur = tl.where(col[None, :] == 0, 1.0 - pr0[:, None], 0.0)
    cur = tl.where(col[None, :] == 1, pr0[:, None], cur)
    cur_n = 2
    for d in range(1, DEPTH):
        start_node = (1 << d) - 1
        nxt = tl.zeros((BLOCK_M, MAXW), dtype=tl.float32)
        for j in range(MAXW // 2):
            node = start_node + j
            logit = tl.load(
                Logits + offs_m * stride_lm + node * stride_li,
                mask=mask_m, other=0.0,
            )
            sr = tl.sigmoid(tl.clamp(2.0 * logit, -30.0, 30.0))
            pv = tl.sum(tl.where(col[None, :] == j, cur, 0.0), axis=1)
            nxt = tl.where(col[None, :] == 2 * j, (pv * (1.0 - sr))[:, None], nxt)
            nxt = tl.where(col[None, :] == 2 * j + 1, (pv * sr)[:, None], nxt)
        cur = nxt
        cur_n = cur_n * 2
    # normalize over first NLEAF entries (reference slices then divides)
    valid = col[None, :] < NLEAF
    s = tl.sum(tl.where(valid, cur, 0.0), axis=1)
    s = tl.maximum(s, 1e-8)
    leaf = cur / s[:, None]
    # exact top-k, lower-index-first ties: repeated argmax with strict >,
    # then renormalize selected weights to sum to 1 like the reference
    work = tl.where(valid & mask_m[:, None], leaf, -1.0)
    sel_idx = tl.zeros((BLOCK_M, TOPK), dtype=tl.int32)
    sel_val = tl.full((BLOCK_M, TOPK), -1.0, dtype=tl.float32)
    for t in tl.static_range(TOPK):
        best = tl.max(work, axis=1)
        is_best = (work == best[:, None]) & valid & mask_m[:, None]
        order = tl.arange(0, MAXW)[None, :]
        masked_order = tl.where(is_best, order, MAXW)
        bi = tl.min(masked_order, axis=1).to(tl.int32)
        sel_idx = tl.where((tl.arange(0, TOPK)[None, :] == t) & mask_m[:, None], bi[:, None], sel_idx)
        sel_val = tl.where((tl.arange(0, TOPK)[None, :] == t) & mask_m[:, None], best[:, None], sel_val)
        work = tl.where(tl.arange(0, MAXW)[None, :] == bi[:, None], -1.0, work)
    wsum = tl.sum(sel_val, axis=1)
    wsum = tl.maximum(wsum, 1e-8)
    sel_val = sel_val / wsum[:, None]
    tl.store(TopIdx + offs_m[:, None] * TOPK + tl.arange(0, TOPK)[None, :],
             sel_idx, mask=mask_m[:, None])
    tl.store(TopW + offs_m[:, None] * TOPK + tl.arange(0, TOPK)[None, :],
             sel_val, mask=mask_m[:, None])


def _grid(m, bm=64):
    return ((m + bm - 1) // bm,)


def triton_router_topk_fwd(node_logits, tree_depth, top_k, num_leaves=None):
    B, I = node_logits.shape
    if num_leaves is None:
        num_leaves = 1 << tree_depth
    top_idx = torch.empty((B, top_k), device=node_logits.device, dtype=torch.int64)
    top_w = torch.empty((B, top_k), device=node_logits.device, dtype=torch.float32)
    _router_cascade_topk_kernel[_grid(B, 64)](
        node_logits, top_idx, top_w,
        node_logits.stride(0), node_logits.stride(1),
        B, I, num_leaves, tree_depth, top_k, 1 << tree_depth, BLOCK_M=64, num_warps=4)
    return top_idx, top_w


class TritonRouterTopkFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_flat, hyperplanes, biases, tree_depth, top_k, num_leaves):
        from affine_ai.core.ast_dag import ternarize
        with torch.no_grad():
            W_route = ternarize(hyperplanes)
            node_logits = torch.nn.functional.linear(x_flat.float(), W_route.float(), biases.float())
            top_idx, top_w = triton_router_topk_fwd(node_logits, tree_depth, top_k, num_leaves)
        ctx.save_for_backward(x_flat, hyperplanes, biases)
        ctx.tree_depth = tree_depth
        ctx.top_k = top_k
        ctx.num_leaves = num_leaves
        return top_idx, top_w

    @staticmethod
    def backward(ctx, grad_idx, grad_w):
        x_flat, hyperplanes, biases = ctx.saved_tensors
        with torch.enable_grad():
            xr = x_flat.detach().requires_grad_(x_flat.requires_grad)
            hr = hyperplanes.detach().requires_grad_(hyperplanes.requires_grad)
            br = biases.detach().requires_grad_(biases.requires_grad)
            import torch.nn.functional as F
            from affine_ai.core.ast_dag import ternarize
            W_route = ternarize(hr)
            node_logits = F.linear(xr.reshape(-1, xr.shape[-1]).float(), W_route.float(), br.float())
            logit_root = node_logits[:, 0:1]
            p_right = torch.sigmoid(logit_root * 2.0)
            current_level_probs = [1.0 - p_right, p_right]
            for depth in range(1, ctx.tree_depth):
                next_level_probs = []
                start_node = (1 << depth) - 1
                for n_idx, p_parent in enumerate(current_level_probs):
                    logit = node_logits[:, start_node + n_idx:start_node + n_idx + 1]
                    pr = torch.sigmoid(logit * 2.0)
                    next_level_probs.append(p_parent * (1.0 - pr))
                    next_level_probs.append(p_parent * pr)
                current_level_probs = next_level_probs
            leaf_probs = torch.cat(current_level_probs, dim=-1)
            routing_probs = leaf_probs[:, :ctx.num_leaves]
            routing_probs = routing_probs / routing_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            top_vals, _ = torch.topk(routing_probs, k=ctx.top_k, dim=-1)
            top_weights = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            torch.autograd.backward(top_weights, grad_w.reshape(top_weights.shape).float())
        return (xr.grad if xr.requires_grad else None,
                hr.grad if hr.requires_grad else None,
                br.grad if br.requires_grad else None,
                None, None, None)


def triton_router_topk(x_flat, hyperplanes, biases, tree_depth, top_k, num_leaves):
    return TritonRouterTopkFunction.apply(x_flat, hyperplanes, biases, tree_depth, top_k, num_leaves)
