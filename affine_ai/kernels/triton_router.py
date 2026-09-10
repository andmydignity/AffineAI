"""Hierarchical sign-router cascade + top-k (CUDA, cuBLAS + Triton).

Mirrors HierarchicalSignRouter.route_tokens + top-k normalize:
  logits [B, I] computed via cuBLAS matmul (dominates cost) before Triton cascade
  -> sigmoid cascade over tree levels -> N leaf probs -> normalize -> exact top-k
  (lower-index-first ties, matching torch.topk which has undefined tie behavior) + weights.

Backward recomputes with plain torch ops under enable_grad (same
pattern as triton_ternary / triton_tree / triton_gla). Elementwise only
(SIMT); the matmul stays in cuBLAS.

TODO: use tl.make_block_ptr for coalesced access.
"""

import torch
from typing import Optional
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
            triton.Config({"BLOCK_M": 16}, num_warps=2),
            triton.Config({"BLOCK_M": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 64}, num_warps=4),
            triton.Config({"BLOCK_M": 64}, num_warps=8),
        ],
        key=["B"],
    )
    @triton.jit
    def _router_cascade_topk_kernel(
        Logits, TopIdx, TopW,
        stride_lm, stride_li,
        B, I,  # noqa: E741
        NLEAF: tl.constexpr, DEPTH: tl.constexpr, TOPK: tl.constexpr,
        MAXW: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        tl.device_assert(DEPTH <= 6, "DEPTH >6 would blow registers (MAXW=1<<DEPTH)")
        if DEPTH > 6:
            return
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < B

        col = tl.arange(0, MAXW)
        z0 = tl.load(Logits + offs_m * stride_lm, mask=mask_m, other=0.0)
        pr0 = tl.sigmoid(tl.clamp(2.0 * z0, -30.0, 30.0))
        cur = tl.where(col[None, :] == 0, 1.0 - pr0[:, None], 0.0)
        cur = tl.where(col[None, :] == 1, pr0[:, None], cur)
        cur_n = 2
        for d in tl.static_range(6):
            if d >= 1 and d < DEPTH:
                start_node = (1 << d) - 1
                nxt = tl.zeros((BLOCK_M, MAXW), dtype=tl.float32)
                for j in tl.static_range(64):
                    if j < (1 << d):
                        node = start_node + j
                        tl.device_assert(node < I, "node OOB")
                        logit = tl.load(
                            Logits + offs_m * stride_lm + node * stride_li,
                            mask=mask_m & (node < I), other=0.0,
                        )
                        sr = tl.where(node < I, tl.sigmoid(tl.clamp(2.0 * logit, -30.0, 30.0)), 0.0)
                        pv = tl.sum(tl.where(col[None, :] == j, cur, 0.0), axis=1)
                        nxt = tl.where(col[None, :] == 2 * j, (pv * (1.0 - sr))[:, None], nxt)
                        nxt = tl.where(col[None, :] == 2 * j + 1, (pv * sr)[:, None], nxt)
                cur = nxt
                cur_n = cur_n * 2
        valid = col[None, :] < NLEAF
        s = tl.sum(tl.where(valid, cur, 0.0), axis=1)
        s = tl.maximum(s, 1e-8)
        leaf = cur / s[:, None]
        work = tl.where(valid & mask_m[:, None], leaf, -1.0)
        sel_idx = tl.zeros((BLOCK_M, TOPK), dtype=tl.int32)
        sel_val = tl.full((BLOCK_M, TOPK), -1.0, dtype=tl.float32)
        for t in tl.static_range(TOPK):
            best = tl.max(work, axis=1)
            is_best = (work == best[:, None]) & valid & mask_m[:, None]
            order = tl.arange(0, MAXW)[None, :]
            masked_order = tl.where(is_best, order, MAXW)
            bi = tl.min(masked_order, axis=1).to(tl.int32)
            bi_valid = bi < MAXW
            sel_idx = tl.where((tl.arange(0, TOPK)[None, :] == t) & mask_m[:, None] & bi_valid[:, None], bi[:, None], sel_idx)
            sel_val = tl.where((tl.arange(0, TOPK)[None, :] == t) & mask_m[:, None] & bi_valid[:, None], best[:, None], sel_val)
            work = tl.where((tl.arange(0, MAXW)[None, :] == bi[:, None]) & bi_valid[:, None], -1.0, work)
        wsum = tl.sum(sel_val, axis=1)
        wsum = tl.maximum(wsum, 1e-8)
        sel_val = sel_val / wsum[:, None]
        tl.store(TopIdx + offs_m[:, None] * TOPK + tl.arange(0, TOPK)[None, :],
                 sel_idx.to(tl.int64), mask=mask_m[:, None])
        tl.store(TopW + offs_m[:, None] * TOPK + tl.arange(0, TOPK)[None, :],
                 sel_val, mask=mask_m[:, None])
else:
    _router_cascade_topk_kernel = None  # type: ignore


def _grid(m, bm=64):
    return ((m + bm - 1) // bm,)


def triton_router_topk_fwd(node_logits, tree_depth, top_k, num_leaves=None):  # noqa: E741
    B, I = node_logits.shape  # noqa: E741
    assert node_logits.shape[1] == (1 << tree_depth) - 1, f"logits width {node_logits.shape[1]} != (1<<DEPTH)-1 {(1<<tree_depth)-1}"
    if num_leaves is None:
        num_leaves = 1 << tree_depth
    if tree_depth > 6:
        raise ValueError(f"tree_depth {tree_depth} exceeds Triton kernel guard (MAXW=1<<DEPTH would blow registers); use torch fallback")
    if (1 << tree_depth) > 64:
        raise ValueError("MAXW exceeds 64 register limit")
    top_k = min(top_k, num_leaves)
    if triton is None or not node_logits.is_cuda or _router_cascade_topk_kernel is None:
        with torch.no_grad():
            cur = torch.zeros(B, 1 << tree_depth, device=node_logits.device, dtype=torch.float32)
            z0 = node_logits[:, 0]
            pr0 = torch.sigmoid(torch.clamp(z0 * 2.0, -30, 30))
            cur[:, 0] = 1.0 - pr0
            cur[:, 1] = pr0
            cur_n = 2
            for d in range(1, tree_depth):
                start_node = (1 << d) - 1
                nxt = torch.zeros_like(cur)
                for j in range(1 << d):
                    node = start_node + j
                    logit = node_logits[:, node]
                    sr = torch.sigmoid(torch.clamp(logit * 2.0, -30, 30))
                    pv = cur[:, j]
                    nxt[:, 2 * j] = pv * (1.0 - sr)
                    nxt[:, 2 * j + 1] = pv * sr
                cur = nxt
                cur_n *= 2
            valid = torch.arange(1 << tree_depth, device=node_logits.device) < num_leaves
            s = cur[:, valid].sum(dim=-1, keepdim=True).clamp(min=1e-8)
            leaf = cur[:, valid] / s
            top_idx = torch.empty(B, top_k, device=node_logits.device, dtype=torch.int64)
            top_w = torch.empty(B, top_k, device=node_logits.device, dtype=torch.float32)
            for b in range(B):
                work = leaf[b].clone()
                for t in range(top_k):
                    maxv = work.max()
                    candidates = (work == maxv).nonzero(as_tuple=False).flatten()
                    bi = int(candidates.min().item()) if len(candidates) > 0 else 0
                    top_idx[b, t] = bi
                    top_w[b, t] = maxv
                    work[bi] = -1.0
                wsum = top_w[b].sum().clamp(min=1e-8)
                top_w[b] /= wsum
            return top_idx, top_w
    top_idx = torch.empty((B, top_k), device=node_logits.device, dtype=torch.int64)
    top_w = torch.empty((B, top_k), device=node_logits.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(B, META["BLOCK_M"]),)  # noqa: E731
    _router_cascade_topk_kernel[grid](
        node_logits, top_idx, top_w,
        node_logits.stride(0), node_logits.stride(1),
        B, I, num_leaves, tree_depth, top_k, 1 << tree_depth)
    return top_idx, top_w


class TritonRouterTopkFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_flat, hyperplanes, biases, tree_depth, top_k, num_leaves):
        from affine_ai.core.ast_dag import ternarize
        top_k = min(top_k, num_leaves)
        with torch.no_grad():
            W_route = ternarize(hyperplanes)
            node_logits = torch.nn.functional.linear(x_flat.float(), W_route.float(), biases.float())
            top_idx, top_w = triton_router_topk_fwd(node_logits, tree_depth, top_k, num_leaves)
        ctx.save_for_backward(x_flat, hyperplanes, biases, top_idx)
        ctx.tree_depth = tree_depth
        ctx.top_k = top_k
        ctx.num_leaves = num_leaves
        return top_idx, top_w

    @staticmethod
    def backward(ctx, grad_idx, grad_w):
        if grad_w is None or not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]):
            return None, None, None, None, None, None
        x_flat, hyperplanes, biases, fwd_top_idx = ctx.saved_tensors
        from affine_ai.core.ast_dag import ternarize
        W_route = ternarize(hyperplanes)
        xr_flat = x_flat.reshape(-1, x_flat.shape[-1])
        node_logits = torch.nn.functional.linear(xr_flat.float(), W_route.float(), biases.float())
        B_nodes = node_logits.shape[0]

        probs_levels = []
        sigmoids = []
        p_r0 = torch.sigmoid(node_logits[:, 0:1] * 2.0)
        sigmoids.append(p_r0)
        cur_p = torch.cat([1.0 - p_r0, p_r0], dim=-1)
        probs_levels.append(cur_p)
        for depth in range(1, ctx.tree_depth):
            start_node = (1 << depth) - 1
            num_nodes = 1 << depth
            ll = node_logits[:, start_node:start_node + num_nodes]
            pr = torch.sigmoid(ll * 2.0)
            sigmoids.append(pr)
            cur_p = torch.stack([cur_p * (1.0 - pr), cur_p * pr], dim=-1).view(B_nodes, -1)
            probs_levels.append(cur_p)

        leaf_probs = cur_p
        routing_probs = leaf_probs[:, :ctx.num_leaves]
        s_routing = routing_probs.sum(dim=-1, keepdim=True)
        routing_norm = routing_probs / s_routing.clamp(min=1e-8)
        top_k = min(ctx.top_k, ctx.num_leaves)
        if fwd_top_idx.shape == (B_nodes, top_k):
            top_idx = fwd_top_idx
            top_vals = torch.gather(routing_norm, -1, top_idx)
        else:
            top_vals = torch.empty(B_nodes, top_k, device=routing_norm.device, dtype=routing_norm.dtype)
            top_idx = torch.empty(B_nodes, top_k, device=routing_norm.device, dtype=torch.int64)
            work = routing_norm.clone()
            for t in range(top_k):
                best = work.max(dim=-1).values
                is_best = work == best.unsqueeze(-1)
                masked_order = torch.where(is_best, torch.arange(ctx.num_leaves, device=work.device), ctx.num_leaves)
                bi = masked_order.min(dim=-1).values
                top_idx[:, t] = bi
                top_vals[:, t] = best
                work[torch.arange(B_nodes), bi] = -1.0
        s_top = top_vals.sum(dim=-1, keepdim=True)
        top_weights = top_vals / s_top.clamp(min=1e-8)

        gw = grad_w.reshape(top_weights.shape).float()

        # Step 1: dL / d(top_vals)
        mask_v = (s_top >= 1e-8).float()
        grad_v = (gw - mask_v * (gw * top_weights).sum(dim=-1, keepdim=True)) / s_top.clamp(min=1e-8)

        # Step 2: dL / d(routing_norm)
        grad_rn = torch.zeros_like(routing_norm)
        grad_rn.scatter_add_(-1, top_idx, grad_v)

        # Step 3: dL / d(routing_probs)
        mask_u = (s_routing >= 1e-8).float()
        grad_u = (grad_rn - mask_u * (grad_rn * routing_norm).sum(dim=-1, keepdim=True)) / s_routing.clamp(min=1e-8)

        # Step 4: dL / d(leaf_probs)
        grad_leaf = torch.zeros_like(leaf_probs)
        grad_leaf[:, :ctx.num_leaves] = grad_u

        # Step 5: Tree cascade backward
        grad_logits = torch.empty_like(node_logits)
        grad_cur = grad_leaf

        for depth in range(ctx.tree_depth - 1, 0, -1):
            start_node = (1 << depth) - 1
            num_nodes = 1 << depth
            pr = sigmoids[depth]
            p_parent = probs_levels[depth - 1]
            g_children = grad_cur.view(B_nodes, num_nodes, 2)
            g_l = g_children[:, :, 0]
            g_r = g_children[:, :, 1]
            grad_logits[:, start_node:start_node + num_nodes] = 2.0 * pr * (1.0 - pr) * p_parent * (g_r - g_l)
            grad_cur = g_l * (1.0 - pr) + g_r * pr

        pr0 = sigmoids[0]
        g_children0 = grad_cur.view(B_nodes, 1, 2)
        g_l0 = g_children0[:, :, 0]
        g_r0 = g_children0[:, :, 1]
        grad_logits[:, 0:1] = 2.0 * pr0 * (1.0 - pr0) * (g_r0 - g_l0)

        grad_x = (grad_logits @ W_route.float()).to(x_flat.dtype).reshape(x_flat.shape) if ctx.needs_input_grad[0] else None
        grad_h = (grad_logits.t() @ xr_flat.float()).to(hyperplanes.dtype) if ctx.needs_input_grad[1] else None
        grad_b = grad_logits.sum(0).to(biases.dtype) if ctx.needs_input_grad[2] else None

        return grad_x, grad_h, grad_b, None, None, None


def triton_router_topk(x_flat, hyperplanes, biases, tree_depth, top_k, num_leaves):
    top_k = min(top_k, num_leaves)
    return TritonRouterTopkFunction.apply(x_flat, hyperplanes, biases, tree_depth, top_k, num_leaves)
