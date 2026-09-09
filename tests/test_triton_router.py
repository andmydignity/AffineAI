"""
Tests for Triton Hierarchical Sign-Router Cascade + Top-K Kernel
"""

import pytest
import torch
import torch.nn.functional as F

from affine_ai.kernels import TRITON_AVAILABLE

try:
    if TRITON_AVAILABLE:
        from affine_ai.kernels.triton_router import (
            triton_router_topk_fwd,
            triton_router_topk,
            TritonRouterTopkFunction,
        )
        HAS_ROUTER_TRITON = True
    else:
        HAS_ROUTER_TRITON = False
except Exception:
    HAS_ROUTER_TRITON = False


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_ROUTER_TRITON, reason="CUDA and Triton required")
def test_triton_router_topk_fwd_parity():
    torch.manual_seed(42)
    device = "cuda"
    B = 32
    tree_depth = 3
    top_k = 2
    num_leaves = 6
    I = (1 << tree_depth) - 1

    node_logits = torch.randn(B, I, device=device)

    # 1. Triton forward
    top_idx, top_w = triton_router_topk_fwd(node_logits, tree_depth, top_k, num_leaves)

    # Verify return dtype is torch.int64
    assert top_idx.dtype == torch.int64
    assert top_w.dtype == torch.float32

    # 2. PyTorch reference
    logit_root = node_logits[:, 0:1]
    p_right = torch.sigmoid(torch.clamp(logit_root * 2.0, -30.0, 30.0))
    current_level_probs = [1.0 - p_right, p_right]
    for depth in range(1, tree_depth):
        next_level_probs = []
        start_node = (1 << depth) - 1
        for n_idx, p_parent in enumerate(current_level_probs):
            logit = node_logits[:, start_node + n_idx:start_node + n_idx + 1]
            pr = torch.sigmoid(torch.clamp(logit * 2.0, -30.0, 30.0))
            next_level_probs.append(p_parent * (1.0 - pr))
            next_level_probs.append(p_parent * pr)
        current_level_probs = next_level_probs
    leaf_probs = torch.cat(current_level_probs, dim=-1)
    routing_probs = leaf_probs[:, :num_leaves]
    routing_probs = routing_probs / routing_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    ref_top_vals, ref_top_idx = torch.topk(routing_probs, k=top_k, dim=-1)
    ref_top_w = ref_top_vals / ref_top_vals.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    assert torch.equal(top_idx, ref_top_idx)
    assert torch.allclose(top_w, ref_top_w, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_ROUTER_TRITON, reason="CUDA and Triton required")
def test_triton_router_backward():
    torch.manual_seed(42)
    device = "cuda"
    B, D = 16, 32
    tree_depth = 4
    top_k = 2
    num_leaves = 16
    I = (1 << tree_depth) - 1

    x_flat = torch.randn(B, D, device=device, requires_grad=True)
    hyperplanes = torch.randn(I, D, device=device, requires_grad=True)
    biases = torch.randn(I, device=device, requires_grad=True)

    top_idx, top_w = triton_router_topk(x_flat, hyperplanes, biases, tree_depth, top_k, num_leaves)
    loss = (top_w * 2.0).sum()
    loss.backward()

    assert x_flat.grad is not None
    assert hyperplanes.grad is not None
    assert biases.grad is not None
    assert not torch.isnan(x_flat.grad).any()
    assert not torch.isnan(hyperplanes.grad).any()
    assert not torch.isnan(biases.grad).any()


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_ROUTER_TRITON, reason="CUDA and Triton required")
@pytest.mark.parametrize("B,D,tree_depth,top_k,num_leaves", [
    (16, 32, 4, 2, 16),
    (32, 64, 3, 2, 6),
    (8, 128, 5, 4, 24),
])
def test_triton_router_backward_parity(B, D, tree_depth, top_k, num_leaves):
    torch.manual_seed(123)
    device = "cuda"
    I = (1 << tree_depth) - 1

    x_flat = torch.randn(B, D, device=device, requires_grad=True)
    hyperplanes = torch.randn(I, D, device=device, requires_grad=True)
    biases = torch.randn(I, device=device, requires_grad=True)

    xr = x_flat.detach().clone().requires_grad_(True)
    hr = hyperplanes.detach().clone().requires_grad_(True)
    br = biases.detach().clone().requires_grad_(True)

    # Reference implementation using PyTorch autograd
    from affine_ai.core.ast_dag import ternarize
    W_route = ternarize(hr)
    node_logits = F.linear(xr.reshape(-1, xr.shape[-1]).float(), W_route.float(), br.float())
    logit_root = node_logits[:, 0:1]
    p_right = torch.sigmoid(logit_root * 2.0)
    current_level_probs = torch.cat([1.0 - p_right, p_right], dim=-1)
    B_nodes = node_logits.shape[0]
    for depth in range(1, tree_depth):
        start_node = (1 << depth) - 1
        num_nodes = 1 << depth
        level_logits = node_logits[:, start_node:start_node + num_nodes]
        pr = torch.sigmoid(level_logits * 2.0)
        pl = 1.0 - pr
        current_level_probs = torch.stack([current_level_probs * pl, current_level_probs * pr], dim=-1).view(B_nodes, -1)
    leaf_probs = current_level_probs
    routing_probs = leaf_probs[:, :num_leaves]
    routing_probs = routing_probs / routing_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    actual_top_k = min(top_k, num_leaves)
    top_vals, _ = torch.topk(routing_probs, k=actual_top_k, dim=-1)
    top_weights = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    grad_w = torch.randn_like(top_weights)
    top_weights.backward(grad_w)

    # Triton implementation
    top_idx, triton_top_w = triton_router_topk(x_flat, hyperplanes, biases, tree_depth, top_k, num_leaves)
    triton_top_w.backward(grad_w)

    assert torch.allclose(x_flat.grad, xr.grad, atol=1e-5), f"x_flat.grad mismatch: {(x_flat.grad - xr.grad).abs().max()}"
    assert torch.allclose(hyperplanes.grad, hr.grad, atol=1e-5), f"hyperplanes.grad mismatch: {(hyperplanes.grad - hr.grad).abs().max()}"
    assert torch.allclose(biases.grad, br.grad, atol=1e-5), f"biases.grad mismatch: {(biases.grad - br.grad).abs().max()}"

