"""
Tests for Triton Permutation Tree Kernel
"""

import pytest
import torch
import torch.nn.functional as F

from affine_ai.kernels import TRITON_AVAILABLE

try:
    if TRITON_AVAILABLE:
        from affine_ai.kernels.triton_tree import (
            triton_tree_perm_fwd,
            triton_tree_perm,
            TritonTreePermFunction,
        )
        HAS_TREE_TRITON = True
    else:
        HAS_TREE_TRITON = False
except Exception:
    HAS_TREE_TRITON = False


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TREE_TRITON, reason="CUDA and Triton required")
@pytest.mark.parametrize("B,D,K,P,Tk", [
    (17, 64, 8, 4, 2),
    (32, 32, 16, 2, 4),
    (64, 128, 8, 4, 2),
])
def test_triton_tree_perm_fwd_parity(B, D, K, P, Tk):
    torch.manual_seed(42)
    device = "cuda"

    r_in = torch.randn(B, D, device=device)
    w_perm = torch.randn(K, P, D, device=device)
    bias = torch.randn(K, D, device=device)
    perms = torch.stack([
        torch.stack([torch.randperm(D, device=device) for _ in range(P)])
        for _ in range(K)
    ]).to(torch.int32)
    top_idx = torch.randint(0, K, (B, Tk), device=device, dtype=torch.int32)
    top_w = torch.rand(B, Tk, device=device)
    top_w = top_w / top_w.sum(dim=-1, keepdim=True)

    out = triton_tree_perm_fwd(r_in, w_perm, bias, perms, top_idx, top_w)

    # Reference implementation
    ref = torch.zeros(B, D, device=device)
    for b in range(B):
        for tk in range(Tk):
            ki = top_idx[b, tk].item()
            tw = top_w[b, tk].item()
            acc = bias[ki].clone()
            for p in range(P):
                acc += w_perm[ki, p] * r_in[b, perms[ki, p].long()]
            acc = torch.clamp(acc, 0.0, 6.0)
            ref[b] += acc * tw

    assert torch.allclose(out, ref, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TREE_TRITON, reason="CUDA and Triton required")
def test_triton_tree_perm_backward():
    torch.manual_seed(42)
    device = "cuda"
    B, D, K, P, Tk = 16, 64, 8, 4, 2

    r_in = torch.randn(B, D, device=device, requires_grad=True)
    w_perm = torch.randn(K, P, D, device=device, requires_grad=True)
    bias = torch.randn(K, D, device=device, requires_grad=True)
    perms = torch.stack([
        torch.stack([torch.randperm(D, device=device) for _ in range(P)])
        for _ in range(K)
    ]).to(torch.int32)
    top_idx = torch.randint(0, K, (B, Tk), device=device, dtype=torch.int32)
    top_w = torch.rand(B, Tk, device=device, requires_grad=True)
    top_w = (top_w / top_w.sum(dim=-1, keepdim=True)).detach().requires_grad_(True)

    out = triton_tree_perm(r_in, w_perm, bias, perms, top_idx, top_w)
    loss = out.sum()
    loss.backward()

    assert r_in.grad is not None
    assert w_perm.grad is not None
    assert bias.grad is not None
    assert not torch.isnan(r_in.grad).any()
    assert not torch.isnan(w_perm.grad).any()
    assert not torch.isnan(bias.grad).any()
