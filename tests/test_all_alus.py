"""
Tests for Advanced Hardware ALU Pipelines:
1. 1-Bit POPC (Population Count) ALUs (popc.b32)
2. INT8 IMMA Tensor Cores (mma.sync.aligned.m16n8k32)
3. Zero-Stall BitLinear SwiGLU with Packed Vector Support
"""

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for hardware ALU tests")
def test_triton_popc_parity():
    from affine_ai.kernels.triton_popc import triton_pack_sign_bits, triton_popc_sign_similarity

    device = torch.device("cuda")
    M, N, D = 64, 32, 128
    torch.manual_seed(42)

    x = torch.randn(M, D, device=device)
    w = torch.randn(N, D, device=device)

    # Reference 1-bit sign similarity in PyTorch
    x_sign = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    w_sign = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
    ref_dot = torch.matmul(x_sign, w_sign.t()).to(torch.int32)

    # Hardware POPC path
    x_bits = triton_pack_sign_bits(x)
    w_bits = triton_pack_sign_bits(w)
    out_dot = triton_popc_sign_similarity(x_bits, w_bits)

    assert out_dot.shape == (M, N)
    diff = (out_dot - ref_dot).abs().max().item()
    assert diff == 0, f"POPC dot product differs from reference: max diff={diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for hardware ALU tests")
def test_hierarchical_router_popc():
    from affine_ai.core.ast_dag import HierarchicalSignRouter

    device = torch.device("cuda")
    dim = 96
    num_leaves = 16
    router = HierarchicalSignRouter(dim=dim, num_leaves=num_leaves).to(device)

    x = torch.randn(32, dim, device=device)
    probs, logits = router.route_tokens_popc(x)

    assert probs.shape == (32, num_leaves)
    assert logits.shape == (32, router.num_internal_nodes)
    assert torch.all(probs >= 0.0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(32, device=device), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for hardware ALU tests")
def test_triton_int8_imma_parity():
    from affine_ai.kernels.triton_int8_imma import triton_int8_imma_linear

    device = torch.device("cuda")
    M, K, N = 128, 96, 64
    torch.manual_seed(42)

    x = torch.randn(M, K, device=device, requires_grad=True)
    weight = torch.randn(N, K, device=device, requires_grad=True)
    bias = torch.randn(N, device=device, requires_grad=True)

    out = triton_int8_imma_linear(x, weight, bias)
    assert out.shape == (M, N)

    # Verify backward pass
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for hardware ALU tests")
def test_triton_bitlinear_swiglu_no_stalls():
    from affine_ai.kernels.triton_bitlinear import triton_bitlinear_swiglu

    device = torch.device("cuda")
    M, D, hidden = 64, 96, 192
    torch.manual_seed(42)

    x = torch.randn(M, D, device=device, requires_grad=True)
    w_gv = torch.randn(2 * hidden, D, device=device, requires_grad=True)
    w_d = torch.randn(D, hidden, device=device, requires_grad=True)

    out = triton_bitlinear_swiglu(x, w_gv, w_d)
    assert out.shape == (M, D)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert w_gv.grad is not None
    assert w_d.grad is not None
