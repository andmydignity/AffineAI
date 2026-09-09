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
@pytest.mark.parametrize("M", [1, 16, 64, 128])
def test_triton_int8_imma_parity(M):
    from affine_ai.kernels.triton_int8_imma import triton_int8_imma_linear

    device = torch.device("cuda")
    K, N = 96, 64
    torch.manual_seed(42)

    x = torch.randn(M, K, device=device, requires_grad=True)
    weight = torch.randn(N, K, device=device, requires_grad=True)
    bias = torch.randn(N, device=device, requires_grad=True)

    out = triton_int8_imma_linear(x, weight, bias)
    assert out.shape == (M, N)

    # PyTorch reference
    sx = (x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5).float() / 127.0).squeeze(-1)
    x_int8 = (x / sx.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    sw = (weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5).float() / 127.0).squeeze(-1)
    w_int8 = (weight / sw.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    ref_out = (torch.matmul(x_int8.float(), w_int8.float().t()) * sx.unsqueeze(-1) * sw.unsqueeze(0)) + bias

    diff = (out - ref_out).abs().max().item()
    assert diff < 1e-3, f"INT8 IMMA forward parity mismatch: max diff={diff}"

    # Verify backward pass mathematical parity
    loss = out.sum()
    loss.backward()

    ref_gx = torch.matmul(torch.ones_like(ref_out), weight)
    ref_gw = torch.matmul(torch.ones_like(ref_out).t(), x)
    ref_gb = torch.ones_like(ref_out).sum(dim=0)

    assert torch.allclose(x.grad, ref_gx, atol=1e-4), "x.grad mismatch"
    assert torch.allclose(weight.grad, ref_gw, atol=1e-4), "weight.grad mismatch"
    assert torch.allclose(bias.grad, ref_gb, atol=1e-4), "bias.grad mismatch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for hardware ALU tests")
@pytest.mark.parametrize("M", [1, 16, 64])
def test_triton_bitlinear_swiglu_no_stalls(M):
    from affine_ai.kernels.triton_bitlinear import triton_bitlinear_swiglu

    device = torch.device("cuda")
    D, hidden = 96, 192
    torch.manual_seed(42)

    x = torch.randn(M, D, device=device, requires_grad=True)
    w_gv = torch.randn(2 * hidden, D, device=device, requires_grad=True)
    w_d = torch.randn(D, hidden, device=device, requires_grad=True)

    out = triton_bitlinear_swiglu(x, w_gv, w_d)
    assert out.shape == (M, D)

    # PyTorch reference with ternary quantization
    gamma_gv = w_gv.abs().mean().clamp(min=1e-5)
    gamma_d = w_d.abs().mean().clamp(min=1e-5)
    w_gv_q = torch.round(torch.clamp(w_gv / gamma_gv, -1.0, 1.0))
    w_d_q = torch.round(torch.clamp(w_d / gamma_d, -1.0, 1.0))

    gv = torch.matmul(x, w_gv_q.t()) * gamma_gv
    g, v = gv[:, :hidden], gv[:, hidden:]
    h_act = (g.sigmoid() * g) * v
    ref = torch.matmul(h_act, w_d_q.t()) * gamma_d

    assert torch.allclose(out, ref, atol=0.5, rtol=1e-3), f"Bitlinear SwiGLU M={M} parity mismatch"

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert w_gv.grad is not None
    assert w_d.grad is not None
