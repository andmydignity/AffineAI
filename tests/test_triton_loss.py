import torch
import torch.nn.functional as F
import pytest
from affine_ai.kernels import triton_fused_linear_cross_entropy, TRITON_AVAILABLE


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA & Triton required")
def test_triton_fused_linear_cross_entropy_parity():
    N = 64
    D = 32
    V = 128

    hidden = torch.randn(N, D, device="cuda", requires_grad=True)
    weight = torch.randn(V, D, device="cuda", requires_grad=True)
    targets = torch.randint(0, V, (N,), device="cuda")

    # 1. PyTorch Reference
    logits = F.linear(hidden, weight)
    ref_loss = F.cross_entropy(logits, targets)
    ref_loss.backward()

    ref_dh = hidden.grad.clone()
    ref_dw = weight.grad.clone()

    hidden.grad.zero_()
    weight.grad.zero_()

    # 2. Triton Fused Loss
    triton_loss = triton_fused_linear_cross_entropy(hidden, weight, targets)
    triton_loss.backward()

    triton_dh = hidden.grad.clone()
    triton_dw = weight.grad.clone()

    # Verify Forward Parity
    assert torch.allclose(triton_loss, ref_loss, atol=2e-3, rtol=2e-3)
    # Verify Backward Gradient Parity
    assert torch.allclose(triton_dh, ref_dh, atol=2e-3, rtol=2e-3)
    assert torch.allclose(triton_dw, ref_dw, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA & Triton required")
def test_triton_fused_linear_cross_entropy_gradcheck():
    N = 17
    D = 32
    V = 65

    hidden = torch.randn(N, D, dtype=torch.float64, device="cuda", requires_grad=True)
    weight = torch.randn(V, D, dtype=torch.float64, device="cuda", requires_grad=True)
    targets = torch.randint(0, V, (N,), device="cuda")

    test = torch.autograd.gradcheck(triton_fused_linear_cross_entropy, (hidden, weight, targets), eps=1e-6, atol=1e-3)
    assert test, "Cross entropy gradcheck failed"


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA & Triton required")
def test_triton_fused_linear_cross_entropy_multi_dim():
    B, T, D, V = 2, 16, 48, 120
    hidden = torch.randn(B, T, D, device="cuda", requires_grad=True)
    weight = torch.randn(V, D, device="cuda", requires_grad=True)
    targets = torch.randint(0, V, (B, T), device="cuda")

    ref_loss = F.cross_entropy(F.linear(hidden, weight).view(-1, V), targets.view(-1))
    ref_loss.backward()
    ref_dh = hidden.grad.clone()

    hidden.grad.zero_()
    triton_loss = triton_fused_linear_cross_entropy(hidden, weight, targets)
    triton_loss.backward()

    assert torch.allclose(triton_loss, ref_loss, atol=2e-3, rtol=2e-3)
    assert torch.allclose(hidden.grad, ref_dh, atol=2e-3, rtol=2e-3)
