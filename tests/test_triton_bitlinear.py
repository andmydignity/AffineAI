import pytest
import torch
import torch.nn.functional as F

from affine_ai.kernels.triton_bitlinear import triton_bitlinear_swiglu, TritonBitLinearSwiGLUFunction


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton bitlinear tests")
@pytest.mark.parametrize("M", [1, 16, 64, 128])
def test_triton_swiglu_parity_and_gradients(M):
    torch.manual_seed(42)
    K, N = 64, 128
    dtype = torch.float32
    x = torch.randn(M, K, device="cuda", dtype=dtype, requires_grad=True)
    w_gv = torch.randn(2 * N, K, device="cuda", dtype=dtype, requires_grad=True)
    w_d = torch.randn(K, N, device="cuda", dtype=dtype, requires_grad=True)

    out = triton_bitlinear_swiglu(x, w_gv, w_d)
    assert out.shape == (M, K)

    # Reference calculation
    gamma_gv = w_gv.abs().mean().clamp(min=1e-5)
    gamma_d = w_d.abs().mean().clamp(min=1e-5)
    w_gv_q = torch.round(torch.clamp(w_gv / gamma_gv, -1.0, 1.0))
    w_d_q = torch.round(torch.clamp(w_d / gamma_d, -1.0, 1.0))

    gv = torch.matmul(x, w_gv_q.t()) * gamma_gv
    g = gv[:, :N]
    v = gv[:, N:]
    h_act = (g.sigmoid() * g) * v
    ref = torch.matmul(h_act, w_d_q.t()) * gamma_d

    assert torch.allclose(out, ref, atol=0.5, rtol=1e-3)

    # Backward pass test
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert w_gv.grad is not None
    assert w_d.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(w_gv.grad).any()
    assert not torch.isnan(w_d.grad).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton bitlinear tests")
def test_triton_swiglu_autograd_context_no_redundant_h_act_issue32():
    """Verify Issue 32: Autograd context does not save redundant h_act in ctx.saved_tensors."""
    torch.manual_seed(42)
    M, K, N = 16, 64, 128
    x = torch.randn(M, K, device="cuda", dtype=torch.float32, requires_grad=True)
    w_gv = torch.randn(2 * N, K, device="cuda", dtype=torch.float32, requires_grad=True)
    w_d = torch.randn(K, N, device="cuda", dtype=torch.float32, requires_grad=True)

    # Run forward via autograd Function
    out = TritonBitLinearSwiGLUFunction.apply(x, w_gv, w_d, None, None)
    
    # Check saved tensors in the grad_fn
    saved = out.grad_fn.saved_tensors
    # Saved tensors are: x_flat, w_gv_q, w_d_q, gv, gamma_gv, gamma_d
    assert len(saved) == 6, f"Expected 6 saved tensors (no redundant h_act), got {len(saved)}"
    shapes = [t.shape for t in saved]
    # Verify no tensor has shape (M, N) which would be h_act
    assert (M, N) not in shapes, "h_act tensor was redundantly saved in autograd context!"

    # Verify backward succeeds by recalculating h_act from gv
    loss = out.sum()
    loss.backward()
    assert w_d.grad is not None
    assert not torch.isnan(w_d.grad).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton bitlinear tests")
def test_triton_swiglu_3d_input():
    torch.manual_seed(42)
    B, T, K, N = 2, 8, 64, 128
    x = torch.randn(B, T, K, device="cuda", dtype=torch.float32, requires_grad=True)
    w_gv = torch.randn(2 * N, K, device="cuda", dtype=torch.float32, requires_grad=True)
    w_d = torch.randn(K, N, device="cuda", dtype=torch.float32, requires_grad=True)

    out = triton_bitlinear_swiglu(x, w_gv, w_d)
    assert out.shape == (B, T, K)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (B, T, K)
