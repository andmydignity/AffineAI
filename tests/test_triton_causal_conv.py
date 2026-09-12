import pytest
import torch
from affine_ai.kernels.triton_causal_conv import triton_causal_conv1d, TritonCausalConv1dFunction


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("K", [4, 8])
def test_triton_causal_conv1d_forward_parity(dtype, has_bias, K):
    torch.manual_seed(42)
    B, T, D = 4, 64, 32
    x = torch.randn(B, T, D, device="cuda", dtype=dtype)
    w = torch.randn(D, K, device="cuda", dtype=dtype)
    b = torch.randn(D, device="cuda", dtype=dtype) if has_bias else None

    # PyTorch reference
    x_pad = torch.nn.functional.pad(x.transpose(1, 2), (K - 1, 0))
    w_2d = w.unsqueeze(1)
    ref = torch.nn.functional.conv1d(x_pad, w_2d, b, groups=D).transpose(1, 2)

    # Triton
    out = triton_causal_conv1d(x, w, b)

    diff = (out - ref).abs().max().item()
    tol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    assert diff < tol, f"Forward diff {diff} exceeds tol {tol}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_bias", [True, False])
def test_triton_causal_conv1d_backward_parity(dtype, has_bias):
    torch.manual_seed(42)
    B, T, D, K = 4, 32, 16, 8

    x1 = torch.randn(B, T, D, device="cuda", dtype=dtype, requires_grad=True)
    w1 = torch.randn(D, K, device="cuda", dtype=dtype, requires_grad=True)
    b1 = torch.randn(D, device="cuda", dtype=dtype, requires_grad=True) if has_bias else None

    x2 = x1.detach().clone().requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)
    b2 = b1.detach().clone().requires_grad_(True) if has_bias else None

    # PyTorch
    x_pad = torch.nn.functional.pad(x1.transpose(1, 2), (K - 1, 0))
    w_2d = w1.unsqueeze(1)
    y1 = torch.nn.functional.conv1d(x_pad, w_2d, b1, groups=D).transpose(1, 2)
    loss1 = (y1 * 0.5).sum()
    loss1.backward()

    # Triton
    y2 = triton_causal_conv1d(x2, w2, b2)
    loss2 = (y2 * 0.5).sum()
    loss2.backward()

    tol = 5e-2 if dtype == torch.bfloat16 else 1e-3
    assert (x1.grad - x2.grad).abs().max().item() < tol
    assert (w1.grad - w2.grad).abs().max().item() < tol
    if has_bias:
        assert (b1.grad - b2.grad).abs().max().item() < tol
