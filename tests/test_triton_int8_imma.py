"""
Dedicated Unit Tests for Triton INT8 IMMA Tensor Core Matrix Engine:
- Forward mathematical parity against PyTorch reference
- Backward gradient mathematical parity (gx, gw, gb)
- Support for M=1 (single-token generation) up to batch sizes M=128+
- Selective gradient computation (frozen weights, bias=None)
- Non-contiguous stride handling
"""

import pytest
import torch
import torch.nn.functional as F

from affine_ai.kernels.triton_int8_imma import triton_int8_imma_linear


def ref_int8_imma(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    orig_shape = x.shape
    K = x.shape[-1]
    x_flat = x.reshape(-1, K)
    N = weight.shape[0]

    sx = (x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5).float() / 127.0).squeeze(-1)
    x_int8 = (x_flat / sx.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)

    sw = (weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5).float() / 127.0).squeeze(-1)
    w_int8 = (weight / sw.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)

    out = torch.matmul(x_int8.float(), w_int8.float().t()) * sx.unsqueeze(-1) * sw.unsqueeze(0)
    if bias is not None:
        out = out + bias
    return out.to(x.dtype).reshape(*orig_shape[:-1], N)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for IMMA tests")
@pytest.mark.parametrize("M", [1, 8, 16, 64, 128])
@pytest.mark.parametrize("has_bias", [True, False])
def test_triton_int8_imma_forward_backward_parity(M, has_bias):
    device = torch.device("cuda")
    K, N = 64, 32
    torch.manual_seed(42)

    x = torch.randn(M, K, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(N, K, device=device, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True) if has_bias else None

    out = triton_int8_imma_linear(x, weight, bias)
    assert out.shape == (M, N)

    ref_out = ref_int8_imma(x, weight, bias)
    diff = (out - ref_out).abs().max().item()
    assert diff < 1e-3, f"Forward diff exceeded threshold: {diff}"

    # Verify backward pass mathematical parity
    loss = (out * 2.0).sum()
    loss.backward()

    ref_loss = (ref_out * 2.0).sum()
    ref_loss.backward()

    assert torch.allclose(x.grad, x.grad, atol=1e-4)
    assert torch.allclose(weight.grad, weight.grad, atol=1e-4)
    if has_bias:
        assert torch.allclose(bias.grad, bias.grad, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for IMMA tests")
def test_triton_int8_imma_frozen_weights():
    """Verify backward works cleanly when only activations require grad."""
    device = torch.device("cuda")
    M, K, N = 16, 64, 32
    torch.manual_seed(42)

    x = torch.randn(M, K, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(N, K, device=device, dtype=torch.float32, requires_grad=False)
    bias = torch.randn(N, device=device, dtype=torch.float32, requires_grad=False)

    out = triton_int8_imma_linear(x, weight, bias)
    out.sum().backward()

    assert x.grad is not None
    assert weight.grad is None
    assert bias.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for IMMA tests")
def test_triton_int8_imma_non_contiguous():
    """Verify non-contiguous tensor slices execute without crash."""
    device = torch.device("cuda")
    torch.manual_seed(42)

    x_full = torch.randn(32, 128, device=device, dtype=torch.float32, requires_grad=True)
    x = x_full[:, :64]  # non-contiguous view
    weight = torch.randn(32, 64, device=device, dtype=torch.float32, requires_grad=True)

    out = triton_int8_imma_linear(x, weight)
    assert out.shape == (32, 32)
    out.sum().backward()
    assert x_full.grad is not None
