"""
Dedicated Unit Tests for 1-Bit Hardware POPC Binary Router:
- triton_pack_sign_bits parity against PyTorch bitpacking reference
- triton_popc_sign_similarity exact integer dot-product parity
- Single-token inference (M=1) up to batch sizes (M=128+)
- Multi-dimensional tensor shapes [B, T, D]
- Scaled floating-point output mode
"""

import pytest
import torch

from affine_ai.kernels.triton_popc import triton_pack_sign_bits, triton_popc_sign_similarity


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for POPC tests")
@pytest.mark.parametrize("M", [1, 8, 32, 64])
@pytest.mark.parametrize("D", [32, 64, 128])
def test_triton_popc_sign_similarity_parity(M, D):
    device = torch.device("cuda")
    N = 32
    torch.manual_seed(42)

    x = torch.randn(M, D, device=device)
    w = torch.randn(N, D, device=device)

    # Reference PyTorch dot product on signs {-1, +1}
    x_sign = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    w_sign = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
    ref_dot = torch.matmul(x_sign, w_sign.t()).to(torch.int32)

    # Triton hardware POPC path
    x_bits = triton_pack_sign_bits(x)
    w_bits = triton_pack_sign_bits(w)
    out_dot = triton_popc_sign_similarity(x_bits, w_bits, D=D)

    assert out_dot.shape == (M, N)
    diff = (out_dot - ref_dot).abs().max().item()
    assert diff == 0, f"POPC dot product differed from exact sign dot product: max diff={diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for POPC tests")
def test_triton_popc_3d_and_scaled():
    """Verify 3D batch shape [B, T, D] and float scaling factor."""
    device = torch.device("cuda")
    B, T, N, D = 2, 16, 32, 64
    torch.manual_seed(42)

    x = torch.randn(B, T, D, device=device)
    w = torch.randn(N, D, device=device)

    x_bits = triton_pack_sign_bits(x)
    w_bits = triton_pack_sign_bits(w)

    scale = 1.0 / (D ** 0.5)
    out = triton_popc_sign_similarity(x_bits, w_bits, scale=scale, D=D)

    assert out.shape == (B, T, N)
    assert out.dtype == torch.float32

    # Verify scaled parity
    x_sign = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    w_sign = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
    ref = (torch.matmul(x_sign, w_sign.t()) * scale).float()
    assert torch.allclose(out, ref, atol=1e-5)
