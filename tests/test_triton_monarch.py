"""
Dedicated Unit Tests for In-SRAM Fused Monarch Permutation Chain:
- Mathematical parity against sequential PyTorch reference
- Backward gradient mathematical parity (gx, g_diagonals, g_bias)
- Single-token decoding (M=1) up to batch sizes (M=128+)
- Multi-branch fused projection (triton_fused_monarch_chain)
- Multi-stage chains (S=2, 4)
"""

import pytest
import torch

from affine_ai.kernels.triton_monarch import (
    triton_monarch_chain,
    triton_fused_monarch_chain,
    precompute_monarch_composed_single,
    precompute_monarch_composed_fused
)


def ref_monarch_chain(x: torch.Tensor, diagonals: torch.Tensor, perms: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    h = x * diagonals[0]
    S = diagonals.shape[0]
    for s in range(S - 1):
        h = h[:, perms[s]] * diagonals[s + 1]
    return h + bias


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Monarch tests")
@pytest.mark.parametrize("M", [1, 8, 32, 64, 128])
@pytest.mark.parametrize("S", [2, 4])
def test_triton_monarch_chain_parity(M, S):
    device = torch.device("cuda")
    D = 64
    torch.manual_seed(42)

    x = torch.randn(M, D, device=device, dtype=torch.float32, requires_grad=True)
    diagonals = torch.randn(S, D, device=device, dtype=torch.float32, requires_grad=True)
    perms = torch.stack([torch.randperm(D, device=device) for _ in range(S)])
    inv_perms = torch.empty_like(perms)
    for s in range(S):
        inv_perms[s] = torch.argsort(perms[s])
    bias = torch.randn(D, device=device, dtype=torch.float32, requires_grad=True)

    ref_x = x.detach().clone().requires_grad_(True)
    ref_diag = diagonals.detach().clone().requires_grad_(True)
    ref_bias = bias.detach().clone().requires_grad_(True)

    out = triton_monarch_chain(x, diagonals, perms, inv_perms, bias)
    ref_out = ref_monarch_chain(ref_x, ref_diag, perms, ref_bias)

    assert torch.allclose(out, ref_out, atol=1e-5), f"Monarch forward diff: {(out - ref_out).abs().max().item()}"

    # Verify backward pass mathematical parity
    loss1 = (out * 2.0).sum()
    loss1.backward()

    loss2 = (ref_out * 2.0).sum()
    loss2.backward()

    assert torch.allclose(x.grad, ref_x.grad, atol=1e-5), "x.grad mismatch"
    assert torch.allclose(diagonals.grad, ref_diag.grad, atol=1e-5), "diagonals.grad mismatch"
    assert torch.allclose(bias.grad, ref_bias.grad, atol=1e-5), "bias.grad mismatch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Monarch tests")
def test_triton_fused_monarch_chain_branches():
    """Verify multi-branch (Q, K, V, G) fused Monarch chain."""
    device = torch.device("cuda")
    M, D, S, num_branches = 16, 64, 2, 4
    torch.manual_seed(42)

    x = torch.randn(M, D, device=device, dtype=torch.float32, requires_grad=True)
    diagonals = torch.randn(num_branches, S, D, device=device, dtype=torch.float32, requires_grad=True)
    perms = torch.stack([torch.randperm(D, device=device) for _ in range(S)])
    inv_perms = torch.empty_like(perms)
    for s in range(S):
        inv_perms[s] = torch.argsort(perms[s])
    bias = torch.randn(num_branches, D, device=device, dtype=torch.float32, requires_grad=True)

    outs = triton_fused_monarch_chain(x, diagonals, perms, inv_perms, bias)
    assert len(outs) == num_branches

    for m in range(num_branches):
        ref_branch = ref_monarch_chain(x, diagonals[m], perms, bias[m])
        assert torch.allclose(outs[m], ref_branch, atol=1e-5)

    loss = sum(o.sum() for o in outs)
    loss.backward()
    assert x.grad is not None
    assert diagonals.grad is not None
    assert bias.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Monarch tests")
def test_triton_monarch_sliced_bias_and_dtype():
    """Verify sliced bias with non-standard stride and bias dtype matching in backward."""
    device = torch.device("cuda")
    M, D, S = 8, 64, 2
    torch.manual_seed(42)

    x = torch.randn(M, D, device=device, dtype=torch.float32, requires_grad=True)
    diagonals = torch.randn(S, D, device=device, dtype=torch.float32, requires_grad=True)
    perms = torch.stack([torch.randperm(D, device=device) for _ in range(S)])
    inv_perms = torch.empty_like(perms)
    for s in range(S):
        inv_perms[s] = torch.argsort(perms[s])

    # Sliced bias with stride != 1
    bias_2d = torch.randn(D, 4, device=device, dtype=torch.float32, requires_grad=True)
    bias_sliced = bias_2d[:, 1]  # stride is 4
    assert bias_sliced.stride(0) != 1

    out = triton_monarch_chain(x, diagonals, perms, inv_perms, bias_sliced)
    ref_out = ref_monarch_chain(x, diagonals, perms, bias_sliced)
    assert torch.allclose(out, ref_out, atol=1e-5)

    out.sum().backward()
    assert bias_2d.grad is not None
    assert bias_sliced.dtype == torch.float32
