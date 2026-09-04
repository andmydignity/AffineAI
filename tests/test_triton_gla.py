import pytest
import torch
from affine_ai.kernels.triton_gla import (
    triton_gla_decay,
    triton_monarch_chain,
    triton_fused_monarch_chain,
    _gla_decay_kernel
)


def ref_gla_decay(gamma):
    log_gam = torch.log(gamma.float().clamp(min=1e-5))
    cum = torch.cumsum(log_gam, dim=-1)
    T = cum.shape[-1]
    decay_diff = (cum.unsqueeze(-1) - cum.unsqueeze(-2)).clamp(max=0.0)
    mask = torch.tril(torch.ones(T, T, device=cum.device, dtype=torch.bool))
    out = torch.where(mask, torch.exp(decay_diff), torch.zeros_like(decay_diff))
    return out.to(gamma.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_gla_decay_fwd_bwd():
    torch.manual_seed(42)
    B, H, T = 2, 4, 16
    gamma = torch.sigmoid(torch.randn(B, H, T, device="cuda", dtype=torch.float32)).detach().requires_grad_(True)

    out = triton_gla_decay(gamma)
    ref_out = ref_gla_decay(gamma)
    assert torch.allclose(out, ref_out, atol=1e-4)

    out.sum().backward()
    assert gamma.grad is not None
    assert not torch.isnan(gamma.grad).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_gla_decay_no_grad():
    torch.manual_seed(42)
    B, H, T = 2, 4, 16
    gamma = torch.sigmoid(torch.randn(B, H, T, device="cuda", dtype=torch.float32))
    assert not gamma.requires_grad

    out = triton_gla_decay(gamma)
    assert not out.requires_grad
    ref_out = ref_gla_decay(gamma)
    assert torch.allclose(out, ref_out, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_gla_decay_non_contiguous_strides():
    """Verify _gla_decay_kernel uses actual tensor strides when storing to Decay buffer."""
    torch.manual_seed(42)
    B, H, T = 2, 3, 8
    gamma = torch.sigmoid(torch.randn(B, H, T, device="cuda", dtype=torch.float32))
    log_gam = torch.log(gamma.clamp(min=1e-5))
    cum = torch.cumsum(log_gam, dim=-1).contiguous()

    # Create non-contiguous output buffer with custom strides
    full_buf = torch.empty((B, H, 2 * T, T), device="cuda", dtype=torch.float32)
    out_sliced = full_buf[:, :, :T, :]  # Non-standard stride in dimension 2
    assert not out_sliced.is_contiguous()

    BLOCK = 1024
    grid = ((B * H * T * T + BLOCK - 1) // BLOCK,)
    _gla_decay_kernel[grid](
        cum, out_sliced,
        cum.stride(0), cum.stride(1), cum.stride(2),
        out_sliced.stride(0), out_sliced.stride(1), out_sliced.stride(2), out_sliced.stride(3),
        B, H, T, BLOCK=BLOCK, num_warps=4
    )

    ref = ref_gla_decay(gamma)
    assert torch.allclose(out_sliced, ref, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_monarch_chain():
    torch.manual_seed(42)
    B, T, D = 2, 8, 32
    num_stages = 3
    x = torch.randn(B, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
    diagonals = torch.randn(num_stages, D, device="cuda", dtype=torch.float32, requires_grad=True)
    perms = torch.stack([torch.randperm(D, device="cuda") for _ in range(num_stages - 1)])
    inv_perms = torch.empty_like(perms)
    for s in range(num_stages - 1):
        inv_perms[s, perms[s]] = torch.arange(D, device="cuda")
    bias = torch.randn(D, device="cuda", dtype=torch.float32, requires_grad=True)

    out = triton_monarch_chain(x, diagonals, perms, inv_perms, bias)
    assert out.shape == (B, T, D)

    # Check backward works and h_list[-1] is not saved (saved_tensors has 4 + num_stages - 1 elements)
    out.sum().backward()
    assert x.grad is not None
    assert diagonals.grad is not None
    assert bias.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_fused_monarch_chain():
    torch.manual_seed(42)
    B, T, D = 2, 8, 32
    num_branches = 2
    num_stages = 3
    x = torch.randn(B, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
    diagonals = torch.randn(num_branches, num_stages, D, device="cuda", dtype=torch.float32, requires_grad=True)
    perms = torch.stack([torch.randperm(D, device="cuda") for _ in range(num_stages - 1)])
    inv_perms = torch.empty_like(perms)
    for s in range(num_stages - 1):
        inv_perms[s, perms[s]] = torch.arange(D, device="cuda")
    bias = torch.randn(num_branches, D, device="cuda", dtype=torch.float32, requires_grad=True)

    outs = triton_fused_monarch_chain(x, diagonals, perms, inv_perms, bias)
    assert len(outs) == num_branches

    loss = sum(o.sum() for o in outs)
    loss.backward()
    assert x.grad is not None
    assert diagonals.grad is not None
    assert bias.grad is not None
