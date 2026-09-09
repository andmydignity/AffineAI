import pytest
import torch
from affine_ai.kernels.triton_perm_proj import triton_fused_perm_proj
from affine_ai.core.associative import PermutationProjection, FusedPermutationProjection


def test_triton_fused_perm_proj_forward_and_backward():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    N, D, M, P = 128, 64, 4, 4
    x = torch.randn(N, D, device="cuda", dtype=torch.float32, requires_grad=True)
    perms = torch.stack([torch.randperm(D, device="cuda", dtype=torch.long) for _ in range(P)])
    inv_perms = torch.empty_like(perms)
    for p in range(P):
        inv_perms[p, perms[p]] = torch.arange(D, device="cuda")
    w = torch.randn(M, P, D, device="cuda", dtype=torch.float32, requires_grad=True)
    biases = torch.randn(M, D, device="cuda", dtype=torch.float32, requires_grad=True)

    # Reference PyTorch
    x_gathered = torch.gather(
        x.unsqueeze(1).expand(-1, P, -1),
        dim=-1,
        index=perms.unsqueeze(0).expand(N, -1, -1)
    )
    out_ref = (x_gathered.unsqueeze(0) * w.unsqueeze(1)).sum(dim=2) + biases.unsqueeze(1)
    loss_ref = (out_ref * 0.5).sum()
    loss_ref.backward()

    # Triton
    x_tri = x.detach().clone().requires_grad_(True)
    w_tri = w.detach().clone().requires_grad_(True)
    biases_tri = biases.detach().clone().requires_grad_(True)

    out_tri = triton_fused_perm_proj(x_tri, w_tri, perms, inv_perms, biases_tri)
    loss_tri = (out_tri * 0.5).sum()
    loss_tri.backward()

    # Verify parity
    diff_fwd = (out_ref - out_tri).abs().max().item()
    diff_gx = (x.grad - x_tri.grad).abs().max().item()
    diff_gw = (w.grad - w_tri.grad).abs().max().item()
    diff_gb = (biases.grad - biases_tri.grad).abs().max().item()

    assert diff_fwd < 1e-4, f"Forward diff too high: {diff_fwd}"
    assert diff_gx < 1e-4, f"Grad x diff too high: {diff_gx}"
    assert diff_gw < 1e-4, f"Grad w diff too high: {diff_gw}"
    assert diff_gb < 1e-4, f"Grad b diff too high: {diff_gb}"


def test_fused_perm_proj_module_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    B, T, D = 4, 32, 64
    x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    mod_single = PermutationProjection(dim=D, num_perms=4, dtype=torch.bfloat16).cuda()
    out_single = mod_single(x)
    assert out_single.shape == (B, T, D)
    out_single.sum().backward()
    assert mod_single.latent_w.grad is not None

    mod_fused = FusedPermutationProjection(dim=D, num_branches=4, num_perms=4, dtype=torch.bfloat16).cuda()
    outs = mod_fused(x)
    assert len(outs) == 4
    for o in outs:
        assert o.shape == (B, T, D)
    sum(outs).sum().backward()
    assert mod_fused.latent_w.grad is not None
