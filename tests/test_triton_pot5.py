import pytest
import torch
import torch.nn.functional as F

from affine_ai.core.ast_dag import pot5_quantize
from affine_ai.kernels.triton_pot5 import (
    triton_pot5_linear,
    triton_pot5_int8_linear,
    triton_pot5_fused_swiglu,
    triton_pot5_bitpacked_linear,
    Triton5StatePOTLinear,
    Triton5StatePOTBitpackedLinear,
    Triton5StatePOTBitpackedResidualLinear,
    pack_pot5_gpu_3bitplane,
    unpack_pot5_gpu_3bitplane,
    _pot5_thresholds,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton POT-5 tests")
def test_triton_pot5_linear_not_dead_layer_issue34():
    """Verify Issue 34: Triton5StatePOTLinear initializes alpha dynamically and is not a 100% dead layer."""
    torch.manual_seed(42)
    layer = Triton5StatePOTLinear(in_features=128, out_features=64, dtype=torch.float32).cuda()
    
    # Check that alpha is NOT hardcoded to 1.0
    assert layer.alpha.item() != 1.0, f"alpha is still hardcoded to 1.0!"
    expected_alpha = (layer.weight.detach().abs().mean() * 1.4).item()
    assert abs(layer.alpha.item() - expected_alpha) < 1e-4

    x = torch.randn(4, 16, 128, device="cuda", dtype=torch.float32)
    out = layer(x)
    assert out.shape == (4, 16, 64)
    # Output must not be all zeros (layer must not be dead)
    assert out.abs().sum().item() > 0.0, "Layer output is all zeros (dead layer)!"

    # Test backward pass
    loss = out.sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert layer.weight.grad.abs().sum().item() > 0.0, "Weight gradient is all zeros!"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton POT-5 tests")
def test_triton_pot5_int8_linear_no_slicing_clamp_issue35():
    """Verify Issue 35: triton_pot5_int8_linear does not clamp/slice K > 16384."""
    torch.manual_seed(42)
    # Test with K = 16416 (> 16384, multiple of 32)
    K = 16416
    M, N = 2, 32
    x = torch.randn(M, K, device="cuda", dtype=torch.float32)
    w = torch.randn(N, K, device="cuda", dtype=torch.float32)

    out = triton_pot5_int8_linear(x, w)
    assert out.shape == (M, N)
    assert not torch.isnan(out).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton POT-5 tests")
def test_pot5_packing_threshold_alignment_issue36_37():
    """Verify Issues 36 & 37: pack_pot5_gpu_3bitplane computes std before padding and aligns thresholds."""
    torch.manual_seed(42)
    N, K = 64, 100  # Non-multiple of 32 to test padding effect on std
    w = torch.randn(N, K, device="cuda", dtype=torch.float32)

    # 1. Without alpha, matches pot5_quantize reference bit-for-bit
    w_nz, w_mag, w_sign, alpha, K_orig = pack_pot5_gpu_3bitplane(w)
    assert K_orig == K
    w_rec = unpack_pot5_gpu_3bitplane(w_nz, w_mag, w_sign, alpha, K_orig)
    w_ref = pot5_quantize(w)
    assert torch.equal(w_rec, w_ref), "Unpacked weights deviate from pot5_quantize reference!"

    # 2. With explicit alpha provided, thresholds match _pot5_thresholds (0.25*alpha, 0.75*alpha)
    target_alpha = torch.tensor([1.25], device="cuda", dtype=torch.float32)
    w_nz_a, w_mag_a, w_sign_a, out_alpha, _ = pack_pot5_gpu_3bitplane(w, alpha=target_alpha)
    assert torch.equal(out_alpha, target_alpha)
    w_rec_a = unpack_pot5_gpu_3bitplane(w_nz_a, w_mag_a, w_sign_a, target_alpha, K_orig)

    t_low, t_high = _pot5_thresholds(target_alpha)
    w_abs = w.abs()
    w_sign_t = w.sign()
    w_full = torch.where(w_abs > t_high, w_sign_t, torch.zeros_like(w))
    w_half = torch.where((w_abs > t_low) & (w_abs <= t_high), w_sign_t, torch.zeros_like(w))
    expected_w = (w_full + w_half * 0.5) * target_alpha
    assert torch.allclose(w_rec_a, expected_w, atol=1e-5), "Packed weights with alpha do not match _pot5_thresholds!"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton POT-5 tests")
def test_pot5_bitpacked_residual_tiled_outliers_issue38():
    """Verify Issue 38: Triton5StatePOTBitpackedResidualLinear runs tiled outlier accumulation."""
    torch.manual_seed(42)
    in_features = 128
    out_features = 64
    w = torch.randn(out_features, in_features, device="cuda", dtype=torch.float32)

    # 10% outliers to verify chunking (> 4096 would be large, but test chunking logic)
    num_outliers = 100
    outlier_idx = torch.randperm(out_features * in_features, device="cuda")[:num_outliers]
    outlier_vals = torch.randn(num_outliers, device="cuda", dtype=torch.float32)

    w_core = w.clone().view(-1)
    w_core[outlier_idx] = 0.0
    w_core = w_core.view(out_features, in_features)
    w_nz, w_mag, w_sign, alpha, _ = pack_pot5_gpu_3bitplane(w_core)

    layer = Triton5StatePOTBitpackedResidualLinear(
        in_features=in_features,
        out_features=out_features,
        w_nz_bits=w_nz,
        w_mag_bits=w_mag,
        w_sign_bits=w_sign,
        alpha=alpha,
        outlier_indices=outlier_idx,
        outlier_values=outlier_vals,
        dtype=torch.float32,
    ).cuda()

    x = torch.randn(8, in_features, device="cuda", dtype=torch.float32)
    out = layer(x)
    assert out.shape == (8, out_features)

    # Compare with dense reference: x @ layer.weight.T
    ref_out = F.linear(x, layer.weight)
    assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton POT-5 tests")
def test_pot5_bitpacked_gemm_coalesced_layout_issue39():
    """Verify Issue 39: Coalesced bitpacked GEMM memory layout gives exact parity with contiguous words."""
    torch.manual_seed(42)
    N, K = 64, 128
    w = torch.randn(N, K, device="cuda", dtype=torch.float32)
    x = torch.randn(4, K, device="cuda", dtype=torch.float32)

    w_nz, w_mag, w_sign, alpha, K_orig = pack_pot5_gpu_3bitplane(w)

    # Run with standard row-major input
    out_std = triton_pot5_bitpacked_linear(x, w_nz, w_mag, w_sign, alpha, K_orig)

    # Run with explicit Fortran/coalesced layout
    w_nz_c = w_nz.transpose(0, 1).contiguous().transpose(0, 1)
    w_mag_c = w_mag.transpose(0, 1).contiguous().transpose(0, 1)
    w_sign_c = w_sign.transpose(0, 1).contiguous().transpose(0, 1)
    assert w_nz_c.stride(0) == 1

    out_coalesced = triton_pot5_bitpacked_linear(x, w_nz_c, w_mag_c, w_sign_c, alpha, K_orig)
    assert torch.equal(out_std, out_coalesced)
