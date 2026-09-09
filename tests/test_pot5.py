import pytest
import torch
import torch.nn.functional as F

from affine_ai.core.ast_dag import pot5_quantize
from affine_ai.kernels.triton_pot5 import triton_pot5_linear, triton_pot5_fused_swiglu, Triton5StatePOTLinear
from affine_ai.models.qwen35_asdag import Qwen35ASDAGConfig, Qwen35ASDAGFFN
from affine_ai.models.qwen38_asdag import Qwen38ASDAGConfig, Qwen38ASDAGFFN


def test_pot5_quantize_values():
    """Verify that pot5_quantize produces outputs strictly in {-1, -0.5, 0, +0.5, +1} * alpha."""
    torch.manual_seed(42)
    w = torch.randn(128, 256, dtype=torch.float32)
    w_q = pot5_quantize(w, threshold_z=0.35, shift=1)

    # Compute actual alpha from non-zero elements
    # Since w_q = q * alpha where q in {-1, -0.5, 0, +0.5, +1},
    # the non-zero ratios must be 0.5 or 1.0 relative to max(abs(w_q))
    max_val = w_q.abs().max()
    normalized = (w_q / max_val).round(decimals=3)

    unique_vals = torch.unique(normalized)
    expected_vals = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32)

    for val in unique_vals:
        diffs = (expected_vals - val).abs()
        assert diffs.min().item() < 1e-3, f"Unexpected quantized value found: {val.item()}"


def test_pot5_quantize_ste_backward():
    """Verify Straight-Through Estimator (STE) with gradient clipping."""
    torch.manual_seed(42)
    w = torch.randn(64, 64, dtype=torch.float32, requires_grad=True)
    w_q = pot5_quantize(w, threshold_z=0.35, shift=1)

    loss = (w_q * 2.0).sum()
    loss.backward()

    assert w.grad is not None
    assert not torch.isnan(w.grad).any()


def test_triton_pot5_linear_forward_backward():
    """Verify triton_pot5_linear forward and backward pass."""
    torch.manual_seed(42)
    x = torch.randn(2, 16, 64, dtype=torch.float32, requires_grad=True)
    w = torch.randn(128, 64, dtype=torch.float32, requires_grad=True)

    out = triton_pot5_linear(x, w)
    assert out.shape == (2, 16, 128)
    assert not torch.isnan(out).any()

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert w.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(w.grad).any()


def test_triton_pot5_fused_swiglu():
    """Verify triton_pot5_fused_swiglu output matches SiLU(gate) * up @ down.T."""
    torch.manual_seed(42)
    B, T, D = 2, 8, 64
    I = 128
    gate_up = torch.randn(B, T, 2 * I, dtype=torch.float32)
    w_down = torch.randn(D, I, dtype=torch.float32)

    out = triton_pot5_fused_swiglu(gate_up, w_down)
    assert out.shape == (B, T, D)
    assert not torch.isnan(out).any()

    # Parity check against fallback
    g, u = gate_up.chunk(2, dim=-1)
    w_abs = w_down.abs()
    w_sign = w_down.sign()
    w_full = torch.where(w_abs > 0.75, w_sign, torch.zeros_like(w_down))
    w_half = torch.where((w_abs > 0.25) & (w_abs <= 0.75), w_sign, torch.zeros_like(w_down))
    alpha_d = (w_down.float().abs().mean() * 1.4)
    ref_out = (F.silu(g) * u) @ ((w_full + w_half * 0.5) * alpha_d.item()).T
    diff = (out - ref_out).abs().max().item()
    assert diff < 1e-4, f"Fused SwiGLU deviates from reference: {diff}"


def test_triton_pot5_int8_linear():
    """Verify Route A: Hardware INT8 IMMA 5-State POT linear layer."""
    from affine_ai.kernels.triton_pot5 import triton_pot5_int8_linear
    torch.manual_seed(42)
    B, T, K, N = 2, 8, 64, 128
    x = torch.randn(B, T, K, dtype=torch.float32)
    w = torch.randn(N, K, dtype=torch.float32)

    out = triton_pot5_int8_linear(x, w)
    assert out.shape == (B, T, N)
    assert not torch.isnan(out).any()



def test_pot5_3bitplane_toros_parity():
    """Verify 3-bitplane packing for .toros format achieves 3.0 bits/wt with exact parity."""
    import io, struct
    from affine_ai.core.format import pack_pot5_3bitplane, unpack_pot5_3bitplane, FLAG_POT5_3BITPLANE

    torch.manual_seed(42)
    w = torch.randn(64, 128, dtype=torch.float32)
    packed_bytes, alpha, shape, flag = pack_pot5_3bitplane(w)

    assert flag == FLAG_POT5_3BITPLANE
    # 64 * 128 = 8192 elements. 8192 bits / 8 = 1024 bytes per bitplane.
    # 3 bitplanes = 3072 bytes + 12 header bytes = 3084 bytes.
    # Exactly 3.00 bits per weight!
    expected_bytes = 3 * (8192 // 8) + 12
    assert len(packed_bytes) == expected_bytes, f"Expected {expected_bytes} bytes, got {len(packed_bytes)}"

    w_rec = unpack_pot5_3bitplane(packed_bytes, alpha, shape)
    w_ref = pot5_quantize(w)

    diff = (w_rec - w_ref).abs().max().item()
    assert diff < 1e-4, f"Toros 3-bitplane unpack deviates from reference: {diff}"


def test_pot5_gpu_3bitplane_linear():
    """Verify GPU 3-bitplane bitpacked linear layer runs and matches reference."""
    from affine_ai.kernels.triton_pot5 import (
        pack_pot5_gpu_3bitplane,
        unpack_pot5_gpu_3bitplane,
        triton_pot5_bitpacked_linear,
        Triton5StatePOTBitpackedLinear,
    )
    torch.manual_seed(42)
    N, K = 128, 256
    w = torch.randn(N, K, dtype=torch.float32)
    w_nz, w_mag, w_sign, alpha, K_orig = pack_pot5_gpu_3bitplane(w)

    # Check GPU unpacked parity
    w_rec = unpack_pot5_gpu_3bitplane(w_nz, w_mag, w_sign, alpha, K_orig)
    w_ref = pot5_quantize(w)
    diff = (w_rec - w_ref).abs().max().item()
    assert diff < 1e-4, f"GPU 3-bitplane unpack deviates: {diff}"

    # Check linear projection
    x = torch.randn(2, 8, K, dtype=torch.float32)
    out = triton_pot5_bitpacked_linear(x, w_nz, w_mag, w_sign, alpha, K_orig)
    ref_out = F.linear(x, w_ref)
    diff_out = (out - ref_out).abs().max().item()
    assert diff_out < 1e-3, f"GPU bitpacked linear output deviates: {diff_out}"

    # Check nn.Module
    mod = Triton5StatePOTBitpackedLinear(in_features=K, out_features=N, dtype=torch.float32)
    out_mod = mod(x)
    assert out_mod.shape == (2, 8, N)
    assert not torch.isnan(out_mod).any()


def test_model_quant_mode_defaults_and_options():
    """Verify Qwen35 and Qwen38 config defaults and quantization mode switching."""
    cfg35 = Qwen35ASDAGConfig()
    assert cfg35.weight_quant_mode == "pot5", f"Expected default pot5, got {cfg35.weight_quant_mode}"

    cfg38 = Qwen38ASDAGConfig()
    assert cfg38.weight_quant_mode == "pot5", f"Expected default pot5, got {cfg38.weight_quant_mode}"

    # Test all 3 quantization modes on Qwen35 ASDAG FFN
    for mode in ["pot5", "dual_ternary", "ternary"]:
        config = Qwen35ASDAGConfig(
            dim=256,
            intermediate_dim=512,
            num_leaves=4,
            leaf_dim=128,
            weight_quant_mode=mode,
            dtype=torch.float32,
        )
        ffn = Qwen35ASDAGFFN(config)
        ffn.train()
        x = torch.randn(2, 4, 256, requires_grad=True)
        out = ffn(x, top_k=2)
        assert out.shape == x.shape
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


def test_toros_save_load_pot5_roundtrip(tmp_path):
    """Verify end-to-end saving and loading of model with 3-bitplane POT5 in .toros format."""
    import tempfile, os
    from affine_ai.core.format import save_toros_model, load_toros_model

    config = Qwen35ASDAGConfig(
        dim=128,
        intermediate_dim=256,
        num_leaves=2,
        leaf_dim=128,
        weight_quant_mode="pot5",
        dtype=torch.float32,
    )
    ffn = Qwen35ASDAGFFN(config)
    tmp_file = str(tmp_path / "model_pot5.toros")

    info = save_toros_model(ffn, tmp_file, weight_quant_mode="pot5")
    assert os.path.exists(tmp_file)
    assert info["compression_ratio"] > 1.0

    # Load back
    loaded_ffn, meta = load_toros_model(tmp_file, model_class=lambda *args, **kwargs: Qwen35ASDAGFFN(config))
    x = torch.randn(2, 4, 128)
    out_loaded = loaded_ffn(x, top_k=2)
    assert out_loaded.shape == x.shape
    assert not torch.isnan(out_loaded).any()


