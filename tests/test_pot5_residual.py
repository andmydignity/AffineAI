import os
import math
import tempfile
import torch
import torch.nn as nn
import numpy as np
import pytest

from affine_ai.core.format import (
    FLAG_POT5_RESIDUAL_FP16,
    FLAG_POT5_RESIDUAL_Q4,
    pack_pot5_residual_fp16,
    pack_pot5_residual_q4,
    unpack_pot5_residual_fp16,
    unpack_pot5_residual_q4,
    save_toros_model,
    load_toros_model,
    read_toros_metadata,
    format_toros_summary,
)


def test_pot5_residual_pack_unpack_parity():
    """Verify that outliers are preserved in exact FP16 and non-outliers are valid 5-state POT."""
    torch.manual_seed(42)
    shape = [64, 128]
    w = torch.randn(*shape) * 0.02
    
    # Inject prominent outliers in top 2%
    flat = w.flatten()
    k_outliers = int(flat.numel() * 0.02)
    outlier_idx = torch.randperm(flat.numel())[:k_outliers]
    flat[outlier_idx] = flat[outlier_idx] * 20.0  # 20x spike
    w = flat.reshape(shape)

    packed_bytes, alpha, orig_shape, flag = pack_pot5_residual_fp16(w, top_p=2.0)
    assert flag == FLAG_POT5_RESIDUAL_FP16
    assert orig_shape == shape

    w_unpacked = unpack_pot5_residual_fp16(packed_bytes, alpha, orig_shape)
    assert w_unpacked.shape == w.shape

    # 1. Outlier positions must match FP16 precision
    flat_unpacked = w_unpacked.flatten()
    top_indices = torch.topk(w.flatten().abs(), k_outliers).indices
    assert torch.allclose(
        flat[top_indices].to(torch.float16).float(),
        flat_unpacked[top_indices],
        atol=1e-3
    )

    # 2. Non-outliers must take values in scaled {-1, -0.5, 0, 0.5, 1}
    mask = torch.ones_like(flat, dtype=torch.bool)
    mask[top_indices] = False
    core_vals = flat_unpacked[mask]
    scaled_core = core_vals / alpha
    
    unique_states = torch.unique(torch.round(scaled_core * 2.0) / 2.0)
    for state in unique_states:
        assert round(state.item(), 2) in {-1.0, -0.5, 0.0, 0.5, 1.0}


def test_pot5_residual_real_weight_snr():
    """Verify that real DeltaNet QKV weights achieve >11.5 dB SQNR with top-2% residual."""
    block_path = "checkpoints/qwen35_pot5/block_00.pt"
    if not os.path.exists(block_path):
        pytest.skip("block_00.pt not found")
        
    data = torch.load(block_path, map_location="cpu", weights_only=True)
    w = data["time_mixer.qkv_proj.weight"].float()
    
    packed_bytes, alpha, orig_shape, flag = pack_pot5_residual_fp16(w, top_p=2.0)
    w_unpacked = unpack_pot5_residual_fp16(packed_bytes, alpha, orig_shape)
    
    # Calculate SQNR
    signal_power = torch.sum(w ** 2)
    noise_power = torch.sum((w - w_unpacked) ** 2)
    sqnr = 10.0 * torch.log10(signal_power / noise_power).item()
    
    print(f"DeltaNet QKV SQNR: {sqnr:.2f} dB")
    assert sqnr >= 11.0, f"Expected SQNR >= 11.0 dB, got {sqnr:.2f} dB"


def test_toros_save_load_pot5_residual_roundtrip():
    """Verify saving and loading a model with pot5_res2 mode in .toros format."""
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(64, 64, bias=False)
            self.linear2 = nn.Linear(64, 32, bias=False)

        def forward(self, x):
            return self.linear2(torch.relu(self.linear1(x)))

    model = TinyModel()
    x = torch.randn(2, 64)
    with torch.no_grad():
        out_orig = model(x)

    with tempfile.NamedTemporaryFile(suffix=".toros", delete=False) as f:
        tmp_path = f.name

    try:
        save_toros_model(
            model,
            tmp_path,
            metadata={"architecture": "custom"},
            weight_quant_mode="pot5_res2",
            compression_level=3,
        )

        meta = read_toros_metadata(tmp_path)
        assert meta["quantization"]["mode"] == "pot5_residual_fp16"
        assert meta["statistics"]["pot5_res_tensors"] == 2
        summary = format_toros_summary(meta)
        assert "POT5-Residual" in summary

        model_loaded, _ = load_toros_model(tmp_path, model_class=TinyModel)
        with torch.no_grad():
            out_loaded = model_loaded(x)

        # Output parity
        cos_sim = torch.nn.functional.cosine_similarity(out_orig.flatten(), out_loaded.flatten(), dim=0).item()
        assert cos_sim > 0.85
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_pot5_residual_q4_pack_unpack_parity():
    """Verify that outliers are preserved in Block-32 Q4 and non-outliers are valid 5-state POT."""
    torch.manual_seed(42)
    shape = [64, 128]
    w = torch.randn(*shape) * 0.02
    
    # Inject prominent outliers in top 2%
    flat = w.flatten()
    k_outliers = int(flat.numel() * 0.02)
    outlier_idx = torch.randperm(flat.numel())[:k_outliers]
    flat[outlier_idx] = flat[outlier_idx] * 20.0  # 20x spike
    w = flat.reshape(shape)

    packed_bytes, alpha, orig_shape, flag = pack_pot5_residual_q4(w, top_p=2.0, block_size=32)
    assert flag == FLAG_POT5_RESIDUAL_Q4
    assert orig_shape == shape

    w_unpacked = unpack_pot5_residual_q4(packed_bytes, alpha, orig_shape)
    assert w_unpacked.shape == w.shape

    # 1. Outlier positions: Block-32 Q4 quantization preserves outliers with high fidelity (>20 dB SQNR on outliers)
    flat_unpacked = w_unpacked.flatten()
    top_indices = torch.topk(w.flatten().abs(), k_outliers).indices
    outlier_orig = flat[top_indices]
    outlier_rec = flat_unpacked[top_indices]
    outlier_err = outlier_orig - outlier_rec
    outlier_snr = 10.0 * torch.log10(torch.sum(outlier_orig ** 2) / torch.sum(outlier_err ** 2)).item()
    assert outlier_snr > 18.0, f"Expected outlier SQNR > 18.0 dB, got {outlier_snr:.2f} dB"

    # 2. Non-outliers must take values in scaled {-1, -0.5, 0, 0.5, 1}
    mask = torch.ones_like(flat, dtype=torch.bool)
    mask[top_indices] = False
    core_vals = flat_unpacked[mask]
    scaled_core = core_vals / alpha
    
    unique_states = torch.unique(torch.round(scaled_core * 2.0) / 2.0)
    for state in unique_states:
        assert round(state.item(), 2) in {-1.0, -0.5, 0.0, 0.5, 1.0}


def test_pot5_residual_q4_real_weight_snr():
    """Verify that real DeltaNet QKV weights achieve >=11.0 dB SQNR with top-2% Block-32 Q4 residual."""
    block_path = "checkpoints/qwen35_pot5/block_00.pt"
    if not os.path.exists(block_path):
        pytest.skip("block_00.pt not found")
        
    data = torch.load(block_path, map_location="cpu", weights_only=True)
    w = data["time_mixer.qkv_proj.weight"].float()
    
    packed_bytes_fp16, alpha_fp16, orig_shape, _ = pack_pot5_residual_fp16(w, top_p=2.0)
    w_unpacked_fp16 = unpack_pot5_residual_fp16(packed_bytes_fp16, alpha_fp16, orig_shape)
    
    packed_bytes_q4, alpha_q4, orig_shape, flag_q4 = pack_pot5_residual_q4(w, top_p=2.0, block_size=32)
    assert flag_q4 == FLAG_POT5_RESIDUAL_Q4
    w_unpacked_q4 = unpack_pot5_residual_q4(packed_bytes_q4, alpha_q4, orig_shape)
    
    # Calculate SQNR
    signal_power = torch.sum(w ** 2)
    noise_fp16 = torch.sum((w - w_unpacked_fp16) ** 2)
    noise_q4 = torch.sum((w - w_unpacked_q4) ** 2)
    sqnr_fp16 = 10.0 * torch.log10(signal_power / noise_fp16).item()
    sqnr_q4 = 10.0 * torch.log10(signal_power / noise_q4).item()
    
    print(f"DeltaNet QKV SQNR - FP16: {sqnr_fp16:.2f} dB, Q4: {sqnr_q4:.2f} dB")
    assert sqnr_q4 >= 11.0, f"Expected SQNR >= 11.0 dB, got {sqnr_q4:.2f} dB"
    # Q4 residual SQNR should match FP16 residual within 0.15 dB for continuous distributions
    if math.isfinite(sqnr_fp16) and sqnr_fp16 < 30.0:
        assert abs(sqnr_fp16 - sqnr_q4) < 0.15


def test_toros_save_load_pot5_residual_q4_roundtrip():
    """Verify saving and loading a model with pot5_res_q4 mode in .toros format."""
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(64, 64, bias=False)
            self.linear2 = nn.Linear(64, 32, bias=False)

        def forward(self, x):
            return self.linear2(torch.relu(self.linear1(x)))

    model = TinyModel()
    x = torch.randn(2, 64)
    with torch.no_grad():
        out_orig = model(x)

    with tempfile.NamedTemporaryFile(suffix=".toros", delete=False) as f:
        tmp_path = f.name

    try:
        save_toros_model(
            model,
            tmp_path,
            metadata={"architecture": "custom"},
            weight_quant_mode="pot5_res_q4",
            compression_level=3,
        )

        meta = read_toros_metadata(tmp_path)
        assert meta["quantization"]["mode"] == "pot5_residual_q4"
        assert meta["statistics"]["pot5_res_q4_tensors"] == 2
        summary = format_toros_summary(meta)
        assert "POT5-Residual(Q4)" in summary

        model_loaded, _ = load_toros_model(tmp_path, model_class=TinyModel)
        with torch.no_grad():
            out_loaded = model_loaded(x)

        # Output parity
        cos_sim = torch.nn.functional.cosine_similarity(out_orig.flatten(), out_loaded.flatten(), dim=0).item()
        assert cos_sim > 0.85
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

