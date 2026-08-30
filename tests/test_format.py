import os
import torch
import numpy as np
from affine_ai import (
    TorosHybridLanguageModel,
    TorosHybridConfig,
    save_toros_model,
    load_toros_model,
    read_toros_metadata
)
from affine_ai.core.format import pack_ternary_tensor, unpack_ternary_tensor, FLAG_TERNARY_2BIT, FLAG_SPARSE_TERNARY


def test_ternary_packing_roundtrip():
    # Dense ternary tensor
    w = torch.tensor([[-0.8, 0.0, 0.8, 0.0], [0.8, -0.8, 0.0, 0.8]], dtype=torch.float32)
    packed, gamma, shape, flag = pack_ternary_tensor(w)
    assert flag == FLAG_TERNARY_2BIT
    
    unpacked = unpack_ternary_tensor(packed, gamma, shape, flag)
    expected = torch.round(w / gamma).clamp(-1.0, 1.0) * gamma
    assert torch.allclose(expected, unpacked, atol=1e-4)


def test_sparse_ternary_packing_roundtrip():
    # Sparse ternary tensor (90% zeros)
    w = torch.zeros(10, 10, dtype=torch.float32)
    w[1, 2] = 0.5
    w[4, 5] = -0.5
    w[8, 9] = 0.5
    
    packed, gamma, shape, flag = pack_ternary_tensor(w)
    assert flag == FLAG_SPARSE_TERNARY
    
    unpacked = unpack_ternary_tensor(packed, gamma, shape, flag)
    expected = torch.round(w / gamma).clamp(-1.0, 1.0) * gamma
    assert torch.allclose(expected, unpacked, atol=1e-4)


def test_toros_save_load_roundtrip(tmp_path):
    config = TorosHybridConfig(
        dim=64, d_byte=32, n_encoder_layers=2, n_predictor_layers=1, n_heads=2, target_patch_size=8
    )
    model = TorosHybridLanguageModel(config)
    
    filepath = str(tmp_path / "test_model.toros")
    stats = model.save_toros(filepath, metadata={"author": "AffineAI", "test": True})
    
    assert os.path.exists(filepath)
    assert stats["compressed_bytes"] < stats["uncompressed_bytes"]
    
    # Test instant metadata read (0.001s)
    meta = read_toros_metadata(filepath)
    assert meta["format"] == "TOROS"
    assert meta["sparsity"]["tree_sparsity"] == 0.9375
    assert meta["user_metadata"]["author"] == "AffineAI"
    
    # Test full model loading
    loaded_model = TorosHybridLanguageModel.from_toros(filepath)
    loaded_model.eval()
    
    # Verify forward pass works seamlessly
    x = torch.randint(0, 256, (2, 32))
    logits, _, _ = loaded_model(x)
    assert logits.shape == (2, 32, 256)
