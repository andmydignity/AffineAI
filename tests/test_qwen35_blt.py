import pytest
import torch
import torch.nn as nn
from affine_ai.models.qwen35_blt import Qwen35BLTConfig, Qwen35BLTLanguageModel


def test_qwen35_blt_config_defaults():
    """Verify BLT configuration defaults (P=16, vocab=256)."""
    cfg = Qwen35BLTConfig()
    assert cfg.vocab_size == 256
    assert cfg.target_patch_size == 16
    assert cfg.d_byte == 128
    assert cfg.dim == 2560
    assert cfg.num_layers == 32


def test_qwen35_blt_forward_and_loss():
    """Verify forward pass, P=16 padding, loss computation, and frozen backbone gradient isolation."""
    torch.manual_seed(42)
    cfg = Qwen35BLTConfig(
        dim=256,
        d_byte=64,
        target_patch_size=16,
        num_layers=2,
        intermediate_dim=512,
        full_attn_interval=2,
        dtype=torch.float32
    )
    model = Qwen35BLTLanguageModel(cfg)
    model.freeze_backbone()

    trainable_params = model.get_trainable_parameters()
    assert len(trainable_params) > 0

    # Ensure backbone weights have requires_grad = False
    for p in model.blocks.parameters():
        assert not p.requires_grad
    for p in model.output_norm.parameters():
        assert not p.requires_grad

    # Test forward with arbitrary byte sequence length (e.g. 35 bytes -> padded to 48 for P=16)
    B, T = 2, 35
    byte_ids = torch.randint(0, 256, (B, T), dtype=torch.long)
    targets = torch.randint(0, 256, (B, T), dtype=torch.long)

    logits, loss = model(byte_ids, targets=targets)

    assert logits.shape == (B, T, 256)
    assert loss is not None
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    # Backward pass
    loss.backward()

    # Trainable parameters must have gradients
    for p in trainable_params:
        assert p.grad is not None, "Trainable param missing gradient"
        assert not torch.isnan(p.grad).any()

    # Backbone parameters must have NO gradients
    for p in model.blocks.parameters():
        assert p.grad is None


def test_qwen35_blt_generate_bytes():
    """Verify autoregressive byte generation produces a byte stream."""
    torch.manual_seed(42)
    cfg = Qwen35BLTConfig(
        dim=256,
        d_byte=64,
        target_patch_size=16,
        num_layers=2,
        intermediate_dim=512,
        full_attn_interval=2,
        dtype=torch.float32
    )
    model = Qwen35BLTLanguageModel(cfg)

    prompt = "Hello, world!"
    out_bytes = model.generate_bytes(prompt, max_new_bytes=10, temperature=0.8)

    assert isinstance(out_bytes, bytes)
    assert len(out_bytes) > len(prompt.encode("utf-8"))
    assert out_bytes.startswith(prompt.encode("utf-8"))
