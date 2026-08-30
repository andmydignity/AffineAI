import torch
import pytest
from affine_ai.models.blt import (
    ByteLocalEncoder,
    EntropyPatcher,
    ByteLocalDecoder,
    ASDAGByteLatentModel
)


def test_byte_local_encoder():
    torch.manual_seed(42)
    B, T = 4, 64
    d_byte = 32
    enc = ByteLocalEncoder(vocab_size=256, d_byte=d_byte, kernel_size=4)
    byte_ids = torch.randint(0, 256, (B, T))
    h_byte, boundary_logits = enc(byte_ids)
    
    assert h_byte.shape == (B, T, d_byte)
    assert boundary_logits.shape == (B, T)
    loss = h_byte.sum() + boundary_logits.sum()
    loss.backward()
    assert enc.byte_embed.weight.grad is not None
    assert enc.conv.weight.grad is not None


def test_entropy_patcher():
    torch.manual_seed(42)
    B, T = 2, 64
    d_byte = 32
    d_model = 64
    patcher = EntropyPatcher(d_byte=d_byte, d_model=d_model, target_patch_size=4)
    h_byte = torch.randn(B, T, d_byte, requires_grad=True)
    boundary_logits = torch.randn(B, T, requires_grad=True)
    
    latent_patches, patch_assignments = patcher(h_byte, boundary_logits, fixed_patch_size=4)
    assert latent_patches.shape == (B, 16, d_model)
    assert patch_assignments.shape == (B, T)
    
    loss = latent_patches.sum()
    loss.backward()
    assert h_byte.grad is not None


def test_byte_local_decoder():
    torch.manual_seed(42)
    B, T = 2, 64
    M = 16
    d_byte = 32
    d_model = 64
    dec = ByteLocalDecoder(vocab_size=256, d_byte=d_byte, d_model=d_model)
    
    h_byte = torch.randn(B, T, d_byte, requires_grad=True)
    latent_patches = torch.randn(B, M, d_model, requires_grad=True)
    patch_assignments = torch.arange(M).repeat_interleave(4).unsqueeze(0).expand(B, -1)
    
    logits = dec(h_byte, latent_patches, patch_assignments)
    assert logits.shape == (B, T, 256)
    loss = logits.sum()
    loss.backward()
    assert dec.lm_head.weight.grad is not None


def test_asdag_byte_latent_model_end_to_end():
    torch.manual_seed(42)
    B, T = 2, 64
    model = ASDAGByteLatentModel(
        vocab_size=256,
        d_byte=32,
        d_model=64,
        n_layers=2,
        n_heads=2,
        target_patch_size=4,
        channel_mixer_type="ternary_swiglu"
    )
    
    byte_ids = torch.randint(0, 256, (B, T))
    targets = torch.randint(0, 256, (B, T))
    
    logits, loss, stats = model(byte_ids, targets=targets)
    assert logits.shape == (B, T, 256)
    assert loss is not None
    assert stats["compression_ratio"] == 4.0
    
    loss.backward()
    
    # Test generation
    prompt = torch.tensor([[ord(c) for c in "Hello"]], dtype=torch.long)
    gen = model.generate(prompt, max_new_bytes=10, temperature=0.7)
    assert gen.shape[1] == 15
