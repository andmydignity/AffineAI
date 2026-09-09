import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.training.distill_bytes import DistillBytesAligner
from affine_ai.models.blt import ByteLocalEncoder, EntropyPatcher


def test_distill_bytes_aligner_initialization():
    aligner = DistillBytesAligner(dim=64, tokenizer_path="tokenizer.json", gguf_path=None)
    assert aligner.dim == 64
    assert aligner.tokenizer is not None


def test_distill_bytes_compute_alignment_loss():
    dim = 64
    aligner = DistillBytesAligner(dim=dim, tokenizer_path="tokenizer.json", gguf_path=None)

    # Sample prompt bytes: "Once upon a time"
    prompt_str = "Once upon a time in a faraway forest"
    byte_seq = list(prompt_str.encode("utf-8"))
    byte_ids = torch.tensor([byte_seq, byte_seq], dtype=torch.long)
    B, T = byte_ids.shape
    M = 4  # 4 latent patches

    # Synthetic patch latents requiring grad
    patch_latents = torch.randn(B, M, dim, requires_grad=True)

    loss_align, cos_sim = aligner.compute_alignment_loss(patch_latents, byte_ids)

    assert isinstance(loss_align, torch.Tensor)
    assert loss_align.item() > 0.0
    assert -1.0 <= cos_sim <= 1.0

    # Ensure gradients flow back to patch_latents
    loss_align.backward()
    assert patch_latents.grad is not None
    assert patch_latents.grad.norm().item() > 0.0


def test_distill_bytes_end_to_end_step():
    dim = 64
    d_byte = 32
    patch_size = 4
    T = 16
    M = T // patch_size

    byte_encoder = ByteLocalEncoder(vocab_size=256, d_byte=d_byte, kernel_size=3, dtype=torch.float32)
    patcher = EntropyPatcher(d_byte=d_byte, d_model=dim, target_patch_size=patch_size, dtype=torch.float32)
    aligner = DistillBytesAligner(dim=dim, tokenizer_path="tokenizer.json", gguf_path=None)

    optimizer = torch.optim.Adam(list(byte_encoder.parameters()) + list(patcher.parameters()), lr=1e-3)

    byte_ids = torch.randint(32, 126, (2, T))

    # Optimization step
    optimizer.zero_grad()
    h_byte, boundary = byte_encoder(byte_ids)
    latent_patches, _ = patcher(h_byte, torch.zeros_like(boundary), fixed_patch_size=patch_size)

    loss_align, initial_cos = aligner.compute_alignment_loss(latent_patches, byte_ids)
    loss_align.backward()
    optimizer.step()

    assert loss_align.item() > 0.0
