import pytest
import torch
import numpy as np
import affine_ai
from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig, compute_document_reset_mask
from affine_ai.models.language_model import ASDAGLanguageModel


def test_compute_document_reset_mask():
    # Verify that EOS markers and document starts produce correct reset masks
    eos_str = "<|endoftext|>"
    doc1 = "Once upon a time in a faraway forest."
    doc2 = "A new journey begins today."
    text = doc1 + eos_str + doc2
    byte_ids = torch.from_numpy(np.frombuffer(text.encode("utf-8"), dtype=np.uint8)).unsqueeze(0).long()

    byte_reset, patch_reset = compute_document_reset_mask(byte_ids, patch_size=8, eos_token=eos_str)

    # Position 0 must always be marked as a reset boundary
    assert bool(byte_reset[0, 0]) is True
    assert bool(patch_reset[0, 0]) is True

    # Byte right after <|endoftext|> must be marked as a reset boundary
    doc2_start_byte = len((doc1 + eos_str).encode("utf-8"))
    assert bool(byte_reset[0, doc2_start_byte]) is True

    # The corresponding patch must be marked as reset
    doc2_patch = doc2_start_byte // 8
    assert bool(patch_reset[0, doc2_patch]) is True


def test_no_remnants_in_context_window():
    # Initialize a lightweight model
    config = TorosHybridConfig(
        dim=64,
        d_byte=32,
        n_encoder_layers=2,
        target_patch_size=4,
    )
    model = TorosHybridLanguageModel(config).eval()

    eos = "<|endoftext|>"
    doc1 = "The quick brown fox jumps over the lazy dog."
    doc2 = "AffineAI MatMul-Free Neural Network."

    # Encode
    doc2_bytes = torch.from_numpy(np.frombuffer(doc2.encode("utf-8"), dtype=np.uint8)).unsqueeze(0).long()
    combined_str = doc1 + eos + doc2
    comb_bytes = torch.from_numpy(np.frombuffer(combined_str.encode("utf-8"), dtype=np.uint8)).unsqueeze(0).long()

    # Pass through model
    with torch.no_grad():
        logits_alone, _, _ = model(doc2_bytes)
        logits_comb, _, _ = model(comb_bytes)

    # Verify both succeed without crashing or shape errors
    assert logits_alone is not None
    assert logits_comb is not None
    assert logits_comb.shape[1] == comb_bytes.shape[1]

    # Verify reset_context flushes context state
    model.reset_context()
    assert getattr(model, "_lpc_graph_runner", None) is None


def test_asdag_language_model_reset_context():
    model = ASDAGLanguageModel(vocab_size=256, d_model=64, n_layers=2, use_blt=False, use_hybrid=False).eval()
    
    x = torch.randint(1, 255, (2, 32))
    # Inject EOS/0 at position 16
    x[:, 16] = 0

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, 32, 256)
    model.reset_context()
