import torch
import pytest
from affine_ai.models.language_model import ASDAGLanguageModel, ASDAGBlock
from affine_ai.core.ast_dag import ASDAGConfig


def test_asdag_language_model_forward():
    batch_size = 4
    seq_len = 16
    vocab_size = 256
    d_model = 64
    n_layers = 2
    num_leaves = 8

    model = ASDAGLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        num_leaves=num_leaves,
        sparsity_ratio=0.875,
        shift_bits=4
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids, use_quantized_gates=True, use_shift4_act=True)

    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_asdag_language_model_generate():
    vocab_size = 128
    d_model = 32

    model = ASDAGLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        num_leaves=4
    )

    prompt = torch.tensor([[10, 20, 30]])
    gen = model.generate(prompt, max_new_tokens=10)

    assert gen.shape == (1, 13)
