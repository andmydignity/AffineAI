import torch
import pytest
from affine_ai.core.bitlinear import BitLinear, TernaryBitLinearSwiGLU
from affine_ai.core.associative import MonarchPermutationChain, FusedMonarchChain, NativeASDAGAssociativeMixer
from affine_ai.models.language_model import ASDAGLanguageModel


def test_bitlinear_ste_quantization():
    torch.manual_seed(42)
    layer = BitLinear(in_features=32, out_features=64, bias=True)
    x = torch.randn(4, 16, 32, dtype=torch.bfloat16)
    out = layer(x)
    assert out.shape == (4, 16, 64)
    loss = out.sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None


def test_ternary_swiglu_channel_mixer():
    torch.manual_seed(42)
    mixer = TernaryBitLinearSwiGLU(dim=32, expand=2)
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)
    out = mixer(x)
    assert out.shape == (2, 8, 32)
    loss = out.sum()
    loss.backward()
    assert mixer.w_gate_val.weight.grad is not None
    assert mixer.w_down.weight.grad is not None


def test_monarch_permutation_chain():
    torch.manual_seed(42)
    monarch = MonarchPermutationChain(dim=32, num_stages=4)
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)
    out = monarch(x)
    assert out.shape == (2, 8, 32)
    loss = out.sum()
    loss.backward()
    assert monarch.diagonals.grad is not None
    assert monarch.bias.grad is not None


def test_fused_monarch_chain():
    torch.manual_seed(42)
    fused = FusedMonarchChain(dim=32, num_branches=4, num_stages=4)
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)
    q, k, v, g = fused(x)
    assert q.shape == (2, 8, 32)
    assert k.shape == (2, 8, 32)
    assert v.shape == (2, 8, 32)
    assert g.shape == (2, 8, 32)


def test_full_option_b_language_model():
    torch.manual_seed(42)
    model = ASDAGLanguageModel(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=4,
        channel_mixer_type="ternary_swiglu"
    )
    x = torch.randint(0, 256, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 256)

    # Test O(1) step generation
    gen = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
    assert gen.shape == (1, 8)
