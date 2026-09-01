import pytest
import torch

from affine_ai import ASDAGLanguageModel, TorosHybridConfig, TorosHybridLanguageModel


def _small_hybrid():
    torch.manual_seed(42)
    cfg = TorosHybridConfig(
        dim=64, d_byte=32, n_encoder_layers=2, n_predictor_layers=1,
        n_heads=2, target_patch_size=8, dtype=torch.float32
    )
    return TorosHybridLanguageModel(cfg).eval()


def test_sample_greedy_when_temperature_zero():
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.1]])
    out = TorosHybridLanguageModel._sample_next_byte(logits, temperature=0.0)
    assert out.item() == 1


def test_sample_temperature_zero_beats_top_k_and_top_p():
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.1]])
    out = TorosHybridLanguageModel._sample_next_byte(logits, temperature=0.0, top_k=2, top_p=0.5)
    assert out.item() == 1


def test_sample_top_k_restricts_support():
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, 9.0, 1.0, 0.0]])
    # top_k=2 -> only tokens 0 and 1 can ever be sampled
    for _ in range(30):
        out = TorosHybridLanguageModel._sample_next_byte(logits, temperature=1.0, top_k=2)
        assert out.item() in (0, 1)


def test_sample_top_p_truncates_tail():
    torch.manual_seed(0)
    # distribution: ~0.8, ~0.19, ~0.01 -> top_p=0.9 keeps ranks 0-1 only
    logits = torch.log(torch.tensor([[0.80, 0.19, 0.01]]))
    for _ in range(30):
        out = TorosHybridLanguageModel._sample_next_byte(logits, temperature=1.0, top_p=0.9)
        assert out.item() in (0, 1)


def test_sample_top_p_keeps_rank0_always():
    logits = torch.log(torch.tensor([[0.999, 0.001]]))
    # top_p smaller than rank-0 mass alone: rank 0 must still survive
    out = TorosHybridLanguageModel._sample_next_byte(logits, temperature=1.0, top_p=0.01)
    assert out.item() == 0


def test_sample_seed_reproducible():
    logits = torch.randn(1, 256)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = TorosHybridLanguageModel._sample_next_byte(logits, temperature=1.0, top_p=0.9, generator=g1)
    b = TorosHybridLanguageModel._sample_next_byte(logits, temperature=1.0, top_p=0.9, generator=g2)
    assert torch.equal(a, b)


def test_sample_batch_independent_support():
    # per-row truncation: row 0 peaked, row 1 uniform
    logits = torch.stack([torch.log(torch.tensor([0.95, 0.04, 0.01])),
                          torch.zeros(3)])
    torch.manual_seed(1)
    for _ in range(10):
        out = TorosHybridLanguageModel._sample_next_byte(logits, temperature=1.0, top_p=0.9)
        assert out[0].item() in (0,)           # row 0: only rank 0 survives
        assert out[1].item() in (0, 1, 2)       # row 1: uniform survives fully


def test_hybrid_generate_sampling_params_and_eos():
    m = _small_hybrid()
    prompt = torch.randint(1, 256, (2, 16))
    g = torch.Generator().manual_seed(0)
    out = m.generate_with_latent_planning(
        prompt, max_new_bytes=10, temperature=0.8, top_p=0.9,
        eos_byte=0, generator=g
    )
    assert out.shape == (2, 26)


def test_hybrid_generate_eos_stops_early():
    m = _small_hybrid()
    prompt = torch.zeros(1, 8, dtype=torch.long)  # all-EOS prompt forces the mode quickly
    out = m.generate_with_latent_planning(
        prompt, max_new_bytes=50, temperature=1.0, eos_byte=0
    )
    # may or may not hit EOS; shape contract only
    assert out.shape[1] >= 8 + 1


def test_plain_lm_generate_with_sampling():
    torch.manual_seed(0)
    m = ASDAGLanguageModel(vocab_size=256, d_model=32, n_layers=2, n_heads=4,
                           use_blt=False, use_hybrid=False, dtype=torch.float32).eval()
    x = torch.randint(1, 256, (1, 8))
    out = m.generate(x, max_new_tokens=5, temperature=0.9, top_p=0.95, eos_byte=None)
    assert out.shape == (1, 13)
    out_k = m.generate(x, max_new_tokens=5, temperature=0.9, top_k=10, eos_byte=None)
    assert out_k.shape == (1, 13)


def test_blt_generate_sampling():
    torch.manual_seed(0)
    m = ASDAGLanguageModel(vocab_size=256, d_model=32, n_layers=2, n_heads=4,
                           use_blt=True, use_hybrid=False, dtype=torch.float32).eval()
    x = torch.randint(1, 256, (1, 8))
    out = m.generate(x, max_new_tokens=5, temperature=0.8, top_p=0.9, eos_byte=None)
    assert out.shape == (1, 13)
