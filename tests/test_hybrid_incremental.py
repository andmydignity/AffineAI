import time

import pytest
import torch

from affine_ai import TorosHybridLanguageModel, TorosHybridConfig


def _small_model(**cfg_kwargs):
    torch.manual_seed(42)
    cfg_kwargs.setdefault("dynamic_patching", False)
    cfg = TorosHybridConfig(
        dim=64, d_byte=32, n_encoder_layers=2,
        n_heads=2, target_patch_size=8, dtype=torch.float32,
        **cfg_kwargs,
    )
    return TorosHybridLanguageModel(cfg).eval()


def test_incremental_matches_full_forward():
    """Last-position logits of one-shot forward_incremental ~= forward()."""
    model = _small_model()
    for T in (8, 9, 15, 16, 17, 24, 25):
        x = torch.randint(1, 256, (1, T))
        with torch.no_grad():
            full_logits, _, _ = model.forward(x)
            inc_logits, state = model.forward_incremental(x, None, return_state=True)
        diff = (full_logits[:, -1] - inc_logits[:, -1]).abs().max().item()
        # Residual comes from int8 quantization rounding inside the fused C++ block
        # kernel (used by forward) vs the float ternary path (used incrementally).
        assert diff < 0.05, (T, diff)
        assert state["n_patches"] == T // 8


def test_incremental_byte_by_byte_matches_one_shot():
    """Stepping one byte at a time must equal feeding the whole sequence once."""
    model = _small_model()
    x = torch.randint(1, 256, (1, 25))
    with torch.no_grad():
        one_shot, _ = model.forward_incremental(x, None, return_state=True)
        state = None
        steps = []
        for t in range(x.shape[1]):
            logits, state = model.forward_incremental(x[:, t:t + 1], state, return_state=True)
            steps.append(logits)
        byte_by_byte = torch.cat(steps, dim=1)
    assert torch.allclose(one_shot, byte_by_byte, atol=1e-5), \
        (one_shot - byte_by_byte).abs().max().item()


def test_incremental_generation_shape_and_eos():
    model = _small_model()
    prompt = torch.randint(1, 256, (2, 12))
    out = model.generate_with_latent_planning(
        prompt, max_new_bytes=20, temperature=0.0, eos_byte=None
    )
    assert out.shape == (2, 32)


def test_generation_faster_than_full_forward_loop():
    """O(1)-per-byte incremental generation must beat O(T^2) full recompute."""
    model = _small_model()
    prompt = torch.randint(1, 256, (1, 32))

    torch.manual_seed(0)
    curr = prompt.clone()
    t0 = time.perf_counter()
    for _ in range(48):
        logits, _, _ = model.forward(curr)
        next_byte = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        curr = torch.cat([curr, next_byte], dim=1)
    t_full = time.perf_counter() - t0

    torch.manual_seed(0)
    t0 = time.perf_counter()
    model.generate_with_latent_planning(prompt, max_new_bytes=48, temperature=0.0)
    t_inc = time.perf_counter() - t0

    assert t_inc < t_full, (t_full, t_inc)


def test_forward_compute_jepa_flag():
    model = _small_model(use_rls_heads=False, use_type_codebook=False)
    x = torch.randint(1, 256, (1, 12))
    y = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        _, loss_full, m_full = model.forward(x, targets=y)
        _, loss_gen_only, m_gen = model.forward(x, targets=y, compute_jepa=False)
    assert "loss_jepa" not in m_full
    assert "loss_jepa" not in m_gen
    assert loss_full.item() == pytest.approx(m_full["loss_gen"])


def test_forward_default_config_jepa_off():
    """Stripped hybrid: JEPA off by default, no System-2 keys."""
    model = _small_model(use_rls_heads=False, use_type_codebook=False)
    x = torch.randint(1, 256, (1, 12))
    y = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        _, loss, m = model.forward(x, targets=y)
    assert "loss_jepa" not in m
    assert loss.item() == pytest.approx(m["loss_gen"])


def test_lpc_step_runs_and_updates():
    model = _small_model()
    assert not hasattr(model, "forward_lpc_step")
    assert not hasattr(model, "local_heads")
    assert not hasattr(model, "predictor")
    model.train()
    x = torch.randint(1, 256, (2, 24))
    y = torch.randint(0, 256, (2, 24))
    _, loss, m = model(x, targets=y)
    loss.backward()
    assert m["loss_gen"] > 0
    assert model.byte_decoder.lm_head.weight.grad is not None
