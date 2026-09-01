import torch
import torch.nn.functional as F

from affine_ai import TorosHybridConfig, TorosHybridLanguageModel


def _model(unl_weight):
    torch.manual_seed(42)
    cfg = TorosHybridConfig(
        dim=64, d_byte=32, n_encoder_layers=2, n_predictor_layers=1, n_heads=2,
        target_patch_size=8, unlikelihood_weight=unl_weight, dtype=torch.float32
    )
    return TorosHybridLanguageModel(cfg)


def _cycling_targets(T=256):
    """Every 3-gram context occurs only once per 256 bytes; window is 64 -> no flags."""
    return torch.arange(T, dtype=torch.long).remainder(256).unsqueeze(0)


def _loop_targets(rep=16):
    """'the ' x16: after the first occurrence every context is a repeat."""
    return torch.tensor([116, 104, 101, 32] * rep, dtype=torch.long).unsqueeze(0)


def test_unlikelihood_zero_on_non_repeating_text():
    m = _model(0.05)
    T = 256
    targets = _cycling_targets(T)
    logits = torch.randn(1, T, 256) * 3.0
    loss = m._unlikelihood_loss(logits, targets)
    assert loss.item() == 0.0


def test_unlikelihood_confident_loop_penalized_hard():
    m = _model(0.05)
    targets = _loop_targets()
    T = targets.shape[1]
    # one-hot-ish logits: model confidently predicts the loop token
    logits = torch.full((1, T, 256), -20.0)
    logits.scatter_(2, targets.unsqueeze(-1), 20.0)
    loss = m._unlikelihood_loss(logits, targets)
    # ~57/64 flagged, each ~-log(1e-6) ~ 13.8
    assert loss.item() > 10.0


def test_unlikelihood_uniform_logits_penalized_weakly():
    m = _model(0.05)
    targets = _loop_targets()
    logits = torch.zeros(1, targets.shape[1], 256)
    loss = m._unlikelihood_loss(logits, targets)
    # self-sharpening property: flat P(x)=1/256 => per-flag ~0.004
    assert 0.0 < loss.item() < 0.01


def test_unlikelihood_flag_positions():
    """Pure loop: only positions after the first context occurrence are flagged."""
    m = _model(0.05)
    n, W = m.config.unlikelihood_n, m.config.unlikelihood_window
    targets = _loop_targets()
    T = targets.shape[1]
    prev = torch.full((1, T, n - 1), -1, dtype=torch.int64)
    for k in range(1, n):
        prev[:, k:, n - 1 - k] = targets[:, :T - k]
    repeat = torch.zeros(1, T, dtype=torch.bool)
    valid_head = torch.arange(T) >= n - 1
    for dt in range(1, W):
        src = torch.clamp(torch.arange(T) - dt, min=0)
        repeat |= (prev[:, src] == prev).all(dim=-1) & valid_head
    repeat &= (prev[:, :, 0] >= 0)
    flagged = int(repeat.sum())
    assert flagged == T - 7  # first 'the ' (4B) + 3B context head are unflagged


def test_unlikelihood_in_forward_metrics_and_grad():
    m = _model(0.05)
    m.train()
    x = torch.randint(1, 256, (2, 64))
    y = torch.randint(1, 256, (2, 64))
    _, loss, met = m(x, targets=y)
    assert "loss_unl" in met
    assert met["loss_unl"] >= 0.0
    loss.backward()
    assert m.byte_decoder.lm_head.weight.grad is not None


def test_unlikelihood_off_by_default():
    cfg = TorosHybridConfig(dim=64, d_byte=32, n_encoder_layers=2, n_heads=2, target_patch_size=8, use_rls_heads=False, use_type_codebook=False)
    assert cfg.unlikelihood_weight == 0.0
    m = TorosHybridLanguageModel(cfg)
    m.train()
    x = torch.randint(1, 256, (2, 64))
    y = torch.randint(1, 256, (2, 64))
    _, loss, met = m(x, targets=y)
    assert "loss_unl" not in met
    _, _, ref = m(x, targets=y)
    assert loss.item() == ref["loss_gen"]
