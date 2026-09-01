import math

import pytest
import torch
import torch.nn.functional as F

from affine_ai import TorosHybridConfig, TorosHybridLanguageModel
from affine_ai.core.associative import NativeASDAGAssociativeMixer


def _mixer(rule, n_heads=2, d_model=64):
    torch.manual_seed(42)
    return NativeASDAGAssociativeMixer(
        d_model=d_model, n_heads=n_heads, proj_type="monarch",
        dtype=torch.float32, rule=rule
    ).eval()


@pytest.mark.parametrize("rule", ["gla", "delta"])
def test_mixer_prefix_causality(rule):
    """Outputs for positions [0:4] must not depend on the sequence length."""
    m = _mixer(rule, n_heads=4)  # H=4: layout bugs are invisible at H=2 in short windows
    torch.manual_seed(3)
    x8 = torch.randn(1, 8, 64)
    with torch.no_grad():
        y8, _ = m(x8, state=None, return_state=True)
        y4, _ = m(x8[:, :4].clone(), state=None, return_state=True)
    assert torch.allclose(y4, y8[:, :4], atol=1e-5), (y4 - y8[:, :4]).abs().max().item()


@pytest.mark.parametrize("rule", ["gla", "delta"])
def test_mixer_carry_split_matches_full(rule):
    """state-carry chunked calls must equal one-shot at any split point."""
    m = _mixer(rule, n_heads=4)
    torch.manual_seed(3)
    x = torch.randn(2, 12, 64)
    with torch.no_grad():
        y_full, s_full = m(x, state=None, return_state=True)
        y_a, s_a = m(x[:, :5], state=None, return_state=True)
        y_b, s_b = m(x[:, 5:], state=s_a, return_state=True)
    assert torch.allclose(torch.cat([y_a, y_b], 1), y_full, atol=1e-5)
    assert torch.allclose(s_b[0], s_full[0], atol=1e-5)


def test_delta_supercession():
    """Slot update in the implemented convention: S <- S(I - b k k^T) + b k v^T.

    After overwriting slot k with v2, the k-parallel component of the old value
    is fully removed (exactly superseded); only the old value's perpendicular
    residue can remain, and it is cancelled for key-k retrieval by the mixer's
    denominator normalization. GLA accumulates both values instead.
    """
    torch.manual_seed(0)
    D = 16
    k = F.normalize(torch.randn(D), dim=0)
    v1 = torch.randn(D)
    v2 = torch.randn(D)

    S = torch.zeros(D, D)
    S = S - S @ torch.outer(k, k) + torch.outer(k, v1)
    S = S - S @ torch.outer(k, k) + torch.outer(k, v2)
    col = S @ k
    residue = col - v2
    k_par = (k @ residue) * k
    assert k_par.norm() < 1e-6          # parallel part exactly superseded

    # GLA contrast: both writes accumulate; the row-query at k returns v1 + v2.
    S_gla = torch.outer(k, v1) + torch.outer(k, v2)
    assert (k @ S_gla - (v1 + v2)).norm() < 1e-5


def test_delta_loop_noise_bounded_vs_gla():
    """On identical repeated (k, v) writes, the delta state's loop-slot component
    converges to a fixed point while GLA's grows without bound (state flooding).

    Uses the mixer's own recurrence convention (right-projection on S):
        delta: S <- g*S - (g*S) @ (b k k^T) + b k v^T
        gla:   S <- g*S + k v^T
    Slot magnitude = ||S @ k|| (the column the loop writes into).
    """
    torch.manual_seed(0)
    D = 16
    k = F.normalize(torch.randn(D), dim=0)
    v = torch.randn(D)
    g, b = 0.93, 0.9

    S_delta, S_gla = torch.zeros(D, D), torch.zeros(D, D)
    slot_d, slot_g = [], []
    for _ in range(300):
        S_delta = g * S_delta - (g * S_delta) @ (b * torch.outer(k, k)) + b * torch.outer(k, v)
        S_gla = g * S_gla + torch.outer(k, v)
        slot_d.append((S_delta @ k).norm().item())
        slot_g.append((S_gla @ k).norm().item())

    # GLA's loop slot converges to the (large) accumulate fixed point kv/g;
    # delta's slot stays at the superseded write magnitude ~ b*v.
    assert slot_g[-1] > slot_d[-1] * 3
    # delta slot bounded near single-write magnitude, not the accumulated one
    fresh = torch.outer(k, v).norm().item()
    assert slot_d[-1] < fresh * 2.0


def test_delta_hybrid_incremental_parity():
    cfg = TorosHybridConfig(
        dim=64, d_byte=32, n_encoder_layers=2, n_heads=2,
        target_patch_size=8, time_mixer_rule="delta", dtype=torch.float32,
        use_rls_heads=False, use_type_codebook=False, dynamic_patching=False, use_growth=False
    )
    torch.manual_seed(42)
    m = TorosHybridLanguageModel(cfg).eval()
    x = torch.randint(1, 256, (1, 24))
    with torch.no_grad():
        full, _, _ = m.forward(x)
        inc, _ = m.forward_incremental(x, None, return_state=True)
    assert torch.allclose(full[:, -1], inc[:, -1], atol=5e-2)


def test_delta_hybrid_train_step():
    cfg = TorosHybridConfig(
        dim=64, d_byte=32, n_encoder_layers=2, n_heads=2,
        target_patch_size=8, time_mixer_rule="delta", dtype=torch.float32,
        use_rls_heads=False, use_type_codebook=False, dynamic_patching=False, use_growth=False
    )
    torch.manual_seed(42)
    m = TorosHybridLanguageModel(cfg)
    m.train()
    x = torch.randint(1, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    _, loss, met = m(x, targets=y)
    loss.backward()
    assert m.byte_decoder.lm_head.weight.grad is not None
    assert math.isfinite(met["loss_gen"])


def test_gla_layout_regression():
    """Sequential T>1 decode must interleave heads correctly (unsqueeze(2) fix).

    Pre-fix, outs held [B,1,H,D] slices; cat+transpose scrambled head/position
    layout for multi-step state-carry calls. H=4 makes the scramble visible.
    """
    m = _mixer("gla", n_heads=4)
    torch.manual_seed(9)
    x = torch.randn(1, 8, 64)
    with torch.no_grad():
        # single-step reference: decode one position at a time through state carries
        _, s = m(x[:, :1], state=None, return_state=True)
        ys = []
        for t in range(1, 8):
            y_t, s = m(x[:, t:t + 1], state=s, return_state=True)
            ys.append(y_t)
        one_by_one = torch.cat(ys, 1)
        chunked, _ = m(x[:, 1:], state=None, return_state=True)
    # compare against chunked call via carry from step 0's state
    with torch.no_grad():
        _, s0 = m(x[:, :1], state=None, return_state=True)
        chunked_from_s0, _ = m(x[:, 1:], state=s0, return_state=True)
    assert torch.allclose(one_by_one, chunked_from_s0, atol=1e-5)
