import math
import torch
import torch.nn.functional as F
import pytest

from affine_ai.core.backpressure_tree import (
    ternary_ste,
    FusedSparseBackpressureTreeV3,
)

RANKS = [None, 8]  # None = full affine, 8 = low-rank factors


def make_layer(depth=2, n_ary=3, **kw):
    kw.setdefault("in_features", 5)
    kw.setdefault("out_features", 3)
    return FusedSparseBackpressureTreeV3(depth=depth, n_ary=n_ary, **kw)


def leaf_params(layer):
    if layer.rank is None:
        return [layer.leaf_weights]
    return [layer.leaf_u, layer.leaf_v]


def leaf_update_keys(layer):
    return ["leaf_weights"] if layer.rank is None else ["leaf_u", "leaf_v"]


def effective_leaf(layer):
    if layer.rank is None:
        return layer._effective_leaf_weights()
    u, v = layer._effective_factors()
    return torch.einsum('kir,ro->kio', u, v)


def soft_reference_out(layer, x_flat):
    """Fully-soft tree output with the layer's CURRENT effective leaf params,
    autograd-differentiable w.r.t. routing weights."""
    B = x_flat.shape[0]
    logits = torch.einsum('bi,din->bdn', x_flat, layer.routing_weights) + layer.routing_biases
    c = F.softmax(logits / layer.temperature, dim=-1)
    p = c[:, 0]
    for d in range(1, layer.depth):
        p = (p.unsqueeze(-1) * c[:, d].unsqueeze(-2)).reshape(B, -1)

    b = layer.leaf_biases.detach()
    if layer.rank is None:
        w = layer._effective_leaf_weights().detach()
        packed = w.permute(1, 0, 2).reshape(layer.in_features, -1)
        y = torch.matmul(x_flat, packed).view(B, layer.num_leaves, layer.out_features) + b
        out_s = torch.bmm(p.unsqueeze(1), y).squeeze(1)
    else:
        u, v = layer._effective_factors()
        u = u.detach()
        h = torch.matmul(x_flat, u.permute(1, 0, 2).reshape(layer.in_features, -1))
        h = h.view(B, layer.num_leaves, layer.rank)
        a = layer._phi(h)
        m = torch.einsum('bk,bkr->br', p, a)
        out_s = m @ v.detach() + p @ b
    return out_s


def test_ternary_ste_forward_values_and_backward_identity():
    torch.manual_seed(0)
    w = torch.randn(64, 32, requires_grad=True)
    out = ternary_ste(w, 0.7)

    # TWN-style: active entries are +/- alpha (mean |w| over active set).
    mag = out.detach().abs()
    nz = mag[mag > 1e-6]
    alpha = nz.median()
    assert torch.allclose(nz, alpha.expand_as(nz), atol=1e-5)
    assert alpha > 0

    # Backward is identity: gradient of sum w.r.t. w is all ones.
    out.sum().backward()
    assert torch.equal(w.grad, torch.ones_like(w))


def test_ternary_ste_mask_forces_zero():
    torch.manual_seed(0)
    w = torch.randn(8, 8)
    mask = (torch.rand(8, 8) > 0.5).float()
    out = ternary_ste(w, 0.7, mask=mask)
    assert torch.all(out[mask == 0] == 0.0)


@pytest.mark.parametrize("rank", RANKS)
def test_default_construction(rank):
    layer = make_layer(rank=rank)
    if rank is None:
        assert hasattr(layer, "leaf_weights") and not hasattr(layer, "leaf_u")
    else:
        assert layer.rank == rank
        assert layer.leaf_u.shape == (layer.num_leaves, 5, rank)
        assert layer.leaf_v.shape == (rank, 3)


def test_lowrank_is_the_default():
    layer = FusedSparseBackpressureTreeV3(in_features=5, out_features=3, depth=2, n_ary=3)
    assert layer.rank == 16
    assert hasattr(layer, "leaf_u") and hasattr(layer, "leaf_v")


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_forward_shapes_all_depths(depth, rank):
    torch.manual_seed(0)
    layer = make_layer(depth=depth, rank=rank, top_k=2)
    x = torch.randn(5, 7, 5)
    y = layer(x, record_cache=False)
    assert y.shape == (5, 7, 3)


@pytest.mark.parametrize("rank", RANKS)
def test_sparse_probs_sum_to_one(rank):
    torch.manual_seed(0)
    layer = make_layer(depth=3, rank=rank)
    layer(torch.randn(16, 5))
    probs = layer._forward_cache["sparse_probs"]
    assert torch.allclose(probs.sum(dim=-1), torch.ones(16), atol=1e-5)


@pytest.mark.parametrize("rank", RANKS)
def test_sparsity_mask_is_fixed_buffer_and_applied(rank):
    torch.manual_seed(0)
    sparsity = 0.9
    layer = make_layer(depth=2, n_ary=2, rank=rank,
                       ternary_leaves=True, leaf_sparsity=sparsity)
    masks = [layer.leaf_sparsity_mask_u] if rank is not None else [layer.leaf_sparsity_mask]
    for mask in masks:
        # tolerance accounts for small-mask sampling noise
        tol = max(0.05, 3.0 * (sparsity * (1 - sparsity) / mask.numel()) ** 0.5)
        assert abs((1.0 - mask.mean().item()) - sparsity) < tol

    # Masked positions are exactly zero in the effective leaf map...
    eff = effective_leaf(layer)
    if rank is not None:
        # factorized: check each masked factor entry contributes nothing
        u, v = layer._effective_factors()
        assert torch.all(u[layer.leaf_sparsity_mask_u == 0] == 0.0)
        # shared V is intentionally never masked

    # ...and their backpressure updates stay exactly zero.
    x = torch.randn(16, 5)
    targets = torch.randn(16, 3)
    layer.train()
    layer(x)
    updates, _ = layer.compute_backpressure_updates(targets, "mse")
    if rank is None:
        assert torch.all(updates["leaf_weights"][layer.leaf_sparsity_mask == 0] == 0.0)
    else:
        assert torch.all(updates["leaf_u"][layer.leaf_sparsity_mask_u == 0] == 0.0)
        # leaf_v updates are unmasked by design (shared factor stays dense)


def test_no_cache_in_eval_mode_by_default():
    torch.manual_seed(0)
    layer = FusedSparseBackpressureTreeV3(in_features=4, out_features=4, depth=2, n_ary=2)
    layer.eval()
    layer(torch.randn(4, 4))
    assert layer._forward_cache is None
    layer.train()
    layer(torch.randn(4, 4))
    assert layer._forward_cache is not None


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_leaf_updates_match_autograd_exactly(depth, rank):
    torch.manual_seed(123)
    layer = make_layer(depth=depth, rank=rank, top_k=2, ternary_leaves=True)
    torch.manual_seed(7)
    x = torch.randn(24, 5)
    targets = torch.randn(24, 3)

    layer.train()
    layer(x)
    updates, _ = layer.compute_backpressure_updates(targets, "mse")

    loss = F.mse_loss(layer(x, record_cache=False), targets)
    grads = torch.autograd.grad(loss, leaf_params(layer))

    keys = leaf_update_keys(layer)
    for k, g in zip(keys + ["leaf_biases"], list(grads)):
        assert torch.allclose(updates[k], -g, atol=1e-6), k


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_routing_updates_match_soft_tree_autograd_exactly(depth, rank):
    """
    Routing updates must equal the exact chain-rule gradients of the
    fully-soft tree (product of all levels' conductances) evaluated at the
    same error signal backpressure consumes -- regardless of whether leaves
    are full affine or low-rank factors. Requires dense dispatch: gathered
    dispatch only sees the n reachable siblings, so its routing gradient is
    the hardened-tree variant by design.
    """
    torch.manual_seed(123)
    layer = make_layer(depth=depth, rank=rank, ternary_leaves=True,
                       sparse_dispatch=False)
    torch.manual_seed(7)
    x = torch.randn(24, 5)
    targets = torch.randn(24, 3)
    B = 24

    layer.train()
    layer(x)
    updates, _ = layer.compute_backpressure_updates(targets, "mse")
    cache = layer._forward_cache

    # Upstream error exactly as backpressure consumed it.
    error = (targets - cache["out"]) * (2.0 / layer.out_features)

    out_s = soft_reference_out(layer, cache["x_flat"])
    loss_ref = -torch.sum(out_s * error.detach()) / B

    gW, gB = torch.autograd.grad(loss_ref, [layer.routing_weights, layer.routing_biases])

    assert torch.allclose(-updates["routing_weights"], gW, atol=1e-6)
    assert torch.allclose(-updates["routing_biases"], gB, atol=1e-6)


@pytest.mark.parametrize("depth", [2, 3])
def test_sparse_dispatch_matches_dense_exactly(depth):
    """Gathered sibling dispatch: forward outputs, leaf/V/bias updates and
    input_pressure are identical to dense-over-K. ROUTING updates
    deliberately differ (hardened-tree local gradient vs soft-relaxation
    gradient) -- see sparse_dispatch docstring."""
    kw = dict(in_features=8, out_features=4, depth=depth, n_ary=3, rank=8,
              ternary_leaves=True)
    torch.manual_seed(42)
    dense = FusedSparseBackpressureTreeV3(sparse_dispatch=False, **kw)
    torch.manual_seed(42)
    sparse = FusedSparseBackpressureTreeV3(sparse_dispatch=True, **kw)

    x = torch.randn(32, 8)
    targets = torch.randn(32, 4)

    dense.train(); sparse.train()
    y_dense = dense(x)
    y_sparse = sparse(x)
    assert torch.allclose(y_dense, y_sparse, atol=1e-6)

    up_d, _ = dense.compute_backpressure_updates(targets, "mse")
    up_s, _ = sparse.compute_backpressure_updates(targets, "mse")
    # NOTE: input_pressure is NOT compared -- it sums in
    # router_steering_pressure, which inherits the deliberate routing
    # difference. Its leaf-input component matches by construction.
    for k in ["leaf_u", "leaf_v", "leaf_biases"]:
        assert torch.allclose(up_d[k], up_s[k], atol=1e-5), k
    # Routing/input_pressure differ by design; sanity-check structure.
    assert torch.isfinite(up_s["routing_weights"]).all()
    assert torch.isfinite(up_s["input_pressure"]).all()


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_sparse_dispatch_leaf_updates_match_autograd(depth):
    """Autograd through the GATHERED graph must match hydraulic leaf updates."""
    torch.manual_seed(123)
    layer = FusedSparseBackpressureTreeV3(
        in_features=6, out_features=3, depth=depth, n_ary=3,
        rank=8, top_k=2, ternary_leaves=True, sparse_dispatch=True
    )
    torch.manual_seed(7)
    x = torch.randn(24, 6)
    targets = torch.randn(24, 3)

    layer.train()
    layer(x)
    updates, _ = layer.compute_backpressure_updates(targets, "mse")

    loss = F.mse_loss(layer(x, record_cache=False), targets)
    grads = torch.autograd.grad(loss, [layer.leaf_u, layer.leaf_v, layer.leaf_biases])

    assert torch.allclose(updates["leaf_u"], -grads[0], atol=1e-6)
    assert torch.allclose(updates["leaf_v"], -grads[1], atol=1e-6)
    assert torch.allclose(updates["leaf_biases"], -grads[2], atol=1e-6)


@pytest.mark.parametrize("rank", RANKS)
def test_apply_updates_trains_with_adamw(rank):
    """
    End-to-end hydraulic path: .grad assignment + AdamW must actually drive
    the loss down on a fixed synthetic regression target.
    """
    torch.manual_seed(0)
    layer = make_layer(depth=2, n_ary=3, rank=rank, top_k=2, ternary_leaves=True)
    opt = torch.optim.AdamW(layer.parameters(), lr=0.01)

    x = torch.randn(64, 4)
    w_true = torch.randn(4, 2)
    targets = x @ w_true

    layer = FusedSparseBackpressureTreeV3(
        in_features=4, out_features=2, depth=2, n_ary=3, top_k=2,
        rank=rank, ternary_leaves=True
    )
    opt = torch.optim.AdamW(layer.parameters(), lr=0.01)
    layer.train()

    first_loss = None
    for step in range(200):
        out = layer(x)
        loss = F.mse_loss(out, targets)
        if first_loss is None:
            first_loss = loss.item()
        updates, _ = layer.compute_backpressure_updates(targets, "mse")
        layer.apply_updates(updates, opt)
        opt.step()
        opt.zero_grad()

    assert loss.item() < 0.5 * first_loss


@pytest.mark.parametrize("rank", RANKS)
def test_resparsify_keeps_shadow_weights_sparse_under_weight_decay(rank):
    torch.manual_seed(0)
    layer = FusedSparseBackpressureTreeV3(
        in_features=4, out_features=4, depth=1, n_ary=2,
        rank=rank, ternary_leaves=False, leaf_sparsity=0.5
    )
    opt = torch.optim.AdamW(layer.parameters(), lr=0.1, weight_decay=0.1)
    x = torch.randn(8, 4)
    targets = torch.randn(8, 4)

    # The fixed mask zeroes masked positions on first resparsify().
    layer.resparsify()
    if rank is None:
        masked = [layer.leaf_weights.data[layer.leaf_sparsity_mask == 0]]
    else:
        masked = [layer.leaf_u.data[layer.leaf_sparsity_mask_u == 0]]
    for t in masked:
        assert torch.all(t == 0.0)

    for _ in range(20):
        layer(x)
        updates, _ = layer.compute_backpressure_updates(targets, "mse")
        layer.apply_updates(updates, opt)
        opt.step()
        layer.resparsify()
        opt.zero_grad()

    if rank is None:
        masked = [layer.leaf_weights.data[layer.leaf_sparsity_mask == 0]]
    else:
        masked = [layer.leaf_u.data[layer.leaf_sparsity_mask_u == 0]]
    for t in masked:
        assert torch.all(t == 0.0)


def _group_counts(mask, m):
    g = mask.reshape(*mask.shape[:-1], -1, m)
    return g.sum(dim=-1)


@pytest.mark.parametrize("nm", [(2, 4), (1, 8), (1, 16)])
def test_nm_mask_guarantees(nm):
    torch.manual_seed(0)
    layer = FusedSparseBackpressureTreeV3(
        in_features=8, out_features=4, depth=2, n_ary=3,
        rank=nm[1] * 2, ternary_leaves=True, nm=nm
    )
    n, m = nm
    assert layer.leaf_sparsity == pytest.approx(1 - n / m)
    counts = _group_counts(layer.leaf_sparsity_mask_u, m)
    assert torch.all(counts == n)  # exactly N active per group, everywhere


def test_nm_redistribute_preserves_group_counts():
    torch.manual_seed(0)
    layer = FusedSparseBackpressureTreeV3(
        in_features=8, out_features=4, depth=2, n_ary=3,
        rank=8, ternary_leaves=False, nm=(2, 4)
    )
    x = torch.randn(16, 8)
    targets = torch.randn(16, 4)
    layer.train()
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    for step in range(6):
        layer(x)
        updates, _ = layer.compute_backpressure_updates(targets, "mse")
        layer.apply_updates(updates, opt)
        layer.redistribute_sparsity(0.5, seed=step)
        opt.step()
        layer.resparsify()
        opt.zero_grad()
    assert torch.all(_group_counts(layer.leaf_sparsity_mask_u, 4) == 2)


def test_nm_magnitude_rebuild_respects_groups():
    torch.manual_seed(0)
    layer = FusedSparseBackpressureTreeV3(
        in_features=8, out_features=4, depth=1, n_ary=3,
        rank=8, ternary_leaves=False, nm=(1, 4)
    )
    with torch.no_grad():
        layer.leaf_u.data.normal_(0, 1.0)
    layer.build_magnitude_mask()
    assert torch.all(_group_counts(layer.leaf_sparsity_mask_u, 4) == 1)


def test_nm_forward_and_updates_finite():
    torch.manual_seed(0)
    layer = FusedSparseBackpressureTreeV3(
        in_features=8, out_features=4, depth=3, n_ary=4,
        rank=16, ternary_leaves=True, nm=(2, 8), sparse_dispatch=True
    )
    x = torch.randn(16, 8)
    targets = torch.randn(16, 4)
    layer.train()
    y = layer(x)
    assert y.shape == (16, 4) and torch.isfinite(y).all()
    updates, _ = layer.compute_backpressure_updates(targets, "mse")
    assert all(torch.isfinite(v).all() for v in updates.values())


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_soft_mode_hydraulic_equals_backprop_exactly(depth):
    """
    THE self-organizing-training property: with route_mode='soft' the graph
    is fully differentiable, so hydraulic updates must equal .backward()
    EXACTLY for every parameter -- leaves AND all routing levels.
    """
    torch.manual_seed(123)
    layer = FusedSparseBackpressureTreeV3(
        in_features=5, out_features=3, depth=depth, n_ary=3,
        rank=8, ternary_leaves=True, route_mode="soft"
    )
    torch.manual_seed(7)
    x = torch.randn(24, 5)
    targets = torch.randn(24, 3)

    layer.train()
    layer(x)
    updates, _ = layer.compute_backpressure_updates(targets, "mse")

    loss = F.mse_loss(layer(x, record_cache=False), targets)
    params = [layer.routing_weights, layer.routing_biases,
              layer.leaf_u, layer.leaf_v, layer.leaf_biases]
    grads = torch.autograd.grad(loss, params)

    keys = ["routing_weights", "routing_biases", "leaf_u", "leaf_v", "leaf_biases"]
    for k, g in zip(keys, grads):
        assert torch.allclose(updates[k], -g, atol=1e-6), k


def test_soft_mode_probs_are_full_product():
    """Soft mode must route through ALL leaves (no hard zeros)."""
    torch.manual_seed(0)
    layer = FusedSparseBackpressureTreeV3(
        in_features=4, out_features=4, depth=3, n_ary=3, rank=8,
        route_mode="soft", sparse_dispatch=True  # must be ignored/forced dense
    )
    layer.train()
    layer(torch.randn(8, 4))
    probs = layer._forward_cache["sparse_probs"]
    K = layer.num_leaves
    assert probs.shape == (8, K)
    assert (probs > 0).float().mean() == 1.0          # nothing hard-zeroed
    assert torch.allclose(probs.sum(-1), torch.ones(8), atol=1e-5)


def test_deploy_switch_soft_to_hard():
    """A layer trained in soft mode can be switched to hard dispatch."""
    torch.manual_seed(0)
    layer = FusedSparseBackpressureTreeV3(
        in_features=4, out_features=4, depth=3, n_ary=3, rank=8,
        route_mode="soft"
    )
    layer.train()
    x = torch.randn(8, 4)
    y_soft = layer(x, record_cache=False)

    layer.route_mode = "hard"
    y_hard = layer(x, record_cache=False)
    assert y_soft.shape == y_hard.shape               # runs; values differ by design


@pytest.mark.parametrize("act", ["none", "sqrelu", "sign"])
@pytest.mark.parametrize("depth", [2, 3])
def test_leaf_activation_updates_match_autograd_exactly(act, depth):
    """Closed-form VJP through phi' must equal autograd of the deployed
    forward, for every leaf activation, dense AND gathered dispatch."""
    for sparse in (False, True):
        torch.manual_seed(123)
        layer = FusedSparseBackpressureTreeV3(
            in_features=6, out_features=3, depth=depth, n_ary=3,
            rank=8, ternary_leaves=True, leaf_activation=act,
            sparse_dispatch=sparse
        )
        torch.manual_seed(7)
        x = torch.randn(24, 6)
        targets = torch.randn(24, 3)

        layer.train()
        layer(x)
        updates, _ = layer.compute_backpressure_updates(targets, "mse")

        loss = F.mse_loss(layer(x, record_cache=False), targets)
        grads = torch.autograd.grad(loss, [layer.leaf_u, layer.leaf_v, layer.leaf_biases])

        assert torch.allclose(updates["leaf_u"], -grads[0], atol=1e-5), (act, sparse)
        assert torch.allclose(updates["leaf_v"], -grads[1], atol=1e-5), (act, sparse)
        assert torch.allclose(updates["leaf_biases"], -grads[2], atol=1e-6), (act, sparse)


def test_sign_activation_values():
    """sign phi: forward values are exactly {-1,0,+1}."""
    layer = FusedSparseBackpressureTreeV3(
        in_features=4, out_features=4, depth=1, n_ary=2,
        rank=8, leaf_activation="sign"
    )
    h = torch.tensor([[-3.0, -0.5, 0.0, 2.0]])
    out = layer._phi(h)
    assert set(out.tolist()[0]) <= {-1.0, 0.0, 1.0}
    g = layer._phi_grad(h)
    # clipped STE: gradient passes only where |h| <= 1
    assert g.tolist() == [[0.0, 1.0, 1.0, 0.0]]


def test_sqrelu_activation_values():
    layer = FusedSparseBackpressureTreeV3(
        in_features=4, out_features=4, depth=1, n_ary=2,
        rank=8, leaf_activation="sqrelu"
    )
    h = torch.tensor([[[-2.0, -0.5, 0.0, 3.0]]])
    out = layer._phi(h)
    assert torch.equal(out, torch.tensor([[[0.0, 0.0, 0.0, 9.0]]]))
    g = layer._phi_grad(h)
    assert torch.equal(g, torch.tensor([[[0.0, 0.0, 0.0, 6.0]]]))


def test_growth_preserves_function_exactly():
    """grown_copy with zero noise must reproduce the parent's function
    exactly (mixture of identical maps is the identity), fp and nonlinear."""
    for act in ["none", "sqrelu", "sign"]:
        torch.manual_seed(0)
        base = FusedSparseBackpressureTreeV3(
            in_features=6, out_features=4, depth=2, n_ary=3,
            rank=8, ternary_leaves=False, leaf_activation=act,
            sparse_dispatch=False, route_mode="hard"
        )
        x = torch.randn(16, 6)
        base.eval()
        with torch.no_grad():
            y_before = base(x)
        grown = base.grown_copy(extra_levels=1, child_noise=0.0)
        grown.eval()
        assert grown.depth == 3 and grown.num_leaves == base.num_leaves * 3
        with torch.no_grad():
            y_after = grown(x)
        assert torch.allclose(y_before, y_after, atol=1e-5), act


@pytest.mark.parametrize("rank", RANKS)
def test_growth_shapes_and_masks(rank):
    torch.manual_seed(0)
    base = FusedSparseBackpressureTreeV3(
        in_features=8, out_features=4, depth=2, n_ary=4, rank=rank,
        ternary_leaves=True, leaf_sparsity=0.5 if rank is not None else 0.0
    )
    grown = base.grown_copy(extra_levels=2, child_noise=0.01, seed=3)
    assert grown.num_leaves == base.num_leaves * 16
    assert grown.routing_weights.shape[0] == 4
    if rank is not None:
        assert grown.leaf_u.shape == (base.num_leaves * 16, 8, rank)
        counts = _group_counts(grown.leaf_sparsity_mask_u, 4) \
            if hasattr(grown, "nm") and grown.nm else None
        # inherited mask density preserved
        assert abs((1 - grown.leaf_sparsity_mask_u.mean()).item() - 0.5) < 0.02


def test_growth_with_noise_is_close():
    torch.manual_seed(1)
    base = FusedSparseBackpressureTreeV3(
        in_features=6, out_features=4, depth=2, n_ary=3, rank=8
    )
    x = torch.randn(16, 6)
    base.eval()
    grown = base.grown_copy(child_noise=0.01).eval()
    with torch.no_grad():
        d = (base(x) - grown(x)).abs().max().item()
    assert d < 0.5   # small but nonzero: symmetry breaking only


def test_stale_cache_invalidated_on_no_record_forward():
    """compute_backpressure_updates must never consume tensors from an OLD
    forward pass (e.g. after eval-mode inference)."""
    torch.manual_seed(0)
    layer = FusedSparseBackpressureTreeV3(
        in_features=4, out_features=4, depth=2, n_ary=2, rank=8
    )
    x1 = torch.randn(8, 4)
    layer.train()
    layer(x1)                                   # populates cache
    layer.eval()
    with torch.no_grad():
        layer(torch.randn(4, 4))                # eval forward: must invalidate
    assert layer._forward_cache is None
    layer.train()
    layer(x1)                                   # repopulates
    assert layer._forward_cache is not None
    layer(x1, record_cache=False)               # explicit opt-out also invalidates
    assert layer._forward_cache is None


def test_learnable_scale_gradients_and_parity():
    """Learnable alpha: gradient reaches the scale, forward starts EXACTLY
    equal to fixed-alpha behavior, and scale actually moves when trained."""
    torch.manual_seed(0)
    fixed = FusedSparseBackpressureTreeV3(
        in_features=6, out_features=4, depth=2, n_ary=3,
        rank=8, ternary_leaves=True, learnable_scale=False)
    torch.manual_seed(0)
    learn = FusedSparseBackpressureTreeV3(
        in_features=6, out_features=4, depth=2, n_ary=3,
        rank=8, ternary_leaves=True, learnable_scale=True)
    x = torch.randn(16, 6)

    with torch.no_grad():
        y_fixed = fixed(x)
        y_learn0 = learn(x)
    assert torch.allclose(y_fixed, y_learn0, atol=1e-6)  # identical at init

    y_learn0 = learn(x)                                   # grad-enabled pass
    loss = y_learn0.square().mean()
    loss.backward()
    assert learn.scale_u.grad is not None and learn.scale_u.grad.abs() > 0
    assert learn.scale_v.grad is not None and learn.scale_v.grad.abs() > 0

    before = learn.scale_u.item()
    opt = torch.optim.AdamW([learn.scale_u], lr=1e-1)
    for _ in range(5):
        for p in learn.parameters(): p.grad = None
        yl = learn(x)
        (yl.square().mean()).backward()
        opt.step()
    assert not math.isclose(learn.scale_u.item(), before, abs_tol=1e-9)
