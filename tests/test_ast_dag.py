import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from affine_ai.core.ast_dag import (
    ASTDAGNode,
    ASTDAGLayer,
    EdgeType,
    bitlinear_add,
    ternarize,
)


def test_bitlinear_ternary_quantization():
    torch.manual_seed(0)
    w = torch.randn(16, 16, requires_grad=True)
    w_q = ternarize(w, threshold_frac=0.7)

    # Check ternary magnitude: entries are {-alpha, 0, +alpha}
    mag = w_q.detach().abs()
    nz = mag[mag > 1e-6]
    alpha = nz.median()
    assert torch.allclose(nz, alpha.expand_as(nz), atol=1e-5)
    assert alpha > 0

    # Check STE backward identity
    loss = w_q.sum()
    loss.backward()
    assert torch.equal(w.grad, torch.ones_like(w))


def test_bitlinear_add_computation():
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    w_ternary = torch.tensor([
        [1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0],
        [0.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0],
    ])
    out = bitlinear_add(w_ternary, x)
    assert out.shape == (4, 2)
    # Manual check first row, first column
    expected_0_0 = x[0, 0] - x[0, 1] + x[0, 3] - x[0, 5] + x[0, 7]
    assert torch.allclose(out[0, 0], expected_0_0, atol=1e-5)


def test_topology_invariant_and_node_initialization():
    dim = 8
    max_sec = 3
    node = ASTDAGNode(node_id=1, dim=dim, max_secondary=max_sec)
    assert node.node_id == 1
    assert node.dim == dim
    assert node.max_secondary == max_sec
    assert node.primary_parent is None
    assert len(node.secondary_parents) == 0
    assert len(node.child_nodes) == 0
    assert node.is_leaf is True
    assert node.accumulated_energy == 0.0


def test_gradient_invariant_secondary_stop_gradient():
    """
    Verifies that backpressure gradients propagate strictly to the primary parent,
    and are completely blocked (zero / detached) to secondary parents.
    """
    torch.manual_seed(42)
    dim = 8
    root = ASTDAGNode(node_id=0, dim=dim)
    root.is_leaf = False

    sec_node = ASTDAGNode(node_id=1, dim=dim)
    sec_node.primary_parent = root
    sec_node.latent_W_primary = nn.Parameter(torch.randn(dim, dim))

    leaf = ASTDAGNode(node_id=2, dim=dim, max_secondary=2)
    leaf.primary_parent = root
    leaf.secondary_parents.append(sec_node)
    leaf.W_context = nn.Parameter(torch.ones(2, dim))

    x = torch.randn(4, dim)

    # 1. Forward passes
    out_root = root.forward_pass(x)
    out_sec = sec_node.forward_pass(out_root)
    out_leaf = leaf.forward_pass(out_root)

    # 2. Local backward pass on leaf with local error
    local_error = torch.randn(4, dim)
    updates, delta_upstream, energy_sig = leaf.local_backward_pass(local_error)

    # Upstream delta strictly matches primary parent shape
    assert delta_upstream.shape == out_root.shape

    # Apply updates to parameters
    assert "latent_W_primary" in updates
    assert "W_context" in updates

    # Check that sec_node parameters did not receive any gradients from leaf backward pass
    assert sec_node.latent_W_primary.grad is None


def test_ast_dag_layer_forward_and_backward():
    torch.manual_seed(123)
    dim = 16
    layer = ASTDAGLayer(dim=dim, initial_branches=3, max_secondary=2)
    x = torch.randn(8, dim)

    layer.train()
    out = layer(x)
    assert out.shape == (8, dim)

    targets = torch.randn(8, dim)
    updates, metrics = layer.compute_backpressure_updates(targets, loss_type="mse")

    assert "router_weights" in updates
    assert "router_biases" in updates
    assert "input_pressure" in updates
    assert updates["input_pressure"].shape == (8, dim)
    assert metrics["num_leaves"] == 3
    assert metrics["mean_energy"] >= 0.0


def test_ast_dag_low_rank_factors():
    torch.manual_seed(123)
    dim = 16
    rank = 4
    layer = ASTDAGLayer(dim=dim, rank=rank, initial_branches=2)
    x = torch.randn(8, dim)

    layer.train()
    out = layer(x)
    assert out.shape == (8, dim)

    targets = torch.randn(8, dim)
    updates, metrics = layer.compute_backpressure_updates(targets, loss_type="mse")
    assert any("latent_U" in k for k in updates.keys())
    assert any("latent_V" in k for k in updates.keys())


def test_ast_dag_training_with_adamw():
    """
    Verifies that applying backpressure updates with AdamW optimizes the layer.
    """
    torch.manual_seed(0)
    dim = 8
    layer = ASTDAGLayer(dim=dim, initial_branches=2, activation="none")
    opt = torch.optim.AdamW(layer.parameters(), lr=0.05)

    x = torch.randn(32, dim)
    target = x @ torch.randn(dim, dim) * 0.5

    layer.train()
    first_loss = None
    for step in range(200):
        out = layer(x)
        loss = F.mse_loss(out, target)
        if first_loss is None:
            first_loss = loss.item()

        updates, _ = layer.compute_backpressure_updates(target, "mse")
        layer.apply_updates(updates, opt)
        opt.step()
        opt.zero_grad()

    final_loss = loss.item()
    assert final_loss < 0.5 * first_loss


def test_topology_engine_splitting_and_peeking():
    torch.manual_seed(42)
    dim = 8
    layer = ASTDAGLayer(
        dim=dim,
        tau_split=0.5,
        tau_peek=0.1,
        initial_branches=2,
        max_secondary=2,
    )
    leaves = layer.leaves
    assert len(leaves) == 2

    # Artificially set energy for leaf 0 to trigger split, leaf 1 to trigger peek
    leaves[0].accumulated_energy = 0.8  # > tau_split (0.5)
    leaves[1].accumulated_energy = 0.3  # tau_peek (0.1) < E < tau_split (0.5)

    stats = layer.step_topology(tau_split=0.5, tau_peek=0.1, n_branches=2)
    assert stats["splits"] == 1
    assert stats["peeks"] == 1
    # Leaf 0 was split into 2 children leaves; leaf 1 remains leaf -> total 3 leaves
    assert len(layer.leaves) == 3

    # Check peeking connection: leaf 1 should now have leaf 0 (or one of its children) in secondary_parents
    assert len(leaves[1].secondary_parents) == 1


def test_topology_engine_utility_pruning():
    torch.manual_seed(42)
    dim = 8
    layer = ASTDAGLayer(
        dim=dim,
        tau_prune=1.0,
        k_prune=2,
        initial_branches=4,
    )
    leaves = layer.leaves
    assert len(leaves) == 4

    # Set utility counters: leaves 0, 1 active, leaves 2, 3 inactive
    leaves[0].utility_counter = 5.0
    leaves[1].utility_counter = 5.0
    leaves[2].utility_counter = 0.0
    leaves[3].utility_counter = 0.0

    # Step 1: counter = 1 (not k_prune)
    stats1 = layer.step_topology(tau_prune=1.0)
    assert stats1["prunes"] == 0

    # Step 2: counter = 2 (triggers prune)
    stats2 = layer.step_topology(tau_prune=1.0)
    assert stats2["prunes"] == 2
    assert len(layer.leaves) == 2


def test_asdag_block_with_ast_dag():
    torch.manual_seed(0)
    from affine_ai.models.language_model import ASDAGBlock
    from affine_ai.core.ast_dag import ASDAGConfig
    d_model = 16
    config = ASDAGConfig(dim=d_model, num_leaves=4)
    block = ASDAGBlock(config)
    assert isinstance(block.asdag, ASTDAGLayer)

    x = torch.randn(2, 6, d_model)
    out = block(x)
    assert out.shape == x.shape


def test_asdag_lm_end_to_end(tmp_path):
    torch.manual_seed(0)
    from affine_ai.models.language_model import ASDAGLanguageModel
    model = ASDAGLanguageModel(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        num_leaves=4,
    )
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    input_ids = torch.randint(0, 32, (4, 8))
    logits = model(input_ids)
    assert logits.shape == (4, 8, 32)

    loss = F.cross_entropy(logits.reshape(-1, 32), torch.randint(0, 32, (4, 8)).reshape(-1))
    loss.backward()
    opt.step()

    # Save and load checkpoint
    path = str(tmp_path / "asdag_ckpt.pt")
    torch.save(model.state_dict(), path)

    loaded = ASDAGLanguageModel(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        num_leaves=4,
    )
    loaded.load_state_dict(torch.load(path))
    loaded.eval()
    with torch.no_grad():
        logits_loaded = loaded(input_ids)
    assert logits_loaded.shape == (4, 8, 32)


def test_bounded_context_gating_and_normalization():
    torch.manual_seed(42)
    dim = 8
    node = ASTDAGNode(node_id=1, dim=dim, max_secondary=2, bounded_gating=True, normalize_context=True)
    sec1 = ASTDAGNode(node_id=2, dim=dim)
    sec2 = ASTDAGNode(node_id=3, dim=dim)
    sec1.cached_output = torch.ones(4, dim) * 10.0
    sec2.cached_output = torch.ones(4, dim) * 10.0
    node.secondary_parents = [sec1, sec2]

    # Set very large context gate values -> tanh saturates to +1.0
    node.W_context.data.fill_(100.0)
    x = torch.zeros(4, dim)
    out = node.forward_pass(x)

    # Context contribution is tanh(100) * 10.0 = 10.0 per parent, total 20.0, scaled by 1/sqrt(3)
    # Plus bias (0), relu6(20/sqrt(3)) = 6.0
    assert torch.allclose(node.cached_pre_act, torch.ones(4, dim) * (20.0 / math.sqrt(3.0)), atol=1e-4)


def test_learnable_scale_gradients():
    torch.manual_seed(0)
    dim = 8
    layer = ASTDAGLayer(dim=dim, learnable_scale=True, initial_branches=2)
    x = torch.randn(8, dim)
    layer.train()
    out = layer(x)
    targets = torch.randn(8, dim)

    updates, _ = layer.compute_backpressure_updates(targets, "mse")
    assert any("scale_w" in k for k in updates.keys())


def test_topological_ordering_peeking():
    torch.manual_seed(42)
    dim = 8
    layer = ASTDAGLayer(dim=dim, tau_peek=0.1, tau_split=1.0, initial_branches=3)
    leaves = layer.leaves

    # Leaf 0 has topo_order 1, leaf 2 has topo_order 3
    # Only higher topo_order node (leaf 2) can peek from lower (leaf 0)
    leaves[0].accumulated_energy = 0.5
    leaves[2].accumulated_energy = 0.5

    layer.step_topology(tau_peek=0.1, tau_split=1.0)
    # Leaf 0 cannot peek at leaf 2 because 2 > 0
    assert len(leaves[0].secondary_parents) == 0
    # Leaf 2 can peek at leaf 0
    assert len(leaves[2].secondary_parents) == 1
    assert leaves[2].secondary_parents[0].node_id == leaves[0].node_id


def test_optimizer_state_migration_on_split():
    torch.manual_seed(0)
    dim = 8
    layer = ASTDAGLayer(dim=dim, initial_branches=2)
    opt = torch.optim.AdamW(layer.parameters(), lr=0.01)

    x = torch.randn(8, dim)
    layer.train()
    out = layer(x)
    loss = out.sum()
    loss.backward()
    opt.step()

    # Trigger split on leaf 0
    leaf0 = layer.leaves[0]
    leaf0.accumulated_energy = 5.0
    layer.step_topology(tau_split=1.0, optimizer=opt)

    # Check that new children inherited momentum state
    for child in leaf0.child_nodes:
        assert child.latent_W_primary in opt.state
        assert "exp_avg" in opt.state[child.latent_W_primary]


def test_forward_batched_dispatch_parity():
    torch.manual_seed(42)
    dim = 16
    layer = ASTDAGLayer(dim=dim, initial_branches=4, rank=None)
    x = torch.randn(12, dim)

    layer.eval()
    with torch.no_grad():
        out_standard = layer(x)
        out_batched = layer.forward_batched_dispatch(x)

    assert torch.allclose(out_standard, out_batched, atol=1e-5)


def test_ast_dag_training_with_hybrid_muon_adamw():
    torch.manual_seed(0)
    dim = 8
    layer = ASTDAGLayer(dim=dim, initial_branches=2, activation="none")
    opt = layer.get_default_optimizer(muon_lr=0.04, adamw_lr=4e-3)

    x = torch.randn(32, dim)
    target = x @ torch.randn(dim, dim) * 0.5

    layer.train()
    first_loss = None
    for step in range(100):
        out = layer(x)
        loss = F.mse_loss(out, target)
        if first_loss is None:
            first_loss = loss.item()

        updates, _ = layer.compute_backpressure_updates(target, "mse")
        layer.apply_updates(updates, opt)
        opt.step()
        opt.zero_grad()

    final_loss = loss.item()
    assert final_loss < 0.5 * first_loss


def test_ast_dag_structured_nm_sparsity():
    torch.manual_seed(42)
    dim = 16
    # 2:4 structured sparsity
    layer = ASTDAGLayer(dim=dim, initial_branches=2, nm=(2, 4), rank=None)
    for leaf in layer.leaves:
        mask = leaf.sparsity_mask
        assert mask.shape == (dim, dim)
        # Check that along last dim, every 4 contiguous elements has exactly 2 active
        g = mask.reshape(-1, 4)
        assert torch.all(g.sum(dim=-1) == 2.0)

    # 1:8 structured sparsity
    layer_18 = ASTDAGLayer(dim=dim, initial_branches=2, nm=(1, 8), rank=None)
    for leaf in layer_18.leaves:
        g = leaf.sparsity_mask.reshape(-1, 8)
        assert torch.all(g.sum(dim=-1) == 1.0)


def test_ast_dag_resparsify_under_weight_decay():
    torch.manual_seed(0)
    dim = 8
    layer = ASTDAGLayer(dim=dim, initial_branches=2, nm=(2, 4), rank=None)
    leaf = layer.leaves[0]
    orig_mask = leaf.sparsity_mask.clone()

    # Introduce weight decay noise at masked positions
    with torch.no_grad():
        leaf.latent_W_primary.data.fill_(1.0)

    # Calling resparsify should re-zero masked positions
    layer.resparsify()
    assert torch.all(leaf.latent_W_primary.data[orig_mask == 0.0] == 0.0)


def test_ast_dag_sparsity_redistribution():
    torch.manual_seed(42)
    dim = 16
    layer = ASTDAGLayer(dim=dim, initial_branches=2, nm=(2, 4), rank=None)
    x = torch.randn(8, dim)
    layer.train()
    out = layer(x)
    targets = torch.randn(8, dim)
    updates, _ = layer.compute_backpressure_updates(targets, "mse")

    # Redistribute sparsity
    layer.redistribute_sparsity(drop_fraction=0.5, seed=42)
    for leaf in layer.leaves:
        g = leaf.sparsity_mask.reshape(-1, 4)
        assert torch.all(g.sum(dim=-1) == 2.0)


def test_ast_dag_lm_with_nm_sparsity():
    torch.manual_seed(0)
    from affine_ai.models.language_model import ASDAGLanguageModel
    model = ASDAGLanguageModel(
        vocab_size=32,
        d_model=16,
        n_layers=2,
        num_leaves=4,
        sparsity_ratio=0.875
    )
    input_ids = torch.randint(0, 32, (2, 8))
    logits = model(input_ids)
    assert logits.shape == (2, 8, 32)


def test_shift4_logarithmic_activation_quantization():
    torch.manual_seed(42)
    from affine_ai.core.ast_dag import quantize_shift4
    # Gaussian distributed activations
    x = torch.randn(10, 16, requires_grad=True)
    q = quantize_shift4(x)

    # Output should retain shape and gradients flow via STE
    loss = q.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape

    # Check power-of-two properties (normalized values are +/- 2^-p)
    scale = x.abs().amax(dim=-1, keepdim=True)
    q_norm = (q / scale).abs()
    nonzero = q_norm[q_norm > 0]
    log_vals = torch.log2(nonzero)
    assert torch.allclose(log_vals, log_vals.round(), atol=1e-5)


def test_power_of_two_context_gates():
    torch.manual_seed(42)
    from affine_ai.core.ast_dag import quantize_power_of_two_gate
    w = torch.randn(4, 16, requires_grad=True)
    q = quantize_power_of_two_gate(w)
    loss = q.sum()
    loss.backward()
    assert w.grad is not None


def test_hierarchical_sign_router():
    torch.manual_seed(42)
    from affine_ai.core.ast_dag import HierarchicalSignRouter
    router = HierarchicalSignRouter(dim=16, num_leaves=8)
    x = torch.randn(4, 16)
    probs, logits = router.route_tokens(x)
    assert probs.shape == (4, 8)
    # Probabilities should sum to 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-4)


def test_homomorphic_subtree_merging():
    torch.manual_seed(42)
    dim = 8
    layer = ASTDAGLayer(dim=dim, initial_branches=4, tau_merge=0.9)
    leaves = layer.leaves

    # Force leaf 0 and leaf 1 to have identical weights (collinear)
    with torch.no_grad():
        leaves[1].latent_W_primary.data.copy_(leaves[0].latent_W_primary.data)
        leaves[1].primary_parent = leaves[0].primary_parent

    # Step topology with tau_merge=0.9
    stats = layer.step_topology(tau_split=100.0, tau_peek=100.0, tau_merge=0.9)
    assert stats["merges"] >= 1


def test_sign_backpressure_updates():
    torch.manual_seed(0)
    dim = 8
    layer = ASTDAGLayer(dim=dim, initial_branches=2)
    x = torch.randn(8, dim)
    layer.train()
    out = layer(x)
    targets = torch.randn(8, dim)

    updates, _ = layer.compute_backpressure_updates(targets, "mse", use_sign_backpressure=True)
    assert "node_1_latent_W_primary" in updates


def test_permutation_leaves():
    torch.manual_seed(42)
    dim = 16
    layer = ASTDAGLayer(dim=dim, initial_branches=4, leaf_mode="permutation", num_permutations=4)
    x = torch.randn(6, dim, requires_grad=True)
    layer.train()
    out = layer(x)
    assert out.shape == (6, dim)

    # Test backpropagation
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (6, dim)

    # Test parity with batched dispatch
    layer.eval()
    with torch.no_grad():
        out1 = layer(x)
        out2 = layer.forward_batched_dispatch(x)
        assert torch.allclose(out1, out2, atol=1e-5)





