import torch
import pytest
from affine_ai.optim.muon import (
    Muon,
    HybridMuonAdamW,
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_batched,
)
from affine_ai.models.language_model import ASDAGLanguageModel


def test_ns5_single_vs_batched_parity():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        B, M, N = 4, 128, 64
        G_stack = torch.randn(B, M, N, device=device, dtype=torch.float32)

        out_batched = zeropower_via_newtonschulz5_batched(G_stack, steps=5)

        for i in range(B):
            out_single = zeropower_via_newtonschulz5(G_stack[i], steps=5)
            diff = (out_batched[i] - out_single).abs().max().item()
            # Tolerance 1e-2 allows for C++ AVX2 SIMD reduction order differences vs PyTorch bmm
            assert diff < 1e-2, f"Parity mismatch on {device}: max diff {diff}"


def test_hybrid_muon_adamw_parameter_classification():
    model = ASDAGLanguageModel(d_model=128, n_layers=2, n_heads=4)
    opt = HybridMuonAdamW(model)

    muon_params = set()
    if opt.muon_opt is not None:
        for group in opt.muon_opt.param_groups:
            muon_params.update(group["params"])

    adamw_params = set()
    if opt.adamw_opt is not None:
        for group in opt.adamw_opt.param_groups:
            adamw_params.update(group["params"])

    # All trainable parameters must be accounted for exactly once
    trainable_params = {p for p in model.parameters() if p.requires_grad}
    assert muon_params.union(adamw_params) == trainable_params
    assert muon_params.isdisjoint(adamw_params)

    # Every parameter in Muon must be a 2D matrix with both dimensions > 1
    for p in muon_params:
        assert p.ndim == 2
        assert p.shape[0] > 1 and p.shape[1] > 1

    # Embeddings must be in AdamW
    if hasattr(model, "tok_embeddings"):
        assert model.tok_embeddings.weight in adamw_params


def test_hybrid_muon_adamw_optimization_step():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ASDAGLanguageModel(d_model=128, n_layers=2, n_heads=4).to(device)
    opt = HybridMuonAdamW(model)

    x = torch.randint(0, 256, (2, 32), device=device)
    targets = torch.randint(0, 256, (2, 32), device=device)

    opt.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 256), targets.view(-1))
    loss.backward()
    opt.step()

    assert not torch.isnan(loss)


def test_muon_step_batched_grouping_parity():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        torch.manual_seed(42)
        shapes = [(32, 32), (32, 32), (32, 128), (128, 32), (32, 128), (16, 48)]
        params_batched = [torch.nn.Parameter(torch.randn(s, device=device)) for s in shapes]
        params_unbatched = [torch.nn.Parameter(p.clone().detach()) for p in params_batched]

        for p_b, p_u in zip(params_batched, params_unbatched):
            g = torch.randn_like(p_b)
            p_b.grad = g.clone()
            p_u.grad = g.clone()

        opt_batched = Muon(params_batched, lr=0.02, momentum=0.95, nesterov=True)
        opt_batched.step()

        # Compute unbatched step manually
        lr = 0.02
        momentum = 0.95
        for p in params_unbatched:
            buf = torch.zeros_like(p.grad)
            buf.mul_(momentum).add_(p.grad)
            update_grad = p.grad.add(buf, alpha=momentum)
            up = zeropower_via_newtonschulz5(update_grad, steps=5)
            p.data.add_(up, alpha=-lr)

        for i, (p_b, p_u) in enumerate(zip(params_batched, params_unbatched)):
            diff = (p_b - p_u).abs().max().item()
            assert diff < 1e-2, f"Parity mismatch on {device} for shape {shapes[i]}: diff {diff}"

