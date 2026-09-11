import pytest
import torch
import torch.nn.functional as F
import numpy as np

from affine_ai.kernels.triton_lpc import triton_fused_lpc_head
from affine_ai.kernels import TRITON_AVAILABLE
from affine_ai.core.lpc import LocalPredictiveHead, LocalPredictiveLanguageModel
from affine_ai.models.language_model import ASDAGLanguageModel
from affine_ai.training.trainer import ASDAGTrainer


def test_triton_lpc_head_cpu_fallback():
    """Verify CPU fallback for LPC head."""
    N, D, V = 32, 64, 128
    h = torch.randn(N, D, requires_grad=True)
    w = torch.randn(V, D, requires_grad=True)
    targets = torch.randint(0, V, (N,))

    ref_logits = F.linear(h, w)
    ref_loss = F.cross_entropy(ref_logits, targets)
    ref_loss.backward()

    ref_dh = h.grad.clone()
    ref_dw = w.grad.clone()

    h.grad.zero_()
    w.grad.zero_()

    lpc_loss = triton_fused_lpc_head(h, w, targets)
    lpc_loss.backward()

    assert torch.allclose(lpc_loss, ref_loss, atol=1e-4)
    assert torch.allclose(h.grad, ref_dh, atol=1e-4)
    assert torch.allclose(w.grad, ref_dw, atol=1e-4)


try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA & Triton required")
@pytest.mark.parametrize("V", [256, 2048])
def test_triton_lpc_head_cuda_parity(V):
    """Verify GPU Triton LPC kernel matches PyTorch reference for V=256 and V=2048."""
    B, T, D = 4, 32, 64
    torch.manual_seed(42)
    h = torch.randn(B, T, D, device="cuda", requires_grad=True)
    w = torch.randn(V, D, device="cuda", requires_grad=True)
    targets = torch.randint(0, V, (B, T), device="cuda")

    # PyTorch reference
    ref_logits = F.linear(h, w)
    ref_loss = F.cross_entropy(ref_logits.view(-1, V), targets.view(-1))
    ref_loss.backward()

    ref_dh = h.grad.clone()
    ref_dw = w.grad.clone()

    h.grad.zero_()
    w.grad.zero_()

    # Triton Fused LPC Head
    triton_loss = triton_fused_lpc_head(h, w, targets)
    triton_loss.backward()

    assert torch.allclose(triton_loss, ref_loss, atol=2e-3, rtol=2e-3)
    assert torch.allclose(h.grad, ref_dh, atol=2e-3, rtol=2e-3)
    assert torch.allclose(w.grad, ref_dw, atol=2e-3, rtol=2e-3)


def test_local_predictive_language_model():
    """Verify LocalPredictiveLanguageModel forward and step execution."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = ASDAGLanguageModel(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        channel_mixer_type="ternary_swiglu",
        use_blt=False
    ).to(device)

    lpc_model = LocalPredictiveLanguageModel(base_model).to(device)
    optimizers = lpc_model.get_default_lpc_optimizers(lr=1e-3)

    assert len(optimizers) == 4 # 2 blocks + enc tail + final head

    x = torch.randint(0, 64, (4, 16), device=device)
    y = torch.randint(0, 64, (4, 16), device=device)

    # Execute LPC forward-only step
    res = lpc_model.forward_lpc_step(x, y, optimizers)

    assert "loss" in res
    assert "layer_losses" in res
    assert len(res["layer_losses"]) == 2
    assert res["loss"] > 0.0

    # Generation check
    gen = lpc_model.generate(x[:1, :4], max_new_tokens=5)
    assert gen.shape == (1, 9)


def test_trainer_with_lpc():
    """Verify ASDAGTrainer with use_lpc=True."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = ASDAGLanguageModel(
        vocab_size=32,
        d_model=32,
        n_layers=2,
        n_heads=2,
        channel_mixer_type="ternary_swiglu",
        use_blt=False
    )

    data = np.random.randint(0, 32, size=1000, dtype=np.uint8)

    trainer = ASDAGTrainer(
        model=base_model,
        train_data=data[:800],
        val_data=data[800:],
        batch_size=4,
        seq_len=16,
        lr=1e-3,
        max_steps=10,
        eval_interval=5,
        use_lpc=True,
        device=device
    )

    initial_loss = trainer.train_step(0)
    assert initial_loss > 0.0

    stats = trainer.train()
    assert "best_val_loss" in stats
    assert stats["best_val_loss"] < 100.0


def test_lpc_with_muon_optimizers():
    """Verify LocalPredictiveLanguageModel with Muon optimizers."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = ASDAGLanguageModel(
        vocab_size=32,
        d_model=32,
        n_layers=2,
        n_heads=2,
        channel_mixer_type="dense_swiglu",
        use_blt=False
    ).to(device)

    lpc_model = LocalPredictiveLanguageModel(base_model).to(device)
    muon_optimizers = lpc_model.get_default_lpc_optimizers(use_muon=True, muon_lr=0.02, lr=3e-3)

    assert len(muon_optimizers) == 4

    x = torch.randint(0, 32, (4, 16), device=device)
    y = torch.randint(0, 32, (4, 16), device=device)

    res = lpc_model.forward_lpc_step(x, y, muon_optimizers)
    assert "loss" in res
    assert res["loss"] > 0.0

