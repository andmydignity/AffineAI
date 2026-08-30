import pytest
import torch
from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig


def test_hybrid_forward_and_joint_loss():
    config = TorosHybridConfig(
        dim=64, d_byte=32, n_encoder_layers=2, n_predictor_layers=1, n_heads=2, target_patch_size=8
    )
    model = TorosHybridLanguageModel(config)
    
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    
    logits, loss, metrics = model(x, targets=y)
    
    assert logits.shape == (2, 32, 256)
    assert loss is not None
    assert loss.item() > 0
    assert "loss_gen" in metrics
    assert "loss_jepa" in metrics
    assert metrics["loss_jepa"] == 0.0  # JEPA off by default: no jepa sub-metrics


def test_hybrid_gradients_and_ema():
    config = TorosHybridConfig(
        dim=64, d_byte=32, n_encoder_layers=2, n_predictor_layers=1, n_heads=2, target_patch_size=8
    )
    model = TorosHybridLanguageModel(config)
    
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    
    _, loss, _ = model(x, targets=y)
    loss.backward()
    
    # Gen loss reaches patcher and decoder; predictor is JEPA-scaffolding only
    assert model.context_encoder.patcher.patch_proj.weight.grad is not None
    assert model.byte_decoder.lm_head.weight.grad is not None


def test_hybrid_stop_grad_target_and_sigreg():
    config = TorosHybridConfig(
        dim=64, d_byte=32, n_encoder_layers=2, n_predictor_layers=1, n_heads=2, target_patch_size=8,
        jepa_loss_weight=0.5
    )
    model = TorosHybridLanguageModel(config)

    # No EMA target encoder should exist
    assert not hasattr(model, "target_encoder")
    assert not hasattr(model.config, "ema_momentum")

    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    _, loss, m = model(x, targets=y)
    assert m["loss_jepa"] > 0.0
    assert "loss_sigreg" in m

    # Collapsed embeddings: constant predictor output sorts to a point mass,
    # far from the N(0,1) quantiles SIGReg matches against -> SIGReg must grow.
    healthy_sigreg = m["loss_sigreg"]
    with torch.no_grad():
        for p in model.predictor.parameters():
            p.zero_()
    _, _, m2 = model(x, targets=y)
    collapsed_sigreg = m2["loss_sigreg"]
    assert collapsed_sigreg > healthy_sigreg


def test_hybrid_generation():
    config = TorosHybridConfig(
        dim=64, d_byte=32, n_encoder_layers=2, n_predictor_layers=1, n_heads=2, target_patch_size=8
    )
    model = TorosHybridLanguageModel(config)
    
    prompt = torch.tensor([[ord('O'), ord('n'), ord('c'), ord('e')]], dtype=torch.long)
    out = model.generate_with_latent_planning(prompt, max_new_bytes=10)
    
    assert out.shape[1] == 14


def test_hybrid_native_lpc():
    config = TorosHybridConfig(
        dim=64, d_byte=32, n_encoder_layers=2, n_predictor_layers=1, n_heads=2, target_patch_size=8
    )
    model = TorosHybridLanguageModel(config)
    optimizers = model.get_default_optimizers(lr=1e-3)
    
    assert len(optimizers) == 3 # 2 encoder layers + 1 tail layer
    
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    
    res = model.forward_lpc_step(x, y, optimizers)
    
    assert "loss" in res
    assert "loss_gen" in res
    assert "loss_jepa" in res
    assert "layer_losses" in res
    assert len(res["layer_losses"]) == 2
    assert res["loss"] > 0.0


def test_hybrid_inference_export(tmp_path):
    config = TorosHybridConfig(
        dim=64, d_byte=32, n_encoder_layers=2, n_predictor_layers=1, n_heads=2, target_patch_size=8
    )
    model = TorosHybridLanguageModel(config)
    
    clean_state = model.export_inference_state_dict()
    for k in clean_state.keys():
        assert not k.startswith("target_encoder")
        assert not k.startswith("local_heads")
        
    save_file = str(tmp_path / "model_clean.pt")
    model.save_inference_checkpoint(save_file)
    
    loaded = torch.load(save_file, weights_only=False)
    assert loaded["scaffolding_stripped"] is True
    assert "model_state_dict" in loaded
