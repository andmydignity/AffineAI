import pytest
import torch
from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig


def test_hybrid_forward_and_joint_loss():
    config = TorosHybridConfig(
        dim=64, d_byte=32, n_encoder_layers=2, n_heads=2, target_patch_size=8
    )
    model = TorosHybridLanguageModel(config)
    
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    
    logits, loss, metrics = model(x, targets=y)
    
    assert logits.shape == (2, 32, 256)
    assert loss is not None
    assert loss.item() > 0
    assert "loss_gen" in metrics
    assert "loss_jepa" not in metrics


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
        dim=64, d_byte=32, n_encoder_layers=2, n_heads=2, target_patch_size=8,
    )
    model = TorosHybridLanguageModel(config)

    assert not hasattr(model, "predictor")
    assert not hasattr(model, "mask_token")
    assert not hasattr(model, "_sigreg")
    assert not hasattr(model, "local_heads")

    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    _, loss, m = model(x, targets=y)
    assert "loss_jepa" not in m
    assert "loss_sigreg" not in m


def test_hybrid_generation():
    config = TorosHybridConfig(
        dim=64, d_byte=32, n_encoder_layers=2, n_predictor_layers=1, n_heads=2, target_patch_size=8
    )
    model = TorosHybridLanguageModel(config)
    
    prompt = torch.tensor([[ord('O'), ord('n'), ord('c'), ord('e')]], dtype=torch.long)
    out = model.generate_with_latent_planning(prompt, max_new_bytes=10, temperature=0.0, eos_byte=None)
    
    assert out.shape[1] == 14


def test_hybrid_native_lpc():
    config = TorosHybridConfig(
        dim=64, d_byte=32, n_encoder_layers=2, n_heads=2, target_patch_size=8
    )
    model = TorosHybridLanguageModel(config)
    assert not hasattr(model, "local_heads") or getattr(model, "local_heads", None) is None
    assert hasattr(model, "forward_lpc_step")
    assert hasattr(model, "enable_lpc")
    assert hasattr(model, "get_default_lpc_optimizers")
    optimizers = model.get_default_optimizers(lr=1e-3)
    assert len(optimizers) == 3
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    _, loss, m = model(x, targets=y)
    loss.backward()
    assert m["loss_gen"] > 0
    model.enable_lpc()
    assert hasattr(model, "local_heads") and model.local_heads is not None
    lpc_opts = model.get_default_lpc_optimizers(lr=1e-3)
    assert len(lpc_opts) == 3
    res = model.forward_lpc_step(x, y, lpc_opts)
    assert "loss" in res and res["loss"] > 0


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
