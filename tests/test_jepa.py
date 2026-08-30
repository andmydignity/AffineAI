import pytest
import torch
from affine_ai.models.jepa import TorosJEPA, TorosJEPAConfig


def test_jepa_forward_and_loss():
    config = TorosJEPAConfig(dim=64, d_byte=32, n_encoder_layers=2, n_predictor_layers=1, n_heads=2, target_patch_size=16)
    jepa = TorosJEPA(config)
    
    x_ctx = torch.randint(0, 256, (2, 32)) # 32 bytes -> 2 patches
    x_tgt = torch.randint(0, 256, (2, 32))
    
    s_pred, loss, metrics = jepa(x_ctx, x_tgt)
    
    assert s_pred.shape == (2, 2, 64)
    assert loss is not None
    assert loss.item() > 0
    assert "loss_invariance" in metrics
    assert "loss_variance" in metrics
    assert "loss_covariance" in metrics
    assert "latent_std" in metrics


def test_jepa_gradients_and_ema_update():
    config = TorosJEPAConfig(dim=64, d_byte=32, n_encoder_layers=2, n_predictor_layers=1, n_heads=2, target_patch_size=16)
    jepa = TorosJEPA(config)
    
    x_ctx = torch.randint(0, 256, (2, 32))
    x_tgt = torch.randint(0, 256, (2, 32))
    
    _, loss, _ = jepa(x_ctx, x_tgt)
    loss.backward()
    
    # Check context encoder has gradients
    assert jepa.context_encoder.patcher.patch_proj.weight.grad is not None
    assert jepa.predictor.pred_proj.weight.grad is not None
    
    # Check target encoder has NO gradients
    for p in jepa.target_encoder.parameters():
        assert p.grad is None
        
    # Check EMA update modifies target encoder weights
    orig_w = jepa.target_encoder.patcher.patch_proj.weight.data.clone()
    jepa.context_encoder.patcher.patch_proj.weight.data.add_(1.0)
    jepa.update_target_encoder(momentum=0.9)
    new_w = jepa.target_encoder.patcher.patch_proj.weight.data
    
    assert not torch.allclose(orig_w, new_w)


def test_jepa_latent_rollout():
    config = TorosJEPAConfig(dim=64, d_byte=32, n_encoder_layers=2, n_predictor_layers=1, n_heads=2, target_patch_size=16)
    jepa = TorosJEPA(config)
    
    ctx = torch.randint(0, 256, (1, 32)) # 32 bytes -> 2 patches
    rollout = jepa.latent_rollout(ctx, num_steps=5)
    
    assert len(rollout) == 5
    for state in rollout:
        assert state.shape == (1, 2, 64)
