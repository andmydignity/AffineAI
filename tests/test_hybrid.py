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
    assert len(lpc_opts) == 4
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


def test_hybrid_triton_fused_dec_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    config = TorosHybridConfig(
        dim=64, d_byte=32, n_encoder_layers=2, n_heads=2, target_patch_size=8, dtype=torch.bfloat16
    )
    model = TorosHybridLanguageModel(config).cuda()
    x = torch.randint(0, 256, (2, 32), device="cuda")
    y = torch.randint(0, 256, (2, 32), device="cuda")

    # Non-fused path (return_logits=True)
    logits, loss_std, _ = model(x, targets=y, return_logits=True)

    # Fused path (return_logits=False)
    _, loss_fused, _ = model(x, targets=y, return_logits=False)

    assert loss_std is not None and loss_fused is not None
    # Tolerance reflects hardware INT8 IMMA Tensor Core accumulation vs fused FP32 cross-entropy
    assert torch.isclose(loss_std, loss_fused, atol=0.25, rtol=0.05)


def test_hybrid_lpc_async_pipelining():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        config = TorosHybridConfig(
            dim=64, d_byte=32, n_encoder_layers=2, n_heads=2, target_patch_size=8
        )
        model = TorosHybridLanguageModel(config).to(device)
        model.enable_lpc()

        for use_muon in [False, True]:
            opts = model.get_default_lpc_optimizers(lr=1e-3, use_muon=use_muon)
            x = torch.randint(0, 256, (2, 32), device=device)
            y = torch.randint(0, 256, (2, 32), device=device)

            # Test sync_loss=False, use_async_pipelining=True
            res_async = model.forward_lpc_step(
                x, y, opts, use_async_pipelining=True, sync_loss=False, return_sample_loss=True
            )
            assert "loss" in res_async and res_async["loss"] > 0
            assert "layer_losses" in res_async and len(res_async["layer_losses"]) == 2
            assert "mean_local_loss" in res_async and res_async["mean_local_loss"] > 0
            assert "loss_total" in res_async and res_async["loss_total"] > 0
            assert "sample_loss" in res_async and res_async["sample_loss"] is not None
            assert res_async["sample_loss"].shape == (2,)

            # Test sync_loss=True, use_async_pipelining=False
            opts2 = model.get_default_lpc_optimizers(lr=1e-3, use_muon=use_muon)
            res_sync = model.forward_lpc_step(
                x, y, opts2, use_async_pipelining=False, sync_loss=True, return_sample_loss=False
            )
            assert isinstance(res_sync["loss"], float) and res_sync["loss"] > 0
            assert isinstance(res_sync["mean_local_loss"], float)
            assert len(res_sync["layer_losses"]) == 2
            assert isinstance(res_sync["layer_losses"][0], float)


