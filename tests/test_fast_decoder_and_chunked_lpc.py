import pytest
import torch
from affine_ai.models.blt import ByteLocalDecoder
from affine_ai.models.hybrid import TorosHybridConfig, TorosHybridLanguageModel
from affine_ai.training.trainer import ASDAGTrainer

def test_byte_decoder_swiglu_and_asdag():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, M = 2, 64, 8
    d_model = 64
    d_byte = 32

    # Test SwiGLU decoder
    dec_swiglu = ByteLocalDecoder(
        vocab_size=256,
        d_byte=d_byte,
        d_model=d_model,
        channel_mixer_type="swiglu",
        dtype=torch.float32,
    ).to(device)

    h_byte = torch.randn(B, T, d_byte, device=device)
    latent_patches = torch.randn(B, M, d_model, device=device)
    patch_assignments = torch.zeros(B, T, dtype=torch.long, device=device)

    logits_swiglu = dec_swiglu(h_byte, latent_patches, patch_assignments)
    assert logits_swiglu.shape == (B, T, 256)

    # Test ASDAG Tree decoder
    dec_asdag = ByteLocalDecoder(
        vocab_size=256,
        d_byte=d_byte,
        d_model=d_model,
        channel_mixer_type="asdag_tree",
        dtype=torch.float32,
    ).to(device)
    logits_asdag = dec_asdag(h_byte, latent_patches, patch_assignments)
    assert logits_asdag.shape == (B, T, 256)

def test_chunked_lpc_hybrid_forward_and_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T = 2, 64
    cfg = TorosHybridConfig(
        dim=64,
        n_encoder_layers=4,
        d_byte=32,
        target_patch_size=8,
        lpc_chunk_size=2,
        decoder_channel_mixer="swiglu",
        dtype=torch.float32,
        use_csa=False,
    )
    model = TorosHybridLanguageModel(cfg).to(device)
    opts = model.get_default_lpc_optimizers(lr=1e-3, use_muon=False)

    # With 4 layers and chunk_size=2, there should be:
    # 2 chunk optimizers + 1 encoder tail + 1 decoder = 4 optimizers
    assert len(opts) == 4
    assert len(model.local_heads) == 2

    x = torch.randint(0, 256, (B, T), device=device)
    y = torch.randint(0, 256, (B, T), device=device)

    res = model.forward_lpc_step(x, y, opts, use_cuda_graph=False)
    assert "loss" in res
    assert not torch.isnan(res["loss"])
    assert len(res["layer_losses"]) == 2

def test_trainer_integration_chunked_lpc():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T = 2, 64
    cfg = TorosHybridConfig(
        dim=64,
        n_encoder_layers=4,
        d_byte=32,
        target_patch_size=8,
        lpc_chunk_size=2,
        decoder_channel_mixer="swiglu",
        dtype=torch.float32,
        use_csa=False,
    )
    model = TorosHybridLanguageModel(cfg).to(device)
    train_data = torch.randint(0, 256, (1000,), dtype=torch.uint8)

    trainer = ASDAGTrainer(
        model=model,
        train_data=train_data,
        batch_size=B,
        seq_len=T,
        max_steps=5,
        device=device,
        use_cuda_graph=False,
        use_muon=False,
    )
    assert trainer.use_priority_replay is False

    loss1 = trainer.train_step(1)
    assert loss1 is not None
    loss2 = trainer.train_step(2)
    assert loss2 is not None
