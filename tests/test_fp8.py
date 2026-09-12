import math
import pytest
import torch
import torch.nn.functional as F

from affine_ai.kernels.triton_quant_swa import (
    pack_fp8_kv,
    unpack_fp8_kv,
    fp8_sliding_window_attn,
    _eager_fp8_swa,
    _is_hopper_or_higher,
)
from affine_ai.core.csa import CompressedSparseAttentionMixer
from affine_ai.core.bitlinear import TernaryBitLinearSwiGLU
from affine_ai.models.hybrid import TorosHybridConfig, TorosHybridLanguageModel


def test_fp8_kv_pack_unpack_fidelity():
    """Verify FP8 KV pack/unpack roundtrip preserves high cosine similarity (>0.99)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, T, D = 2, 4, 32, 64
    x = torch.randn(B, H, T, D, device=device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

    packed, scale = pack_fp8_kv(x)
    assert packed.shape == x.shape
    if hasattr(torch, "float8_e4m3fn"):
        assert packed.dtype == torch.float8_e4m3fn

    x_rec = unpack_fp8_kv(packed, scale)
    assert x_rec.shape == x.shape
    assert x_rec.dtype == x.dtype

    cos_sim = F.cosine_similarity(x.float(), x_rec.float(), dim=-1).mean().item()
    assert cos_sim > 0.99, f"Expected cosine similarity > 0.99, got {cos_sim:.4f}"


def test_fp8_sliding_window_attention_parity():
    """Verify FP8 SWA produces valid outputs matching eager reference."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    B, H, T, D = 2, 4, 64, 64

    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)

    k_pack, k_scale = pack_fp8_kv(k)
    v_pack, v_scale = pack_fp8_kv(v)

    out_eager = _eager_fp8_swa(q, k_pack, k_scale, v_pack, v_scale, window=32, sink=True)
    out_dispatched = fp8_sliding_window_attn(q, k_pack, k_scale, v_pack, v_scale, window=32, sink=True)

    assert out_eager.shape == (B, H, T, D)
    assert out_dispatched.shape == (B, H, T, D)
    assert not torch.isnan(out_dispatched).any()
    assert torch.allclose(out_eager, out_dispatched, atol=1e-3, rtol=1e-3)


def test_csa_fp8_kv_caching():
    """Verify CSA2 works seamlessly with kv_quant='fp8' in full and reuse modes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    B, T, D = 2, 32, 64

    csa_full = CompressedSparseAttentionMixer(
        d_model=D, n_heads=4, mode="full", kv_quant="fp8", use_tree_sga=False, dtype=dtype
    ).to(device)
    csa_reuse = CompressedSparseAttentionMixer(
        d_model=D, n_heads=4, mode="reuse", kv_quant="fp8", shared_source=csa_full, use_tree_sga=False, dtype=dtype
    ).to(device)

    x = torch.randn(B, T, D, device=device, dtype=dtype)
    out_full, _ = csa_full(x)
    out_reuse, _ = csa_reuse(x)

    assert out_full.shape == (B, T, D)
    assert out_reuse.shape == (B, T, D)
    assert not torch.isnan(out_full).any()
    assert not torch.isnan(out_reuse).any()


def test_fp8_swiglu_forward_and_backward():
    """Verify TernaryBitLinearSwiGLU with use_fp8=True computes forward and backward gradients."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    B, T, D = 2, 16, 64

    swiglu = TernaryBitLinearSwiGLU(dim=D, expand=2, dtype=dtype, use_fp8=True).to(device)
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)

    out = swiglu(x)
    assert out.shape == (B, T, D)
    assert not torch.isnan(out).any()

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert swiglu.w_gate_val.weight.grad is not None


def test_toros_hybrid_with_fp8():
    """Verify end-to-end TorosHybridLanguageModel with use_fp8=True."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    B, T = 2, 64

    cfg = TorosHybridConfig(
        dim=64,
        d_byte=32,
        n_encoder_layers=4,
        n_heads=2,
        target_patch_size=8,
        lpc_chunk_size=2,
        decoder_channel_mixer="swiglu",
        use_csa=True,
        csa_every_n=2,
        use_fp8=True,
        dtype=dtype,
    )
    model = TorosHybridLanguageModel(cfg).to(device)
    assert model.config.csa_kv_quant == "fp8"

    tokens = torch.randint(0, 256, (B, T), dtype=torch.long, device=device)

    # 1. Forward Pass
    logits, loss, _ = model(tokens, tokens)
    assert logits.shape == (B, T, 256)
    assert loss is not None
    assert not torch.isnan(loss)

    # 2. LPC Training Step
    opts = model.get_default_lpc_optimizers(lr=1e-3, use_muon=False)
    res = model.forward_lpc_step(tokens, tokens, optimizers=opts, use_cuda_graph=False)
    assert "loss" in res
    assert not torch.isnan(res["loss"])

    # 3. Incremental Stepping
    h_small, _ = model.forward_incremental(tokens[:, :1])
    assert h_small.shape == (B, 1, 256)


def test_fp8_default_on_sm89(monkeypatch):
    """Verify FP8 is automatically enabled by default on SM89+ (Ada/Hopper/Blackwell) and disabled on SM < 89."""
    from affine_ai.kernels.triton_quant_swa import is_sm89_or_higher
    from affine_ai.models.blt import ByteLocalDecoder

    # 1. Simulate Ada Lovelace SM89 (RTX 4090 / 4080 / L4 / L40)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (8, 9))

    assert is_sm89_or_higher() is True

    cfg_ada = TorosHybridConfig()
    assert cfg_ada.use_fp8 is True
    assert cfg_ada.csa_kv_quant == "fp8"

    csa_ada = CompressedSparseAttentionMixer(d_model=64, n_heads=4)
    assert csa_ada.kv_quant == "fp8"

    decoder_ada = ByteLocalDecoder(d_model=64, d_byte=32)
    assert decoder_ada.use_fp8 is True

    swiglu_ada = TernaryBitLinearSwiGLU(dim=32)
    assert swiglu_ada.use_fp8 is True

    # Explicit override should be respected on Ada
    cfg_ada_off = TorosHybridConfig(use_fp8=False)
    assert cfg_ada_off.use_fp8 is False

    cfg_ada_int4 = TorosHybridConfig(csa_kv_quant="int4")
    assert cfg_ada_int4.csa_kv_quant == "int4"

    # 2. Simulate Hopper SM90
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (9, 0))
    assert is_sm89_or_higher() is True

    cfg_hopper = TorosHybridConfig()
    assert cfg_hopper.use_fp8 is True
    assert cfg_hopper.csa_kv_quant == "fp8"

    # 3. Simulate Blackwell SM100
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))
    assert is_sm89_or_higher() is True

    cfg_blackwell = TorosHybridConfig()
    assert cfg_blackwell.use_fp8 is True
    assert cfg_blackwell.csa_kv_quant == "fp8"

    # 4. Simulate Ampere SM86 (e.g. RTX 3050/3080/3090, A100 is 80)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (8, 6))
    assert is_sm89_or_higher() is False

    cfg_ampere = TorosHybridConfig()
    assert cfg_ampere.use_fp8 is False
    assert cfg_ampere.csa_kv_quant == "int4"

    csa_ampere = CompressedSparseAttentionMixer(d_model=64, n_heads=4)
    assert csa_ampere.kv_quant == "int4"

    decoder_ampere = ByteLocalDecoder(d_model=64, d_byte=32)
    assert decoder_ampere.use_fp8 is False

    # Explicit activation should work on Ampere
    cfg_ampere_on = TorosHybridConfig(use_fp8=True)
    assert cfg_ampere_on.use_fp8 is True
    assert cfg_ampere_on.csa_kv_quant == "fp8"
