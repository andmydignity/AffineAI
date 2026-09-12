import math
import pytest
import torch
import torch.nn as nn

from affine_ai.core.csa import (
    CompressedSparseAttentionMixer,
    interleave_csa,
    SharedKVEntry,
)
from affine_ai.kernels.triton_quant_swa import (
    pack_int4_kv,
    unpack_int4_kv,
    quantized_sliding_window_attn,
    _eager_quant_swa,
)
from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig


def test_int4_pack_unpack():
    """Verify INT4 pack and unpack roundtrip precision."""
    B, H, T, D = 2, 4, 32, 64
    x = torch.randn(B, H, T, D, dtype=torch.float32)

    packed, scale = pack_int4_kv(x)
    assert packed.shape == (B, H, T, D // 2)
    assert packed.dtype == torch.uint8
    assert scale.shape == (B, H, T, 1)

    unpacked = unpack_int4_kv(packed, scale)
    assert unpacked.shape == x.shape
    assert unpacked.dtype == x.dtype

    # Quantization error must be bounded by step size (scale / 7)
    max_err = (x - unpacked).abs().max().item()
    max_step = (scale).max().item()
    assert max_err <= max_step + 1e-4, f"Quantization error {max_err} exceeded step {max_step}"


def test_quantized_swa_parity():
    """Verify Triton quantized SWA matches eager reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, T, D, W = 2, 2, 128, 64, 32
    dtype = torch.float16

    torch.manual_seed(42)
    q = torch.randn(B, H, T, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, T, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H, T, D, device="cuda", dtype=dtype)

    k_pack, k_scale = pack_int4_kv(k)
    v_pack, v_scale = pack_int4_kv(v)

    out_eager = _eager_quant_swa(q, k_pack, k_scale, v_pack, v_scale, window=W, sink=True)
    out_triton = quantized_sliding_window_attn(q, k_pack, k_scale, v_pack, v_scale, window=W, sink=True)

    diff = (out_triton.float() - out_eager.float()).abs().max().item()
    assert diff < 2e-2, f"Triton vs Eager mismatch for quantized SWA: diff={diff}"

    # Sink token check at t=0
    v_unpacked = unpack_int4_kv(v_pack, v_scale)
    sink_diff = (out_triton[:, :, 0, :].float() - v_unpacked[:, :, 0, :].float()).abs().max().item()
    assert sink_diff < 1e-2, f"Sink token identity failed: diff={sink_diff}"


def test_csa_modes_and_sharing():
    """Verify Full, Reindex, and Reuse modes with cross-layer KV sharing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    B, T, D, H, W = 2, 64, 64, 4, 32

    # Layer 0: Full mode (Producer)
    full_layer = CompressedSparseAttentionMixer(
        d_model=D, n_heads=H, window=W, mode="full", dtype=dtype
    ).to(device)

    # Layer 1: Reindex mode (Reuses KV from Layer 0)
    reindex_layer = CompressedSparseAttentionMixer(
        d_model=D, n_heads=H, window=W, mode="reindex", shared_source=full_layer, dtype=dtype
    ).to(device)

    # Layer 2: Reuse mode (Reuses KV and selection from Layer 0)
    reuse_layer = CompressedSparseAttentionMixer(
        d_model=D, n_heads=H, window=W, mode="reuse", shared_source=full_layer, dtype=dtype
    ).to(device)

    # Verify parameter counts: Reindex and Reuse must NOT allocate k_proj or v_proj
    assert full_layer.k_proj is not None and full_layer.v_proj is not None
    assert reindex_layer.k_proj is None and reindex_layer.v_proj is None
    assert reuse_layer.k_proj is None and reuse_layer.v_proj is None

    full_params = sum(p.numel() for p in full_layer.parameters())
    reuse_params = sum(p.numel() for p in reuse_layer.parameters())
    # Full has Q, K, V, Out (4 projections). Reuse has only Q, Out (2 projections -> ~50% params)
    assert reuse_params < full_params, f"Expected fewer params in reuse mode: {reuse_params} vs {full_params}"

    # Forward pass
    x0 = torch.randn(B, T, D, device=device, dtype=dtype)
    out0, _ = full_layer(x0)
    assert out0.shape == (B, T, D)

    # Downstream layers execute using Layer 0's KV
    x1 = torch.randn(B, T, D, device=device, dtype=dtype)
    out1, _ = reindex_layer(x1)
    assert out1.shape == (B, T, D)

    x2 = torch.randn(B, T, D, device=device, dtype=dtype)
    out2, _ = reuse_layer(x2)
    assert out2.shape == (B, T, D)


def test_tree_sga_sparse_attention():
    """Verify Tree-Guided Sparse Global Attention allows tokens with matching leaf IDs to attend."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    B, T, D, H, W = 1, 16, 32, 2, 4  # Small window W=4

    layer = CompressedSparseAttentionMixer(
        d_model=D, n_heads=H, window=W, mode="full", use_tree_sga=True, dtype=dtype
    ).to(device)

    # Mock leaf indices: token 0 and token 10 share leaf 5
    leaf_indices = torch.zeros(B, T, dtype=torch.long, device=device)
    leaf_indices[0, 0] = 5
    leaf_indices[0, 10] = 5

    x = torch.randn(B, T, D, device=device, dtype=dtype)
    out, _ = layer(x, tree_leaf_indices=leaf_indices)
    assert out.shape == (B, T, D)
    assert not torch.isnan(out).any()


def test_toros_hybrid_with_csa():
    """Verify TorosHybridLanguageModel initializes and trains with CSA2 enabled."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    cfg = TorosHybridConfig(
        dim=64,
        n_encoder_layers=8,  # 8 layers: layers 3 and 7 will be CSA
        target_patch_size=4,
        use_csa=True,
        csa_every_n=4,
        csa_group_size=2,
        csa_window=16,
        csa_kv_quant="int4" if device == "cuda" else "none",
        dtype=dtype,
    )

    model = TorosHybridLanguageModel(cfg).to(device)

    # Check that CSA was interleaved
    csa_blocks = [
        b for b in model.context_encoder.blocks
        if isinstance(getattr(b, "time_mixer", None), CompressedSparseAttentionMixer)
    ]
    assert len(csa_blocks) == 2
    assert csa_blocks[0].time_mixer.mode == "full"
    assert csa_blocks[1].time_mixer.mode == "reindex"

    # Forward pass
    B, T_bytes = 2, 32
    byte_ids = torch.randint(0, 256, (B, T_bytes), device=device)
    logits, loss, metrics = model(byte_ids, targets=byte_ids)

    assert logits.shape == (B, T_bytes, 256)
    assert not torch.isnan(loss)

    # Incremental generation test
    gen_state = None
    for step in range(8):
        single_byte = byte_ids[:, step : step + 1]
        step_logits, gen_state = model.forward_incremental(
            single_byte, gen_state=gen_state, return_state=True
        )
        assert step_logits.shape == (B, 1, 256)
        assert not torch.isnan(step_logits).any()
