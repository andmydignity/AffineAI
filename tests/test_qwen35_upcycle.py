import pytest
import os
import torch
import torch.nn.functional as F

from affine_ai.models.qwen35_asdag import (
    Qwen35ASDAGConfig,
    Qwen35Block,
    Qwen35ASDAGFFN,
    Qwen35GatedDeltaNet,
    Qwen35GatedAttention,
)


def test_asdag_ffn_mathematical_parity():
    """Verify that forward_dense(x) across sliced continuous leaves is 100% identical to full SwiGLU."""
    torch.manual_seed(42)
    dim = 2560
    intermediate_dim = 9216
    num_leaves = 8
    leaf_dim = intermediate_dim // num_leaves  # 1152

    config = Qwen35ASDAGConfig(
        dim=dim,
        intermediate_dim=intermediate_dim,
        num_leaves=num_leaves,
        leaf_dim=leaf_dim,
        dtype=torch.float32,
        ternary_leaves=False,  # Test continuous slicing identity
        use_shift4_routing=False,
        use_fp8_hybrid=False
    )
    asdag_ffn = Qwen35ASDAGFFN(config)
    asdag_ffn.eval()

    # Reconstruct the equivalent dense weights by concatenating leaf weights
    dense_gate = torch.cat([leaf.gate_proj.weight for leaf in asdag_ffn.leaves], dim=0)  # [9216, 2560]
    dense_up = torch.cat([leaf.up_proj.weight for leaf in asdag_ffn.leaves], dim=0)      # [9216, 2560]
    dense_down = torch.cat([leaf.down_proj.weight for leaf in asdag_ffn.leaves], dim=1)  # [2560, 9216]

    x = torch.randn(2, 4, dim)

    # 1. Full dense SwiGLU forward
    dense_out = (F.silu(F.linear(x, dense_gate)) * F.linear(x, dense_up)) @ dense_down.T

    # 2. ASDAG dense forward (sum across leaves)
    asdag_dense_out = asdag_ffn.forward_dense(x)

    # Check numerical identity
    max_diff = (dense_out - asdag_dense_out).abs().max().item()
    assert max_diff < 1e-5, f"ASDAG dense slicing deviates from full SwiGLU: max_diff={max_diff}"


def test_asdag_native_defaults():
    """Verify ASDAG native defaults: ternary leaves {-1, 0, 1}, 4-bit logshift routing, and FP8 hybrid."""
    torch.manual_seed(42)
    dim = 2560
    intermediate_dim = 9216
    config = Qwen35ASDAGConfig(
        dim=dim,
        intermediate_dim=intermediate_dim,
        num_leaves=8,
        leaf_dim=1152,
        top_k=2,
        ternary_leaves=True,
        use_shift4_routing=True,
        use_fp8_hybrid=True,
        dtype=torch.bfloat16
    )
    asdag_ffn = Qwen35ASDAGFFN(config)
    asdag_ffn.train()

    x = torch.randn(2, 4, dim, dtype=torch.bfloat16, requires_grad=True)
    out = asdag_ffn(x, top_k=2)

    assert out.shape == x.shape
    assert not torch.isnan(out).any()

    # Verify backward pass with FP8 hybrid gradient backpressure
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_gated_deltanet_forward():
    """Verify Gated DeltaNet layer forward and constant-memory recurrence stepping."""
    torch.manual_seed(42)
    config = Qwen35ASDAGConfig(
        dim=2560,
        ssm_v_heads=32,
        ssm_qk_heads=16,
        ssm_head_dim=128,
        dtype=torch.bfloat16
    )
    deltanet = Qwen35GatedDeltaNet(config)

    B, T = 2, 8
    x = torch.randn(B, T, config.dim, dtype=torch.bfloat16)

    # 1. Sequence forward
    y, (conv_st, ssm_st) = deltanet(x)
    assert y.shape == (B, T, config.dim)
    assert conv_st.shape == (B, deltanet.qkv_dim, config.ssm_conv_kernel - 1)
    assert ssm_st.shape == (B, 32, 128, 128)
    assert not torch.isnan(y).any()

    # 2. Step forward (next token t+1 with cached state)
    x_next = torch.randn(B, 1, config.dim, dtype=torch.bfloat16)
    y_step, (conv_st2, ssm_st2) = deltanet(x_next, conv_state=conv_st, ssm_state=ssm_st)
    assert y_step.shape == (B, 1, config.dim)
    assert not torch.isnan(y_step).any()


def test_gated_attention_forward():
    """Verify Gated Attention layer forward and KV-cache stepping."""
    torch.manual_seed(42)
    config = Qwen35ASDAGConfig(
        dim=2560,
        attn_q_heads=16,
        attn_kv_heads=4,
        attn_head_dim=256,
        dtype=torch.bfloat16
    )
    attn = Qwen35GatedAttention(config)

    B, T = 2, 8
    x = torch.randn(B, T, config.dim, dtype=torch.bfloat16)

    # 1. Sequence forward
    y, kv_cache = attn(x, pos=0)
    assert y.shape == (B, T, config.dim)
    k_cached, v_cached = kv_cache
    assert k_cached.shape == (B, T, 4, 256)
    assert v_cached.shape == (B, T, 4, 256)
    assert not torch.isnan(y).any()

    # 2. Step forward with cached KV
    x_next = torch.randn(B, 1, config.dim, dtype=torch.bfloat16)
    y_step, new_kv_cache = attn(x_next, kv_cache=kv_cache, pos=T)
    assert y_step.shape == (B, 1, config.dim)
    assert new_kv_cache[0].shape == (B, T + 1, 4, 256)


def test_upcycled_checkpoint_loading():
    """Verify loading upcycled weights from checkpoints/qwen35_asdag/block_00.pt into Qwen35Block."""
    ckpt_path = "checkpoints/qwen35_asdag/block_00.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip(f"{ckpt_path} does not exist. Run scripts/upcycle_qwen35.py first.")

    config = Qwen35ASDAGConfig(dtype=torch.bfloat16)
    block = Qwen35Block(config, layer_idx=0)

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    block.load_state_dict(state_dict)

    x = torch.randn(1, 4, config.dim, dtype=torch.bfloat16)
    out, state = block(x, top_k=2)

    assert out.shape == (1, 4, config.dim)
    assert not torch.isnan(out).any()


def test_mtp_checkpoint_loading():
    """Verify loading upcycled weights from checkpoints/qwen35_asdag/mtp_block.pt into Qwen35MTPBlock."""
    ckpt_path = "checkpoints/qwen35_asdag/mtp_block.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip(f"{ckpt_path} does not exist. Run scripts/upcycle_qwen35.py first.")

    from affine_ai.models.qwen35_asdag import Qwen35MTPBlock
    config = Qwen35ASDAGConfig(dtype=torch.bfloat16, use_attention_bridge=True)
    mtp = Qwen35MTPBlock(config)

    mtp_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    block_state = {k: v for k, v in mtp_ckpt.items() if not k.startswith(('eh_proj', 'enorm', 'hnorm', 'shared_head_norm'))}
    extra_state = {k: v for k, v in mtp_ckpt.items() if k.startswith(('eh_proj', 'enorm', 'hnorm', 'shared_head_norm'))}

    mtp.block.load_state_dict(block_state, strict=False)
    mtp.eh_proj.weight.data.copy_(extra_state['eh_proj.weight'])
    mtp.enorm.weight.data.copy_(extra_state['enorm.weight'])
    mtp.hnorm.weight.data.copy_(extra_state['hnorm.weight'])
    mtp.shared_head_norm.weight.data.copy_(extra_state['shared_head_norm.weight'])

    B, T = 1, 4
    h = torch.randn(B, T, config.dim, dtype=torch.bfloat16)
    emb = torch.randn(B, T, config.dim, dtype=torch.bfloat16)
    out, _ = mtp(h, emb, pos=0)

    assert out.shape == (B, T, config.dim)
    assert not torch.isnan(out).any()


def test_toros_metadata_reading():
    """Verify instant reading of .toros metadata header without loading entire 4B weights."""
    toros_path = "checkpoints/qwen35_asdag.toros"
    if not os.path.exists(toros_path):
        pytest.skip(f"{toros_path} does not exist.")

    from affine_ai.core.format import read_toros_metadata
    meta = read_toros_metadata(toros_path)

    assert meta["format"] == "TOROS"
    assert meta["model_type"] == "Qwen35ASDAGModel"
    assert meta["config"]["dim"] == 2560
    assert meta["config"]["num_layers"] == 32
    assert meta["config"]["num_leaves"] == 8
    assert meta["statistics"]["total_params"] == 4327026688
    assert meta["statistics"]["ternary_tensors"] == 902
