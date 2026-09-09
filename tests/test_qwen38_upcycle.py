import pytest
import torch
import torch.nn.functional as F

from affine_ai.models.qwen38_asdag import (
    Qwen38ASDAGConfig,
    Qwen38ASDAGLeaf,
    Qwen38ASDAGFFN,
    Qwen38GatedDeltaNet,
    Qwen38GatedAttention,
    Qwen38Block,
    apply_nm_sparsity
)


def test_qwen38_config_invariants():
    """Verify Qwen3.8-27B architectural specifications and native ASDAG invariants."""
    config = Qwen38ASDAGConfig()

    assert config.dim == 5120
    assert config.vocab_size == 248320
    assert config.num_layers == 64
    assert config.intermediate_dim == 17408
    assert config.num_leaves == 8
    assert config.leaf_dim == 2176  # 17408 // 8
    assert config.top_k == 2
    assert config.nm_n == 1
    assert config.nm_m == 16
    assert config.use_nm_sparsity is True
    assert config.ternary_leaves is True
    assert config.ternary_embedding is False  # Sacred layer unquantized in BF16
    assert config.ssm_v_heads == 48
    assert config.ssm_qk_heads == 16
    assert config.attn_q_heads == 24
    assert config.attn_kv_heads == 4


def test_qwen38_1_to_16_structured_sparsity():
    """Verify that 1:16 structured sparsity zeros exactly 93.75% of weights."""
    w = torch.randn(64, 5120)
    w_sparse = apply_nm_sparsity(w, n=1, m=16)

    # Check zero fraction
    zero_frac = (w_sparse == 0).float().mean().item()
    assert abs(zero_frac - 0.9375) < 1e-5

    # Check that in every contiguous chunk of 16, exactly 1 weight is non-zero
    w_chunks = w_sparse.reshape(-1, 16)
    non_zeros_per_chunk = (w_chunks != 0).sum(dim=-1)
    assert (non_zeros_per_chunk == 1).all()


def test_qwen38_leaf_forward_and_backward():
    """Verify that ASDAG leaf executes with ternary weights and 1:16 sparsity, flowing STE gradients."""
    config = Qwen38ASDAGConfig()
    leaf = Qwen38ASDAGLeaf(config.dim, config.leaf_dim, config.dim, dtype=torch.float32)
    leaf.train()

    x = torch.randn(2, 4, config.dim, requires_grad=True)
    out = leaf(x)
    assert out.shape == (2, 4, config.dim)

    loss = out.sum()
    loss.backward()

    assert leaf.gate_proj.weight.grad is not None
    assert leaf.up_proj.weight.grad is not None
    assert leaf.down_proj.weight.grad is not None
    assert x.grad is not None


def test_qwen38_gated_deltanet_forward():
    """Verify 48 V-head / 16 QK-head Gated DeltaNet forward pass and recurrence stepping."""
    config = Qwen38ASDAGConfig()
    deltanet = Qwen38GatedDeltaNet(config).float()
    deltanet.eval()

    B, T = 2, 4
    x = torch.randn(B, T, config.dim)
    out, (next_conv, next_ssm) = deltanet(x)

    assert out.shape == (B, T, config.dim)
    assert next_conv.shape == (B, deltanet.qkv_dim, config.ssm_conv_kernel - 1)
    assert next_ssm.shape == (B, config.ssm_v_heads, config.ssm_head_dim, config.ssm_head_dim)


def test_qwen38_gated_attention_forward():
    """Verify 24 Q-head / 4 KV-head Gated Attention forward pass and KV-caching."""
    config = Qwen38ASDAGConfig()
    attn = Qwen38GatedAttention(config).float()
    attn.eval()

    B, T = 2, 4
    x = torch.randn(B, T, config.dim)
    out, (k_cache, v_cache) = attn(x)

    assert out.shape == (B, T, config.dim)
    assert k_cache.shape == (B, T, config.attn_kv_heads, config.attn_head_dim)
    assert v_cache.shape == (B, T, config.attn_kv_heads, config.attn_head_dim)


def test_qwen38_block_execution():
    """Verify full Qwen3.8 ASDAG block forward on both DeltaNet and Attention layers."""
    config = Qwen38ASDAGConfig()
    x = torch.randn(2, 4, config.dim, dtype=config.dtype)

    b0 = Qwen38Block(config, layer_idx=0)
    out0, st0 = b0(x)
    assert out0.shape == (2, 4, config.dim)

    b3 = Qwen38Block(config, layer_idx=3)
    out3, st3 = b3(x)
    assert out3.shape == (2, 4, config.dim)
