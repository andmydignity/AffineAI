import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.models.qwen35_asdag import Qwen35ASDAGConfig, Qwen35Block
from affine_ai.models.attention_bridge import AttentionBridge, CrossArchitectureAttentionBridge


def test_attention_bridge_forward():
    dim = 64
    bridge = AttentionBridge(dim=dim, hidden_dim=96, dtype=torch.float32)
    x = torch.randn(2, 8, dim)
    out = bridge(x)
    
    assert out.shape == x.shape
    # Since down projection is initialized near zero, out should be close to x
    assert torch.allclose(out, x, atol=1e-1)


def test_cross_architecture_attention_bridge_distill():
    config = Qwen35ASDAGConfig(
        dim=64,
        intermediate_dim=128,
        num_layers=4,
        full_attn_interval=2,
        ssm_v_heads=4,
        ssm_qk_heads=2,
        ssm_head_dim=16,
        attn_q_heads=4,
        attn_kv_heads=2,
        attn_head_dim=16,
        rope_dim=16,
        dtype=torch.float32
    )
    cab = CrossArchitectureAttentionBridge(config, include_teacher=True)
    x = torch.randn(2, 8, config.dim)

    # 1. Test standard recurrent forward
    out, next_state = cab(x)
    assert out.shape == x.shape
    assert next_state is not None
    assert len(next_state) == 2  # (conv_state, ssm_state)

    # 2. Test distillation mode
    bridged_out, next_state, loss_cab = cab.forward_distill(x)
    assert bridged_out.shape == x.shape
    assert loss_cab.item() > 0.0

    # Test backward through bridge
    loss_cab.backward()
    assert cab.bridge.gate_up.weight.grad is not None
    assert cab.bridge.down.weight.grad is not None


def test_qwen35_block_with_attention_bridge():
    config = Qwen35ASDAGConfig(
        dim=64,
        intermediate_dim=128,
        num_layers=4,
        full_attn_interval=2,
        ssm_v_heads=4,
        ssm_qk_heads=2,
        ssm_head_dim=16,
        attn_q_heads=4,
        attn_kv_heads=2,
        attn_head_dim=16,
        use_attention_bridge=True,
        dtype=torch.float32
    )
    # Layer 1 has is_full_attention = True ((1+1)%2 == 0)
    block = Qwen35Block(config, layer_idx=1)
    assert isinstance(block.time_mixer, CrossArchitectureAttentionBridge)

    x = torch.randn(2, 8, config.dim)
    out, state = block(x)
    assert out.shape == x.shape
    assert state is not None  # State is recurrent conv/ssm state, NOT quadratic kv cache!
