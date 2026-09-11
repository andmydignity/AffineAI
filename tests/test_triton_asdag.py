"""
Tests for Fused Triton ASDAG Kernel
"""

import pytest
import torch
import torch.nn.functional as F

from affine_ai.kernels import TRITON_AVAILABLE
try:
    if TRITON_AVAILABLE:
        from affine_ai.kernels.triton_asdag import fused_asdag_forward_triton
        HAS_ASDAG_TRITON = True
    else:
        HAS_ASDAG_TRITON = False
except Exception:
    HAS_ASDAG_TRITON = False


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_ASDAG_TRITON, reason="CUDA and ASDAG Triton kernel required")
def test_triton_fused_asdag_parity():
    torch.manual_seed(42)
    device = "cuda"
    B, D = 32, 64
    K = 8
    M_max = 2

    x = torch.randn(B, D, device=device)
    w_stack = torch.randint(-1, 2, (K, D, D), device=device, dtype=torch.float32)
    bias_stack = torch.randn(K, D, device=device)
    routing_logits = torch.randn(B, K, device=device)
    routing_probs = F.softmax(routing_logits, dim=-1)

    context_gates = torch.randn(K, M_max, D, device=device)
    peer_outputs = torch.randn(K, M_max, B, D, device=device)
    norm_factors = torch.ones(K, device=device) * (1.0 / (1.0 + M_max) ** 0.5)

    # 1. PyTorch reference forward
    ref_outs = []
    for k in range(K):
        y_prim = F.linear(x, w_stack[k]) + bias_stack[k]
        h_ctx = torch.zeros_like(y_prim)
        for s in range(M_max):
            h_ctx += context_gates[k, s] * peer_outputs[k, s]
        y_v = (y_prim + h_ctx) * norm_factors[k]
        y_act = F.relu6(y_v)
        ref_outs.append(y_act)
    stacked_ref = torch.stack(ref_outs, dim=1)
    ref_composite = torch.einsum('bk, bkd -> bd', routing_probs, stacked_ref)

    # 2. Triton fused forward
    triton_out = fused_asdag_forward_triton(
        x=x,
        w_stack=w_stack,
        bias_stack=bias_stack,
        routing_probs=routing_probs,
        context_gates=context_gates,
        peer_outputs=peer_outputs,
        norm_factors=norm_factors,
        activation="relu6"
    )

    # 3. Assert mathematical parity
    assert torch.allclose(ref_composite, triton_out, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_ASDAG_TRITON, reason="CUDA and ASDAG Triton kernel required")
def test_triton_fused_asdag_d512():
    """Verify D >= 512 does not suffer from silent channel truncation (Issue 1)."""
    torch.manual_seed(42)
    device = "cuda"
    B, D = 4, 512
    K = 4
    x = torch.randn(B, D, device=device)
    w_stack = torch.randint(-1, 2, (K, D, D), device=device, dtype=torch.float32)
    bias_stack = torch.randn(K, D, device=device)
    routing_probs = F.softmax(torch.randn(B, K, device=device), dim=-1)
    context_gates = torch.zeros(K, 1, D, device=device)
    norm_factors = torch.ones(K, device=device)

    ref_outs = []
    for k in range(K):
        y_prim = F.linear(x, w_stack[k]) + bias_stack[k]
        ref_outs.append(F.relu6(y_prim))
    stacked_ref = torch.stack(ref_outs, dim=1)
    ref = torch.einsum('bk, bkd -> bd', routing_probs, stacked_ref)

    triton_out = fused_asdag_forward_triton(
        x=x, w_stack=w_stack, bias_stack=bias_stack, routing_probs=routing_probs,
        context_gates=context_gates, norm_factors=norm_factors, activation="relu6",
    )
    assert torch.allclose(ref, triton_out, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_ASDAG_TRITON, reason="CUDA and ASDAG Triton kernel required")
def test_triton_fused_asdag_peer_3d_and_zero_sum_gate():
    """Verify 3D peer_outputs stride (Issue 3) and zero-sum gate accumulation (Issue 2)."""
    torch.manual_seed(42)
    device = "cuda"
    B, D = 8, 64
    K = 4
    x = torch.randn(B, D, device=device)
    w_stack = torch.randn(K, D, D, device=device)
    bias_stack = torch.randn(K, D, device=device)
    routing_probs = F.softmax(torch.randn(B, K, device=device), dim=-1)
    norm_factors = torch.ones(K, device=device)

    # Gate summing to 0 (e.g. [+1, -1] or sum == 0)
    context_gates = torch.zeros(K, 1, D, device=device)
    context_gates[:, 0, :D//2] = 1.0
    context_gates[:, 0, D//2:] = -1.0
    assert context_gates.sum(dim=-1).abs().max().item() == 0.0  # sums to 0!

    # 3D peer_outputs of shape (K, B, D)
    peer_outputs_3d = torch.randn(K, B, D, device=device)

    triton_out = fused_asdag_forward_triton(
        x=x, w_stack=w_stack, bias_stack=bias_stack, routing_probs=routing_probs,
        context_gates=context_gates, peer_outputs=peer_outputs_3d, norm_factors=norm_factors, activation="relu6",
    )
    assert triton_out.shape == (B, D)
    assert not torch.isnan(triton_out).any()


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_ASDAG_TRITON, reason="CUDA and ASDAG Triton kernel required")
def test_triton_fused_asdag_fp16():
    """Verify FP16 precision execution (Issue 4)."""
    torch.manual_seed(42)
    device = "cuda"
    B, D = 8, 64
    K = 4
    x = torch.randn(B, D, device=device, dtype=torch.float16)
    w_stack = torch.randn(K, D, D, device=device, dtype=torch.float16)
    bias_stack = torch.randn(K, D, device=device, dtype=torch.float16)
    routing_probs = F.softmax(torch.randn(B, K, device=device, dtype=torch.float16), dim=-1)
    context_gates = torch.randn(K, 1, D, device=device, dtype=torch.float16)
    norm_factors = torch.ones(K, device=device, dtype=torch.float16)

    triton_out = fused_asdag_forward_triton(
        x=x, w_stack=w_stack, bias_stack=bias_stack, routing_probs=routing_probs,
        context_gates=context_gates, norm_factors=norm_factors, activation="relu6",
    )
    assert triton_out.dtype == torch.float16
    assert not torch.isnan(triton_out).any()
