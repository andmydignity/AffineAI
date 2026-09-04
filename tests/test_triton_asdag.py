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
