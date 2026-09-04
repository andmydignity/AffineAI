"""
Custom Triton Kernel: 2D Grid-Tiled ASDAG Forward Dispatch
===========================================================
Parallelizes grid across BOTH (Batch Tiles, Leaves) so each GPU thread block
loads its leaf weight tile ONCE into SRAM and processes all tokens in parallel.
Zero loops inside thread blocks, zero redundant DRAM traffic.
"""

import math
from typing import Optional, Tuple
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_asdag_2d_grid_kernel(
    X_ptr,              # (B, D)
    W_stack_ptr,        # (K, D, D)
    Bias_stack_ptr,     # (K, D)
    Routing_ptr,        # (B, K)
    Context_ptr,        # (K, M_max, D)
    Peer_ptr,           # (K, M_max, B, D)
    Norm_Factors_ptr,   # (K,)
    Leaf_Outs_ptr,      # (B, K, D)
    stride_xb, stride_xd,
    stride_wk, stride_wd1, stride_wd2,
    stride_bk, stride_bd,
    stride_rb, stride_rk,
    stride_ck, stride_cm, stride_cd,
    stride_pok, stride_pos, stride_pob, stride_pod,
    stride_lob, stride_lok, stride_lod,
    B_SZ,
    DIM: tl.constexpr,
    MAX_SECONDARY: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_PEER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)  # Parallelized over leaves!

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < B_SZ
    mask_d = offs_d < DIM

    # 1. Load input batch tile: (BLOCK_M, BLOCK_D)
    x_ptrs = X_ptr + offs_m[:, None] * stride_xb + offs_d[None, :] * stride_xd
    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # 2. Load this leaf's weight tile ONCE: (BLOCK_D, BLOCK_D)
    w_ptrs = W_stack_ptr + pid_k * stride_wk + offs_d[:, None] * stride_wd2 + offs_d[None, :] * stride_wd1
    w_k = tl.load(w_ptrs, mask=mask_d[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # 3. Load this leaf's bias: (BLOCK_D,)
    b_ptrs = Bias_stack_ptr + pid_k * stride_bk + offs_d * stride_bd
    b_k = tl.load(b_ptrs, mask=mask_d, other=0.0).to(tl.float32)

    # 4. Compute primary transformation: (BLOCK_M, BLOCK_D)
    y_prim = tl.dot(x, w_k, input_precision="ieee") + b_k[None, :]

    # 4b. Secondary parent context accumulation
    if HAS_PEER:
        h_ctx = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for s in range(MAX_SECONDARY):
            c_s = tl.load(Context_ptr + pid_k * stride_ck + s * stride_cm + offs_d * stride_cd, mask=mask_d, other=0.0).to(tl.float32)
            p_s = tl.load(Peer_ptr + pid_k * stride_pok + s * stride_pos + offs_m[:, None] * stride_pob + offs_d[None, :] * stride_pod, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            h_ctx += c_s[None, :] * p_s
        y_prim = y_prim + h_ctx

    # 5. Variance scaling
    norm_factor = tl.load(Norm_Factors_ptr + pid_k)
    y_v = y_prim * norm_factor

    # 6. Activation
    if ACTIVATION == 1:
        y_v = tl.minimum(tl.maximum(y_v, 0.0), 6.0)
    elif ACTIVATION == 2:
        y_v = tl.where(y_v >= 0.0, 1.0, -1.0)

    # 7. Store intermediate leaf output tile
    lo_ptrs = Leaf_Outs_ptr + offs_m[:, None] * stride_lob + pid_k * stride_lok + offs_d[None, :] * stride_lod
    tl.store(lo_ptrs, y_v.to(tl.float32), mask=mask_m[:, None] & mask_d[None, :])


def fused_asdag_2d_triton(
    x: torch.Tensor,
    w_stack: torch.Tensor,
    bias_stack: torch.Tensor,
    routing_probs: torch.Tensor,
    context_gates: torch.Tensor,
    norm_factors: torch.Tensor,
    activation: str = "relu6",
    peer_outputs: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    2D Grid-Tiled Fused ASDAG Forward Pass in Triton.
    """
    B, D = x.shape
    K = w_stack.shape[0]
    M_max = context_gates.shape[1] if context_gates.ndim >= 3 else 1

    leaf_outs = torch.empty((B, K, D), device=x.device, dtype=x.dtype)
    act_code = 1 if activation == "relu6" else (2 if activation == "sign" else 0)

    BLOCK_M = 64
    BLOCK_D = triton.next_power_of_2(D)

    grid = (triton.cdiv(B, BLOCK_M), K)

    has_peer = peer_outputs is not None
    dummy_peer = x if not has_peer else peer_outputs

    _fused_asdag_2d_grid_kernel[grid](
        x, w_stack, bias_stack, routing_probs, context_gates, dummy_peer, norm_factors, leaf_outs,
        x.stride(0), x.stride(1),
        w_stack.stride(0), w_stack.stride(1), w_stack.stride(2),
        bias_stack.stride(0), bias_stack.stride(1),
        routing_probs.stride(0), routing_probs.stride(1),
        context_gates.stride(0), context_gates.stride(1), context_gates.stride(2),
        dummy_peer.stride(0) if has_peer else 0,
        dummy_peer.stride(1) if has_peer else 0,
        dummy_peer.stride(2) if has_peer else 0,
        dummy_peer.stride(3) if has_peer else 0,
        leaf_outs.stride(0), leaf_outs.stride(1), leaf_outs.stride(2),
        B,
        DIM=D,
        MAX_SECONDARY=M_max,
        ACTIVATION=act_code,
        HAS_PEER=has_peer,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
    )

    # Reduction across leaves with routing probs
    return torch.einsum('bk, bkd -> bd', routing_probs, leaf_outs)


fused_asdag_forward_triton = fused_asdag_2d_triton
