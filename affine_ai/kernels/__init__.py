"""
AffineAI Custom Hardware Acceleration Kernels (Triton & CUDA)
"""

try:
    from affine_ai.kernels.triton_rms_norm import triton_rms_norm
    from affine_ai.kernels.triton_cross_entropy import triton_fused_linear_cross_entropy
    from affine_ai.kernels.triton_asdag import fused_asdag_forward_triton
    from affine_ai.kernels.triton_lpc import triton_fused_lpc_head
    from affine_ai.kernels.triton_ternary import (
        triton_ternary_linear,
        triton_ternary_twin,
        triton_row_amax,
        triton_ternary_linear_fwd,
        triton_fp32_linear,
        triton_ternary_linear_gw,
    )
    from affine_ai.kernels.triton_tree import triton_tree_perm
    from affine_ai.kernels.triton_gla import triton_monarch_chain, triton_fused_monarch_chain, triton_gla_decay
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False
    fused_asdag_forward_triton = None
    triton_rms_norm = None
    triton_fused_linear_cross_entropy = None
    triton_fused_lpc_head = None
    triton_ternary_linear = None
    triton_ternary_twin = None
    triton_row_amax = None
    triton_ternary_linear_fwd = None
    triton_fp32_linear = None
    triton_ternary_linear_gw = None
    triton_tree_perm = None
    triton_monarch_chain = None
    triton_fused_monarch_chain = None
    triton_gla_decay = None

__all__ = [
    "fused_asdag_forward_triton",
    "triton_rms_norm",
    "triton_fused_linear_cross_entropy",
    "triton_fused_lpc_head",
    "triton_ternary_linear",
    "triton_ternary_twin",
    "triton_row_amax",
    "triton_ternary_linear_fwd",
    "triton_fp32_linear",
    "triton_ternary_linear_gw",
    "triton_tree_perm",
    "triton_monarch_chain",
    "triton_fused_monarch_chain",
    "triton_gla_decay",
    "TRITON_AVAILABLE",
]

