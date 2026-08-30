"""
AffineAI Custom Hardware Acceleration Kernels (Triton & CUDA)
"""

try:
    from affine_ai.kernels.triton_rms_norm import triton_rms_norm
    from affine_ai.kernels.triton_cross_entropy import triton_fused_linear_cross_entropy
    from affine_ai.kernels.triton_asdag import fused_asdag_forward_triton
    from affine_ai.kernels.triton_lpc import triton_fused_lpc_head
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False
    fused_asdag_forward_triton = None
    triton_rms_norm = None
    triton_fused_linear_cross_entropy = None
    triton_fused_lpc_head = None

__all__ = [
    "fused_asdag_forward_triton",
    "triton_rms_norm",
    "triton_fused_linear_cross_entropy",
    "triton_fused_lpc_head",
    "TRITON_AVAILABLE",
]

