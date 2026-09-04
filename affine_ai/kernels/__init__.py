"""
AffineAI Custom Hardware Acceleration Kernels (Triton & CUDA)
"""

def _optional_import(module_name, symbol):
    try:
        import importlib
        module = importlib.import_module(module_name)
        return getattr(module, symbol)
    except Exception:
        return None


triton_rms_norm = _optional_import("affine_ai.kernels.triton_rms_norm", "triton_rms_norm")
triton_fused_add_rms_norm = _optional_import("affine_ai.kernels.triton_rms_norm", "triton_fused_add_rms_norm")
triton_fused_linear_cross_entropy = _optional_import(
    "affine_ai.kernels.triton_cross_entropy", "triton_fused_linear_cross_entropy")
try:
    from affine_ai.kernels.triton_asdag import fused_asdag_forward_triton
except Exception:
    fused_asdag_forward_triton = None
triton_fused_lpc_head = _optional_import("affine_ai.kernels.triton_lpc", "triton_fused_lpc_head")
triton_ternary_linear = _optional_import("affine_ai.kernels.triton_ternary", "triton_ternary_linear")
triton_ternary_twin = _optional_import("affine_ai.kernels.triton_ternary", "triton_ternary_twin")
triton_row_amax = _optional_import("affine_ai.kernels.triton_ternary", "triton_row_amax")
triton_ternary_linear_fwd = _optional_import("affine_ai.kernels.triton_ternary", "triton_ternary_linear_fwd")
triton_fp32_linear = _optional_import("affine_ai.kernels.triton_ternary", "triton_fp32_linear")
triton_ternary_linear_gw = _optional_import("affine_ai.kernels.triton_ternary", "triton_ternary_linear_gw")
triton_tree_perm = _optional_import("affine_ai.kernels.triton_tree", "triton_tree_perm")
triton_monarch_chain = _optional_import("affine_ai.kernels.triton_gla", "triton_monarch_chain")
triton_fused_monarch_chain = _optional_import("affine_ai.kernels.triton_gla", "triton_fused_monarch_chain")
triton_gla_decay = _optional_import("affine_ai.kernels.triton_gla", "triton_gla_decay")
triton_router_topk = _optional_import("affine_ai.kernels.triton_router", "triton_router_topk")
triton_unpack_ternary_2bit = _optional_import("affine_ai.kernels.triton_ternary", "triton_unpack_ternary_2bit")
triton_pack_ternary_2bit = _optional_import("affine_ai.kernels.triton_ternary", "triton_pack_ternary_2bit")
triton_pack_sign_bits = _optional_import("affine_ai.kernels.triton_popc", "triton_pack_sign_bits")
triton_popc_sign_similarity = _optional_import("affine_ai.kernels.triton_popc", "triton_popc_sign_similarity")
triton_int8_imma_linear = _optional_import("affine_ai.kernels.triton_int8_imma", "triton_int8_imma_linear")

TRITON_AVAILABLE = triton_rms_norm is not None

__all__ = [
    "fused_asdag_forward_triton",
    "triton_rms_norm",
    "triton_fused_add_rms_norm",
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
    "triton_router_topk",
    "triton_unpack_ternary_2bit",
    "triton_pack_ternary_2bit",
    "triton_pack_sign_bits",
    "triton_popc_sign_similarity",
    "triton_int8_imma_linear",
    "TRITON_AVAILABLE",
]

