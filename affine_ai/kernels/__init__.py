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
triton_quantize_x = _optional_import("affine_ai.kernels.triton_ternary", "triton_quantize_x")
triton_tree_perm = _optional_import("affine_ai.kernels.triton_tree", "triton_tree_perm")
triton_monarch_chain = _optional_import("affine_ai.kernels.triton_monarch", "triton_monarch_chain")
if triton_monarch_chain is None:
    triton_monarch_chain = _optional_import("affine_ai.kernels.triton_gla", "triton_monarch_chain")
triton_fused_monarch_chain = _optional_import("affine_ai.kernels.triton_gla", "triton_fused_monarch_chain")
triton_gla_decay = _optional_import("affine_ai.kernels.triton_gla", "triton_gla_decay")
triton_router_topk = _optional_import("affine_ai.kernels.triton_router", "triton_router_topk")
triton_unpack_ternary_2bit = _optional_import("affine_ai.kernels.triton_ternary", "triton_unpack_ternary_2bit")
triton_pack_ternary_2bit = _optional_import("affine_ai.kernels.triton_ternary", "triton_pack_ternary_2bit")
triton_pack_sign_bits = _optional_import("affine_ai.kernels.triton_popc", "triton_pack_sign_bits")
triton_popc_sign_similarity = _optional_import("affine_ai.kernels.triton_popc", "triton_popc_sign_similarity")
triton_int8_imma_linear = _optional_import("affine_ai.kernels.triton_int8_imma", "triton_int8_imma_linear")
triton_pot5_linear = _optional_import("affine_ai.kernels.triton_pot5", "triton_pot5_linear")
triton_pot5_int8_linear = _optional_import("affine_ai.kernels.triton_pot5", "triton_pot5_int8_linear")
triton_pot5_fused_swiglu = _optional_import("affine_ai.kernels.triton_pot5", "triton_pot5_fused_swiglu")
triton_pot5_bitpacked_linear = _optional_import("affine_ai.kernels.triton_pot5", "triton_pot5_bitpacked_linear")
pack_pot5_gpu_3bitplane = _optional_import("affine_ai.kernels.triton_pot5", "pack_pot5_gpu_3bitplane")
unpack_pot5_gpu_3bitplane = _optional_import("affine_ai.kernels.triton_pot5", "unpack_pot5_gpu_3bitplane")
Triton5StatePOTBitpackedLinear = _optional_import("affine_ai.kernels.triton_pot5", "Triton5StatePOTBitpackedLinear")

# Fused Byte Local Encoder & Patcher
triton_fused_byte_encoder = _optional_import("affine_ai.kernels.triton_byte_encoder", "triton_fused_byte_encoder")
TritonByteEncoderFunction = _optional_import("affine_ai.kernels.triton_byte_encoder", "TritonByteEncoderFunction")
triton_patch_mean_pool = _optional_import("affine_ai.kernels.triton_byte_encoder", "triton_patch_mean_pool")
triton_patch_weighted_pool = _optional_import("affine_ai.kernels.triton_byte_encoder", "triton_patch_weighted_pool")


# BitLinear SwiGLU
triton_bitlinear_swiglu = _optional_import("affine_ai.kernels.triton_bitlinear", "triton_bitlinear_swiglu")
TritonBitLinearSwiGLUFunction = _optional_import("affine_ai.kernels.triton_bitlinear", "TritonBitLinearSwiGLUFunction")

# Fused AdamW Step
TritonAdamW = _optional_import("affine_ai.kernels.triton_adamw", "TritonAdamW")
triton_adamw_step = _optional_import("affine_ai.kernels.triton_adamw", "triton_adamw_step")

# Fused Permutation Projection
triton_fused_perm_proj = _optional_import("affine_ai.kernels.triton_perm_proj", "triton_fused_perm_proj")

if triton_adamw_step is None:
    try:
        import math
        from affine_ai.kernels.triton_adamw import _adamw_kernel
        def triton_adamw_step(p, grad, exp_avg, exp_avg_sq, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=1, master_p=None):
            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            step_size = lr / bc1
            bc2_sqrt = math.sqrt(bc2)
            N = p.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(N, BLOCK_SIZE),)
            has_master = master_p is not None
            mp = master_p if has_master else p
            _adamw_kernel[grid](
                p, grad, exp_avg, exp_avg_sq,
                lr, beta1, beta2, eps, weight_decay,
                step_size, bc2_sqrt, N, mp,
                HAS_MASTER=has_master, BLOCK_SIZE=BLOCK_SIZE
            )
    except Exception:
        triton_adamw_step = None

try:
    import torch
    import triton
    TRITON_AVAILABLE = torch.cuda.is_available() and (triton_rms_norm is not None)
    if torch.cuda.is_available():
        try:
            _cap = tuple(torch.cuda.get_device_capability())
            # Ampere (sm_80) is the minimum supported arch: INT8 m16n8k32,
            # BF16/TF32 Tensor Cores and large-SMEM autotune configs assume it.
            # FP8 paths additionally need Ada (sm_89+); they guard separately.
            _AMPERE_MIN = _cap < (8, 0)
        except Exception:
            _AMPERE_MIN = False
        if _AMPERE_MIN:
            import warnings as _warnings
            _warnings.warn(
                f"Unsupported GPU sm_{_cap[0]}{_cap[1]}: Triton kernels require Ampere (sm_80)+. "
                "Falling back to PyTorch paths where available.",
                stacklevel=2,
            )
            TRITON_AVAILABLE = False
except Exception:
    TRITON_AVAILABLE = False

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
    "triton_pot5_linear",
    "triton_pot5_int8_linear",
    "triton_pot5_fused_swiglu",
    "triton_pot5_bitpacked_linear",
    "pack_pot5_gpu_3bitplane",
    "unpack_pot5_gpu_3bitplane",
    "Triton5StatePOTBitpackedLinear",
    "triton_fused_byte_encoder",
    "TritonByteEncoderFunction",
    "triton_patch_mean_pool",
    "triton_patch_weighted_pool",
    "triton_bitlinear_swiglu",
    "TritonBitLinearSwiGLUFunction",
    "TritonAdamW",
    "triton_adamw_step",
    "triton_fused_perm_proj",
    "TRITON_AVAILABLE",
]


