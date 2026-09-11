"""
AffineAI Custom Hardware Acceleration Kernels (Triton & CUDA)
"""

import torch
import warnings

_cap = (0, 0)
_IS_TURING = False
_IS_AMPERE_PLUS = False
_UNSUPPORTED = False

if torch.cuda.is_available():
    try:
        _dev = torch.cuda.current_device()
        _cap = tuple(torch.cuda.get_device_capability(_dev))
        _UNSUPPORTED = _cap < (7, 5)
        _IS_TURING = (7, 5) <= _cap < (8, 0)
        _IS_AMPERE_PLUS = _cap >= (8, 0)
    except Exception:
        _UNSUPPORTED = False

    if _UNSUPPORTED:
        _cap_str = f"{_cap[0]}{_cap[1]}"
        warnings.warn(
            f"Unsupported GPU sm_{_cap_str}: Triton kernels require Turing sm_75+ (Ampere sm_80+ for bf16/int8). "
            "Falling back to PyTorch paths.",
            stacklevel=2,
        )
    elif _IS_TURING:
        warnings.warn(
            f"Turing GPU sm_{_cap[0]}{_cap[1]}: FP16 AMP via Triton (bf16→fp16, INT8/FP8 → fp16 fallback, 64KB SMEM caps, BLOCK≤64). "
            "Use dtype=torch.float16 + torch.amp.GradScaler() for training.",
            stacklevel=2,
        )


def _optional_import(module_name, symbol):
    try:
        import importlib
        module = importlib.import_module(module_name)
        return getattr(module, symbol)
    except Exception as e:
        warnings.warn(f"{module_name}:{symbol} unavailable: {e}", stacklevel=2)
        return None


triton_rms_norm = _optional_import("affine_ai.kernels.triton_rms_norm", "triton_rms_norm")
triton_fused_add_rms_norm = _optional_import("affine_ai.kernels.triton_rms_norm", "triton_fused_add_rms_norm")
triton_fused_linear_cross_entropy = _optional_import(
    "affine_ai.kernels.triton_cross_entropy", "triton_fused_linear_cross_entropy")
fused_asdag_forward_triton = _optional_import("affine_ai.kernels.triton_asdag", "fused_asdag_forward_triton")
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
triton_differentiable_popc_similarity = _optional_import("affine_ai.kernels.triton_popc", "triton_differentiable_popc_similarity")
TritonPopcSignSimilarityFunction = _optional_import("affine_ai.kernels.triton_popc", "TritonPopcSignSimilarityFunction")
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
try:
    from affine_ai.kernels.triton_adamw import triton_adamw_step as _tmp_step
    triton_adamw_step = _tmp_step
except Exception:
    triton_adamw_step = None

# Fused Permutation Projection
triton_fused_perm_proj = _optional_import("affine_ai.kernels.triton_perm_proj", "triton_fused_perm_proj")

# Sliding Window Attention
sliding_window_attn = _optional_import("affine_ai.kernels.triton_sliding_window", "sliding_window_attn")

if triton_adamw_step is None:
    try:
        import math
        from affine_ai.kernels.triton_adamw import _adamw_kernel
        def triton_adamw_step(p, grad, exp_avg, exp_avg_sq, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=1, master_p=None):
            # I-02/03: fallback shim checks triton and cuda at call time, not import time
            # A-04: heuristic BLOCK_SIZE 256/512/1024 by N same as TritonAdamW, and CPU guard
            import torch as _torch
            try:
                import triton as _triton
            except Exception:
                _triton = None
            if not p.is_cuda or _triton is None or not _torch.cuda.is_available():
                # CPU fallback to torch AdamW logic (A-04)
                has_master = master_p is not None
                mp = master_p if has_master else p
                grad_f32 = grad.float()
                if has_master:
                    if weight_decay != 0.0:
                        mp.mul_(1.0 - lr * weight_decay)
                    exp_avg.mul_(beta1).add_(grad_f32, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad_f32, grad_f32, value=1.0 - beta2)
                    # Use passed betas for correct math
                    bc1 = 1.0 - beta1 ** step
                    bc2 = 1.0 - beta2 ** step
                    step_size_fb = lr / bc1
                    bc2_sqrt_fb = math.sqrt(bc2)
                    denom = (exp_avg_sq.sqrt() / bc2_sqrt_fb).add_(eps)
                    mp.addcdiv_(exp_avg, denom, value=-step_size_fb)
                    p.copy_(mp)
                else:
                    if weight_decay != 0.0:
                        p.mul_(1.0 - lr * weight_decay)
                    exp_avg.mul_(beta1).add_(grad_f32, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad_f32, grad_f32, value=1.0 - beta2)
                    bc1 = 1.0 - beta1 ** step
                    bc2 = 1.0 - beta2 ** step
                    step_size_fb = lr / bc1
                    bc2_sqrt_fb = math.sqrt(bc2)
                    denom = (exp_avg_sq.sqrt() / bc2_sqrt_fb).add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size_fb)
                return
            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            step_size = lr / bc1
            bc2_sqrt = math.sqrt(bc2)
            N = p.numel()
            # A-04: heuristic same as TritonAdamW: 256/512/1024 by N
            if N > 1 << 18:
                BLOCK_SIZE = 1024
            elif N > 1 << 14:
                BLOCK_SIZE = 512
            else:
                BLOCK_SIZE = 256
            grid = (_triton.cdiv(N, BLOCK_SIZE),)
            has_master = master_p is not None
            mp = master_p if has_master else p
            has_wd = weight_decay != 0.0
            _adamw_kernel[grid](
                p, grad, exp_avg, exp_avg_sq,
                lr, beta1, beta2, eps, weight_decay,
                step_size, bc2_sqrt, N, mp,
                HAS_MASTER=has_master, HAS_WD=has_wd, BLOCK_SIZE=BLOCK_SIZE
            )
    except Exception as e:
        import warnings
        warnings.warn(f"triton_adamw_step shim unavailable: {e}", stacklevel=2)
        triton_adamw_step = None

try:
    import triton  # noqa: F401
    _ALL_TRITON_SYMBOLS = [
        triton_rms_norm, triton_fused_add_rms_norm, triton_fused_linear_cross_entropy,
        fused_asdag_forward_triton, triton_fused_lpc_head,
        triton_ternary_linear, triton_ternary_twin, triton_row_amax,
        triton_ternary_linear_fwd, triton_fp32_linear, triton_ternary_linear_gw,
        triton_quantize_x, triton_tree_perm, triton_monarch_chain,
        triton_fused_monarch_chain, triton_gla_decay, triton_router_topk,
        triton_unpack_ternary_2bit, triton_pack_ternary_2bit,
        triton_pack_sign_bits, triton_popc_sign_similarity,
        triton_differentiable_popc_similarity, TritonPopcSignSimilarityFunction,
        triton_int8_imma_linear, triton_pot5_linear, triton_pot5_int8_linear,
        triton_pot5_fused_swiglu, triton_pot5_bitpacked_linear,
        pack_pot5_gpu_3bitplane, unpack_pot5_gpu_3bitplane,
        Triton5StatePOTBitpackedLinear,
        triton_fused_byte_encoder, TritonByteEncoderFunction,
        triton_patch_mean_pool, triton_patch_weighted_pool,
        triton_bitlinear_swiglu, TritonBitLinearSwiGLUFunction,
        TritonAdamW, triton_adamw_step, triton_fused_perm_proj, sliding_window_attn,
    ]
    TRITON_AVAILABLE = torch.cuda.is_available() and not _UNSUPPORTED and any(s is not None for s in _ALL_TRITON_SYMBOLS)
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
    "triton_differentiable_popc_similarity",
    "TritonPopcSignSimilarityFunction",
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
    "sliding_window_attn",
    "TRITON_AVAILABLE",
]


