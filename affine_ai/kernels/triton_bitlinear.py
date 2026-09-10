"""
Custom Triton Kernel: Fused BitLinear SwiGLU (No 2:4 Mask)
=============================================================
Fuses Gate, Value, SiLU non-linearity, and Down projections directly
in GPU SRAM registers with BF16 AMP.
Note: No 2:4 structured sparsity mask is applied; earlier header overstated
"Fused 2:4 Sparse" — this kernel is fused SiLU+Down only.
"""

import math
import warnings
from typing import Optional, Tuple, Any
import torch
import triton
import triton.language as tl


def _is_turing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        cap = torch.cuda.get_device_capability()
        return (7, 5) <= tuple(cap) < (8, 0)
    except Exception:
        return False


def _maybe_cast_fp16_for_turing(t: torch.Tensor) -> torch.Tensor:
    if _is_turing() and t.dtype == torch.bfloat16:
        warnings.warn("Turing sm_75: bf16 unsupported, casting to fp16 (acc fp32)", stacklevel=3)
        return t.to(torch.float16)
    return t


def _prune_turing_block(block: int) -> int:
    if _is_turing() and block > 64:
        warnings.warn(f"Turing sm_75: clamping BLOCK {block} -> 64 (64KB SMEM)", stacklevel=3)
        return 64
    return block


_swiglu_autotune_configs = [
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
]

def _turing_prune_configs(configs):
    if not _is_turing():
        return configs
    pruned = []
    for c in configs:
        bm = c.kwargs.get('BLOCK_M', 32)
        bn = c.kwargs.get('BLOCK_N', 32)
        bk = c.kwargs.get('BLOCK_K', 32)
        if bm <= 64 and bn <= 64 and bk <= 64:
            pruned.append(c)
        else:
            warnings.warn(f"Turing sm_75: pruning BLOCK config {c.kwargs} >64", stacklevel=2)
    return pruned if pruned else configs

_swiglu_autotune_configs = _turing_prune_configs(_swiglu_autotune_configs)


@triton.autotune(configs=_swiglu_autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _swiglu_down_fwd_kernel(
    GV, W_D, OUT, H_ACT,
    stride_gvm, stride_gvn,
    stride_wdk, stride_wdn,
    stride_outm, stride_outk,
    stride_hm, stride_hn,
    Gamma_d_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Intra-Kernel SRAM Fusion Forward:
    - Loads gate and val tiles from DRAM into SRAM registers.
    - Standard CUDA SIMT ALUs compute SiLU(gate) * val directly in SRAM registers.
    - Tensor Cores accumulate tl.dot(h_act, W_down.t()) in SRAM registers.
    - Eliminates DRAM roundtrips; stores final output directly to DRAM.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    gamma_d = tl.load(Gamma_d_ptr).to(tl.float32)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        gate_ptrs = GV + offs_m[:, None] * stride_gvm + offs_n[None, :] * stride_gvn
        val_ptrs = GV + offs_m[:, None] * stride_gvm + (offs_n[None, :] + N) * stride_gvn

        gate = tl.load(gate_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        val = tl.load(val_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

        # Standard CUDA Core SIMT activation in SRAM registers
        gate_f = gate.to(tl.float32)
        val_f = val.to(tl.float32)
        sig = tl.sigmoid(gate_f)
        h_act = (gate_f * sig) * val_f

        # Save h_act to DRAM in fp32 for numerical stability (or recompute in bwd)
        # Per-program host branch on pid_k==0 (constexpr per program); alternative is HAS_STORE constexpr or tl.where
        if pid_k == 0:
            h_ptrs = H_ACT + offs_m[:, None] * stride_hm + offs_n[None, :] * stride_hn
            tl.store(h_ptrs, h_act.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])

        # Load W_down tile [K, N] transposed to [N, K]
        wd_ptrs = W_D + offs_k[None, :] * stride_wdk + offs_n[:, None] * stride_wdn
        wd = tl.load(wd_ptrs, mask=mask_k[None, :] & mask_n[:, None], other=0.0)

        # Tensor Core dot-product accumulation in SRAM (ieee for fp32)
        # Mixed precision: h_act fp32 cast to bf16 for dot gives ~1e-3 precision vs torch fp32; acceptable for SwiGLU path
        acc += tl.dot(h_act.to(W_D.dtype.element_ty), wd, out_dtype=tl.float32, input_precision="ieee")

    acc = acc * gamma_d
    out_ptrs = OUT + offs_m[:, None] * stride_outm + offs_k[None, :] * stride_outk
    tl.store(out_ptrs, acc.to(OUT.dtype.element_ty), mask=mask_m[:, None] & mask_k[None, :])


@triton.autotune(configs=_swiglu_autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _swiglu_bwd_kernel(
    GO, W_D, GV, G_GV,
    stride_gom, stride_gok,
    stride_wdk, stride_wdn,
    stride_gvm, stride_gvn,
    stride_ggvm, stride_ggvn,
    Gamma_d_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Intra-Kernel SRAM Fusion Backward:
    - Tensor Cores accumulate g_hact = GO @ W_D directly in SRAM registers.
    - Standard CUDA SIMT ALUs compute dSiLU and activation backward in SRAM registers.
    - Writes g_gate and g_val directly into G_GV [M, 2*N] in DRAM (zero g_hact DRAM roundtrip).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    gamma_d = tl.load(Gamma_d_ptr).to(tl.float32)

    # Accumulate g_hact in SRAM via Tensor Cores: GO @ W_D
    g_hact = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        go_ptrs = GO + offs_m[:, None] * stride_gom + offs_k[None, :] * stride_gok
        go = tl.load(go_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        wd_ptrs = W_D + offs_k[:, None] * stride_wdk + offs_n[None, :] * stride_wdn
        wd = tl.load(wd_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        g_hact += tl.dot(go, wd, out_dtype=tl.float32, input_precision="ieee")

    g_hact = g_hact * gamma_d

    # Load gate & val in SRAM
    gate_ptrs = GV + offs_m[:, None] * stride_gvm + offs_n[None, :] * stride_gvn
    val_ptrs = GV + offs_m[:, None] * stride_gvm + (offs_n[None, :] + N) * stride_gvn
    gate = tl.load(gate_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
    val = tl.load(val_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

    # CUDA Core SIMT activation gradient in SRAM
    sig = tl.sigmoid(gate)
    dsilu = sig * (1.0 + gate * (1.0 - sig))

    dval = g_hact * (gate * sig)
    dgate = g_hact * (val * dsilu)

    # Write out g_gate & g_val directly into G_GV [M, 2*N]
    g_gate_ptrs = G_GV + offs_m[:, None] * stride_ggvm + offs_n[None, :] * stride_ggvn
    g_val_ptrs = G_GV + offs_m[:, None] * stride_ggvm + (offs_n[None, :] + N) * stride_ggvn

    tl.store(g_gate_ptrs, dgate.to(G_GV.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])
    tl.store(g_val_ptrs, dval.to(G_GV.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


class TritonBitLinearSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_gate_val: torch.Tensor,
        w_down: torch.Tensor,
        gamma_gv: Any = None,
        gamma_d: Any = None
    ) -> torch.Tensor:
        if _is_turing():
            if x.dtype == torch.bfloat16:
                warnings.warn("Turing sm_75: bf16 -> fp16 (bitlinear fwd, acc fp32)", stacklevel=2)
                x = x.to(torch.float16)
            if w_gate_val.dtype == torch.bfloat16:
                warnings.warn("Turing sm_75: bf16 -> fp16 (bitlinear w_gv, acc fp32)", stacklevel=2)
                w_gate_val = w_gate_val.to(torch.float16)
            if w_down.dtype == torch.bfloat16:
                warnings.warn("Turing sm_75: bf16 -> fp16 (bitlinear w_down, acc fp32)", stacklevel=2)
                w_down = w_down.to(torch.float16)
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1]).contiguous()
        M, K = x_flat.shape
        N = w_down.shape[1]

        w_gv_f = w_gate_val.contiguous().to(x_flat.dtype)
        w_d_f = w_down.contiguous().to(x_flat.dtype)

        if gamma_gv is None:
            gamma_gv = w_gv_f.abs().mean().clamp(min=1e-5)
        elif not isinstance(gamma_gv, torch.Tensor):
            gamma_gv = torch.tensor(gamma_gv, device=x_flat.device, dtype=w_gv_f.dtype)

        if gamma_d is None:
            gamma_d = w_d_f.abs().mean().clamp(min=1e-5)
        elif not isinstance(gamma_d, torch.Tensor):
            gamma_d = torch.tensor(gamma_d, device=x_flat.device, dtype=w_d_f.dtype)

        # True BitLinear ternary quantization: round(clip(w / gamma, -1, 1))
        w_gv_q = torch.round(torch.clamp(w_gv_f / gamma_gv, -1.0, 1.0)).to(w_gv_f.dtype)
        w_d_q = torch.round(torch.clamp(w_d_f / gamma_d, -1.0, 1.0)).to(w_d_f.dtype)

        gamma_d_tensor = gamma_d.to(torch.float32) if isinstance(gamma_d, torch.Tensor) else torch.tensor(float(gamma_d), device=x.device, dtype=torch.float32)

        # Step 1: Compute Gate & Val projections via Tensor Cores with quantized weights * gamma
        gv = torch.matmul(x_flat, w_gv_q.t()) * gamma_gv  # [M, 2*N]

        # Step 2 & 3: Fused in-SRAM SiLU(gate) * val + Down projection via Triton
        out = torch.empty((M, K), device=x_flat.device, dtype=x_flat.dtype)
        h_act = torch.empty((M, N), device=x_flat.device, dtype=torch.float32)

        grid_fwd = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(K, META["BLOCK_K"]))
        _swiglu_down_fwd_kernel[grid_fwd](
            gv, w_d_q, out, h_act,
            gv.stride(0), gv.stride(1),
            w_d_q.stride(0), w_d_q.stride(1),
            out.stride(0), out.stride(1),
            h_act.stride(0), h_act.stride(1),
            gamma_d_tensor,
            M, N, K,
        )

        ctx.save_for_backward(x_flat, w_gv_q, w_d_q, gv, h_act, gamma_gv, gamma_d_tensor)
        ctx.orig_shape = orig_shape
        return out.reshape(*orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_flat, w_gv_q, w_d_q, gv, h_act, gamma_gv, gamma_d = ctx.saved_tensors

        go_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()  # [M, K]
        M, K = go_flat.shape
        N = h_act.shape[1]

        # 1. Gradients for W_down (STE: grad flows through quantized weights scaled by gamma)
        # h_act is stored fp32 for precision; cast to go dtype for the cuBLAS matmul.
        g_w_down = torch.matmul(go_flat.t(), h_act.to(go_flat.dtype)) * gamma_d.to(go_flat.dtype) if ctx.needs_input_grad[2] else None

        # 2. Gradients through SwiGLU non-linearity directly fused in SRAM
        g_gv = torch.empty((M, 2 * N), dtype=x_flat.dtype, device=x_flat.device)
        grid_bwd = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
        _swiglu_bwd_kernel[grid_bwd](
            go_flat, w_d_q, gv, g_gv,
            go_flat.stride(0), go_flat.stride(1),
            w_d_q.stride(0), w_d_q.stride(1),
            gv.stride(0), gv.stride(1),
            g_gv.stride(0), g_gv.stride(1),
            gamma_d,
            M, N, K,
        )

        # 3. Gradients for W_gate_val & X (STE)
        g_w_gate_val = torch.matmul(g_gv.t(), x_flat) * gamma_gv if ctx.needs_input_grad[1] else None
        g_x = torch.matmul(g_gv, w_gv_q) * gamma_gv if ctx.needs_input_grad[0] else None

        if g_x is not None:
            g_x = g_x.reshape(*ctx.orig_shape)

        return g_x, g_w_gate_val, g_w_down, None, None


def triton_bitlinear_swiglu(
    x: torch.Tensor,
    w_gate_val: torch.Tensor,
    w_down: torch.Tensor,
    gamma_gv: Any = None,
    gamma_d: Any = None
) -> torch.Tensor:
    """
    High-Performance Fused BitLinear SwiGLU on CUDA with Intra-Kernel SRAM Fusion.
    Turing sm_75: fp16 AMP, acc fp32.
    """
    if _is_turing():
        if x.dtype == torch.bfloat16:
            warnings.warn("Turing sm_75: bf16 -> fp16 (bitlinear api, acc fp32)", stacklevel=2)
            x = x.to(torch.float16)
        if w_gate_val.dtype == torch.bfloat16:
            w_gate_val = w_gate_val.to(torch.float16)
        if w_down.dtype == torch.bfloat16:
            w_down = w_down.to(torch.float16)
    return TritonBitLinearSwiGLUFunction.apply(x, w_gate_val, w_down, gamma_gv, gamma_d)
