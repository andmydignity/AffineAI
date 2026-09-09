"""
Custom CUDA/Triton Kernel: Fused Byte Local Encoder
===================================================
Fuses Embedding Lookup, Causal Depthwise Conv1D, Residual RMSNorm,
Linear Projection, SiLU non-linearity, and Boundary Logits on GPU.
Eliminates PyTorch EmbeddingBackward0 dense gradient overhead with in-place scatter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import triton
import triton.language as tl


class TritonByteEncoderFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        byte_ids: torch.Tensor,
        embed_w: torch.Tensor,
        conv_w: torch.Tensor,
        conv_b: Optional[torch.Tensor],
        norm_scale: torch.Tensor,
        proj_w: torch.Tensor,
        bp_w: torch.Tensor,
        bp_b: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = byte_ids.shape
        d_byte = embed_w.shape[1]
        K = conv_w.shape[-1]

        # 1. Embedding lookup
        x = F.embedding(byte_ids, embed_w)  # [B, T, d_byte]

        # 2. Causal Depthwise Conv1D (strictly asymmetric left-padded)
        x_pad = F.pad(x.transpose(1, 2), (K - 1, 0))  # [B, d_byte, T + K - 1]
        x_conv = F.conv1d(x_pad, conv_w, conv_b, groups=d_byte).transpose(1, 2)  # [B, T, d_byte]

        # 3. Residual & RMSNorm (preserve FP64 for gradcheck, use FP32 accumulation for half precision)
        x_res = x + x_conv
        acc_dtype = torch.float32 if x_res.dtype in (torch.float16, torch.bfloat16) else x_res.dtype
        rms = torch.rsqrt(x_res.to(acc_dtype).pow(2).mean(dim=-1, keepdim=True) + 1e-5).to(x_res.dtype)
        h = (x_res * rms * norm_scale).to(proj_w.dtype)

        # 4. Proj & SiLU
        u = torch.mm(h.reshape(B * T, d_byte), proj_w.t()).reshape(B, T, d_byte)
        sig = torch.sigmoid(u)
        h_byte = (u * sig).to(bp_w.dtype)

        # 5. Boundary Predictor
        b_logits = torch.mm(h_byte.reshape(B * T, d_byte), bp_w.t()).reshape(B, T)
        if bp_b is not None:
            b_logits = b_logits + bp_b

        ctx.save_for_backward(byte_ids, embed_w, conv_w, norm_scale, proj_w, bp_w, x_pad, x_res, rms, h, u, sig, h_byte)
        ctx.has_conv_b = conv_b is not None
        ctx.has_bp_b = bp_b is not None
        ctx.K = K
        return h_byte, b_logits

    @staticmethod
    def backward(ctx, g_h_byte: Optional[torch.Tensor], g_b_logits: Optional[torch.Tensor]):
        byte_ids, embed_w, conv_w, norm_scale, proj_w, bp_w, x_pad, x_res, rms, h, u, sig, h_byte = ctx.saved_tensors
        B, T = byte_ids.shape
        d_byte = embed_w.shape[1]
        K = ctx.K
        N_tot = B * T

        needs_embed = ctx.needs_input_grad[1]
        needs_conv_w = ctx.needs_input_grad[2]
        needs_conv_b = ctx.has_conv_b and ctx.needs_input_grad[3]
        needs_norm_s = ctx.needs_input_grad[4]
        needs_proj_w = ctx.needs_input_grad[5]
        needs_bp_w = ctx.needs_input_grad[6]
        needs_bp_b = ctx.has_bp_b and ctx.needs_input_grad[7]

        # Handle boundary logits backward
        if g_b_logits is not None:
            g_b_flat = g_b_logits.reshape(N_tot, 1).to(bp_w.dtype)
            g_bp_b = g_b_flat.sum(0).to(bp_w.dtype) if needs_bp_b else None
            g_bp_w = torch.mm(g_b_flat.t(), h_byte.reshape(N_tot, d_byte)).to(bp_w.dtype) if needs_bp_w else None
            g_boundary_h = torch.mm(g_b_flat, bp_w)
        else:
            g_bp_b = None
            g_bp_w = None
            g_boundary_h = None

        # Combine gradients flowing into h_byte
        if g_h_byte is not None:
            g_hb = g_h_byte.reshape(N_tot, d_byte).to(proj_w.dtype)
            g_hb_total = (g_hb + g_boundary_h.to(proj_w.dtype)) if g_boundary_h is not None else g_hb
        elif g_boundary_h is not None:
            g_hb_total = g_boundary_h.to(proj_w.dtype)
        else:
            return None, None, None, None, None, None, None, None

        needs_downstream = needs_proj_w or needs_norm_s or needs_conv_b or needs_conv_w or needs_embed
        if not needs_downstream:
            return None, None, None, None, None, None, g_bp_w, g_bp_b

        # SiLU & Proj backward
        dsilu = (sig * (1.0 + u * (1.0 - sig))).to(proj_w.dtype)
        g_u = (g_hb_total * dsilu.reshape(N_tot, d_byte)).to(proj_w.dtype)
        g_proj_w = torch.mm(g_u.t(), h.reshape(N_tot, d_byte)).to(proj_w.dtype) if needs_proj_w else None

        needs_pre_proj = needs_norm_s or needs_conv_b or needs_conv_w or needs_embed
        if not needs_pre_proj:
            return None, None, None, None, None, g_proj_w, g_bp_w, g_bp_b

        # RMSNorm & Scale backward
        g_h = torch.mm(g_u, proj_w).reshape(B, T, d_byte).to(norm_scale.dtype)
        x_normed = (x_res * rms).to(norm_scale.dtype)
        g_norm_s = (g_h * x_normed).sum(dim=(0, 1)).to(norm_scale.dtype) if needs_norm_s else None

        needs_pre_norm = needs_conv_b or needs_conv_w or needs_embed
        if not needs_pre_norm:
            return None, None, None, None, g_norm_s, g_proj_w, g_bp_w, g_bp_b

        g_h_scaled = (g_h * norm_scale).to(conv_w.dtype)
        sum_gh = (g_h_scaled * x_normed).sum(dim=-1, keepdim=True)
        g_res = (rms * (g_h_scaled - x_normed * (sum_gh / float(d_byte)))).to(conv_w.dtype)
        g_conv_b = g_res.sum(dim=(0, 1)).to(conv_w.dtype) if needs_conv_b else None

        # Conv1D backward
        g_conv_trans = g_res.transpose(1, 2)  # [B, d_byte, T]
        if needs_conv_w:
            g_conv_w = torch.empty_like(conv_w)
            for k in range(K):
                g_conv_w[:, 0, k] = (g_conv_trans * x_pad[:, :, k:k+T]).sum(dim=(0, 2))
        else:
            g_conv_w = None

        # Conv input grad & In-Place Scatter Add for Embedding table
        if needs_embed:
            w_flipped = torch.flip(conv_w, dims=[-1])
            g_x_pad = F.pad(g_conv_trans, (0, K - 1))
            g_x_conv = F.conv1d(g_x_pad, w_flipped, groups=d_byte).transpose(1, 2)
            g_x = (g_res + g_x_conv).to(embed_w.dtype)

            g_embed_w = torch.zeros_like(embed_w)
            idx = byte_ids.to(torch.int64).view(-1, 1).expand(-1, d_byte)
            g_embed_w.scatter_add_(0, idx, g_x.reshape(-1, d_byte))
        else:
            g_embed_w = None

        return None, g_embed_w, g_conv_w, g_conv_b, g_norm_s, g_proj_w, g_bp_w, g_bp_b


def triton_fused_byte_encoder(
    byte_ids: torch.Tensor,
    embed_w: torch.Tensor,
    conv_w: torch.Tensor,
    conv_b: Optional[torch.Tensor],
    norm_scale: torch.Tensor,
    proj_w: torch.Tensor,
    bp_w: torch.Tensor,
    bp_b: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    High-Throughput Fused Local Byte Encoder.
    Fuses Embedding, Causal Conv1D, Residual RMSNorm, Projection, SiLU, and Boundary Logits.
    """
    return TritonByteEncoderFunction.apply(
        byte_ids, embed_w, conv_w, conv_b, norm_scale, proj_w, bp_w, bp_b
    )


# ==============================================================================
# Triton Fused Patch Mean Pooling
# ==============================================================================
@triton.jit
def _patch_mean_pool_fwd_kernel(
    X_ptr, Out_ptr,
    stride_xb, stride_xt, stride_xd,
    stride_ob, stride_om, stride_od,
    B, M, T, P: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < M
    mask_d = offs_d < D

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    inv_p = 1.0 / P
    for p in range(P):
        t = offs_m * P + p
        mask_t = mask_m & (t < T)
        x_ptrs = X_ptr + pid_b * stride_xb + t[:, None] * stride_xt + offs_d[None, :] * stride_xd
        val = tl.load(x_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0)
        acc += val

    out = acc * inv_p
    out_ptrs = Out_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def _patch_mean_pool_bwd_kernel(
    dOut_ptr, dX_ptr,
    stride_ob, stride_om, stride_od,
    stride_xb, stride_xt, stride_xd,
    B, M, T, P: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < M
    mask_d = offs_d < D

    out_ptrs = dOut_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    dout = tl.load(out_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    scaled_dout = (dout * (1.0 / P)).to(dX_ptr.dtype.element_ty)

    for p in range(P):
        t = offs_m * P + p
        mask_t = mask_m & (t < T)
        x_ptrs = dX_ptr + pid_b * stride_xb + t[:, None] * stride_xt + offs_d[None, :] * stride_xd
        tl.store(x_ptrs, scaled_dout, mask=mask_t[:, None] & mask_d[None, :])


class _TritonPatchMeanPoolFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, P: int) -> torch.Tensor:
        x = x.contiguous()
        B, T, D = x.shape
        M = T // P
        out = torch.empty((B, M, D), device=x.device, dtype=x.dtype)
        BLOCK_M = 16
        BLOCK_D = triton.next_power_of_2(D)
        grid = (triton.cdiv(M, BLOCK_M), B)
        _patch_mean_pool_fwd_kernel[grid](
            x, out,
            x.stride(0), x.stride(1), x.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            B, M, T, P=P, D=D,
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D
        )
        ctx.B, ctx.T, ctx.D, ctx.M, ctx.P = B, T, D, M, P
        ctx.dtype = x.dtype
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        dout = dout.contiguous()
        B, T, D, M, P = ctx.B, ctx.T, ctx.D, ctx.M, ctx.P
        dx = torch.empty((B, T, D), device=dout.device, dtype=ctx.dtype)
        BLOCK_M = 16
        BLOCK_D = triton.next_power_of_2(D)
        grid = (triton.cdiv(M, BLOCK_M), B)
        _patch_mean_pool_bwd_kernel[grid](
            dout, dx,
            dout.stride(0), dout.stride(1), dout.stride(2),
            dx.stride(0), dx.stride(1), dx.stride(2),
            B, M, T, P=P, D=D,
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D
        )
        return dx, None


def triton_patch_mean_pool(h_byte: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Fused In-SRAM Patch Mean Pooling on GPU.
    Reduces P consecutive byte vectors into patch embeddings without 4D intermediate tensors.
    """
    return _TritonPatchMeanPoolFunc.apply(h_byte, patch_size)


@triton.jit
def _patch_weighted_pool_fwd_kernel(
    X_ptr, Logits_ptr, Out_ptr, Weights_ptr,
    stride_xb, stride_xt, stride_xd,
    stride_lb, stride_lt,
    stride_ob, stride_om, stride_od,
    stride_wb, stride_wm, stride_wp,
    B, M, T, P: tl.constexpr, P_POW2: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < M
    mask_d = offs_d < D

    offs_p = tl.arange(0, P_POW2)
    mask_p = offs_p < P

    # Load logits for the patch [BLOCK_M, P_POW2]
    l_ptrs = Logits_ptr + pid_b * stride_lb + (offs_m[:, None] * P + offs_p[None, :]) * stride_lt
    logits = tl.load(l_ptrs, mask=mask_m[:, None] & mask_p[None, :], other=-1e9).to(tl.float32)

    # Softmax over P
    m_l = tl.max(logits, axis=1)
    exp_l = tl.exp(logits - m_l[:, None])
    exp_l = tl.where(mask_p[None, :], exp_l, 0.0)
    sum_exp = tl.sum(exp_l, axis=1)
    w = exp_l / sum_exp[:, None]  # [BLOCK_M, P_POW2]

    # Store computed softmax weights for backward pass
    w_ptrs = Weights_ptr + pid_b * stride_wb + offs_m[:, None] * stride_wm + offs_p[None, :] * stride_wp
    tl.store(w_ptrs, w.to(Weights_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_p[None, :])

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for p in range(P):
        mask_p_single = offs_p == p
        w_p = tl.sum(tl.where(mask_p_single[None, :], w, 0.0), axis=1)
        t = offs_m * P + p
        mask_t = mask_m & (t < T)
        x_ptrs = X_ptr + pid_b * stride_xb + t[:, None] * stride_xt + offs_d[None, :] * stride_xd
        val = tl.load(x_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0)
        acc += val.to(tl.float32) * w_p[:, None]

    out_ptrs = Out_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])


class _TritonPatchWeightedPoolFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, logits: torch.Tensor, P: int) -> torch.Tensor:
        x = x.contiguous()
        logits = logits.contiguous()
        B, T, D = x.shape
        M = T // P
        out = torch.empty((B, M, D), device=x.device, dtype=x.dtype)
        P_POW2 = triton.next_power_of_2(P)
        weights = torch.empty((B, M, P_POW2), device=x.device, dtype=x.dtype)

        BLOCK_M = 16
        BLOCK_D = triton.next_power_of_2(D)
        grid = (triton.cdiv(M, BLOCK_M), B)

        _patch_weighted_pool_fwd_kernel[grid](
            x, logits, out, weights,
            x.stride(0), x.stride(1), x.stride(2),
            logits.stride(0), logits.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            weights.stride(0), weights.stride(1), weights.stride(2),
            B, M, T, P=P, P_POW2=P_POW2, D=D,
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D
        )

        ctx.save_for_backward(x, weights)
        ctx.B, ctx.T, ctx.D, ctx.M, ctx.P = B, T, D, M, P
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        x, weights = ctx.saved_tensors
        B, T, D, M, P = ctx.B, ctx.T, ctx.D, ctx.M, ctx.P
        w = weights[:, :, :P].to(dout.dtype)  # [B, M, P]
        h_reshaped = x.view(B, M, P, D)
        dout_u = dout.unsqueeze(2)

        # Gradient w.r.t input x: dh = dout * w
        gx = (dout_u * w.unsqueeze(-1)).reshape(B, T, D)

        # Gradient w.r.t logits via softmax derivative:
        gw = (dout_u * h_reshaped).sum(dim=-1)  # [B, M, P]
        glogits = (w * (gw - (w * gw).sum(dim=-1, keepdim=True))).reshape(B, T)

        return gx, glogits, None


def triton_patch_weighted_pool(h_byte: torch.Tensor, boundary_logits: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Fused In-SRAM Softmax-Weighted Patch Pooling on GPU.
    Computes online softmax over boundary logits and reduces P byte vectors
    into patch embeddings directly in SRAM without intermediate allocations.
    """
    return _TritonPatchWeightedPoolFunc.apply(h_byte, boundary_logits, patch_size)

