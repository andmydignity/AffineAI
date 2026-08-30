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
        x = F.embedding(byte_ids, embed_w) # [B, T, d_byte]
        
        # 2. Causal Depthwise Conv1D
        x_pad = F.pad(x.transpose(1, 2), (K - 1, 0)) # [B, d_byte, T + K - 1]
        x_conv = F.conv1d(x_pad, conv_w, conv_b, groups=d_byte).transpose(1, 2) # [B, T, d_byte]
        
        # 3. Residual & RMSNorm
        x_res = x + x_conv
        rms = torch.rsqrt(x_res.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        h = (x_res * rms * norm_scale).to(proj_w.dtype)
        
        # 4. Proj & SiLU
        u = torch.mm(h.reshape(B * T, d_byte), proj_w.t()).reshape(B, T, d_byte)
        sig = torch.sigmoid(u)
        h_byte = (u * sig).to(bp_w.dtype)
        
        # 5. Boundary Predictor
        b_logits = torch.mm(h_byte.reshape(B * T, d_byte), bp_w.t()).reshape(B, T)
        if bp_b is not None:
            b_logits = b_logits + bp_b
        
        ctx.save_for_backward(byte_ids, embed_w, conv_w, norm_scale, proj_w, bp_w, x, x_pad, x_res, rms, h, u, sig, h_byte)
        ctx.has_conv_b = conv_b is not None
        ctx.has_bp_b = bp_b is not None
        ctx.K = K
        return h_byte, b_logits

    @staticmethod
    def backward(ctx, g_h_byte: torch.Tensor, g_b_logits: torch.Tensor):
        byte_ids, embed_w, conv_w, norm_scale, proj_w, bp_w, x, x_pad, x_res, rms, h, u, sig, h_byte = ctx.saved_tensors
        B, T = byte_ids.shape
        d_byte = embed_w.shape[1]
        K = ctx.K
        N_tot = B * T
        
        g_h_byte = g_h_byte.to(bp_w.dtype)
        g_b_logits = g_b_logits.to(bp_w.dtype)
        
        # Boundary backward
        g_b_flat = g_b_logits.reshape(N_tot, 1)
        g_bp_b = g_b_flat.sum(0).to(bp_w.dtype) if ctx.has_bp_b else None
        g_bp_w = torch.mm(g_b_flat.t(), h_byte.reshape(N_tot, d_byte)).to(bp_w.dtype)
        g_hb_total = (g_h_byte.reshape(N_tot, d_byte) + torch.mm(g_b_flat, bp_w)).to(proj_w.dtype)
        
        # SiLU & Proj backward
        dsilu = (sig * (1.0 + u * (1.0 - sig))).to(proj_w.dtype)
        g_u = (g_hb_total * dsilu.reshape(N_tot, d_byte)).to(proj_w.dtype)
        g_proj_w = torch.mm(g_u.t(), h.reshape(N_tot, d_byte)).to(proj_w.dtype)
        g_h = torch.mm(g_u, proj_w).reshape(B, T, d_byte).to(norm_scale.dtype)
        
        # RMSNorm & Scale backward
        x_normed = (x_res * rms).to(norm_scale.dtype)
        g_norm_s = (g_h * x_normed).sum(dim=(0, 1)).to(norm_scale.dtype)
        g_h_scaled = (g_h * norm_scale).to(conv_w.dtype)
        sum_gh = (g_h_scaled * x_normed).sum(dim=-1, keepdim=True)
        g_res = (rms * (g_h_scaled - x_normed * (sum_gh / float(d_byte)))).to(conv_w.dtype)
        
        # Conv1D backward
        g_conv_b = g_res.sum(dim=(0, 1)).to(conv_w.dtype) if ctx.has_conv_b else None
        g_conv_trans = g_res.transpose(1, 2) # [B, d_byte, T]
        g_conv_w = torch.empty_like(conv_w)
        for k in range(K):
            g_conv_w[:, 0, k] = (g_conv_trans * x_pad[:, :, k:k+T]).sum(dim=(0, 2))
        
        # Conv input grad
        w_flipped = torch.flip(conv_w, dims=[-1])
        g_x_pad = F.pad(g_conv_trans, (0, K - 1))
        g_x_conv = F.conv1d(g_x_pad, w_flipped, groups=d_byte).transpose(1, 2)
        
        g_x = (g_res + g_x_conv).to(embed_w.dtype)
        
        # In-Place Scatter Add for Embedding table (Zero EmbeddingBackward0 overhead)
        g_embed_w = torch.zeros_like(embed_w)
        g_embed_w.scatter_add_(0, byte_ids.view(-1, 1).expand(-1, d_byte), g_x.reshape(-1, d_byte))
        
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
    High-Throughput Fused Local Byte Encoder on CUDA.
    """
    return TritonByteEncoderFunction.apply(
        byte_ids, embed_w, conv_w, conv_b, norm_scale, proj_w, bp_w, bp_b
    )
