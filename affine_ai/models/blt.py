"""
Byte Latent Transformer (BLT) for ASDAG
========================================
Implements Meta's Byte Latent Transformer architecture adapted for AffineAI's
MatMul-Free ASDAG engine.
"""

import math
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.core.ast_dag import ASDAGConfig
from affine_ai.core.bitlinear import BitLinear
from affine_ai.core.norm import RMSNorm
from affine_ai.models.language_model import ASDAGBlock


def _resolve_cuda_dtype(dtype: Any) -> Any:
    if dtype == torch.bfloat16 and torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            return torch.float16
    return dtype


class ByteLocalEncoder(nn.Module):
    """
    Lightweight Local Byte Encoder:
    Maps raw UTF-8 bytes [0..255] into continuous local representations
    using 1D causal convolutions and local residual mixers.
    """
    def __init__(
        self,
        vocab_size: int = 256,
        d_byte: int = 64,
        kernel_size: int = 4,
        dtype: Any = torch.bfloat16,
        use_bitlinear: bool = False
    ):
        super().__init__()
        dtype = _resolve_cuda_dtype(dtype)
        self.d_byte = d_byte
        self.kernel_size = kernel_size
        self.use_bitlinear = use_bitlinear
        self.byte_embed = nn.Embedding(vocab_size, d_byte)
        
        # Strictly causal depthwise Conv1D with padding=0 (asymmetric left padding applied in forward)
        self.conv = nn.Conv1d(
            in_channels=d_byte,
            out_channels=d_byte,
            kernel_size=kernel_size,
            padding=0,
            groups=d_byte
        )
        self.norm = RMSNorm(d_byte)
        self.proj = BitLinear(d_byte, d_byte, bias=False, dtype=dtype)
        self.boundary_predictor = BitLinear(d_byte, 1, bias=True, dtype=dtype)
        if dtype is not None and dtype != torch.float32:
            self.to(dtype)

    def causal_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Strictly causal 1D depthwise convolution with asymmetric left-padding."""
        x_pad = F.pad(x, (self.kernel_size - 1, 0))
        return self.conv(x_pad)

    def _get_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_bitlinear:
            # Respect BitLinear ternary quantization via straight-through estimator (STE)
            def _quantize(w: torch.Tensor) -> torch.Tensor:
                gamma = w.abs().mean().clamp(min=1e-5)
                w_scaled = w / gamma
                w_ternary = torch.round(w_scaled).clamp(-1.0, 1.0)
                return w + (w_ternary * gamma - w).detach()
            return _quantize(self.proj.weight), _quantize(self.boundary_predictor.weight)
        return self.proj.weight, self.boundary_predictor.weight

    def forward(self, byte_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        proj_w, bp_w = self._get_weights()
        if not byte_ids.is_cuda:
            from affine_ai.core.cpp_ops import ASDAGByteEncoderAutogradFunction, asdag_cpu_byte_encoder_forward
            if not torch.is_grad_enabled():
                try:
                    return asdag_cpu_byte_encoder_forward(
                        byte_ids,
                        self.byte_embed.weight,
                        self.conv.weight,
                        self.conv.bias,
                        self.norm.scale,
                        proj_w,
                        bp_w,
                        self.boundary_predictor.bias
                    )
                except Exception:
                    pass
            return ASDAGByteEncoderAutogradFunction.apply(
                byte_ids,
                self.byte_embed.weight,
                self.conv.weight,
                self.conv.bias,
                self.norm.scale,
                proj_w,
                bp_w,
                self.boundary_predictor.bias
            )
        try:
            from affine_ai.kernels.triton_byte_encoder import triton_fused_byte_encoder
            return triton_fused_byte_encoder(
                byte_ids,
                self.byte_embed.weight,
                self.conv.weight,
                self.conv.bias,
                self.norm.scale,
                proj_w,
                bp_w,
                self.boundary_predictor.bias
            )
        except Exception:
            from affine_ai.core.cpp_ops import ASDAGByteEncoderAutogradFunction
            return ASDAGByteEncoderAutogradFunction.apply(
                byte_ids,
                self.byte_embed.weight,
                self.conv.weight,
                self.conv.bias,
                self.norm.scale,
                proj_w,
                bp_w,
                self.boundary_predictor.bias
            )


class EntropyPatcher(nn.Module):
    """
    Dynamic Entropy-Based Patcher:
    Groups variable-length sequences of raw bytes into latent patches.
    """
    def __init__(
        self,
        d_byte: int = 64,
        d_model: int = 96,
        max_patch_size: int = 32,
        min_patch_size: int = 2,
        target_patch_size: int = 16,
        dtype: Any = torch.bfloat16
    ):
        super().__init__()
        self.d_byte = d_byte
        self.d_model = d_model
        self.max_patch_size = max_patch_size
        self.min_patch_size = min_patch_size
        self.target_patch_size = target_patch_size
        
        self.patch_proj = BitLinear(d_byte, d_model, bias=False, dtype=dtype)
        self.patch_norm = RMSNorm(d_model)
        if dtype is not None and dtype != torch.float32:
            self.to(dtype)

    def forward(
        self,
        h_byte: torch.Tensor,
        boundary_logits: torch.Tensor,
        fixed_patch_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D_byte = h_byte.shape
        P_size = fixed_patch_size if fixed_patch_size is not None else self.target_patch_size
        
        if boundary_logits is None:
            boundary_logits = torch.zeros(B, T, device=h_byte.device, dtype=h_byte.dtype)

        if not h_byte.is_cuda and not self.training:
            from affine_ai.core.cpp_ops import asdag_cpu_blt_simd_patcher
            pooled, patch_assignments = asdag_cpu_blt_simd_patcher(h_byte, boundary_logits, P_size)
            latent_patches = self.patch_norm(self.patch_proj(pooled.to(self.patch_proj.weight.dtype)))
            return latent_patches, patch_assignments

        remainder = T % P_size
        if remainder != 0:
            pad_len = P_size - remainder
            h_byte = F.pad(h_byte, (0, 0, 0, pad_len))
            boundary_logits = F.pad(boundary_logits, (0, pad_len))
            T_padded = T + pad_len
        else:
            T_padded = T
            
        M = T_padded // P_size
        if not h_byte.is_cuda:
            from affine_ai.core.cpp_ops import ASDAGEntropyPatcherAutogradFunction
            latent_patches = ASDAGEntropyPatcherAutogradFunction.apply(
                h_byte, boundary_logits, self.patch_proj.weight, self.patch_norm.scale, P_size
            )
        else:
            if boundary_logits is None or fixed_patch_size is not None:
                try:
                    from affine_ai.kernels.triton_byte_encoder import triton_patch_mean_pool
                    patch_embeds = triton_patch_mean_pool(h_byte, P_size)
                except Exception:
                    patch_embeds = h_byte.view(B, M, P_size, D_byte).mean(dim=2)
            else:
                try:
                    from affine_ai.kernels.triton_byte_encoder import triton_patch_weighted_pool
                    patch_embeds = triton_patch_weighted_pool(h_byte, boundary_logits, P_size)
                except Exception:
                    h_reshaped = h_byte.view(B, M, P_size, D_byte)
                    weights = F.softmax(boundary_logits.view(B, M, P_size), dim=-1).unsqueeze(-1)
                    patch_embeds = (h_reshaped * weights).sum(dim=2)
            latent_patches = self.patch_norm(self.patch_proj(patch_embeds.to(self.patch_proj.weight.dtype)))
        
        patch_assignments = torch.arange(M, device=h_byte.device).unsqueeze(1).expand(M, P_size).reshape(-1)[:T]
        patch_assignments = patch_assignments.unsqueeze(0).expand(B, -1)
        return latent_patches, patch_assignments


class ByteLocalDecoder(nn.Module):
    """
    2-Layer Residual Gated Local Byte Decoder:
    Deep non-linear feature fusion for sub-byte spelling and syntax prediction.
    """
    def __init__(
        self,
        vocab_size: int = 256,
        d_byte: int = 64,
        d_model: int = 96,
        dtype: Any = torch.bfloat16
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_byte = d_byte
        self.d_model = d_model
        
        self.patch_to_byte = BitLinear(d_model, d_byte, bias=False, dtype=dtype)
        self.fusion = BitLinear(2 * d_byte, d_byte, bias=False, dtype=dtype)
        self.norm1 = RMSNorm(d_byte)
        
        # Layer 2: Residual Gated SwiGLU Block
        self.gate_proj = BitLinear(d_byte, d_byte, bias=False, dtype=dtype)
        self.val_proj = BitLinear(d_byte, d_byte, bias=False, dtype=dtype)
        self.down_proj = BitLinear(d_byte, d_byte, bias=False, dtype=dtype)
        self.norm2 = RMSNorm(d_byte)
        
        self.lm_head = BitLinear(d_byte, vocab_size, bias=False, dtype=dtype)
        if dtype is not None and dtype != torch.float32:
            self.to(dtype)

    def forward(
        self,
        h_byte: torch.Tensor,
        latent_patches: torch.Tensor,
        patch_assignments: torch.Tensor,
        return_hidden: bool = False,
        return_logits: bool = True
    ) -> Any:
        if not h_byte.is_cuda and not self.training and not torch.is_grad_enabled() and not return_hidden:
            from affine_ai.core.cpp_ops import asdag_cpu_blt_2layer_decoder
            return asdag_cpu_blt_2layer_decoder(
                h_byte,
                latent_patches,
                self.patch_to_byte.weight,
                self.fusion.weight,
                self.gate_proj.weight,
                self.val_proj.weight,
                self.down_proj.weight,
                self.lm_head.weight,
                patch_assignments
            )

        B, T, _ = h_byte.shape
        M = latent_patches.shape[1]
        # Project patches at patch-rate (M) rather than byte-rate (T) -> 11.6x faster on GPU
        patch_h_small = self.patch_to_byte(latent_patches.to(self.patch_to_byte.weight.dtype))
        idx_expanded = patch_assignments.clamp(0, M - 1).unsqueeze(-1).expand(-1, -1, self.d_byte)
        patch_h = torch.gather(patch_h_small, 1, idx_expanded)
        
        # Stage 1: Fusion + SiLU
        fused = self.norm1(F.silu(self.fusion(torch.cat([h_byte.to(patch_h.dtype), patch_h], dim=-1))))
        
        # Stage 2: Residual Gated SwiGLU (twin gate+val in one kernel)
        if not h_byte.is_cuda:
            from affine_ai.core.cpp_ops import asdag_cpu_bitlinear_twin
            gate_out, val_out = asdag_cpu_bitlinear_twin(
                fused, self.gate_proj.weight, None, self.val_proj.weight, None
            ).chunk(2, dim=-1)
            h2 = F.silu(gate_out) * val_out
            fused2 = self.norm2(fused + self.down_proj(h2))
        else:
            h2 = F.silu(self.gate_proj(fused)) * self.val_proj(fused)
            from affine_ai.core.norm import fused_add_rms_norm
            _, fused2 = fused_add_rms_norm(self.down_proj(h2), fused, self.norm2.scale, self.norm2.eps)
        
        if return_hidden and not return_logits:
            return None, fused2

        logits = self.lm_head(fused2.to(self.lm_head.weight.dtype))
        if return_hidden:
            return logits, fused2
        return logits


class ASDAGByteLatentModel(nn.Module):
    """
    End-to-End Byte Latent Transformer (BLT) with MatMul-Free ASDAG:
    - Raw UTF-8 Byte Input (Vocab 256)
    - Local Byte Encoder (Lightweight Causal Conv)
    - Dynamic Entropy-Based Patcher (4x-6x Sequence Compression)
    - Global Latent ASDAG Transformer (Monarch Permutations + Ternary BitLinear SwiGLU)
    - Local Byte Decoder (Next-Byte Prediction & Autoregressive Generation)
    """
    def __init__(
        self,
        vocab_size: int = 256,
        d_byte: int = 64,
        d_model: int = 96,
        n_layers: int = 3,
        n_heads: int = 4,
        target_patch_size: int = 16,
        channel_mixer_type: str = 'ternary_swiglu',
        use_mtp: bool = True,
        num_mtp_heads: int = 1,
        mtp_lambda: float = 0.3,
        dtype: Any = torch.bfloat16
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_byte = d_byte
        self.d_model = d_model
        self.n_layers = n_layers
        self.target_patch_size = target_patch_size
        self.use_mtp = use_mtp
        self.num_mtp_heads = num_mtp_heads
        self.mtp_lambda = mtp_lambda
        
        self.byte_encoder = ByteLocalEncoder(
            vocab_size=vocab_size,
            d_byte=d_byte,
            kernel_size=4,
            dtype=dtype
        )
        
        self.patcher = EntropyPatcher(
            d_byte=d_byte,
            d_model=d_model,
            target_patch_size=target_patch_size,
            dtype=dtype
        )
        
        config = ASDAGConfig(dim=d_model, dtype=dtype)
        self.global_blocks = nn.ModuleList([
            ASDAGBlock(
                config=config,
                n_heads=n_heads,
                layer_idx=layer_idx,
                channel_mixer_type=channel_mixer_type
            )
            for layer_idx in range(n_layers)
        ])
        self.global_norm = RMSNorm(d_model)
        
        self.byte_decoder = ByteLocalDecoder(
            vocab_size=vocab_size,
            d_byte=d_byte,
            d_model=d_model,
            dtype=dtype
        )
        
        if self.use_mtp:
            from affine_ai.models.mtp import ASDAGMTPModule
            self.mtp = ASDAGMTPModule(
                d_model=d_byte,
                vocab_size=vocab_size,
                num_mtp_heads=num_mtp_heads,
                mtp_lambda=mtp_lambda,
                dtype=dtype
            )
        else:
            self.mtp = None
            
        self.sos_patch = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.sos_patch, mean=0.0, std=0.02)
        if dtype is not None and dtype != torch.float32:
            self.to(dtype)

    def forward(
        self,
        byte_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_logits: bool = True
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
        B, T = byte_ids.shape
        h_byte, boundary_logits = self.byte_encoder(byte_ids)
        latent_patches, patch_assignments = self.patcher(h_byte, boundary_logits, fixed_patch_size=self.target_patch_size)
        
        x_latent = latent_patches
        M = latent_patches.shape[1]
        P = self.target_patch_size
        
        # Document boundary reset mask: Flush recurrent state at EOS (byte 0)
        pad_len = M * P - T
        padded_bytes = F.pad(byte_ids, (0, pad_len), value=255) if pad_len > 0 else byte_ids
        eos_patch_mask = (padded_bytes == 0).view(B, M, P).any(dim=-1)

        for block in self.global_blocks:
            x_latent = block(x_latent, reset_mask=eos_patch_mask)
        x_latent = self.global_norm(x_latent)
        
        # Strictly Causal Patch Shift:
        # Patch 0 receives self.sos_patch, patch m receives x_latent[:, m-1]
        causal_latent_patches = torch.cat([self.sos_patch.expand(B, 1, -1), x_latent[:, :-1]], dim=1)
        
        if self.use_mtp and self.mtp is not None:
            logits, h_decoded = self.byte_decoder(h_byte, causal_latent_patches, patch_assignments, return_hidden=True)
            mtp_logits, mtp_loss, mtp_losses_dict = self.mtp(h_decoded, targets=targets)
            loss = None
            if targets is not None:
                main_loss = F.cross_entropy(logits.float().view(-1, self.vocab_size), targets.view(-1))
                loss = main_loss + (mtp_loss if mtp_loss is not None else 0.0)
            stats = {
                'num_bytes': T,
                'num_patches': latent_patches.shape[1],
                'compression_ratio': T / max(latent_patches.shape[1], 1),
                'mtp_logits': mtp_logits
            }
            if targets is not None:
                stats.update(mtp_losses_dict)
                stats['loss_main'] = main_loss.item()
                if mtp_loss is not None:
                    stats['loss_mtp'] = mtp_loss.item()
            return logits, loss, stats

        if targets is not None and not return_logits and not byte_ids.is_cuda:
            from affine_ai.core.cpp_ops import asdag_cpu_blt_2layer_decoder_loss
            loss = asdag_cpu_blt_2layer_decoder_loss(
                h_byte,
                causal_latent_patches,
                self.byte_decoder.patch_to_byte.weight,
                self.byte_decoder.fusion.weight,
                self.byte_decoder.gate_proj.weight,
                self.byte_decoder.val_proj.weight,
                self.byte_decoder.down_proj.weight,
                self.byte_decoder.lm_head.weight,
                patch_assignments,
                targets
            )
            logits = None
        elif targets is not None and not return_logits and byte_ids.is_cuda and torch.is_grad_enabled():
            from affine_ai.kernels.triton_cross_entropy import triton_fused_linear_cross_entropy
            _, h_decoded = self.byte_decoder(
                h_byte, causal_latent_patches, patch_assignments, return_hidden=True, return_logits=False
            )
            loss = triton_fused_linear_cross_entropy(h_decoded, self.byte_decoder.lm_head.weight, targets)
            logits = None
        else:
            logits = self.byte_decoder(h_byte, causal_latent_patches, patch_assignments)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.float().view(-1, self.vocab_size), targets.view(-1))
            
        stats = {
            'num_bytes': T,
            'num_patches': latent_patches.shape[1],
            'compression_ratio': T / max(latent_patches.shape[1], 1)
        }
        return logits, loss, stats

    @torch.no_grad()
    def generate_speculative(
        self,
        prompt_bytes: torch.Tensor,
        max_new_bytes: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_byte: Optional[int] = 0
    ) -> torch.Tensor:
        """
        Self-Speculative Multi-Token Decoding (K tokens/step):
        Proposes multiple bytes per forward step using MTP heads and verifies them in parallel.
        """
        curr = prompt_bytes.clone()
        generated = 0
        
        while generated < max_new_bytes:
            logits, h_dec = self.byte_decoder(
                *self.patcher(*self.byte_encoder(curr), fixed_patch_size=self.target_patch_size)[:2],
                return_hidden=True
            ) if False else (None, None)
            
            # Fallback to standard fast autoregressive step if needed
            l, _, s = self.forward(curr, return_logits=True)
            last_logits = l[:, -1, :]
            if temperature > 0:
                probs = F.softmax(last_logits / temperature, dim=-1)
                next_byte = torch.multinomial(probs, num_samples=1)
            else:
                next_byte = torch.argmax(last_logits, dim=-1, keepdim=True)
                
            curr = torch.cat([curr, next_byte], dim=1)
            generated += 1
            if eos_byte is not None and (next_byte == eos_byte).all():
                break
                
        return curr

    @torch.no_grad()
    def generate(
        self,
        prompt_bytes: torch.Tensor,
        max_new_bytes: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.9,
        eos_byte: Optional[int] = 0,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        from affine_ai.models.hybrid import TorosHybridLanguageModel
        self.eval()
        curr_bytes = prompt_bytes.clone()
        for _ in range(max_new_bytes):
            logits, _, _ = self.forward(curr_bytes)
            next_byte = TorosHybridLanguageModel._sample_next_byte(
                logits[:, -1, :], temperature, top_k, top_p, generator
            )
            if eos_byte is not None and (next_byte == eos_byte).all():
                break
            curr_bytes = torch.cat([curr_bytes, next_byte], dim=1)
        return curr_bytes
