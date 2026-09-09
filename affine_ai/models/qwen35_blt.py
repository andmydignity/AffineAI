"""
Qwen3.5 Byte Latent Transformer (BLT) Language Model
====================================================
Adapts the pretrained 32-layer Qwen3.5 ASDAG backbone into a True Byte-In, Byte-Out
Byte Latent Transformer (BLT), completely eliminating the 1.21 GB (248,320 x 2560)
token embedding table and replacing it with a ~1.6 MB local byte encoder/decoder.

Architecture:
  UTF-8 Bytes [0..255]
    -> ByteLocalEncoder (128-dim, causal conv)
    -> EntropyPatcher (P=16 byte compression -> [B, M, 2560])
    -> 32x Qwen3.5 ASDAG Blocks (GatedDeltaNet / Attention + POT5 FFN)
    -> Causal Shifted Patches
    -> ByteLocalDecoder (2-Layer SwiGLU + Cross-Fusion)
    -> Next-Byte Logits [B, T, 256]
"""

import os
import gc
import math
from dataclasses import dataclass, fields
from typing import Optional, Tuple, Dict, Any, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.models.qwen35_asdag import Qwen35ASDAGConfig, Qwen35Block, Qwen35RMSNorm
from affine_ai.models.blt import ByteLocalEncoder, EntropyPatcher, ByteLocalDecoder


@dataclass
class Qwen35BLTConfig(Qwen35ASDAGConfig):
    """Configuration for Qwen3.5 Byte Latent Transformer."""
    d_byte: int = 128
    target_patch_size: int = 16
    vocab_size: int = 256  # 256 raw bytes


class Qwen35BLTLanguageModel(nn.Module):
    """
    Qwen3.5 Byte Latent Transformer (BLT):
    - Zero token embedding table (saves 1,212.5 MB).
    - Raw UTF-8 bytes in [0..255].
    - Target patch size P=16 (16x sequence compression through 32 layers).
    - 256 byte output logits.
    """
    def __init__(self, config: Optional[Qwen35BLTConfig] = None):
        super().__init__()
        self.config = config or Qwen35BLTConfig()
        cfg = self.config

        # 1. Local Byte Encoder (Vocab 256 -> d_byte=128)
        self.byte_encoder = ByteLocalEncoder(
            vocab_size=256,
            d_byte=cfg.d_byte,
            kernel_size=4,
            dtype=cfg.dtype
        )

        # 2. Entropy / Fixed Patcher (d_byte -> dim=2560, P=16)
        self.patcher = EntropyPatcher(
            d_byte=cfg.d_byte,
            d_model=cfg.dim,
            target_patch_size=cfg.target_patch_size,
            dtype=cfg.dtype
        )

        # 3. Start-of-Sequence Patch for causal decoding
        self.sos_patch = nn.Parameter(
            torch.randn(1, 1, cfg.dim, dtype=cfg.dtype) * (1.0 / math.sqrt(cfg.dim))
        )

        # 4. Global 32-Layer ASDAG Backbone
        self.blocks = nn.ModuleList([
            Qwen35Block(cfg, layer_idx=i) for i in range(cfg.num_layers)
        ])
        self.output_norm = Qwen35RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)

        # 5. Local Byte Decoder (dim=2560 + d_byte=128 -> Vocab 256)
        self.byte_decoder = ByteLocalDecoder(
            vocab_size=256,
            d_byte=cfg.d_byte,
            d_model=cfg.dim,
            dtype=cfg.dtype
        )

        if cfg.dtype is not None and cfg.dtype != torch.float32:
            self.to(cfg.dtype)

    def load_backbone(self, checkpoint_dir: str = "checkpoints/qwen35_pot5"):
        """Loads the 32 pretrained Qwen3.5 ASDAG blocks and output norm."""
        print(f">>> Loading Qwen3.5 ASDAG Backbone from {checkpoint_dir}...")
        norm_path = os.path.join(checkpoint_dir, "output_norm.pt")
        if os.path.exists(norm_path):
            st = torch.load(norm_path, map_location="cpu", weights_only=True)
            if isinstance(st, torch.Tensor):
                self.output_norm.weight.data.copy_(st.to(self.output_norm.weight.dtype))
            else:
                self.output_norm.load_state_dict(st)
            del st

        loaded_blocks = 0
        for i in range(self.config.num_layers):
            bp = os.path.join(checkpoint_dir, f"block_{i:02d}.pt")
            if os.path.exists(bp):
                st = torch.load(bp, map_location="cpu", weights_only=True)
                self.blocks[i].load_state_dict(st, strict=False)
                del st
                loaded_blocks += 1

        gc.collect()
        print(f">>> Loaded {loaded_blocks}/{self.config.num_layers} backbone blocks successfully!")

    def freeze_backbone(self):
        """Freezes all 32 backbone blocks and output norm, keeping only local byte modules trainable."""
        for p in self.blocks.parameters():
            p.requires_grad = False
        for p in self.output_norm.parameters():
            p.requires_grad = False
        print(">>> Backbone frozen: Only ByteLocalEncoder, Patcher, and ByteLocalDecoder are trainable.")

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Returns parameters of the local byte interface (~822k parameters, ~1.6 MB)."""
        params = []
        params.extend(self.byte_encoder.parameters())
        params.extend(self.patcher.parameters())
        params.append(self.sos_patch)
        params.extend(self.byte_decoder.parameters())
        return [p for p in params if p.requires_grad]

    def forward(
        self,
        byte_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        fixed_patch_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass:
          byte_ids: [B, T] uint8 or int64 raw bytes [0..255]
          targets:  optional [B, T] next-byte targets
        """
        B, T = byte_ids.shape
        P = fixed_patch_size or self.config.target_patch_size

        # Pad sequence to multiple of P if needed
        remainder = T % P
        pad_len = (P - remainder) % P
        if pad_len > 0:
            byte_ids = F.pad(byte_ids, (0, pad_len), value=0)
            if targets is not None:
                targets = F.pad(targets, (0, pad_len), value=-100)
            T_padded = byte_ids.shape[1]
        else:
            T_padded = T

        # 1. Local Byte Encoder (Causal 1D conv + local embeddings)
        h_byte, boundary = self.byte_encoder(byte_ids)

        # 2. Dynamic/Fixed Patcher (groups P bytes into 1 latent patch)
        latent_patches, patch_assignments = self.patcher(
            h_byte, torch.zeros_like(boundary), fixed_patch_size=P
        )
        M = latent_patches.shape[1]

        # 3. Global 32-Layer ASDAG Backbone
        curr_h = latent_patches
        for block in self.blocks:
            curr_h, _ = block(curr_h)

        final_h = self.output_norm(curr_h)

        # 4. Causal shift of patches: byte t conditions on patch (t // P - 1)
        causal_patches = torch.cat([self.sos_patch.expand(B, 1, -1), final_h[:, :-1]], dim=1)

        # 5. Local Byte Decoder (fuses local byte history with global patch context)
        logits = self.byte_decoder(h_byte, causal_patches, patch_assignments)

        # Strip padding from logits and targets if padded
        if pad_len > 0:
            logits = logits[:, :T]
            if targets is not None:
                targets = targets[:, :T]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1), ignore_index=-100)

        return logits, loss

    @torch.no_grad()
    def generate_bytes(
        self,
        prompt_bytes: Union[bytes, str],
        max_new_bytes: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: Optional[str] = None
    ) -> bytes:
        """Autoregressively generates new bytes from a prompt string or byte stream."""
        self.eval()
        if isinstance(prompt_bytes, str):
            raw_bytes = prompt_bytes.encode("utf-8")
        else:
            raw_bytes = prompt_bytes

        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        generated = list(raw_bytes)
        for _ in range(max_new_bytes):
            input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            logits, _ = self.forward(input_tensor)
            next_logits = logits[0, -1, :]  # [256]

            if temperature <= 0.0:
                next_byte = int(next_logits.argmax().item())
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0.0
                    probs = probs / probs.sum().clamp_min(1e-8)
                next_byte = int(torch.multinomial(probs, num_samples=1).item())

            generated.append(next_byte)
            # Stop if null byte or end of stream marker reached
            if next_byte == 0:
                break

        return bytes(generated)
