"""
Multi-Token Prediction (MTP) for ASDAG & Byte-Latent Transformer (BLT)
=====================================================================
Implements DeepSeek-V3 / Meta AI style Multi-Token Prediction modules:
- Sequential auxiliary prediction heads for offsets k in {2, ..., K}
- Joint pretraining foresight loss: L_MTP = L_1 + sum(lambda_k * L_k)
- Self-speculative multi-token decoding engine (K tokens/step without extra draft model)
"""

import math
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.core.bitlinear import BitLinear
from affine_ai.core.norm import RMSNorm


class ASDAGMTPHead(nn.Module):
    """
    Lightweight MTP Auxiliary Prediction Module for offset k.
    DeepSeek-V3 / Meta style projection + normalization + LM head.
    """
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        k_offset: int = 2,
        dtype: Any = torch.float32
    ):
        super().__init__()
        self.k_offset = k_offset
        self.norm = RMSNorm(d_model)
        self.proj = BitLinear(d_model, d_model, bias=False, dtype=dtype)
        self.norm_out = RMSNorm(d_model)
        self.lm_head = BitLinear(d_model, vocab_size, bias=False, dtype=dtype)
        
        # Initialize weights
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        if dtype is not None and dtype != torch.float32:
            self.to(dtype)

    def forward(self, h_latent: torch.Tensor) -> torch.Tensor:
        """
        Takes latent representation h_t from trunk / previous head and predicts target at t + k_offset.
        """
        h_norm = self.norm(h_latent)
        h_act = F.silu(self.proj(h_norm))
        h_fused = self.norm_out(h_latent + h_act)
        logits = self.lm_head(h_fused)
        return logits


class ASDAGMTPModule(nn.Module):
    """
    Multi-Token Prediction (MTP) System:
    Manages (K - 1) sequential auxiliary prediction heads for k in [2 .. K].
    """
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        num_mtp_heads: int = 1,
        mtp_lambda: float = 0.3,
        dtype: Any = torch.float32
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_mtp_heads = num_mtp_heads
        self.mtp_lambda = mtp_lambda
        
        self.heads = nn.ModuleList([
            ASDAGMTPHead(d_model=d_model, vocab_size=vocab_size, k_offset=k + 2, dtype=dtype)
            for k in range(num_mtp_heads)
        ])

    def forward(
        self,
        h_trunk: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        ignore_index: int = -100
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        """
        Computes forward predictions and composite MTP loss across all k in [2 .. K].
        """
        all_logits = []
        total_aux_loss = 0.0
        losses_dict = {}
        
        curr_h = h_trunk
        for idx, head in enumerate(self.heads):
            k = head.k_offset # 2, 3, 4, ...
            logits_k = head(curr_h)
            all_logits.append(logits_k)
            
            if targets is not None:
                # Target for offset k: targets[:, k-1:] aligned with logits[:, :-(k-1)]
                shift_len = k - 1
                if targets.shape[1] > shift_len:
                    tgt_k = targets[:, shift_len:]
                    pred_k = logits_k[:, :-shift_len]
                    loss_k = F.cross_entropy(
                        pred_k.reshape(-1, self.vocab_size),
                        tgt_k.reshape(-1),
                        ignore_index=ignore_index
                    )
                    weight = self.mtp_lambda / (1.0 + 0.2 * idx)
                    total_aux_loss = total_aux_loss + weight * loss_k
                    losses_dict[f"loss_mtp_k{k}"] = loss_k.item()
                    
            curr_h = curr_h.detach() if self.training else curr_h

        loss_tensor = total_aux_loss if targets is not None else None
        return all_logits, loss_tensor, losses_dict
