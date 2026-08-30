"""
Memory-Efficient Chunked & Fused Cross-Entropy Loss
===================================================
Computes Cross-Entropy loss via smart auto-dispatch:
- For small byte-level vocabularies (V <= 256): Dispatches to high-speed cuBLAS Tensor Cores.
- For multi-byte patched heads (V = P * 256): Projects (B, N, D) -> (B, N, P, 256) -> (B, T, 256) for exact byte-level loss.
- For standard/large vocabularies (V > 256 unpatched): Dispatches to Triton Fused Linear Cross-Entropy with
  online Log-Sum-Exp in SRAM, preventing multi-gigabyte DRAM logit allocations.
- For CPU fallback: Employs memory-efficient sequential micro-chunking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import affine_ai.kernels as kernels


class ChunkedCrossEntropyLoss(nn.Module):
    """
    Smart auto-dispatching Cross Entropy Loss module.
    
    Args:
        chunk_size: Sequence chunk size for micro-projections (default 32).
        ignore_index: Target index to ignore in loss calculation (default -100).
    """
    def __init__(self, chunk_size: int = 32, ignore_index: int = -100):
        super().__init__()
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Normalized output states of shape (B, N, d_model)
            lm_head_weight: Linear projection weights of shape (out_features, d_model)
            targets: Target token IDs of shape (B, T)
            
        Returns:
            Scalar cross-entropy loss
        """
        B = hidden_states.shape[0]
        N = hidden_states.shape[1]
        T = targets.shape[1]
        out_features = lm_head_weight.shape[0]

        # Multi-byte patched head: N != T, out_features = P * 256, where P = T // N
        if N != T and out_features % 256 == 0:
            P = out_features // 256
            if N * P == T:
                logits = F.linear(hidden_states, lm_head_weight)  # (B, N, P * 256)
                logits_byte = logits.view(B, N, P, 256).reshape(B, T, 256)
                return F.cross_entropy(logits_byte.reshape(-1, 256), targets.reshape(-1), ignore_index=self.ignore_index)

        # Standard unpatched sequence: N == T
        if hidden_states.is_cuda and lm_head_weight.is_cuda and targets.is_cuda:
            # 1. For large vocabularies (unpatched), fuse linear projection & online LSE in Triton SRAM
            if out_features > 256 and getattr(kernels, "TRITON_AVAILABLE", False) and self.ignore_index == -100:
                return kernels.triton_fused_linear_cross_entropy(hidden_states, lm_head_weight, targets)

            # 2. For byte-level vocabularies (V <= 256), use monolithic cuBLAS Tensor Cores
            elif out_features <= 256:
                logits = F.linear(hidden_states, lm_head_weight)
                return F.cross_entropy(logits.reshape(-1, out_features), targets.reshape(-1), ignore_index=self.ignore_index)

        # 3. CPU / Fallback micro-chunking to conserve RAM
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_targets = targets.reshape(-1)
        total_tokens = flat_hidden.shape[0]

        if total_tokens == 0:
            return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        valid_mask = flat_targets != self.ignore_index
        valid_tokens = valid_mask.sum()
        if valid_tokens == 0:
            return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        batch_dim = hidden_states.shape[0] if hidden_states.ndim > 2 else 1
        chunk_step = max(self.chunk_size * batch_dim, 1)

        total_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        for start_idx in range(0, total_tokens, chunk_step):
            end_idx = min(start_idx + chunk_step, total_tokens)
            h_chunk = flat_hidden[start_idx:end_idx]
            t_chunk = flat_targets[start_idx:end_idx]

            logits_chunk = F.linear(h_chunk, lm_head_weight)
            chunk_loss = F.cross_entropy(logits_chunk, t_chunk, reduction="sum", ignore_index=self.ignore_index)
            total_loss = total_loss + chunk_loss

        return total_loss / valid_tokens
