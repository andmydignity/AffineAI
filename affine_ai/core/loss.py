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

import warnings
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
                # Avoid full (B,N,P*256) alloc: fuse per-patch via triton_fused_linear_cross_entropy as drop-in loop, else chunk
                if hidden_states.is_cuda and lm_head_weight.is_cuda and targets.is_cuda and getattr(kernels, "TRITON_AVAILABLE", False) and self.ignore_index == -100 and getattr(kernels, "triton_fused_linear_cross_entropy", None) is not None:
                    try:
                        total_loss = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)
                        total_valid = 0
                        for p_idx in range(P):
                            w_slice = lm_head_weight[p_idx*256:(p_idx+1)*256]  # [256, D]
                            targets_p = targets.view(B, N, P)[:, :, p_idx]  # [B, N]
                            loss_p = kernels.triton_fused_linear_cross_entropy(hidden_states, w_slice, targets_p)
                            valid_p = int((targets_p != self.ignore_index).sum().item())
                            total_loss = total_loss + loss_p * valid_p
                            total_valid += valid_p
                        if total_valid > 0:
                            return total_loss / total_valid
                        return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
                    except Exception as e:
                        warnings.warn(f"patched triton_fused_linear_cross_entropy per-patch failed: {e}; falling back to chunked", stacklevel=2)
                # Chunked fallback to avoid full alloc (also handles non-100 ignore_index or no Triton)
                # Process hidden in micro-chunks over N dimension
                chunk_n = max(self.chunk_size, 1)
                flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
                # We'll chunk over N*B tokens and project per patch then gather
                total_loss = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)
                total_valid = 0
                B_ = hidden_states.shape[0]
                N_ = hidden_states.shape[1]
                # Chunk over B*N hidden positions
                total_positions = B_ * N_
                batch_dim = B_
                chunk_step = max(chunk_n * batch_dim // P if P else chunk_n, 1)
                # Simpler: loop over hidden chunks, compute logits per patch chunk-wise
                for start in range(0, total_positions, max(self.chunk_size * batch_dim, 1)):
                    end = min(start + max(self.chunk_size * batch_dim, 1), total_positions)
                    h_chunk = flat_hidden[start:end]  # [C, D]
                    # Determine corresponding target indices: need mapping from flat position to (b,n) and patch
                    # For chunked we compute all P patches at once but chunked to limit memory: logits_chunk [C, P*256] = [C, out_features]
                    logits_c = F.linear(h_chunk, lm_head_weight)  # [C, P*256] - still moderate because C is chunked
                    # Map h_chunk positions to byte targets: each hidden position expands to P byte targets
                    # Build flat byte targets for this chunk
                    # flat index -> b = idx // N_, n = idx % N_
                    # byte targets for this chunk are targets[b, n*P:(n+1)*P] flattened
                    # Use vectorized gather via view
                    flat_targets_full = targets.view(B_, N_, P).reshape(B_*N_, P)  # [B*N, P]
                    t_chunk = flat_targets_full[start:end].reshape(-1)  # [C*P]
                    logits_byte_c = logits_c.view(-1, P, 256).reshape(-1, 256)  # [C*P, 256]
                    valid = t_chunk != self.ignore_index
                    if valid.any():
                        loss_c = F.cross_entropy(logits_byte_c[valid], t_chunk[valid], reduction="sum")
                        total_loss = total_loss + loss_c
                        total_valid += int(valid.sum().item())
                if total_valid == 0:
                    return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
                return total_loss / total_valid

        # Standard unpatched sequence: N == T
        if hidden_states.is_cuda and lm_head_weight.is_cuda and targets.is_cuda:
            # 1. Fuse linear projection & online LSE in Triton SRAM directly
            if getattr(kernels, "TRITON_AVAILABLE", False) and self.ignore_index == -100 and getattr(kernels, "triton_fused_linear_cross_entropy", None) is not None:
                try:
                    return kernels.triton_fused_linear_cross_entropy(hidden_states, lm_head_weight, targets)
                except Exception as e:
                    warnings.warn(f"triton_fused_linear_cross_entropy failed: {e}; using chunked fallback", stacklevel=2)

            # 2. Chunked fallback to avoid OOM (copy CPU logic style, chunk over tokens)
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
            total_loss = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)
            for start_idx in range(0, total_tokens, chunk_step):
                end_idx = min(start_idx + chunk_step, total_tokens)
                h_chunk = flat_hidden[start_idx:end_idx]
                t_chunk = flat_targets[start_idx:end_idx]
                logits_chunk = F.linear(h_chunk, lm_head_weight)
                chunk_loss = F.cross_entropy(logits_chunk, t_chunk, reduction="sum", ignore_index=self.ignore_index)
                total_loss = total_loss + chunk_loss
            return (total_loss / valid_tokens).to(hidden_states.dtype)

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
