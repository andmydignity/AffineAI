import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.core.norm import RMSNorm


class RLSPredictiveHead(nn.Module):
    """
    RLS closed-form posterior local head — conjugate alternative to LocalPredictiveHead.

    Linear map h @ W^T with RMSNorm, where W [vocab, d] has a shared input
    precision P [d, d] (ridge posterior covariance). update() does recursive
    least squares on one-hot targets; uncertainty() returns per-position
    predictive variance h^T P h.

    Double-writer note: weight is updated via RLS (no grad) as a posterior
    mean; exclude this head's weight from AdamW/Muon optimizer or set
    requires_grad=False before training to avoid optimizer+RLS double writes.
    See docs: either `head.weight.requires_grad=False` or filter param groups.

    Drop-in parallel to LocalPredictiveHead:
      forward(h, targets=None) -> (logits, loss) contract parity.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int = 256,
        dtype: Optional[torch.dtype] = torch.float32,
        forgetting: float = 0.999,
        ridge: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.forgetting = forgetting
        self.ridge = ridge
        self.norm = RMSNorm(d_model)
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model, dtype=dtype) * (1.0 / math.sqrt(d_model)))
        # Shared precision P = (X^T X / lambda + ridge*I)^-1, start as I/ridge
        self.register_buffer("P", torch.eye(d_model, dtype=torch.float32) / ridge)
        # _updates as int for backward compat, plus buffer for checkpoint
        self._updates: int = 0
        self.register_buffer("_updates_buf", torch.tensor(0, dtype=torch.long))

    def _sync_updates(self):
        # Keep buffer in sync with int
        if self._updates_buf.item() != self._updates:
            self._updates_buf.fill_(self._updates)

    def forward(
        self,
        h: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ):
        h_n = self.norm(h)
        gamma = self.weight.abs().mean().clamp(min=1e-5)
        w_ternary = torch.round(self.weight / gamma).clamp(-1.0, 1.0)
        w_quant = self.weight + (w_ternary * gamma - self.weight).detach()
        logits = F.linear(h_n.to(self.weight.dtype), w_quant)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size).float(),
            targets.view(-1),
            ignore_index=ignore_index,
        )
        return logits, loss

    @torch.no_grad()
    def update(self, h: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100):
        """Batched RLS: assimilate (h, targets) pairs into W and P. No autograd.
        Uses top-loss subsampling when N>64 (keeps informative samples) and
        Joseph-form symmetrization with denom clamp for stability.
        """
        h_n = self.norm(h).float().reshape(-1, self.d_model)  # [N, d]
        tgt = targets.reshape(-1)  # [N]
        valid = tgt != ignore_index
        if not valid.any():
            return
        h_n = h_n[valid]
        tgt = tgt[valid]
        N = h_n.shape[0]
        if N > 64:
            # Top-loss selection: keep samples with largest CE under current head
            # Fallback to randperm if ranking fails; documented.
            try:
                logits_tmp = F.linear(h_n, self.weight.float())
                per_loss = F.cross_entropy(logits_tmp, tgt, reduction='none')
                _, top_idx = torch.topk(per_loss, 64)
                h_n = h_n[top_idx]
                tgt = tgt[top_idx]
            except Exception:
                idx = torch.randperm(N, device=h_n.device)[:64]
                h_n = h_n[idx]
                tgt = tgt[idx]
            N = 64
        lam = self.forgetting
        for i in range(N):
            x = h_n[i]
            y_idx = tgt[i]
            Px = self.P @ x
            denom = lam + (x @ Px)
            denom = denom.clamp(min=1e-6)
            k = Px / denom
            pred = self.weight.float() @ x
            y_onehot = torch.zeros_like(pred)
            y_onehot.scatter_(0, y_idx.unsqueeze(0), 1.0)
            err = y_onehot - pred
            self.weight.data.add_(torch.outer(err, k).to(self.weight.dtype))
            # Joseph-lite update: P = (P - k (x^T P))/lam, then symmetrize cadence
            self.P.sub_(torch.outer(k, x @ self.P))
            self.P.div_(lam)
            self._updates += 1
            self._updates_buf.fill_(self._updates)
            if self._updates % 16 == 0:
                # Symmetrize every 16 steps (Joseph stabilization cadence)
                self.P.copy_((self.P + self.P.T) * 0.5)

    def uncertainty(self, h: torch.Tensor) -> torch.Tensor:
        """Per-position predictive variance proxy: h^T P h, shape [B, T] or [B*T]."""
        h_n = self.norm(h).float()
        orig_shape = h_n.shape[:-1]
        h_flat = h_n.reshape(-1, self.d_model)
        var = torch.einsum("nd,dd,nd->n", h_flat, self.P, h_flat)
        return var.view(orig_shape)

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        # Ensure int _updates also persisted (buffer already in sd)
        sd["_updates_int"] = torch.tensor(self._updates, dtype=torch.long)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        # Restore int from buffer or legacy key
        if "_updates_int" in state_dict:
            self._updates = int(state_dict["_updates_int"].item())
            self._updates_buf.fill_(self._updates)
        elif "_updates_buf" in state_dict:
            self._updates = int(state_dict["_updates_buf"].item())
        elif "_updates" in state_dict and isinstance(state_dict["_updates"], torch.Tensor):
            self._updates = int(state_dict["_updates"].item())
            self._updates_buf.fill_(self._updates)
        return super().load_state_dict(state_dict, strict=strict)
