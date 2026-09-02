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

    Drop-in parallel to LocalPredictiveHead:
      forward(h, targets=None) -> (logits, loss) contract parity.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int = 256,
        dtype: any = torch.float32,
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
        self._updates = 0

    def forward(
        self,
        h: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ):
        h_n = self.norm(h)
        logits = F.linear(h_n.to(self.weight.dtype), self.weight)
        if targets is None:
            return logits, None
        # Masked CE, ignore_index rows excluded from loss (but h shape unchanged)
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size).float(),
            targets.view(-1),
            ignore_index=ignore_index,
        )
        return logits, loss

    @torch.no_grad()
    def update(self, h: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100):
        """Batched RLS: assimilate (h, targets) pairs into W and P. No autograd."""
        h_n = self.norm(h).float().reshape(-1, self.d_model)  # [N, d]
        tgt = targets.reshape(-1)  # [N]
        valid = tgt != ignore_index
        if not valid.any():
            return
        h_n = h_n[valid]
        tgt = tgt[valid]
        N = h_n.shape[0]
        lam = self.forgetting
        # Iterate rows (sequential dependency on P); keep on device, avoid .item() host sync
        for i in range(N):
            x = h_n[i]
            y_idx = tgt[i]
            Px = self.P @ x
            denom = lam + (x @ Px)
            k = Px / denom
            pred = self.weight.float() @ x
            y_onehot = torch.zeros_like(pred)
            y_onehot.scatter_(0, y_idx.unsqueeze(0), 1.0)
            err = y_onehot - pred
            self.weight.data.add_(torch.outer(err, k).to(self.weight.dtype))
            self.P.sub_(torch.outer(k, x @ self.P))
            self.P.div_(lam)
            self._updates += 1
            if self._updates % 64 == 0:
                self.P.copy_((self.P + self.P.T) * 0.5)

    def uncertainty(self, h: torch.Tensor) -> torch.Tensor:
        """Per-position predictive variance proxy: h^T P h, shape [B, T] or [B*T]."""
        h_n = self.norm(h).float()
        # h_n: [..., d], P: [d, d]
        # var = h^T P h
        orig_shape = h_n.shape[:-1]
        h_flat = h_n.reshape(-1, self.d_model)
        var = torch.einsum("nd,dd,nd->n", h_flat, self.P, h_flat)
        return var.view(orig_shape)
