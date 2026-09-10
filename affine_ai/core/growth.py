import math
from collections import deque
from typing import Optional, Tuple

import torch


class StickBreakingGrowthController:
    """
    Heuristic stick-breaking-inspired growth criterion for sparse trees/patchers.

    NOTE: This is a heuristic, not a full Dirichlet posterior. It mimics
    truncated stick-breaking intuition: unexplained surprisal maps to pseudo-mass
    on an unseen component via sigmoid, weighted by alpha0. For a proper Bayesian
    treatment, replace with variational inference or exact Dirichlet updates.
    Purely inference-side, no gradients. Keep device tensors to avoid sync.

    The heuristic: unexplained = sigmoid((surprisal - 1.0)*2), grow_mass = unexplained * alpha0/(1+alpha0)
    This is intentionally simple and avoids over-engineered posterior sampling.
    """

    def __init__(self, alpha0: float = 1.0, threshold: float = 0.15, window: int = 128):
        if alpha0 <= 0:
            raise ValueError(f"alpha0 must be >0, got {alpha0}")
        if not 0 < threshold < 1:
            raise ValueError(f"threshold must be in (0,1), got {threshold}")
        if window <= 0:
            raise ValueError(f"window must be >0, got {window}")
        self.alpha0 = float(alpha0)
        self.threshold = float(threshold)
        self.window = int(window)
        self._surprisals: deque[float] = deque(maxlen=window)
        # Keep tensor on device for no-sync path
        self._surprisal_tensor: Optional[torch.Tensor] = None

    @torch.no_grad()
    def update(self, routing_probs: torch.Tensor):
        """routing_probs: [B, K] or [B, T, K] softmax probs. No grad, device-local."""
        if not isinstance(routing_probs, torch.Tensor):
            raise TypeError(f"routing_probs must be Tensor, got {type(routing_probs)}")
        if routing_probs.dim() not in (2, 3):
            raise ValueError(f"routing_probs dim must be 2 or 3, got {routing_probs.dim()}")
        if routing_probs.shape[-1] <= 0:
            raise ValueError("routing_probs last dim must be >0")
        # Ensure probabilities valid (no_nan, sum ~1)
        if torch.isnan(routing_probs).any() or torch.isinf(routing_probs).any():
            raise ValueError("routing_probs contains NaN/Inf")
        if routing_probs.dim() == 3:
            routing_probs = routing_probs.reshape(-1, routing_probs.shape[-1])
        max_p = routing_probs.max(dim=-1).values  # [N] stays on device
        # Compute surprisal without .item() sync: accumulate on CPU via deque of floats still needs item,
        # so keep as tensor mean then single item for deque (min sync). Alternative is to keep window as tensor.
        # Use .detach() and single .item() is one sync per update, acceptable vs per-element.
        # To reduce sync, batch updates: keep _surprisal_tensor append
        surprisal_t = -torch.log(max_p.clamp(min=1e-6)).mean()  # stays on routing_probs device
        # One sync to store in deque for backward compat; keep tensor version as well
        surprisal = float(surprisal_t.item())
        self._surprisals.append(surprisal)
        # Keep tensor window for no-sync should_grow if possible
        if self._surprisal_tensor is None:
            self._surprisal_tensor = surprisal_t.detach().unsqueeze(0)
        else:
            self._surprisal_tensor = torch.cat([self._surprisal_tensor, surprisal_t.detach().unsqueeze(0)], dim=0)
            if self._surprisal_tensor.numel() > self.window:
                self._surprisal_tensor = self._surprisal_tensor[-self.window:]

    @torch.no_grad()
    def should_grow(self) -> Tuple[bool, float]:
        if len(self._surprisals) < 16:
            return False, 0.0
        # Vectorized window mean: use tensor path if available, else numpy (no Python sum loop)
        if self._surprisal_tensor is not None and self._surprisal_tensor.numel() == len(self._surprisals):
            mean_surprisal = float(self._surprisal_tensor.float().mean().item())
        else:
            mean_surprisal = float(np.mean(np.fromiter(self._surprisals, dtype=np.float64)))
        # Map surprisal to pseudo mass on new component (higher surprisal -> grow)
        # Heuristic: unexplained mass ~ sigmoid((surprisal - 1.0) * 2)
        unexplained = 1.0 / (1.0 + math.exp(-(mean_surprisal - 1.0) * 2.0))
        # Weight by alpha0
        grow_mass = unexplained * self.alpha0 / (1.0 + self.alpha0)
        return grow_mass > self.threshold, grow_mass

    def reset(self):
        self._surprisals.clear()
        self._surprisal_tensor = None
