import math
from typing import Optional, Tuple

import torch


class StickBreakingGrowthController:
    """
    Truncated stick-breaking growth criterion for sparse trees/patchers.

    Maintains a pseudo-count for an unseen component (alpha0) in a Dirichlet
    posterior. Surprisal under current routing (max prob) moves posterior mass
    toward growth. Purely inference-side, no gradients.
    """

    def __init__(self, alpha0: float = 1.0, threshold: float = 0.15, window: int = 128):
        self.alpha0 = alpha0
        self.threshold = threshold
        self.window = window
        self._surprisals: list[float] = []

    def update(self, routing_probs: torch.Tensor):
        """routing_probs: [B, K] or [B, T, K] softmax probs."""
        if routing_probs.dim() == 3:
            routing_probs = routing_probs.reshape(-1, routing_probs.shape[-1])
        max_p = routing_probs.max(dim=-1).values  # [N]
        surprisal = -torch.log(max_p.clamp(min=1e-6)).mean().item()
        self._surprisals.append(surprisal)
        if len(self._surprisals) > self.window:
            self._surprisals.pop(0)

    def should_grow(self) -> Tuple[bool, float]:
        if len(self._surprisals) < 16:
            return False, 0.0
        mean_surprisal = sum(self._surprisals) / len(self._surprisals)
        # Map surprisal to pseudo mass on new component (higher surprisal -> grow)
        # Heuristic: unexplained mass ~ sigmoid((surprisal - 1.0) * 2)
        unexplained = 1.0 / (1.0 + math.exp(-(mean_surprisal - 1.0) * 2.0))
        # Weight by alpha0
        grow_mass = unexplained * self.alpha0 / (1.0 + self.alpha0)
        return grow_mass > self.threshold, grow_mass

    def reset(self):
        self._surprisals.clear()
