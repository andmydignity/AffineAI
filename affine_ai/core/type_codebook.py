import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentTypeCodebook(nn.Module):
    """
    Type-vs-instance conditioning: discrete code over patch latents.

    Online Gaussian types with stick-breaking birth and BMR merging.
    Embedding is zero-init scale, so enrichment is function-preserving at init.
    """

    def __init__(self, dim: int, max_types: int = 32, birth_threshold: float = 1.5, merge_threshold: float = 0.8):
        super().__init__()
        self.dim = dim
        self.max_types = max_types
        self.birth_threshold = birth_threshold
        self.merge_threshold = merge_threshold
        self.embedding = nn.Embedding(max_types, dim)
        nn.init.zeros_(self.embedding.weight)
        self.scale = nn.Parameter(torch.zeros(1))
        self.register_buffer("means", torch.zeros(max_types, dim))
        self.register_buffer("counts", torch.zeros(max_types))
        self.register_buffer("num_types", torch.tensor(1))
        # init first type at zero
        self.means[0] = torch.zeros(dim)

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """latents [B, M, dim] -> enriched [B, M, dim], ids [B, M]"""
        B, M, D = latents.shape
        active = int(self.num_types.item())
        if active == 0:
            return latents, torch.zeros(B, M, dtype=torch.long, device=latents.device)
        # distance to each active type mean
        # latents [B*M, D], means [K, D] -> dist [B*M, K]
        flat = latents.reshape(-1, D)
        dists = torch.cdist(flat, self.means[:active])
        ids = dists.argmin(dim=-1).view(B, M)
        # enrichment: zero-init scale -> identity at start
        enrich = self.embedding(ids) * self.scale
        return latents + enrich, ids

    @torch.no_grad()
    def update(self, latents: torch.Tensor):
        """Online update: EMA means, stick-breaking birth for far latents."""
        flat = latents.detach().reshape(-1, self.dim)
        active = int(self.num_types.item())
        if active == 0:
            return
        dists = torch.cdist(flat, self.means[:active])
        min_dist, nearest = dists.min(dim=-1)
        # Birth: far latents spawn new type if room
        for i in range(flat.shape[0]):
            if min_dist[i].item() > self.birth_threshold and active < self.max_types:
                self.means[active] = flat[i]
                self.counts[active] = 1.0
                active += 1
                self.num_types.fill_(active)
                # recompute dists after birth (simple: continue)
                dists = torch.cdist(flat, self.means[:active])
                min_dist, nearest = dists.min(dim=-1)
            else:
                k = int(nearest[i].item())
                # EMA update
                self.counts[k] += 1
                n = self.counts[k].item()
                lr = 1.0 / n
                self.means[k] = (1 - lr) * self.means[k] + lr * flat[i]

    @torch.no_grad()
    def bmr_merge(self) -> int:
        """BMR-style merge: merge closest pair if distance < merge_threshold."""
        active = int(self.num_types.item())
        if active <= 1:
            return 0
        # Find closest pair among active types
        dists = torch.cdist(self.means[:active], self.means[:active])
        dists.fill_diagonal_(float("inf"))
        min_val, min_idx = dists.min(dim=-1)
        # global min
        best = min_val.argmin().item()
        other = int(min_idx[best].item())
        if min_val[best].item() >= self.merge_threshold:
            return 0
        a, b = best, other
        if a > b:
            a, b = b, a
        # Merge b into a (keep a, drop b)
        ca, cb = self.counts[a].item(), self.counts[b].item()
        if ca + cb == 0:
            return 0
        self.means[a] = (ca * self.means[a] + cb * self.means[b]) / (ca + cb)
        self.counts[a] = ca + cb
        # Shift down types after b
        for k in range(b, active - 1):
            self.means[k] = self.means[k + 1]
            self.counts[k] = self.counts[k + 1]
        self.means[active - 1].zero_()
        self.counts[active - 1] = 0
        self.num_types.fill_(active - 1)
        # Merge embeddings similarly (zero-init so merging is trivial, but keep consistent)
        with torch.no_grad():
            w = self.embedding.weight
            # Move b's embedding into a's slot via average (zero-init -> still zero)
            w[a] = (w[a] * ca + w[b] * cb) / (ca + cb) if (ca + cb) > 0 else w[a]
            # Shift
            for k in range(b, active - 1):
                w[k] = w[k + 1]
            w[active - 1].zero_()
        return 1

    def extra_repr(self):
        return f"dim={self.dim}, max_types={self.max_types}, active={int(self.num_types.item())}"


def rank_windows_by_info_gain(head, windows: List[torch.Tensor]) -> List[int]:
    """
    Info-gain data selection: rank candidate windows by predicted variance
    reduction (h^T P h). head must have .P [d, d] and .norm.
    windows: list of h tensors [B, T, d] or [N, d]
    Returns indices sorted descending by mean information gain.
    """
    scores = []
    for w in windows:
        var = head.uncertainty(w)  # [...,]
        scores.append(var.mean().item())
    # Higher variance = more informative
    return sorted(range(len(windows)), key=lambda i: scores[i], reverse=True)
