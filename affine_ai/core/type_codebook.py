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
        flat = latents.reshape(-1, D)
        dists = torch.cdist(flat.float(), self.means[:active].float())
        ids = dists.argmin(dim=-1).view(B, M)
        enrich = self.embedding(ids) * self.scale
        return latents + enrich, ids

    @torch.no_grad()
    def update(self, latents: torch.Tensor):
        flat = latents.detach().reshape(-1, self.dim)
        active = int(self.num_types.item())
        if active == 0:
            return
        dists = torch.cdist(flat.float(), self.means[:active].float())
        min_dist, nearest = dists.min(dim=-1)
        birth_mask = (min_dist > self.birth_threshold) & (active < self.max_types)
        if birth_mask.any():
            birth_idx = torch.where(birth_mask)[0]
            n_birth = min(int(birth_mask.sum().item()), self.max_types - active)
            for j in range(n_birth):
                idx = int(birth_idx[j].item())
                self.means[active] = flat[idx]
                self.counts[active] = 1.0
                active += 1
            self.num_types.fill_(active)
            if n_birth > 0 and n_birth < flat.shape[0]:
                dists = torch.cdist(flat.float(), self.means[:active].float())
                min_dist, nearest = dists.min(dim=-1)
                birth_mask = torch.zeros_like(birth_mask)
        mask = ~birth_mask
        if mask.any():
            flat_rem = flat[mask]
            nearest_rem = nearest[mask]
            for k in range(active):
                sel = nearest_rem == k
                if not sel.any():
                    continue
                cnt_add = int(sel.sum().item())
                sum_k = flat_rem[sel].sum(dim=0)
                old_c = float(self.counts[k].item())
                new_c = old_c + cnt_add
                if old_c == 0:
                    self.means[k] = sum_k / cnt_add
                else:
                    self.means[k] = (self.means[k] * old_c + sum_k) / new_c
                self.counts[k] = new_c

    @torch.no_grad()
    def bmr_merge(self) -> int:
        """BMR-style merge: merge closest pair if distance < merge_threshold."""
        active = int(self.num_types.item())
        if active <= 1:
            return 0
        dists = torch.cdist(self.means[:active].float(), self.means[:active].float())
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
        if b < active - 1:
            self.means[b : active - 1] = self.means[b + 1 : active].clone()
            self.counts[b : active - 1] = self.counts[b + 1 : active].clone()
        self.means[active - 1].zero_()
        self.counts[active - 1] = 0
        self.num_types.fill_(active - 1)
        with torch.no_grad():
            w = self.embedding.weight
            w[a] = (w[a] * ca + w[b] * cb) / (ca + cb) if (ca + cb) > 0 else w[a]
            if b < active - 1:
                w[b : active - 1] = w[b + 1 : active].clone()
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
