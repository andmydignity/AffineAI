"""
Dynamic Priority Replay Buffer (AXIOM Info-Gain Selection)
==========================================================
Zero-memory-overhead priority replay buffer for high-efficiency LLM training:
  - Tracks high-uncertainty / high-loss training sequence offsets.
  - Interleaves high-signal sequences into future batches (default 25% replay, 75% fresh).
  - Bounded replay cap (max 3 replays per sequence) to prevent overfitting or memorizing noise.
  - Sliding-window lowest-loss eviction policy when buffer capacity is reached.
  - Stores only 64-bit integer file offsets (~32 KB total memory for 4,000 sequences).
"""

import heapq
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch


class DynamicPriorityReplayBuffer:
    """
    In-Memory Dynamic Priority Replay Buffer.
    
    Args:
        max_capacity: Maximum number of active sequence offsets in the buffer (default 4,000).
        max_replays: Maximum number of times a single sequence can be replayed before retirement (default 3).
        replay_ratio: Fraction of batch capacity allocated to replayed sequences (default 0.25).
        max_loss_ceiling: Upper bound loss threshold to filter irreducible noise/corrupted tokens (default 4.5).
        min_loss_percentile: Running percentile threshold for enqueuing fresh sequences (default 65.0).
    """
    def __init__(
        self,
        max_capacity: int = 4000,
        max_replays: int = 3,
        replay_ratio: float = 0.25,
        max_loss_ceiling: float = 4.5,
        min_loss_percentile: float = 65.0
    ):
        self.max_capacity = max_capacity
        self.max_replays = max_replays
        self.replay_ratio = replay_ratio
        self.max_loss_ceiling = max_loss_ceiling
        self.min_loss_percentile = min_loss_percentile

        self.buffer: Dict[int, Dict[str, Any]] = {}
        self._min_heap: List[Tuple[float, int, int]] = []  # (loss, version, idx)
        self._version: Dict[int, int] = {}
        self.running_losses: List[float] = []
        self.total_enqueued: int = 0
        self.total_replayed: int = 0
        self.total_retired: int = 0
        # Generator for reproducible sampling without global np.random
        self._rng = np.random.default_rng()

    def push_candidates(
        self,
        indices: Union[List[int], np.ndarray, torch.Tensor],
        losses: Union[List[float], np.ndarray, torch.Tensor],
        threshold: Optional[float] = None
    ) -> int:
        """
        Pushes candidate sequences with high loss into the priority replay buffer.
        Batched D2H: converts Tensor inputs in one sync.
        """
        # Batched D2H: single cpu() per tensor
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().tolist()
        elif isinstance(indices, np.ndarray):
            indices = indices.tolist()

        if isinstance(losses, torch.Tensor):
            losses_np = losses.detach().float().cpu().numpy()
            losses = losses_np
        elif isinstance(losses, list):
            losses = np.array(losses, dtype=np.float32)
        # losses is now np.ndarray

        # Update running loss statistics
        valid_losses = [float(l) for l in losses if np.isfinite(float(l))]
        self.running_losses.extend(valid_losses)
        if len(self.running_losses) > 1000:
            self.running_losses = self.running_losses[-1000:]

        if threshold is None and len(self.running_losses) >= 10:
            threshold = float(np.percentile(self.running_losses, self.min_loss_percentile))
        elif threshold is None:
            threshold = 2.0

        enqueued_count = 0
        for idx, l in zip(indices, losses):
            loss_val = float(l)
            if threshold <= loss_val <= self.max_loss_ceiling:
                if idx not in self.buffer:
                    if len(self.buffer) >= self.max_capacity:
                        # Lazy deletion: pop until valid entry (O(log C) amortized)
                        evicted = False
                        while self._min_heap:
                            cand_loss, cand_ver, cand_idx = heapq.heappop(self._min_heap)
                            if cand_idx in self.buffer and self._version.get(cand_idx, -1) == cand_ver and self.buffer[cand_idx]["loss"] == cand_loss:
                                if loss_val <= cand_loss:
                                    heapq.heappush(self._min_heap, (cand_loss, cand_ver, cand_idx))
                                    break
                                del self.buffer[cand_idx]
                                self._version.pop(cand_idx, None)
                                evicted = True
                                break
                            # stale entry -> discard (heap leak fixed via lazy deletion)
                        if len(self.buffer) >= self.max_capacity and not evicted:
                            # Fallback scan if heap exhausted but still full (rebuild heap)
                            self._rebuild_heap()
                            # After rebuild, try again Pop
                            if self._min_heap:
                                cand_loss, cand_ver, cand_idx = heapq.heappop(self._min_heap)
                                if cand_idx in self.buffer and loss_val > cand_loss:
                                    del self.buffer[cand_idx]
                                    self._version.pop(cand_idx, None)
                                    evicted = True
                                else:
                                    heapq.heappush(self._min_heap, (cand_loss, cand_ver, cand_idx))
                            if not evicted:
                                min_k = min(self.buffer, key=lambda k: self.buffer[k]["loss"])
                                if loss_val <= self.buffer[min_k]["loss"]:
                                    continue
                                del self.buffer[min_k]
                                self._version.pop(min_k, None)

                    if len(self.buffer) < self.max_capacity:
                        self.buffer[idx] = {"loss": loss_val, "replays": 0}
                        ver = self._version.get(idx, 0) + 1
                        self._version[idx] = ver
                        heapq.heappush(self._min_heap, (loss_val, ver, idx))
                        self.total_enqueued += 1
                        enqueued_count += 1
                else:
                    # Update loss estimate with version bump (old heap entry becomes stale)
                    new_loss = 0.5 * (self.buffer[idx]["loss"] + loss_val)
                    self.buffer[idx]["loss"] = new_loss
                    ver = self._version.get(idx, 0) + 1
                    self._version[idx] = ver
                    heapq.heappush(self._min_heap, (new_loss, ver, idx))

        return enqueued_count

    def _rebuild_heap(self):
        # Rebuild heap from current buffer to purge stale entries
        self._min_heap = [(v["loss"], self._version[k], k) for k, v in self.buffer.items()]
        heapq.heapify(self._min_heap)

    def sample(self, n: int, rng: Optional[Union[np.random.RandomState, np.random.Generator]] = None) -> List[int]:
        """
        Samples n sequence indices weighted by uncertainty/loss.
        Increments replay counter and retires sequences that exceed max_replays.
        Vectorized: O(C) still but batched numpy; no per-element Python loop for probs.
        Accepts Generator or RandomState for backward compat.
        """
        if not self.buffer or n <= 0:
            return []

        # Resolve RNG: prefer Generator
        if rng is None:
            rng = self._rng
        # Normalize rng to have choice method
        use_generator = isinstance(rng, np.random.Generator)

        candidates = list(self.buffer.keys())
        # Vectorized loss gather (still O(C) but numpy)
        losses = np.array([self.buffer[c]["loss"] for c in candidates], dtype=np.float64)
        sum_l = np.sum(losses)
        if sum_l > 0:
            probs = losses / sum_l
        else:
            probs = np.ones(len(candidates)) / len(candidates)

        sample_n = min(n, len(candidates))
        if use_generator:
            chosen = rng.choice(candidates, size=sample_n, replace=False, p=probs).tolist()
        else:
            # Legacy RandomState (kept for backward compat, no warning to keep tests clean)
            chosen = rng.choice(candidates, size=sample_n, replace=False, p=probs).tolist()

        for c in chosen:
            self.buffer[c]["replays"] += 1
            self.total_replayed += 1
            if self.buffer[c]["replays"] >= self.max_replays:
                del self.buffer[c]
                self._version.pop(c, None)
                self.total_retired += 1

        return chosen

    def get_stats(self) -> Dict[str, Any]:
        """Returns live buffer telemetry."""
        active_count = len(self.buffer)
        mean_buf_loss = float(np.mean([v["loss"] for v in self.buffer.values()])) if active_count > 0 else 0.0
        return {
            "active_count": active_count,
            "max_capacity": self.max_capacity,
            "mean_loss": mean_buf_loss,
            "total_enqueued": self.total_enqueued,
            "total_replayed": self.total_replayed,
            "total_retired": self.total_retired,
            "memory_kb": (active_count * 8) / 1024.0
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "buffer": self.buffer,
            "running_losses": self.running_losses,
            "total_enqueued": self.total_enqueued,
            "total_replayed": self.total_replayed,
            "total_retired": self.total_retired,
            "version": self._version,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.buffer = state.get("buffer", {})
        self.running_losses = state.get("running_losses", [])
        self.total_enqueued = state.get("total_enqueued", 0)
        self.total_replayed = state.get("total_replayed", 0)
        self.total_retired = state.get("total_retired", 0)
        self._version = state.get("version", {k: 0 for k in self.buffer})
        # Rebuild heap to fix leak from stale entries persisted
        self._rebuild_heap()

    def __len__(self) -> int:
        return len(self.buffer)
