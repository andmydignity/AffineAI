"""
Distributed helpers for multi-GPU DDP on Kaggle 2xT4.

Provides a thin wrapper around torch.distributed that keeps single-GPU
fallback when not in distributed mode. Auto-detects WORLD_SIZE/RANK env
vars set by torchrun / Kaggle DDP launchers.

Kaggle 2xT4 uses 2 shards: train_data[rank::world_size] for raw byte
streams. For PaddedDataLoader/HFStreamDataLoader the loader is sharded
by rank or via DistributedSampler when possible.
"""

import os
from typing import Optional

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """Return True if torch.distributed is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Global rank (0 when not distributed)."""
    if is_distributed():
        return dist.get_rank()
    # fallback to env when not yet initialized (pre-init call)
    try:
        return int(os.environ.get("RANK", "0"))
    except Exception:
        return 0


def get_world_size() -> int:
    """World size (1 when not distributed)."""
    if is_distributed():
        return dist.get_world_size()
    ws = os.environ.get("WORLD_SIZE", "")
    if ws:
        try:
            return int(ws)
        except Exception:
            pass
    return 1


def is_main_process() -> bool:
    """True only on rank 0 (or when not distributed)."""
    return get_rank() == 0


def _get_local_rank() -> int:
    """Local rank from env or derived from global rank."""
    if "LOCAL_RANK" in os.environ:
        try:
            return int(os.environ["LOCAL_RANK"])
        except Exception:
            pass
    # fall back to global rank modulo device count
    try:
        has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
        n = torch.cuda.device_count() if has_cuda else 1
        return get_rank() % max(1, n)
    except Exception:
        return get_rank()


def get_local_device() -> str:
    """Device string for this rank: cuda:{local_rank} or cpu/cuda."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return f"cuda:{_get_local_rank()}"
    # single-GPU fallback or CPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def setup_distributed(backend: str = "nccl") -> None:
    """
    Initialize torch.distributed process group if in a distributed launch.
    Auto-detects via WORLD_SIZE/RANK env vars (torchrun / Kaggle). Safe to
    call when not distributed — becomes a no-op. Sets cuda device to
    local_rank when CUDA is available. Falls back to gloo if nccl fails
    or CUDA is unavailable.
    """
    if is_distributed():
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            try:
                torch.cuda.set_device(_get_local_rank())
            except Exception:
                pass
        return

    world_size = get_world_size()
    has_rank_env = ("RANK" in os.environ) or ("WORLD_SIZE" in os.environ)
    if world_size <= 1 and not has_rank_env:
        return

    if backend.lower() == "nccl" and not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        backend = "gloo"

    try:
        dist.init_process_group(backend=backend)
    except Exception:
        # retry with gloo if nccl failed (e.g. CPU-only or missing nccl)
        if backend.lower() != "gloo":
            try:
                dist.init_process_group(backend="gloo")
            except Exception:
                return
        else:
            return

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            torch.cuda.set_device(_get_local_rank())
        except Exception:
            pass


def cleanup_distributed() -> None:
    """Destroy process group if initialized."""
    if is_distributed():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def barrier() -> None:
    """Barrier across all ranks (no-op when not distributed)."""
    if is_distributed():
        try:
            dist.barrier()
        except Exception:
            pass


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    In-place all-reduce SUM across ranks.

    Args:
        tensor: tensor to reduce (modified in place). No-op when not
                distributed. Returns the same tensor for chaining.
    """
    if is_distributed() and tensor is not None:
        try:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        except Exception:
            pass
    return tensor
