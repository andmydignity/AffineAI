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


def all_reduce_avg(tensor: torch.Tensor) -> torch.Tensor:
    """
    In-place all-reduce AVG across ranks.
    """
    if is_distributed() and tensor is not None:
        try:
            if hasattr(dist.ReduceOp, "AVG"):
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            else:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor.div_(get_world_size())
        except Exception:
            pass
    return tensor


def broadcast_parameters(model: torch.nn.Module, root: int = 0) -> None:
    """
    Broadcast model parameters and buffers from root rank to all ranks.
    Ensures identical initialization across multi-GPU processes.
    """
    if not (is_distributed() and get_world_size() > 1):
        return
    for p in model.parameters():
        try:
            dist.broadcast(p.data, src=root)
        except Exception:
            pass
    for b in model.buffers():
        try:
            dist.broadcast(b.data, src=root)
        except Exception:
            pass


def find_free_port() -> int:
    """Finds an available TCP port on localhost for multi-process rendezvous."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return int(port)


def parse_devices(devices: Optional[object] = None) -> list:
    """
    Normalizes a device specification into a list of CUDA device integer IDs.
    Supports:
        - None / 'auto' / 'all': all available CUDA GPUs
        - int: count of GPUs (e.g. 2 -> [0, 1])
        - list / tuple of ints: [0, 1]
        - str: '0,1', 'cuda:0,cuda:1', 'cuda:0', '0'
        - 'cpu': empty list []
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return []

    n_gpus = torch.cuda.device_count()

    if devices is None:
        return list(range(n_gpus))

    if isinstance(devices, str):
        dev_str = devices.strip().lower()
        if dev_str in ("auto", "all"):
            return list(range(n_gpus))
        if dev_str == "cpu":
            return []
        parts = [p.strip() for p in dev_str.split(",") if p.strip()]
        out = []
        for p in parts:
            p_clean = p.replace("cuda:", "").strip()
            if p_clean.isdigit():
                d_id = int(p_clean)
                if d_id < n_gpus:
                    out.append(d_id)
        return out if out else [0]

    if isinstance(devices, int):
        count = max(1, min(devices, n_gpus))
        return list(range(count))

    if isinstance(devices, (list, tuple)):
        out = []
        for d in devices:
            if isinstance(d, int) and d < n_gpus:
                out.append(d)
            elif isinstance(d, str):
                d_clean = d.replace("cuda:", "").strip()
                if d_clean.isdigit() and int(d_clean) < n_gpus:
                    out.append(int(d_clean))
        return out if out else [0]

    return [0]


def _worker_wrapper(
    rank: int,
    world_size: int,
    devices: list,
    master_port: int,
    worker_fn: object,
    results_dict: object,
    args: tuple,
    kwargs: dict,
) -> None:
    """Internal process target for torch.multiprocessing.spawn."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    local_gpu = devices[rank] if (devices and rank < len(devices)) else rank
    os.environ["LOCAL_RANK"] = str(local_gpu)

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.set_device(local_gpu)

    try:
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=world_size,
            rank=rank,
        )
    except Exception:
        if torch.cuda.is_available():
            dist.init_process_group(
                backend="gloo",
                world_size=world_size,
                rank=rank,
            )

    try:
        res = worker_fn(*args, **kwargs)
        if rank == 0 and results_dict is not None:
            results_dict["result"] = res
    finally:
        cleanup_distributed()


def spawn_multiprocess_training(
    worker_fn: object,
    devices: list,
    *args,
    **kwargs,
) -> object:
    """
    Natively launches multi-process training across the specified devices
    using torch.multiprocessing.spawn. Returns result from rank 0.
    """
    world_size = len(devices)
    if world_size <= 1:
        return worker_fn(*args, **kwargs)

    master_port = find_free_port()
    import torch.multiprocessing as mp

    manager = mp.Manager()
    results_dict = manager.dict()

    mp.spawn(
        _worker_wrapper,
        args=(world_size, devices, master_port, worker_fn, results_dict, args, kwargs),
        nprocs=world_size,
        join=True,
    )

    return results_dict.get("result", None)

