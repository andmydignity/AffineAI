"""
Unified High-Performance Training Pipeline for ASDAG LLMs
==========================================================
Encapsulates all ASDAG optimizations out of the box:
- Pure MatMul-free channel mixing with 1:8 / 1:16 Structured Sparsity
- Power-of-two gating & Shift4 logarithmic activations (2^-p)
- Memory-efficient fused cross-entropy loss
- Linear Warmup + Bounded Cosine Annealing Learning Rate Scheduler
- CPU physical-core optimization and SMT thread tuning
- Optional PyTorch 2.x torch.compile() support
- Gradient Clipping for Numerical Stability
- Hardware-aligned batch dispatch
"""

import gc
import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union, Tuple
import numpy as np

from affine_ai.models.language_model import ASDAGLanguageModel
from affine_ai.core.loss import ChunkedCrossEntropyLoss

# Distributed helpers (safe on single-GPU; no-ops when not in DDP)
try:
    from affine_ai.training.distributed import (
        setup_distributed,
        cleanup_distributed,
        is_distributed,
        get_rank,
        get_world_size,
        is_main_process,
        get_local_device,
        barrier,
        all_reduce_sum,
    )
except Exception:  # pragma: no cover
    # Fallback stubs if distributed module missing
    def setup_distributed(backend="nccl"):  # type: ignore
        return None

    def cleanup_distributed():  # type: ignore
        return None

    def is_distributed():  # type: ignore
        return False

    def get_rank():  # type: ignore
        return 0

    def get_world_size():  # type: ignore
        return 1

    def is_main_process():  # type: ignore
        return True

    def get_local_device():  # type: ignore
        return "cpu"

    def barrier():  # type: ignore
        return None

    def all_reduce_sum(tensor):  # type: ignore
        return tensor


def _is_turing() -> bool:
    try:
        from affine_ai.kernels import _IS_TURING as _T  # type: ignore

        return bool(_T)
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return (7, 5) <= tuple(cap) < (8, 0)
    except Exception:
        pass
    return False


def get_turing_dtype(dtype=None):
    try:
        if torch.cuda.is_available():
            cap = tuple(torch.cuda.get_device_capability())
            if cap < (8, 0):
                return torch.float16
    except Exception:
        pass
    return dtype


def suggest_batch_size(device: Optional[str] = None) -> int:
    """Throughput-optimal batch size for training on this machine.

    On GPU: B96 maximizes throughput (~212k-250k tok/s) while keeping
    peak VRAM comfortably within budget (~1.7 GB, avoiding OOM seen at B256).
    On CPU: sweet spot is ~1.5x physical cores (B12 on 8 cores;
    mild oversubscription feeds the memory-bound phases). Scales
    linearly: B24 on 16 cores, etc.
    """
    if device is not None and "cuda" in str(device):
        return 96
    if device is None and torch.cuda.is_available():
        return 96
    return max(1, (3 * get_cpu_physical_cores() + 1) // 2)


def get_cpu_physical_cores() -> int:
    """Returns number of physical CPU cores (bypassing SMT hyperthreading).

    Detects real SMT pairing via sysfs; falls back to the total//2 heuristic
    only when topology is unreadable (non-Linux). On non-SMT machines
    (e.g. ARM) this returns the full core count instead of halving it.
    """
    count = os.cpu_count() or 1
    try:
        seen = set()
        for cpu in range(count):
            with open(
                f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            ) as f:
                first = f.read().strip().split(",")[0].split("-")[0]
                if first not in seen:
                    seen.add(first)
        if seen:
            return len(seen)
    except OSError:
        pass
    return count // 2 if count >= 8 else max(1, count)


class ASDAGTrainer:
    """
    High-Performance Trainer for ASDAG Language Models.
    Optimized for both GPU and CPU execution.
    """

    def __init__(
        self,
        model: Any,
        train_data: Any,
        val_data: Optional[Any] = None,
        batch_size: Optional[int] = None,
        seq_len: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        max_steps: int = 1000,
        warmup_steps: int = 100,
        eval_interval: int = 100,
        eval_iters: int = 20,
        grad_clip: float = 1.0,
        device: Optional[str] = None,
        use_quantized_gates: bool = True,
        use_shift4_act: bool = True,
        compile_model: bool = False,
        num_threads: Optional[int] = None,
        use_backpressure: bool = False,
        use_lpc: bool = True,
        use_muon: bool = True,
        muon_lr: float = 0.03,
        use_priority_replay: bool = False,
        replay_ratio: float = 0.25,
        replay_buffer_capacity: int = 4000,
        replay_max_replays: int = 3,
        use_cuda_graph: bool = True,
        pad_id: int = 0,
        ignore_index: int = -100,
        context_window: Optional[int] = None,
        inject_eos: bool = True,
        eos_token: Union[str, bytes, int] = "<|endoftext|>",
        as_stream: bool = False,
        use_mtp: Optional[bool] = None,
        num_mtp_heads: Optional[int] = None,
        mtp_lambda: Optional[float] = None,
        channel_mixer: Optional[str] = "asdag_tree",
        time_mixer: Optional[str] = None,
        swa_every_n: Optional[int] = None,
        swa_window: Optional[int] = None,
        distributed: Optional[bool] = None,
    ):
        self.pad_id = pad_id
        self.ignore_index = ignore_index
        _ws_env = os.environ.get("WORLD_SIZE", "")
        _rank_env = os.environ.get("RANK", "")
        _auto_ws = 1
        if _ws_env:
            try:
                _auto_ws = int(_ws_env)
            except Exception:
                _auto_ws = 1
        if distributed is None:
            if _auto_ws > 1 or _rank_env != "":
                distributed = True
            elif is_distributed():
                distributed = True
            else:
                distributed = False
        self.distributed = bool(distributed)
        self.is_distributed = self.distributed
        if self.distributed:
            try:
                setup_distributed(
                    backend="nccl" if torch.cuda.is_available() else "gloo"
                )
            except Exception:
                pass
            self.rank = get_rank()
            self.world_size = get_world_size()
            self.is_main = is_main_process()
            try:
                _lr_env = os.environ.get("LOCAL_RANK", "")
                self.local_rank = (
                    int(_lr_env)
                    if _lr_env != ""
                    else (
                        self.rank
                        % max(
                            1,
                            torch.cuda.device_count()
                            if torch.cuda.is_available()
                            else 1,
                        )
                    )
                )
            except Exception:
                self.local_rank = self.rank
            _has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
            if _has_cuda:
                _eff = self.local_rank % torch.cuda.device_count()
                try:
                    torch.cuda.set_device(_eff)
                except Exception:
                    pass
                self.device = f"cuda:{_eff}"
            else:
                self.device = device or "cpu"
        else:
            self.rank = 0
            self.world_size = 1
            self.is_main = True
            self.local_rank = 0
            _has_cuda_fallback = (
                torch.cuda.is_available() and torch.cuda.device_count() > 0
            )
            self.device = device or ("cuda" if _has_cuda_fallback else "cpu")
        self.train_sampler = None
        self.use_cuda_graph = (
            ("cuda" in str(self.device))
            if use_cuda_graph is None
            else bool(use_cuda_graph)
        )
        self.use_backpressure = use_backpressure
        self.use_lpc = use_lpc
        self.use_muon = use_muon
        self.muon_lr = muon_lr
        self.use_priority_replay = use_priority_replay
        self.replay_ratio = replay_ratio
        self.inject_eos = inject_eos
        self.eos_token = eos_token
        self.as_stream = as_stream
        self.use_mtp = use_mtp
        self.num_mtp_heads = num_mtp_heads
        self.mtp_lambda = mtp_lambda
        self.channel_mixer = channel_mixer
        self.time_mixer = time_mixer
        self._last_train_fresh_ix = []

        if self.use_priority_replay:
            from affine_ai.core.priority_replay import DynamicPriorityReplayBuffer

            self.replay_buffer = DynamicPriorityReplayBuffer(
                max_capacity=replay_buffer_capacity,
                max_replays=replay_max_replays,
                replay_ratio=replay_ratio,
            )
        else:
            self.replay_buffer = None

        # CPU Threading Optimization
        if self.device == "cpu":
            target_threads = num_threads or get_cpu_physical_cores()
            try:
                torch.set_num_threads(target_threads)
                torch.set_num_interop_threads(1)
            except Exception:
                pass

        self.model = model.to(self.device)
        if getattr(self, "is_distributed", False) and self.world_size > 1:
            try:
                if (
                    torch.cuda.is_available()
                    and torch.cuda.device_count() > 0
                    and "cuda" in str(self.device)
                ):
                    _eff_ddp = self.local_rank % torch.cuda.device_count()
                    self.model = torch.nn.parallel.DistributedDataParallel(
                        self.model,
                        device_ids=[_eff_ddp],
                        output_device=_eff_ddp,
                        find_unused_parameters=False,
                    )
                else:
                    self.model = torch.nn.parallel.DistributedDataParallel(
                        self.model,
                        find_unused_parameters=False,
                    )
            except Exception:
                pass

        # Context window / max sequence length configuration
        if context_window is not None:
            self.seq_len = int(context_window)
        else:
            self.seq_len = int(seq_len)
        self.context_window = self.seq_len
        if hasattr(self.model, "context_window"):
            self.model.context_window = self.context_window
        if hasattr(self.model, "config") and hasattr(
            self.model.config, "context_window"
        ):
            self.model.config.context_window = self.context_window
        if hasattr(self.model, "config") and hasattr(self.model.config, "max_seq_len"):
            self.model.config.max_seq_len = self.context_window

        # Configure architectural mixers if requested
        if channel_mixer is not None:
            if hasattr(self.model, "channel_mixer_type"):
                self.model.channel_mixer_type = channel_mixer
            if hasattr(self.model, "config") and hasattr(
                self.model.config, "channel_mixer_type"
            ):
                self.model.config.channel_mixer_type = channel_mixer

        if time_mixer is not None:
            if hasattr(self.model, "time_mixer_rule"):
                self.model.time_mixer_rule = time_mixer
            if hasattr(self.model, "config") and hasattr(
                self.model.config, "time_mixer_rule"
            ):
                self.model.config.time_mixer_rule = time_mixer

        # Configure Multi-Token Prediction (MTP) if requested
        from affine_ai.models.hybrid import TorosHybridLanguageModel

        _unwrap = (
            self.model.module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model
        )
        if swa_every_n is not None and swa_every_n:
            if hasattr(_unwrap, "config"):
                if hasattr(_unwrap.config, "swa_every_n"):
                    _unwrap.config.swa_every_n = swa_every_n
                if swa_window is not None and hasattr(_unwrap.config, "swa_window"):
                    _unwrap.config.swa_window = swa_window
            from affine_ai.core.swa import interleave_swa
            interleave_swa(_unwrap, every_n=swa_every_n,
                           window=swa_window or 256)
        if isinstance(_unwrap, TorosHybridLanguageModel):
            self.hybrid = _unwrap
        elif getattr(_unwrap, "hybrid", None) is not None:
            self.hybrid = _unwrap.hybrid
        elif getattr(self.model, "hybrid", None) is not None:
            self.hybrid = self.model.hybrid
        else:
            self.hybrid = None

        if self.use_mtp is not None:
            if self.hybrid is not None:
                if self.use_mtp:
                    self.hybrid.enable_mtp(
                        num_mtp_heads=self.num_mtp_heads
                        if self.num_mtp_heads is not None
                        else 2,
                        mtp_lambda=self.mtp_lambda
                        if self.mtp_lambda is not None
                        else 0.3,
                    )
                else:
                    self.hybrid.disable_mtp()
            elif hasattr(self.model, "use_mtp"):
                self.model.use_mtp = bool(self.use_mtp)
                if self.num_mtp_heads is not None and hasattr(
                    self.model, "num_mtp_heads"
                ):
                    self.model.num_mtp_heads = self.num_mtp_heads
                if self.mtp_lambda is not None and hasattr(self.model, "mtp_lambda"):
                    self.model.mtp_lambda = self.mtp_lambda

        if self.device == "cpu":
            try:
                from affine_ai.core.numa import node_count, interleave_model_weights

                if node_count() > 1:
                    interleave_model_weights(self.model)
            except Exception:
                pass
        if (
            compile_model
            and hasattr(torch, "compile")
            and not getattr(self, "is_distributed", False)
        ):
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = suggest_batch_size(self.device)
        self.lr = lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.grad_clip = grad_clip
        self.use_quantized_gates = use_quantized_gates
        self.use_shift4_act = use_shift4_act

        from affine_ai.data.dataloader import PaddedDataLoader, HFStreamDataLoader

        # Ingest train data
        if isinstance(train_data, (PaddedDataLoader, HFStreamDataLoader)):
            self.train_loader = train_data
            self.train_data = None
        elif isinstance(train_data, (list, tuple)) and not isinstance(
            train_data, (torch.Tensor, np.ndarray)
        ):
            self.train_loader = PaddedDataLoader(
                train_data,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                pad_id=self.pad_id,
                ignore_index=self.ignore_index,
                device=self.device,
                shuffle=True,
                pad_remainder=True,
                inject_eos=self.inject_eos,
                eos_token=self.eos_token,
                as_stream=self.as_stream,
            )
            self.train_data = None
        else:
            self.train_loader = None
            if isinstance(train_data, np.ndarray):
                self.train_data = torch.from_numpy(train_data.astype(np.int64))
            elif isinstance(train_data, torch.Tensor):
                self.train_data = train_data.to(torch.long)
            else:
                self.train_data = train_data

        # Ingest val data
        if isinstance(val_data, (PaddedDataLoader, HFStreamDataLoader)):
            self.val_loader = val_data
            self.val_data = None
        elif isinstance(val_data, (list, tuple)) and not isinstance(
            val_data, (torch.Tensor, np.ndarray)
        ):
            self.val_loader = PaddedDataLoader(
                val_data,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                pad_id=self.pad_id,
                ignore_index=self.ignore_index,
                device=self.device,
                shuffle=False,
                pad_remainder=True,
                inject_eos=self.inject_eos,
                eos_token=self.eos_token,
                as_stream=self.as_stream,
            )
            self.val_data = None
        elif val_data is not None:
            self.val_loader = None
            if isinstance(val_data, np.ndarray):
                self.val_data = torch.from_numpy(val_data.astype(np.int64))
            elif isinstance(val_data, torch.Tensor):
                self.val_data = val_data.to(torch.long)
            else:
                self.val_data = val_data
        else:
            self.val_loader = None
            self.val_data = None

        if getattr(self, "is_distributed", False) and self.world_size > 1:
            try:
                _rank = self.rank
                _ws = self.world_size
                if self.train_data is not None:
                    if isinstance(self.train_data, torch.Tensor):
                        self.train_data = self.train_data[_rank::_ws].contiguous()
                    elif isinstance(self.train_data, np.ndarray):
                        self.train_data = self.train_data[_rank::_ws].copy()
                    elif hasattr(self.train_data, "__getitem__"):
                        try:
                            self.train_data = self.train_data[_rank::_ws]
                        except Exception:
                            pass
                elif self.train_loader is not None:
                    try:
                        from torch.utils.data.distributed import DistributedSampler

                        if hasattr(self.train_loader, "data"):
                            _d = self.train_loader.data
                            if isinstance(_d, np.ndarray):
                                if _d.ndim == 1:
                                    _sharded = _d[_rank::_ws].copy()
                                    self.train_loader.data = _sharded
                                    if hasattr(self.train_loader, "stream_len"):
                                        self.train_loader.stream_len = len(_sharded)
                            elif isinstance(_d, torch.Tensor):
                                _sharded = _d[_rank::_ws].contiguous()
                                self.train_loader.data = _sharded
                                if hasattr(self.train_loader, "stream_len"):
                                    self.train_loader.stream_len = len(_sharded)
                            elif isinstance(_d, list):
                                self.train_loader.data = _d[_rank::_ws]
                        if (
                            hasattr(self.train_loader, "data")
                            and isinstance(self.train_loader.data, list)
                            and not getattr(self.train_loader, "is_stream", False)
                        ):
                            try:
                                _ds_len = len(self.train_loader.data)

                                class _IdxDataset(torch.utils.data.Dataset):
                                    def __len__(self_inner):
                                        return _ds_len

                                    def __getitem__(self_inner, idx):
                                        return idx

                                self.train_sampler = DistributedSampler(
                                    _IdxDataset(),
                                    num_replicas=_ws,
                                    rank=_rank,
                                    shuffle=getattr(self.train_loader, "shuffle", True),
                                )
                            except Exception:
                                self.train_sampler = None
                    except Exception:
                        pass
            except Exception:
                pass

        # Allocate pinned staging buffers on CPU for zero-copy DMA to CUDA
        if "cuda" in str(self.device):
            try:
                self._pinned_buf_x = torch.empty(
                    (self.batch_size, self.seq_len), dtype=torch.long, pin_memory=True
                )
                self._pinned_buf_y = torch.empty(
                    (self.batch_size, self.seq_len), dtype=torch.long, pin_memory=True
                )
            except Exception:
                self._pinned_buf_x = None
                self._pinned_buf_y = None
        else:
            self._pinned_buf_x = None
            self._pinned_buf_y = None

        # Fused / standard AdamW
        fused = self.device == "cuda" and hasattr(optim.AdamW, "_fused")
        from affine_ai.models.hybrid import TorosHybridLanguageModel

        _unwrap2 = (
            self.model.module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model
        )
        if isinstance(_unwrap2, TorosHybridLanguageModel):
            self.hybrid = _unwrap2
        elif getattr(_unwrap2, "hybrid", None) is not None:
            self.hybrid = _unwrap2.hybrid
        elif getattr(self.model, "hybrid", None) is not None:
            self.hybrid = self.model.hybrid
        else:
            self.hybrid = None

        if self.hybrid is not None:
            hybrid = self.hybrid
            can_lpc = (
                self.use_lpc
                and hasattr(hybrid, "enable_lpc")
                and hasattr(hybrid, "get_default_lpc_optimizers")
            )
            has_legacy_local = (
                self.use_lpc and getattr(hybrid, "local_heads", None) is not None
            )
            if can_lpc:
                try:
                    try:
                        hybrid.enable_lpc(device=self.device)
                    except TypeError:
                        hybrid.enable_lpc()
                    has_local_lpc = getattr(hybrid, "local_heads", None) is not None
                except Exception:
                    has_local_lpc = False
                if has_local_lpc:
                    lpc_kwargs = {}
                    if hasattr(hybrid.get_default_lpc_optimizers, "__code__"):
                        import inspect

                        sig = inspect.signature(hybrid.get_default_lpc_optimizers)
                    if "capturable" in sig.parameters:
                        lpc_kwargs["capturable"] = self.use_cuda_graph
                    try:
                        self.hybrid_optimizers = hybrid.get_default_lpc_optimizers(
                            lr=lr,
                            weight_decay=weight_decay,
                            use_muon=self.use_muon,
                            **lpc_kwargs,
                        )
                    except TypeError:
                        self.hybrid_optimizers = hybrid.get_default_lpc_optimizers(
                            lr=lr,
                            weight_decay=weight_decay,
                            use_muon=self.use_muon,
                            muon_lr=self.muon_lr,
                        )
                    self.lpc_model = None
                    self.lpc_optimizers = None
                    self.optimizer = None
                else:
                    self.hybrid_optimizers = None
                    self.lpc_model = None
                    self.lpc_optimizers = None
                    adamw_kwargs = {
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "fused": fused,
                    }
                    if self.use_cuda_graph:
                        adamw_kwargs["capturable"] = True
                    self.optimizer = optim.AdamW(
                        self.model.parameters(), **adamw_kwargs
                    )
            elif has_legacy_local:
                self.hybrid_optimizers = hybrid.get_default_optimizers(
                    lr=lr,
                    weight_decay=weight_decay,
                    use_muon=self.use_muon,
                    muon_lr=self.muon_lr,
                )
                self.lpc_model = None
                self.lpc_optimizers = None
                self.optimizer = None
            else:
                self.hybrid_optimizers = None
                self.lpc_model = None
                self.lpc_optimizers = None
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    fused=fused,
                )
        elif self.use_lpc:
            from affine_ai.core.lpc import LocalPredictiveLanguageModel

            self.lpc_model = LocalPredictiveLanguageModel(self.model).to(self.device)
            self.lpc_optimizers = self.lpc_model.get_default_lpc_optimizers(
                lr=lr,
                weight_decay=weight_decay,
                use_muon=self.use_muon,
                muon_lr=self.muon_lr,
            )
            self.hybrid_optimizers = None
            self.optimizer = None
        else:
            self.lpc_model = None
            self.lpc_optimizers = None
            self.hybrid_optimizers = None
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay, fused=fused
            )

        self.loss_fn = nn.CrossEntropyLoss()

        self.scaler = None
        self.use_amp = False
        try:
            _model_dtype = getattr(getattr(self.model, "config", None), "dtype", None)
            _is_fp16 = _model_dtype == torch.float16
            if _is_turing() and get_turing_dtype(_model_dtype) == torch.float16:
                _is_fp16 = True
            if _is_fp16 and "cuda" in str(self.device) and torch.cuda.is_available():
                if hasattr(torch.amp, "GradScaler"):
                    self.scaler = torch.amp.GradScaler("cuda")
                else:
                    self.scaler = torch.cuda.amp.GradScaler()
                self.use_amp = True
                import warnings

                warnings.warn(
                    "Turing fp16 AMP enabled: using torch.amp.GradScaler for fp16 training on sm_75.",
                    stacklevel=2,
                )
            elif _is_fp16 and "cuda" in str(self.device):
                import warnings

                warnings.warn(
                    "FP16 training without CUDA GradScaler (CPU or non-CUDA device); scaler not created.",
                    stacklevel=2,
                )
        except Exception:
            self.scaler = None
            self.use_amp = False

    def get_batch(self, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        loader = self.train_loader if split == "train" else self.val_loader
        if loader is not None:
            if (
                split == "train"
                and getattr(self, "use_priority_replay", False)
                and getattr(self, "replay_buffer", None) is not None
                and getattr(loader, "stream_len", None) is not None
                and hasattr(loader, "get_batch_by_indices")
            ):
                max_units = (
                    (loader.stream_len - 1) // loader.seq_len
                    if loader.is_stream
                    else len(loader.data)
                )
                if max_units > 0:
                    n_replay = min(
                        int(self.batch_size * self.replay_ratio),
                        len(self.replay_buffer),
                    )
                    n_fresh = self.batch_size - n_replay
                    fresh_ix = torch.randint(0, max_units, (n_fresh,)).tolist()
                    replayed_ix = (
                        self.replay_buffer.sample(n_replay) if n_replay > 0 else []
                    )
                    self._last_train_fresh_ix = fresh_ix
                    all_ix = fresh_ix + replayed_ix
                    return loader.get_batch_by_indices(all_ix)

            iter_attr = f"_{split}_iter"
            it = getattr(self, iter_attr, None)
            if it is None:
                it = iter(loader)
                setattr(self, iter_attr, it)
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                setattr(self, iter_attr, it)
                x, y = next(it)
            return x, y

        data = self.train_data if split == "train" else self.val_data
        if data is None:
            raise ValueError(f"No {split} data available in trainer.")
        high = len(data) - self.seq_len - 1
        if high <= 0:
            raise ValueError(
                f"Dataset too small ({len(data)}) for seq_len {self.seq_len}"
            )
        if (
            split == "train"
            and getattr(self, "use_priority_replay", False)
            and getattr(self, "replay_buffer", None) is not None
        ):
            n_replay = min(
                int(self.batch_size * self.replay_ratio), len(self.replay_buffer)
            )
            n_fresh = self.batch_size - n_replay
            fresh_ix = torch.randint(0, high, (n_fresh,)).tolist()
            replayed_ix = self.replay_buffer.sample(n_replay) if n_replay > 0 else []
            self._last_train_fresh_ix = fresh_ix
            all_ix = fresh_ix + replayed_ix
            ix = torch.tensor(all_ix, dtype=torch.long)
        else:
            ix = torch.randint(0, high, (self.batch_size,))
            if split == "train":
                self._last_train_fresh_ix = ix.tolist()
        offsets = torch.arange(self.seq_len, device=ix.device)
        idx = ix.unsqueeze(1) + offsets.unsqueeze(0)
        idx_next = idx + 1

        if data.is_cuda:
            if idx.device != data.device:
                idx = idx.to(data.device, non_blocking=True)
                idx_next = idx_next.to(data.device, non_blocking=True)
            x = data[idx]
            y = data[idx_next]
        elif (
            "cuda" in str(self.device)
            and getattr(self, "_pinned_buf_x", None) is not None
        ):
            self._pinned_buf_x.copy_(data[idx])
            self._pinned_buf_y.copy_(data[idx_next])
            x = self._pinned_buf_x.to(self.device, non_blocking=True)
            y = self._pinned_buf_y.to(self.device, non_blocking=True)
        else:
            x = data[idx].to(self.device, non_blocking=True)
            y = data[idx_next].to(self.device, non_blocking=True)
        return x, y

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr * (step + 1) / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(
            1, self.max_steps - self.warmup_steps
        )
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if (
            getattr(self, "val_loader", None) is None
            and getattr(self, "val_data", None) is None
        ):
            return {"val_loss": 0.0, "val_bpc": 0.0, "val_ppl": 1.0}
        self.model.eval()
        losses = []
        for _ in range(self.eval_iters):
            x, y = self.get_batch("val")
            if getattr(self, "hybrid", None) is not None:
                _, loss, _ = self.hybrid(x, targets=y, return_logits=False)
                losses.append(loss.item())
            else:
                logits = self.model(
                    x,
                    use_quantized_gates=self.use_quantized_gates,
                    use_shift4_act=self.use_shift4_act,
                )
                loss = self.loss_fn(logits.view(-1, self.model.vocab_size), y.view(-1))
                losses.append(loss.item())
        self.model.train()
        mean_loss = float(np.mean(losses))
        bpc = mean_loss / math.log(2)
        ppl = math.exp(min(mean_loss, 20.0))
        return {"val_loss": mean_loss, "val_bpc": bpc, "val_ppl": ppl}

    @torch.no_grad()
    def routing_health(self, n_batches: int = 2) -> Dict[int, Dict[str, float]]:
        stats: Dict[int, Dict[str, float]] = {}
        hybrid = getattr(self, "hybrid", None)
        if hybrid is None:
            return stats
        blocks = getattr(getattr(hybrid, "context_encoder", None), "blocks", None)
        if not blocks:
            return stats
        was_training = hybrid.training
        hybrid.train()
        try:
            counts = []
            for _ in range(n_batches):
                x, _ = self.get_batch("val")
                hybrid(x)
                per_layer = []
                for blk in blocks:
                    cm = getattr(blk, "channel_mixer", None) or getattr(
                        blk, "asdag", None
                    )
                    rp = getattr(cm, "_last_routing_probs", None)
                    per_layer.append(None if rp is None else rp.detach().float().cpu())
                counts.append(per_layer)
            for li in range(len(blocks)):
                parts = [
                    c[li].reshape(-1, c[li].shape[-1])
                    for c in counts
                    if c[li] is not None
                ]
                if not parts:
                    continue
                p = torch.cat(parts, dim=0)
                K = p.shape[-1]
                t1 = p.argmax(dim=-1).reshape(-1)
                tot = t1.numel()
                top1 = 0.0
                dead = 0
                ent = 0.0
                for k in range(K):
                    f = float((t1 == k).sum()) / max(1, tot)
                    if f > top1:
                        top1 = f
                    if f < 0.01:
                        dead += 1
                    if f > 0:
                        ent -= f * math.log(f)
                stats[li] = {
                    "top1": top1,
                    "dead": float(dead),
                    "leaves": float(K),
                    "entropy": ent,
                    "entropy_max": math.log(K),
                }
        finally:
            if was_training:
                hybrid.train()
            else:
                hybrid.eval()
        return stats

    def train_step_backpressure(
        self, step: int, use_sign_backpressure: bool = False
    ) -> float:
        """
        Executes a zero-autograd closed-form local backpressure training step (1.B).
        Eliminates reverse-mode tape memory allocations.
        """
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        x, y = self.get_batch("train")

        logits = self.model(
            x,
            use_quantized_gates=self.use_quantized_gates,
            use_shift4_act=self.use_shift4_act,
            record_cache=True,
        )

        B, T, V = logits.shape
        # In-place softmax derivative without dense one-hot allocation
        probs = F.softmax(logits.detach(), dim=-1)
        # Error for backpressure: (1[target] - probs) / (B*T)
        scale_val = torch.tensor(-1.0, device=probs.device, dtype=probs.dtype)
        logits_error = -probs
        logits_error.scatter_add_(
            -1, y.unsqueeze(-1), scale_val.expand_as(y.unsqueeze(-1))
        )
        logits_error = -logits_error / float(B * T)

        loss = self.loss_fn(logits.view(-1, V), y.view(-1))

        self.optimizer.zero_grad(set_to_none=True)
        self.model.backward_backpressure(
            logits_error, use_sign_backpressure=use_sign_backpressure
        )

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        return loss.item()

    def train_step(self, step: int, sync_loss: bool = False) -> Any:
        """Executes a single optimized training step (Autograd, Backpressure, LPC, or Hybrid)."""
        if getattr(self, "hybrid", None) is not None:
            lr = self.get_lr(step)
            lr_ratio = lr / max(1e-8, self.lr)
            _graphed = bool(self.use_cuda_graph) and "cuda" in str(self.device)
            if (
                self.use_lpc
                and self.hybrid_optimizers is not None
                and hasattr(self.hybrid, "forward_lpc_step")
            ):
                if _graphed and not getattr(self, "_graph_lr_pinned", False):
                    # CUDA graphs bake host-side LR floats as launch constants: a
                    # per-step schedule can never take effect inside replay (all
                    # steps would silently reuse the capture-step LR). Pin base LRs
                    # once so warmup, recording, and replay agree (constant-LR).
                    for opt in self.hybrid_optimizers or []:
                        if hasattr(opt, "muon_opt") and opt.muon_opt:
                            for pg in opt.muon_opt.param_groups:
                                pg["lr"] = self.muon_lr
                        if hasattr(opt, "adamw_opt") and opt.adamw_opt:
                            for pg in opt.adamw_opt.param_groups:
                                pg["lr"] = self.lr
                        if not hasattr(opt, "muon_opt"):
                            for pg in opt.param_groups:
                                pg["lr"] = lr
                    self._graph_lr_pinned = True
                if not _graphed:
                    for opt in self.hybrid_optimizers:
                        if hasattr(opt, "muon_opt") and opt.muon_opt:
                            for pg in opt.muon_opt.param_groups:
                                pg["lr"] = self.muon_lr * lr_ratio
                        if hasattr(opt, "adamw_opt") and opt.adamw_opt:
                            for pg in opt.adamw_opt.param_groups:
                                pg["lr"] = self.lr * lr_ratio
                        if not hasattr(opt, "muon_opt"):
                            for pg in opt.param_groups:
                                pg["lr"] = lr
                x, y = self.get_batch("train")
                res = self.hybrid.forward_lpc_step(
                    x,
                    y,
                    self.hybrid_optimizers,
                    grad_clip=self.grad_clip,
                    sync_loss=sync_loss,
                    return_sample_loss=self.use_priority_replay,
                    use_cuda_graph=self.use_cuda_graph,
                )
                if (
                    self.use_priority_replay
                    and self.replay_buffer is not None
                    and "sample_loss" in res
                    and res["sample_loss"] is not None
                ):
                    n_fresh = len(getattr(self, "_last_train_fresh_ix", []))
                    if n_fresh > 0:
                        self.replay_buffer.push_candidates(
                            self._last_train_fresh_ix, res["sample_loss"][:n_fresh]
                        )
                return res["loss"]
            else:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
                x, y = self.get_batch("train")
                self.optimizer.zero_grad()
                logits, loss, _ = self.hybrid(x, targets=y, return_logits=True)
                if getattr(self, "scaler", None) is not None:
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.optimizer.step()
                if self.use_priority_replay and self.replay_buffer is not None:
                    n_fresh = len(getattr(self, "_last_train_fresh_ix", []))
                    if n_fresh > 0:
                        with torch.no_grad():
                            s_loss = (
                                F.cross_entropy(
                                    logits.view(-1, 256), y.view(-1), reduction="none"
                                )
                                .view(x.shape[0], -1)
                                .mean(dim=-1)
                            )
                            self.replay_buffer.push_candidates(
                                self._last_train_fresh_ix, s_loss[:n_fresh]
                            )
                return loss.item() if sync_loss else loss.detach()

        if self.use_backpressure:
            return self.train_step_backpressure(step)

        if self.use_lpc:
            lr = self.get_lr(step)
            lr_ratio = lr / max(1e-8, self.lr)
            for opt in self.lpc_optimizers:
                if hasattr(opt, "muon_opt") and opt.muon_opt:
                    for pg in opt.muon_opt.param_groups:
                        pg["lr"] = self.muon_lr * lr_ratio
                if hasattr(opt, "adamw_opt") and opt.adamw_opt:
                    for pg in opt.adamw_opt.param_groups:
                        pg["lr"] = self.lr * lr_ratio
                if not hasattr(opt, "muon_opt"):
                    for pg in opt.param_groups:
                        pg["lr"] = lr
            x, y = self.get_batch("train")
            res = self.lpc_model.forward_lpc_step(
                x, y, self.lpc_optimizers, grad_clip=self.grad_clip, sync_loss=sync_loss
            )
            return res["loss"]

        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        x, y = self.get_batch("train")
        logits = self.model(
            x,
            use_quantized_gates=self.use_quantized_gates,
            use_shift4_act=self.use_shift4_act,
        )
        loss = self.loss_fn(logits.view(-1, self.model.vocab_size), y.view(-1))

        self.optimizer.zero_grad(set_to_none=True)
        if getattr(self, "scaler", None) is not None:
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
        if self.use_priority_replay and self.replay_buffer is not None:
            n_fresh = len(getattr(self, "_last_train_fresh_ix", []))
            if n_fresh > 0:
                with torch.no_grad():
                    s_loss = (
                        F.cross_entropy(
                            logits.view(-1, self.model.vocab_size),
                            y.view(-1),
                            reduction="none",
                        )
                        .view(x.shape[0], -1)
                        .mean(dim=-1)
                    )
                    self.replay_buffer.push_candidates(
                        self._last_train_fresh_ix, s_loss[:n_fresh]
                    )
        return loss.item() if sync_loss else loss.detach()

    def train(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        if getattr(self, "is_distributed", False):
            try:
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    torch.cuda.set_device(self.local_rank % torch.cuda.device_count())
                else:
                    torch.cuda.set_device(self.local_rank)
            except Exception:
                pass
        self.model.train()
        best_val_loss = float("inf")
        start_time = time.time()

        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        try:
            for step in range(self.max_steps):
                if getattr(self, "train_sampler", None) is not None:
                    try:
                        self.train_sampler.set_epoch(step)
                    except Exception:
                        pass
                loss_val = self.train_step(step, sync_loss=False)
                if step % 500 == 499:
                    gc.collect()

                if step % self.eval_interval == 0 or step == self.max_steps - 1:
                    if getattr(self, "is_distributed", False) and not getattr(
                        self, "is_main", True
                    ):
                        continue
                    eval_metrics = self.evaluate()
                    try:
                        _rh = self.routing_health(n_batches=2)
                        if _rh:
                            _worst_li = min(
                                _rh,
                                key=lambda li: (
                                    _rh[li]["entropy"]
                                    / max(1e-9, _rh[li]["entropy_max"])
                                ),
                            )
                            _w = _rh[_worst_li]
                            print(
                                f"Routing health: worst layer {_worst_li} "
                                f"top1={_w['top1']:.2f} dead={int(_w['dead'])}/{int(_w['leaves'])} "
                                f"ent={_w['entropy']:.2f}/{_w['entropy_max']:.2f}",
                                flush=True,
                            )
                            if (
                                _w["entropy"] < 0.4 * _w["entropy_max"]
                                or _w["dead"] > 0.25 * _w["leaves"]
                            ):
                                import warnings

                                warnings.warn(
                                    f"Router starvation signs at layer {_worst_li}: "
                                    f"top1={_w['top1']:.2f}, dead={int(_w['dead'])}/{int(_w['leaves'])}. "
                                    f"Consider expert_bias_rate>0.",
                                    stacklevel=2,
                                )
                    except Exception:
                        pass
                    if getattr(self, "is_distributed", False) and self.world_size > 1:
                        try:
                            _t = torch.tensor(
                                eval_metrics["val_loss"],
                                device=self.device
                                if isinstance(self.device, torch.device)
                                else torch.device(self.device)
                                if "cuda" in str(self.device)
                                and torch.cuda.is_available()
                                else torch.device("cpu"),
                            )
                            all_reduce_sum(_t)
                            _t = _t / float(self.world_size)
                            eval_metrics["val_loss"] = float(_t.item())
                            eval_metrics["val_bpc"] = eval_metrics[
                                "val_loss"
                            ] / math.log(2)
                            eval_metrics["val_ppl"] = math.exp(
                                min(eval_metrics["val_loss"], 20.0)
                            )
                        except Exception:
                            pass
                    if eval_metrics["val_loss"] < best_val_loss:
                        best_val_loss = eval_metrics["val_loss"]
                        if save_path and getattr(self, "is_main", True):
                            import os as _os

                            _dir = _os.path.dirname(save_path)
                            if _dir:
                                _os.makedirs(_dir, exist_ok=True)
                            try:
                                _state = (
                                    self.model.module.state_dict()
                                    if isinstance(
                                        self.model,
                                        torch.nn.parallel.DistributedDataParallel,
                                    )
                                    else self.model.state_dict()
                                )
                            except Exception:
                                _state = self.model.state_dict()
                            torch.save(_state, save_path)
        finally:
            if gc_was_enabled:
                gc.enable()
            if getattr(self, "is_distributed", False):
                try:
                    barrier()
                except Exception:
                    pass

        total_time = time.time() - start_time
        if getattr(self, "is_distributed", False) and self.world_size > 1:
            try:
                _dev = (
                    self.device
                    if isinstance(self.device, torch.device)
                    else torch.device(self.device)
                    if "cuda" in str(self.device) and torch.cuda.is_available()
                    else torch.device("cpu")
                )
                _b = torch.tensor(best_val_loss, device=_dev)
                all_reduce_sum(_b)
                _b = _b / float(self.world_size)
                best_val_loss = float(_b.item())
            except Exception:
                pass
        return {
            "best_val_loss": best_val_loss,
            "best_val_bpc": best_val_loss / math.log(2),
            "best_val_ppl": math.exp(min(best_val_loss, 20.0)),
            "total_time_seconds": total_time,
        }


def train(
    model: Any,
    train_data: Any,
    val_data: Optional[Any] = None,
    batch_size: Optional[int] = None,
    seq_len: int = 512,
    context_window: Optional[int] = None,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    max_steps: int = 1000,
    warmup_steps: int = 100,
    eval_interval: int = 100,
    eval_iters: int = 20,
    grad_clip: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
    save_path: Optional[str] = None,
    use_cuda_graph: bool = True,
    use_muon: bool = True,
    muon_lr: float = 0.02,
    pad_id: int = 0,
    ignore_index: int = -100,
    # Context window & Token injection options
    inject_eos: bool = True,
    eos_token: Union[str, bytes, int] = "<|endoftext|>",
    as_stream: bool = False,
    # Multi-Token Prediction (MTP) options
    use_mtp: Optional[bool] = None,
    num_mtp_heads: Optional[int] = None,
    mtp_lambda: Optional[float] = None,
    # Architectural Mixer options
    channel_mixer: Optional[str] = "asdag_tree",
    time_mixer: Optional[str] = None,
    swa_every_n: Optional[int] = None,
    swa_window: Optional[int] = None,
    # Priority Replay (AXIOM Info-Gain Selection) options
    use_priority_replay: bool = False,
    replay_ratio: float = 0.25,
    replay_buffer_capacity: int = 4000,
    replay_max_replays: int = 3,
    distributed: Optional[bool] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    High-level, zero-friction training entry point for AffineAI models.

    Automatically handles:
    - Data ingestion via PaddedDataLoader (guaranteeing invariant static shapes).
    - Hardware INT8 IMMA Tensor Cores and zero-overhead CUDA Graphs on GPU.
    - Automatic <|endoftext|> injection into documents and sequences.
    - Multi-Token Prediction (MTP) heads & loss management.
    - Channel mixer (ternary_swiglu, asdag_tree, classic_mlp, dense_swiglu) & Time mixer (gla) configuration.
    - Dynamic Priority Replay (AXIOM info-gain selection) on high-uncertainty sequences.
    - Muon (for 2D projection weights) + AdamW (for norms, embeddings, 1D vectors).
    - Periodic evaluation and checkpoint saving.

    Parameters:
        model: TorosHybridLanguageModel, ASDAGLanguageModel, or any AffineAI model.
        train_data: List of strings/bytes, 1D numpy array, 1D torch Tensor, memmap, or PaddedDataLoader.
        val_data: Optional validation data (same formats as train_data).
        batch_size: Training batch size (auto-tuned if None).
        seq_len: Target sequence length in bytes or tokens (default: 512).
        context_window: Context window / sequence length alias (overrides seq_len if provided).
        lr: Learning rate for AdamW (default: 1e-3).
        weight_decay: Weight decay (default: 0.01).
        max_steps: Total training steps (default: 1000).
        warmup_steps: Linear LR warmup steps (default: 100).
        eval_interval: Evaluation interval in steps (default: 100).
        eval_iters: Number of batches to evaluate on (default: 20).
        grad_clip: Maximum gradient norm (default: 1.0).
        device: 'cuda', 'cpu', or torch.device (defaults to auto-detect).
        save_path: Optional file path to save the best model weights.
        use_cuda_graph: True to enable CUDA Graph replay (default: True on CUDA).
        use_muon: True to enable Muon Newton-Schulz optimization (default: True).
        muon_lr: Learning rate for Muon (default: 0.02).
        pad_id: Padding token/byte ID (default: 0).
        ignore_index: Target label for padded positions (default: -100).
        inject_eos: Whether to automatically inject <|endoftext|> to each sample/document (default: True).
        eos_token: End-of-text marker/token string, bytes, or int ID (default: "<|endoftext|>").
        as_stream: Whether to concatenate sequence inputs into a continuous 1D stream joined by EOS (default: False).
        use_mtp: Whether to enable Multi-Token Prediction (MTP) auxiliary heads (default: None).
        num_mtp_heads: Number of future prediction heads (default: 1 for ASDAG, 2 for TorosHybrid).
        mtp_lambda: Loss weighting coefficient for MTP auxiliary predictions (default: 0.3).
        channel_mixer: Channel mixer type ('ternary_swiglu', 'asdag_tree', 'classic_mlp', 'dense_swiglu').
        time_mixer: Time mixer rule ('gla').
        use_priority_replay: Whether to enable AXIOM Dynamic Priority Replay (default: True).
        replay_ratio: Fraction of batch capacity reserved for high-uncertainty replayed sequences (default: 0.25).
        replay_buffer_capacity: Maximum number of active sequence offsets in the replay buffer (default: 4000).
        replay_max_replays: Maximum replays per sequence before retirement (default: 3).
    """
    trainer = ASDAGTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        batch_size=batch_size,
        seq_len=seq_len,
        context_window=context_window,
        lr=lr,
        weight_decay=weight_decay,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        eval_interval=eval_interval,
        eval_iters=eval_iters,
        grad_clip=grad_clip,
        device=str(device) if device is not None else None,
        use_cuda_graph=use_cuda_graph,
        use_muon=use_muon,
        muon_lr=muon_lr,
        pad_id=pad_id,
        ignore_index=ignore_index,
        inject_eos=inject_eos,
        eos_token=eos_token,
        as_stream=as_stream,
        use_mtp=use_mtp,
        num_mtp_heads=num_mtp_heads,
        mtp_lambda=mtp_lambda,
        channel_mixer=channel_mixer,
        time_mixer=time_mixer,
        swa_every_n=swa_every_n,
        swa_window=swa_window,
        use_priority_replay=use_priority_replay,
        replay_ratio=replay_ratio,
        replay_buffer_capacity=replay_buffer_capacity,
        replay_max_replays=replay_max_replays,
        distributed=distributed,
        **kwargs,
    )
    return trainer.train(save_path=save_path)
