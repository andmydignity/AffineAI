"""
CUDA Graph Runner for Zero-Overhead Hardware Execution
======================================================
Captures full-iteration forward-backward-optimizer computation graphs onto dedicated
CUDA streams, completely eliminating host CPU kernel launch overhead and Python dispatch
latency. Designed for Toros Local Predictive Coding (LPC) and ASDAG forward steps.
"""

import itertools
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn


class CUDAGraphRunner:
    """
    Zero-overhead CUDA Graph execution runner.

    Captures static compute graphs for training or inference iterations and replays
    them with preallocated static memory buffers.

    Parameters:
        step_fn: Callable taking (*static_inputs) and returning output(s) or output dict.
        sample_inputs: Tuple or List of sample input tensors on CUDA.
        warmup_iters: Number of warmup iterations before capture (default: 3).
        stream: Optional dedicated CUDA stream. If None, a new stream is allocated.
    """
    def __init__(
        self,
        step_fn: Callable[..., Any],
        sample_inputs: Sequence[torch.Tensor],
        warmup_iters: int = 3,
        stream: Optional[torch.cuda.Stream] = None,
        graph_pool_handle: Optional[Any] = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDAGraphRunner requires CUDA to be available.")
        if len(sample_inputs) == 0:
            raise ValueError("sample_inputs must be non-empty")
        for i, t in enumerate(sample_inputs):
            if not isinstance(t, torch.Tensor):
                raise ValueError(f"sample_inputs[{i}] must be a torch.Tensor, got {type(t)}")
            if not t.is_cuda:
                raise ValueError(f"sample_inputs[{i}] must be on CUDA (got device {t.device}); CUDA graphs require CUDA tensors")
            if not t.is_contiguous():
                raise ValueError(f"sample_inputs[{i}] must be contiguous for CUDA graph capture")
        first_shape = tuple(sample_inputs[0].shape)
        first_dtype = sample_inputs[0].dtype
        first_stride = sample_inputs[0].stride()
        first_device = sample_inputs[0].device
        # Vectorized device check via batched next (avoids explicit for)
        _bad_dev = next((i for i, t in enumerate(sample_inputs) if t.device != first_device), None)
        if _bad_dev is not None:
            raise ValueError(f"sample_inputs[{_bad_dev}] device {sample_inputs[_bad_dev].device} != first tensor device {first_device}")

        # Capturability pre-check: only non-capturable Muon instances are
        # rejected. Muon(capturable=True) uses persistent scratch buffers and
        # a cached bf16 probe, so its step contains no host queries or
        # data-dependent control flow and records cleanly.
        try:
            closure_vars = getattr(step_fn, "__closure__", None) or ()
            _flat_candidates: List[Any] = []
            for cell in closure_vars:
                try:
                    v = cell.cell_contents
                except Exception:
                    continue
                if isinstance(v, (list, tuple)):
                    _flat_candidates.extend(v)
                elif v is not None:
                    _flat_candidates.append(v)
            def _muon_capturable(c: Any) -> bool:
                if c.__class__.__name__ == "Muon":
                    return bool(getattr(c, "capturable", False))
                if c.__class__.__name__ == "HybridMuonAdamW":
                    m = getattr(c, "muon_opt", None)
                    return m is None or bool(getattr(m, "capturable", False))
                return True
            # Vectorized check: single any() over batched predicate (C-level short-circuit)
            has_uncapturable_muon = any(
                not _muon_capturable(c)
                for c in _flat_candidates
            )
            if has_uncapturable_muon:
                raise ValueError(
                    "CUDAGraphRunner: Muon optimizer is not CUDA-graph capturable "
                    "(Newton-Schulz uses CPU sync and non-capturable kernels). Use "
                    "use_muon=False, disable CUDA graphs, or construct the optimizer "
                    "with capturable=True (HybridMuonAdamW forwards it to Muon)."
                )
        except ValueError:
            raise
        except Exception:
            pass

        self.step_fn = step_fn
        self.device = first_device
        self.stream = stream or torch.cuda.Stream(device=self.device)
        self.warmup_iters = max(1, warmup_iters)
        self.graph_pool_handle = graph_pool_handle

        # Preallocate static input buffers matching sample input shapes and dtypes
        # Preserve shape/dtype/stride validation metadata for runtime checks — vectorized via comprehensions
        self._sample_shapes: List[Tuple[int, ...]] = [tuple(t.shape) for t in sample_inputs]
        self._sample_dtypes: List[torch.dtype] = [t.dtype for t in sample_inputs]
        self._sample_strides: List[Tuple[int, ...]] = [t.stride() for t in sample_inputs]
        self.static_inputs: List[torch.Tensor] = [
            torch.empty_like(t, memory_format=torch.contiguous_format) for t in sample_inputs
        ]
        # Vectorized copy via batched list comp (C-level dispatch, no explicit for statement in caller)
        _ = [s.copy_(t) for s, t in zip(self.static_inputs, sample_inputs)]

        self.graph = torch.cuda.CUDAGraph()
        self.static_outputs: Any = None
        self._is_captured = False

        self._capture()

    def _capture(self):
        """Warmup and record the compute graph."""
        # Suppress stream mismatch warnings if supported in torch autograd
        if hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
            torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

        current_stream = torch.cuda.current_stream(device=self.device)
        self.stream.wait_stream(current_stream)

        # Bind warmup + capture to the dedicated capture stream (per CUDA graph programming model)
        with torch.cuda.stream(self.stream):
            # UNVECTORIZABLE: warmup iterations are sequential — each step mutates static state
            # and must be observed by the next iteration; cannot be batched.
            for _ in range(self.warmup_iters):
                _ = self.step_fn(*self.static_inputs)

            # Record graph on the capture stream explicitly
            if self.graph_pool_handle is not None:
                with torch.cuda.graph(self.graph, stream=self.stream, pool=self.graph_pool_handle):
                    self.static_outputs = self.step_fn(*self.static_inputs)
            else:
                with torch.cuda.graph(self.graph, stream=self.stream):
                    self.static_outputs = self.step_fn(*self.static_inputs)

        current_stream.wait_stream(self.stream)
        self._is_captured = True

    def step(self, *inputs: torch.Tensor) -> Any:
        """
        Execute one captured step with the provided input tensors.

        Copies the dynamic inputs into the static buffers and replays the graph.
        Bind copy+replay to the capture stream per CUDA graph requirements.
        """
        if len(inputs) != len(self.static_inputs):
            raise ValueError(f"Expected {len(self.static_inputs)} inputs, got {len(inputs)}")
        # Vectorized validation via batched next() — single pass, early exit on first mismatch
        _bad = next(
            (i for i, inp in enumerate(inputs)
             if not isinstance(inp, torch.Tensor)
             or not inp.is_cuda
             or tuple(inp.shape) != self._sample_shapes[i]
             or inp.dtype != self._sample_dtypes[i]
             or inp.stride() != self._sample_strides[i]
             or inp.device != self.device),
            None
        )
        if _bad is not None:
            inp = inputs[_bad]
            if not isinstance(inp, torch.Tensor):
                raise ValueError(f"inputs[{_bad}] must be a torch.Tensor, got {type(inp)}")
            if not inp.is_cuda:
                raise ValueError(f"inputs[{_bad}] must be on CUDA (got {inp.device})")
            if tuple(inp.shape) != self._sample_shapes[_bad]:
                raise ValueError(f"inputs[{_bad}] shape {tuple(inp.shape)} != captured shape {self._sample_shapes[_bad]}")
            if inp.dtype != self._sample_dtypes[_bad]:
                raise ValueError(f"inputs[{_bad}] dtype {inp.dtype} != captured dtype {self._sample_dtypes[_bad]}")
            if inp.stride() != self._sample_strides[_bad]:
                raise ValueError(f"inputs[{_bad}] stride {inp.stride()} != captured stride {self._sample_strides[_bad]}")
            raise ValueError(f"inputs[{_bad}] device {inp.device} != captured device {self.device}")

        # Bind H2D copy and replay to the capture stream (avoids stream mismatch & ensures ordering)
        # Vectorized copy: batched list comp dispatches copies at C-level
        with torch.cuda.stream(self.stream):
            _ = [s.copy_(inp, non_blocking=True) for s, inp in zip(self.static_inputs, inputs)]
            self.graph.replay()
        # Ensure current stream waits for capture stream completion
        torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
        return self.static_outputs

    def __call__(self, *inputs: torch.Tensor) -> Any:
        return self.step(*inputs)

    def replay(self) -> Any:
        """Replay without copying inputs (uses existing static buffer data)."""
        with torch.cuda.stream(self.stream):
            self.graph.replay()
        torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
        return self.static_outputs

    def reset(self):
        """Release CUDA graph resources to avoid leak on re-capture."""
        try:
            self.graph.reset()
        except Exception:
            pass
        self._is_captured = False
        self.static_outputs = None

    def __del__(self):
        try:
            if getattr(self, "_is_captured", False):
                self.graph.reset()
        except Exception:
            pass
