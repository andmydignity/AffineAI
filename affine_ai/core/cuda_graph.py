"""
CUDA Graph Runner for Zero-Overhead Hardware Execution
======================================================
Captures full-iteration forward-backward-optimizer computation graphs onto dedicated
CUDA streams, completely eliminating host CPU kernel launch overhead and Python dispatch
latency. Designed for Toros Local Predictive Coding (LPC) and ASDAG forward steps.
"""

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
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDAGraphRunner requires CUDA to be available.")

        self.step_fn = step_fn
        self.device = sample_inputs[0].device
        self.stream = stream or torch.cuda.Stream(device=self.device)
        self.warmup_iters = max(1, warmup_iters)

        # Preallocate static input buffers matching sample input shapes and dtypes
        self.static_inputs: List[torch.Tensor] = [
            torch.empty_like(t, memory_format=torch.contiguous_format) for t in sample_inputs
        ]
        for s_buf, t in zip(self.static_inputs, sample_inputs):
            s_buf.copy_(t)

        self.graph = torch.cuda.CUDAGraph()
        self.static_outputs: Any = None

        self._capture()

    def _capture(self):
        """Warmup and record the compute graph."""
        # Suppress stream mismatch warnings if supported in torch autograd
        if hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
            torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

        current_stream = torch.cuda.current_stream(device=self.device)
        self.stream.wait_stream(current_stream)

        with torch.cuda.stream(self.stream):
            # Warmup iterations on the capture stream
            for _ in range(self.warmup_iters):
                _ = self.step_fn(*self.static_inputs)

            # Record graph
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.static_outputs = self.step_fn(*self.static_inputs)

        current_stream.wait_stream(self.stream)

    def step(self, *inputs: torch.Tensor) -> Any:
        """
        Execute one captured step with the provided input tensors.

        Copies the dynamic inputs into the static buffers and replays the graph.
        """
        assert len(inputs) == len(self.static_inputs), (
            f"Expected {len(self.static_inputs)} inputs, got {len(inputs)}"
        )
        for s_buf, inp in zip(self.static_inputs, inputs):
            s_buf.copy_(inp)

        self.graph.replay()
        return self.static_outputs

    def __call__(self, *inputs: torch.Tensor) -> Any:
        return self.step(*inputs)

    def replay(self) -> Any:
        """Replay without copying inputs (uses existing static buffer data)."""
        self.graph.replay()
        return self.static_outputs
