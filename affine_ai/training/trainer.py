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
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
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
        model: ASDAGLanguageModel,
        train_data: Union[torch.Tensor, np.ndarray],
        val_data: Union[torch.Tensor, np.ndarray],
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
        muon_lr: float = 0.03
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_backpressure = use_backpressure
        self.use_lpc = use_lpc
        self.use_muon = use_muon
        self.muon_lr = muon_lr
        
        # CPU Threading Optimization
        if self.device == "cpu":
            target_threads = num_threads or get_cpu_physical_cores()
            try:
                torch.set_num_threads(target_threads)
                torch.set_num_interop_threads(1)
            except Exception:
                pass

        self.model = model.to(self.device)
        if self.device == "cpu":
            try:
                from affine_ai.core.numa import node_count, interleave_model_weights
                if node_count() > 1:
                    interleave_model_weights(self.model)
            except Exception:
                pass
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = suggest_batch_size(self.device)
        self.seq_len = seq_len
        self.lr = lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.grad_clip = grad_clip
        self.use_quantized_gates = use_quantized_gates
        self.use_shift4_act = use_shift4_act

        if isinstance(train_data, np.ndarray):
            self.train_data = torch.from_numpy(train_data.astype(np.int64))
        else:
            self.train_data = train_data.to(torch.long)

        if isinstance(val_data, np.ndarray):
            self.val_data = torch.from_numpy(val_data.astype(np.int64))
        else:
            self.val_data = val_data.to(torch.long)

        # Allocate pinned staging buffers on CPU for zero-copy DMA to CUDA
        if "cuda" in str(self.device):
            try:
                self._pinned_buf_x = torch.empty((self.batch_size, self.seq_len), dtype=torch.long, pin_memory=True)
                self._pinned_buf_y = torch.empty((self.batch_size, self.seq_len), dtype=torch.long, pin_memory=True)
            except Exception:
                self._pinned_buf_x = None
                self._pinned_buf_y = None
        else:
            self._pinned_buf_x = None
            self._pinned_buf_y = None

        # Fused / standard AdamW
        fused = (self.device == "cuda" and hasattr(optim.AdamW, "_fused"))
        if getattr(self.model, "hybrid", None) is not None:
            hybrid = self.model.hybrid
            can_lpc = self.use_lpc and hasattr(hybrid, 'enable_lpc') and hasattr(hybrid, 'get_default_lpc_optimizers')
            has_legacy_local = self.use_lpc and getattr(hybrid, 'local_heads', None) is not None
            if can_lpc:
                try:
                    try:
                        hybrid.enable_lpc(device=self.device)
                    except TypeError:
                        hybrid.enable_lpc()
                    has_local_lpc = getattr(hybrid, 'local_heads', None) is not None
                except Exception:
                    has_local_lpc = False
                if has_local_lpc:
                    lpc_kwargs = {}
                    if hasattr(hybrid.get_default_lpc_optimizers, '__code__'):
                        import inspect
                        sig = inspect.signature(hybrid.get_default_lpc_optimizers)
                        if 'muon_lr' in sig.parameters:
                            lpc_kwargs['muon_lr'] = self.muon_lr
                    try:
                        self.hybrid_optimizers = hybrid.get_default_lpc_optimizers(
                            lr=lr, weight_decay=weight_decay, use_muon=self.use_muon, **lpc_kwargs
                        )
                    except TypeError:
                        self.hybrid_optimizers = hybrid.get_default_lpc_optimizers(
                            lr=lr, weight_decay=weight_decay, use_muon=self.use_muon, muon_lr=self.muon_lr
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
                        fused=fused
                    )
            elif has_legacy_local:
                self.hybrid_optimizers = hybrid.get_default_optimizers(
                    lr=lr, weight_decay=weight_decay, use_muon=self.use_muon, muon_lr=self.muon_lr
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
                    fused=fused
                )
        elif self.use_lpc:
            from affine_ai.core.lpc import LocalPredictiveLanguageModel
            self.lpc_model = LocalPredictiveLanguageModel(self.model).to(self.device)
            self.lpc_optimizers = self.lpc_model.get_default_lpc_optimizers(
                lr=lr, weight_decay=weight_decay, use_muon=self.use_muon, muon_lr=self.muon_lr
            )
            self.hybrid_optimizers = None
            self.optimizer = None
        else:
            self.lpc_model = None
            self.lpc_optimizers = None
            self.hybrid_optimizers = None
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                fused=fused
            )

        self.loss_fn = nn.CrossEntropyLoss()

    def get_batch(self, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        high = len(data) - self.seq_len - 1
        if high <= 0:
            raise ValueError(f"Dataset too small ({len(data)}) for seq_len {self.seq_len}")
        ix = torch.randint(0, high, (self.batch_size,))
        offsets = torch.arange(self.seq_len, device=ix.device)
        idx = ix.unsqueeze(1) + offsets.unsqueeze(0)
        idx_next = idx + 1

        if data.is_cuda:
            if idx.device != data.device:
                idx = idx.to(data.device, non_blocking=True)
                idx_next = idx_next.to(data.device, non_blocking=True)
            x = data[idx]
            y = data[idx_next]
        elif "cuda" in str(self.device) and getattr(self, "_pinned_buf_x", None) is not None:
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
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        losses = []
        for _ in range(self.eval_iters):
            x, y = self.get_batch("val")
            if getattr(self.model, "hybrid", None) is not None:
                _, loss, _ = self.model.hybrid(x, targets=y, return_logits=False)
                losses.append(loss.item())
            else:
                logits = self.model(
                    x,
                    use_quantized_gates=self.use_quantized_gates,
                    use_shift4_act=self.use_shift4_act
                )
                loss = self.loss_fn(logits.view(-1, self.model.vocab_size), y.view(-1))
                losses.append(loss.item())
        self.model.train()
        mean_loss = float(np.mean(losses))
        bpc = mean_loss / math.log(2)
        ppl = math.exp(min(mean_loss, 20.0))
        return {"val_loss": mean_loss, "val_bpc": bpc, "val_ppl": ppl}

    def train_step_backpressure(self, step: int, use_sign_backpressure: bool = False) -> float:
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
            record_cache=True
        )

        B, T, V = logits.shape
        # In-place softmax derivative without dense one-hot allocation
        probs = F.softmax(logits.detach(), dim=-1)
        # Error for backpressure: (1[target] - probs) / (B*T)
        scale_val = torch.tensor(-1.0, device=probs.device, dtype=probs.dtype)
        logits_error = -probs
        logits_error.scatter_add_(-1, y.unsqueeze(-1), scale_val.expand_as(y.unsqueeze(-1)))
        logits_error = -logits_error / float(B * T)

        loss = self.loss_fn(logits.view(-1, V), y.view(-1))

        self.optimizer.zero_grad(set_to_none=True)
        self.model.backward_backpressure(logits_error, use_sign_backpressure=use_sign_backpressure)

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        return loss.item()

    def train_step(self, step: int, sync_loss: bool = False) -> Any:
        """Executes a single optimized training step (Autograd, Backpressure, LPC, or Hybrid)."""
        if getattr(self.model, "hybrid", None) is not None:
            lr = self.get_lr(step)
            lr_ratio = lr / max(1e-8, self.lr)
            if self.use_lpc and self.hybrid_optimizers is not None and hasattr(self.model.hybrid, 'forward_lpc_step'):
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
                res = self.model.hybrid.forward_lpc_step(
                    x, y, self.hybrid_optimizers, grad_clip=self.grad_clip, sync_loss=sync_loss
                )
                return res["loss"]
            else:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
                x, y = self.get_batch("train")
                self.optimizer.zero_grad()
                _, loss, _ = self.model.hybrid(x, targets=y, return_logits=False)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
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
            res = self.lpc_model.forward_lpc_step(x, y, self.lpc_optimizers, grad_clip=self.grad_clip, sync_loss=sync_loss)
            return res["loss"]

        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        x, y = self.get_batch("train")
        logits = self.model(
            x,
            use_quantized_gates=self.use_quantized_gates,
            use_shift4_act=self.use_shift4_act
        )
        loss = self.loss_fn(logits.view(-1, self.model.vocab_size), y.view(-1))

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        return loss.item() if sync_loss else loss.detach()

    def train(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        self.model.train()
        best_val_loss = float("inf")
        start_time = time.time()

        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        try:
            for step in range(self.max_steps):
                loss_val = self.train_step(step, sync_loss=False)
                if step % 500 == 499:
                    gc.collect()

                if step % self.eval_interval == 0 or step == self.max_steps - 1:
                    eval_metrics = self.evaluate()
                    if eval_metrics["val_loss"] < best_val_loss:
                        best_val_loss = eval_metrics["val_loss"]
                        if save_path:
                            torch.save(self.model.state_dict(), save_path)
        finally:
            if gc_was_enabled:
                gc.enable()

        total_time = time.time() - start_time
        return {
            "best_val_loss": best_val_loss,
            "best_val_bpc": best_val_loss / math.log(2),
            "best_val_ppl": math.exp(min(best_val_loss, 20.0)),
            "total_time_seconds": total_time
        }
