"""
Local Predictive Coding (LPC) Core Modules & Model Wrapper
===========================================================
Provides forward-only, layer-wise decoupled credit assignment without cross-layer backward passes.
"""

import math
from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.core.norm import RMSNorm
from affine_ai.models.language_model import ASDAGLanguageModel


class LocalPredictiveHead(nn.Module):
    """
    MatMul-free Local Predictive Head via ternary BitLinear.
    Replaces dense GEMM h @ W^T with ternary weight additions (STE).
    """
    def __init__(
        self,
        d_model: int,
        vocab_size: int = 256,
        dtype: Any = torch.float32
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.norm = RMSNorm(d_model)
        self.weight = nn.Parameter(
            torch.randn(vocab_size, d_model, dtype=dtype) * (1.0 / math.sqrt(d_model))
        )

    def forward(
        self,
        h: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        ignore_index: int = -100
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h_norm = self.norm(h)
        gamma = self.weight.abs().mean().clamp(min=1e-5)
        w_scaled = self.weight / gamma
        w_ternary = torch.round(w_scaled).clamp(-1.0, 1.0)
        w_quant = self.weight + (w_ternary * gamma - self.weight).detach()
        if targets is not None:
            if h.is_cuda:
                try:
                    from affine_ai.kernels.triton_lpc import triton_fused_lpc_head
                    loss = triton_fused_lpc_head(h_norm, w_quant, targets, ignore_index=ignore_index)
                    # If Triton returns NaN (can happen with wide BF16 models), fall through
                    if not loss.isnan().any():
                        return h_norm, loss
                except Exception:
                    pass
            # Cast logits to float32 before cross_entropy — prevents BF16 overflow (>65504)
            # in wide models (dim >= 512) where logit magnitudes can exceed BF16 range.
            logits = F.linear(h_norm, w_quant)
            loss = F.cross_entropy(logits.float().view(-1, self.vocab_size), targets.view(-1), ignore_index=ignore_index)
            return h_norm, loss
        logits = F.linear(h_norm, w_quant)
        return logits, None


class LocalPredictiveLanguageModel(nn.Module):
    """
    End-to-End Local Predictive Coding Model Wrapper for ASDAGLanguageModel.
    Decouples layer execution such that each block computes, predicts, and updates
    in-place during forward streaming with zero cross-layer autograd graph.
    """
    def __init__(
        self,
        base_model: ASDAGLanguageModel,
        dtype: Any = torch.float32,
        tie_heads: bool = True,
        stride: int = 1
    ):
        super().__init__()
        self.base_model = base_model
        self.d_model = base_model.d_model
        self.vocab_size = base_model.vocab_size
        self.n_layers = len(base_model.blocks)
        self.tie_heads = tie_heads
        self.stride = stride

        # Local predictive heads for all intermediate layers
        self.local_heads = nn.ModuleList([
            LocalPredictiveHead(self.d_model, self.vocab_size, dtype=dtype)
            for _ in range(self.n_layers)
        ])

        if self.tie_heads:
            for head in self.local_heads:
                head.weight = self.base_model.tok_embeddings.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Standard full-model forward pass for evaluation and inference."""
        return self.base_model(input_ids)

    def get_default_lpc_optimizers(
        self,
        lr: float = 3e-3,
        weight_decay: float = 0.01,
        use_muon: bool = True,
        muon_lr: float = 0.03,
        muon_momentum: float = 0.95,
        capturable: Optional[bool] = None,
    ) -> List[Any]:
        """
        Creates independent, decoupled optimizers for each layer (AdamW or HybridMuonAdamW).
        Allows immediate, in-place parameter updates per block.

        Async-safe layout (new): [block0+head0, ..., blockN-1+headN-1, enc_tail, final]
        i.e. n_layers layer opts + 1 dedicated shared-encoder opt + 1 final readout opt.
        Shared encoder params (tok_embeddings, and aliased head weights when
        tie_heads=True) live ONLY in enc_tail, stepped AFTER the per-layer
        pipeline drains — so per-layer async overlap is race-free.
        Old callers passing n_layers+1 optimizers still work (enc step skipped).
        """
        if capturable is None:
            try:
                capturable = next(self.parameters()).is_cuda
            except Exception:
                capturable = False
        optimizers = []
        if use_muon:
            from affine_ai.optim.muon import HybridMuonAdamW
            if self.tie_heads:
                # head.weight aliases tok_embeddings.weight: exclude weight from
                # layer opts (own block + head.norm only), tail owns embedding.
                p0 = nn.ModuleList([self.base_model.blocks[0], self.local_heads[0].norm])
                optimizers.append(HybridMuonAdamW(p0, muon_lr=muon_lr, adamw_lr=lr, muon_momentum=muon_momentum, adamw_weight_decay=weight_decay, capturable=capturable))

                for i in range(1, self.n_layers):
                    pi = nn.ModuleList([self.base_model.blocks[i], self.local_heads[i].norm])
                    optimizers.append(HybridMuonAdamW(pi, muon_lr=muon_lr, adamw_lr=lr, muon_momentum=muon_momentum, adamw_weight_decay=weight_decay, capturable=capturable))
            else:
                # Optimizer 0: Block 0 + LocalHead 0 (no shared encoder params)
                p0 = nn.ModuleList([self.base_model.blocks[0], self.local_heads[0]])
                optimizers.append(HybridMuonAdamW(p0, muon_lr=muon_lr, adamw_lr=lr, muon_momentum=muon_momentum, adamw_weight_decay=weight_decay, capturable=capturable))

                for i in range(1, self.n_layers):
                    pi = nn.ModuleList([self.base_model.blocks[i], self.local_heads[i]])
                    optimizers.append(HybridMuonAdamW(pi, muon_lr=muon_lr, adamw_lr=lr, muon_momentum=muon_momentum, adamw_weight_decay=weight_decay, capturable=capturable))

            # Dedicated tail for shared encoder — stepped after loop, not pipelined.
            enc_tail = nn.ModuleList([self.base_model.tok_embeddings])
            optimizers.append(HybridMuonAdamW(enc_tail, muon_lr=muon_lr, adamw_lr=lr, muon_momentum=muon_momentum, adamw_weight_decay=weight_decay, capturable=capturable))

            p_final = nn.ModuleList([self.base_model.norm_f, self.base_model.lm_head])
            optimizers.append(HybridMuonAdamW(p_final, muon_lr=muon_lr, adamw_lr=lr, muon_momentum=muon_momentum, adamw_weight_decay=weight_decay, capturable=capturable))
            return optimizers

        # Standard decoupled AdamW
        enc_ids = {id(p) for p in self.base_model.tok_embeddings.parameters()}
        def _layer_params(bi: int) -> List[Any]:
            ps = list(self.base_model.blocks[bi].parameters())
            for p in self.local_heads[bi].parameters():
                if id(p) not in enc_ids:  # skip aliased embedding weight when tie_heads
                    ps.append(p)
            return ps
        p0 = _layer_params(0)
        adamw_kwargs: Dict[str, Any] = {"lr": lr, "weight_decay": weight_decay}
        if capturable:
            adamw_kwargs["capturable"] = True
        optimizers.append(torch.optim.AdamW(p0, **adamw_kwargs))

        for i in range(1, self.n_layers):
            pi = _layer_params(i)
            optimizers.append(torch.optim.AdamW(pi, **adamw_kwargs))

        enc_params = list(self.base_model.tok_embeddings.parameters())
        optimizers.append(torch.optim.AdamW(enc_params, **adamw_kwargs))

        p_final = (
            list(self.base_model.norm_f.parameters()) +
            list(self.base_model.lm_head.parameters())
        )
        optimizers.append(torch.optim.AdamW(p_final, **adamw_kwargs))
        return optimizers

    def try_compile_lpc_blocks(self) -> bool:
        if getattr(self, '_lpc_blocks_compiled', False):
            return True
        try:
            import torch as _torch
            if not _torch.cuda.is_available():
                return False
            try:
                if tuple(_torch.cuda.get_device_capability()) < (8, 9):
                    return False
            except Exception:
                return False
            for blk in self.base_model.blocks:
                try:
                    blk.forward = _torch.compile(blk.forward, mode="reduce-overhead", dynamic=False, fullgraph=False)
                except Exception:
                    return False
            self._lpc_blocks_compiled = True
            return True
        except Exception:
            return False

    def capture_lpc_graph(
        self,
        sample_input_ids: torch.Tensor,
        sample_targets: torch.Tensor,
        optimizers: List[Any],
        warmup_iters: int = 3,
        grad_clip: float = 1.0,
        ignore_index: int = -100,
        stride: Optional[int] = None,
    ) -> Any:
        from affine_ai.core.cuda_graph import CUDAGraphRunner

        def step_fn(bx: torch.Tensor, by: torch.Tensor):
            return self.forward_lpc_step(
                bx, by, optimizers,
                grad_clip=grad_clip,
                ignore_index=ignore_index,
                use_async_pipelining=False,
                stride=stride,
                sync_loss=False,
                use_cuda_graph=False,
                use_compiled_blocks=False,
            )

        return CUDAGraphRunner(
            step_fn=step_fn,
            sample_inputs=(sample_input_ids, sample_targets),
            warmup_iters=warmup_iters,
        )

    def forward_lpc_step(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        optimizers: List[Any],
        grad_clip: float = 1.0,
        ignore_index: int = -100,
        use_async_pipelining: bool = True,
        stride: Optional[int] = None,
        sync_loss: bool = False,
        use_cuda_graph: bool = False,
        use_compiled_blocks: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes a complete forward-only Local Predictive Coding step.
        Supports asynchronous CUDA stream pipelining and strided token error sampling.
        """
        is_cuda = input_ids.is_cuda
        if use_compiled_blocks and is_cuda:
            self.try_compile_lpc_blocks()
        if use_cuda_graph and is_cuda and not sync_loss:
            runner = getattr(self, "_lpc_graph_runner", None)
            if (
                runner is not None
                and runner.static_inputs[0].shape == input_ids.shape
                and runner.static_inputs[0].dtype == input_ids.dtype
            ):
                return runner.step(input_ids, targets)
            try:
                self._lpc_graph_runner = self.capture_lpc_graph(
                    input_ids, targets, optimizers,
                    warmup_iters=3,
                    grad_clip=grad_clip,
                    ignore_index=ignore_index,
                    stride=stride,
                )
                return self._lpc_graph_runner.step(input_ids, targets)
            except Exception:
                self._lpc_graph_runner = None
        B, T = input_ids.shape
        # New async-safe layout has n_layers+2 optimizers (layer opts + enc tail + final).
        has_enc_tail = len(optimizers) == self.n_layers + 2
        if has_enc_tail:
            optimizers[-2].zero_grad(set_to_none=is_cuda)
        x = self.base_model.tok_embeddings(input_ids)
        layer_losses = []

        eff_stride = stride if stride is not None else self.stride
        sub_targets = targets[:, ::eff_stride] if eff_stride > 1 else targets

        if is_cuda and use_async_pipelining:
            if getattr(self, '_bwd_stream', None) is None or self._bwd_stream.device != input_ids.device:
                self._bwd_stream = torch.cuda.Stream(device=input_ids.device)
            bwd_stream = self._bwd_stream
            bwd_event = None
        else:
            bwd_stream = None
            bwd_event = None

        curr_h = x

        for idx, block in enumerate(self.base_model.blocks):
            if idx == 0 and has_enc_tail:
                curr_h_in = curr_h
            else:
                curr_h = curr_h.detach()
                curr_h_in = curr_h.requires_grad_(True)

            # Block forward transformation
            next_h = block(curr_h_in)
            h_sub = next_h[:, ::eff_stride] if eff_stride > 1 else next_h

            # Local predictive head forward
            _, loss_i = self.local_heads[idx](h_sub, targets=sub_targets, ignore_index=ignore_index)

            # Asynchronous Pipelined Execution on CUDA via Double-Buffering
            if is_cuda and use_async_pipelining:
                if bwd_event is not None:
                    torch.cuda.current_stream().wait_event(bwd_event)

                fwd_event = torch.cuda.Event()
                fwd_event.record(torch.cuda.current_stream())
                with torch.cuda.stream(bwd_stream):
                    bwd_stream.wait_event(fwd_event)
                    opt_i = optimizers[idx]
                    opt_i.zero_grad(set_to_none=True)
                    loss_i.backward()
                    if grad_clip > 0:
                        params = [p for pg in opt_i.param_groups for p in pg['params']]
                        torch.nn.utils.clip_grad_norm_(params, grad_clip, foreach=True)
                    opt_i.step()
                    opt_i.zero_grad(set_to_none=True)
                    bwd_event = torch.cuda.Event()
                    bwd_event.record(bwd_stream)

                layer_losses.append(loss_i.item() if sync_loss else loss_i.detach())
            else:
                opt_i = optimizers[idx]
                opt_i.zero_grad(set_to_none=is_cuda)
                loss_i.backward()
                if grad_clip > 0:
                    params = [p for pg in opt_i.param_groups for p in pg['params']]
                    torch.nn.utils.clip_grad_norm_(params, grad_clip, foreach=True)
                opt_i.step()
                opt_i.zero_grad(set_to_none=is_cuda)
                layer_losses.append(loss_i.item() if sync_loss else loss_i.detach())

            curr_h = next_h.detach() if (idx == 0 and has_enc_tail) else next_h

        if is_cuda and use_async_pipelining and bwd_event is not None:
            torch.cuda.current_stream().wait_event(bwd_event)

        if has_enc_tail:
            opt_enc = optimizers[-2]
            if grad_clip > 0:
                enc_params = [p for pg in opt_enc.param_groups for p in pg['params'] if p.grad is not None]
                if enc_params:
                    torch.nn.utils.clip_grad_norm_(enc_params, grad_clip, foreach=True)
            opt_enc.step()
            opt_enc.zero_grad(set_to_none=is_cuda)

        # Final output layer update
        curr_h_in = curr_h.detach().requires_grad_(True)
        final_h = self.base_model.norm_f(curr_h_in)
        if final_h.is_cuda:
            try:
                from affine_ai.kernels.triton_cross_entropy import triton_fused_linear_cross_entropy
                loss_final = triton_fused_linear_cross_entropy(
                    final_h, self.base_model.lm_head.weight, targets, ignore_index=ignore_index
                )
            except Exception:
                final_logits = self.base_model.lm_head(final_h)
                loss_final = F.cross_entropy(final_logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=ignore_index)
        else:
            final_logits = self.base_model.lm_head(final_h)
            loss_final = F.cross_entropy(final_logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=ignore_index)

        opt_final = optimizers[-1]
        opt_final.zero_grad(set_to_none=is_cuda)
        loss_final.backward()
        if grad_clip > 0:
            params = []
            for pg in opt_final.param_groups:
                params.extend(pg['params'])
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt_final.step()
        opt_final.zero_grad(set_to_none=is_cuda)

        if sync_loss:
            return {
                "loss": loss_final.item(),
                "layer_losses": layer_losses,
                "mean_local_loss": sum(layer_losses) / len(layer_losses) if layer_losses else loss_final.item()
            }
        else:
            loss_final_det = loss_final.detach()
            return {
                "loss": loss_final_det,
                "layer_losses": layer_losses,
                "mean_local_loss": torch.stack(layer_losses).mean() if layer_losses else loss_final_det
            }

    @torch.no_grad()
    def generate(self, *args, **kwargs) -> torch.Tensor:
        """Delegates generation to underlying base model."""
        return self.base_model.generate(*args, **kwargs)
