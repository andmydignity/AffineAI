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
        logits = F.linear(h_norm, w_quant)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=ignore_index)
            return h_norm, loss
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
        muon_momentum: float = 0.95
    ) -> List[Any]:
        """
        Creates independent, decoupled optimizers for each layer (AdamW or HybridMuonAdamW).
        Allows immediate, in-place parameter updates per block.
        """
        optimizers = []
        if use_muon:
            from affine_ai.optim.muon import HybridMuonAdamW
            # Optimizer 0: Embeddings + Block 0 + LocalHead 0
            p0 = nn.ModuleList([self.base_model.tok_embeddings, self.base_model.blocks[0], self.local_heads[0]])
            optimizers.append(HybridMuonAdamW(p0, muon_lr=muon_lr, adamw_lr=lr, muon_momentum=muon_momentum, adamw_weight_decay=weight_decay))

            for i in range(1, self.n_layers):
                pi = nn.ModuleList([self.base_model.blocks[i], self.local_heads[i]])
                optimizers.append(HybridMuonAdamW(pi, muon_lr=muon_lr, adamw_lr=lr, muon_momentum=muon_momentum, adamw_weight_decay=weight_decay))

            p_final = nn.ModuleList([self.base_model.norm_f, self.base_model.lm_head])
            optimizers.append(HybridMuonAdamW(p_final, muon_lr=muon_lr, adamw_lr=lr, muon_momentum=muon_momentum, adamw_weight_decay=weight_decay))
            return optimizers

        # Standard decoupled AdamW
        p0 = (
            list(self.base_model.tok_embeddings.parameters()) +
            list(self.base_model.blocks[0].parameters()) +
            list(self.local_heads[0].parameters())
        )
        optimizers.append(torch.optim.AdamW(p0, lr=lr, weight_decay=weight_decay))

        for i in range(1, self.n_layers):
            pi = (
                list(self.base_model.blocks[i].parameters()) +
                list(self.local_heads[i].parameters())
            )
            optimizers.append(torch.optim.AdamW(pi, lr=lr, weight_decay=weight_decay))

        p_final = (
            list(self.base_model.norm_f.parameters()) +
            list(self.base_model.lm_head.parameters())
        )
        optimizers.append(torch.optim.AdamW(p_final, lr=lr, weight_decay=weight_decay))
        return optimizers

    def forward_lpc_step(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        optimizers: List[Any],
        grad_clip: float = 1.0,
        ignore_index: int = -100,
        use_async_pipelining: bool = True,
        stride: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Executes a complete forward-only Local Predictive Coding step.
        Supports asynchronous CUDA stream pipelining and strided token error sampling.
        """
        B, T = input_ids.shape
        x = self.base_model.tok_embeddings(input_ids)
        layer_losses = []

        eff_stride = stride if stride is not None else self.stride
        sub_targets = targets[:, ::eff_stride] if eff_stride > 1 else targets

        is_cuda = input_ids.is_cuda
        if is_cuda and use_async_pipelining:
            if getattr(self, '_cuda_streams', None) is None or len(self._cuda_streams) != len(self.base_model.blocks):
                self._cuda_streams = [torch.cuda.Stream(device=input_ids.device) for _ in range(len(self.base_model.blocks))]
        elif not is_cuda and use_async_pipelining:
            import concurrent.futures
            if getattr(self, '_cpu_executor', None) is None:
                self._cpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        curr_h = x
        prev_future = None

        for idx, block in enumerate(self.base_model.blocks):
            # Detach to guarantee zero cross-layer autograd tape (strictly O(1) memory)
            curr_h = curr_h.detach()
            curr_h_in = curr_h.requires_grad_(True)

            # Block forward transformation
            next_h = block(curr_h_in)
            h_sub = next_h[:, ::eff_stride] if eff_stride > 1 else next_h

            # Local predictive head forward
            _, loss_i = self.local_heads[idx](h_sub, targets=sub_targets, ignore_index=ignore_index)

            # Asynchronous Pipelined Execution on CUDA
            if is_cuda and use_async_pipelining:
                stream = self._cuda_streams[idx]
                with torch.cuda.stream(stream):
                    opt_i = optimizers[idx]
                    opt_i.zero_grad(set_to_none=True)
                    loss_i.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(block.parameters(), grad_clip)
                    opt_i.step()
                    opt_i.zero_grad(set_to_none=True)
                layer_losses.append(loss_i.item())
            else:
                opt_i = optimizers[idx]
                opt_i.zero_grad(set_to_none=False)
                loss_i.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(block.parameters(), grad_clip)
                opt_i.step()
                opt_i.zero_grad(set_to_none=False)
                layer_losses.append(loss_i.item())

            curr_h = next_h

        # Final output layer update
        curr_h_in = curr_h.detach().requires_grad_(True)
        final_h = self.base_model.norm_f(curr_h_in)
        final_logits = self.base_model.lm_head(final_h)
        loss_final = F.cross_entropy(final_logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=ignore_index)

        opt_final = optimizers[-1]
        opt_final.zero_grad(set_to_none=False if not is_cuda else True)
        loss_final.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.base_model.lm_head.parameters(), grad_clip)
        opt_final.step()
        opt_final.zero_grad(set_to_none=False if not is_cuda else True)

        if is_cuda and use_async_pipelining:
            for s in self._cuda_streams:
                torch.cuda.current_stream().wait_stream(s)

        return {
            "loss": loss_final.item(),
            "layer_losses": layer_losses,
            "mean_local_loss": sum(layer_losses) / len(layer_losses) if layer_losses else loss_final.item()
        }

    @torch.no_grad()
    def generate(self, *args, **kwargs) -> torch.Tensor:
        """Delegates generation to underlying base model."""
        return self.base_model.generate(*args, **kwargs)
