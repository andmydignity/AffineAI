"""
Muon (Momentumized Orthogonal Newton-Schulz Optimizer)
======================================================
Applies 5th-order Newton-Schulz matrix polar decomposition to 2D weight matrices,
combining Nesterov momentum with spectral orthogonal updates.
Vectors, embeddings, and 1D parameters are optimized via AdamW.
"""

import math
import warnings
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonal polar factor of G.
    Accelerates 2D weight matrix optimization to the theoretical limit of spectral geometry.
    """
    assert len(G.shape) == 2, f"Expected 2D tensor, got shape {G.shape}"
    
    if not G.is_cuda:
        from affine_ai.core.cpp_ops import get_asdag_cpu_ops
        ops = get_asdag_cpu_ops()
        if ops and hasattr(ops, 'newton_schulz5'):
            return ops.newton_schulz5(G, steps, eps)

    a, b, c = (3.4445, -4.7750, 2.0315)
    orig_dtype = G.dtype
    X = G.bfloat16() if (orig_dtype == torch.bfloat16 or (G.is_cuda and torch.cuda.is_bf16_supported())) else G.float()
    X = X / (X.norm() + eps)  # Spectral norm normalization
    
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
        
    m = X.size(0)
    A = torch.empty((m, m), device=X.device, dtype=X.dtype)
    B = torch.empty((m, m), device=X.device, dtype=X.dtype)
    X_buf = torch.empty_like(X)

    for _ in range(steps):
        torch.mm(X, X.T, out=A)
        torch.addmm(A, A, A, beta=b, alpha=c, out=B)
        torch.addmm(X, B, X, beta=a, alpha=1.0, out=X_buf)
        X, X_buf = X_buf, X
        
    if transposed:
        X = X.T
        
    return X.to(orig_dtype)


def zeropower_via_newtonschulz5_batched(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Batched Newton-Schulz iteration for 3D tensor of shape [B, M, N].
    Fuses multiple 2D matrices into batched GEMM (bmm/baddbmm) operations.
    """
    assert len(G.shape) == 3, f"Expected 3D tensor, got shape {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    orig_dtype = G.dtype
    X = G.bfloat16() if (orig_dtype == torch.bfloat16 or (G.is_cuda and torch.cuda.is_bf16_supported())) else G.float()
    norms = torch.linalg.vector_norm(X, dim=(1, 2), keepdim=True)
    X = X / (norms + eps)
    
    transposed = X.size(1) > X.size(2)
    if transposed:
        X = X.transpose(1, 2)
        
    batch_size, m, _ = X.shape
    A = torch.empty((batch_size, m, m), device=X.device, dtype=X.dtype)
    B = torch.empty((batch_size, m, m), device=X.device, dtype=X.dtype)
    X_buf = torch.empty_like(X)

    for _ in range(steps):
        torch.bmm(X, X.transpose(1, 2), out=A)
        torch.baddbmm(A, A, A, beta=b, alpha=c, out=B)
        torch.baddbmm(X, B, X, beta=a, alpha=1.0, out=X_buf)
        X, X_buf = X_buf, X
        
    if transposed:
        X = X.transpose(1, 2)
        
    return X.to(orig_dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer for 2D parameter tensors.
    Applies Newton-Schulz orthogonalization to momentum-filtered gradients.

    capturable: when True, the Newton-Schulz iteration reuses pre-allocated
        scratch buffers and caches the bf16-capability probe, so that
        ``step()`` contains no host queries or data-dependent control flow
        and can be recorded inside a CUDA graph. Requires at least one eager
        warmup step before capture so that momentum and scratch buffers are
        materialized. Numerics are bit-identical to the legacy path.
    """
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        capturable: bool = False,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay, use_muon=True)
        super().__init__(params, defaults)
        self.capturable = capturable
        # Cache for the bf16-capability probe (a driver query, illegal inside
        # graph capture). Keyed by (device.type, device.index); populated on
        # eager steps (warmup) so capture hits the cache.
        self._bf16_cache: Dict[Tuple[str, Optional[int]], bool] = {}
        # Pre-allocated Newton-Schulz scratch buffers, keyed by
        # (shape, work_dtype, device, count). Allocated on first eager use,
        # reused on every step to avoid per-step torch.empty inside capture.
        self._ns_buffers: Dict[Any, Dict[str, torch.Tensor]] = {}

    def _ns_work_dtype(self, G: torch.Tensor, orig_dtype: torch.dtype) -> torch.dtype:
        # Mirrors the legacy condition in zeropower_via_newtonschulz5 without
        # issuing the torch.cuda.is_bf16_supported() driver query on every
        # step (queries are illegal inside graph capture, so the result is
        # cached per device after the first eager step / warmup).
        if orig_dtype == torch.bfloat16:
            return torch.bfloat16
        if G.is_cuda:
            key = (G.device.type, G.device.index)
            v = self._bf16_cache.get(key)
            if v is None:
                v = bool(torch.cuda.is_bf16_supported())
                self._bf16_cache[key] = v
            if v:
                return torch.bfloat16
        return torch.float32

    def _ns_single_static(self, g_can: torch.Tensor, steps: int, eps: float = 1e-7) -> torch.Tensor:
        # Capturable single-matrix Newton-Schulz. g_can must already be in
        # canonical (m <= n) orientation. Same math as the module-level
        # zeropower_via_newtonschulz5 (same coefficients, same op order), but
        # all scratch (X, X_buf, A, B) comes from persistent buffers.
        orig_dtype = g_can.dtype
        wdtype = self._ns_work_dtype(g_can, orig_dtype)
        m, n = g_can.shape
        key = ((m, n), str(wdtype), str(g_can.device))
        st = self._ns_buffers.get(key)
        if st is None:
            st = {
                "X": torch.empty((m, n), device=g_can.device, dtype=wdtype),
                "Xb": torch.empty((m, n), device=g_can.device, dtype=wdtype),
                "A": torch.empty((m, m), device=g_can.device, dtype=wdtype),
                "B": torch.empty((m, m), device=g_can.device, dtype=wdtype),
            }
            self._ns_buffers[key] = st
        X, Xb, A, B = st["X"], st["Xb"], st["A"], st["B"]
        X.copy_(g_can.to(wdtype))
        X.div_(X.norm() + eps)
        a, b, c = (3.4445, -4.7750, 2.0315)
        for _ in range(steps):
            torch.mm(X, X.T, out=A)
            torch.addmm(A, A, A, beta=b, alpha=c, out=B)
            torch.addmm(X, B, X, beta=a, alpha=1.0, out=Xb)
            X, Xb = Xb, X
        return X.to(orig_dtype)

    def _ns_batched_static(self, G_batch: torch.Tensor, steps: int, eps: float = 1e-7) -> torch.Tensor:
        # Capturable batched Newton-Schulz for (count, m, n) canonical input.
        # Same math as zeropower_via_newtonschulz5_batched; scratch persists
        # in self._ns_buffers under a count-qualified key.
        orig_dtype = G_batch.dtype
        wdtype = self._ns_work_dtype(G_batch, orig_dtype)
        count, m, n = G_batch.shape
        key = ((count, m, n), str(wdtype), str(G_batch.device))
        st = self._ns_buffers.get(key)
        if st is None:
            st = {
                "X": torch.empty((count, m, n), device=G_batch.device, dtype=wdtype),
                "Xb": torch.empty((count, m, n), device=G_batch.device, dtype=wdtype),
                "A": torch.empty((count, m, m), device=G_batch.device, dtype=wdtype),
                "B": torch.empty((count, m, m), device=G_batch.device, dtype=wdtype),
            }
            self._ns_buffers[key] = st
        X, Xb, A, B = st["X"], st["Xb"], st["A"], st["B"]
        X.copy_(G_batch.to(wdtype))
        X.div_(torch.linalg.vector_norm(X, dim=(1, 2), keepdim=True) + eps)
        a, b, c = (3.4445, -4.7750, 2.0315)
        for _ in range(steps):
            torch.bmm(X, X.transpose(1, 2), out=A)
            torch.baddbmm(A, A, A, beta=b, alpha=c, out=B)
            torch.baddbmm(X, B, X, beta=a, alpha=1.0, out=Xb)
            X, Xb = Xb, X
        return X.to(orig_dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            weight_decay = group.get('weight_decay', 0.0)

            shape_to_params = {}
            non_2d_params = []

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if weight_decay != 0.0:
                    g = g.add(p, alpha=weight_decay)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                
                update_grad = g.add(buf, alpha=momentum) if nesterov else buf

                if len(p.shape) == 2 and p.shape[0] > 1 and p.shape[1] > 1:
                    orig_shape = p.shape
                    needs_transpose = orig_shape[0] > orig_shape[1]
                    canonical_shape = (min(orig_shape[0], orig_shape[1]), max(orig_shape[0], orig_shape[1]))
                    g_can = update_grad.t().contiguous() if needs_transpose else update_grad
                    key = (canonical_shape, p.device, p.dtype)
                    if key not in shape_to_params:
                        shape_to_params[key] = []
                    shape_to_params[key].append((p, orig_shape, update_grad, g_can, needs_transpose))
                else:
                    non_2d_params.append((p, update_grad))

            for p, update_grad in non_2d_params:
                p.data.add_(update_grad, alpha=-lr)

            for key, items in shape_to_params.items():
                if len(items) == 1:
                    p, orig_shape, update_grad, g_can, needs_transpose = items[0]
                    if self.capturable:
                        upd = self._ns_single_static(g_can, steps=ns_steps)
                        if needs_transpose:
                            upd = upd.t()
                        p.data.add_(upd.view(orig_shape), alpha=-lr)
                    else:
                        g_2d = update_grad.view(orig_shape[0], -1)
                        update = zeropower_via_newtonschulz5(g_2d, steps=ns_steps).view(orig_shape)
                        p.data.add_(update, alpha=-lr)
                else:
                    if self.capturable:
                        G_batch = torch.stack([item[3] for item in items], dim=0)
                        updates_batch = self._ns_batched_static(G_batch, steps=ns_steps)
                    else:
                        G_batch = torch.stack([item[3] for item in items], dim=0)
                        updates_batch = zeropower_via_newtonschulz5_batched(G_batch, steps=ns_steps)
                    for i, (p, orig_shape, _, _, needs_transpose) in enumerate(items):
                        update = updates_batch[i].t() if needs_transpose else updates_batch[i]
                        p.data.add_(update.view(orig_shape), alpha=-lr)

        return loss


class HybridMuonAdamW:
    """
    Unified Hybrid Optimizer combining Muon for 2D matrix weights
    and AdamW for embeddings, norms, and 1D vector parameters.
    """
    def __init__(
        self,
        model: nn.Module,
        muon_lr: float = 0.03,
        adamw_lr: float = 3e-3,
        muon_momentum: float = 0.95,
        adamw_weight_decay: float = 0.01,
        muon_weight_decay: float = 0.0,
        fused: bool = True,
        capturable: bool = False,
    ):
        self.capturable = capturable
        muon_params = []
        adamw_decay_params = []
        adamw_nodecay_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            
            is_2d_matrix = (p.ndim == 2 and p.shape[0] > 1 and p.shape[1] > 1)
            is_special = any(k in name.lower() for k in (
                "diagonals", "diagonal", "embed", "tok_embeddings", "lm_head",
                "bias", "norm", "scale", "decay", "sos_patch", "boundary_predictor", "conv"
            ))

            if is_2d_matrix and not is_special:
                # 2D BitLinear/Linear projection matrices to Muon
                muon_params.append(p)
            else:
                # Embeddings, 1D vectors, norms, biases, convs, scales to AdamW
                if any(k in name.lower() for k in ("bias", "norm", "scale", "decay", "diagonals", "diagonal")):
                    adamw_nodecay_params.append(p)
                else:
                    adamw_decay_params.append(p)

        self.optimizers = []

        if len(muon_params) > 0:
            self.muon_opt = Muon(
                muon_params,
                lr=muon_lr,
                momentum=muon_momentum,
                weight_decay=muon_weight_decay,
                capturable=capturable,
            )
            self.optimizers.append(self.muon_opt)
        else:
            self.muon_opt = None
            warnings.warn(
                "HybridMuonAdamW: no 2D matrix parameters found for Muon; "
                "this optimizer is pure AdamW despite the Muon name. "
                "Check parameter shapes/names if Muon was expected.",
                stacklevel=2,
            )

        adamw_groups = []
        if len(adamw_decay_params) > 0:
            adamw_groups.append({"params": adamw_decay_params, "weight_decay": adamw_weight_decay})
        if len(adamw_nodecay_params) > 0:
            adamw_groups.append({"params": adamw_nodecay_params, "weight_decay": 0.0})

        if len(adamw_groups) > 0:
            is_cuda = next(model.parameters()).is_cuda if list(model.parameters()) else False
            adamw_kwargs: Dict[str, Any] = {"lr": adamw_lr}
            if capturable:
                adamw_kwargs["capturable"] = True
            if is_cuda and fused:
                try:
                    self.adamw_opt = torch.optim.AdamW(adamw_groups, fused=True, **adamw_kwargs)
                except Exception:
                    self.adamw_opt = torch.optim.AdamW(adamw_groups, **adamw_kwargs)
            else:
                self.adamw_opt = torch.optim.AdamW(adamw_groups, **adamw_kwargs)
            self.optimizers.append(self.adamw_opt)
        else:
            self.adamw_opt = None

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        for opt in self.optimizers:
            opt.step()

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def state_dict(self) -> Dict[str, Any]:
        return {
            "muon_opt": self.muon_opt.state_dict() if self.muon_opt else None,
            "adamw_opt": self.adamw_opt.state_dict() if self.adamw_opt else None
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if self.muon_opt and state_dict.get("muon_opt"):
            self.muon_opt.load_state_dict(state_dict["muon_opt"])
        if self.adamw_opt and state_dict.get("adamw_opt"):
            self.adamw_opt.load_state_dict(state_dict["adamw_opt"])
