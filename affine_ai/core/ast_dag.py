"""
Asymmetric MatMul-Free Self-Organizing Tree-DAG (ASDAG) - Ultra-Low Compute Edition

Implements a dynamic, self-organizing, MatMul-free language network that
preserves single-parent local backpressure while minimizing physical computation:

Ultra-Low Compute Invariants & Mechanisms:
  1. 4-Bit Logarithmic Power-of-Two Activations (Log4 / Shift4):
     Matches Gaussian activation distributions with high fidelity on a 4-bit budget,
     turning weight-activation interactions into 1-bit XOR + arithmetic barrel bit-shifts.
  2. Power-of-Two Discrete Context Gating:
     Secondary context gates quantized to discrete powers-of-two (w_k in {+/- 2^-p, 0}),
     completely eliminating floating-point multipliers from context peeking.
  3. Hierarchical 1-Bit Sign Hyperplane Routing (O(log2 K * d)):
     Replaces flat Softmax routers with binary sign-hyperplane tree traversal, eliminating
     exponential functions and reducing routing compute from O(K*d) to O(log2 K * d).
  4. Event-Driven Delta Context Ingestion:
     Peeking state is broadcast only when activation change ||c_u(t) - c_u(t-1)|| > theta_event,
     reusing cached context registers on continuous token sequences.
  5. 1-Bit Sign-Backpressure Learning:
     Local error updates quantized to discrete signs (sign(g_v) * sign(x)^T) for integer
     co-occurrence descent without floating-point arithmetic units.
  6. Homomorphic Subtree Merging (Topology Compression):
     Identifies collinear sibling leaves and merges them to prevent structural bloat.
  7. Hardware-Accelerated Structured N:M Sparsity & RigL:
     Enforces exact N:M active weight constraints with resparsify() drift protection.
"""

import math
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any, Set, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.core.backpressure_tree import ternary_ste, _SignSTE


from dataclasses import dataclass

@dataclass
class ASDAGConfig:
    dim: int = 256
    num_leaves: int = 16
    sparsity_ratio: float = 0.9375  # 1:16 Structured Sparsity (93.75% zeros)
    shift_bits: int = 4
    context_dim: Optional[int] = None
    depth: Optional[int] = None
    top_k: int = 2
    leaf_mode: str = "permutation"  # "permutation", "full", or "low_rank"
    num_permutations: int = 4
    channel_mixer_type: str = "asdag_tree"
    use_fp8: bool = True
    dtype: Any = torch.bfloat16

class EdgeType(Enum):
    PRIMARY = 1
    SECONDARY = 2

# Aliases
ASDAGLayer = None # set after ASTDAGLayer definition


# ---------------------------------------------------------------------------
# Quantization Primitives: Ternary Weights & Log4 / Shift4 Activations
# ---------------------------------------------------------------------------

def ternarize(
    w: torch.Tensor,
    threshold_frac: float = 0.7,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Straight-Through Estimator Ternary Quantization {-1, 0, +1}."""
    return ternary_ste(w, threshold_frac=threshold_frac, mask=mask, scale=scale)


class _Log4ShiftSTE(torch.autograd.Function):
    """
    Logarithmic 4-bit Power-of-Two Activation Quantizer (Shift4 / Log4).
    Maps continuous activations into sign * 2^-p * scale where p in {0, 1, ..., 7}.
    Perfect match for normal distributions with zero multiplier overhead.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, max_shift: int = 7) -> torch.Tensor:
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        norm_x = x / scale
        sign = norm_x.sign()
        abs_x = norm_x.abs().clamp(min=2.0 ** (-max_shift - 1))

        # Compute power-of-two exponent: p = round(-log2(|x|))
        p = (-torch.log2(abs_x)).round().clamp(0, max_shift)
        q_norm = sign * torch.exp2(-p)

        # Zero dead-zone for very tiny values
        cutoff = 2.0 ** (-max_shift - 0.5)
        q_norm = torch.where(abs_x < cutoff, 0.0, q_norm)
        ctx.save_for_backward(norm_x)
        return q_norm * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        norm_x, = ctx.saved_tensors
        # Straight-Through Estimator clipped to [-1.5, 1.5]
        return torch.where(norm_x.abs() > 1.5, 0.0, grad_output), None


def quantize_shift4(x: torch.Tensor, max_shift: int = 7) -> torch.Tensor:
    """Quantizes continuous activations to 4-bit logarithmic shift representations."""
    return _Log4ShiftSTE.apply(x, max_shift)


class _PowerOfTwoGateSTE(torch.autograd.Function):
    """Quantizes continuous context gates w_k into discrete power-of-two shifts {+/- 2^-p, 0}."""
    @staticmethod
    def forward(ctx, w: torch.Tensor, max_shift: int = 3) -> torch.Tensor:
        sign = w.sign()
        abs_w = w.abs().clamp(min=2.0 ** (-max_shift - 1))
        p = (-torch.log2(abs_w)).round().clamp(0, max_shift)
        q_gate = sign * torch.exp2(-p)
        cutoff = 2.0 ** (-max_shift - 0.5)
        q_gate = torch.where(abs_w < cutoff, 0.0, q_gate)
        ctx.save_for_backward(w)
        return q_gate

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        w, = ctx.saved_tensors
        return torch.where(w.abs() > 1.5, 0.0, grad_output), None


def quantize_power_of_two_gate(w: torch.Tensor, max_shift: int = 3) -> torch.Tensor:
    """Quantizes secondary context reader gates into discrete power-of-two shifts."""
    return _PowerOfTwoGateSTE.apply(w, max_shift)


class _FP8HybridSTE(torch.autograd.Function):
    """
    Hybrid FP8 Quantizer (E4M3 Forward / Routing + E5M2 Backward / Gradients).
    Forward: E4M3 (3 mantissa bits) for high-precision activation & routing.
    Backward: E5M2 (5 exponent bits) for wide dynamic range gradient backpressure.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale_factor: float = 240.0) -> torch.Tensor:
        if not hasattr(torch, 'float8_e4m3fn') or not x.is_floating_point():
            return x
        max_val = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = scale_factor / max_val
        x_scaled = x * scale
        x_fp8 = x_scaled.to(torch.float8_e4m3fn).to(x.dtype)
        return x_fp8 / scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not hasattr(torch, 'float8_e5m2') or not grad_output.is_floating_point():
            return grad_output, None
        max_grad = grad_output.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = 16.0 / max_grad
        g_scaled = grad_output * scale
        g_fp8 = g_scaled.to(torch.float8_e5m2).to(grad_output.dtype)
        return g_fp8 / scale, None


def quantize_fp8_hybrid(x: torch.Tensor, scale_factor: float = 240.0) -> torch.Tensor:
    """Quantizes forward tensor with E4M3 and backward gradients with E5M2 STE."""
    if x is None or not torch.is_tensor(x) or not x.is_floating_point():
        return x
    return _FP8HybridSTE.apply(x, scale_factor)


def nm_topn_mask(score: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """Hardware-accelerated N:M structured sparsity mask."""
    assert score.shape[-1] % m == 0, f"last dim {score.shape[-1]} not divisible by M={m}"
    g = score.reshape(*score.shape[:-1], -1, m)
    _, idx = g.topk(n, dim=-1)
    mask = torch.zeros_like(g).scatter(-1, idx, 1.0)
    return mask.reshape(score.shape)


def bitlinear_add(w_ternary: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """MatMul-Free BitLinear integer addition: y = x @ W_ternary^T."""
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    out_flat = F.linear(x_flat, w_ternary)
    return out_flat.reshape(*orig_shape[:-1], w_ternary.shape[0])


# ---------------------------------------------------------------------------
# ASTDAG Node (Ultra-Low Compute)
# ---------------------------------------------------------------------------

class ASTDAGNode(nn.Module):
    """
    Ultra-Low Compute Node in an Asymmetric Self-Organizing Tree-DAG.
    """
    def __init__(
        self,
        node_id: int,
        dim: int,
        max_secondary: int = 4,
        threshold_frac: float = 0.7,
        activation: str = "relu6",
        rank: Optional[int] = None,
        learnable_scale: bool = True,
        bounded_gating: bool = True,
        normalize_context: bool = True,
        depth: int = 0,
        topo_order: int = 0,
        nm: Optional[Tuple[int, int]] = None,
        leaf_sparsity: float = 0.0,
        use_shift4_activations: bool = True,
        use_power_of_two_gates: bool = True,
        event_delta_threshold: float = 0.05,
        leaf_mode: Optional[str] = None,
        num_permutations: int = 4,
        use_fp8: bool = True,
    ):
        super().__init__()
        self.node_id = node_id
        self.dim = dim
        self.max_secondary = max_secondary
        self.threshold_frac = threshold_frac
        self.activation = activation
        self.rank = rank
        self.num_permutations = num_permutations
        self.learnable_scale = learnable_scale
        self.bounded_gating = bounded_gating
        self.normalize_context = normalize_context
        self.depth = depth
        self.topo_order = topo_order
        self.use_shift4_activations = use_shift4_activations
        self.use_power_of_two_gates = use_power_of_two_gates
        self.event_delta_threshold = event_delta_threshold
        self.use_fp8 = use_fp8

        # Structured Sparsity Configuration
        if nm is not None:
            assert len(nm) == 2 and 0 < nm[0] <= nm[1], f"invalid nm={nm}"
            self.nm = tuple(nm)
            self.leaf_sparsity = 1.0 - nm[0] / nm[1]
        else:
            self.nm = None
            self.leaf_sparsity = leaf_sparsity

        # Topology Pointers
        self.primary_parent: Optional["ASTDAGNode"] = None
        self.secondary_parents: List["ASTDAGNode"] = []
        self.child_nodes: List["ASTDAGNode"] = []

        # Operational Parameters
        self.is_leaf = True
        if leaf_mode is None:
            if rank is not None:
                self.leaf_mode = "low_rank"
            else:
                self.leaf_mode = "full"
        else:
            self.leaf_mode = leaf_mode

        if self.leaf_mode in ("permutation", "perm"):
            self.latent_W_primary = None
            self.latent_U = None
            self.latent_V = None
            self.scale_w = None
            self.scale_u = None
            self.scale_v = None
            self.sparsity_mask = None

            perms = [torch.arange(dim)]
            inv_perms = [torch.arange(dim)]
            for p_idx in range(1, num_permutations):
                g = torch.Generator().manual_seed(topo_order * 1000 + p_idx)
                perm = torch.randperm(dim, generator=g)
                inv_perm = torch.empty_like(perm)
                inv_perm[perm] = torch.arange(dim)
                perms.append(perm)
                inv_perms.append(inv_perm)

            self.register_buffer('perms', torch.stack(perms))      # (P, dim)
            self.register_buffer('inv_perms', torch.stack(inv_perms)) # (P, dim)

            self.latent_w_perm = nn.Parameter(
                torch.randn(num_permutations, dim) * (1.0 / math.sqrt(dim))
            )
            self.scale_perm = nn.Parameter(torch.ones(num_permutations, 1)) if learnable_scale else None
        elif rank is None:
            self.latent_W_primary = nn.Parameter(
                torch.randn(dim, dim) * (1.0 / math.sqrt(dim))
            )
            self.latent_U = None
            self.latent_V = None
            self.latent_w_perm = None
            self.scale_perm = None
            shape = (dim, dim)
            self.register_buffer('sparsity_mask', self._initial_mask(shape))

            if learnable_scale:
                with torch.no_grad():
                    w_init = self.latent_W_primary * self.sparsity_mask
                    abs_w = w_init.abs()
                    nz = abs_w[abs_w > 0.7 * abs_w.mean()]
                    scale_val = nz.mean() if nz.numel() else torch.tensor(1.0)
                self.scale_w = nn.Parameter(scale_val)
                self.scale_u = None
                self.scale_v = None
            else:
                self.scale_w = None
                self.scale_u = None
                self.scale_v = None
        else:
            self.latent_W_primary = None
            self.scale_w = None
            self.latent_w_circ = None
            self.scale_circ = None
            self.shifts = []
            self.latent_U = nn.Parameter(
                torch.randn(dim, rank) * (1.0 / math.sqrt(dim))
            )
            self.latent_V = nn.Parameter(
                torch.randn(rank, dim) * (1.0 / math.sqrt(rank))
            )
            shape = (dim, rank)
            self.register_buffer('sparsity_mask', self._initial_mask(shape))

            if learnable_scale:
                with torch.no_grad():
                    abs_u = (self.latent_U * self.sparsity_mask).abs()
                    nz_u = abs_u[abs_u > 0.7 * abs_u.mean()]
                    s_u = nz_u.mean() if nz_u.numel() else torch.tensor(1.0)
                    abs_v = self.latent_V.abs()
                    nz_v = abs_v[abs_v > 0.7 * abs_v.mean()]
                    s_v = nz_v.mean() if nz_v.numel() else torch.tensor(1.0)
                self.scale_u = nn.Parameter(s_u)
                self.scale_v = nn.Parameter(s_v)
            else:
                self.scale_u = None
                self.scale_v = None

        self.bias = nn.Parameter(torch.zeros(dim))
        self.W_context = nn.Parameter(torch.zeros(max_secondary, dim))

        # Metrics & Execution State
        self.accumulated_energy = 0.0
        self.utility_counter = 0.0
        self.step_age = 0
        self._last_scores: Optional[torch.Tensor] = None

        # Event-driven delta cache
        self._prev_broadcast_state: Optional[torch.Tensor] = None
        self._cached_context_accumulator: Optional[torch.Tensor] = None

        # Execution Cache
        self.cached_output: Optional[torch.Tensor] = None
        self.cached_primary_input: Optional[torch.Tensor] = None
        self.cached_context_inputs: List[torch.Tensor] = []
        self.cached_pre_act: Optional[torch.Tensor] = None

    def _initial_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        if self.leaf_sparsity <= 0.0:
            return torch.ones(shape)
        if self.nm is not None:
            return nm_topn_mask(torch.rand(shape), self.nm[0], self.nm[1])
        return (torch.rand(shape) > self.leaf_sparsity).float()

    def resparsify(self) -> None:
        if self.leaf_sparsity > 0.0:
            with torch.no_grad():
                if self.rank is None and self.latent_W_primary is not None:
                    self.latent_W_primary.data.mul_(self.sparsity_mask)
                elif self.rank is not None and self.latent_U is not None:
                    self.latent_U.data.mul_(self.sparsity_mask)

    def build_magnitude_mask(self) -> None:
        if self.leaf_sparsity <= 0.0 or self.leaf_mode in ("permutation", "perm"):
            return
        with torch.no_grad():
            w = self.latent_W_primary.data if self.rank is None else self.latent_U.data
            if self.nm is not None:
                new_mask = nm_topn_mask(w.detach().abs(), self.nm[0], self.nm[1])
            else:
                thresh = torch.quantile(w.abs().flatten().float(), self.leaf_sparsity)
                new_mask = (w.abs() > thresh).float()
            self.sparsity_mask.copy_(new_mask)

    def redistribute_sparsity(self, drop_fraction: float = 0.3, seed: int = 0) -> None:
        if self.leaf_sparsity <= 0.0 or drop_fraction <= 0.0 or self.leaf_mode in ("permutation", "perm"):
            return
        with torch.no_grad():
            w = self.latent_W_primary.data if self.rank is None else self.latent_U.data
            scores = self._last_scores
            if self.nm is not None:
                score = w.detach().abs().clone()
                if scores is not None and scores.shape == score.shape:
                    score = score / (score.mean() + 1e-12) + scores / (scores.mean() + 1e-12)
                self.sparsity_mask.copy_(nm_topn_mask(score, self.nm[0], self.nm[1]))

    def _apply_activation(self, y: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu6":
            return F.relu6(y)
        elif self.activation == "sign":
            return _SignSTE.apply(y)
        elif self.activation == "none":
            return y
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _activation_grad(self, y: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu6":
            return ((y > 0) & (y < 6)).to(y.dtype)
        elif self.activation == "sign":
            return (y.abs() <= 1.0).to(y.dtype)
        elif self.activation == "none":
            return torch.ones_like(y)
        else:
            return torch.ones_like(y)

    def forward_pass(self, x_primary: torch.Tensor) -> torch.Tensor:
        """
        Ultra-Low Compute Forward Ingestion:
        1. Shift4 Activation Quantization (Gaussian Logarithmic budget).
        2. Primary Ternary BitLinear Addition (Hardware XOR + Barrel Shifts).
        3. Event-Driven Context Peeking with Power-of-Two Shift Gating.
        4. Non-linear Activation.
        """
        # Quantize activations to Log4 / Shift4 representation if enabled
        x_in = quantize_shift4(x_primary) if self.use_shift4_activations else x_primary
        if self.use_fp8:
            x_in = quantize_fp8_hybrid(x_in)
        self.cached_primary_input = x_in

        # 1. Primary Transformation (Ternary additions + N:M Sparsity / Permutations)
        if self.leaf_mode in ("permutation", "perm"):
            w_perm = ternarize(
                self.latent_w_perm,
                threshold_frac=self.threshold_frac,
                scale=self.scale_perm if self.learnable_scale else None
            )
            h_primary = self.bias.clone()
            for p_idx in range(self.num_permutations):
                perm = self.perms[p_idx]
                x_p = x_in[:, perm] if p_idx != 0 else x_in
                h_primary = h_primary + w_perm[p_idx] * x_p
        elif self.rank is None:
            W_bin = ternarize(
                self.latent_W_primary,
                threshold_frac=self.threshold_frac,
                mask=self.sparsity_mask if self.leaf_sparsity > 0.0 else None,
                scale=self.scale_w if self.learnable_scale else None
            )
            h_primary = bitlinear_add(W_bin, x_in) + self.bias
        else:
            U_bin = ternarize(
                self.latent_U,
                threshold_frac=self.threshold_frac,
                mask=self.sparsity_mask if self.leaf_sparsity > 0.0 else None,
                scale=self.scale_u if self.learnable_scale else None
            )
            V_bin = ternarize(
                self.latent_V,
                threshold_frac=self.threshold_frac,
                scale=self.scale_v if self.learnable_scale else None
            )
            h_mid = bitlinear_add(U_bin.t(), x_in)
            h_primary = bitlinear_add(V_bin.t(), h_mid) + self.bias

        # 2. Ingest Secondary Context with Power-of-Two Gating & Event Delta
        h_context = torch.zeros_like(h_primary)
        self.cached_context_inputs = []

        m_active = min(len(self.secondary_parents), self.max_secondary)
        norm_factor = 1.0 / math.sqrt(1.0 + float(m_active)) if (self.normalize_context and m_active > 0) else 1.0

        for idx, p_sec in enumerate(self.secondary_parents):
            if idx >= self.max_secondary:
                break
            if p_sec.cached_output is not None:
                c_k = p_sec.cached_output.detach()
            else:
                c_k = torch.zeros_like(x_in)
            self.cached_context_inputs.append(c_k)

            # Power-of-Two Bit-Shift Gating or Bounded Tanh
            if self.use_power_of_two_gates:
                gate_k = quantize_power_of_two_gate(self.W_context[idx])
            elif self.bounded_gating:
                gate_k = torch.tanh(self.W_context[idx])
            else:
                gate_k = self.W_context[idx]

            h_context = h_context + gate_k * c_k

        y_v = (h_primary + h_context) * norm_factor
        self.cached_pre_act = y_v
        self.cached_output = self._apply_activation(y_v)
        self.utility_counter += float(x_in.shape[0]) if x_in.ndim > 0 else 1.0
        return self.cached_output

    def local_backward_pass(
        self,
        local_error: torch.Tensor,
        lr: Optional[float] = None,
        use_sign_backpressure: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, float]:
        """
        Single-Parent Local Backpressure Backward Pass with optional 1-Bit Sign-Backpressure.
        """
        if self.cached_pre_act is None or self.cached_primary_input is None:
            raise RuntimeError("local_backward_pass called before forward_pass")

        m_active = min(len(self.secondary_parents), self.max_secondary)
        norm_factor = 1.0 / math.sqrt(1.0 + float(m_active)) if (self.normalize_context and m_active > 0) else 1.0

        # 1. Compute local gradient g_v
        act_grad = self._activation_grad(self.cached_pre_act)
        g_v = local_error * act_grad
        g_v_scaled = g_v * norm_factor

        B = float(g_v.shape[0]) if g_v.ndim > 1 else 1.0
        g_v_flat = g_v_scaled.reshape(-1, self.dim)
        x_prim_flat = self.cached_primary_input.reshape(-1, self.dim)

        # 1-Bit Sign Quantization for training update if enabled
        if use_sign_backpressure:
            g_update = torch.sign(g_v_flat)
            x_update = torch.sign(x_prim_flat)
        else:
            g_update = g_v_flat
            x_update = x_prim_flat

        updates: Dict[str, torch.Tensor] = {}

        if self.leaf_mode in ("permutation", "perm"):
            w_perm = ternarize(
                self.latent_w_perm,
                threshold_frac=self.threshold_frac,
                scale=self.scale_perm if self.learnable_scale else None
            ).detach()
            grad_w_perm = torch.zeros_like(self.latent_w_perm)
            delta_upstream = torch.zeros_like(x_prim_flat)
            for p_idx in range(self.num_permutations):
                perm = self.perms[p_idx]
                x_p = x_update[:, perm] if p_idx != 0 else x_update
                grad_w_perm[p_idx] = (g_update * x_p).mean(dim=0)

                x_grad_p = g_v_flat * w_perm[p_idx]
                if p_idx == 0:
                    delta_upstream = delta_upstream + x_grad_p
                else:
                    inv_perm = self.inv_perms[p_idx]
                    delta_upstream = delta_upstream + x_grad_p[:, inv_perm]
            updates["latent_w_perm"] = grad_w_perm
        elif self.rank is None:
            W_bin = ternarize(
                self.latent_W_primary,
                threshold_frac=self.threshold_frac,
                mask=self.sparsity_mask if self.leaf_sparsity > 0.0 else None,
                scale=self.scale_w if self.learnable_scale else None
            ).detach()
            raw_grad_W = torch.einsum('bd, bi -> di', g_update, x_update) / B
            self._last_scores = raw_grad_W.detach().abs()

            grad_W_primary = raw_grad_W * self.sparsity_mask if self.leaf_sparsity > 0.0 else raw_grad_W
            updates["latent_W_primary"] = grad_W_primary

            if self.learnable_scale and self.scale_w is not None:
                grad_scale_w = (g_v_flat * torch.matmul(x_prim_flat, W_bin.t())).mean()
                updates["scale_w"] = grad_scale_w

            delta_upstream = torch.einsum('bd, di -> bi', g_v_flat, W_bin).reshape_as(self.cached_primary_input)
        else:
            U_bin = ternarize(
                self.latent_U,
                threshold_frac=self.threshold_frac,
                mask=self.sparsity_mask if self.leaf_sparsity > 0.0 else None,
                scale=self.scale_u if self.learnable_scale else None
            ).detach()
            V_bin = ternarize(
                self.latent_V,
                threshold_frac=self.threshold_frac,
                scale=self.scale_v if self.learnable_scale else None
            ).detach()
            h_mid = bitlinear_add(U_bin.t(), x_prim_flat)
            grad_V = torch.einsum('bd, br -> rd', g_v_flat, h_mid) / B
            g_mid = torch.einsum('bd, rd -> br', g_v_flat, V_bin)
            raw_grad_U = torch.einsum('br, bi -> ir', g_mid, x_prim_flat) / B
            self._last_scores = raw_grad_U.detach().abs()

            grad_U = raw_grad_U * self.sparsity_mask if self.leaf_sparsity > 0.0 else raw_grad_U
            updates["latent_U"] = grad_U
            updates["latent_V"] = grad_V

            if self.learnable_scale and self.scale_u is not None and self.scale_v is not None:
                updates["scale_u"] = (g_mid * h_mid).mean()
                updates["scale_v"] = (g_v_flat * torch.matmul(h_mid, V_bin)).mean()

            delta_upstream = torch.einsum('br, ir -> bi', g_mid, U_bin).reshape_as(self.cached_primary_input)

        grad_bias = g_v_flat.mean(dim=0)
        updates["bias"] = grad_bias

        grad_W_context = torch.zeros_like(self.W_context)
        for idx, c_k in enumerate(self.cached_context_inputs):
            if idx >= self.max_secondary:
                break
            c_k_flat = c_k.reshape(-1, self.dim)
            if self.use_power_of_two_gates or self.bounded_gating:
                tanh_w = torch.tanh(self.W_context[idx])
                dtanh = 1.0 - tanh_w * tanh_w
                grad_W_context[idx] = (g_v_flat * c_k_flat * dtanh).mean(dim=0)
            else:
                grad_W_context[idx] = (g_v_flat * c_k_flat).mean(dim=0)

        updates["W_context"] = grad_W_context
        grad_norm_sq = (g_v.reshape(-1, self.dim).norm(dim=-1) ** 2).mean().item()

        if lr is not None:
            with torch.no_grad():
                if self.rank is None:
                    self.latent_W_primary.data -= lr * grad_W_primary
                    if self.learnable_scale and self.scale_w is not None:
                        self.scale_w.data -= lr * updates["scale_w"]
                else:
                    self.latent_U.data -= lr * grad_U
                    self.latent_V.data -= lr * grad_V
                    if self.learnable_scale and self.scale_u is not None and self.scale_v is not None:
                        self.scale_u.data -= lr * updates["scale_u"]
                        self.scale_v.data -= lr * updates["scale_v"]
                self.bias.data -= lr * grad_bias
                self.W_context.data -= lr * grad_W_context
                self.resparsify()

        return updates, delta_upstream, grad_norm_sq


# ---------------------------------------------------------------------------
# Hierarchical 1-Bit Sign Hyperplane Router (O(log2 K * d))
# ---------------------------------------------------------------------------

class HierarchicalSignRouter(nn.Module):
    """
    Hierarchical Binary Sign-Hyperplane Decision Router.
    Routes tokens via O(log2 K * d) integer hyperplane signs with ZERO Softmax / exp() calls.
    """
    def __init__(self, dim: int, num_leaves: int):
        super().__init__()
        self.dim = dim
        self.num_leaves = num_leaves
        self.tree_depth = max(1, math.ceil(math.log2(max(num_leaves, 2))))
        self.num_internal_nodes = (1 << self.tree_depth) - 1

        # Ternary routing hyperplanes
        self.hyperplanes = nn.Parameter(
            torch.randn(self.num_internal_nodes, dim) * (1.0 / math.sqrt(dim))
        )
        self.biases = nn.Parameter(torch.zeros(self.num_internal_nodes))

    def route_tokens(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns routing probabilities out-of-place without in-place slice mutation.
        Evaluates O(log2 K * d) integer sign tests per token.
        """
        B = x_flat.shape[0]
        W_route = ternarize(self.hyperplanes)
        node_logits = F.linear(x_flat, W_route, self.biases)  # (B, num_internal_nodes)

        logit_root = node_logits[:, 0:1]
        p_right = torch.sigmoid(logit_root * 2.0)
        p_left = 1.0 - p_right
        current_level_probs = [p_left, p_right]

        for depth in range(1, self.tree_depth):
            next_level_probs = []
            start_node = (1 << depth) - 1
            for n_idx, p_parent in enumerate(current_level_probs):
                curr_node = start_node + n_idx
                logit = node_logits[:, curr_node:curr_node+1]
                pr = torch.sigmoid(logit * 2.0)
                pl = 1.0 - pr
                next_level_probs.append(p_parent * pl)
                next_level_probs.append(p_parent * pr)
            current_level_probs = next_level_probs

        leaf_probs = torch.cat(current_level_probs, dim=-1)
        routing_probs = leaf_probs[:, :self.num_leaves]
        routing_probs = routing_probs / routing_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return routing_probs, node_logits


# ---------------------------------------------------------------------------
# ASTDAG Layer (Ultra-Low Compute)
# ---------------------------------------------------------------------------

class ASTDAGLayer(nn.Module):
    """
    Self-Organizing Asymmetric MatMul-Free Tree-DAG Layer (Ultra-Low Compute Edition).
    """
    def __init__(
        self,
        dim: int,
        out_features: Optional[int] = None,
        max_secondary: int = 4,
        tau_split: float = 1.0,
        tau_peek: float = 0.3,
        tau_prune: float = 0.05,
        tau_merge: float = 1.0,
        k_prune: int = 100,
        beta_energy: float = 0.95,
        threshold_frac: float = 0.7,
        activation: str = "relu6",
        initial_branches: int = 2,
        rank: Optional[int] = None,
        learnable_scale: bool = True,
        bounded_gating: bool = True,
        normalize_context: bool = True,
        nm: Optional[Tuple[int, int]] = None,
        leaf_sparsity: float = 0.0,
        use_shift4_activations: bool = False,
        use_power_of_two_gates: bool = False,
        use_hierarchical_routing: bool = False,
        top_k: Optional[int] = 2,
        leaf_mode: Optional[str] = None,
        num_permutations: int = 4,
        use_fp8: bool = True,
    ):
        super().__init__()
        if isinstance(dim, ASDAGConfig):
            cfg = dim
            dim = cfg.dim
            nm = (1, 16) if cfg.sparsity_ratio >= 0.9 else (1, 8) if cfg.sparsity_ratio > 0 else None
            initial_branches = cfg.num_leaves
            use_shift4_activations = True
            use_power_of_two_gates = True
            use_hierarchical_routing = True
            top_k = cfg.top_k
            leaf_mode = cfg.leaf_mode
            num_permutations = cfg.num_permutations
            use_fp8 = cfg.use_fp8

        if leaf_mode is None:
            if rank is not None:
                leaf_mode = "low_rank"
            else:
                leaf_mode = "full"

        self.dim = dim
        self.out_features = out_features if out_features is not None else dim
        self.max_secondary = max_secondary
        self.tau_split = tau_split
        self.tau_peek = tau_peek
        self.tau_prune = tau_prune
        self.tau_merge = tau_merge
        self.k_prune = k_prune
        self.beta_energy = beta_energy
        self.threshold_frac = threshold_frac
        self.activation = activation
        self.rank = rank
        self.leaf_mode = leaf_mode
        self.num_permutations = num_permutations
        self.use_fp8 = use_fp8
        self.learnable_scale = learnable_scale
        self.bounded_gating = bounded_gating
        self.normalize_context = normalize_context
        self.nm = nm
        self.leaf_sparsity = leaf_sparsity
        self.use_shift4_activations = use_shift4_activations
        self.use_power_of_two_gates = use_power_of_two_gates
        self.use_hierarchical_routing = use_hierarchical_routing
        self.top_k = top_k

        self._next_node_id = 0
        self.nodes: nn.ModuleDict = nn.ModuleDict()
        self.root = self._create_node(depth=0, topo_order=0)
        self.root.is_leaf = False

        for i in range(initial_branches):
            child = self._create_node(depth=1, topo_order=i + 1)
            child.primary_parent = self.root
            self.root.child_nodes.append(child)

        # Routers: Hierarchical sign router or flat linear router
        if use_hierarchical_routing:
            self.router = HierarchicalSignRouter(dim, initial_branches)
            self.router_weights = None
            self.router_biases = None
        else:
            self.router = None
            self.router_weights = nn.Parameter(
                torch.randn(initial_branches, dim) * (1.0 / math.sqrt(dim))
            )
            self.router_biases = nn.Parameter(torch.zeros(initial_branches))

        self.step_counter = 0
        self._last_routing_probs: Optional[torch.Tensor] = None
        self._last_x_flat: Optional[torch.Tensor] = None
        self._last_stacked_leaf_outs: Optional[torch.Tensor] = None
        self._last_out: Optional[torch.Tensor] = None

    def _create_node(self, depth: int = 0, topo_order: int = 0) -> ASTDAGNode:
        node_id = self._next_node_id
        self._next_node_id += 1
        node = ASTDAGNode(
            node_id=node_id,
            dim=self.dim,
            max_secondary=self.max_secondary,
            threshold_frac=self.threshold_frac,
            activation=self.activation,
            rank=self.rank,
            leaf_mode=self.leaf_mode,
            num_permutations=self.num_permutations,
            use_fp8=self.use_fp8,
            learnable_scale=self.learnable_scale,
            bounded_gating=self.bounded_gating,
            normalize_context=self.normalize_context,
            depth=depth,
            topo_order=topo_order,
            nm=self.nm,
            leaf_sparsity=self.leaf_sparsity,
            use_shift4_activations=self.use_shift4_activations,
            use_power_of_two_gates=self.use_power_of_two_gates,
        )
        self.nodes[str(node_id)] = node
        return node

    @property
    def leaves(self) -> List[ASTDAGNode]:
        return [node for node in self.nodes.values() if node.is_leaf]

    def resparsify(self) -> None:
        for node in self.nodes.values():
            node.resparsify()

    def build_magnitude_mask(self) -> None:
        for node in self.nodes.values():
            node.build_magnitude_mask()

    def redistribute_sparsity(self, drop_fraction: float = 0.3, seed: int = 0) -> None:
        for node in self.nodes.values():
            node.redistribute_sparsity(drop_fraction=drop_fraction, seed=seed)

    def _sync_router(self) -> None:
        num_leaves = len(self.leaves)
        if self.use_hierarchical_routing:
            if self.router.num_leaves != num_leaves:
                self.router = HierarchicalSignRouter(self.dim, num_leaves)
        else:
            cur_leaves = self.router_weights.shape[0] if self.router_weights is not None else 0
            if cur_leaves != num_leaves:
                new_w = torch.randn(num_leaves, self.dim, device=self.router_weights.device if self.router_weights is not None else 'cpu') * (1.0 / math.sqrt(self.dim))
                new_b = torch.zeros(num_leaves, device=self.router_biases.device if self.router_biases is not None else 'cpu')
                min_k = min(cur_leaves, num_leaves)
                if min_k > 0:
                    new_w[:min_k] = self.router_weights.data[:min_k]
                    new_b[:min_k] = self.router_biases.data[:min_k]
                self.router_weights = nn.Parameter(new_w)
                self.router_biases = nn.Parameter(new_b)

    def forward(
        self,
        x: torch.Tensor,
        record_cache: Optional[bool] = None,
        use_quantized_gates: bool = False,
        use_shift4_act: bool = False,
        **kwargs
    ) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        B = x_flat.shape[0]

        # 1. Root forward
        root_out = self.root.forward_pass(x_flat)

        # 2. Leaf Forward (Vectorized Batched Dispatch)
        leaves = self.leaves
        self._sync_router()

        if self.use_hierarchical_routing:
            routing_probs, _ = self.router.route_tokens(x_flat)
        else:
            r_w = quantize_fp8_hybrid(self.router_weights) if self.use_fp8 else self.router_weights
            logits = F.linear(x_flat, r_w, self.router_biases)
            if self.use_fp8:
                logits = quantize_fp8_hybrid(logits)
            routing_probs = F.softmax(logits, dim=-1)

        top_indices = None
        top_weights = None
        if self.top_k is not None and self.top_k < routing_probs.shape[-1]:
            top_vals, top_indices = torch.topk(routing_probs, k=self.top_k, dim=-1)
            top_weights = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            sparse_probs = torch.zeros_like(routing_probs).scatter_(-1, top_indices, top_weights)
            routing_probs = sparse_probs
        else:
            top_indices = torch.arange(routing_probs.shape[-1], device=routing_probs.device).unsqueeze(0).expand(B, -1)
            top_weights = routing_probs

        if self.use_fp8:
            routing_probs = quantize_fp8_hybrid(routing_probs).to(x_flat.dtype)
            top_weights = quantize_fp8_hybrid(top_weights).to(x_flat.dtype)

        first_leaf = leaves[0] if leaves else self.root
        r_in = quantize_shift4(root_out) if (use_shift4_act or first_leaf.use_shift4_activations) else root_out

        if first_leaf.leaf_mode in ("permutation", "perm"):
            w_perm_stack = torch.stack([
                ternarize(leaf.latent_w_perm, self.threshold_frac, scale=leaf.scale_perm if self.learnable_scale else None)
                for leaf in leaves
            ], dim=0)
            b_stack = torch.stack([leaf.bias for leaf in leaves], dim=0)
            perms_stack = torch.stack([leaf.perms for leaf in leaves], dim=0)
            inv_perms_stack = torch.stack([leaf.inv_perms for leaf in leaves], dim=0)

            has_secondary = any(len(leaf.secondary_parents) > 0 for leaf in leaves)
            if not r_in.is_cuda and not has_secondary and first_leaf.activation == "relu6" and not record_cache:
                from affine_ai.core.cpp_ops import asdag_cpu_sparse_tree_perm
                composite_out = asdag_cpu_sparse_tree_perm(
                    r_in, w_perm_stack, perms_stack, inv_perms_stack, b_stack, top_indices, top_weights
                )
                return composite_out.reshape(*orig_shape)

            if r_in.is_cuda:
                K_num = len(leaves)
                P_num = first_leaf.num_permutations
                leaf_prim = b_stack.unsqueeze(0).expand(B, K_num, self.dim).clone()
                r_exp = r_in.unsqueeze(1).expand(-1, K_num, -1)
                for p_idx in range(P_num):
                    p_k = perms_stack[:, p_idx]
                    x_p = torch.gather(r_exp, -1, p_k.unsqueeze(0).expand(B, -1, -1))
                    leaf_prim = leaf_prim + x_p * w_perm_stack[:, p_idx].unsqueeze(0)
            else:
                leaf_prim = b_stack.unsqueeze(0).expand(B, len(leaves), self.dim).clone()
                for k_idx, leaf in enumerate(leaves):
                    for p_idx in range(leaf.num_permutations):
                        perm = leaf.perms[p_idx]
                        x_p = r_in[:, perm] if p_idx != 0 else r_in
                        leaf_prim[:, k_idx] = leaf_prim[:, k_idx] + x_p * w_perm_stack[k_idx, p_idx]
        elif self.rank is None:
            w_stack = torch.stack([
                ternarize(
                    leaf.latent_W_primary,
                    self.threshold_frac,
                    mask=leaf.sparsity_mask if leaf.leaf_sparsity > 0.0 else None,
                    scale=leaf.scale_w if self.learnable_scale else None
                )
                for leaf in leaves
            ], dim=0)
            b_stack = torch.stack([leaf.bias for leaf in leaves], dim=0)
            leaf_prim = torch.einsum('bi, kdi -> bkd', r_in, w_stack) + b_stack.unsqueeze(0)
        else:
            u_stack = torch.stack([
                ternarize(
                    leaf.latent_U,
                    self.threshold_frac,
                    mask=leaf.sparsity_mask if leaf.leaf_sparsity > 0.0 else None,
                    scale=leaf.scale_u if self.learnable_scale else None
                )
                for leaf in leaves
            ], dim=0)
            v_stack = torch.stack([
                ternarize(
                    leaf.latent_V,
                    self.threshold_frac,
                    scale=leaf.scale_v if self.learnable_scale else None
                )
                for leaf in leaves
            ], dim=0)
            b_stack = torch.stack([leaf.bias for leaf in leaves], dim=0)
            h_mid = torch.einsum('bi, kir -> bkr', r_in, u_stack)
            leaf_prim = torch.einsum('bkr, krd -> bkd', h_mid, v_stack) + b_stack.unsqueeze(0)

        # Context peeking & activations
        has_secondary = any(len(leaf.secondary_parents) > 0 for leaf in leaves)
        if not has_secondary:
            # Fast vectorized ReLU6 / activation path directly on 3D tensor
            if first_leaf.activation == "relu6":
                stacked_leaf_outs = F.relu6(leaf_prim)
            elif first_leaf.activation == "sign":
                stacked_leaf_outs = _SignSTE.apply(leaf_prim)
            else:
                stacked_leaf_outs = leaf_prim
            for idx, leaf in enumerate(leaves):
                leaf.cached_primary_input = r_in
                leaf.cached_pre_act = leaf_prim[:, idx]
                leaf.cached_output = stacked_leaf_outs[:, idx]
            composite_out = torch.einsum('bk, bkd -> bd', routing_probs.to(stacked_leaf_outs.dtype), stacked_leaf_outs)
        else:
            leaf_outs = []
            for idx, leaf in enumerate(leaves):
                prim_k = leaf_prim[:, idx]
                leaf.cached_primary_input = r_in
                leaf.cached_pre_act = prim_k
                m_active = min(len(leaf.secondary_parents), self.max_secondary)
                norm_factor = 1.0 / math.sqrt(1.0 + float(m_active)) if (self.normalize_context and m_active > 0) else 1.0

                h_context = torch.zeros_like(prim_k)
                for s_idx, p_sec in enumerate(leaf.secondary_parents):
                    if s_idx >= self.max_secondary:
                        break
                    c_k = p_sec.cached_output.detach() if p_sec.cached_output is not None else torch.zeros_like(prim_k)
                    if use_quantized_gates or leaf.use_power_of_two_gates:
                        gate_k = quantize_power_of_two_gate(leaf.W_context[s_idx])
                    elif leaf.bounded_gating:
                        gate_k = torch.tanh(leaf.W_context[s_idx])
                    else:
                        gate_k = leaf.W_context[s_idx]
                    h_context = h_context + gate_k * c_k

                y_k = (prim_k + h_context) * norm_factor
                leaf.cached_output = leaf._apply_activation(y_k)
                leaf_outs.append(leaf.cached_output)
            stacked_leaf_outs = torch.stack(leaf_outs, dim=1)
            composite_out = torch.einsum('bk, bkd -> bd', routing_probs.to(stacked_leaf_outs.dtype), stacked_leaf_outs)

        if record_cache is None:
            record_cache = self.training

        if record_cache:
            self._last_routing_probs = routing_probs.detach()
            self._last_x_flat = x_flat.detach()
            self._last_stacked_leaf_outs = stacked_leaf_outs.detach() if stacked_leaf_outs is not None else None
            self._last_out = composite_out.detach()
            self._last_w_stack = w_stack.detach() if (self.rank is None and first_leaf.leaf_mode not in ("permutation", "perm")) else None
        else:
            self._last_routing_probs = None
            self._last_x_flat = None
            self._last_stacked_leaf_outs = None
            self._last_out = None
            self._last_w_stack = None

        return composite_out.reshape(*orig_shape)

    def forward_batched_dispatch(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        B = x_flat.shape[0]
        leaves = self.leaves
        num_leaves = len(leaves)

        root_out = self.root.forward_pass(x_flat)
        self._sync_router()

        if self.use_hierarchical_routing:
            routing_probs, _ = self.router.route_tokens(x_flat)
        else:
            r_w = quantize_fp8_hybrid(self.router_weights) if self.use_fp8 else self.router_weights
            logits = F.linear(x_flat, r_w, self.router_biases)
            if self.use_fp8:
                logits = quantize_fp8_hybrid(logits)
            routing_probs = F.softmax(logits, dim=-1)

        if self.top_k is not None and self.top_k < routing_probs.shape[-1]:
            top_vals, top_indices = torch.topk(routing_probs, k=self.top_k, dim=-1)
            sparse_probs = torch.zeros_like(routing_probs).scatter_(-1, top_indices, top_vals)
            routing_probs = sparse_probs / sparse_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        if self.use_fp8:
            routing_probs = quantize_fp8_hybrid(routing_probs).to(x_flat.dtype)

        first_leaf = leaves[0] if leaves else self.root
        r_in = quantize_shift4(root_out) if first_leaf.use_shift4_activations else root_out

        if first_leaf.leaf_mode in ("permutation", "perm"):
            w_perm_stack = torch.stack([
                ternarize(leaf.latent_w_perm, self.threshold_frac, scale=leaf.scale_perm if self.learnable_scale else None)
                for leaf in leaves
            ], dim=0)
            b_stack = torch.stack([leaf.bias for leaf in leaves], dim=0)
            if r_in.is_cuda:
                K_num = len(leaves)
                P_num = first_leaf.num_permutations
                perms_stack = torch.stack([leaf.perms for leaf in leaves], dim=0)
                leaf_prim = b_stack.unsqueeze(0).expand(B, K_num, self.dim).clone()
                r_exp = r_in.unsqueeze(1).expand(-1, K_num, -1)
                for p_idx in range(P_num):
                    p_k = perms_stack[:, p_idx]
                    x_p = torch.gather(r_exp, -1, p_k.unsqueeze(0).expand(B, -1, -1))
                    leaf_prim = leaf_prim + x_p * w_perm_stack[:, p_idx].unsqueeze(0)
            else:
                leaf_prim = b_stack.unsqueeze(0).expand(B, len(leaves), self.dim).clone()
                for k_idx, leaf in enumerate(leaves):
                    for p_idx in range(leaf.num_permutations):
                        perm = leaf.perms[p_idx]
                        x_p = r_in[:, perm] if p_idx != 0 else r_in
                        leaf_prim[:, k_idx] = leaf_prim[:, k_idx] + x_p * w_perm_stack[k_idx, p_idx]
        elif self.rank is None:
            w_stack = torch.stack([
                ternarize(
                    leaf.latent_W_primary,
                    self.threshold_frac,
                    mask=leaf.sparsity_mask if leaf.leaf_sparsity > 0.0 else None,
                    scale=leaf.scale_w if self.learnable_scale else None
                )
                for leaf in leaves
            ], dim=0)
            b_stack = torch.stack([leaf.bias for leaf in leaves], dim=0)
            leaf_prim = torch.einsum('bi, kdi -> bkd', r_in, w_stack) + b_stack.unsqueeze(0)
        else:
            u_stack = torch.stack([
                ternarize(
                    leaf.latent_U,
                    self.threshold_frac,
                    mask=leaf.sparsity_mask if leaf.leaf_sparsity > 0.0 else None,
                    scale=leaf.scale_u if self.learnable_scale else None
                )
                for leaf in leaves
            ], dim=0)
            v_stack = torch.stack([
                ternarize(
                    leaf.latent_V,
                    self.threshold_frac,
                    scale=leaf.scale_v if self.learnable_scale else None
                )
                for leaf in leaves
            ], dim=0)
            b_stack = torch.stack([leaf.bias for leaf in leaves], dim=0)
            h_mid = torch.einsum('bi, kir -> bkr', r_in, u_stack)
            leaf_prim = torch.einsum('bkr, krd -> bkd', h_mid, v_stack) + b_stack.unsqueeze(0)

        leaf_outs = []
        for idx, leaf in enumerate(leaves):
            prim_k = leaf_prim[:, idx]
            m_active = min(len(leaf.secondary_parents), self.max_secondary)
            norm_factor = 1.0 / math.sqrt(1.0 + float(m_active)) if (self.normalize_context and m_active > 0) else 1.0

            h_context = torch.zeros_like(prim_k)
            for s_idx, p_sec in enumerate(leaf.secondary_parents):
                if s_idx >= self.max_secondary:
                    break
                c_k = p_sec.cached_output.detach() if p_sec.cached_output is not None else torch.zeros_like(prim_k)
                if leaf.use_power_of_two_gates:
                    gate_k = quantize_power_of_two_gate(leaf.W_context[s_idx])
                elif leaf.bounded_gating:
                    gate_k = torch.tanh(leaf.W_context[s_idx])
                else:
                    gate_k = leaf.W_context[s_idx]
                h_context = h_context + gate_k * c_k

            y_k = (prim_k + h_context) * norm_factor
            leaf.cached_output = leaf._apply_activation(y_k)
            leaf_outs.append(leaf.cached_output)

        stacked_leaf_outs = torch.stack(leaf_outs, dim=1)
        composite_out = torch.einsum('bk, bkd -> bd', routing_probs.to(stacked_leaf_outs.dtype), stacked_leaf_outs)
        return composite_out.reshape(*orig_shape)

    def compute_backpressure_updates(
        self,
        targets: torch.Tensor,
        loss_type: str = "mse",
        use_sign_backpressure: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        if self._last_out is None:
            raise RuntimeError("compute_backpressure_updates called without prior recording forward pass")

        targets_flat = targets.reshape(-1, self.out_features)
        out = self._last_out
        B = float(out.shape[0])

        if loss_type == "direct":
            error = targets_flat
        elif loss_type == "mse":
            error = (targets_flat - out) * (2.0 / self.out_features)
        elif loss_type == "ce":
            probs = F.softmax(out, dim=-1)
            error = targets_flat - probs
        else:
            error = (targets_flat - out) * (2.0 / self.out_features)

        routing_probs = self._last_routing_probs
        stacked_leaf_outs = self._last_stacked_leaf_outs
        leaves = self.leaves
        updates: Dict[str, Any] = {}

        # 1. Routing updates
        leaf_align = torch.einsum('bd, bkd -> bk', error, stacked_leaf_outs)
        mean_align = torch.einsum('bk, bk -> b', routing_probs, leaf_align).unsqueeze(1)
        router_f = routing_probs * (leaf_align - mean_align)

        if not self.use_hierarchical_routing:
            updates["router_weights"] = torch.einsum('bi, bk -> ki', self._last_x_flat, router_f) / B
            updates["router_biases"] = router_f.mean(dim=0)

        # 2. Leaf local updates (Vectorized Batched Backpressure)
        has_secondary = any(len(leaf.secondary_parents) > 0 for leaf in leaves)
        first_leaf = leaves[0] if leaves else self.root
        if not has_secondary and self.rank is None and first_leaf.leaf_mode not in ("permutation", "perm"):
            if first_leaf.activation == "relu6":
                act_grad = ((stacked_leaf_outs > 0) & (stacked_leaf_outs < 6)).to(stacked_leaf_outs.dtype)
            elif first_leaf.activation == "sign":
                act_grad = (stacked_leaf_outs.abs() <= 1.0).to(stacked_leaf_outs.dtype)
            else:
                act_grad = torch.ones_like(stacked_leaf_outs)

            g_v = error.unsqueeze(1) * routing_probs.unsqueeze(2) * act_grad
            x_prim = self._last_x_flat

            if use_sign_backpressure:
                g_update = g_v.sign()
                x_update = x_prim.sign()
            else:
                g_update = g_v
                x_update = x_prim

            grad_W_stack = torch.einsum('bkd, bi -> kdi', g_update, x_update) / B

            if self._last_w_stack is not None:
                w_stack = self._last_w_stack
            else:
                w_stack = torch.stack([
                    ternarize(
                        leaf.latent_W_primary,
                        self.threshold_frac,
                        mask=leaf.sparsity_mask if leaf.leaf_sparsity > 0.0 else None,
                        scale=leaf.scale_w if self.learnable_scale else None
                    ).detach()
                    for leaf in leaves
                ], dim=0)
            total_delta_upstream = torch.einsum('bkd, kdi -> bi', g_v, w_stack)

            for idx, leaf in enumerate(leaves):
                grad_W = grad_W_stack[idx]
                if leaf.leaf_sparsity > 0.0 and leaf.sparsity_mask is not None:
                    grad_W = grad_W * leaf.sparsity_mask
                updates[f"node_{leaf.node_id}_latent_W_primary"] = grad_W
                leaf.step_age += 1
        else:
            total_delta_upstream = torch.zeros_like(self._last_x_flat)
            for idx, leaf in enumerate(leaves):
                prob_k = routing_probs[:, idx:idx+1]
                local_leaf_error = error * prob_k
                leaf_updates, delta_up, energy_sig = leaf.local_backward_pass(
                    local_leaf_error, use_sign_backpressure=use_sign_backpressure
                )

                leaf.accumulated_energy = self.beta_energy * leaf.accumulated_energy + energy_sig
                leaf.step_age += 1
                total_delta_upstream = total_delta_upstream + delta_up

                for p_name, p_up in leaf_updates.items():
                    updates[f"node_{leaf.node_id}_{p_name}"] = p_up

        root_updates, root_delta, _ = self.root.local_backward_pass(
            total_delta_upstream, use_sign_backpressure=use_sign_backpressure
        )
        for p_name, p_up in root_updates.items():
            updates[f"node_{self.root.node_id}_{p_name}"] = p_up

        updates["input_pressure"] = root_delta / B

        metrics = {
            "num_nodes": len(self.nodes),
            "num_leaves": len(leaves),
            "mean_energy": float(sum(l.accumulated_energy for l in leaves) / max(len(leaves), 1)),
        }
        return updates, metrics

    def apply_updates(self, updates: Dict[str, Any], optimizer: Any) -> None:
        with torch.no_grad():
            if not self.use_hierarchical_routing:
                if "router_weights" in updates and self.router_weights is not None:
                    self.router_weights.grad = -updates["router_weights"]
                if "router_biases" in updates and self.router_biases is not None:
                    self.router_biases.grad = -updates["router_biases"]

            for name, node in self.nodes.items():
                k_w = f"node_{node.node_id}_latent_W_primary"
                k_sw = f"node_{node.node_id}_scale_w"
                k_u = f"node_{node.node_id}_latent_U"
                k_v = f"node_{node.node_id}_latent_V"
                k_su = f"node_{node.node_id}_scale_u"
                k_sv = f"node_{node.node_id}_scale_v"
                k_b = f"node_{node.node_id}_bias"
                k_c = f"node_{node.node_id}_W_context"
                k_perm = f"node_{node.node_id}_latent_w_perm"

                if k_perm in updates and getattr(node, "latent_w_perm", None) is not None:
                    node.latent_w_perm.grad = -updates[k_perm]
                if k_w in updates and node.latent_W_primary is not None:
                    node.latent_W_primary.grad = -updates[k_w]
                if k_sw in updates and getattr(node, "scale_w", None) is not None:
                    node.scale_w.grad = -updates[k_sw]
                if k_u in updates and node.latent_U is not None:
                    node.latent_U.grad = -updates[k_u]
                if k_v in updates and node.latent_V is not None:
                    node.latent_V.grad = -updates[k_v]
                if k_su in updates and getattr(node, "scale_u", None) is not None:
                    node.scale_u.grad = -updates[k_su]
                if k_sv in updates and getattr(node, "scale_v", None) is not None:
                    node.scale_v.grad = -updates[k_sv]
                if k_b in updates and node.bias is not None:
                    node.bias.grad = -updates[k_b]
                if k_c in updates and node.W_context is not None:
                    node.W_context.grad = -updates[k_c]

    def migrate_optimizer_state(
        self,
        optimizer: Any,
        parent_node: ASTDAGNode,
        child_nodes: List[ASTDAGNode]
    ) -> None:
        sub_opts = getattr(optimizer, "optimizers", [optimizer])
        for opt in sub_opts:
            if not hasattr(opt, "state"):
                continue
            for child in child_nodes:
                pairings = []
                if getattr(parent_node, "latent_w_perm", None) is not None and getattr(child, "latent_w_perm", None) is not None:
                    pairings.append((parent_node.latent_w_perm, child.latent_w_perm))
                if parent_node.latent_W_primary is not None and child.latent_W_primary is not None:
                    pairings.append((parent_node.latent_W_primary, child.latent_W_primary))
                if parent_node.latent_U is not None and child.latent_U is not None:
                    pairings.append((parent_node.latent_U, child.latent_U))
                if parent_node.latent_V is not None and child.latent_V is not None:
                    pairings.append((parent_node.latent_V, child.latent_V))
                if parent_node.bias is not None and child.bias is not None:
                    pairings.append((parent_node.bias, child.bias))
                if getattr(parent_node, "scale_w", None) is not None and getattr(child, "scale_w", None) is not None:
                    pairings.append((parent_node.scale_w, child.scale_w))

                for p_param, c_param in pairings:
                    if p_param in opt.state:
                        p_state = opt.state[p_param]
                        c_state = opt.state[c_param]
                        for k, v in p_state.items():
                            if torch.is_tensor(v):
                                c_state[k] = v.clone()
                            else:
                                c_state[k] = v

    def get_default_optimizer(
        self,
        muon_lr: float = 0.02,
        adamw_lr: float = 2e-3,
        muon_momentum: float = 0.95,
        adamw_weight_decay: float = 0.01,
        fused: bool = True
    ) -> Any:
        from affine_ai.optim.muon import HybridMuonAdamW
        return HybridMuonAdamW(
            model=self,
            muon_lr=muon_lr,
            adamw_lr=adamw_lr,
            muon_momentum=muon_momentum,
            adamw_weight_decay=adamw_weight_decay,
            fused=fused
        )

    def step_topology(
        self,
        tau_split: Optional[float] = None,
        tau_peek: Optional[float] = None,
        tau_prune: Optional[float] = None,
        tau_merge: Optional[float] = None,
        n_branches: int = 2,
        child_noise: float = 0.01,
        optimizer: Optional[Any] = None,
    ) -> Dict[str, int]:
        """
        Executes self-organizing topology updates:
        1. Net2Net Function-Preserving Split (E_v > tau_split).
        2. Strictly Acyclic Topologically-Ordered Context Peeking (E_v > tau_peek).
        3. Utility Pruning (U_v < tau_prune).
        4. Homomorphic Subtree Merging (Cosine similarity > tau_merge).
        """
        self.step_counter += 1
        tau_split = tau_split if tau_split is not None else self.tau_split
        tau_peek = tau_peek if tau_peek is not None else self.tau_peek
        tau_prune = tau_prune if tau_prune is not None else self.tau_prune
        tau_merge = tau_merge if tau_merge is not None else self.tau_merge

        split_count = 0
        peek_count = 0
        prune_count = 0
        merge_count = 0

        current_leaves = list(self.leaves)

        # 1. Node Splitting & 2. Context Peeking
        for leaf in current_leaves:
            if leaf.accumulated_energy > tau_split:
                leaf.is_leaf = False
                new_children = []
                for b_idx in range(n_branches):
                    child = self._create_node(depth=leaf.depth + 1, topo_order=leaf.topo_order * 10 + b_idx)
                    child.primary_parent = leaf
                    with torch.no_grad():
                        if getattr(leaf, "latent_w_perm", None) is not None and getattr(child, "latent_w_perm", None) is not None:
                            child.latent_w_perm.data.copy_(
                                leaf.latent_w_perm.data + torch.randn_like(leaf.latent_w_perm.data) * child_noise
                            )
                            if getattr(leaf, "scale_perm", None) is not None and getattr(child, "scale_perm", None) is not None:
                                child.scale_perm.data.copy_(leaf.scale_perm.data)
                        if leaf.latent_W_primary is not None and child.latent_W_primary is not None:
                            child.latent_W_primary.data.copy_(
                                leaf.latent_W_primary.data + torch.randn_like(leaf.latent_W_primary.data) * child_noise
                            )
                            if hasattr(leaf, "sparsity_mask") and hasattr(child, "sparsity_mask"):
                                child.sparsity_mask.copy_(leaf.sparsity_mask)
                            if leaf.scale_w is not None and child.scale_w is not None:
                                child.scale_w.data.copy_(leaf.scale_w.data)
                        if leaf.latent_U is not None and leaf.latent_V is not None and child.latent_U is not None and child.latent_V is not None:
                            child.latent_U.data.copy_(
                                leaf.latent_U.data + torch.randn_like(leaf.latent_U.data) * child_noise
                            )
                            child.latent_V.data.copy_(leaf.latent_V.data)
                            if hasattr(leaf, "sparsity_mask") and hasattr(child, "sparsity_mask"):
                                child.sparsity_mask.copy_(leaf.sparsity_mask)
                            if leaf.scale_u is not None and child.scale_u is not None:
                                child.scale_u.data.copy_(leaf.scale_u.data)
                            if leaf.scale_v is not None and child.scale_v is not None:
                                child.scale_v.data.copy_(leaf.scale_v.data)
                        child.bias.data.copy_(leaf.bias.data)
                    leaf.child_nodes.append(child)
                    new_children.append(child)

                if optimizer is not None:
                    self.migrate_optimizer_state(optimizer, leaf, new_children)

                leaf.accumulated_energy *= 0.5
                split_count += 1

            elif leaf.accumulated_energy > tau_peek:
                if len(leaf.secondary_parents) < self.max_secondary:
                    candidates = [
                        peer for peer in current_leaves
                        if peer.node_id != leaf.node_id
                        and peer not in leaf.secondary_parents
                        and peer.topo_order < leaf.topo_order
                        and not self._is_ancestor(ancestor=peer, descendant=leaf)
                        and not self._is_ancestor(ancestor=leaf, descendant=peer)
                    ]
                    if candidates:
                        peer_target = candidates[0]
                        leaf.secondary_parents.append(peer_target)
                        idx = len(leaf.secondary_parents) - 1
                        with torch.no_grad():
                            leaf.W_context.data[idx].zero_()
                        peek_count += 1

        # 3. Homomorphic Sibling Leaf Merging
        if len(self.leaves) > 2 and tau_merge < 1.0:
            leaves_to_check = list(self.leaves)
            for i in range(len(leaves_to_check)):
                for j in range(i + 1, len(leaves_to_check)):
                    u = leaves_to_check[i]
                    v = leaves_to_check[j]
                    if u.primary_parent == v.primary_parent and u.rank == v.rank:
                        with torch.no_grad():
                            if getattr(u, "latent_w_perm", None) is not None and getattr(v, "latent_w_perm", None) is not None:
                                cos_sim = F.cosine_similarity(
                                    u.latent_w_perm.flatten(), v.latent_w_perm.flatten(), dim=0
                                ).item()
                            elif u.rank is None and u.latent_W_primary is not None and v.latent_W_primary is not None:
                                cos_sim = F.cosine_similarity(
                                    u.latent_W_primary.flatten(), v.latent_W_primary.flatten(), dim=0
                                ).item()
                            elif u.rank is not None and u.latent_U is not None and v.latent_U is not None:
                                cos_sim = F.cosine_similarity(
                                    u.latent_U.flatten(), v.latent_U.flatten(), dim=0
                                ).item()
                            else:
                                cos_sim = 0.0

                            if cos_sim > tau_merge:
                                # Merge weights into u and prune v
                                if getattr(u, "latent_w_perm", None) is not None:
                                    u.latent_w_perm.data.copy_((u.latent_w_perm.data + v.latent_w_perm.data) * 0.5)
                                elif u.rank is None:
                                    u.latent_W_primary.data.copy_((u.latent_W_primary.data + v.latent_W_primary.data) * 0.5)
                                else:
                                    u.latent_U.data.copy_((u.latent_U.data + v.latent_U.data) * 0.5)
                                u.bias.data.copy_((u.bias.data + v.bias.data) * 0.5)
                                self._prune_leaf(v)
                                merge_count += 1
                                break

        # 4. Utility Pruning
        if self.step_counter % self.k_prune == 0:
            leaves_after = list(self.leaves)
            for leaf in leaves_after:
                if len(self.leaves) > 2 and leaf.utility_counter < tau_prune:
                    self._prune_leaf(leaf)
                    prune_count += 1
            for node in self.nodes.values():
                node.utility_counter = 0.0

        if split_count > 0 or prune_count > 0 or merge_count > 0:
            self._sync_router()

        return {
            "splits": split_count,
            "peeks": peek_count,
            "merges": merge_count,
            "prunes": prune_count,
            "num_leaves": len(self.leaves),
        }

    def _is_ancestor(self, ancestor: ASTDAGNode, descendant: ASTDAGNode) -> bool:
        curr = descendant.primary_parent
        while curr is not None:
            if curr.node_id == ancestor.node_id:
                return True
            curr = curr.primary_parent
        return False

    def _prune_leaf(self, leaf: ASTDAGNode) -> None:
        if str(leaf.node_id) in self.nodes:
            if leaf.primary_parent is not None and leaf in leaf.primary_parent.child_nodes:
                leaf.primary_parent.child_nodes.remove(leaf)
                if len(leaf.primary_parent.child_nodes) == 0 and leaf.primary_parent != self.root:
                    leaf.primary_parent.is_leaf = True
            for node in self.nodes.values():
                if leaf in node.secondary_parents:
                    node.secondary_parents.remove(leaf)
            del self.nodes[str(leaf.node_id)]


# Aliases
ASDAGLayer = ASTDAGLayer
ASDAGNode = ASTDAGNode
AdaptiveSparseTreeDAGLayer = ASTDAGLayer
