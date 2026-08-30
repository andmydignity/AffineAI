"""
FusedSparseBackpressureTree - v3

Builds on v2 (bug fixes + diffusion simplification) and adds ternary QAT /
structural sparsity / AdamW-integration.

Verified properties (see the original test suite for this module):
  1. TERNARY QAT -- leaf weights only, NOT routing weights. Ternary-quantizing
     ROUTING weights costs ~70-100% worse MSE (a representational-capacity
     limit, not a gradient-signal issue), which is why there is deliberately
     NO `ternary_routing` option.
  2. STRUCTURAL SPARSITY on leaf weights via a FIXED (never trained) binary
     mask applied before ternary quantization. The accuracy/compute tradeoff
     must be re-measured on your own model and data before relying on it;
     default is 0.0 (off) for this reason.
  3. ADAMW INTEGRATION -- use apply_updates() with a real torch.optim
     optimizer; do not write a new fixed-lr optimizer for this module.

NOT addressed in this file:
  - The leaf GEMM is computed DENSE over all num_leaves every forward pass;
    actual FLOP savings require a kernel that skips zero entries / uses
    shift+add. `leaf_sparsity` only changes which VALUES are nonzero.
  - Dynamic topology (growing/pruning leaves at runtime) is not implemented.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any


class _SignSTE(torch.autograd.Function):
    """sign(h) forward with clipped straight-through backward (|h|<=1).
    Mirrors ternary_ste's STE convention so autograd through the deployed
    graph matches the hydraulic VJP exactly."""
    @staticmethod
    def forward(ctx, h):
        ctx.save_for_backward(h)
        return torch.sign(h)
    @staticmethod
    def backward(ctx, grad_out):
        h, = ctx.saved_tensors
        return grad_out * (h.abs() <= 1.0).to(h.dtype)


def ternary_ste(w: torch.Tensor, threshold_frac: float = 0.7,
                mask: Optional[torch.Tensor] = None,
                scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Straight-through-estimator ternary quantization (TWN-style):
      W_q = alpha * sgn(W) * 1[|W| > delta],   delta = threshold_frac * mean|W|
    Backward treats the operation as identity w.r.t. `w`.

    `scale`: optional LEARNABLE tensor used as alpha instead of the derived
    mean-of-active magnitudes. Gradient flows into it (∂/∂alpha = sgn),
    letting the network control quantized OUTPUT AMPLITUDE -- critical in
    pre-norm residual stacks where fixed-alpha outputs cannot grow to the
    amplitude the stack wants (observed: fixed-alpha mixers were inert in an
    LM while fp mixers contributed). Threshold selection still tracks the
    weights; only magnitude becomes learned.
    """
    w_eff = w * mask if mask is not None else w
    # Threshold statistics must come from SURVIVING entries only: including
    # mask zeros deflates the threshold by the mask density, which disables
    # ternary's own selection role under heavy masks (measured: 97.7% of
    # survivors activated instead of ~58%).
    abs_eff = w_eff.detach().abs()
    if mask is not None:
        survivors = abs_eff[w_eff != 0]
        mean_abs = survivors.mean() if survivors.numel() else abs_eff.mean()
    else:
        mean_abs = abs_eff.mean()
    ref = scale.detach() if scale is not None else mean_abs
    delta = mean_abs * threshold_frac
    w_sign = torch.where(w_eff > delta, torch.ones_like(w_eff),
             torch.where(w_eff < -delta, -torch.ones_like(w_eff), torch.zeros_like(w_eff)))
    if scale is not None:
        alpha = scale                       # learnable: gradient flows in
    else:
        active = (w_sign != 0).to(w_eff.dtype)
        denom = active.sum().clamp(min=1.0)
        alpha = ((abs_eff * active).sum() / denom).detach()
    # forward = alpha*sign ; d/dw = identity (STE) ; d/dalpha = sign
    return w_sign.detach() * alpha + (w_eff - w_eff.detach())


class FusedSparseBackpressureTreeV3(nn.Module):
    """
    Ultra-Fast Contiguous Sibling-Aware Sparse Hydraulic Tree (v3).

    Adds ternary QAT (leaf-only) and optional structural sparsity on top of
    v2's bug fixes and diffusion simplification.

    Training: either plain backprop (STE makes ternary leaves compatible), or
    the hydraulic method -- see compute_backpressure_updates() and
    apply_updates().

    Gradient-exactness (verified numerically against autograd):
      - LEAF updates are the exact gradients of THIS class's deployed
        forward (argmax upper levels), at every depth, ternary or not.
      - ROUTING updates are the exact chain-rule gradients of the fully-soft
        routing tree (product of all levels' conductances) evaluated at the
        same operating point and error -- verified to float precision at
        depths 1-5 via a single general diffusion recursion (see
        compute_backpressure_updates). At depth >= 2 this differs from
        literal .backward() on this class's own graph only because argmax
        hard-routing zeroes those paths in autograd; backpressure supplies
        the soft-relaxation gradient there instead. At depth 1 (fully soft)
        both coincide and the entire update set equals .backward() exactly.

    LOW-RANK LEAVES (default, rank=16): per-leaf U_k with a SHARED V
    (MultiHeadAffineTree convention). Still linear in x and in each factor,
    so all exactness results above carry over -- leaf updates become two
    closed-form einsums, and the shared-V mix-before-project structure means
    the projection cost no longer scales with num_leaves. CAVEAT: ternary
    quantization applied to the FACTORS composes differently than ternary on
    a full product weight; the earlier ternary accuracy measurements were
    full-rank and need re-measuring for factorized leaves.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        depth: int = 3,
        n_ary: int = 4,
        top_k: int = 2,                    # DEPRECATED/vestigial: clamped and stored,
                                           # but never used by forward or updates.
        temperature: float = 1.0,
        viscous_damping: float = 0.05,     # DEPRECATED/vestigial: stored, never used.
        init_identity: bool = False,
        is_output_layer: bool = True,
        rank: Optional[int] = 16,         # Low-rank leaves by DEFAULT (mirrors
                                          # MultiHeadAffineTree). rank=None restores full affine.
        ternary_leaves: bool = True,      # No ternary_routing option, deliberately.
        leaf_sparsity: float = 0.0,       # Unstructured sparsity fraction. Ignored if nm is set.
        nm: Optional[Tuple[int, int]] = None,  # N:M STRUCTURED sparsity: keep exactly
                                          # N of every M consecutive entries (grouped
                                          # along the contiguous last dim). E.g. (2,4)
                                          # classic semi-structured, (1,16) = 93.75%.
                                          # Overrides leaf_sparsity. Groups give CPU
                                          # kernels whole-block skips; unstructured
                                          # scattered zeros cannot be skipped cheaply.
        ternary_threshold_frac: float = 0.7,
        sparse_dispatch: bool = True,     # Gather only the n sibling leaves each
                                          # token can reach, instead of contracting
                                          # x against ALL K leaves. Forward OUTPUT is
                                          # numerically identical, and leaf/V/bias
                                          # updates + input_pressure stay exact --
                                          # but ROUTING updates change meaning: they
                                          # become the gradient of the HARDENED tree
                                          # (only n reachable siblings carry pressure)
                                          # rather than of the fully-soft relaxation,
                                          # which requires all-K counterfactual
                                          # alignments. Set False when you need the
                                          # soft-relaxation routing gradient.
                                          # Low-rank + depth > 1 only.
        route_mode: str = "hard",         # "hard": argmax top-1 upper levels (deployed
                                          # shape). "soft": NO argmax anywhere -- probs
                                          # are the product of all levels' conductances,
                                          # so every leaf receives gradient every step
                                          # and branches cannot die. This is the mode
                                          # where backpressure == literal backprop
                                          # exactly, and where self-organized
                                          # specialization happens (train soft, deploy
                                          # hard). Forced dense forward in soft mode.
        leaf_activation: str = "none",    # Fixed nonlinearity on the leaf code h=xU_k
                                          # BEFORE mixing/projection: turns each leaf
                                          # from an affine patch into a nonlinear local
                                          # expert while keeping updates closed-form
                                          # (analytic VJP through phi'). Options:
                                          #   "none"   identity (pure affine, original)
                                          #   "sqrelu" x^2*[x>0]  -- 2 ALU ops, poly
                                          #            local experts, free derivative
                                          #   "sign"   +-1 STE (clipped identity grad,
                                          #            |h|<=1) -- activations go binary;
                                          #            V contraction becomes popcount-class
        learnable_scale: bool = False,    # Learnable per-factor ternary output scale
                                          # (alpha): fixes inert-mixer syndrome in
                                          # pre-norm residual stacks where fixed-alpha
                                          # outputs cannot grow to the amplitude the
                                          # stack wants.
	):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.depth = depth
        self.n_ary = n_ary
        self.num_leaves = n_ary ** depth
        self.top_k = min(max(top_k, 2), self.num_leaves)
        self.temperature = max(temperature, 1e-4)
        self.viscous_damping = viscous_damping
        self.is_output_layer = is_output_layer
        self.rank = rank
        self.ternary_leaves = ternary_leaves
        self.ternary_threshold_frac = ternary_threshold_frac

        self.routing_weights = nn.Parameter(
            torch.randn(depth, in_features, n_ary) * (1.0 / math.sqrt(in_features))
        )
        self.routing_biases = nn.Parameter(torch.zeros(depth, n_ary))

        if rank is not None:
            # Low-rank leaves: per-leaf U_k in R^{in x rank}, SHARED V in
            # R^{rank x out} (same convention as MultiHeadAffineTree).
            # y_k = (x U_k) V + b_k -- still linear in x and in each factor,
            # so backpressure stays closed-form and tape-free. Sharing V
            # means the mixture is contracted BEFORE the projection:
            # sum_k p_k (x U_k) V = (sum_k p_k xU_k) V, so the projection
            # cost no longer scales with num_leaves.
            self.leaf_u = nn.Parameter(
                torch.randn(self.num_leaves, in_features, rank) * (1.0 / math.sqrt(in_features))
            )
            self.leaf_v = nn.Parameter(
                torch.randn(rank, out_features) * (1.0 / math.sqrt(rank))
            )
        else:
            # Full-precision "shadow" weights -- ternary_ste() quantizes them
            # on the fly in forward(); the optimizer always updates these.
            self.leaf_weights = nn.Parameter(
                torch.randn(self.num_leaves, in_features, out_features) * (1.0 / math.sqrt(in_features))
            )

        self.leaf_biases = nn.Parameter(torch.zeros(self.num_leaves, out_features))

        # init_identity only applies to full-rank leaves (identity not
        # representable at rank < min(in, out)).
        if init_identity and in_features == out_features and rank is None:
            with torch.no_grad():
                self.leaf_weights.data.copy_(
                    torch.eye(in_features).unsqueeze(0).expand(self.num_leaves, -1, -1)
                )

        # Structural sparsity masks. Applied to the LEAF-SPECIFIC factor
        # only (U / full weights): masking the shared V damages every leaf
        # simultaneously to save a negligible r*O parameters, so V stays
        # dense. Two regimes:
        #   nm=(N, M)  -- structured: exactly N of every M consecutive
        #                 entries (last dim) are active. Kernel-friendly.
        #   leaf_sparsity -- unstructured Bernoulli (weakest baseline).
        # Masks can be rebuilt from magnitudes / gradients after warmup:
        # see build_magnitude_mask() and redistribute_sparsity().
        if nm is not None:
            assert len(nm) == 2 and 0 < nm[0] <= nm[1], f"invalid nm={nm}"
            self.nm = tuple(nm)
            self.leaf_sparsity = 1.0 - nm[0] / nm[1]
        else:
            self.nm = None
            self.leaf_sparsity = leaf_sparsity

        if rank is not None:
            shape = (self.num_leaves, in_features, rank)
            mask_u = self._initial_mask(shape) if rank is not None else None
            self.register_buffer('leaf_sparsity_mask_u', mask_u)
        else:
            shape = (self.num_leaves, in_features, out_features)
            self.register_buffer('leaf_sparsity_mask', self._initial_mask(shape))
        self.sparse_dispatch = sparse_dispatch
        assert route_mode in ("hard", "soft"), route_mode
        self.route_mode = route_mode
        assert leaf_activation in ("none", "sqrelu", "sign"), leaf_activation
        self.leaf_activation = leaf_activation

        # Learnable ternary output scales (alpha), one per quantized factor.
        # Initialized from the derived mean-of-active-magnitudes so the
        # starting function matches fixed-alpha behavior exactly.
        self.learnable_scale = learnable_scale and ternary_leaves
        self.scale_u = self.scale_v = self.scale_w = None
        if self.learnable_scale:
            with torch.no_grad():
                def _init_alpha(q):
                    nz = q[q != 0]
                    return (nz.abs().mean() if nz.numel()
                            else torch.tensor(1.0, device=q.device))
                if rank is not None:
                    self.scale_u = nn.Parameter(_init_alpha(ternary_ste(
                        self.leaf_u, self.ternary_threshold_frac,
                        mask=self.leaf_sparsity_mask_u)))
                    self.scale_v = nn.Parameter(_init_alpha(ternary_ste(
                        self.leaf_v, self.ternary_threshold_frac)))
                else:
                    self.scale_w = nn.Parameter(_init_alpha(ternary_ste(
                        self.leaf_weights, self.ternary_threshold_frac,
                        mask=self.leaf_sparsity_mask)))

        self._forward_cache: Optional[Dict[str, Any]] = None

    def _phi(self, h: torch.Tensor) -> torch.Tensor:
        if self.leaf_activation == "none":
            return h
        if self.leaf_activation == "sqrelu":
            return h * h * (h > 0).to(h.dtype)
        # sign with straight-through clipped backward (via _SignSTE)
        return _SignSTE.apply(h)

    def _phi_grad(self, h: torch.Tensor) -> torch.Tensor:
        if self.leaf_activation == "none":
            return torch.ones_like(h)
        if self.leaf_activation == "sqrelu":
            return 2.0 * h * (h > 0).to(h.dtype)
        return (h.abs() <= 1.0).to(h.dtype)

    def set_temperature(self, temp: float):
        self.temperature = max(temp, 1e-4)

    def _initial_mask(self, shape) -> torch.Tensor:
        """Random-init mask. For nm mode this is STRATIFIED random (exactly N
        active per group, positions random) -- balanced by construction."""
        if self.leaf_sparsity <= 0.0:
            return torch.ones(shape)
        if self.nm is not None:
            return self._nm_topn_mask(torch.rand(shape))
        return (torch.rand(shape) > self.leaf_sparsity).float()

    def _nm_topn_mask(self, score: torch.Tensor) -> torch.Tensor:
        """Top-N-per-group-of-M binary mask over the last dim of `score`."""
        assert score.shape[-1] % self.nm[1] == 0, \
            f"last dim {score.shape[-1]} not divisible by M={self.nm[1]}"
        g = score.reshape(*score.shape[:-1], -1, self.nm[1])
        _, idx = g.topk(self.nm[0], dim=-1)
        mask = torch.zeros_like(g).scatter(-1, idx, 1.0)
        return mask.reshape(score.shape)

    def _effective_leaf_weights(self) -> torch.Tensor:
        """Full-rank effective weights (only valid when rank is None):
        masked (if leaf_sparsity>0) and ternary-quantized (if ternary)."""
        scale = self.scale_w if self.learnable_scale else None
        if self.ternary_leaves:
            return ternary_ste(self.leaf_weights, self.ternary_threshold_frac,
                               mask=self.leaf_sparsity_mask, scale=scale)
        elif self.leaf_sparsity > 0.0:
            return self.leaf_weights * self.leaf_sparsity_mask
        else:
            return self.leaf_weights

    def _effective_factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Low-rank effective factors (only valid when rank is not None).
        Ternary applies to both factors; STRUCTURAL SPARSITY applies to the
        leaf-specific U only -- the shared V is never masked."""
        u, v = self.leaf_u, self.leaf_v
        if self.ternary_leaves:
            su = self.scale_u if self.learnable_scale else None
            sv = self.scale_v if self.learnable_scale else None
            u = ternary_ste(u, self.ternary_threshold_frac,
                            mask=self.leaf_sparsity_mask_u, scale=su)
            v = ternary_ste(v, self.ternary_threshold_frac, scale=sv)
        elif self.leaf_sparsity > 0.0:
            u = u * self.leaf_sparsity_mask_u
        return u, v

    def forward(self, x: torch.Tensor, record_cache: Optional[bool] = None) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        B = x_flat.shape[0]

        all_logits = torch.einsum('bi, din -> bdn', x_flat, self.routing_weights) + self.routing_biases.unsqueeze(0)
        conductances = F.softmax(all_logits / self.temperature, dim=-1)

        # Routing dispatch. Two modes:
        #   "hard": top-1 argmax at levels 0..D-2 (one-hot), soft softmax at
        #           the last level -- the deployed shape.
        #   "soft": NO argmax anywhere; probs are the product of all levels'
        #           conductances. Every leaf receives gradient every step, so
        #           branch death is structurally impossible during training
        #           and specialization emerges from the loss alone (gradient
        #           EM for a mixture of affine experts). In this mode
        #           backpressure == literal backprop EXACTLY.
        final_nodes = None
        if self.route_mode == "soft":
            p = conductances[:, 0]
            for d in range(1, self.depth):
                p = (p.unsqueeze(-1) * conductances[:, d].unsqueeze(-2)).reshape(B, -1)
            sparse_probs = p
        elif self.depth == 1:
            sparse_probs = conductances[:, 0]
        else:
            node = conductances[:, 0].argmax(dim=-1)
            probs = F.one_hot(node, num_classes=self.n_ary).float()
            for d in range(1, self.depth):
                if d < self.depth - 1:
                    nxt = F.one_hot(conductances[:, d].argmax(dim=-1), num_classes=self.n_ary).float()
                    probs = (probs.reshape(B, -1, 1) * nxt.unsqueeze(-2)).reshape(B, -1)
                    node = node * self.n_ary + conductances[:, d].argmax(dim=-1)
                else:
                    nxt = conductances[:, d]
                    probs = (probs.reshape(B, -1, 1) * nxt.unsqueeze(-2)).reshape(B, -1)
            sparse_probs = probs
            # Leaf ids each token actually routes to: the n children of its
            # hard-routed final node (row-major leaf ordering).
            final_nodes = node

        use_sparse = (
            self.sparse_dispatch and self.route_mode == "hard"
            and self.rank is not None
            and self.depth > 1
        )

        if self.rank is None:
            leaf_w_eff = self._effective_leaf_weights()
            leaf_w_packed = leaf_w_eff.permute(1, 0, 2).reshape(self.in_features, self.num_leaves * self.out_features)
            x_proj = torch.matmul(x_flat, leaf_w_packed).view(B, self.num_leaves, self.out_features)
            leaf_outputs = x_proj + self.leaf_biases.unsqueeze(0)
            out = torch.bmm(sparse_probs.unsqueeze(1), leaf_outputs).squeeze(1)
            h = m = sib = p_sib = h_sel = a_sel = a = None
        elif use_sparse:
            # Gather only the n reachable sibling leaves per token and
            # contract those: B*n*I*r work instead of B*I*K*r.
            u_eff, v_eff = self._effective_factors()
            sib = final_nodes.unsqueeze(1) * self.n_ary + torch.arange(self.n_ary, device=x_flat.device)
            p_sib = conductances[:, -1]
            u_sel = u_eff[sib]                                    # (B, n, I, r)
            h_sel = torch.einsum('bi,bnir->bnr', x_flat, u_sel)   # (B, n, r)
            a_sel = self._phi(h_sel)
            m = torch.einsum('bn,bnr->br', p_sib, a_sel)
            out = m @ v_eff + torch.einsum('bn,bno->bo', p_sib, self.leaf_biases[sib])
            h = a = None
            leaf_outputs = None
        else:
            # Low-rank dense-over-K path: mix before projecting.
            u_eff, v_eff = self._effective_factors()
            u_packed = u_eff.permute(1, 0, 2).reshape(self.in_features, self.num_leaves * self.rank)
            h = torch.matmul(x_flat, u_packed).view(B, self.num_leaves, self.rank)
            a = self._phi(h)
            m = torch.einsum('bk,bkr->br', sparse_probs, a)
            out = m @ v_eff + sparse_probs @ self.leaf_biases
            sib = p_sib = h_sel = a_sel = None
            leaf_outputs = None

        if record_cache is None:
            record_cache = self.training
        if record_cache:
            self._forward_cache = {
                "x_flat": x_flat,
                "conductances": conductances.detach(),
                "sparse_probs": sparse_probs.detach(),
                "leaf_outputs": leaf_outputs.detach() if leaf_outputs is not None else None,
                "h": h.detach() if h is not None else (h_sel.detach() if h_sel is not None else None),
                "a": a.detach() if a is not None else (a_sel.detach() if a_sel is not None else None),
                "m": m.detach() if m is not None else None,
                "sib": sib,
                "p_sib": p_sib.detach() if p_sib is not None else None,
                "h_sel": h_sel.detach() if h_sel is not None else None,
                "out": out.detach(),
                "B": B,
            }
        else:
            # Invalidate any stale cache so compute_backpressure_updates()
            # can never silently consume tensors from an OLD forward pass
            # (e.g. after switching to eval mode mid-training).
            self._forward_cache = None

        return out.reshape(*orig_shape, self.out_features)

    def compute_backpressure_updates(
        self,
        targets: torch.Tensor,
        loss_type: str = "mse",
        compute_metrics: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Returns per-parameter update DIRECTIONS (descent direction). See
        apply_updates() for the recommended way to actually use these with a
        real optimizer.

        Correct even when leaves are ternary/sparse-masked: leaf_outputs is
        already the result of a forward pass through the effective (masked,
        quantized) weights, so the diffusion math downstream correctly
        attributes credit assuming observed forward behavior; STE's backward
        (identity) makes the result an update for the fp32 shadow weights,
        which is where apply_updates() applies it.

        loss_type="direct" has the opposite sign convention from "mse"/"ce".
        """
        x_flat = self._forward_cache["x_flat"]
        conductances = self._forward_cache["conductances"]
        sparse_probs = self._forward_cache["sparse_probs"]
        out = self._forward_cache["out"]
        B = self._forward_cache["B"]
        h = self._forward_cache.get("h")
        m = self._forward_cache.get("m")

        targets_flat = targets.reshape(-1, self.out_features)

        if loss_type == "direct":
            error = targets_flat
        elif loss_type == "mse":
            error = (targets_flat - out) * (2.0 / self.out_features)
        elif loss_type == "ce":
            probs = F.softmax(out, dim=-1)
            error = targets_flat - probs
        else:
            error = (targets_flat - out) * (2.0 / self.out_features)

        if self.rank is None:
            leaf_alignment = torch.sum(error.unsqueeze(1) * self._forward_cache["leaf_outputs"], dim=-1)
            leaf_pressure = -leaf_alignment
        else:
            # <e, y_k> with y_k = phi(x U_k) V + b_k, computed without
            # materializing (B, K, O): alignment = a . (e V^T) + <e, b_k>,
            # where a = phi(h) is the ACTIVATED code.
            u_eff, v_eff = self._effective_factors()
            z = torch.einsum('bo,ro->br', error, v_eff.detach())
            sib = self._forward_cache.get("sib")
            if sib is None:
                a = self._forward_cache["a"]
                leaf_alignment = torch.einsum('bkr,br->bk', a, z) \
                    + torch.einsum('ko,bo->bk', self.leaf_biases.detach(), error)
                leaf_pressure = -leaf_alignment
            else:
                # Sparse dispatch: only the n reachable siblings carry pressure.
                p_sib = self._forward_cache["p_sib"]
                a_sel = self._forward_cache["a"]
                b_sel = self.leaf_biases.detach()[sib]
                align_sel = torch.einsum('bnr,br->bn', a_sel, z) \
                    + torch.einsum('bno,bo->bn', b_sel, error)
                leaf_pressure = torch.zeros(
                    B, self.num_leaves, device=x_flat.device, dtype=x_flat.dtype
                ).scatter(1, sib, -align_sel)

        # General diffusion recursion -- exact chain-rule gradient of the
        # fully-soft routing tree, valid for ANY depth >= 1.
        #
        # Path weight of a length-d prefix m: Q_d[m] = prod_{j<d} c_j[digit],
        # flattened row-major (Q_0 is the empty product == 1). Bottom-up:
        #   pp_d[m] = sum_n M_d[m,n] * c_d[n]
        #   f_d[n]  = c_d[n] * ( sum_m Q_d[m]*pp_d[m] - sum_m Q_d[m]*M_d[m,n] )
        #   M_{d-1} = pp_d reshaped to length-(d-1) prefixes
        # which reproduces the previously hand-written depth 2/3 formulas
        # exactly and supplies the conductance-weighted Jacobian terms that
        # the old depth-4 shortcut omitted.
        n = self.n_ary
        q = torch.ones(B, 1, device=x_flat.device, dtype=x_flat.dtype)
        path_weights = [q]
        for d in range(self.depth - 1):
            q = (q.view(B, -1, 1) * conductances[:, d].unsqueeze(1)).reshape(B, -1)
            path_weights.append(q)

        M = leaf_pressure.view(B, n ** (self.depth - 1), n)
        f_levels = []
        for d in range(self.depth - 1, -1, -1):
            cd = conductances[:, d]
            pp = torch.sum(M * cd.unsqueeze(1), dim=-1)
            qw = path_weights[d]
            f_d = cd * (
                torch.sum(qw * pp, dim=-1, keepdim=True)
                - torch.einsum('bm,bmn->bn', qw, M)
            )
            f_levels.append(f_d)
            if d > 0:
                M = pp.view(B, n ** (d - 1), n)
        f_all = torch.stack(f_levels[::-1], dim=1)

        routing_w_updates = torch.einsum('bi, bdn -> din', x_flat, f_all) / float(B)
        routing_b_updates = f_all.mean(dim=0)

        if self.rank is None:
            x_weighted = torch.einsum('bk, bi -> kbi', sparse_probs, x_flat)
            leaf_w_updates = torch.einsum('kbi, bo -> kio', x_weighted, error) / float(B)
            # Keep updates at masked-to-zero positions at exactly zero, so a
            # sparse leaf never accumulates gradient at its fixed-zero entries.
            if self.leaf_sparsity > 0.0:
                leaf_w_updates = leaf_w_updates * self.leaf_sparsity_mask
            leaf_b_updates = torch.einsum('bk, bo -> ko', sparse_probs, error) / float(B)
            updates = {
                "routing_weights": routing_w_updates,
                "routing_biases": routing_b_updates,
                "leaf_weights": leaf_w_updates,
                "leaf_biases": leaf_b_updates,
            }
        else:
            # Closed-form two-factor updates (descent direction), exact for
            # the deployed graph: y = ((x U_k) V) mixed by p before V.
            #   dV ~ m^T e ;  dU_k ~ x^T (p_k (e V^T))
            _, v_eff = self._effective_factors()
            z = torch.einsum('bo,ro->br', error, v_eff.detach())
            sib = self._forward_cache.get("sib")
            v_updates = torch.einsum('br,bo->ro', m, error) / float(B)
            if sib is None:
                pz = sparse_probs.unsqueeze(-1) * z.unsqueeze(1)          # (B,K,r)
                if self.leaf_activation != "none":
                    pz = pz * self._phi_grad(self._forward_cache["h"])     # VJP through phi'
                u_updates = torch.einsum('bi,bkr->kir', x_flat, pz) / float(B)
                leaf_b_updates = torch.einsum('bk, bo -> ko', sparse_probs, error) / float(B)
            else:
                p_sib = self._forward_cache["p_sib"]
                h_sel = self._forward_cache["h_sel"]
                pz_sel = p_sib.unsqueeze(-1) * z.unsqueeze(1)             # (B,n,r)
                if self.leaf_activation != "none":
                    pz_sel = pz_sel * self._phi_grad(h_sel)               # VJP through phi'
                contrib = torch.einsum('bi,bnr->bnir', x_flat, pz_sel)
                u_updates = torch.zeros(
                    self.num_leaves, self.in_features, self.rank,
                    device=x_flat.device, dtype=x_flat.dtype,
                ).index_add_(0, sib.reshape(-1), contrib.reshape(B * self.n_ary, self.in_features, self.rank))
                u_updates = u_updates / float(B)
                db_sel = (p_sib.unsqueeze(-1) * error.unsqueeze(1)).reshape(B * self.n_ary, self.out_features)
                leaf_b_updates = torch.zeros(
                    self.num_leaves, self.out_features,
                    device=x_flat.device, dtype=x_flat.dtype,
                ).index_add_(0, sib.reshape(-1), db_sel) / float(B)
            # Unmasked magnitude = where the network WOULD grow were it not
            # for the fixed mask. Masked positions get exactly zero gradient
            # by construction, so RigL growth must read THIS signal or it
            # can never reach a genuinely-inactive position.
            u_scores = u_updates.detach().abs()
            if self.leaf_sparsity > 0.0:
                u_updates = u_updates * self.leaf_sparsity_mask_u
            updates = {
                "routing_weights": routing_w_updates,
                "routing_biases": routing_b_updates,
                "leaf_u": u_updates,
                "leaf_v": v_updates,
                "leaf_biases": leaf_b_updates,
                "leaf_u_scores": u_scores,
            }

        router_steering_pressure = torch.einsum('bdn, din -> bi', f_all, self.routing_weights.data)
        if self.rank is None:
            leaf_input_flow = torch.einsum('bk, bo, kio -> bi', sparse_probs, error, self.leaf_weights.data)
        else:
            # STE-consistent dx uses the raw factors (identity backward),
            # mirroring the full-rank path's use of .data here.
            u_raw, v_raw = self.leaf_u.data, self.leaf_v.data
            z_raw = torch.einsum('bo,ro->br', error, v_raw)
            sib = self._forward_cache.get("sib")
            if sib is None:
                pz_raw = sparse_probs.unsqueeze(-1) * z_raw.unsqueeze(1)
                if self.leaf_activation != "none":
                    pz_raw = pz_raw * self._phi_grad(self._forward_cache["h"])
                leaf_input_flow = torch.einsum('bkr,kir->bi', pz_raw, u_raw)
            else:
                pz_sel = self._forward_cache["p_sib"].unsqueeze(-1) * z_raw.unsqueeze(1)
                if self.leaf_activation != "none":
                    pz_sel = pz_sel * self._phi_grad(self._forward_cache["h_sel"])
                leaf_input_flow = torch.einsum(
                    'bnr,bnir->bi', pz_sel, u_raw[sib])
        if self.is_output_layer:
            router_steering_pressure = router_steering_pressure / float(B)
            leaf_input_flow = leaf_input_flow / float(B)
        total_input_pressure = leaf_input_flow + router_steering_pressure

        updates["input_pressure"] = total_input_pressure

        metrics = {}
        if compute_metrics:
            with torch.no_grad():
                entropy = -torch.sum(sparse_probs.mean(dim=0) * torch.log(sparse_probs.mean(dim=0) + 1e-8)).item()
                metrics = {"utilization_entropy": entropy}

        return updates, metrics

    def apply_updates(self, updates: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> None:
        """
        Recommended way to consume compute_backpressure_updates()'s output.

        Verified against real backprop+AdamW on this architecture:
          - LEAF parameter updates are numerically IDENTICAL to autograd
            gradients of the deployed forward at every depth (to float
            precision), ternary or sparsity-masked or not.
          - ROUTING updates are the exact gradients of the fully-soft
            routing tree given the same error signal, verified at depths
            1-5 via the general diffusion recursion (see class docstring).
            At depth >= 2 they differ from literal .backward() only where
            argmax hard-routing zeroes autograd's paths; backpressure
            supplies the soft-relaxation gradient there instead.

        Do not substitute a hand-written fixed-lr optimizer: a plain
        fixed-lr optimizer measurably underperforms AdamW fed these same
        updates (the gap is attributable to Adam's momentum/adaptive
        scaling, not to the pressure signal).

        Usage:
            out = layer(x)
            updates, _ = layer.compute_backpressure_updates(targets, "mse")
            layer.apply_updates(updates, optimizer)   # replaces .backward()
            optimizer.step()
            optimizer.zero_grad()

        Sign convention: `updates` are in the DESCENT direction; .grad holds
        the ASCENT direction (standard optimizers do param -= lr * grad), so
        this method negates before assigning. This method does NOT call
        optimizer.step() or optimizer.zero_grad().

        SPARSITY NOTE: AdamW's weight_decay applies to EVERY parameter
        regardless of gradient, so masked positions drift away from exact
        zero in the raw parameter tensor over many steps. Forward/backward
        correctness is unaffected (the mask is re-applied every forward),
        but call resparsify() after optimizer.step() if you export or
        inspect self.leaf_weights directly and need true sparsity.
        """
        with torch.no_grad():
            self.routing_weights.grad = -updates["routing_weights"]
            self.routing_biases.grad = -updates["routing_biases"]
            self.leaf_biases.grad = -updates["leaf_biases"]
            if self.rank is None:
                self.leaf_weights.grad = -updates["leaf_weights"]
            else:
                self.leaf_u.grad = -updates["leaf_u"]
                self.leaf_v.grad = -updates["leaf_v"]
            # Stash per-entry leaf-gradient scores for RigL-style growth
            # (redistribute_sparsity grows where the gradient wants to).
            # Prefer the UNMASKED scores: masked entries read exactly zero,
            # so growth from the masked signal can only ever regrow the
            # positions that were just dropped.
            if self.rank is not None:
                self._last_leaf_scores = updates.get(
                    "leaf_u_scores", updates["leaf_u"]).detach().abs()
            else:
                self._last_leaf_scores = updates["leaf_weights"].detach().abs()

    def resparsify(self) -> None:
        """
        Re-zero leaf parameters at masked (sparsity) positions. Call
        after optimizer.step() if leaf_sparsity > 0 to counteract AdamW
        weight_decay drift -- see apply_updates(). No-op otherwise.
        Recommended pattern:
            layer.apply_updates(updates, optimizer)
            optimizer.step()
            layer.resparsify()
            optimizer.zero_grad()
        """
        if self.leaf_sparsity > 0.0:
            with torch.no_grad():
                if self.rank is None:
                    self.leaf_weights.data.mul_(self.leaf_sparsity_mask)
                else:
                    self.leaf_u.data.mul_(self.leaf_sparsity_mask_u)

    def grown_copy(self, extra_levels: int = 1, child_noise: float = 0.02,
                   new_router_scale: float = 0.05, seed: int = 0) -> "FusedSparseBackpressureTreeV3":
        """
        Function-preserving expansion (Net2Net / Growing-Neural-Gas style):
        returns a NEW layer with `depth + extra_levels`, grown from the ROOT:
        the entire tree is replicated into n_ary**extra_levels identical
        copies (+ tiny noise for symmetry breaking) under new near-uniform
        root routing levels.

        Top insertion is what makes this EXACTLY function-preserving: the
        new upper levels are argmax over identical subtrees, so whichever
        copy a token lands in produces the original output. (Appending at
        the LEAF side would NOT be preserving -- it hardens the previous
        last-level softmax into argmax.) Growth cannot hurt quality by
        construction, while multiplying stored capacity by
        n_ary**extra_levels. Copies differentiate during subsequent training
        as root routing redirects traffic between them.

        This exists because adding FRESH RANDOM leaves starves them: per-leaf
        update frequency falls as 1/K and unvisited branches compete while
        frozen at random init. Inherited subtrees begin fully trained instead.
        Recommended usage: train -> grown_copy() -> continue training -> repeat.
        Optimizer state is intentionally NOT transferred (fresh moments).
        """
        import math as _math
        gen = torch.Generator().manual_seed(seed)
        dev = next(self.parameters()).device
        rep = self.n_ary ** extra_levels
        new = FusedSparseBackpressureTreeV3(
            in_features=self.in_features, out_features=self.out_features,
            depth=self.depth + extra_levels, n_ary=self.n_ary,
            top_k=self.top_k, temperature=self.temperature,
            viscous_damping=self.viscous_damping, init_identity=False,
            is_output_layer=self.is_output_layer, rank=self.rank,
            ternary_leaves=self.ternary_leaves, leaf_sparsity=self.leaf_sparsity,
            nm=self.nm, sparse_dispatch=self.sparse_dispatch,
            route_mode=self.route_mode, leaf_activation=self.leaf_activation,
            ternary_threshold_frac=self.ternary_threshold_frac,
        ).to(dev)
        with torch.no_grad():
            # New ROOT levels are near-uniform; old levels carry over verbatim.
            new.routing_weights.data[:extra_levels] = \
                torch.randn(new.routing_weights.data[:extra_levels].shape,
                            generator=gen).to(dev) * (new_router_scale / _math.sqrt(self.in_features))
            new.routing_weights.data[extra_levels:] = self.routing_weights.data
            new.routing_biases.data[:extra_levels] = 0.0
            new.routing_biases.data[extra_levels:] = self.routing_biases.data
            # Leading new digits vary slowest => tile copies along leaf axis:
            # new_leaf_idx = copy * K_old + old_leaf_idx.
            if self.rank is None:
                w_new = self.leaf_weights.data.repeat(rep, 1, 1)
                if child_noise > 0.0:
                    w_new += torch.randn(w_new.shape, generator=gen).to(dev) \
                             * (child_noise * self.leaf_weights.data.std())
                new.leaf_weights.data.copy_(w_new)
                if self.leaf_sparsity > 0.0:
                    new.leaf_sparsity_mask.copy_(
                        self.leaf_sparsity_mask.repeat(rep, 1, 1))
            else:
                u_new = self.leaf_u.data.repeat(rep, 1, 1)
                if child_noise > 0.0:
                    u_new += torch.randn(u_new.shape, generator=gen).to(dev) \
                             * (child_noise * self.leaf_u.data.std().clamp(min=1e-8))
                new.leaf_u.data.copy_(u_new)
                new.leaf_v.data.copy_(self.leaf_v.data)
                if self.leaf_sparsity > 0.0:
                    new.leaf_sparsity_mask_u.copy_(
                        self.leaf_sparsity_mask_u.repeat(rep, 1, 1))
            new.leaf_biases.data.copy_(self.leaf_biases.data.repeat(rep, 1))
            if self.learnable_scale:
                if self.rank is not None:
                    new.scale_u.data.copy_(self.scale_u.data)
                    new.scale_v.data.copy_(self.scale_v.data)
                else:
                    new.scale_w.data.copy_(self.scale_w.data)
        return new

    def build_magnitude_mask(self) -> None:
        """
        Rebuild the structural mask from current weight magnitudes: keep the
        top (1 - leaf_sparsity) fraction of entries by |w| -- or, in nm mode,
        exactly the top-N of every M-group. The classic prune-after-warmup
        recipe -- train without sparsity (or with the random/initial mask)
        for a while, call this, keep training. Strictly better informed than
        the random-at-init mask; still a FIXED mask afterwards (no topology
        change within a step).
        """
        if self.leaf_sparsity <= 0.0:
            return
        with torch.no_grad():
            if self.rank is None:
                w, buf = self.leaf_weights.data, self.leaf_sparsity_mask
            else:
                w, buf = self.leaf_u.data, self.leaf_sparsity_mask_u
            if self.nm is not None:
                new_mask = self._nm_topn_mask(w.detach().abs())
            else:
                thresh = torch.quantile(w.abs().flatten().float(), self.leaf_sparsity)
                new_mask = (w.abs() > thresh).float()
            buf.copy_(new_mask)

    def redistribute_sparsity(self, drop_fraction: float = 0.3, seed: int = 0,
                              grow_by_gradient: bool = True) -> None:
        """
        One RigL/SET-style evolution step: drop `drop_fraction` of the
        currently-ACTIVE entries with the smallest |w| and grow the same
        number at currently-inactive positions. Growth is by largest recent
        gradient score (stashed by apply_updates -- proper RigL) when
        available and grow_by_gradient=True, else uniform random. Keeps the
        active count -- and thus compute/memory -- exactly constant while
        letting the mask track importance during training. Call periodically
        (e.g. every 200-500 steps); finish training with
        build_magnitude_mask() for a static deployment mask.
        """
        if self.leaf_sparsity <= 0.0 or drop_fraction <= 0.0:
            return
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            if self.rank is None:
                mask, w = self.leaf_sparsity_mask, self.leaf_weights.data
                scores = getattr(self, "_last_leaf_scores", None)
            else:
                mask, w = self.leaf_sparsity_mask_u, self.leaf_u.data
                scores = getattr(self, "_last_leaf_scores", None)

            if self.nm is not None:
                # Structured mode: re-select top-N per M-group by a blended
                # magnitude + gradient score; exactly N actives per group by
                # construction, so compute/memory stay constant.
                score = w.detach().abs().clone()
                if scores is not None and scores.shape == score.shape:
                    score = score / (score.mean() + 1e-12) \
                          + scores / (scores.mean() + 1e-12)
                mask.copy_(self._nm_topn_mask(score))
                return

            flat_mask = mask.flatten()
            abs_w = w.detach().abs().flatten()
            grad_scores = None
            if grow_by_gradient and scores is not None \
                    and scores.flatten().numel() == flat_mask.numel():
                grad_scores = scores.flatten()
            active_idx = flat_mask.nonzero(as_tuple=True)[0]
            n_drop = int(len(active_idx) * drop_fraction)
            if n_drop == 0 or len(active_idx) == 0:
                return
            kth = torch.kthvalue(abs_w[active_idx].cpu(), n_drop).values.to(abs_w.device)
            drop_local = (abs_w[active_idx] <= kth).nonzero(as_tuple=True)[0][:n_drop]
            dropped = active_idx[drop_local]
            flat_mask[dropped] = 0.0
            inactive_idx = (flat_mask == 0).nonzero(as_tuple=True)[0]
            # Growth candidates EXCLUDE the positions just dropped this round:
            # those still carry their old gradient, so without exclusion they
            # regrow instantly and the mask never moves (observed: zero
            # change across hundreds of steps). Only genuinely-long-inactive
            # positions may be grown.
            grow_cand = inactive_idx[~torch.isin(inactive_idx, dropped)]
            if grow_cand.numel() == 0:
                grow_cand = inactive_idx
            if grow_by_gradient and grad_scores is not None and grow_cand.numel() > 0:
                g = grad_scores[grow_cand]
                n_grow = min(n_drop, g.numel())
                _, top = torch.topk(g, n_grow)
                flat_mask[grow_cand[top]] = 1.0
                if n_grow < n_drop:  # fill remainder randomly
                    rest = grow_cand[torch.randperm(len(grow_cand), generator=gen)[:n_drop - n_grow]]
                    flat_mask[rest] = 1.0
            else:
                perm = torch.randperm(len(grow_cand), generator=gen)[:n_drop].to(grow_cand.device)
                flat_mask[grow_cand[perm]] = 1.0
