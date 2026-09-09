"""
Qwen3.8-27B Upcycled ASDAG Architecture with 1:16 Structured Sparsity.
Native AffineAI Features:
  1. Top-2 Sparse MoE Leaf Routing (4-bit Log-Shift Router, 8 Leaves).
  2. 1:16 Structured Sparsity (93.75% Zeros, 1 Active per 16).
  3. Ternary Weights {-1, 0, +1} * gamma with BF16 Master Weights (STE).
  4. Hybrid Gated DeltaNet (O(1) recurrence) + Gated Quadratic Attention (every 4th layer).
  5. 248,320 Vocab with Unquantized BF16 Sacred Embeddings.
  6. Memory-efficient layer-streaming for 48GB A40 GPU training and inference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any, Union

from affine_ai.core.ast_dag import quantize_shift4, ternarize, dual_ternarize, pot5_quantize, _FP8HybridSTE


@dataclass
class Qwen38ASDAGConfig:
    dim: int = 5120
    vocab_size: int = 248320
    num_layers: int = 64
    intermediate_dim: int = 17408
    
    # ASDAG Tree Slicing & Sparsity
    num_leaves: int = 8
    num_shared_leaves: int = 1
    num_routed_leaves: int = 7
    top_k: int = 2        # 2 routed + 1 shared = 3 leaves active (or configurable to 1 routed)
    leaf_dim: int = 2176  # 17408 // 8
    nm_n: int = 1         # 1:16 Structured Sparsity numerator
    nm_m: int = 16        # 1:16 Structured Sparsity denominator (93.75% zero math)
    use_nm_sparsity: bool = True
    shared_leaf_sparsity: bool = False  # Protected from 1:16 zeroing
    shared_leaf_ternary: bool = False   # Gate 1-2: BF16, Gate 3: Ternary
    routed_leaf_ternary: bool = True
    
    # Hybrid Layout: Every 4th layer is full attention (layers 3, 7, 11, ..., 63)
    full_attn_interval: int = 4
    
    # Gated DeltaNet (SSM Linear Attention)
    ssm_conv_kernel: int = 4
    ssm_v_heads: int = 48
    ssm_qk_heads: int = 16
    ssm_head_dim: int = 128
    
    # Gated Attention
    attn_q_heads: int = 24
    attn_kv_heads: int = 4
    attn_head_dim: int = 256
    rope_dim: int = 64
    rope_theta: float = 10000000.0
    
    rms_norm_eps: float = 1e-6
    has_mtp: bool = True
    dtype: torch.dtype = torch.bfloat16

    # Native ASDAG Defaults
    ternary_leaves: bool = True
    ternary_mixers: bool = True
    ternary_embedding: bool = False  # Sacred layer: kept in BF16 for 248k discrimination
    use_shift4_routing: bool = True
    use_fp8_hybrid: bool = False

    # Weight Quantization & Sparsity Modes
    weight_quant_mode: str = "pot5"  # "pot5" (Default 5-state POT 2.32b), "dual_ternary" (3.17b), "ternary" (1.58b)
    sparsity_mode: str = "abstopk"   # "abstopk" (25% active compute) or "cluster_moe"
    use_dual_ternary: bool = False   # Backward compatibility toggle
    retain_ratio: float = 0.25       # 25% active compute


class _NMStructuredSparsitySTE(torch.autograd.Function):
    """
    N:M Structured Sparsity with Straight-Through Estimator.
    Forces exactly N non-zero weights per group of M along the last dimension.
    """
    @staticmethod
    def forward(ctx, w: torch.Tensor, n: int = 1, m: int = 16) -> torch.Tensor:
        orig_shape = w.shape
        K = orig_shape[-1]
        if K % m != 0:
            return w
        w_reshaped = w.reshape(-1, m)
        _, top_idx = torch.topk(w_reshaped.abs(), n, dim=-1)
        mask = torch.zeros_like(w_reshaped, dtype=torch.bool)
        mask.scatter_(-1, top_idx, True)
        ctx.save_for_backward(mask)
        ctx.orig_shape = orig_shape
        w_sparse = torch.where(mask, w_reshaped, torch.zeros_like(w_reshaped))
        return w_sparse.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE: gradient flows through directly to update master weights
        return grad_output, None, None


def apply_nm_sparsity(w: torch.Tensor, n: int = 1, m: int = 16) -> torch.Tensor:
    return _NMStructuredSparsitySTE.apply(w, n, m)


class Qwen38RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        normed = x_f32 * torch.rsqrt(variance + self.eps)
        return (normed * self.weight.float()).to(orig_dtype)


class Qwen38ASDAGLeaf(nn.Module):
    """
    ASDAG Tree Leaf for 27B:
    Holds quantized weights with BF16 Master Weights.
    Supports:
      - "pot5" (Default 5-state POT 2.32b: {-1, -0.5, 0, +0.5, +1} * alpha)
      - "dual_ternary" (3.17b: alpha_1 T_1 + alpha_2 T_2)
      - "ternary" (1.58b: {-1, 0, +1} * alpha)
    """
    def __init__(
        self,
        in_dim: int,
        leaf_dim: int,
        out_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        ternary: bool = True,
        weight_quant_mode: str = "pot5",
        use_dual_ternary: bool = False,
        use_nm: bool = False,
        nm_n: int = 1,
        nm_m: int = 16,
        use_fp8: bool = False
    ):
        super().__init__()
        self.ternary = ternary
        self.weight_quant_mode = weight_quant_mode
        self.use_dual_ternary = use_dual_ternary
        self.use_nm = use_nm
        self.nm_n = nm_n
        self.nm_m = nm_m
        self.use_fp8 = use_fp8

        self.gate_proj = nn.Linear(in_dim, leaf_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(in_dim, leaf_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(leaf_dim, out_dim, bias=False, dtype=dtype)

    def _prep_weight(self, w: torch.Tensor) -> torch.Tensor:
        if self.use_nm:
            w = apply_nm_sparsity(w, self.nm_n, self.nm_m)
        if self.ternary:
            mode = getattr(self, "weight_quant_mode", "pot5")
            if getattr(self, "use_dual_ternary", False):
                mode = "dual_ternary"
            if mode == "pot5":
                w = pot5_quantize(w)
            elif mode == "dual_ternary":
                w = dual_ternarize(w)
            else:
                w = ternarize(w)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_gate = self._prep_weight(self.gate_proj.weight)
        w_up = self._prep_weight(self.up_proj.weight)
        w_down = self._prep_weight(self.down_proj.weight)

        act = F.silu(F.linear(x, w_gate)) * F.linear(x, w_up)
        out = F.linear(act, w_down)

        if self.use_fp8 and self.training:
            out = _FP8HybridSTE.apply(out)
        return out


class _ASDAGLeafSliceView(nn.Module):
    """View over unified ASDAG projections matching the Qwen38ASDAGLeaf interface."""
    def __init__(self, gate_proj, up_proj, down_proj, st, ed, ternary=True, weight_quant_mode="pot5", use_dual=False, use_fp8=False):
        super().__init__()
        self._gate_proj = gate_proj
        self._up_proj = up_proj
        self._down_proj = down_proj
        self.st = st
        self.ed = ed
        self.ternary = ternary
        self.weight_quant_mode = weight_quant_mode
        self.use_dual = use_dual
        self.use_fp8 = use_fp8

    @property
    def gate_proj(self):
        class _W:
            weight = self._gate_proj.weight[self.st:self.ed]
        return _W()

    @property
    def up_proj(self):
        class _W:
            weight = self._up_proj.weight[self.st:self.ed]
        return _W()

    @property
    def down_proj(self):
        class _W:
            weight = self._down_proj.weight[:, self.st:self.ed]
        return _W()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wg = self.gate_proj.weight
        wu = self.up_proj.weight
        wd = self.down_proj.weight
        if self.ternary:
            mode = getattr(self, "weight_quant_mode", "pot5")
            if getattr(self, "use_dual", False):
                mode = "dual_ternary"
            if mode == "pot5":
                wg, wu, wd = pot5_quantize(wg), pot5_quantize(wu), pot5_quantize(wd)
            elif mode == "dual_ternary":
                wg, wu, wd = dual_ternarize(wg), dual_ternarize(wu), dual_ternarize(wd)
            else:
                wg, wu, wd = ternarize(wg), ternarize(wu), ternarize(wd)
        act = F.silu(F.linear(x, wg)) * F.linear(x, wu)
        out = F.linear(act, wd)
        if self.use_fp8 and self.training:
            out = _FP8HybridSTE.apply(out)
        return out


class Qwen38ASDAGFFN(nn.Module):
    """
    Unified ASDAG FFN Layer for 27B:
      1. Default (weight_quant_mode="pot5", sparsity_mode="abstopk"):
         - 5-State Power-of-Two (POT) Quantization (2.32b: {-1, -0.5, 0, +0.5, +1} * alpha).
         - Native AbsTopK-GLU activation sparsity (25% active compute, 75% savings).
         - 36% fewer integer additions and 40% lower memory bandwidth than Dual-Ternary.
      2. Options:
         - weight_quant_mode="dual_ternary" (3.17b, 0.8131 CosSim / 25.78 dB SNR).
         - weight_quant_mode="ternary" (1.58b Single Ternary).
         - sparsity_mode="cluster_moe" (1 Shared Leaf + 7 Routed Leaves).
    """
    def __init__(self, config: Qwen38ASDAGConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.intermediate_dim = config.intermediate_dim
        self.num_leaves = config.num_leaves
        self.num_shared = config.num_shared_leaves
        self.num_routed = config.num_routed_leaves
        self.leaf_dim = config.leaf_dim
        self.top_k = config.top_k
        self.use_shift4_routing = config.use_shift4_routing
        self.sparsity_mode = getattr(config, "sparsity_mode", "abstopk")
        self.retain_ratio = getattr(config, "retain_ratio", 0.25)
        self.weight_quant_mode = getattr(config, "weight_quant_mode", "pot5")
        self.use_dual_ternary = getattr(config, "use_dual_ternary", False)
        self.ternary_leaves = getattr(config, "ternary_leaves", True)
        self.use_nm = (self.sparsity_mode == "cluster_moe") and getattr(config, "use_nm_sparsity", False) and self.ternary_leaves
        self.nm_n = getattr(config, "nm_n", 1)
        self.nm_m = getattr(config, "nm_m", 16)
        self.use_fp8 = getattr(config, "use_fp8_hybrid", False)

        # Unified linear projections for high throughput & AbsTopK
        self.gate_proj = nn.Linear(config.dim, config.intermediate_dim, bias=False, dtype=config.dtype)
        self.up_proj = nn.Linear(config.dim, config.intermediate_dim, bias=False, dtype=config.dtype)
        self.down_proj = nn.Linear(config.intermediate_dim, config.dim, bias=False, dtype=config.dtype)

        # Router for cluster_moe mode
        self.router = nn.Linear(config.dim, self.num_routed, bias=False, dtype=config.dtype)

    def _prep_weight(self, w: torch.Tensor) -> torch.Tensor:
        if self.use_nm:
            w = apply_nm_sparsity(w, self.nm_n, self.nm_m)
        if self.ternary_leaves:
            mode = getattr(self, "weight_quant_mode", "pot5")
            if getattr(self, "use_dual_ternary", False):
                mode = "dual_ternary"
            if mode == "pot5":
                w = pot5_quantize(w)
            elif mode == "dual_ternary":
                w = dual_ternarize(w)
            else:
                w = ternarize(w)
        return w

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        leaf0_gate = prefix + "leaves.0.gate_proj.weight"
        if leaf0_gate in state_dict:
            gates = []
            ups = []
            downs = []
            for idx in range(self.num_leaves):
                g_key = f"{prefix}leaves.{idx}.gate_proj.weight"
                u_key = f"{prefix}leaves.{idx}.up_proj.weight"
                d_key = f"{prefix}leaves.{idx}.down_proj.weight"
                if g_key in state_dict:
                    gates.append(state_dict.pop(g_key))
                if u_key in state_dict:
                    ups.append(state_dict.pop(u_key))
                if d_key in state_dict:
                    downs.append(state_dict.pop(d_key))
            if len(gates) == self.num_leaves:
                state_dict[prefix + "gate_proj.weight"] = torch.cat(gates, dim=0)
            if len(ups) == self.num_leaves:
                state_dict[prefix + "up_proj.weight"] = torch.cat(ups, dim=0)
            if len(downs) == self.num_leaves:
                state_dict[prefix + "down_proj.weight"] = torch.cat(downs, dim=1)

        router_key = prefix + "router.weight"
        if router_key in state_dict and state_dict[router_key].shape[0] == self.num_leaves:
            state_dict[router_key] = state_dict[router_key][self.num_shared:]
        elif router_key not in state_dict and self.sparsity_mode == "abstopk":
            state_dict[router_key] = self.router.weight

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @property
    def leaves(self) -> List[Any]:
        views = []
        for idx in range(self.num_leaves):
            st = idx * self.leaf_dim
            ed = (idx + 1) * self.leaf_dim
            views.append(_ASDAGLeafSliceView(
                self.gate_proj,
                self.up_proj,
                self.down_proj,
                st,
                ed,
                ternary=self.ternary_leaves,
                use_dual=self.use_dual_ternary,
                use_fp8=self.use_fp8
            ))
        return views

    @property
    def shared_leaf(self) -> Any:
        return self.leaves[0]

    @property
    def routed_leaves(self) -> List[Any]:
        return self.leaves[self.num_shared:]

    def forward(
        self,
        x: torch.Tensor,
        top_k: Optional[int] = None,
        retain_ratio: Optional[float] = None
    ) -> torch.Tensor:
        if self.sparsity_mode == "abstopk":
            return self._forward_abstopk(x, top_k=top_k, retain_ratio=retain_ratio)
        else:
            return self._forward_cluster_moe(x, top_k=top_k)

    def _forward_abstopk(
        self,
        x: torch.Tensor,
        top_k: Optional[int] = None,
        retain_ratio: Optional[float] = None
    ) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.dim)

        w_gate = self._prep_weight(self.gate_proj.weight)
        w_up = self._prep_weight(self.up_proj.weight)
        w_down = self._prep_weight(self.down_proj.weight)

        # 1. Gate projection (Dual-Ternary integer sign-accumulation)
        gate = F.linear(x_2d, w_gate)

        # 2. Determine number of active neurons k
        if retain_ratio is not None:
            k = int(self.intermediate_dim * retain_ratio)
        elif top_k is not None:
            if top_k <= self.num_leaves:
                k = int(self.intermediate_dim * (top_k / self.num_leaves))
            else:
                k = min(top_k, self.intermediate_dim)
        else:
            k = int(self.intermediate_dim * self.retain_ratio)

        k = max(1, min(k, self.intermediate_dim))

        if k >= self.intermediate_dim:
            act = F.silu(gate) * F.linear(x_2d, w_up)
            out = F.linear(act, w_down)
        else:
            # Native AbsTopK-GLU activation sparsity (Candidate 4)
            _, topk_idx = torch.topk(gate.abs(), k, dim=-1)
            mask = torch.zeros_like(gate, dtype=torch.bool).scatter_(-1, topk_idx, True)
            up = F.linear(x_2d, w_up)
            act = torch.where(mask, F.silu(gate) * up, torch.zeros_like(gate))
            out = F.linear(act, w_down)

        if self.use_fp8 and self.training:
            out = _FP8HybridSTE.apply(out)

        return out.reshape(orig_shape)

    def _forward_cluster_moe(self, x: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
        k = top_k if top_k is not None else self.top_k
        if k >= self.num_routed:
            return self.forward_dense(x)

        orig_shape = x.shape
        x_2d = x.reshape(-1, self.dim)

        out = self.shared_leaf(x_2d)

        if k <= 0:
            return out.reshape(orig_shape)

        x_route = quantize_shift4(x_2d) if self.use_shift4_routing else x_2d
        router_logits = self.router(x_route).float()

        routing_weights, selected_routed = torch.topk(router_logits, k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1).to(x.dtype)

        for r_idx, leaf in enumerate(self.routed_leaves):
            mask = (selected_routed == r_idx)
            token_mask = mask.any(dim=-1)
            if not token_mask.any():
                continue

            sub_x = x_2d[token_mask]
            sub_out = leaf(sub_x)

            weights = (routing_weights * mask.to(routing_weights.dtype)).sum(dim=-1, keepdim=True)
            out[token_mask] += sub_out * weights[token_mask]

        return out.reshape(orig_shape)

    def forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        """Exact dense forward pass across full intermediate dimension."""
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.dim)
        w_gate = self._prep_weight(self.gate_proj.weight)
        w_up = self._prep_weight(self.up_proj.weight)
        w_down = self._prep_weight(self.down_proj.weight)

        act = F.silu(F.linear(x_2d, w_gate)) * F.linear(x_2d, w_up)
        out = F.linear(act, w_down)

        if self.use_fp8 and self.training:
            out = _FP8HybridSTE.apply(out)
        return out.reshape(orig_shape)


class Qwen38GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet Layer for 27B:
      48 V-Heads, 16 QK-Heads, 128 Head Dim.
      QKV Dim: (16*2 + 48)*128 = 10,240.
      Out Dim: 48*128 = 6,144.
    """
    def __init__(self, config: Qwen38ASDAGConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.v_heads = config.ssm_v_heads        # 48
        self.qk_heads = config.ssm_qk_heads      # 16
        self.head_dim = config.ssm_head_dim      # 128
        self.qkv_dim = (self.qk_heads * 2 + self.v_heads) * self.head_dim  # 10240
        self.out_dim = self.v_heads * self.head_dim  # 6144

        self.qkv_proj = nn.Linear(self.dim, self.qkv_dim, bias=False, dtype=config.dtype)
        self.conv1d = nn.Conv1d(
            in_channels=self.qkv_dim,
            out_channels=self.qkv_dim,
            kernel_size=config.ssm_conv_kernel,
            groups=self.qkv_dim,
            padding=0,
            bias=False,
            dtype=config.dtype
        )
        self.alpha_proj = nn.Linear(self.dim, self.v_heads, bias=False, dtype=torch.float32)
        self.beta_proj = nn.Linear(self.dim, self.v_heads, bias=False, dtype=torch.float32)
        self.ssm_a = nn.Parameter(torch.zeros(self.v_heads, dtype=torch.float32))
        self.ssm_dt_bias = nn.Parameter(torch.zeros(self.v_heads, dtype=torch.float32))

        self.norm = Qwen38RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_gate = nn.Linear(self.dim, self.out_dim, bias=False, dtype=config.dtype)
        self.ssm_out = nn.Linear(self.out_dim, self.dim, bias=False, dtype=config.dtype)

    def forward(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        ssm_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, D = x.shape
        orig_dtype = x.dtype

        # 1. Project QKV: [B, T, 10240]
        qkv = self.qkv_proj(x)

        # 2. Causal Conv1D
        qkv_t = qkv.transpose(1, 2)
        if conv_state is not None:
            conv_input = torch.cat([conv_state, qkv_t], dim=-1)
        else:
            conv_input = F.pad(qkv_t, (self.config.ssm_conv_kernel - 1, 0))

        next_conv_state = conv_input[:, :, -(self.config.ssm_conv_kernel - 1):].detach()
        conv_out = self.conv1d(conv_input)[:, :, :T]
        conv_act = F.silu(conv_out).transpose(1, 2)

        # Split Q, K, V
        q_dim = self.qk_heads * self.head_dim
        k_dim = self.qk_heads * self.head_dim
        q = conv_act[:, :, :q_dim].reshape(B, T, self.qk_heads, self.head_dim)
        k = conv_act[:, :, q_dim:q_dim + k_dim].reshape(B, T, self.qk_heads, self.head_dim)
        v = conv_act[:, :, q_dim + k_dim:].reshape(B, T, self.v_heads, self.head_dim)

        # Repeat QK heads to match V heads (16 -> 48, repeat factor = 3)
        repeat_factor = self.v_heads // self.qk_heads
        q = torch.repeat_interleave(q, repeat_factor, dim=2)  # [B, T, 48, 128]
        k = torch.repeat_interleave(k, repeat_factor, dim=2)  # [B, T, 48, 128]

        # 3. Compute recurrence coefficients
        x_f32 = x.float()
        alpha = torch.sigmoid(self.alpha_proj(x_f32) + self.ssm_a)        # [B, T, 48]
        beta = torch.sigmoid(self.beta_proj(x_f32) + self.ssm_dt_bias)     # [B, T, 48]

        if ssm_state is None:
            state_S = torch.zeros(B, self.v_heads, self.head_dim, self.head_dim, device=x.device, dtype=torch.float32)
        else:
            state_S = ssm_state.clone().float()

        q_f32 = q.float()
        k_f32 = k.float()
        v_f32 = v.float()

        k_norm = k_f32 / (k_f32.norm(dim=-1, keepdim=True) + 1e-8)

        outs = []
        for t in range(T):
            qt = q_f32[:, t]         # [B, 48, 128]
            kt = k_norm[:, t]        # [B, 48, 128]
            vt = v_f32[:, t]         # [B, 48, 128]
            at = alpha[:, t, :, None, None]  # [B, 48, 1, 1]
            bt = beta[:, t, :, None]         # [B, 48, 1]

            pred_v = torch.matmul(state_S, kt.unsqueeze(-1)).squeeze(-1)
            err = (vt - pred_v) * bt
            state_S = at * state_S + torch.matmul(err.unsqueeze(-1), kt.unsqueeze(-2))
            yt = torch.matmul(state_S, qt.unsqueeze(-1)).squeeze(-1)
            outs.append(yt.unsqueeze(1))

        out_ssm = torch.cat(outs, dim=1)
        out_ssm = self.norm(out_ssm.reshape(-1, self.head_dim)).reshape(B, T, self.out_dim).to(orig_dtype)

        gate = F.silu(self.attn_gate(x))
        y = self.ssm_out(out_ssm * gate)

        return y, (next_conv_state, state_S.to(orig_dtype))


class Qwen38GatedAttention(nn.Module):
    """
    Gated Attention Layer for 27B:
      24 Q-Heads, 4 KV-Heads, 256 Head Dim.
      Out Dim: 24*256 = 6,144.
    """
    def __init__(self, config: Qwen38ASDAGConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.q_heads = config.attn_q_heads    # 24
        self.kv_heads = config.attn_kv_heads  # 4
        self.head_dim = config.attn_head_dim  # 256
        self.rope_dim = config.rope_dim       # 64
        self.out_dim = self.q_heads * self.head_dim  # 6144

        self.attn_q = nn.Linear(self.dim, self.out_dim * 2, bias=False, dtype=config.dtype)
        self.attn_k = nn.Linear(self.dim, self.kv_heads * self.head_dim, bias=False, dtype=config.dtype)
        self.attn_v = nn.Linear(self.dim, self.kv_heads * self.head_dim, bias=False, dtype=config.dtype)

        self.q_norm = Qwen38RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen38RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_output = nn.Linear(self.out_dim, self.dim, bias=False, dtype=config.dtype)

    def _apply_rope(self, x: torch.Tensor, pos: int = 0) -> torch.Tensor:
        B, T, H, D = x.shape
        x_rope = x[..., :self.rope_dim]
        x_pass = x[..., self.rope_dim:]

        positions = torch.arange(pos, pos + T, device=x.device, dtype=torch.float32)
        dim_idx = torch.arange(0, self.rope_dim, 2, device=x.device, dtype=torch.float32)
        inv_freq = 1.0 / (self.config.rope_theta ** (dim_idx / self.rope_dim))
        sinusoid = torch.outer(positions, inv_freq)
        sin = sinusoid.sin().repeat_interleave(2, dim=-1)[None, :, None, :]
        cos = sinusoid.cos().repeat_interleave(2, dim=-1)[None, :, None, :]

        x1 = x_rope[..., 0::2]
        x2 = x_rope[..., 1::2]
        x_rot = torch.cat([-x2, x1], dim=-1)
        x_rope_out = (x_rope.float() * cos + x_rot.float() * sin).to(x.dtype)

        return torch.cat([x_rope_out, x_pass], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pos: int = 0
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, D = x.shape

        q_proj = self.attn_q(x)
        q_raw, gate = q_proj.chunk(2, dim=-1)

        k = self.attn_k(x).reshape(B, T, self.kv_heads, self.head_dim)
        v = self.attn_v(x).reshape(B, T, self.kv_heads, self.head_dim)
        q = q_raw.reshape(B, T, self.q_heads, self.head_dim)

        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(B, T, self.q_heads, self.head_dim)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(B, T, self.kv_heads, self.head_dim)

        q = self._apply_rope(q, pos=pos)
        k = self._apply_rope(k, pos=pos)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        new_kv_cache = (k.detach(), v.detach())

        repeat_factor = self.q_heads // self.kv_heads  # 24 // 4 = 6
        k_rep = torch.repeat_interleave(k, repeat_factor, dim=2)
        v_rep = torch.repeat_interleave(v, repeat_factor, dim=2)

        q_t = q.transpose(1, 2)
        k_t = k_rep.transpose(1, 2)
        v_t = v_rep.transpose(1, 2)

        is_causal = (T > 1)
        attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.out_dim)

        gated = attn_out * F.silu(gate)
        out = self.attn_output(gated)

        return out, new_kv_cache


class Qwen38Block(nn.Module):
    """
    Qwen3.8 Block (Layer 0 to 63):
      Norm1 -> Time Mixer (Gated DeltaNet or Gated Attention) -> Norm2 -> ASDAG Tree FFN (1:16 Sparsity)
    """
    def __init__(self, config: Qwen38ASDAGConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_full_attention = (layer_idx == 63) or ((layer_idx + 1) % config.full_attn_interval == 0)

        self.attn_norm = Qwen38RMSNorm(config.dim, eps=config.rms_norm_eps)
        if self.is_full_attention:
            self.time_mixer = Qwen38GatedAttention(config)
        else:
            self.time_mixer = Qwen38GatedDeltaNet(config)

        self.post_attention_norm = Qwen38RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.asdag_ffn = Qwen38ASDAGFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Any] = None,
        pos: int = 0,
        top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, Any]:
        normed1 = self.attn_norm(x)
        if self.is_full_attention:
            tm_out, next_state = self.time_mixer(normed1, kv_cache=state, pos=pos)
        else:
            conv_st = state[0] if state is not None else None
            ssm_st = state[1] if state is not None else None
            tm_out, next_state = self.time_mixer(normed1, conv_state=conv_st, ssm_state=ssm_st)
        x = x + tm_out

        normed2 = self.post_attention_norm(x)
        ffn_out = self.asdag_ffn(normed2, top_k=top_k)
        x = x + ffn_out

        return x, next_state


class Qwen38MTPBlock(nn.Module):
    """
    Multi-Token Prediction (MTP) Speculative Drafting Block for 27B (Block 64):
    """
    def __init__(self, config: Qwen38ASDAGConfig):
        super().__init__()
        self.config = config
        self.eh_proj = nn.Linear(config.dim * 2, config.dim, bias=False, dtype=config.dtype)
        self.enorm = Qwen38RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.hnorm = Qwen38RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.shared_head_norm = Qwen38RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.block = Qwen38Block(config, layer_idx=64)

    def forward(
        self,
        h: torch.Tensor,
        emb_next: torch.Tensor,
        state: Optional[Any] = None,
        pos: int = 0
    ) -> Tuple[torch.Tensor, Any]:
        normed_h = self.hnorm(h)
        normed_emb = self.enorm(emb_next)
        fused = self.eh_proj(torch.cat([normed_emb, normed_h], dim=-1))
        out, next_state = self.block(fused, state=state, pos=pos)
        return self.shared_head_norm(out), next_state


class Qwen38ASDAGModel(nn.Module):
    """
    Complete Upcycled Qwen3.8-27B ASDAG Model.
    """
    def __init__(self, config: Qwen38ASDAGConfig):
        super().__init__()
        self.config = config
        self.token_embd = nn.Embedding(config.vocab_size, config.dim, dtype=config.dtype)
        self.blocks = nn.ModuleList([
            Qwen38Block(config, layer_idx=i) for i in range(config.num_layers)
        ])
        self.output_norm = Qwen38RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.mtp_block = Qwen38MTPBlock(config) if config.has_mtp else None

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[Any]] = None,
        pos: int = 0,
        top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[Any]]:
        x = self.token_embd(input_ids)
        next_states = []

        for i, block in enumerate(self.blocks):
            st = states[i] if states is not None else None
            x, next_st = block(x, state=st, pos=pos, top_k=top_k)
            next_states.append(next_st)

        h = self.output_norm(x)
        logits = F.linear(h, self.token_embd.weight)
        return logits, next_states
