import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any, Union


@dataclass
class Qwen35ASDAGConfig:
    dim: int = 2560
    vocab_size: int = 248320
    num_layers: int = 32
    intermediate_dim: int = 9216
    
    # ASDAG Tree Slicing
    num_leaves: int = 8
    top_k: int = 2
    leaf_dim: int = 1152  # 9216 // 8
    
    # Hybrid Layout: Every 4th layer is full attention (layers 3, 7, 11, 15, 19, 23, 27, 31)
    full_attn_interval: int = 4
    
    # Gated DeltaNet (SSM Linear Attention)
    ssm_conv_kernel: int = 4
    ssm_v_heads: int = 32
    ssm_qk_heads: int = 16
    ssm_head_dim: int = 128
    
    # Gated Attention
    attn_q_heads: int = 16
    attn_kv_heads: int = 4
    attn_head_dim: int = 256
    rope_dim: int = 64
    rope_theta: float = 10000000.0
    
    rms_norm_eps: float = 1e-6
    has_mtp: bool = True
    dtype: torch.dtype = torch.bfloat16


class Qwen35RMSNorm(nn.Module):
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


class Qwen35ASDAGLeaf(nn.Module):
    """A single ASDAG leaf containing a slice of the SwiGLU intermediate channels."""
    def __init__(self, in_dim: int, leaf_dim: int, out_dim: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.gate_proj = nn.Linear(in_dim, leaf_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(in_dim, leaf_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(leaf_dim, out_dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen35ASDAGFFN(nn.Module):
    """
    ASDAG Tree FFN:
    Partitions the 9216 intermediate channels into K leaves (e.g. 8 leaves x 1152 channels).
    When evaluating all leaves (top_k=K), it is mathematically identical to the dense SwiGLU FFN.
    When evaluating top_k < K, it slashes intermediate compute proportionally (e.g. 75% for top_k=2).
    """
    def __init__(self, config: Qwen35ASDAGConfig):
        super().__init__()
        self.config = config
        self.num_leaves = config.num_leaves
        self.leaf_dim = config.leaf_dim
        self.top_k = config.top_k
        self.dim = config.dim

        self.leaves = nn.ModuleList([
            Qwen35ASDAGLeaf(config.dim, config.leaf_dim, config.dim, dtype=config.dtype)
            for _ in range(self.num_leaves)
        ])
        self.router = nn.Linear(config.dim, self.num_leaves, bias=False, dtype=config.dtype)

    def forward(self, x: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass with sparse routing to Top-k leaves.
        """
        k = top_k if top_k is not None else self.top_k
        if k >= self.num_leaves:
            return self.forward_dense(x)

        orig_shape = x.shape
        x_2d = x.reshape(-1, self.dim)
        N = x_2d.shape[0]

        # Route tokens
        router_logits = self.router(x_2d).float()  # [N, num_leaves]
        routing_weights, selected_leaves = torch.topk(router_logits, k, dim=-1)  # [N, k]
        routing_weights = F.softmax(routing_weights, dim=-1).to(x.dtype)

        # Evaluate selected leaves
        out = torch.zeros_like(x_2d)
        for leaf_idx, leaf in enumerate(self.leaves):
            # Find tokens routed to this leaf
            mask = (selected_leaves == leaf_idx)  # [N, k]
            token_mask = mask.any(dim=-1)         # [N]
            if not token_mask.any():
                continue

            sub_x = x_2d[token_mask]
            sub_out = leaf(sub_x)  # [N_sub, dim]

            # Weight by router probability
            weights = (routing_weights * mask.to(routing_weights.dtype)).sum(dim=-1, keepdim=True)
            out[token_mask] += sub_out * weights[token_mask]

        return out.reshape(orig_shape)

    def forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        """Exact sum across all leaves (identical to full 9216-dim dense FFN)."""
        out = torch.zeros_like(x)
        for leaf in self.leaves:
            out += leaf(x)
        return out


class Qwen35GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet Layer (SSM Linear Attention with Delta Rule):
      1. Causal Conv1D (kernel_size=4, groups=8192) on QKV projections.
      2. Multi-head Delta recurrence with data-dependent decay alpha and rate beta.
      3. Head RMSNorm (128 dims).
      4. Output gating and linear projection back to hidden dimension.
    """
    def __init__(self, config: Qwen35ASDAGConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.v_heads = config.ssm_v_heads        # 32
        self.qk_heads = config.ssm_qk_heads      # 16
        self.head_dim = config.ssm_head_dim      # 128
        self.qkv_dim = (self.qk_heads * 2 + self.v_heads) * self.head_dim  # 8192
        self.out_dim = self.v_heads * self.head_dim  # 4096

        self.qkv_proj = nn.Linear(self.dim, self.qkv_dim, bias=False, dtype=config.dtype)
        self.conv1d = nn.Conv1d(
            in_channels=self.qkv_dim,
            out_channels=self.qkv_dim,
            kernel_size=config.ssm_conv_kernel,
            groups=self.qkv_dim,
            padding=0,  # Manual causal padding handled in forward
            bias=False,
            dtype=config.dtype
        )
        self.alpha_proj = nn.Linear(self.dim, self.v_heads, bias=False, dtype=torch.float32)
        self.beta_proj = nn.Linear(self.dim, self.v_heads, bias=False, dtype=torch.float32)
        self.ssm_a = nn.Parameter(torch.zeros(self.v_heads, dtype=torch.float32))
        self.ssm_dt_bias = nn.Parameter(torch.zeros(self.v_heads, dtype=torch.float32))

        self.norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_gate = nn.Linear(self.dim, self.out_dim, bias=False, dtype=config.dtype)
        self.ssm_out = nn.Linear(self.out_dim, self.dim, bias=False, dtype=config.dtype)

    def forward(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        ssm_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of Gated DeltaNet.
        x: [B, T, dim]
        Returns: output [B, T, dim], (next_conv_state, next_ssm_state)
        """
        B, T, D = x.shape
        orig_dtype = x.dtype

        # 1. Project QKV: [B, T, 8192]
        qkv = self.qkv_proj(x)

        # 2. Causal Conv1D
        qkv_t = qkv.transpose(1, 2)  # [B, 8192, T]
        if conv_state is not None:
            # Prepend saved conv state (last kernel_size - 1 tokens)
            conv_input = torch.cat([conv_state, qkv_t], dim=-1)
        else:
            conv_input = F.pad(qkv_t, (self.config.ssm_conv_kernel - 1, 0))

        next_conv_state = conv_input[:, :, -(self.config.ssm_conv_kernel - 1):].detach()
        conv_out = self.conv1d(conv_input)[:, :, :T]  # [B, 8192, T]
        conv_act = F.silu(conv_out).transpose(1, 2)   # [B, T, 8192]

        # Split Q, K, V
        # Q: [B, T, 16, 128], K: [B, T, 16, 128], V: [B, T, 32, 128]
        q_dim = self.qk_heads * self.head_dim
        k_dim = self.qk_heads * self.head_dim
        q = conv_act[:, :, :q_dim].reshape(B, T, self.qk_heads, self.head_dim)
        k = conv_act[:, :, q_dim:q_dim + k_dim].reshape(B, T, self.qk_heads, self.head_dim)
        v = conv_act[:, :, q_dim + k_dim:].reshape(B, T, self.v_heads, self.head_dim)

        # Repeat QK heads to match V heads (16 -> 32)
        q = torch.repeat_interleave(q, 2, dim=2)  # [B, T, 32, 128]
        k = torch.repeat_interleave(k, 2, dim=2)  # [B, T, 32, 128]

        # 3. Compute recurrence coefficients
        x_f32 = x.float()
        alpha = torch.sigmoid(self.alpha_proj(x_f32) + self.ssm_a)        # [B, T, 32]
        beta = torch.sigmoid(self.beta_proj(x_f32) + self.ssm_dt_bias)     # [B, T, 32]

        # 4. Delta Recurrence: S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_{t-1} k_t) k_t^T
        if ssm_state is None:
            state_S = torch.zeros(B, self.v_heads, self.head_dim, self.head_dim, device=x.device, dtype=torch.float32)
        else:
            state_S = ssm_state.clone().float()

        q_f32 = q.float()
        k_f32 = k.float()
        v_f32 = v.float()

        # Unit-normalize keys for stable delta rule
        k_norm = k_f32 / (k_f32.norm(dim=-1, keepdim=True) + 1e-8)

        outs = []
        for t in range(T):
            qt = q_f32[:, t]         # [B, 32, 128]
            kt = k_norm[:, t]        # [B, 32, 128]
            vt = v_f32[:, t]         # [B, 32, 128]
            at = alpha[:, t, :, None, None]  # [B, 32, 1, 1]
            bt = beta[:, t, :, None]         # [B, 32, 1]

            # Current state projection on key: S_{t-1} k_t
            pred_v = torch.matmul(state_S, kt.unsqueeze(-1)).squeeze(-1)  # [B, 32, 128]
            err = (vt - pred_v) * bt  # [B, 32, 128]

            # Delta update: S_t = alpha * S_{t-1} + err @ kt^T
            state_S = at * state_S + torch.matmul(err.unsqueeze(-1), kt.unsqueeze(-2))

            # Query state: y_t = S_t q_t
            yt = torch.matmul(state_S, qt.unsqueeze(-1)).squeeze(-1)  # [B, 32, 128]
            outs.append(yt.unsqueeze(1))

        out_ssm = torch.cat(outs, dim=1)  # [B, T, 32, 128]
        out_ssm = self.norm(out_ssm.reshape(-1, self.head_dim)).reshape(B, T, self.out_dim).to(orig_dtype)

        # 5. Output Gating & Projection
        gate = F.silu(self.attn_gate(x))
        y = self.ssm_out(out_ssm * gate)

        return y, (next_conv_state, state_S.to(orig_dtype))


class Qwen35GatedAttention(nn.Module):
    """
    Gated Attention Layer (Full Quadratic Attention with QK Norm and Rotary Embedding):
    Executed on layers 3, 7, 11, 15, 19, 23, 27, 31.
    """
    def __init__(self, config: Qwen35ASDAGConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.q_heads = config.attn_q_heads    # 16
        self.kv_heads = config.attn_kv_heads  # 4
        self.head_dim = config.attn_head_dim  # 256
        self.rope_dim = config.rope_dim       # 64
        self.out_dim = self.q_heads * self.head_dim  # 4096

        self.attn_q = nn.Linear(self.dim, self.out_dim * 2, bias=False, dtype=config.dtype)  # Q + Gate
        self.attn_k = nn.Linear(self.dim, self.kv_heads * self.head_dim, bias=False, dtype=config.dtype)
        self.attn_v = nn.Linear(self.dim, self.kv_heads * self.head_dim, bias=False, dtype=config.dtype)

        self.q_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_output = nn.Linear(self.out_dim, self.dim, bias=False, dtype=config.dtype)

    def _apply_rope(self, x: torch.Tensor, pos: int = 0) -> torch.Tensor:
        """Applies Rotary Position Embedding to first rope_dim dimensions."""
        B, T, H, D = x.shape
        x_rope = x[..., :self.rope_dim]
        x_pass = x[..., self.rope_dim:]

        positions = torch.arange(pos, pos + T, device=x.device, dtype=torch.float32)
        dim_idx = torch.arange(0, self.rope_dim, 2, device=x.device, dtype=torch.float32)
        inv_freq = 1.0 / (self.config.rope_theta ** (dim_idx / self.rope_dim))
        sinusoid = torch.outer(positions, inv_freq)  # [T, rope_dim//2]
        sin = sinusoid.sin().repeat_interleave(2, dim=-1)[None, :, None, :]  # [1, T, 1, rope_dim]
        cos = sinusoid.cos().repeat_interleave(2, dim=-1)[None, :, None, :]

        # Rotate pairs
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
        orig_dtype = x.dtype

        # 1. Project Q, Gate, K, V
        q_proj = self.attn_q(x)  # [B, T, 8192]
        q_raw, gate = q_proj.chunk(2, dim=-1)  # each [B, T, 4096]

        k = self.attn_k(x).reshape(B, T, self.kv_heads, self.head_dim)
        v = self.attn_v(x).reshape(B, T, self.kv_heads, self.head_dim)
        q = q_raw.reshape(B, T, self.q_heads, self.head_dim)

        # 2. QK Norm
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(B, T, self.q_heads, self.head_dim)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(B, T, self.kv_heads, self.head_dim)

        # 3. Apply RoPE
        q = self._apply_rope(q, pos=pos)
        k = self._apply_rope(k, pos=pos)

        # 4. KV Cache Update
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        new_kv_cache = (k.detach(), v.detach())

        # 5. GQA Scaled Dot-Product Attention
        k_rep = torch.repeat_interleave(k, self.q_heads // self.kv_heads, dim=2)  # [B, S, 16, 256]
        v_rep = torch.repeat_interleave(v, self.q_heads // self.kv_heads, dim=2)

        # Transpose for attention: [B, 16, T, 256]
        q_t = q.transpose(1, 2)
        k_t = k_rep.transpose(1, 2)
        v_t = v_rep.transpose(1, 2)

        is_causal = (T > 1)
        attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.out_dim)

        # 6. Output Gating & Projection
        gated = attn_out * F.silu(gate)
        out = self.attn_output(gated)

        return out, new_kv_cache


class Qwen35Block(nn.Module):
    """
    Qwen3.5 Block:
      Norm1 -> Time Mixer (DeltaNet or Attention) -> Norm2 -> ASDAG Tree FFN
    """
    def __init__(self, config: Qwen35ASDAGConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_full_attention = ((layer_idx + 1) % config.full_attn_interval == 0)

        self.attn_norm = Qwen35RMSNorm(config.dim, eps=config.rms_norm_eps)
        if self.is_full_attention:
            self.time_mixer = Qwen35GatedAttention(config)
        else:
            self.time_mixer = Qwen35GatedDeltaNet(config)

        self.post_attention_norm = Qwen35RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.asdag_ffn = Qwen35ASDAGFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Any] = None,
        pos: int = 0,
        top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, Any]:
        # 1. Time Mixer with residual
        normed1 = self.attn_norm(x)
        if self.is_full_attention:
            tm_out, next_state = self.time_mixer(normed1, kv_cache=state, pos=pos)
        else:
            conv_st = state[0] if state is not None else None
            ssm_st = state[1] if state is not None else None
            tm_out, next_state = self.time_mixer(normed1, conv_state=conv_st, ssm_state=ssm_st)
        x = x + tm_out

        # 2. ASDAG FFN with residual
        normed2 = self.post_attention_norm(x)
        ffn_out = self.asdag_ffn(normed2, top_k=top_k)
        x = x + ffn_out

        return x, next_state


class Qwen35MTPBlock(nn.Module):
    """
    Multi-Token Prediction (MTP) Speculative Drafting Block (Block 32):
    Predicts the subsequent token (t+1) from hidden state + next token embedding.
    """
    def __init__(self, config: Qwen35ASDAGConfig):
        super().__init__()
        self.config = config
        self.eh_proj = nn.Linear(config.dim * 2, config.dim, bias=False, dtype=config.dtype)
        self.enorm = Qwen35RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.hnorm = Qwen35RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.shared_head_norm = Qwen35RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.block = Qwen35Block(config, layer_idx=32)

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


class Qwen35ASDAGModel(nn.Module):
    """
    Complete Upcycled Qwen3.5-4B Model with AffineAI ASDAG Architecture.
    """
    def __init__(self, config: Qwen35ASDAGConfig):
        super().__init__()
        self.config = config
        self.token_embd = nn.Embedding(config.vocab_size, config.dim, dtype=config.dtype)
        self.blocks = nn.ModuleList([
            Qwen35Block(config, layer_idx=i) for i in range(config.num_layers)
        ])
        self.output_norm = Qwen35RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.mtp_block = Qwen35MTPBlock(config) if config.has_mtp else None

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
        # Tied LM Head: multiply by token_embd.weight
        logits = F.linear(h, self.token_embd.weight)
        return logits, next_states
