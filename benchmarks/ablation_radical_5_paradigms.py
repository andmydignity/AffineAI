#!/usr/bin/env python3
"""
Ablation of 5 Radical Academic Paradigms for Breaking the Sample Efficiency Ceiling
===================================================================================
Evaluates 5 distinct paradigms from academic literature to break the ~5.3 PPL plateau:
  - Arm 0: Baseline Model (Quantum Tensor Network MPS, 250k params)
  - Arm 1: Episodic Associative Memory (Modern Continuous Hopfield / kNN-LM)
  - Arm 2: Test-Time Training (TTT-Linear Fast-Weight Plasticity)
  - Arm 3: Epistemic Active Sampling (AXIOM / Free Energy Information Gain)
  - Arm 4: Dual-Horizon Hierarchical Rollout (Macro-Patch Speculation + Micro Infill)
  - Arm 5: Slot / Prototype Competitive Binding (RIMs / Invariant Structural Slots)

Environment: NVIDIA RTX 3050 GPU, SimpleStories (100MB cap), HybridMuonAdamW, Sampling Probe.
Does NOT modify any production files.
"""

import os
import sys
import math
import time
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.kernels.triton_rms_norm import triton_rms_norm
from affine_ai.core.norm import RMSNorm
from affine_ai.core.associative import FusedMonarchChain, MonarchPermutationChain
from affine_ai.core.bitlinear import BitLinear
from affine_ai.models.blt import ByteLocalEncoder, EntropyPatcher
from affine_ai.core.rls_head import RLSPredictiveHead
from affine_ai.optim.muon import HybridMuonAdamW


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.75,
    top_k: int = 40,
    top_p: float = 0.9,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    scores = logits.float() / max(temperature, 1e-4)
    if top_k is not None and 0 < top_k < scores.size(-1):
        v, _ = torch.topk(scores, min(top_k, scores.size(-1)), dim=-1)
        scores = scores.masked_fill(scores < v[:, -1:], float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
        probs = F.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = (cumulative - probs) > top_p
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        scores = torch.full_like(scores, float("-inf")).scatter(-1, sorted_idx, sorted_scores)
    probs = F.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


# ---------------------------------------------------------------------------
# Core Modules & Paradigms
# ---------------------------------------------------------------------------

class DilatedConvPrefix(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 4, dilation: int = 1):
        super().__init__()
        self.eff_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=0, dilation=dilation, groups=dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        x_pad = F.pad(x, (0, 0, self.eff_pad, 0)).to(self.conv.weight.dtype)
        x_conv = self.conv(x_pad.transpose(1, 2)).transpose(1, 2).to(x.dtype)
        return self.act(x_conv)


class FastGLAMixer(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, num_stages: int = 4, seed_offset: int = 0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkvg_proj = FusedMonarchChain(dim=d_model, num_branches=4, num_stages=num_stages, seed_offset=seed_offset * 10 + 1)
        self.out_proj = MonarchPermutationChain(dim=d_model, num_stages=num_stages, seed_offset=seed_offset * 10 + 5)
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)
        self.gate_decay = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.gate_decay.bias, 3.0)

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.n_heads, self.d_head
        q_raw, k_raw, v_raw, g_raw = self.qkvg_proj(x)
        phi_q = (F.elu(self.q_norm(q_raw.view(B, T, H, D))) + 1.0).transpose(1, 2)
        phi_k = (F.elu(self.k_norm(k_raw.view(B, T, H, D))) + 1.0).transpose(1, 2)
        v = v_raw.view(B, T, H, D).transpose(1, 2)
        g = F.silu(g_raw)

        gamma = torch.sigmoid(self.gate_decay(x.to(self.gate_decay.weight.dtype))).transpose(1, 2)
        log_gam = torch.log(gamma.clamp(min=1e-5))
        cum_log_gam = torch.cumsum(log_gam, dim=-1)
        decay_diff = (cum_log_gam.unsqueeze(-1) - cum_log_gam.unsqueeze(-2)).clamp(max=0.0)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        decay_mat = torch.where(causal_mask, torch.exp(decay_diff), torch.zeros_like(decay_diff))

        scores = (torch.matmul(phi_q, phi_k.transpose(-1, -2)) * decay_mat.to(phi_q.dtype)).to(v.dtype)
        num = torch.matmul(scores, v)
        den = scores.sum(dim=-1, keepdim=True).clamp(min=1e-5)
        y = (num / den).transpose(1, 2).reshape(B, T, C).to(x.dtype)
        return self.out_proj(y * g.to(y.dtype))


class MatrixProductStateMixer(nn.Module):
    """Quantum Tensor Network: Matrix Product Operator (MPO) Contraction."""
    def __init__(self, dim: int = 128, hidden: int = 256, bond_dim: int = 16):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.bond_dim = bond_dim
        self.d1, self.d2 = 16, 8
        self.h1, self.h2 = 16, 16
        self.core1 = nn.Parameter(torch.randn(self.d1, self.h1, bond_dim) * 0.05)
        self.core2 = nn.Parameter(torch.randn(bond_dim, self.d2, self.h2) * 0.05)
        self.norm = RMSNorm(hidden)
        self.w_out = BitLinear(hidden, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        x_tens = x.view(B, T, self.d1, self.d2)
        c1 = self.core1.to(x_tens.dtype)
        c2 = self.core2.to(x_tens.dtype)
        t1 = torch.einsum('btij,ikr->btjkr', x_tens, c1)
        t2 = torch.einsum('btjkr,rjs->btks', t1, c2)
        h = t2.reshape(B, T, self.hidden)
        return self.w_out(self.norm(h))


class TTTLinearMixer(nn.Module):
    """
    Paradigm 2: Test-Time Training (TTT-Linear / Delta-Net Fast Weights).
    Maintains an online adaptive weight matrix S_t updated via causal Delta rule:
    S_t = S_{t-1} * (I - beta * k_t k_t^T) + beta * v_t k_t^T
    """
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.w_q = BitLinear(dim, dim, bias=False)
        self.w_k = BitLinear(dim, dim, bias=False)
        self.w_v = BitLinear(dim, dim, bias=False)
        self.beta_proj = nn.Linear(dim, 1, bias=True)
        nn.init.constant_(self.beta_proj.bias, -1.0)
        self.norm = RMSNorm(dim)
        self.w_out = BitLinear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q = F.normalize(self.w_q(x), dim=-1)
        k = F.normalize(self.w_k(x), dim=-1)
        v = self.w_v(x)
        beta = torch.sigmoid(self.beta_proj(x.to(self.beta_proj.weight.dtype))) # [B, T, 1]

        # Fast-weight sequential unroll
        out = torch.zeros_like(v)
        S = torch.zeros(B, D, D, device=x.device, dtype=v.dtype)
        for t in range(T):
            kt = k[:, t : t + 1, :] # [B, 1, D]
            vt = v[:, t : t + 1, :] # [B, 1, D]
            bt = beta[:, t : t + 1, :].to(v.dtype) # [B, 1, 1]
            qt = q[:, t : t + 1, :] # [B, 1, D]

            # Delta-rule step on fast weights S
            error = vt - torch.bmm(kt, S.transpose(1, 2)) # [B, 1, D]
            S = S + bt * torch.bmm(kt.transpose(1, 2), error)
            out[:, t : t + 1, :] = torch.bmm(qt, S.transpose(1, 2))

        return self.w_out(self.norm(out))


class SlotPrototypeMixer(nn.Module):
    """
    Paradigm 5: Slot / Prototype Competitive Structural Binding (RIMs / Slot Attention).
    Channel features compete to bind to K invariant structural prototype slots.
    """
    def __init__(self, dim: int = 128, num_slots: int = 8):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.w_q = BitLinear(dim, dim, bias=False)
        # Learnable invariant prototype slots
        self.slots = nn.Parameter(torch.randn(num_slots, dim) * 0.1)
        self.slot_transform = BitLinear(dim, dim, bias=False)
        self.norm = RMSNorm(dim)
        self.w_out = BitLinear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q = self.w_q(x) # [B, T, D]
        slots = self.slots.to(x.dtype)
        # Cross-attention / competitive binding to invariant slots
        scores = torch.matmul(q, slots.t()) / math.sqrt(D) # [B, T, K]
        attn = F.softmax(scores, dim=-1) # [B, T, K]
        # Transformed slots
        t_slots = self.slot_transform(slots) # [K, D]
        bound = torch.matmul(attn, t_slots) # [B, T, D]
        return self.w_out(self.norm(bound))


# ---------------------------------------------------------------------------
# Decoders & Episodic Memory
# ---------------------------------------------------------------------------

class EpisodicHopfieldMemory(nn.Module):
    """
    Paradigm 1: Modern Continuous Hopfield Episodic Memory (kNN-LM).
    Retrieves immediate 1-shot past byte associations via continuous Hopfield energy.
    """
    def __init__(self, d_model: int = 64, vocab_size: int = 256):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.w_q = BitLinear(d_model, d_model, bias=False)
        self.w_k = BitLinear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, 1, bias=True)
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def forward(self, h2_norm: torch.Tensor, base_logits: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = h2_norm.shape
        if targets is None or T <= 1:
            return base_logits

        # Compute continuous Hopfield energy over strictly past sequence history (j < t)
        q = F.normalize(self.w_q(h2_norm), dim=-1)
        k = F.normalize(self.w_k(h2_norm), dim=-1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D) # [B, T, T]
        causal_mask = torch.tril(torch.ones(T, T, device=h2_norm.device, dtype=torch.bool), diagonal=-1)
        scores = scores.masked_fill(~causal_mask, -1e9)
        attn_raw = F.softmax(scores, dim=-1)
        pos_mask = (torch.arange(T, device=h2_norm.device) > 0).view(1, T, 1)
        attn = torch.nan_to_num(attn_raw, nan=0.0) * pos_mask.to(attn_raw.dtype)

        # Scatter attention weights into vocabulary distribution
        tgt_exp = targets.unsqueeze(1).expand(-1, T, -1) # [B, T, T]
        p_mem = torch.zeros(B, T, self.vocab_size, device=h2_norm.device, dtype=attn.dtype).scatter_add(-1, tgt_exp, attn)
        mem_logits = torch.log(p_mem.clamp(min=1e-6))

        gate_raw = torch.sigmoid(self.gate_proj(h2_norm.to(self.gate_proj.weight.dtype))).to(base_logits.dtype)
        gate = gate_raw * pos_mask.to(base_logits.dtype)
        return (1.0 - gate) * base_logits + gate * mem_logits.to(base_logits.dtype)


class HierarchicalByteDecoder(nn.Module):
    """
    Paradigm 4: Dual-Horizon Hierarchical Rollout Decoder.
    Byte decoder is conditioned on both current patch latent AND speculative future macro-patch.
    """
    def __init__(self, vocab_size: int = 256, d_byte: int = 64, d_model: int = 128, conv_k: int = 8, use_speculation: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_byte = d_byte
        self.d_model = d_model
        self.conv_k = conv_k
        self.use_speculation = use_speculation

        # If speculative, context is [current_patch, future_patch]
        in_patch_dim = 2 * d_model if use_speculation else d_model
        self.patch_to_byte = BitLinear(in_patch_dim, d_byte, bias=False)
        self.fusion = BitLinear(2 * d_byte, d_byte, bias=False)
        self.norm1_scale = nn.Parameter(torch.ones(d_byte))

        self.temporal_conv = nn.Conv1d(in_channels=d_byte, out_channels=d_byte, kernel_size=conv_k, padding=0, groups=d_byte, bias=False)
        nn.init.normal_(self.temporal_conv.weight, mean=0.0, std=0.02)
        self.conv_act = nn.SiLU()

        self.gate_proj = BitLinear(d_byte, d_byte, bias=False)
        self.val_proj = BitLinear(d_byte, d_byte, bias=False)
        self.down_proj = BitLinear(d_byte, d_byte, bias=False)
        self.norm2_scale = nn.Parameter(torch.ones(d_byte))
        self.lm_head = BitLinear(d_byte, vocab_size, bias=False)

    def forward(self, h_byte, latent_patches, patch_assignments, speculative_patches: Optional[torch.Tensor] = None):
        B, T, _ = h_byte.shape
        M = latent_patches.shape[1]

        idx_expanded = patch_assignments.clamp(0, M - 1).unsqueeze(-1).expand(-1, -1, self.d_model)
        patch_curr = torch.gather(latent_patches, 1, idx_expanded)

        if self.use_speculation and speculative_patches is not None:
            spec_expanded = patch_assignments.clamp(0, M - 1).unsqueeze(-1).expand(-1, -1, self.d_model)
            patch_next = torch.gather(speculative_patches, 1, spec_expanded)
            combined = torch.cat([patch_curr, patch_next], dim=-1)
        else:
            combined = patch_curr

        patch_h = self.patch_to_byte(combined)
        fused_raw = self.fusion(torch.cat([h_byte, patch_h], dim=-1))
        fused = triton_rms_norm(F.silu(fused_raw), self.norm1_scale)
        fused_pad = F.pad(fused, (0, 0, self.conv_k - 1, 0)).to(self.temporal_conv.weight.dtype)
        fused_conv = self.temporal_conv(fused_pad.transpose(1, 2)).transpose(1, 2).to(fused.dtype)
        fused = self.conv_act(fused_conv)

        gate = F.silu(self.gate_proj(fused))
        val = self.val_proj(fused)
        h2 = self.down_proj(gate * val)
        h2_norm = triton_rms_norm(h2, self.norm2_scale)
        return self.lm_head(h2_norm), h2_norm


# ---------------------------------------------------------------------------
# Integrated Architecture Model
# ---------------------------------------------------------------------------

class GenericEncoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, layer_idx: int = 0, channel_mode: str = "mps"):
        super().__init__()
        self.norm1_scale = nn.Parameter(torch.ones(dim))
        self.norm2_scale = nn.Parameter(torch.ones(dim))
        self.conv_prefix = DilatedConvPrefix(dim, kernel_size=4, dilation=1)
        self.time_mixer = FastGLAMixer(d_model=dim, n_heads=n_heads, seed_offset=layer_idx * 10)

        if channel_mode == "mps":
            self.channel_mixer = MatrixProductStateMixer(dim, hidden=256, bond_dim=16)
        elif channel_mode == "ttt_linear":
            self.channel_mixer = TTTLinearMixer(dim)
        elif channel_mode == "slot_prototype":
            self.channel_mixer = SlotPrototypeMixer(dim, num_slots=8)
        else:
            raise ValueError(f"Unknown channel_mode: {channel_mode}")

    def forward(self, x):
        h1 = triton_rms_norm(x, self.norm1_scale)
        h1 = self.conv_prefix(h1)
        x = x + self.time_mixer(h1)
        h2 = triton_rms_norm(x, self.norm2_scale)
        x = x + self.channel_mixer(h2)
        return x


class AcademicParadigmsEvaluationModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 128,
        d_byte: int = 64,
        n_encoder_layers: int = 4,
        n_heads: int = 4,
        target_patch_size: int = 16,
        channel_mode: str = "mps",
        use_hopfield_memory: bool = False,
        use_macro_speculation: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.d_byte = d_byte
        self.n_encoder_layers = n_encoder_layers
        self.target_patch_size = target_patch_size
        self.channel_mode = channel_mode
        self.use_hopfield_memory = use_hopfield_memory
        self.use_macro_speculation = use_macro_speculation

        self.byte_encoder = ByteLocalEncoder(vocab_size=vocab_size, d_byte=d_byte, kernel_size=4)
        self.patcher = EntropyPatcher(d_byte=d_byte, d_model=dim, target_patch_size=target_patch_size)

        self.blocks = nn.ModuleList([
            GenericEncoderBlock(dim=dim, n_heads=n_heads, layer_idx=i, channel_mode=channel_mode)
            for i in range(n_encoder_layers)
        ])

        self.layer_readout_weights = nn.Parameter(torch.zeros(n_encoder_layers + 1))
        with torch.no_grad():
            self.layer_readout_weights.fill_(0.0)
            self.layer_readout_weights[-1] = 1.5

        self.norm_out_scale = nn.Parameter(torch.ones(dim))
        self.sos_patch = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Macro-Speculator (Paradigm 4)
        if use_macro_speculation:
            self.macro_speculator = nn.Sequential(
                BitLinear(dim, dim, bias=False),
                RMSNorm(dim),
                nn.SiLU(),
                BitLinear(dim, dim, bias=False)
            )
        else:
            self.macro_speculator = None

        self.byte_decoder = HierarchicalByteDecoder(
            vocab_size=vocab_size,
            d_byte=d_byte,
            d_model=dim,
            conv_k=8,
            use_speculation=use_macro_speculation
        )

        # Episodic Hopfield Memory (Paradigm 1)
        if use_hopfield_memory:
            self.hopfield_memory = EpisodicHopfieldMemory(d_model=d_byte, vocab_size=vocab_size)
        else:
            self.hopfield_memory = None

    def forward(self, byte_ids, targets=None):
        B, T = byte_ids.shape
        P = self.target_patch_size
        h_byte, boundary_logits = self.byte_encoder(byte_ids)

        is_delimiter = (byte_ids == 32) | (byte_ids == 44) | (byte_ids == 46) | (byte_ids == 10)
        boundary_boost = torch.where(is_delimiter, torch.tensor(3.0, device=byte_ids.device), torch.tensor(0.0, device=byte_ids.device))
        boundary_in = boundary_logits + boundary_boost
        latent_patches, patch_assignments = self.patcher(h_byte, boundary_in, fixed_patch_size=P)

        layer_outputs = [latent_patches]
        h = latent_patches
        for block in self.blocks:
            h = block(h)
            layer_outputs.append(h)

        normed_weights = F.softmax(self.layer_readout_weights, dim=0)
        h_dense = sum(w * hl for w, hl in zip(normed_weights, layer_outputs))
        h_latent = triton_rms_norm(h_dense, self.norm_out_scale)

        causal_patches = torch.cat([self.sos_patch.expand(B, 1, -1), h_latent[:, :-1]], dim=1)

        spec_patches = None
        if self.use_macro_speculation and self.macro_speculator is not None:
            # Speculate next macro-state from current causal patch
            spec_patches = self.macro_speculator(causal_patches)

        base_logits, h2_norm = self.byte_decoder(h_byte, causal_patches, patch_assignments, speculative_patches=spec_patches)

        if self.use_hopfield_memory and self.hopfield_memory is not None:
            logits = self.hopfield_memory(h2_norm, base_logits, targets=targets)
        else:
            logits = base_logits

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return loss
        return logits


# ---------------------------------------------------------------------------
# Training Harness
# ---------------------------------------------------------------------------

def run_experiment_arm(
    arm_name: str,
    channel_mode: str,
    use_hopfield: bool,
    use_speculation: bool,
    use_epistemic_sampling: bool,
    train_data: np.memmap,
    val_data: np.memmap,
    steps: int = 400,
    batch_size: int = 16,
    seq_len: int = 256,
    muon_lr: float = 0.02,
    adamw_lr: float = 3e-3,
    device: str = "cuda"
):
    print(f"\n>>> Running Arm: {arm_name}")
    set_seed(42)

    model = AcademicParadigmsEvaluationModel(
        dim=128,
        d_byte=64,
        n_encoder_layers=4,
        n_heads=4,
        channel_mode=channel_mode,
        use_hopfield_memory=use_hopfield,
        use_macro_speculation=use_speculation
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"    Params: {param_count:,d} | Channel: {channel_mode} | Hopfield: {use_hopfield} | Speculation: {use_speculation} | Epistemic Sampling: {use_epistemic_sampling}")

    optimizer = HybridMuonAdamW(
        model=model,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        muon_momentum=0.95,
        adamw_weight_decay=0.01
    )
    muon_p = sum(p.numel() for p in optimizer.muon_opt.param_groups[0]['params']) if optimizer.muon_opt else 0
    adamw_p = sum(p.numel() for g in optimizer.adamw_opt.param_groups for p in g['params']) if optimizer.adamw_opt else 0
    print(f"    Optimizer: HybridMuonAdamW (Muon: {muon_p:,d} params, AdamW: {adamw_p:,d} params)")

    warmup_steps = 35

    def update_lr(opt, s):
        if s < warmup_steps:
            ratio = float(s) / float(max(1, warmup_steps))
        else:
            progress = float(s - warmup_steps) / float(max(1, steps - warmup_steps))
            ratio = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        if opt.muon_opt is not None:
            for g in opt.muon_opt.param_groups:
                g['lr'] = muon_lr * ratio
        if opt.adamw_opt is not None:
            for g in opt.adamw_opt.param_groups:
                g['lr'] = adamw_lr * ratio

    val_batches = []
    np.random.seed(1337)
    high_val = len(val_data) - seq_len - 1
    for _ in range(12):
        ix = np.random.randint(0, high_val, size=batch_size)
        x_val = torch.from_numpy(np.stack([val_data[i : i + seq_len] for i in ix])).long().to(device)
        y_val = torch.from_numpy(np.stack([val_data[i + 1 : i + seq_len + 1] for i in ix])).long().to(device)
        val_batches.append((x_val, y_val))

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for x_v, y_v in val_batches:
            loss_v = model(x_v, targets=y_v)
            losses.append(loss_v.item())
        model.train()
        mean_loss = float(np.mean(losses))
        ppl = math.exp(min(mean_loss, 20.0))
        bpc = mean_loss / math.log(2)
        return mean_loss, ppl, bpc

    init_loss, init_ppl, init_bpc = evaluate()
    print(f"    Step   0 | Val Loss: {init_loss:.4f} | Val PPL: {init_ppl:.2f}")

    high_train = len(train_data) - seq_len - 1
    step_times = []

    # Online RLS evaluator for Epistemic Sampling (Paradigm 3)
    rls_sampler = RLSPredictiveHead(d_model=64, vocab_size=256) if use_epistemic_sampling else None

    for step in range(1, steps + 1):
        update_lr(optimizer, step)

        t0 = time.time()
        optimizer.zero_grad()

        if use_epistemic_sampling:
            # Sample 3 candidate chunks and select the one with highest epistemic surprise / loss
            cand_ixs = [np.random.randint(0, high_train, size=batch_size) for _ in range(3)]
            best_ix = cand_ixs[0]
            max_var = -1.0
            with torch.no_grad():
                for c_ix in cand_ixs:
                    x_cand = torch.from_numpy(np.stack([train_data[i : i + seq_len] for i in c_ix])).long().to(device)
                    y_cand = torch.from_numpy(np.stack([train_data[i + 1 : i + seq_len + 1] for i in c_ix])).long().to(device)
                    # Quick forward loss as proxy for information gain
                    loss_cand = model(x_cand, targets=y_cand)
                    if loss_cand.item() > max_var:
                        max_var = loss_cand.item()
                        best_ix = c_ix
            ix = best_ix
        else:
            ix = np.random.randint(0, high_train, size=batch_size)

        x = torch.from_numpy(np.stack([train_data[i : i + seq_len] for i in ix])).long().to(device)
        y = torch.from_numpy(np.stack([train_data[i + 1 : i + seq_len + 1] for i in ix])).long().to(device)

        loss = model(x, targets=y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t1 = time.time()

        step_times.append((t1 - t0) * 1000)
        loss_val = loss.item()

        if step % 80 == 0 or step == steps:
            val_loss, val_ppl, val_bpc = evaluate()
            avg_ms = np.mean(step_times[-50:]) if len(step_times) >= 50 else np.mean(step_times)
            print(f"    Step {step:3d}/{steps} | Train Loss: {loss_val:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:6.2f} | {avg_ms:5.1f} ms/step")

    # In-distribution sampling probe from SimpleStories validation slice
    prompt_bytes = bytes(val_data[:48].tolist())
    prompt_ids = torch.from_numpy(val_data[:48].astype(np.int64)).unsqueeze(0).to(device)

    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(42)

    with torch.no_grad():
        curr_ids = prompt_ids
        for _ in range(96):
            logits = model(curr_ids)
            next_token = sample_next_token(logits[:, -1, :], temperature=0.75, top_k=40, top_p=0.9, generator=rng)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

    generated_bytes = bytes(curr_ids[0].tolist())
    gen_slice = generated_bytes[len(prompt_bytes):]
    four_grams = [gen_slice[i:i+4] for i in range(len(gen_slice) - 3)]
    distinct_4gram = len(set(four_grams)) / max(1, len(four_grams)) if four_grams else 0.0
    sample_text = gen_slice[:80].decode("utf-8", errors="replace").replace("\n", "\\n")

    print(f"    Sampling Probe (T=0.75, top-p=0.9): Distinct-4gram = {distinct_4gram:.3f} | Sample: \"{sample_text}\"")

    warm_times = step_times[5:] if len(step_times) > 5 else step_times
    return {
        "arm": arm_name,
        "params": param_count,
        "final_val_loss": val_loss,
        "final_val_ppl": val_ppl,
        "val_bpc": val_bpc,
        "avg_ms_step": float(np.mean(warm_times)),
        "distinct_4gram": distinct_4gram,
        "sample_text": sample_text
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 88)
    print(f"   EVALUATING 5 RADICAL ACADEMIC PARADIGMS FOR RAPID LEARNING (Device: {device})")
    print("   Dataset: SimpleStories (Capped to 100.0 MB)")
    print("=" * 88)

    data_path = "data/simplestories_eos.bin"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        return

    MAX_BYTES = 100 * 1024 * 1024
    raw_data = np.memmap(data_path, dtype=np.uint8, mode="r")
    total_len = min(len(raw_data), MAX_BYTES)
    data = raw_data[:total_len]
    split = int(0.95 * total_len)
    train_data = data[:split]
    val_data = data[split:]

    arms = [
        # Arm 0: Baseline (Quantum Tensor Network MPS, 250k)
        ("Arm 0: Baseline (Quantum Tensor MPS, 250k)", "mps", False, False, False),

        # Arm 1: Episodic Associative Memory (Modern Hopfield / kNN-LM)
        ("Arm 1: Episodic Associative Memory (Modern Hopfield)", "mps", True, False, False),

        # Arm 2: Test-Time Training (TTT-Linear Fast Weights)
        ("Arm 2: Test-Time Training (TTT Fast Weights)", "ttt_linear", False, False, False),

        # Arm 3: Epistemic Active Sampling (AXIOM Info-Gain)
        ("Arm 3: Epistemic Active Sampling (AXIOM Surprisal)", "mps", False, False, True),

        # Arm 4: Dual-Horizon Hierarchical Rollout (Macro Speculation)
        ("Arm 4: Dual-Horizon Rollout (Macro Speculation)", "mps", False, True, False),

        # Arm 5: Slot / Prototype Competitive Binding (RIMs / Slots)
        ("Arm 5: Slot / Prototype Structural Binding", "slot_prototype", False, False, False),
    ]

    results = []
    for name, c_mode, hopfield, spec, epistemic in arms:
        res = run_experiment_arm(
            arm_name=name,
            channel_mode=c_mode,
            use_hopfield=hopfield,
            use_speculation=spec,
            use_epistemic_sampling=epistemic,
            train_data=train_data,
            val_data=val_data,
            steps=400,
            batch_size=16,
            seq_len=256,
            muon_lr=0.02,
            adamw_lr=3e-3,
            device=device
        )
        results.append(res)

    print("\n" + "=" * 102)
    print("               FINAL RESULTS: 5 RADICAL ACADEMIC PARADIGMS ON SIMPLESTORIES")
    print("=" * 102)
    print(f"{'Arm':<56} | {'Params':<8} | {'Val PPL':<8} | {'Val BPC':<8} | {'ms/step':<8} | {'Dist-4g':<8}")
    print("-" * 102)
    for r in results:
        print(f"{r['arm']:<56} | {r['params']:<8,d} | {r['final_val_ppl']:<8.2f} | {r['val_bpc']:<8.3f} | {r['avg_ms_step']:<8.1f} | {r['distinct_4gram']:<8.3f}")
    print("=" * 102)


if __name__ == "__main__":
    main()
