#!/usr/bin/env python3
"""
TorosHybrid vs Arm 2 (Quantum Tensor Network MPS) at Iso-Parameter Size
======================================================================
Direct head-to-head empirical comparison of TorosHybrid vs Quantum Tensor Network (MPS):
  - Iso-Budget 250k: TorosHybrid SwiGLU (250,070) vs Arm 2 MPS (250,070) [EXACT MATCH]
  - Iso-Budget 486k: TorosHybrid SwiGLU (486,614) vs Arm 2 Wide MPS (485,590) vs Arm 2 Deep MPS (468,655)
  - Native TorosHybrid: Adaptive Sparse Tree DAG (179,234)

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
from affine_ai.models.language_model import AdaptiveSparseTreeDAGLayer
from affine_ai.core.ast_dag import ASDAGConfig
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
# Channel Mixers
# ---------------------------------------------------------------------------

class SwiGLUMixer(nn.Module):
    """Configurable SwiGLU MLP mixer."""
    def __init__(self, dim: int = 128, hidden: int = 256):
        super().__init__()
        self.w_gate = BitLinear(dim, hidden, bias=False)
        self.w_val = BitLinear(dim, hidden, bias=False)
        self.w_out = BitLinear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w_out(F.silu(self.w_gate(x)) * self.w_val(x))


class ConfigurableMPSMixer(nn.Module):
    """
    Quantum Tensor Network: Matrix Product Operator (MPO) Contraction.
    Decomposes channel mixing into a chain of 3-way entangled core tensors.
    """
    def __init__(self, dim: int = 128, hidden: int = 256, bond_dim: int = 16):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.bond_dim = bond_dim
        self.d1 = 16
        self.d2 = 8
        if hidden == 256:
            self.h1, self.h2 = 16, 16
        elif hidden == 512:
            self.h1, self.h2 = 32, 16
        else:
            self.h1, self.h2 = 16, hidden // 16

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


class ASDAGTreeMixer(nn.Module):
    """Adaptive Sparse Tree DAG Channel Mixer (Production TorosHybrid default)."""
    def __init__(self, dim: int = 128):
        super().__init__()
        cfg = ASDAGConfig(dim=dim)
        self.asdag = AdaptiveSparseTreeDAGLayer(cfg)

    def forward(self, x):
        return self.asdag(x)


# ---------------------------------------------------------------------------
# Scaffold Blocks
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


class HybridEncoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, layer_idx: int = 0, mixer_spec: Dict[str, Any] = None):
        super().__init__()
        self.norm1_scale = nn.Parameter(torch.ones(dim))
        self.norm2_scale = nn.Parameter(torch.ones(dim))
        self.conv_prefix = DilatedConvPrefix(dim, kernel_size=4, dilation=1)
        self.time_mixer = FastGLAMixer(d_model=dim, n_heads=n_heads, seed_offset=layer_idx * 10)

        m_type = mixer_spec.get("type", "swiglu")
        if m_type == "swiglu":
            hidden = mixer_spec.get("hidden", 256)
            self.channel_mixer = SwiGLUMixer(dim, hidden=hidden)
        elif m_type == "mps":
            hidden = mixer_spec.get("hidden", 256)
            bond_dim = mixer_spec.get("bond_dim", 16)
            self.channel_mixer = ConfigurableMPSMixer(dim, hidden=hidden, bond_dim=bond_dim)
        elif m_type == "asdag":
            self.channel_mixer = ASDAGTreeMixer(dim)
        else:
            raise ValueError(f"Unknown mixer: {m_type}")

    def forward(self, x):
        h1 = triton_rms_norm(x, self.norm1_scale)
        h1 = self.conv_prefix(h1)
        x = x + self.time_mixer(h1)
        h2 = triton_rms_norm(x, self.norm2_scale)
        x = x + self.channel_mixer(h2)
        return x


class TemporalByteDecoder(nn.Module):
    def __init__(self, vocab_size: int = 256, d_byte: int = 64, d_model: int = 128, conv_k: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_byte = d_byte
        self.d_model = d_model
        self.conv_k = conv_k

        self.patch_to_byte = BitLinear(d_model, d_byte, bias=False)
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

    def forward(self, h_byte, latent_patches, patch_assignments):
        B, T, _ = h_byte.shape
        M = latent_patches.shape[1]
        idx_expanded = patch_assignments.clamp(0, M - 1).unsqueeze(-1).expand(-1, -1, self.d_model)
        patch_context = torch.gather(latent_patches, 1, idx_expanded)
        patch_h = self.patch_to_byte(patch_context)

        fused_raw = self.fusion(torch.cat([h_byte, patch_h], dim=-1))
        fused = triton_rms_norm(F.silu(fused_raw), self.norm1_scale)
        fused_pad = F.pad(fused, (0, 0, self.conv_k - 1, 0)).to(self.temporal_conv.weight.dtype)
        fused_conv = self.temporal_conv(fused_pad.transpose(1, 2)).transpose(1, 2).to(fused.dtype)
        fused = self.conv_act(fused_conv)

        gate = F.silu(self.gate_proj(fused))
        val = self.val_proj(fused)
        h2 = self.down_proj(gate * val)
        h2_norm = triton_rms_norm(h2, self.norm2_scale)
        return self.lm_head(h2_norm)


class IsoSizeEvaluationModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 128,
        d_byte: int = 64,
        n_encoder_layers: int = 4,
        n_heads: int = 4,
        target_patch_size: int = 16,
        mixer_spec: Dict[str, Any] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.d_byte = d_byte
        self.n_encoder_layers = n_encoder_layers
        self.target_patch_size = target_patch_size

        self.byte_encoder = ByteLocalEncoder(vocab_size=vocab_size, d_byte=d_byte, kernel_size=4)
        self.patcher = EntropyPatcher(d_byte=d_byte, d_model=dim, target_patch_size=target_patch_size)

        self.blocks = nn.ModuleList([
            HybridEncoderBlock(dim=dim, n_heads=n_heads, layer_idx=i, mixer_spec=mixer_spec)
            for i in range(n_encoder_layers)
        ])

        self.layer_readout_weights = nn.Parameter(torch.zeros(n_encoder_layers + 1))
        with torch.no_grad():
            self.layer_readout_weights.fill_(0.0)
            self.layer_readout_weights[-1] = 1.5

        self.norm_out_scale = nn.Parameter(torch.ones(dim))
        self.sos_patch = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.byte_decoder = TemporalByteDecoder(vocab_size=vocab_size, d_byte=d_byte, d_model=dim, conv_k=8)

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
        logits = self.byte_decoder(h_byte, causal_patches, patch_assignments)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return loss
        return logits


def run_experiment_arm(
    arm_name: str,
    n_layers: int,
    mixer_spec: Dict[str, Any],
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

    model = IsoSizeEvaluationModel(
        dim=128,
        d_byte=64,
        n_encoder_layers=n_layers,
        n_heads=4,
        mixer_spec=mixer_spec
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"    Params: {param_count:,d} | Layers: {n_layers} | Spec: {mixer_spec}")

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

    for step in range(1, steps + 1):
        update_lr(optimizer, step)

        t0 = time.time()
        optimizer.zero_grad()

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
    print("=" * 85)
    print(f"   TOROSHYBRID VS ARM 2 (QUANTUM TENSOR NETWORK MPS) AT SAME SIZE (Device: {device})")
    print("   Dataset: SimpleStories (Capped to 100.0 MB)")
    print("=" * 85)

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
        # --- Iso-Budget 250k: Exact match at 250,070 parameters ---
        ("Arm 0: TorosHybrid SwiGLU (Iso-250k, H=102)", 4, {"type": "swiglu", "hidden": 102}),
        ("Arm 1: Arm 2 MPS Tensor Network (Iso-250k, r=16)", 4, {"type": "mps", "hidden": 256, "bond_dim": 16}),

        # --- Iso-Budget 486k: Full standard capacity match ---
        ("Arm 2: TorosHybrid SwiGLU (Full 486k, H=256)", 4, {"type": "swiglu", "hidden": 256}),
        ("Arm 3: Arm 2 Wide MPS (Iso-486k, r=50, H=512)", 4, {"type": "mps", "hidden": 512, "bond_dim": 50}),
        ("Arm 4: Arm 2 Deep MPS (Iso-486k, 9-Layers, r=16)", 9, {"type": "mps", "hidden": 256, "bond_dim": 16}),

        # --- Native TorosHybrid ASDAG Tree ---
        ("Arm 5: TorosHybrid ASDAG Tree (Native Default)", 4, {"type": "asdag"}),
    ]

    results = []
    for name, n_l, spec in arms:
        res = run_experiment_arm(
            arm_name=name,
            n_layers=n_l,
            mixer_spec=spec,
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

    print("\n" + "=" * 98)
    print("            FINAL RESULTS: TOROSHYBRID VS ARM 2 (MPS) AT ISO-PARAMETER SIZE")
    print("=" * 98)
    print(f"{'Arm':<52} | {'Params':<8} | {'Val PPL':<8} | {'Val BPC':<8} | {'ms/step':<8} | {'Dist-4g':<8}")
    print("-" * 98)
    for r in results:
        print(f"{r['arm']:<52} | {r['params']:<8,d} | {r['final_val_ppl']:<8.2f} | {r['val_bpc']:<8.3f} | {r['avg_ms_step']:<8.1f} | {r['distinct_4gram']:<8.3f}")
    print("=" * 98)


if __name__ == "__main__":
    main()
