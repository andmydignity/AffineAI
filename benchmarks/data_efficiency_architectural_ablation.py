#!/usr/bin/env python3
"""
Architectural Data-Efficiency Ablation
======================================
Tests architectural inductive biases for sample efficiency on byte-level text
WITHOUT modifying core library files, utilizing Triton GPU kernels.

Arms:
  0. Baseline: Standard 4-layer ASDAG (Monarch GLA + Ternary SwiGLU, uniform decay)
  1. Multi-scale & Persistent Memory: Head timescale hierarchy (fast, medium, long, persistent g=1.0)
  2. Causal Depthwise Conv Prefix: k=4 depthwise 1D conv per block (local n-gram offloading)
  3. Dense Multi-Layer Readout: Highway skip from embedding + all blocks to LM head
  4. Synergistic Combined: Multi-scale memory + Conv prefix + Dense readout
"""

import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Triton Hardware Acceleration Kernels
from affine_ai.kernels.triton_rms_norm import triton_rms_norm
from affine_ai.kernels.triton_cross_entropy import triton_fused_linear_cross_entropy
from affine_ai.core.norm import RMSNorm
from affine_ai.core.associative import FusedMonarchChain, MonarchPermutationChain
from affine_ai.core.bitlinear import TernaryBitLinearSwiGLU

from typing import Optional

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
    """Sample next token using temperature, top-k, and nucleus (top-p) filtering."""
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


class FastGLAMixer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        num_stages: int = 4,
        seed_offset: int = 0,
        use_multiscale: bool = False,
        dtype: any = torch.float32
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_multiscale = use_multiscale

        self.qkvg_proj = FusedMonarchChain(
            dim=d_model,
            num_branches=4,
            num_stages=num_stages,
            seed_offset=seed_offset * 10 + 1,
            dtype=dtype
        )
        self.out_proj = MonarchPermutationChain(
            dim=d_model,
            num_stages=num_stages,
            seed_offset=seed_offset * 10 + 5,
            dtype=dtype
        )

        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)

        self.gate_decay = nn.Linear(d_model, n_heads, bias=True, dtype=dtype)
        if use_multiscale:
            with torch.no_grad():
                self.gate_decay.bias.copy_(torch.tensor([0.0, 2.0, 4.0, 14.0][:n_heads]))
                self.persistent_head_idx = n_heads - 1
        else:
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
        if self.use_multiscale:
            gamma = gamma.clone()
            gamma[:, self.persistent_head_idx, :] = 1.0

        log_gam = torch.log(gamma.clamp(min=1e-5))
        cum_log_gam = torch.cumsum(log_gam, dim=-1)
        decay_diff = (cum_log_gam.unsqueeze(-1) - cum_log_gam.unsqueeze(-2)).clamp(max=0.0)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        decay_mat = torch.where(causal_mask, torch.exp(decay_diff), torch.zeros_like(decay_diff))

        scores = torch.matmul(phi_q, phi_k.transpose(-1, -2)) * decay_mat
        num = torch.matmul(scores, v)
        den = scores.sum(dim=-1, keepdim=True).clamp(min=1e-5)
        y = (num / den).transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(y * g)
        return out


class CausalConvPrefix(nn.Module):
    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=dim,
            bias=False
        )
        self.act = nn.SiLU()

    def forward(self, x):
        T = x.shape[1]
        x_conv = self.conv(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        return self.act(x_conv)


class AblationBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        use_multiscale: bool = False,
        use_conv_prefix: bool = False,
        layer_idx: int = 0
    ):
        super().__init__()
        self.dim = dim
        self.use_conv_prefix = use_conv_prefix
        self.norm1_scale = nn.Parameter(torch.ones(dim))
        self.norm2_scale = nn.Parameter(torch.ones(dim))

        if use_conv_prefix:
            self.conv_prefix = CausalConvPrefix(dim, kernel_size=4)

        self.time_mixer = FastGLAMixer(
            d_model=dim,
            n_heads=n_heads,
            num_stages=4,
            seed_offset=layer_idx * 10,
            use_multiscale=use_multiscale
        )
        self.channel_mixer = TernaryBitLinearSwiGLU(dim, expand=2)

    def forward(self, x):
        h1 = triton_rms_norm(x, self.norm1_scale)
        if self.use_conv_prefix:
            h1 = self.conv_prefix(h1)
        x = x + self.time_mixer(h1)

        h2 = triton_rms_norm(x, self.norm2_scale)
        x = x + self.channel_mixer(h2)
        return x


class AblationModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        use_multiscale: bool = False,
        use_conv_prefix: bool = False,
        use_dense_readout: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.use_dense_readout = use_dense_readout

        self.embed = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList([
            AblationBlock(
                dim=dim,
                n_heads=n_heads,
                use_multiscale=use_multiscale,
                use_conv_prefix=use_conv_prefix,
                layer_idx=i
            ) for i in range(n_layers)
        ])

        if use_dense_readout:
            self.layer_weights = nn.Parameter(torch.zeros(n_layers + 1))
            with torch.no_grad():
                self.layer_weights.fill_(0.0)
                self.layer_weights[-1] = 1.5

        self.norm_f_scale = nn.Parameter(torch.ones(dim))
        self.lm_head_weight = nn.Parameter(torch.randn(vocab_size, dim) * 0.02)

    def forward(self, input_ids, targets=None):
        x = self.embed(input_ids)
        layer_outputs = [x]

        for block in self.blocks:
            x = block(x)
            if self.use_dense_readout:
                layer_outputs.append(x)

        if self.use_dense_readout:
            normed_weights = F.softmax(self.layer_weights, dim=0)
            h_final = sum(w * h for w, h in zip(normed_weights, layer_outputs))
        else:
            h_final = x

        h_norm = triton_rms_norm(h_final, self.norm_f_scale)

        if targets is not None:
            loss = triton_fused_linear_cross_entropy(
                h_norm.view(-1, self.dim),
                self.lm_head_weight,
                targets.view(-1)
            )
            return loss
        else:
            logits = F.linear(h_norm, self.lm_head_weight)
            return logits


def run_experiment_arm(
    arm_name: str,
    model_kwargs: dict,
    train_data: np.memmap,
    val_data: np.memmap,
    steps: int = 350,
    batch_size: int = 16,
    seq_len: int = 256,
    lr: float = 3e-3,
    device: str = "cuda"
):
    print(f"\n>>> Running Arm: {arm_name}")
    set_seed(42)

    model = AblationModel(**model_kwargs).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"    Params: {param_count:,d}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup_steps = 30

    def get_lr(s):
        if s < warmup_steps:
            return lr * (s + 1) / warmup_steps
        progress = (s - warmup_steps) / max(1, steps - warmup_steps)
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # 10 fixed validation batches
    val_batches = []
    np.random.seed(1337)
    high_val = len(val_data) - seq_len - 1
    for _ in range(10):
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
        step_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = step_lr

        ix = np.random.randint(0, high_train, size=batch_size)
        x = torch.from_numpy(np.stack([train_data[i : i + seq_len] for i in ix])).long().to(device)
        y = torch.from_numpy(np.stack([train_data[i + 1 : i + seq_len + 1] for i in ix])).long().to(device)

        t0 = time.time()
        optimizer.zero_grad()
        loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t1 = time.time()

        step_times.append((t1 - t0) * 1000)
        loss_val = loss.item()

        if step % 70 == 0 or step == steps:
            val_loss, val_ppl, val_bpc = evaluate()
            avg_ms = np.mean(step_times[-50:]) if len(step_times) >= 50 else np.mean(step_times)
            print(f"    Step {step:3d}/{steps} | Train Loss: {loss_val:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:6.2f} | {avg_ms:5.1f} ms/step")

    # Sampling decode probe
    prompt = b"Once upon a time, there was a little "
    prompt_ids = torch.tensor(list(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
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
    gen_slice = generated_bytes[len(prompt):]
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
    print(f"   ARCHITECTURAL DATA-EFFICIENCY ABLATION BENCHMARK (Device: {device})")
    print("   Testing Inductive Biases on Byte-Level Language Modeling with Triton Kernels")
    print("=" * 85)

    data_path = "data/tinystories_extracted.bin"
    if not os.path.exists(data_path):
        data_path = "data/tinystories_eos.bin"

    print(f"Dataset: {data_path} ({os.path.getsize(data_path) / (1024*1024):.1f} MB)")
    data = np.memmap(data_path, dtype=np.uint8, mode="r")
    split = int(0.95 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    print(f"Train Bytes: {len(train_data):,d} | Val Bytes: {len(val_data):,d}")

    STEPS = 350
    BATCH_SIZE = 16
    SEQ_LEN = 256
    DIM = 128
    N_LAYERS = 4
    N_HEADS = 4

    arms = [
        (
            "Arm 0: Baseline ASDAG (Flat Decay)",
            {"dim": DIM, "n_layers": N_LAYERS, "n_heads": N_HEADS, "use_multiscale": False, "use_conv_prefix": False, "use_dense_readout": False}
        ),
        (
            "Arm 1: Multi-Scale & Persistent State (g=1.0)",
            {"dim": DIM, "n_layers": N_LAYERS, "n_heads": N_HEADS, "use_multiscale": True, "use_conv_prefix": False, "use_dense_readout": False}
        ),
        (
            "Arm 2: Causal Depthwise Conv Prefix (k=4)",
            {"dim": DIM, "n_layers": N_LAYERS, "n_heads": N_HEADS, "use_multiscale": False, "use_conv_prefix": True, "use_dense_readout": False}
        ),
        (
            "Arm 3: Dense Multi-Layer Readout Highway",
            {"dim": DIM, "n_layers": N_LAYERS, "n_heads": N_HEADS, "use_multiscale": False, "use_conv_prefix": False, "use_dense_readout": True}
        ),
        (
            "Arm 4: Combined Best (MultiScale + Conv + Dense)",
            {"dim": DIM, "n_layers": N_LAYERS, "n_heads": N_HEADS, "use_multiscale": True, "use_conv_prefix": True, "use_dense_readout": True}
        ),
    ]

    results = []
    for name, kwargs in arms:
        res = run_experiment_arm(
            arm_name=name,
            model_kwargs=kwargs,
            train_data=train_data,
            val_data=val_data,
            steps=STEPS,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            lr=3e-3,
            device=device
        )
        results.append(res)

    print("\n" + "=" * 95)
    print("                    FINAL ABLATION RESULTS SUMMARY")
    print("=" * 95)
    print(f"{'Arm':<48} | {'Params':<8} | {'Val PPL':<8} | {'Val BPC':<8} | {'ms/step':<8} | {'Dist-4g':<8}")
    print("-" * 95)
    for r in results:
        print(f"{r['arm']:<48} | {r['params']:<8,d} | {r['final_val_ppl']:<8.2f} | {r['val_bpc']:<8.3f} | {r['avg_ms_step']:<8.1f} | {r['distinct_4gram']:<8.3f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
