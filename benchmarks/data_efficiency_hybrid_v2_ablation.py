#!/usr/bin/env python3
"""
TorosHybrid Data-Efficiency Frontier Ablation (Round 2)
======================================================
Tests 4 advanced architectural and curriculum data-efficiency mechanisms
against the newly updated TorosHybrid default baseline.

Baseline:
  - Arm 0: Current TorosHybrid defaults (Trunk Conv Prefix k=4 + Dense Readout Highway)

New Exploration Arms:
  - Arm 1: Decoder Temporal Continuity (k=8 Causal Conv in ByteLocalDecoder)
  - Arm 2: Pyramidal Dilated Convolutions (Dilations 1, 2, 4, 8 across blocks)
  - Arm 3: Word-Boundary Aligned Patching (Entropy/delimiter-weighted patch pooling)
  - Arm 4: Active Information-Gain Filtering (rho-Loss: screen 2xB candidates, train top B)
"""

import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Hardware acceleration kernels
from affine_ai.kernels.triton_rms_norm import triton_rms_norm
from affine_ai.kernels.triton_cross_entropy import triton_fused_linear_cross_entropy
from affine_ai.core.norm import RMSNorm
from affine_ai.core.associative import FusedMonarchChain, MonarchPermutationChain
from affine_ai.core.bitlinear import TernaryBitLinearSwiGLU, BitLinear
from affine_ai.models.blt import ByteLocalEncoder, EntropyPatcher, ByteLocalDecoder

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Modular Components for the 4 Arms
# ---------------------------------------------------------------------------

class DilatedConvPrefix(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 4, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.eff_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
            groups=dim,
            bias=False
        )
        self.act = nn.SiLU()

    def forward(self, x):
        T = x.shape[1]
        x_pad = F.pad(x, (0, 0, self.eff_pad, 0)).to(self.conv.weight.dtype)
        x_conv = self.conv(x_pad.transpose(1, 2)).transpose(1, 2).to(x.dtype)
        return self.act(x_conv)


class FastGLAMixer(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, num_stages: int = 4, seed_offset: int = 0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkvg_proj = FusedMonarchChain(
            dim=d_model,
            num_branches=4,
            num_stages=num_stages,
            seed_offset=seed_offset * 10 + 1
        )
        self.out_proj = MonarchPermutationChain(
            dim=d_model,
            num_stages=num_stages,
            seed_offset=seed_offset * 10 + 5
        )
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
    def __init__(self, dim: int, n_heads: int = 4, dilation: int = 1, layer_idx: int = 0):
        super().__init__()
        self.norm1_scale = nn.Parameter(torch.ones(dim))
        self.norm2_scale = nn.Parameter(torch.ones(dim))
        self.conv_prefix = DilatedConvPrefix(dim, kernel_size=4, dilation=dilation)
        self.time_mixer = FastGLAMixer(d_model=dim, n_heads=n_heads, seed_offset=layer_idx * 10)
        self.channel_mixer = TernaryBitLinearSwiGLU(dim, expand=2)

    def forward(self, x):
        h1 = triton_rms_norm(x, self.norm1_scale)
        h1_conv = self.conv_prefix(h1)
        x = x + self.time_mixer(h1_conv)
        h2 = triton_rms_norm(x, self.norm2_scale)
        x = x + self.channel_mixer(h2)
        return x


class TemporalByteDecoder(nn.Module):
    """
    ByteLocalDecoder with optional Arm 1: Temporal Continuity Causal Depthwise Conv.
    Expands the decoder's temporal context from 4 bytes to 11+ bytes.
    """
    def __init__(
        self,
        vocab_size: int = 256,
        d_byte: int = 64,
        d_model: int = 128,
        use_temporal_conv: bool = False,
        conv_k: int = 8
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_byte = d_byte
        self.d_model = d_model
        self.use_temporal_conv = use_temporal_conv
        self.conv_k = conv_k

        self.patch_to_byte = BitLinear(d_model, d_byte, bias=False)
        self.fusion = BitLinear(2 * d_byte, d_byte, bias=False)
        self.norm1_scale = nn.Parameter(torch.ones(d_byte))

        if use_temporal_conv:
            self.temporal_conv = nn.Conv1d(
                in_channels=d_byte,
                out_channels=d_byte,
                kernel_size=conv_k,
                padding=0,
                groups=d_byte,
                bias=False
            )
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

        # Stage 1: Fusion
        fused_raw = self.fusion(torch.cat([h_byte, patch_h], dim=-1))
        fused = triton_rms_norm(F.silu(fused_raw), self.norm1_scale)

        # Stage 1.5: Optional Temporal Continuity Conv (Arm 1)
        if self.use_temporal_conv:
            fused_pad = F.pad(fused, (0, 0, self.conv_k - 1, 0)).to(self.temporal_conv.weight.dtype)
            fused_conv = self.temporal_conv(fused_pad.transpose(1, 2)).transpose(1, 2).to(fused.dtype)
            fused = self.conv_act(fused_conv)

        # Stage 2: SwiGLU + LM Head
        gate = F.silu(self.gate_proj(fused))
        val = self.val_proj(fused)
        h2 = self.down_proj(gate * val)
        h2_norm = triton_rms_norm(h2, self.norm2_scale)
        logits = self.lm_head(h2_norm)
        return logits


class TestTorosHybridModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 128,
        d_byte: int = 64,
        n_encoder_layers: int = 4,
        n_heads: int = 4,
        target_patch_size: int = 16,
        use_decoder_temporal_conv: bool = False,
        use_pyramidal_dilations: bool = False,
        use_word_aligned_patching: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.d_byte = d_byte
        self.n_encoder_layers = n_encoder_layers
        self.target_patch_size = target_patch_size
        self.use_word_aligned_patching = use_word_aligned_patching

        self.byte_encoder = ByteLocalEncoder(vocab_size=vocab_size, d_byte=d_byte, kernel_size=4)
        self.patcher = EntropyPatcher(d_byte=d_byte, d_model=dim, target_patch_size=target_patch_size)

        # Dilations for blocks
        dilations = [2 ** i for i in range(n_encoder_layers)] if use_pyramidal_dilations else [1] * n_encoder_layers
        self.blocks = nn.ModuleList([
            HybridEncoderBlock(dim=dim, n_heads=n_heads, dilation=dilations[i], layer_idx=i)
            for i in range(n_encoder_layers)
        ])

        # Dense readout highway
        self.layer_readout_weights = nn.Parameter(torch.zeros(n_encoder_layers + 1))
        with torch.no_grad():
            self.layer_readout_weights.fill_(0.0)
            self.layer_readout_weights[-1] = 1.5

        self.norm_out_scale = nn.Parameter(torch.ones(dim))
        self.sos_patch = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.byte_decoder = TemporalByteDecoder(
            vocab_size=vocab_size,
            d_byte=d_byte,
            d_model=dim,
            use_temporal_conv=use_decoder_temporal_conv,
            conv_k=8
        )

    def forward(self, byte_ids, targets=None, return_per_item_loss=False):
        B, T = byte_ids.shape
        P = self.target_patch_size
        h_byte, boundary_logits = self.byte_encoder(byte_ids)

        if self.use_word_aligned_patching:
            # Bias boundary logits on word delimiters (space=32, punctuation)
            is_delimiter = (byte_ids == 32) | (byte_ids == 44) | (byte_ids == 46) | (byte_ids == 10)
            boundary_boost = torch.where(is_delimiter, torch.tensor(3.0, device=byte_ids.device), torch.tensor(0.0, device=byte_ids.device))
            boundary_in = boundary_logits + boundary_boost
        else:
            boundary_in = torch.zeros_like(boundary_logits)

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
            if return_per_item_loss:
                # Per-sequence loss for active screening
                ce = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), reduction='none')
                return ce.view(B, T).mean(dim=1)
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return loss
        return logits


# ---------------------------------------------------------------------------
# Training Harness
# ---------------------------------------------------------------------------

def run_experiment_arm(
    arm_name: str,
    model_kwargs: dict,
    train_data: np.memmap,
    val_data: np.memmap,
    use_active_filtering: bool = False,
    steps: int = 350,
    batch_size: int = 16,
    seq_len: int = 256,
    lr: float = 3e-3,
    device: str = "cuda"
):
    print(f"\n>>> Running Arm: {arm_name}")
    set_seed(42)

    model = TestTorosHybridModel(**model_kwargs).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"    Params: {param_count:,d}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup_steps = 30

    def get_lr(s):
        if s < warmup_steps:
            return lr * (s + 1) / warmup_steps
        progress = (s - warmup_steps) / max(1, steps - warmup_steps)
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

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

        t0 = time.time()
        optimizer.zero_grad()

        if use_active_filtering:
            # Draw 2xB candidates, score with no_grad, train on top B with highest reducible error
            candidate_size = batch_size * 2
            ix_cand = np.random.randint(0, high_train, size=candidate_size)
            x_cand = torch.from_numpy(np.stack([train_data[i : i + seq_len] for i in ix_cand])).long().to(device)
            y_cand = torch.from_numpy(np.stack([train_data[i + 1 : i + seq_len + 1] for i in ix_cand])).long().to(device)

            with torch.no_grad():
                scores = model(x_cand, targets=y_cand, return_per_item_loss=True)
                topk_idx = torch.topk(scores, k=batch_size).indices

            x = x_cand[topk_idx]
            y = y_cand[topk_idx]
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

        if step % 70 == 0 or step == steps:
            val_loss, val_ppl, val_bpc = evaluate()
            avg_ms = np.mean(step_times[-50:]) if len(step_times) >= 50 else np.mean(step_times)
            print(f"    Step {step:3d}/{steps} | Train Loss: {loss_val:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:6.2f} | {avg_ms:5.1f} ms/step")

    # Greedy decode probe
    prompt = b"Once upon a time, there was a little "
    prompt_ids = torch.tensor(list(prompt), dtype=torch.long, device=device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        curr_ids = prompt_ids
        for _ in range(64):
            logits = model(curr_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

    generated_bytes = bytes(curr_ids[0].tolist())
    gen_slice = generated_bytes[len(prompt):]
    four_grams = [gen_slice[i:i+4] for i in range(len(gen_slice) - 3)]
    distinct_4gram = len(set(four_grams)) / max(1, len(four_grams)) if four_grams else 0.0
    sample_text = gen_slice[:48].decode("utf-8", errors="replace").replace("\n", "\\n")

    print(f"    Repetition Probe: Distinct-4gram = {distinct_4gram:.3f} | Sample: \"{sample_text}\"")

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
    print(f"   TOROS-HYBRID DATA-EFFICIENCY FRONTIER ABLATION (Round 2) (Device: {device})")
    print("   Evaluating 4 Advanced Architectural & Active Learning Inductive Levers")
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
    D_BYTE = 64
    N_LAYERS = 4
    N_HEADS = 4

    arms = [
        (
            "Arm 0: Baseline (Current TorosHybrid Defaults)",
            {"dim": DIM, "d_byte": D_BYTE, "n_encoder_layers": N_LAYERS, "n_heads": N_HEADS,
             "use_decoder_temporal_conv": False, "use_pyramidal_dilations": False, "use_word_aligned_patching": False},
            False
        ),
        (
            "Arm 1: Decoder Temporal Continuity (k=8 Conv)",
            {"dim": DIM, "d_byte": D_BYTE, "n_encoder_layers": N_LAYERS, "n_heads": N_HEADS,
             "use_decoder_temporal_conv": True, "use_pyramidal_dilations": False, "use_word_aligned_patching": False},
            False
        ),
        (
            "Arm 2: Pyramidal Dilated Convs (d=1,2,4,8)",
            {"dim": DIM, "d_byte": D_BYTE, "n_encoder_layers": N_LAYERS, "n_heads": N_HEADS,
             "use_decoder_temporal_conv": False, "use_pyramidal_dilations": True, "use_word_aligned_patching": False},
            False
        ),
        (
            "Arm 3: Word-Boundary Aligned Patching",
            {"dim": DIM, "d_byte": D_BYTE, "n_encoder_layers": N_LAYERS, "n_heads": N_HEADS,
             "use_decoder_temporal_conv": False, "use_pyramidal_dilations": False, "use_word_aligned_patching": True},
            False
        ),
        (
            "Arm 4: Active Info-Gain Filtering (rho-Loss)",
            {"dim": DIM, "d_byte": D_BYTE, "n_encoder_layers": N_LAYERS, "n_heads": N_HEADS,
             "use_decoder_temporal_conv": False, "use_pyramidal_dilations": False, "use_word_aligned_patching": False},
            True
        ),
    ]

    results = []
    for name, kwargs, use_filter in arms:
        res = run_experiment_arm(
            arm_name=name,
            model_kwargs=kwargs,
            train_data=train_data,
            val_data=val_data,
            use_active_filtering=use_filter,
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
