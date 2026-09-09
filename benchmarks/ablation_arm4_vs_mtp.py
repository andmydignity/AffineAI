#!/usr/bin/env python3
"""
TorosHybrid Data-Efficiency Ablation: Arm 4 vs Arm 4 with MTP (SimpleStories 100MB)
===================================================================================
Compares:
  - Arm 4: Combined Best (Trunk Conv k=4 + Readout + Decoder Conv k=8 + Word-Aligned, No MTP)
  - Arm 4 + MTP (1 Head, lambda=0.3): Auxiliary foresight loss on decoder byte hidden state
  - Arm 4 + MTP (2 Heads, lambda=0.3): 2 sequential foresight heads (k=2, k=3)

Hardware: NVIDIA RTX 3050 GPU with Triton Hardware Kernels & HybridMuonAdamW.
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

# Hardware acceleration kernels & core modules
from affine_ai.kernels.triton_rms_norm import triton_rms_norm
from affine_ai.core.norm import RMSNorm
from affine_ai.core.associative import FusedMonarchChain, MonarchPermutationChain
from affine_ai.core.bitlinear import TernaryBitLinearSwiGLU, BitLinear
from affine_ai.models.blt import ByteLocalEncoder, EntropyPatcher, ByteLocalDecoder
from affine_ai.models.mtp import ASDAGMTPModule
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
        x_pad = F.pad(x, (0, 0, self.eff_pad, 0)).to(self.conv.weight.dtype)
        x_conv = self.conv(x_pad.transpose(1, 2)).transpose(1, 2).to(x.dtype)
        return self.act(x_conv)


class FastGLAMixer(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, num_stages: int = 4, seed_offset: int = 0):
        super().__init__()
        assert d_model % n_heads == 0
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
    def __init__(self, dim: int, n_heads: int = 4, dilation: int = 1, layer_idx: int = 0, use_conv_prefix: bool = True):
        super().__init__()
        self.norm1_scale = nn.Parameter(torch.ones(dim))
        self.norm2_scale = nn.Parameter(torch.ones(dim))
        self.use_conv_prefix = use_conv_prefix
        if use_conv_prefix:
            self.conv_prefix = DilatedConvPrefix(dim, kernel_size=4, dilation=dilation)
        else:
            self.conv_prefix = None
        self.time_mixer = FastGLAMixer(d_model=dim, n_heads=n_heads, seed_offset=layer_idx * 10)
        self.channel_mixer = TernaryBitLinearSwiGLU(dim, expand=2)

    def forward(self, x):
        h1 = triton_rms_norm(x, self.norm1_scale)
        if self.use_conv_prefix and self.conv_prefix is not None:
            h1 = self.conv_prefix(h1)
        x = x + self.time_mixer(h1)
        h2 = triton_rms_norm(x, self.norm2_scale)
        x = x + self.channel_mixer(h2)
        return x


class TemporalByteDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        d_byte: int = 64,
        d_model: int = 128,
        use_temporal_conv: bool = True,
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

    def forward(self, h_byte, latent_patches, patch_assignments, return_hidden: bool = False):
        B, T, _ = h_byte.shape
        M = latent_patches.shape[1]
        idx_expanded = patch_assignments.clamp(0, M - 1).unsqueeze(-1).expand(-1, -1, self.d_model)
        patch_context = torch.gather(latent_patches, 1, idx_expanded)
        patch_h = self.patch_to_byte(patch_context)

        # Stage 1: Fusion
        fused_raw = self.fusion(torch.cat([h_byte, patch_h], dim=-1))
        fused = triton_rms_norm(F.silu(fused_raw), self.norm1_scale)

        # Stage 1.5: Temporal Continuity Conv (Arm 4)
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

        if return_hidden:
            return logits, h2_norm
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
        use_conv_prefix: bool = True,
        use_dense_readout: bool = True,
        use_decoder_temporal_conv: bool = True,
        use_word_aligned_patching: bool = True,
        use_mtp: bool = False,
        num_mtp_heads: int = 1,
        mtp_lambda: float = 0.3
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.d_byte = d_byte
        self.n_encoder_layers = n_encoder_layers
        self.target_patch_size = target_patch_size
        self.use_conv_prefix = use_conv_prefix
        self.use_dense_readout = use_dense_readout
        self.use_word_aligned_patching = use_word_aligned_patching
        self.use_mtp = use_mtp
        self.num_mtp_heads = num_mtp_heads
        self.mtp_lambda = mtp_lambda

        self.byte_encoder = ByteLocalEncoder(vocab_size=vocab_size, d_byte=d_byte, kernel_size=4)
        self.patcher = EntropyPatcher(d_byte=d_byte, d_model=dim, target_patch_size=target_patch_size)

        self.blocks = nn.ModuleList([
            HybridEncoderBlock(dim=dim, n_heads=n_heads, dilation=1, layer_idx=i, use_conv_prefix=use_conv_prefix)
            for i in range(n_encoder_layers)
        ])

        if use_dense_readout:
            self.layer_readout_weights = nn.Parameter(torch.zeros(n_encoder_layers + 1))
            with torch.no_grad():
                self.layer_readout_weights.fill_(0.0)
                self.layer_readout_weights[-1] = 1.5
        else:
            self.layer_readout_weights = None

        self.norm_out_scale = nn.Parameter(torch.ones(dim))
        self.sos_patch = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.byte_decoder = TemporalByteDecoder(
            vocab_size=vocab_size,
            d_byte=d_byte,
            d_model=dim,
            use_temporal_conv=use_decoder_temporal_conv,
            conv_k=8
        )

        if use_mtp:
            self.mtp = ASDAGMTPModule(
                d_model=d_byte,
                vocab_size=vocab_size,
                num_mtp_heads=num_mtp_heads,
                mtp_lambda=mtp_lambda
            )
        else:
            self.mtp = None

    def forward(self, byte_ids, targets=None):
        B, T = byte_ids.shape
        P = self.target_patch_size
        h_byte, boundary_logits = self.byte_encoder(byte_ids)

        if self.use_word_aligned_patching:
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

        if self.use_dense_readout and self.layer_readout_weights is not None:
            normed_weights = F.softmax(self.layer_readout_weights, dim=0)
            h_dense = sum(w * hl for w, hl in zip(normed_weights, layer_outputs))
            h_latent = triton_rms_norm(h_dense, self.norm_out_scale)
        else:
            h_latent = triton_rms_norm(layer_outputs[-1], self.norm_out_scale)

        causal_patches = torch.cat([self.sos_patch.expand(B, 1, -1), h_latent[:, :-1]], dim=1)

        if self.use_mtp and self.mtp is not None:
            logits, h_decoded = self.byte_decoder(h_byte, causal_patches, patch_assignments, return_hidden=True)
        else:
            logits = self.byte_decoder(h_byte, causal_patches, patch_assignments)
            h_decoded = None

        if targets is not None:
            loss_gen = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            if self.use_mtp and self.mtp is not None and h_decoded is not None:
                _, mtp_loss, _ = self.mtp(h_decoded, targets=targets)
                total_loss = loss_gen + (mtp_loss if mtp_loss is not None else 0.0)
                return total_loss, loss_gen
            return loss_gen, loss_gen
        return logits


def run_experiment_arm(
    arm_name: str,
    model_kwargs: dict,
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

    model = TestTorosHybridModel(**model_kwargs).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"    Params: {param_count:,d}")
    print(f"    Active Inductive Biases:")
    print(f"      - Trunk Conv Prefix (k=4): {getattr(model, 'use_conv_prefix', False)}")
    print(f"      - Dense Readout Highway   : {getattr(model, 'use_dense_readout', False)}")
    print(f"      - Decoder Temporal Conv   : {getattr(model.byte_decoder, 'use_temporal_conv', False)}")
    print(f"      - Word-Aligned Patching   : {getattr(model, 'use_word_aligned_patching', False)}")
    print(f"      - Multi-Token Prediction  : {getattr(model, 'use_mtp', False)} (heads={getattr(model, 'num_mtp_heads', 0)}, lambda={getattr(model, 'mtp_lambda', 0.0)})")

    optimizer = HybridMuonAdamW(
        model=model,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        muon_momentum=0.95,
        adamw_weight_decay=0.01
    )
    muon_p = sum(p.numel() for p in optimizer.muon_opt.param_groups[0]['params']) if optimizer.muon_opt else 0
    adamw_p = sum(p.numel() for g in optimizer.adamw_opt.param_groups for p in g['params']) if optimizer.adamw_opt else 0
    print(f"    Optimizer: HybridMuonAdamW (Muon: {muon_p:,d} params [lr={muon_lr}], AdamW: {adamw_p:,d} params [lr={adamw_lr}])")

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
            res = model(x_v, targets=y_v)
            loss_v = res[1] if isinstance(res, tuple) else res
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

        out = model(x, targets=y)
        if isinstance(out, tuple):
            loss_total, loss_gen = out
        else:
            loss_total, loss_gen = out, out

        loss_total.backward()

        if step == 1:
            if getattr(model, 'use_dense_readout', False) and model.layer_readout_weights is not None:
                assert model.layer_readout_weights.grad is not None, "Dense readout highway received no grad!"
                grad_norm = model.layer_readout_weights.grad.norm().item()
                print(f"    [Active Verification] Step 1 layer_readout_weights grad norm: {grad_norm:.5f}")
            if getattr(model, 'use_conv_prefix', False) and model.blocks[0].conv_prefix is not None:
                assert model.blocks[0].conv_prefix.conv.weight.grad is not None, "Trunk conv prefix received no grad!"
                c_grad_norm = model.blocks[0].conv_prefix.conv.weight.grad.norm().item()
                print(f"    [Active Verification] Step 1 trunk conv prefix grad norm: {c_grad_norm:.5f}")
            if getattr(model, 'use_mtp', False) and model.mtp is not None:
                assert model.mtp.heads[0].proj.weight.grad is not None, "MTP proj received no grad!"
                m_grad_norm = model.mtp.heads[0].proj.weight.grad.norm().item()
                print(f"    [Active Verification] Step 1 MTP Head 0 proj grad norm: {m_grad_norm:.5f}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t1 = time.time()

        step_times.append((t1 - t0) * 1000)
        loss_val = loss_gen.item()

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
    print(f"   TOROS-HYBRID MTP ABLATION ON SIMPLESTORIES 100MB (Device: {device})")
    print("   Comparison: Arm 4 vs Arm 4 + MTP (Multi-Token Prediction)")
    print("=" * 85)

    data_path = "data/simplestories_eos.bin"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        return

    # Cap to exactly 100 MB
    MAX_BYTES = 100 * 1024 * 1024
    raw_data = np.memmap(data_path, dtype=np.uint8, mode="r")
    total_len = min(len(raw_data), MAX_BYTES)
    data = raw_data[:total_len]
    split = int(0.95 * total_len)
    train_data = data[:split]
    val_data = data[split:]
    print(f"Source File: {data_path} (File size: {len(raw_data)/(1024**3):.2f} GB)")
    print(f"Active Slice: {total_len:,d} bytes ({total_len/(1024*1024):.1f} MB)")
    print(f"Train Bytes : {len(train_data):,d} (95.0 MB) | Val Bytes: {len(val_data):,d} (5.0 MB)")

    STEPS = 400
    BATCH_SIZE = 16
    SEQ_LEN = 256
    DIM = 128
    D_BYTE = 64
    N_LAYERS = 4
    N_HEADS = 4

    arms = [
        (
            "Arm 4: Combined Best (Decoder Conv + Word-Aligned, No MTP)",
            {
                "dim": DIM, "d_byte": D_BYTE, "n_encoder_layers": N_LAYERS, "n_heads": N_HEADS,
                "use_conv_prefix": True, "use_dense_readout": True,
                "use_decoder_temporal_conv": True, "use_word_aligned_patching": True,
                "use_mtp": False, "num_mtp_heads": 0, "mtp_lambda": 0.0
            }
        ),
        (
            "Arm 4 + MTP (1 Head, lambda=0.3)",
            {
                "dim": DIM, "d_byte": D_BYTE, "n_encoder_layers": N_LAYERS, "n_heads": N_HEADS,
                "use_conv_prefix": True, "use_dense_readout": True,
                "use_decoder_temporal_conv": True, "use_word_aligned_patching": True,
                "use_mtp": True, "num_mtp_heads": 1, "mtp_lambda": 0.3
            }
        ),
        (
            "Arm 4 + MTP (2 Heads, lambda=0.3)",
            {
                "dim": DIM, "d_byte": D_BYTE, "n_encoder_layers": N_LAYERS, "n_heads": N_HEADS,
                "use_conv_prefix": True, "use_dense_readout": True,
                "use_decoder_temporal_conv": True, "use_word_aligned_patching": True,
                "use_mtp": True, "num_mtp_heads": 2, "mtp_lambda": 0.3
            }
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
            muon_lr=0.02,
            adamw_lr=3e-3,
            device=device
        )
        results.append(res)

    print("\n" + "=" * 95)
    print("              FINAL MTP ABLATION RESULTS SUMMARY (SIMPLESTORIES 100MB)")
    print("=" * 95)
    print(f"{'Arm':<52} | {'Params':<8} | {'Val PPL':<8} | {'Val BPC':<8} | {'ms/step':<8} | {'Dist-4g':<8}")
    print("-" * 95)
    for r in results:
        print(f"{r['arm']:<52} | {r['params']:<8,d} | {r['final_val_ppl']:<8.2f} | {r['val_bpc']:<8.3f} | {r['avg_ms_step']:<8.1f} | {r['distinct_4gram']:<8.3f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
