#!/usr/bin/env python3
"""
TinyStories 100MB Benchmark: 1:16 Sparsity, Top-2 Routing (LPC vs Standard Backprop)
====================================================================================
Evaluates the full ASDAG architecture on a 100MB slice of TinyStories with:
- 1:16 Structured Sparsity (sparsity_ratio = 0.9375)
- Top-2 Routing (top_k = 2)
- Training Paradigm: Local Predictive Coding (LPC) OR Standard End-to-End Backprop
- Muon + AdamW optimization
"""

import math
import os
import sys
import time
import argparse
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from affine_ai.models.language_model import ASDAGLanguageModel
from affine_ai.core.lpc import LocalPredictiveLanguageModel
from affine_ai.optim.muon import HybridMuonAdamW


class TinyStories100MBLoader:
    """100MB Dataset loader with 90/10 split."""
    def __init__(self, path="data/tinystories_extracted.bin", max_bytes=100_000_000, seq_len=128, split_ratio=0.9):
        raw = np.fromfile(path, dtype=np.uint8)
        self.total = min(len(raw), max_bytes)
        self.data = raw[:self.total]
        self.split = int(self.total * split_ratio)
        self.seq_len = seq_len
        self.train_data = torch.from_numpy(self.data[:self.split].astype(np.int64))
        self.val_data = torch.from_numpy(self.data[self.split:].astype(np.int64))

        print("=" * 75)
        print("                 TINYSTORIES 100MB DATASET LOADED")
        print("=" * 75)
        print(f"Total Bytes : {self.total:,} bytes ({self.total / (1024*1024):.2f} MB)")
        print(f"Train Split : {len(self.train_data):,} bytes (90%)")
        print(f"Val Split   : {len(self.val_data):,} bytes (10%)")
        print("=" * 75)

    def get_batch(self, split: str, bs: int, device: torch.device):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(0, len(data) - self.seq_len - 1, (bs,))
        x = torch.stack([data[i:i + self.seq_len] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + self.seq_len] for i in ix]).to(device)
        return x, y


@torch.no_grad()
def evaluate_ppl(model: nn.Module, loader: TinyStories100MBLoader, device: torch.device, iters: int = 25, bs: int = 16) -> Tuple[float, float]:
    """Computes validation loss and perplexity on held-out 10MB slice."""
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    total_nll, total_tok = 0.0, 0

    for _ in range(iters):
        x, y = loader.get_batch("val", bs, device)
        logits = model(x)
        total_nll += crit(logits.view(-1, 256), y.view(-1)).item()
        total_tok += y.numel()

    model.train()
    val_loss = total_nll / max(total_tok, 1)
    val_ppl = math.exp(min(val_loss, 20.0))
    return val_loss, val_ppl


def run_100mb_test(
    steps: int = 500,
    batch_size: int = 16,
    seq_len: int = 128,
    d_model: int = 384,
    n_layers: int = 8,
    n_heads: int = 6,
    lr: float = 3e-3,
    muon_lr: float = 0.02,
    eval_interval: int = 50,
    mode: str = "backprop",
    device_str: str = "auto"
):
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"\nDevice: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Architecture Config:")
    print(f"  - Hidden Dim (d_model)   : {d_model}")
    print(f"  - Layers (n_layers)      : {n_layers}")
    print(f"  - Attention Heads        : {n_heads}")
    print(f"  - Channel Mixer          : ASDAG Tree (1:16 Structured Sparsity, 93.75% Zeros)")
    print(f"  - Routing Mode           : Top-2 Hierarchical Sign-Hyperplane (top_k=2)")
    print(f"  - Training Paradigm      : {mode.upper()} ({'Forward-Only LPC' if mode=='lpc' else 'End-to-End Global Backpropagation'})")
    print(f"  - Optimizer              : {'Decoupled Per-Layer' if mode=='lpc' else 'Global'} Hybrid Muon (Newton-Schulz) + AdamW")
    print(f"  - Batch Size & Context   : bs={batch_size}, seq_len={seq_len} ({batch_size * seq_len:,} tok/step)")
    print(f"  - Total Steps            : {steps:,}")

    # 1. Load Dataset
    loader = TinyStories100MBLoader(path="data/tinystories_extracted.bin", max_bytes=100_000_000, seq_len=seq_len)

    # 2. Build ASDAG Language Model with 1:16 Sparsity and Top-2 Routing
    torch.manual_seed(42)
    base_model = ASDAGLanguageModel(
        vocab_size=256,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        num_leaves=16,
        sparsity_ratio=0.9375,  # 1:16 Structured Sparsity (1 active of 16)
        top_k=2,                # Top-2 routing
        leaf_mode="permutation",
        channel_mixer_type="asdag_tree",
        use_blt=False,
        dtype=torch.float32
    ).to(device)

    num_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"\nModel Parameters: {num_params:,}")

    # 3. Setup Training Infrastructure
    if mode == "lpc":
        eval_model = LocalPredictiveLanguageModel(base_model).to(device)
        optimizers = eval_model.get_default_lpc_optimizers(
            use_muon=True,
            muon_lr=muon_lr,
            lr=lr,
            weight_decay=0.01
        )
    else: # backprop
        eval_model = base_model
        global_optimizer = HybridMuonAdamW(
            model=base_model,
            muon_lr=muon_lr,
            adamw_lr=lr,
            muon_momentum=0.95,
            adamw_weight_decay=0.01
        )

    # 4. Initial Baseline Validation
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    init_val_loss, init_val_ppl = evaluate_ppl(eval_model, loader, device, iters=15, bs=8)
    print(f"\nStep 0 | Initial Val Loss: {init_val_loss:.4f} | Initial Val PPL: {init_val_ppl:6.2f}")

    start_time = time.time()
    tokens_processed = 0

    print("\n" + "-" * 75)
    print(f"{'Step':<10} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val PPL':<10} | {'Speed (tok/s)':<14} | {'Elapsed':<8}")
    print("-" * 75)

    for step in range(1, steps + 1):
        if mode == "lpc":
            x, y = loader.get_batch("train", batch_size, device)
            stats = eval_model.forward_lpc_step(x, y, optimizers, grad_clip=1.0)
            train_loss = stats["loss"]
        else:  # Standard Backpropagation with micro-batching to fit in VRAM
            global_optimizer.zero_grad(set_to_none=True)
            micro_bs = min(8, batch_size)
            accum_steps = max(1, batch_size // micro_bs)
            total_loss_val = 0.0
            for _ in range(accum_steps):
                xm, ym = loader.get_batch("train", micro_bs, device)
                logits = base_model(xm)
                loss = F.cross_entropy(logits.view(-1, 256), ym.view(-1)) / accum_steps
                loss.backward()
                total_loss_val += loss.item() * accum_steps
            if torch.cuda.is_available():
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
            global_optimizer.step()
            train_loss = total_loss_val / accum_steps

        tokens_processed += batch_size * seq_len

        if step % eval_interval == 0 or step == steps:
            val_loss, val_ppl = evaluate_ppl(eval_model, loader, device, iters=20, bs=8)
            elapsed = time.time() - start_time
            tok_per_sec = tokens_processed / elapsed
            print(f"Step {step:<5d}/{steps} | {train_loss:<12.4f} | {val_loss:<10.4f} | {val_ppl:<10.2f} | {tok_per_sec:>12,.0f} | {elapsed:>6.1f}s", flush=True)

        if step % (eval_interval * 2) == 0 or step == steps:
            # Generate sample story
            prompt_str = "Once upon a time, Lily found a"
            prompt_bytes = torch.tensor([[ord(c) for c in prompt_str]], dtype=torch.long, device=device)
            gen_bytes = base_model.generate(prompt_bytes, max_new_tokens=60, temperature=0.7)
            gen_text = bytes(gen_bytes[0].tolist()).decode("utf-8", errors="ignore")
            print(f"  >> Sample @ Step {step}: \"{gen_text}\"\n", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_vram_mb = 0.0

    total_time = time.time() - start_time
    final_val_loss, final_val_ppl = evaluate_ppl(eval_model, loader, device, iters=30, bs=batch_size)

    print("=" * 75)
    print(f"             TINYSTORIES 100MB {mode.upper()} TEST RESULTS")
    print("=" * 75)
    print(f"Dataset Size          : {loader.total / (1024*1024):.2f} MB (TinyStories)")
    print(f"Architecture          : ASDAG Tree (1:16 Sparsity, Top-2 Routing)")
    print(f"Training Paradigm     : {mode.upper()} ({'Forward-Only LPC' if mode=='lpc' else 'End-to-End Global Backprop'})")
    print(f"Total Steps           : {steps:,}")
    print(f"Total Time            : {total_time:.2f} seconds")
    print(f"Average Throughput    : {tokens_processed / total_time:,.0f} tokens / second")
    print(f"Peak VRAM Allocated   : {peak_vram_mb:.1f} MB")
    print(f"Initial Val Loss / PPL: {init_val_loss:.4f} / {init_val_ppl:.2f}")
    print(f"Final Val Loss / PPL  : {final_val_loss:.4f} / {final_val_ppl:.2f}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ASDAG 1:16 Top-2 (LPC vs Backprop) on 100MB TinyStories")
    parser.add_argument("--mode", type=str, default="backprop", choices=["backprop", "lpc"], help="Training mode")
    parser.add_argument("--steps", type=int, default=500, help="Number of training steps")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=384, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--lr", type=float, default=3e-3, help="AdamW learning rate")
    parser.add_argument("--muon_lr", type=float, default=0.02, help="Muon learning rate")
    parser.add_argument("--eval_interval", type=int, default=50, help="Eval interval")
    args = parser.parse_args()

    run_100mb_test(
        steps=args.steps,
        batch_size=args.bs,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        lr=args.lr,
        muon_lr=args.muon_lr,
        eval_interval=args.eval_interval,
        mode=args.mode
    )
