#!/usr/bin/env python3
"""
Benchmark: Local Predictive Coding (LPC) vs Standard Backprop vs Closed-Form Backpressure
========================================================================================
Evaluates whether a multi-layer MatMul-Free ASDAG language model can train effectively
using forward-only, decoupled local predictive coding without cross-layer backward passes.

Dataset:
  - 10 MB slice of TinyStories (data/tinystories_extracted.bin)
  - 90% Train (9 MB) / 10% Val (1 MB)
  - Byte-level vocabulary (256), sequence length 128

Training Variants:
  1. "backprop"     : Standard global end-to-end autograd backward pass.
  2. "backpressure" : Closed-form tape-free hydraulic backpressure error diffusion.
  3. "local_pred"   : Forward-only Local Predictive Coding (LPC) with decoupled layer-wise error heads.
"""

import math
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from affine_ai.models.language_model import ASDAGLanguageModel, ASDAGBlock
from affine_ai.core.norm import RMSNorm
from affine_ai.optim.muon import HybridMuonAdamW


class MemmappedStories10MB:
    """Byte-level data loader for a 10MB slice of TinyStories."""
    def __init__(self, path="data/tinystories_extracted.bin", max_bytes=10_000_000, seq_len=128, split_ratio=0.9):
        raw = np.memmap(path, dtype=np.uint8, mode="r")
        self.total = min(len(raw), max_bytes)
        self.data = raw[:self.total]
        self.split = int(self.total * split_ratio)
        self.seq_len = seq_len
        print(f"Dataset Loaded: {self.total:,} bytes ({self.total / (1024*1024):.2f} MB)")
        print(f"  Train Split : {self.split:,} bytes ({split_ratio*100:.0f}%)")
        print(f"  Val Split   : {self.total - self.split:,} bytes ({(1-split_ratio)*100:.0f}%)")

    def batch(self, split: str, bs: int, device: torch.device):
        if split == "train":
            lo, hi = 0, self.split - self.seq_len - 1
        else:
            lo, hi = self.split, self.total - self.seq_len - 1
        ix = np.random.randint(lo, hi, size=bs)
        x = np.stack([self.data[i:i+self.seq_len] for i in ix])
        y = np.stack([self.data[i+1:i+1+self.seq_len] for i in ix])
        return (torch.from_numpy(x.astype(np.int64)).to(device),
                torch.from_numpy(y.astype(np.int64)).to(device))


class LocalPredictiveModel(nn.Module):
    """
    Wrapper around ASDAGLanguageModel that equips each block with a local predictive head.
    Enables true forward-only, decoupled in-place local error training without cross-layer autograd.
    """
    def __init__(self, base_model: ASDAGLanguageModel):
        super().__init__()
        self.base_model = base_model
        self.d_model = base_model.d_model
        self.vocab_size = base_model.vocab_size
        self.n_layers = len(base_model.blocks)

        # Local predictive heads for each intermediate layer
        # Each layer has a lightweight RMSNorm + Linear head mapping to vocabulary
        self.local_heads = nn.ModuleList([
            nn.Sequential(
                RMSNorm(self.d_model),
                nn.Linear(self.d_model, self.vocab_size, bias=False)
            )
            for _ in range(self.n_layers)
        ])

    def forward(self, input_ids: torch.Tensor):
        return self.base_model(input_ids)

    def forward_local_predictive_step(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        optimizers: list,
        criterion: nn.CrossEntropyLoss
    ) -> float:
        """
        Executes a forward-only local predictive coding step.
        For each layer:
          1. Computes local block output with detached inputs (no gradient to previous layers).
          2. Computes local prediction of the target token.
          3. Evaluates local prediction error and immediately updates layer parameters.
          4. Memory for the layer is freed before moving to the next.
        """
        B, T = input_ids.shape
        x = self.base_model.tok_embeddings(input_ids)
        total_loss = 0.0

        curr_h = x

        for idx, block in enumerate(self.base_model.blocks):
            # Detach input to strictly isolate gradients between layers
            if idx > 0:
                curr_h = curr_h.detach()
                curr_h.requires_grad_(True)

            # Block forward
            next_h = block(curr_h)

            # Local prediction head for this layer
            local_logits = self.local_heads[idx](next_h)
            loss_i = criterion(local_logits.view(-1, self.vocab_size), targets.view(-1))

            # Optimize this layer strictly locally
            opt_i = optimizers[idx]
            opt_i.zero_grad()
            loss_i.backward()
            torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
            opt_i.step()

            total_loss += loss_i.item()
            curr_h = next_h

        # Final head update
        curr_h = curr_h.detach()
        curr_h.requires_grad_(True)
        final_h = self.base_model.norm_f(curr_h)
        final_logits = self.base_model.lm_head(final_h)
        loss_final = criterion(final_logits.view(-1, self.vocab_size), targets.view(-1))

        opt_final = optimizers[-1]
        opt_final.zero_grad()
        loss_final.backward()
        opt_final.step()

        return loss_final.item()


@torch.no_grad()
def eval_ppl(model: nn.Module, data: MemmappedStories10MB, device: torch.device, batches: int = 25, bs: int = 64) -> float:
    """Evaluates validation perplexity on held-out slice."""
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    total_nll, total_tok = 0.0, 0

    for _ in range(batches):
        x, y = data.batch("val", bs, device)
        logits = model(x)
        total_nll += crit(logits.view(-1, 256), y.view(-1)).item()
        total_tok += y.numel()

    model.train()
    return math.exp(total_nll / max(total_tok, 1))


def run_single_variant(
    kind: str,
    data: MemmappedStories10MB,
    steps: int = 1000,
    batch_size: int = 32,
    lr: float = 3e-3,
    d_model: int = 96,
    n_layers: int = 3,
    n_heads: int = 4,
    device: torch.device = torch.device("cpu")
) -> dict:
    """Runs a complete training run for a single credit assignment variant."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Base model configuration
    base_model = ASDAGLanguageModel(
        vocab_size=256,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        channel_mixer_type="ternary_swiglu",
        use_blt=False,
        dtype=torch.float32  # Stable FP32 for comparative analysis
    ).to(device)

    nparams = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"\n========================================================")
    print(f"Variant: '{kind.upper()}' | Parameters: {nparams:,}")
    print(f"========================================================")

    criterion = nn.CrossEntropyLoss()

    if kind == "local_pred":
        model = LocalPredictiveModel(base_model).to(device)
        # Create separate decoupled optimizer per block
        layer_optimizers = []
        # First optimizer includes embedding + block 0 + local_head 0
        layer_optimizers.append(torch.optim.AdamW(
            list(model.base_model.tok_embeddings.parameters()) +
            list(model.base_model.blocks[0].parameters()) +
            list(model.local_heads[0].parameters()),
            lr=lr, weight_decay=0.01
        ))
        for i in range(1, n_layers):
            layer_optimizers.append(torch.optim.AdamW(
                list(model.base_model.blocks[i].parameters()) +
                list(model.local_heads[i].parameters()),
                lr=lr, weight_decay=0.01
            ))
        # Final head optimizer
        layer_optimizers.append(torch.optim.AdamW(
            list(model.base_model.norm_f.parameters()) +
            list(model.base_model.lm_head.parameters()),
            lr=lr, weight_decay=0.01
        ))
    else:
        model = base_model
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)

    # Initial Validation PPL
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    initial_ppl = eval_ppl(model, data, device, batches=10, bs=batch_size)
    print(f"  Step 0: Initial Val PPL = {initial_ppl:.2f}")

    t0 = time.time()
    loss_history = []

    for step in range(1, steps + 1):
        x, y = data.batch("train", batch_size, device)

        if kind == "backprop":
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, 256), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step_loss = loss.item()

        elif kind == "backpressure":
            # Hydraulic backpressure mode
            opt.zero_grad()
            logits = model(x)
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                y_onehot = F.one_hot(y, num_classes=256).to(logits.dtype)
                logits_error = (y_onehot - probs) / float(x.size(0) * x.size(1))
                loss = criterion(logits.view(-1, 256), y.view(-1))
            model.backward_backpressure(logits_error)
            opt.step()
            sched.step()
            step_loss = loss.item()

        elif kind == "local_pred":
            step_loss = model.forward_local_predictive_step(x, y, layer_optimizers, criterion)

        loss_history.append(step_loss)

        if step % 200 == 0 or step == steps:
            val_ppl = eval_ppl(model, data, device, batches=20, bs=batch_size)
            elapsed = time.time() - t0
            tok_s = (step * batch_size * data.seq_len) / elapsed
            print(f"  Step {step:>5}/{steps} | Loss: {step_loss:.3f} | Val PPL: {val_ppl:.2f} | Speed: {tok_s:,.0f} tok/s | Elapsed: {elapsed:.1f}s", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_vram_mb = 0.0

    total_time = time.time() - t0
    final_ppl = eval_ppl(model, data, device, batches=40, bs=batch_size)

    return {
        "kind": kind,
        "final_ppl": final_ppl,
        "initial_ppl": initial_ppl,
        "total_time_s": total_time,
        "tok_per_sec": (steps * batch_size * data.seq_len) / total_time,
        "peak_vram_mb": peak_vram_mb,
        "loss_history": loss_history
    }


def main():
    parser = argparse.ArgumentParser(description="Test Local Predictive Coding on TinyStories 10MB")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps per variant")
    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=96, help="Model hidden dimension")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of ASDAG blocks")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--variants", nargs="+", default=["backprop", "local_pred", "backpressure"],
                        help="List of variants to test: backprop, local_pred, backpressure")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f"TinyStories 10MB Local Predictive Coding Benchmark")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Config: d_model={args.d_model}, n_layers={args.n_layers}, n_heads={args.n_heads}, steps={args.steps}")
    print(f"========================================================")

    data = MemmappedStories10MB("data/tinystories_extracted.bin", max_bytes=10_000_000, seq_len=128, split_ratio=0.9)

    results = {}
    for var in args.variants:
        res = run_single_variant(
            kind=var,
            data=data,
            steps=args.steps,
            batch_size=args.bs,
            lr=args.lr,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            device=device
        )
        results[var] = res

    print("\n" + "=" * 80)
    print("                TINYSTORIES 10MB BENCHMARK COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Method':<18} | {'Initial PPL':<12} | {'Final Val PPL':<14} | {'Speed (tok/s)':<15} | {'Peak VRAM':<12}")
    print("-" * 80)
    for k, v in results.items():
        vram_str = f"{v['peak_vram_mb']:.1f} MB" if device.type == "cuda" else "N/A"
        print(f"{k.upper():<18} | {v['initial_ppl']:<12.2f} | {v['final_ppl']:<14.2f} | {v['tok_per_sec']:>13,.0f} | {vram_str:>10}")
    print("=" * 80)


if __name__ == "__main__":
    main()
