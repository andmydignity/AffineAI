#!/usr/bin/env python3
"""
Comprehensive Training Speed Benchmark:
Single Ternary vs. Double Ternary vs. 5-State Power-of-Two (POT)

Measures forward pass, backward pass, optimizer step, training throughput (tok/s),
and peak VRAM footprint across:
1. Isolated ASDAG FFN Layer (AbsTopK 25% Sparsity, Qwen3.5-4B scale: Dim=2560, Hidden=9216)
2. Full Transformer Block (Qwen35Block: DeltaNet/Attention + ASDAG FFN + RMSNorms)
"""

import os
import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.models.qwen35_asdag import Qwen35ASDAGConfig, Qwen35ASDAGFFN, Qwen35Block


def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_ffn_benchmark(batch_size=2, seq_len=256, dim=2560, intermediate_dim=9216, warmup=10, iters=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_tokens = batch_size * seq_len
    modes = [
        ("Single Ternary (1.58b)", "ternary"),
        ("5-State POT (2.32b)", "pot5"),
        ("Double Ternary (3.17b)", "dual_ternary"),
    ]

    results = {}

    print("\n" + "=" * 96)
    print(f"  ISOLATED ASDAG FFN TRAINING SPEEDTEST (Dim={dim}, Intermediate={intermediate_dim})")
    print(f"  Batch: {batch_size}, SeqLen: {seq_len} ({total_tokens} tokens/step) | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 96)

    for label, mode in modes:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        cfg = Qwen35ASDAGConfig(
            dim=dim,
            intermediate_dim=intermediate_dim,
            weight_quant_mode=mode,
            sparsity_mode="abstopk",
            retain_ratio=0.25,
            dtype=torch.bfloat16,
        )
        ffn = Qwen35ASDAGFFN(cfg).to(device)
        ffn.train()
        optimizer = torch.optim.AdamW(ffn.parameters(), lr=1e-4)

        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(warmup):
            optimizer.zero_grad(set_to_none=True)
            out = ffn(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
        sync_device()

        # Timed benchmark with CUDA events
        fwd_times = []
        bwd_times = []
        step_times = []

        for _ in range(iters):
            sync_device()
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            out = ffn(x)
            sync_device()
            t1 = time.perf_counter()

            loss = out.sum()
            loss.backward()
            sync_device()
            t2 = time.perf_counter()

            optimizer.step()
            sync_device()
            t3 = time.perf_counter()

            fwd_times.append((t1 - t0) * 1000.0)
            bwd_times.append((t2 - t1) * 1000.0)
            step_times.append((t3 - t0) * 1000.0)

        avg_fwd = sum(fwd_times) / len(fwd_times)
        avg_bwd = sum(bwd_times) / len(bwd_times)
        avg_step = sum(step_times) / len(step_times)
        tok_per_sec = total_tokens / (avg_step / 1000.0)
        peak_vram = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0

        results[mode] = {
            "label": label,
            "fwd_ms": avg_fwd,
            "bwd_ms": avg_bwd,
            "step_ms": avg_step,
            "tok_per_sec": tok_per_sec,
            "peak_vram_mb": peak_vram,
        }

    return results


def run_block_benchmark(batch_size=2, seq_len=256, dim=2560, intermediate_dim=9216, warmup=8, iters=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_tokens = batch_size * seq_len
    modes = [
        ("Single Ternary (1.58b)", "ternary"),
        ("5-State POT (2.32b)", "pot5"),
        ("Double Ternary (3.17b)", "dual_ternary"),
    ]

    results = {}

    print("\n" + "=" * 96)
    print(f"  FULL TRANSFORMER BLOCK TRAINING SPEEDTEST (Qwen35Block: DeltaNet + ASDAG FFN)")
    print(f"  Batch: {batch_size}, SeqLen: {seq_len} ({total_tokens} tokens/step) | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 96)

    for label, mode in modes:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        cfg = Qwen35ASDAGConfig(
            dim=dim,
            intermediate_dim=intermediate_dim,
            weight_quant_mode=mode,
            sparsity_mode="abstopk",
            retain_ratio=0.25,
            dtype=torch.bfloat16,
        )
        block = Qwen35Block(cfg, layer_idx=0).to(device)
        block.train()
        optimizer = torch.optim.AdamW(block.parameters(), lr=1e-4)

        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(warmup):
            optimizer.zero_grad(set_to_none=True)
            out, _ = block(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
        sync_device()

        # Timed benchmark
        fwd_times = []
        bwd_times = []
        step_times = []

        for _ in range(iters):
            sync_device()
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            out, _ = block(x)
            sync_device()
            t1 = time.perf_counter()

            loss = out.sum()
            loss.backward()
            sync_device()
            t2 = time.perf_counter()

            optimizer.step()
            sync_device()
            t3 = time.perf_counter()

            fwd_times.append((t1 - t0) * 1000.0)
            bwd_times.append((t2 - t1) * 1000.0)
            step_times.append((t3 - t0) * 1000.0)

        avg_fwd = sum(fwd_times) / len(fwd_times)
        avg_bwd = sum(bwd_times) / len(bwd_times)
        avg_step = sum(step_times) / len(step_times)
        tok_per_sec = total_tokens / (avg_step / 1000.0)
        peak_vram = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0

        results[mode] = {
            "label": label,
            "fwd_ms": avg_fwd,
            "bwd_ms": avg_bwd,
            "step_ms": avg_step,
            "tok_per_sec": tok_per_sec,
            "peak_vram_mb": peak_vram,
        }

    return results


def print_comparison_table(title, results, baseline_key="dual_ternary"):
    baseline_step = results[baseline_key]["step_ms"]
    baseline_tok = results[baseline_key]["tok_per_sec"]

    print("\n" + "-" * 96)
    print(f" {title.upper()} ")
    print("-" * 96)
    header = f"{'Quantization Mode':<26} | {'Forward':<10} | {'Backward':<10} | {'Step (F+B+Opt)':<15} | {'Throughput':<15} | {'Speedup vs Dual':<12}"
    print(header)
    print("-" * 96)

    for key, r in results.items():
        speedup = baseline_step / r["step_ms"]
        speedup_str = f"{speedup:.2f}x faster" if speedup >= 1.0 else f"{speedup:.2f}x slower"
        line = (
            f"{r['label']:<26} | "
            f"{r['fwd_ms']:>7.2f} ms | "
            f"{r['bwd_ms']:>7.2f} ms | "
            f"{r['step_ms']:>10.2f} ms    | "
            f"{r['tok_per_sec']:>10.1f} tok/s | "
            f"{speedup_str:<12}"
        )
        print(line)
    print("-" * 96)


if __name__ == "__main__":
    print("================================================================================================")
    print("      TRAINING SPEEDTEST: SINGLE TERNARY vs 5-STATE POT vs DOUBLE TERNARY")
    print("================================================================================================")

    ffn_res = run_ffn_benchmark(batch_size=2, seq_len=256, dim=2560, intermediate_dim=9216, warmup=8, iters=25)
    print_comparison_table("Isolated ASDAG FFN Layer Speed Comparison", ffn_res)

    block_res = run_block_benchmark(batch_size=2, seq_len=256, dim=2560, intermediate_dim=9216, warmup=6, iters=20)
    print_comparison_table("Full Transformer Block Speed Comparison", block_res)
