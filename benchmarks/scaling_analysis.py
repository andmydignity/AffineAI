"""
Multi-Scale Complexity and Bottleneck Profiler
==============================================
Systematically measures runtime scaling across:
1. Sequence Length (T in [512, 1024, 2048, 4096]) at CONSTANT total tokens (B * T = 32,768)
   -> An ideal linear model should take constant time. Any growth indicates superlinear (O(T^2)) scaling!
2. Batch Size (B in [8, 16, 32, 64, 96]) at fixed T = 1024
   -> Measures SM saturation efficiency, memory bandwidth cliffs, and launch overhead.
3. Model Dimension (dim in [96, 144, 192, 288]) at fixed B=32, T=1024
   -> Detects tile quantization cliffs and scaling exponent alpha (O(D) vs O(D^2)).
"""

import sys
import os
import time
import math
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig


def measure_step(model, optimizers, bx, by, warmup=5, iters=15):
    # Warmup
    for _ in range(warmup):
        model.forward_lpc_step(bx, by, optimizers, use_async_pipelining=True, sync_loss=False)
    torch.cuda.synchronize()

    # Time breakdown with CUDA events
    e_start = torch.cuda.Event(enable_timing=True)
    e_bwd = torch.cuda.Event(enable_timing=True)

    latencies = []
    for _ in range(iters):
        e_start.record()
        step_res = model.forward_lpc_step(bx, by, optimizers, use_async_pipelining=True, sync_loss=False)
        e_bwd.record()
        torch.cuda.synchronize()
        latencies.append(e_start.elapsed_time(e_bwd))

    mean_ms = float(np.median(latencies))
    tokens_per_step = bx.shape[0] * bx.shape[1]
    tok_s = (tokens_per_step / (mean_ms / 1000.0))
    return mean_ms, tok_s


def test_seq_len_scaling(device="cuda"):
    print("\n" + "=" * 95)
    print(" 1. SEQUENCE LENGTH SCALING (AT CONSTANT TOTAL TOKENS B * T = 32,768)")
    print(" Theory: Ideal linear attention + patching should have constant latency (~0% slope).")
    print(" Any superlinear scaling indicates an O(T^2) hidden attention or patcher bottleneck!")
    print("=" * 95)
    header = f"{'Seq Len (T)':<12} | {'Batch (B)':<10} | {'Tokens/Step':<14} | {'Step Time (ms)':<16} | {'Throughput (tok/s)':<20} | {'Relative Cost':<14}"
    print(header)
    print("-" * 95)

    TOTAL_TOKENS = 32768
    configs = [
        (512, TOTAL_TOKENS // 512),
        (1024, TOTAL_TOKENS // 1024),
        (2048, TOTAL_TOKENS // 2048),
        (4096, TOTAL_TOKENS // 4096),
    ]

    base_time = None
    results = []

    for T, B in configs:
        cfg = TorosHybridConfig(
            dim=144,
            n_encoder_layers=6,
            channel_mixer_type="asdag_tree",
            dtype=torch.bfloat16
        )
        model = TorosHybridLanguageModel(cfg).to(device)
        model.enable_lpc(dtype=torch.bfloat16, device=device)
        optimizers = model.get_default_lpc_optimizers(use_muon=True)

        bx = torch.randint(0, 256, (B, T), device=device)
        by = torch.randint(0, 256, (B, T), device=device)

        mean_ms, tok_s = measure_step(model, optimizers, bx, by)
        if base_time is None:
            base_time = mean_ms
        rel = mean_ms / base_time

        print(f"{T:<12} | {B:<10} | {B * T:<14} | {mean_ms:<16.2f} | {tok_s:<20,.0f} | {rel:<14.2f}x")
        results.append((T, B, mean_ms, tok_s, rel))

        del model, optimizers, bx, by
        torch.cuda.empty_cache()

    return results


def test_batch_size_scaling(device="cuda"):
    print("\n" + "=" * 95)
    print(" 2. BATCH SIZE SCALING (FIXED SEQ LEN T = 1024)")
    print(" Theory: At small B, launch latency dominates. At large B, SM saturation hits throughput peak.")
    print(" Checks for memory cliff / allocator thrashing at large batch sizes.")
    print("=" * 95)
    header = f"{'Batch (B)':<10} | {'Seq Len (T)':<12} | {'Tokens/Step':<14} | {'Step Time (ms)':<16} | {'Throughput (tok/s)':<20} | {'Efficiency':<14}"
    print(header)
    print("-" * 95)

    batches = [8, 16, 32, 64, 96]
    T = 1024
    results = []

    cfg = TorosHybridConfig(
        dim=144,
        n_encoder_layers=6,
        channel_mixer_type="asdag_tree",
        dtype=torch.bfloat16
    )
    model = TorosHybridLanguageModel(cfg).to(device)
    model.enable_lpc(dtype=torch.bfloat16, device=device)
    optimizers = model.get_default_lpc_optimizers(use_muon=True)

    max_tok_s = 0
    for B in batches:
        bx = torch.randint(0, 256, (B, T), device=device)
        by = torch.randint(0, 256, (B, T), device=device)

        mean_ms, tok_s = measure_step(model, optimizers, bx, by)
        max_tok_s = max(max_tok_s, tok_s)
        results.append((B, T, mean_ms, tok_s))

    for B, T, mean_ms, tok_s in results:
        eff = (tok_s / max_tok_s) * 100.0
        print(f"{B:<10} | {T:<12} | {B * T:<14} | {mean_ms:<16.2f} | {tok_s:<20,.0f} | {eff:<14.1f}%")

    del model, optimizers
    torch.cuda.empty_cache()
    return results


def test_dimension_scaling(device="cuda"):
    print("\n" + "=" * 95)
    print(" 3. MODEL DIMENSION SCALING (B = 32, T = 1024, FIXED 6 LAYERS)")
    print(" Theory: Linear GEMMs scale as O(D^2). Routing and normalizations scale as O(D).")
    print(" Detects tile quantization cliffs (e.g. non-power-of-2 dimensions like 144 vs 256).")
    print("=" * 95)
    header = f"{'Hidden Dim (D)':<16} | {'Param Count':<14} | {'Step Time (ms)':<16} | {'Throughput (tok/s)':<20} | {'Exponent (alpha)':<16}"
    print(header)
    print("-" * 95)

    dims = [96, 144, 192, 288]
    B, T = 32, 1024
    results = []

    base_d = dims[0]
    base_ms = None

    for d in dims:
        cfg = TorosHybridConfig(
            dim=d,
            n_encoder_layers=6,
            channel_mixer_type="asdag_tree",
            dtype=torch.bfloat16
        )
        model = TorosHybridLanguageModel(cfg).to(device)
        model.enable_lpc(dtype=torch.bfloat16, device=device)
        optimizers = model.get_default_lpc_optimizers(use_muon=True)

        total_params = sum(p.numel() for p in model.parameters())

        bx = torch.randint(0, 256, (B, T), device=device)
        by = torch.randint(0, 256, (B, T), device=device)

        mean_ms, tok_s = measure_step(model, optimizers, bx, by)

        if base_ms is None:
            base_ms = mean_ms
            alpha = 1.0
        else:
            # T_ms = k * D^alpha => alpha = log(T_ms / base_ms) / log(d / base_d)
            alpha = math.log(mean_ms / base_ms) / math.log(d / base_d)

        print(f"{d:<16} | {total_params:<14,} | {mean_ms:<16.2f} | {tok_s:<20,.0f} | {alpha:<16.2f}")
        results.append((d, total_params, mean_ms, tok_s, alpha))

        del model, optimizers, bx, by
        torch.cuda.empty_cache()

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("Error: CUDA is required for multi-scale GPU profiling.")
        sys.exit(1)

    print("=" * 95)
    print("   TOROS HYBRID / ASDAG MULTI-SCALE BOTTLENECK & SCALING ANALYSIS")
    print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
    print("=" * 95)

    seq_res = test_seq_len_scaling(device)
    batch_res = test_batch_size_scaling(device)
    dim_res = test_dimension_scaling(device)

    print("\n" + "=" * 95)
    print("   SCALING ANALYSIS SUMMARY & ANOMALY DETECTION")
    print("=" * 95)

    # 1. Check Seq Len Anomaly:
    t512_time = seq_res[0][2]
    t4096_time = seq_res[-1][2]
    seq_ratio = t4096_time / t512_time
    print(f"\n[Seq Len Scaling Check (T=512 -> 4096 at constant 32k tokens)]:")
    print(f"  Latency ratio T=4096 / T=512: {seq_ratio:.2f}x")
    if seq_ratio > 1.35:
        print(f"  --> ALERT: Superlinear scaling detected! Latency grew {seq_ratio:.2f}x at constant token budget.")
        print(f"      Investigating intra-chunk quadratic terms or patch allocation bounds.")
    else:
        print(f"  --> HEALTHY: Latency is roughly constant ({seq_ratio:.2f}x), confirming linear attention O(T) behavior.")

    # 2. Check Dimension Anomaly:
    dim_alpha = dim_res[-1][4]
    print(f"\n[Dimension Scaling Check (D=96 -> 288)]:")
    print(f"  Empirical scaling exponent alpha: D^{dim_alpha:.2f}")
    if dim_alpha > 1.8:
        print(f"  --> Heavy compute: Quadratic O(D^2) regime dominated by dense GEMMs.")
    elif dim_alpha < 1.0:
        print(f"  --> Sublinear/Memory-bound: Overhead is dominated by fixed launch and routing costs.")
    else:
        print(f"  --> Typical mixed scaling (O(D^{dim_alpha:.2f})), healthy balance of linear and quadratic components.")


if __name__ == "__main__":
    main()
