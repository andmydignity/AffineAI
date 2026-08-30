#!/usr/bin/env python3
"""
ASDAG Execution Showdown: GPU (RTX 3050) vs CPU (C++ AVX2 Native Engine)
========================================================================
Measures latency (ms) and throughput (tok/s) across exact batch sizes:
  512, 2048, 8192, 32768 tokens (Dim=96, Leaves=16)
"""

import subprocess
import time
import torch
import torch.nn.functional as F

from affine_ai.core.ast_dag import ASTDAGLayer


def benchmark_gpu(tokens_list, dim=96, leaves=16, iters=30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = ASTDAGLayer(dim=dim, initial_branches=leaves, rank=None).to(device).eval()

    gpu_results = {}
    for N in tokens_list:
        x = torch.randn(N, dim, device=device)
        # Warmup
        for _ in range(5): _ = layer.forward_batched_dispatch(x)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(iters):
            _ = layer.forward_batched_dispatch(x)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        tok_s = N / dt
        gpu_results[N] = {"ms": dt * 1000.0, "tok_s": tok_s}

    return gpu_results


def run_showdown():
    tokens_list = [512, 2048, 8192, 32768]
    print("Measuring GPU (RTX 3050)...")
    gpu_res = benchmark_gpu(tokens_list)

    print("Measuring CPU (C++ AVX2 Engine)...")
    cmd = ["./cpp_infer/asdag_bench"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Parse C++ output
    # Format:
    # --- Tokens:    512 (Dim=96) ---
    #   Dense FP32 MLP (C++ Multi-threaded):   7.58 ms |      67562 tok/s
    #   ASDAG Native C++ SIMD Engine       :   3.50 ms |     146112 tok/s  [ 2.16x Faster! ]
    cpu_res = {}
    lines = res.stdout.split("\n")
    current_tokens = None
    for line in lines:
        if "--- Tokens:" in line:
            current_tokens = int(line.split(":")[1].split("(")[0].strip())
        elif "ASDAG Native C++ SIMD Engine" in line and current_tokens is not None:
            parts = line.split(":")
            subparts = parts[1].split("|")
            ms = float(subparts[0].replace("ms", "").strip())
            tok_s = float(subparts[1].split("[")[0].replace("tok/s", "").strip())
            cpu_res[current_tokens] = {"ms": ms, "tok_s": tok_s}

    print("\n" + "=" * 92)
    print("                 ASDAG ARCHITECTURE: GPU (RTX 3050) vs CPU (Native C++ AVX2)")
    print("=" * 92)
    print(f"{'Batch Size':<14} | {'GPU (RTX 3050) Latency':<24} | {'CPU (C++ AVX2) Latency':<24} | {'Speed Winner':<18}")
    print("-" * 92)

    for N in tokens_list:
        g_ms = gpu_res[N]["ms"]
        g_tok = gpu_res[N]["tok_s"]
        c_ms = cpu_res[N]["ms"]
        c_tok = cpu_res[N]["tok_s"]

        if c_tok > g_tok:
            winner = f"CPU ({c_tok/g_tok:.2f}x faster)"
        else:
            winner = f"GPU ({g_tok/c_tok:.2f}x faster)"

        print(f"{N:<6} tokens   | {g_ms:>6.2f} ms ({g_tok:>9,.0f} tok/s) | {c_ms:>6.2f} ms ({c_tok:>9,.0f} tok/s) | {winner:<18}")

    print("=" * 92)


if __name__ == "__main__":
    run_showdown()
