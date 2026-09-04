import os
import sys
import time
import torch
import torch.nn as nn
from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig

def benchmark_hybrid(
    dim=136,
    d_byte=64,
    n_layers=4,
    n_heads=4,
    patch_size=16,
    batch_sizes=[16, 32, 64, 96, 128],
    seq_len=512,
    warmup=5,
    steps=20,
    dtype=torch.float32,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA not available! Aborting GPU benchmark.", flush=True)
        return

    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 105, flush=True)
    print(f"  TOROS-HYBRID SPEED BENCHMARK: LPC BLT 1:16 Top-2 Tree on {gpu_name}", flush=True)
    print("=" * 105, flush=True)
    print(f"Config: dim={dim}, d_byte={d_byte}, layers={n_layers}, heads={n_heads}, patch_size={patch_size}", flush=True)
    print(f"Mixers: Time=Monarch GLA, Channel=ASDAG Tree (1:16 structured sparsity, Top-2 Sign Routing)", flush=True)
    print(f"Precision: {dtype} | Context length: T={seq_len} bytes", flush=True)
    print("-" * 105, flush=True)

    config = TorosHybridConfig(
        dim=dim,
        d_byte=d_byte,
        n_encoder_layers=n_layers,
        n_heads=n_heads,
        target_patch_size=patch_size,
        channel_mixer_type="asdag_tree",
        time_mixer_rule="gla",
        dtype=dtype,
    )

    model = TorosHybridLanguageModel(config).to(device)
    model.enable_lpc(device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,d}", flush=True)
    print("-" * 105, flush=True)

    # 1. Benchmark Inference / Forward-only
    print(f"\n--- [1] FORWARD EVALUATION / INFERENCE (torch.no_grad) ---", flush=True)
    print(f"{'Batch':<8} | {'Tokens/Step':<12} | {'Step Time':<14} | {'Throughput':<18} | {'Peak VRAM':<12}", flush=True)
    print("-" * 72, flush=True)
    model.eval()

    with torch.no_grad():
        for B in batch_sizes:
            x = torch.randint(0, 256, (B, seq_len), device=device)
            # Warmup
            for _ in range(warmup):
                _ = model(x)
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(steps):
                _ = model(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            dt = (t1 - t0) / steps
            total_tok = B * seq_len
            tok_per_sec = total_tok / dt
            vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"B={B:<6} | {total_tok:<12,d} | {dt * 1000:8.2f} ms     | {tok_per_sec:12,.0f} tok/s  | {vram:8.1f} MB", flush=True)

    # 2. Benchmark LPC Training (Forward-Only Per-Layer Local Updates)
    print(f"\n--- [2] LPC TRAINING STEP (Forward-Only Local Plasticity, 0 Cross-Layer Graph) ---", flush=True)
    print(f"{'Batch':<8} | {'Tokens/Step':<12} | {'Step Time':<14} | {'Throughput':<18} | {'Peak VRAM':<12}", flush=True)
    print("-" * 72, flush=True)
    model.train()
    opts_lpc = model.get_default_lpc_optimizers(lr=1e-3, use_muon=False)

    for B in batch_sizes:
        x = torch.randint(0, 256, (B, seq_len), device=device)
        y = torch.randint(0, 256, (B, seq_len), device=device)

        try:
            # Warmup
            for _ in range(warmup):
                _ = model.forward_lpc_step(x, y, opts_lpc, sync_loss=False)
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(steps):
                _ = model.forward_lpc_step(x, y, opts_lpc, sync_loss=False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            dt = (t1 - t0) / steps
            total_tok = B * seq_len
            tok_per_sec = total_tok / dt
            vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"B={B:<6} | {total_tok:<12,d} | {dt * 1000:8.2f} ms     | {tok_per_sec:12,.0f} tok/s  | {vram:8.1f} MB", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"B={B:<6} | OOM in LPC training!", flush=True)
            torch.cuda.empty_cache()

    # 3. Benchmark Standard Full-Graph Autograd Backprop
    print(f"\n--- [3] STANDARD BACKPROP TRAINING (Full End-to-End Autograd Backward) ---", flush=True)
    print(f"{'Batch':<8} | {'Tokens/Step':<12} | {'Step Time':<14} | {'Throughput':<18} | {'Peak VRAM':<12}", flush=True)
    print("-" * 72, flush=True)
    opt_standard = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for B in batch_sizes:
        x = torch.randint(0, 256, (B, seq_len), device=device)
        y = torch.randint(0, 256, (B, seq_len), device=device)

        try:
            # Warmup
            for _ in range(warmup):
                opt_standard.zero_grad(set_to_none=True)
                _, loss, _ = model(x, targets=y)
                loss.backward()
                opt_standard.step()
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(steps):
                opt_standard.zero_grad(set_to_none=True)
                _, loss, _ = model(x, targets=y)
                loss.backward()
                opt_standard.step()
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            dt = (t1 - t0) / steps
            total_tok = B * seq_len
            tok_per_sec = total_tok / dt
            vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"B={B:<6} | {total_tok:<12,d} | {dt * 1000:8.2f} ms     | {tok_per_sec:12,.0f} tok/s  | {vram:8.1f} MB", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"B={B:<6} | OOM during standard backprop backward pass!", flush=True)
            torch.cuda.empty_cache()

    print("=" * 105, flush=True)

if __name__ == "__main__":
    benchmark_hybrid()
