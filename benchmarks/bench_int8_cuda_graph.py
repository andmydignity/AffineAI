"""
Benchmark: Ampere Hardware INT8 IMMA Tensor Cores + Zero-Overhead CUDA Graph Runner
===================================================================================
Compares:
1. LPC Training Step - Eager PyTorch (with INT8 IMMA Tensor Cores)
2. LPC Training Step - CUDA Graph Replay (Zero CPU overhead + INT8 IMMA)
"""

import time
import torch
from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig


def benchmark():
    if not torch.cuda.is_available():
        print("CUDA not available. Aborting.")
        return

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 90)
    print(f"  BENCHMARK: Hardware INT8 IMMA Tensor Cores + CUDA Graph Runner on {gpu_name}")
    print("=" * 90)

    dim = 128
    d_byte = 64
    n_layers = 4
    patch_size = 16
    seq_len = 512
    batch_size = 32
    warmup_steps = 10
    bench_steps = 30

    config = TorosHybridConfig(
        context_dim=dim,
        byte_dim=d_byte,
        num_layers=n_layers,
        target_patch_size=patch_size,
        vocab_size=256,
        num_experts=1,
    )

    print(f"Model: {dim=}, {d_byte=}, {n_layers=}, {patch_size=}, {seq_len=}, {batch_size=}")
    tokens_per_step = batch_size * seq_len

    # --- Mode 1: Eager Training ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_eager = TorosHybridLanguageModel(config).to(device).bfloat16()
    opts_eager = model_eager.get_default_lpc_optimizers(use_muon=True, capturable=False)

    bx = torch.randint(0, 256, (batch_size, seq_len), device=device)
    by = torch.randint(0, 256, (batch_size, seq_len), device=device)

    # Warmup eager
    for _ in range(warmup_steps):
        _ = model_eager.forward_lpc_step(bx, by, opts_eager, use_async_pipelining=False, sync_loss=False)
    torch.cuda.synchronize()

    start_eager = time.perf_counter()
    for _ in range(bench_steps):
        _ = model_eager.forward_lpc_step(bx, by, opts_eager, use_async_pipelining=False, sync_loss=False)
    torch.cuda.synchronize()
    time_eager = (time.perf_counter() - start_eager) / bench_steps
    tok_per_sec_eager = tokens_per_step / time_eager
    vram_eager = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # --- Mode 2: CUDA Graph Replay ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_graph = TorosHybridLanguageModel(config).to(device).bfloat16()
    opts_graph = model_graph.get_default_lpc_optimizers(use_muon=True, capturable=True)

    # Capture graph
    runner = model_graph.capture_lpc_graph(bx, by, opts_graph, warmup_iters=warmup_steps)
    torch.cuda.synchronize()

    start_graph = time.perf_counter()
    for _ in range(bench_steps):
        _ = runner.step(bx, by)
    torch.cuda.synchronize()
    time_graph = (time.perf_counter() - start_graph) / bench_steps
    tok_per_sec_graph = tokens_per_step / time_graph
    vram_graph = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Results
    speedup = time_eager / time_graph

    print("\n" + "-" * 90)
    print(f"{'Mode':<30} | {'Latency (ms)':<15} | {'Throughput (tok/s)':<22} | {'Peak VRAM':<12}")
    print("-" * 90)
    print(f"{'1. INT8 IMMA (Eager)':<30} | {time_eager * 1000:<15.2f} | {tok_per_sec_eager:<22,.1f} | {vram_eager:<12.1f} MB")
    print(f"{'2. INT8 IMMA + CUDA Graph':<30} | {time_graph * 1000:<15.2f} | {tok_per_sec_graph:<22,.1f} | {vram_graph:<12.1f} MB")
    print("-" * 90)
    print(f"CUDA Graph Speedup: {speedup:.2f}x ({speedup * 100 - 100:+.1f}% throughput gain, 0 launch overhead)\n")


if __name__ == "__main__":
    benchmark()
