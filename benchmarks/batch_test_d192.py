"""
Batch Size Scaling Benchmark over B=160 at D=192
================================================
Evaluates throughput, latency, and VRAM scaling for TorosHybrid
at D=192 with batch sizes B in [32, 64, 96, 128, 160, 192, 224, 256].
"""

import sys
import os
import time
import math
import torch
import numpy as np

from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig


def run_batch_sweep():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("Error: CUDA required.")
        sys.exit(1)

    print("=" * 105)
    print("   BATCH SCALING BENCHMARK AT D=192 (GOING OVER B=160)")
    print(f"   GPU: {torch.cuda.get_device_name(0)} | Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.0f} MiB")
    print(f"   Model Config: TorosHybrid (dim=192, layers=6, asdag_tree, BF16)")
    print("=" * 105)

    batches = [32, 64, 96, 128, 160, 192, 224, 256]
    seq_len = 1024

    cfg = TorosHybridConfig(
        dim=192,
        n_encoder_layers=6,
        channel_mixer_type="asdag_tree",
        dtype=torch.bfloat16
    )
    model = TorosHybridLanguageModel(cfg).to(device)
    model.enable_lpc(dtype=torch.bfloat16, device=device)
    optimizers = model.get_default_lpc_optimizers(use_muon=True)

    header = f"{'Batch (B)':<10} | {'Seq Len':<8} | {'Tokens/Step':<14} | {'Step Time (ms)':<16} | {'Throughput (tok/s)':<20} | {'Peak VRAM':<12} | {'Efficiency':<10}"
    print(header)
    print("-" * 105)

    results = []
    max_tok_s = 0

    for B in batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            bx = torch.randint(0, 256, (B, seq_len), device=device)
            by = torch.randint(0, 256, (B, seq_len), device=device)

            # Warmup
            for _ in range(4):
                model.forward_lpc_step(bx, by, optimizers, use_async_pipelining=True, sync_loss=False)
            torch.cuda.synchronize()

            # Timed iterations
            latencies = []
            e_start = torch.cuda.Event(enable_timing=True)
            e_end = torch.cuda.Event(enable_timing=True)

            iters = 10
            for _ in range(iters):
                e_start.record()
                step_res = model.forward_lpc_step(bx, by, optimizers, use_async_pipelining=True, sync_loss=False)
                e_end.record()
                torch.cuda.synchronize()
                latencies.append(e_start.elapsed_time(e_end))

            mean_ms = float(np.median(latencies))
            total_tokens = B * seq_len
            tok_s = total_tokens / (mean_ms / 1000.0)
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

            max_tok_s = max(max_tok_s, tok_s)
            results.append((B, seq_len, total_tokens, mean_ms, tok_s, peak_vram_mb))

            del bx, by
        except torch.cuda.OutOfMemoryError:
            print(f"{B:<10} | {seq_len:<8} | {B * seq_len:<14} | OOM EXCEEDED    | N/A                  | >4000 MiB    | N/A")
            break
        except Exception as e:
            print(f"{B:<10} | ERROR: {e}")
            break

    print("-" * 105)
    print("\nFINAL SCALING SUMMARY (RELATIVE TO PEAK):")
    for B, S, tok_count, mean_ms, tok_s, peak_vram in results:
        eff = (tok_s / max_tok_s) * 100.0
        print(f"  B = {B:<3} ({tok_count:>7,} tok/step): {mean_ms:>7.2f} ms | {tok_s:>8,.0f} tok/s | Peak VRAM: {peak_vram:>6.1f} MiB | {eff:>5.1f}% of peak")

    print("\n" + "=" * 105)


if __name__ == "__main__":
    run_batch_sweep()
