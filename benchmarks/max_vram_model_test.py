"""
Empirical Maximum Model Size Search on Host VRAM
================================================
Searches for the largest possible TorosHybrid model (parameter count and dimensions)
that can execute full Forward + Backward + Optimizer steps on this host GPU
without Out-of-Memory (OOM).
"""

import sys
import gc
import torch
import numpy as np
from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig


def format_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def test_config(dim: int, layers: int, batch_size: int = 16, seq_len: int = 512, device: str = "cuda"):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    n_heads = max(4, dim // 64)
    while dim % n_heads != 0:
        n_heads -= 1

    cfg = TorosHybridConfig(
        dim=dim,
        n_heads=n_heads,
        n_encoder_layers=layers,
        channel_mixer_type="ternary_swiglu",
        dtype=torch.bfloat16
    )

    model = TorosHybridLanguageModel(cfg).to(device)
    model.enable_lpc(dtype=torch.bfloat16, device=device)
    optimizers = model.get_default_lpc_optimizers(use_muon=True)

    param_count = sum(p.numel() for p in model.parameters())

    bx = torch.randint(0, 256, (batch_size, seq_len), device=device)
    by = torch.randint(0, 256, (batch_size, seq_len), device=device)

    # Warmup step
    step_res = model.forward_lpc_step(bx, by, optimizers, use_async_pipelining=True, sync_loss=False)

    # Benchmark 3 steps
    latencies = []
    e_start = torch.cuda.Event(enable_timing=True)
    e_end = torch.cuda.Event(enable_timing=True)

    for step in range(3):
        e_start.record()
        step_res = model.forward_lpc_step(bx, by, optimizers, use_async_pipelining=True, sync_loss=False)
        e_end.record()
        torch.cuda.synchronize()
        latencies.append(e_start.elapsed_time(e_end))

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    step_time_ms = float(np.median(latencies))
    tok_s = (batch_size * seq_len) / (step_time_ms / 1000.0)

    del model, optimizers, bx, by, step_res
    gc.collect()
    torch.cuda.empty_cache()

    return param_count, peak_vram_mb, step_time_ms, tok_s


def main():
    if not torch.cuda.is_available():
        print("Error: CUDA is required.")
        sys.exit(1)

    device_props = torch.cuda.get_device_properties(0)
    total_vram_mb = device_props.total_memory / (1024 ** 2)

    print("=" * 110)
    print(f"   MAXIMUM MODEL SIZE SEARCH (LOCAL GPU: {device_props.name})")
    print(f"   Usable VRAM: {total_vram_mb:.1f} MiB | Precision: BF16 | Method: Layerwise LPC + HybridMuonAdamW")
    print(f"   Channel Mixer: Ternary BitLinear SwiGLU (Expand=2x) | Time Mixer: Monarch GLA")
    print(f"   Batch Size: 16 | Sequence Length: 512 (8,192 tokens/step)")
    print("=" * 110)

    # Systematic scaling sweep from 25M to 800M+
    candidates = [
        (512, 16),   # ~25M
        (576, 24),   # ~50M
        (768, 24),   # ~90M
        (1024, 24),  # ~155M
        (1280, 24),  # ~240M
        (1536, 24),  # ~345M
        (1792, 24),  # ~470M
        (1920, 24),  # ~540M
        (2048, 24),  # ~615M
        (2176, 24),  # ~695M
        (2304, 24),  # ~780M
        (2432, 24),  # ~870M
    ]

    header = f"{'Config (D x L)':<16} | {'Params':<12} | {'Peak VRAM (MiB)':<20} | {'VRAM %':<10} | {'Step Time (ms)':<16} | {'Speed (tok/s)':<16} | {'Status':<10}"
    print(header)
    print("-" * 110)

    largest_successful = None

    for dim, layers in candidates:
        name = f"D={dim}, L={layers}"
        try:
            params, vram, step_ms, tok_s = test_config(dim, layers)
            vram_pct = (vram / total_vram_mb) * 100.0
            print(f"{name:<16} | {format_params(params):<12} | {vram:>10.1f} MiB ({vram_pct:>4.1f}%) | {vram_pct:>5.1f}%    | {step_ms:>10.2f} ms   | {tok_s:>10,.0f} tok/s | SUCCESS")
            largest_successful = (name, params, vram, step_ms, tok_s)
        except torch.cuda.OutOfMemoryError:
            print(f"{name:<16} | {'---':<12} | {'> ' + str(int(total_vram_mb)) + ' MiB':<20} | {'>100%':<10} | {'---':<16} | {'---':<16} | OOM")
            break
        except Exception as e:
            print(f"{name:<16} | ERROR: {e}")
            break

    print("-" * 110)
    if largest_successful:
        cfg_name, p_count, p_vram, s_time, s_spd = largest_successful
        print(f"\nLARGEST MODEL TRAINABLE ON THIS {total_vram_mb:.0f} MiB GPU:")
        print(f"  Architecture   : {cfg_name} (Ternary SwiGLU + Monarch GLA)")
        print(f"  Parameter Count: {p_count:,} ({format_params(p_count)})")
        print(f"  Peak VRAM Used : {p_vram:.1f} MiB ({(p_vram / total_vram_mb) * 100.0:.1f}% of total VRAM)")
        print(f"  Step Latency   : {s_time:.2f} ms")
        print(f"  Training Speed : {s_spd:,.0f} tokens/second")
    print("=" * 110)


if __name__ == "__main__":
    main()
