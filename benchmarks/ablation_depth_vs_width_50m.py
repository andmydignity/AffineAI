#!/usr/bin/env python3
"""
Depth vs. Width Iso-Parameter Ablation on SimpleStories (~50M Parameters, 300 Steps)
=====================================================================================
5 iso-parameter candidates (all ~48-51M params, ternary_swiglu channel mixer):
  1. Deep-L20      D=640,  L=20, H=10  -- 49.8M
  2. Deep-L16      D=704,  L=16, H=11  -- 48.2M
  3. Balanced-L12  D=832,  L=12, H=13  -- 50.5M
  4. Wide-L8       D=1024, L=8,  H=16  -- 50.9M
  5. Shallow-L6    D=1152, L=6,  H=18  -- 48.3M

Note: L=32/24 diverge under LPC (too many independent Muon optimizer groups).
LPC is stable up to ~L=20 at muon_lr=2e-3.

LR: muon_lr=2e-3, adamw_lr=3e-4, 30-step linear warmup + cosine annealing.
NaN safety: fp32 cross_entropy in LPC heads (fixed in lpc.py).
"""

import os, sys, math, time, gc
import numpy as np
import torch

from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig
from affine_ai.data.dataloader import PaddedDataLoader


def get_lr(step, warmup=30, base_lr=2e-3, min_lr=1e-4, total_steps=300):
    if step < warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def set_lr(optimizers, lr):
    for opt in optimizers:
        for pg in opt.param_groups:
            pg["lr"] = lr


def evaluate_val_ppl(model, val_loader, max_batches=20):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, (bx, by) in enumerate(val_loader):
            if i >= max_batches:
                break
            try:
                logits, loss, _ = model(bx, targets=by, return_logits=False)
                valid_mask = (by != -100)
                n_tokens = valid_mask.sum().item()
                if n_tokens > 0 and isinstance(loss, torch.Tensor) and not loss.isnan():
                    total_loss += loss.item() * n_tokens
                    total_tokens += n_tokens
            except Exception:
                pass
    model.train()
    if total_tokens == 0:
        return float("inf"), float("inf")
    mean_loss = total_loss / total_tokens
    return mean_loss, math.exp(min(mean_loss, 20.0))


def run_ablation():
    if not torch.cuda.is_available():
        print("Error: CUDA required."); sys.exit(1)

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    data_path = "data/simplestories_eos.bin"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found."); sys.exit(1)

    print("=" * 115)
    print("  ~50M PARAMETER DEPTH vs. WIDTH ABLATION ON SIMPLESTORIES (300 STEPS)")
    print(f"  GPU: {gpu_name}  |  Data: {data_path}")
    print(f"  LR: muon=2e-3, adamw=3e-4, 30-step warmup + cosine annealing")
    print("=" * 115)
    sys.stdout.flush()

    raw_data = np.memmap(data_path, dtype=np.uint8, mode="r")
    val_size = 500_000
    train_data = raw_data[:-val_size]
    val_data   = raw_data[-val_size:]

    batch_size     = 16
    seq_len        = 512
    total_steps    = 300
    tokens_per_step = batch_size * seq_len
    muon_base_lr   = 2e-3
    adamw_base_lr  = 3e-4
    grad_clip      = 1.0

    val_loader = PaddedDataLoader(val_data, batch_size=batch_size, seq_len=seq_len, device=device)

    candidates = [
        {"name": "Deep-L20",     "dim": 640,  "layers": 20, "heads": 10},
        {"name": "Deep-L16",     "dim": 704,  "layers": 16, "heads": 11},
        {"name": "Balanced-L12", "dim": 832,  "layers": 12, "heads": 13},
        {"name": "Wide-L8",      "dim": 1024, "layers": 8,  "heads": 16},
        {"name": "Shallow-L6",   "dim": 1152, "layers": 6,  "heads": 18},
    ]

    results = []

    for c in candidates:
        name, dim, layers, heads = c["name"], c["dim"], c["layers"], c["heads"]

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        torch.manual_seed(42)
        np.random.seed(42)
        train_loader = PaddedDataLoader(train_data, batch_size=batch_size, seq_len=seq_len, device=device)
        train_iter = iter(train_loader)

        cfg = TorosHybridConfig(
            dim=dim, n_heads=heads, n_encoder_layers=layers,
            channel_mixer_type="ternary_swiglu", dtype=torch.bfloat16,
        )
        model = TorosHybridLanguageModel(cfg).to(device)
        model.enable_lpc(dtype=torch.bfloat16, device=device)
        optimizers = model.get_default_lpc_optimizers(
            lr=adamw_base_lr, muon_lr=muon_base_lr, use_muon=True,
        )
        params_m = sum(p.numel() for p in model.parameters()) / 1e6

        print(f"\n{'='*80}")
        print(f"  [{name}]  Dim={dim}  L={layers}  H={heads}  ({params_m:.2f}M params)")
        print(f"{'='*80}")
        sys.stdout.flush()

        # Warmup (not timed, no CUDA graph)
        for s in range(3):
            bx, by = next(train_iter)
            set_lr(optimizers, get_lr(s, base_lr=muon_base_lr, total_steps=total_steps))
            model.forward_lpc_step(bx, by, optimizers, grad_clip=grad_clip,
                                   sync_loss=False, use_cuda_graph=False, use_async_pipelining=False)
        torch.cuda.synchronize()

        step_times   = []
        train_losses = []
        nan_count    = 0

        t_start = time.perf_counter()
        for step in range(1, total_steps + 1):
            bx, by = next(train_iter)
            set_lr(optimizers, get_lr(step, base_lr=muon_base_lr, total_steps=total_steps))
            sync_now = (step % 50 == 0 or step == total_steps)

            t0 = time.perf_counter()
            res = model.forward_lpc_step(bx, by, optimizers, grad_clip=grad_clip,
                                         sync_loss=sync_now, use_cuda_graph=False, use_async_pipelining=False)
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)

            if sync_now:
                cur_loss = res["loss"]
                if isinstance(cur_loss, torch.Tensor):
                    cur_loss = cur_loss.item()
                is_bad = math.isnan(cur_loss) or math.isinf(cur_loss)
                if is_bad:
                    nan_count += 1
                    train_losses.append(float("nan"))
                    status = " [NaN!]"
                else:
                    train_losses.append(cur_loss)
                    status = ""
                ppl_str  = f"{math.exp(min(cur_loss, 20.0)):.2f}" if not is_bad else "NaN"
                elapsed  = time.perf_counter() - t_start
                tok_s_now = (step * tokens_per_step) / elapsed
                print(f"  Step {step:3d}/300 | Loss: {cur_loss:.4f} | PPL: {ppl_str} | "
                      f"{step_times[-1]*1000:.1f} ms | {tok_s_now:,.0f} tok/s{status}")
                sys.stdout.flush()

        avg_ms    = (sum(step_times) / len(step_times)) * 1000.0
        p95_ms    = float(np.percentile(step_times, 95)) * 1000.0
        tok_s     = tokens_per_step / (avg_ms / 1000.0)
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

        print(f"\n  Evaluating val PPL (20 batches)...")
        sys.stdout.flush()
        val_loss, val_ppl = evaluate_val_ppl(model, val_loader, max_batches=20)

        valid_losses = [l for l in train_losses if not (isinstance(l, float) and math.isnan(l))]
        final_train_loss = valid_losses[-1] if valid_losses else float("nan")
        final_train_ppl  = (math.exp(min(final_train_loss, 20.0))
                            if not math.isnan(final_train_loss) else float("nan"))

        print(f"  Train PPL@300: {final_train_ppl:.2f} | Val PPL: {val_ppl:.2f} | "
              f"Avg: {avg_ms:.1f} ms | p95: {p95_ms:.1f} ms | {tok_s:,.0f} tok/s | "
              f"{peak_vram:.0f} MB VRAM | NaN steps: {nan_count}")
        sys.stdout.flush()

        results.append(dict(
            name=name, dim=dim, layers=layers, params_m=params_m,
            avg_ms=avg_ms, p95_ms=p95_ms, tok_s=tok_s, vram_mb=peak_vram,
            train_ppl=final_train_ppl, val_ppl=val_ppl, nan_count=nan_count,
        ))

        del model, optimizers, train_loader
        gc.collect()
        torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 130)
    print("        ~50M PARAM DEPTH vs. WIDTH ABLATION -- FINAL SUMMARY  (SimpleStories, 300 steps, muon_lr=2e-3)")
    print("=" * 130)
    print(f"  {'Config':<15} | {'Dim x L':<9} | {'Params':>8} | {'AvgMS':>8} | {'p95MS':>8} | {'Tok/s':>12} | {'VRAM':>7} | {'TrainPPL':>10} | {'ValPPL':>9}")
    print("-" * 130)
    valid_ppls = [r["val_ppl"] for r in results if not math.isnan(r["val_ppl"])]
    best_val   = min(valid_ppls) if valid_ppls else None
    best_spd   = max(r["tok_s"] for r in results)
    for r in results:
        pm  = " *" if (best_val and not math.isnan(r["val_ppl"]) and r["val_ppl"] == best_val) else ""
        sm  = " ^" if r["tok_s"] == best_spd else ""
        cfg = f"{r['dim']}x{r['layers']}"
        vs  = f"{r['val_ppl']:.2f}"   if not math.isnan(r["val_ppl"])   else "NaN"
        ts  = f"{r['train_ppl']:.2f}" if not math.isnan(r["train_ppl"]) else "NaN"
        print(f"  {r['name']:<15} | {cfg:<9} | {r['params_m']:>6.2f}M  | "
              f"{r['avg_ms']:>6.1f}ms | {r['p95_ms']:>6.1f}ms | {r['tok_s']:>11,.0f}  | "
              f"{r['vram_mb']:>5.0f}MB | {ts:>10} | {vs:>9}{pm}{sm}")
    print("=" * 130)
    print("  * = Best Val PPL   ^ = Highest Throughput")
    print()
    sys.stdout.flush()


if __name__ == "__main__":
    run_ablation()
