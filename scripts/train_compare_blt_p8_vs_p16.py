#!/usr/bin/env python3
"""
Head-to-Head Comparison: 1M TorosHybrid BLT (P=8 vs P=16) on SimpleStories
==========================================================================
- Architecture: 1M-ish TorosHybridLanguageModel (dim=272, layers=8, d_byte=136 [2:1 ratio])
- All default hybrid settings:
    * use_csa = True (CSA2 enabled by default)
    * csa_kv_quant = "int4" (Triton 4-bit packed KV cache)
    * use_tree_sga = True (ASDAG Tree-Guided Sparse Global Attention)
    * csa_every_n = 4, csa_group_size = 4
    * conv_kernel_size = 8
    * channel_mixer_type = "asdag_tree"
- Training: 1,000 steps per model on SimpleStories (B=96, T=512, Muon+AdamW)
- Metrics: Step-by-step Val Loss, Val PPL, Val BPC, Throughput (tok/s), Step latency (ms)
"""

import os
import sys
import time
import math
import numpy as np
import torch

from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig
from affine_ai.data.dataloader import PaddedDataLoader
from affine_ai.training.trainer import ASDAGTrainer


def train_model(patch_size: int, train_bytes: np.ndarray, val_bytes: np.ndarray, device: str):
    print("\n" + "=" * 95)
    print(f"   STARTING RUN: 1M TorosHybrid BLT (P={patch_size}) on SimpleStories (1,000 Steps)")
    print("=" * 95)

    batch_size = 96
    seq_len = 512
    bytes_per_step = batch_size * seq_len
    max_steps = 1000

    # Ensure deterministic data stream across both runs
    torch.manual_seed(42)
    np.random.seed(42)

    train_loader = PaddedDataLoader(
        train_bytes,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        as_stream=True,
    )
    val_loader = PaddedDataLoader(
        val_bytes,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        as_stream=True,
    )

    cfg = TorosHybridConfig(
        dim=272,
        n_encoder_layers=8,
        d_byte=136,  # 2:1 Encoder:Decoder ratio
        target_patch_size=patch_size,
        dtype=torch.bfloat16,
    )
    model = TorosHybridLanguageModel(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model Configuration: dim={cfg.dim}, layers={cfg.n_encoder_layers}, d_byte={cfg.d_byte}, P={patch_size}")
    print(f"Total Parameters   : {total_params:,} ({total_params/1e6:.3f}M)")
    print(f"CSA2 Configuration : use_csa={cfg.use_csa}, csa_kv_quant={cfg.csa_kv_quant}, use_tree_sga={cfg.use_tree_sga}")

    trainer = ASDAGTrainer(
        model=model,
        train_data=train_loader,
        val_data=val_loader,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=1e-3,
        weight_decay=0.01,
        max_steps=max_steps,
        warmup_steps=100,
        eval_interval=200,
        eval_iters=20,
        grad_clip=1.0,
        device=device,
        use_cuda_graph=True,
        use_muon=True,
        muon_lr=0.02,
    )

    eval_history = {}
    init_eval = trainer.evaluate()
    eval_history[0] = init_eval
    print(f" >>> [INITIAL @ Step 0] Val Loss: {init_eval['val_loss']:.4f} | Val PPL: {init_eval['val_ppl']:>7.2f} | Val BPC: {init_eval['val_bpc']:.3f}\n")

    start_time = time.time()
    t_prev = start_time
    total_tokens = 0

    for step in range(1, max_steps + 1):
        step_loss = trainer.train_step(step)
        total_tokens += bytes_per_step

        if step % 100 == 0 or step == 10:
            t_now = time.time()
            elapsed_interval = t_now - t_prev
            steps_in_interval = step if step == 10 else 100
            avg_step_ms = (elapsed_interval / steps_in_interval) * 1000.0
            tok_per_sec = (steps_in_interval * bytes_per_step) / max(1e-6, elapsed_interval)
            total_elapsed = t_now - start_time
            t_prev = t_now

            loss_val = float(step_loss) if not hasattr(step_loss, "item") else float(step_loss.item())
            ppl_val = math.exp(min(loss_val, 20.0))
            bpc_val = loss_val / math.log(2)

            print(
                f"Step {step:5d}/{max_steps} | "
                f"Train Loss: {loss_val:.4f} | "
                f"Train PPL: {ppl_val:>6.2f} | "
                f"{avg_step_ms:>5.1f} ms/step | "
                f"{tok_per_sec:>7,.0f} tok/s | "
                f"Elapsed: {total_elapsed:>5.1f}s",
                flush=True,
            )

        if step % trainer.eval_interval == 0 or step == max_steps:
            ev = trainer.evaluate()
            eval_history[step] = ev
            print(
                f" >>> [EVAL @ Step {step:5d}] "
                f"Val Loss: {ev['val_loss']:.4f} | "
                f"Val PPL: {ev['val_ppl']:>6.2f} | "
                f"Val BPC: {ev['val_bpc']:.3f}",
                flush=True,
            )

    total_time = time.time() - start_time
    avg_speed = total_tokens / max(1e-6, total_time)
    avg_ms = (total_time / max_steps) * 1000.0

    return {
        "patch_size": patch_size,
        "params": total_params,
        "total_time": total_time,
        "avg_speed": avg_speed,
        "avg_ms": avg_ms,
        "eval_history": eval_history,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 95)
    print("      HEAD-TO-HEAD COMPARISON: 1M TOROSHYBRID BLT (P=8 vs P=16) ON SIMPLESTORIES")
    print("=" * 95)

    data_path = "data/simplestories_eos.bin"
    if not os.path.exists(data_path):
        print(f"Error: dataset file '{data_path}' not found!")
        sys.exit(1)

    raw_data = np.memmap(data_path, dtype=np.uint8, mode="r")
    total_bytes = len(raw_data)
    split = int(0.95 * total_bytes)
    train_bytes = raw_data[:split]
    val_bytes = raw_data[split:]

    print(f"Dataset             : SimpleStories ({total_bytes / (1024*1024):.1f} MB, {total_bytes:,} bytes)")
    print(f"Train / Val Split   : Train={len(train_bytes) / (1024*1024):.1f} MB | Val={len(val_bytes) / (1024*1024):.1f} MB")
    print(f"Hardware Target     : {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Run Model A: P=8
    res_p8 = train_model(patch_size=8, train_bytes=train_bytes, val_bytes=val_bytes, device=device)

    # Clean VRAM between runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run Model B: P=16
    res_p16 = train_model(patch_size=16, train_bytes=train_bytes, val_bytes=val_bytes, device=device)

    # Head-to-head summary
    print("\n" + "=" * 95)
    print("                       HEAD-TO-HEAD RESULTS SUMMARY (STEP 1,000)")
    print("=" * 95)
    print(f"{'Metric':<25} | {'BLT (P=8)':<22} | {'BLT (P=16)':<22} | {'Delta (P=8 vs P=16)':<20}")
    print("-" * 95)

    p8_hist = res_p8["eval_history"]
    p16_hist = res_p16["eval_history"]

    eval_steps = [0, 200, 400, 600, 800, 1000]
    for st in eval_steps:
        if st in p8_hist and st in p16_hist:
            p8_loss = p8_hist[st]["val_loss"]
            p8_ppl = p8_hist[st]["val_ppl"]
            p16_loss = p16_hist[st]["val_loss"]
            p16_ppl = p16_hist[st]["val_ppl"]
            ppl_diff = p8_ppl - p16_ppl
            winner = "P=8 is better" if ppl_diff < 0 else "P=16 is better" if ppl_diff > 0 else "Tied"
            print(f"Step {st:<4d} Val Loss       | {p8_loss:<22.4f} | {p16_loss:<22.4f} | {p8_loss - p16_loss:+.4f}")
            print(f"Step {st:<4d} Val PPL        | {p8_ppl:<22.2f} | {p16_ppl:<22.2f} | {ppl_diff:+.2f} ({winner})")

    p8_bpc = p8_hist[1000]["val_bpc"]
    p16_bpc = p16_hist[1000]["val_bpc"]
    print(f"{'Final Val BPC':<25} | {p8_bpc:<22.3f} | {p16_bpc:<22.3f} | {p8_bpc - p16_bpc:+.3f}")
    print(f"{'Throughput (tok/s)':<25} | {res_p8['avg_speed']:<22,.0f} | {res_p16['avg_speed']:<22,.0f} | {res_p16['avg_speed'] / res_p8['avg_speed']:.2f}x faster")
    print(f"{'Step Latency (ms)':<25} | {res_p8['avg_ms']:<22.1f} | {res_p16['avg_ms']:<22.1f} | {res_p8['avg_ms'] - res_p16['avg_ms']:+.1f} ms")
    print(f"{'Total Time (min)':<25} | {res_p8['total_time']/60:<22.2f} | {res_p16['total_time']/60:<22.2f} | {res_p8['total_time']/60 - res_p16['total_time']/60:+.2f} min")
    print("=" * 95)


if __name__ == "__main__":
    main()
