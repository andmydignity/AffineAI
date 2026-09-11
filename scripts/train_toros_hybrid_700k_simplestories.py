#!/usr/bin/env python3
"""
Train 700k TorosHybrid ASDAG Model on SimpleStories (1,000 steps)
================================================================
- Architecture: TorosHybridLanguageModel (dim=288, n_encoder_layers=7 -> ~705k params)
- Mixer: Adaptive Sparse Tree DAG (channel_mixer_type="asdag_tree", DeepSeek Expert Bias)
- Precision: BFloat16
- Data: data/simplestories_eos.bin (SimpleStories, 2.14 GB)
- Settings: All defaults (B=32, T=512, Muon+AdamW, CUDA Graphs)
- Metrics: Throughput (tok/s, ms/step), Train Loss/PPL/BPC, Validation Loss/PPL/BPC
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
from affine_ai.core.format import save_toros_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 95)
    print("      TRAINING 700k TOROSHYBRID ASDAG ON SIMPLESTORIES (1,000 STEPS, ALL DEFAULTS)")
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

    batch_size = 32
    seq_len = 512
    bytes_per_step = batch_size * seq_len
    max_steps = 1000

    print(f"Dataset             : SimpleStories ({total_bytes / (1024*1024):.1f} MB, {total_bytes:,} bytes)")
    print(f"Train / Val Split   : Train={len(train_bytes) / (1024*1024):.1f} MB | Val={len(val_bytes) / (1024*1024):.1f} MB")
    print(f"Batch Configuration : batch_size={batch_size}, seq_len={seq_len} ({bytes_per_step:,} bytes/tokens per step)")
    print(f"Hardware Target     : {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

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
        dim=288,
        n_encoder_layers=7,
        channel_mixer_type="asdag_tree",
        dtype=torch.bfloat16,
    )
    model = TorosHybridLanguageModel(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model Architecture  : TorosHybrid (dim={cfg.dim}, layers={cfg.n_encoder_layers})")
    print(f"Total Parameters    : {total_params:,d} parameters (~705k)")
    print(f"Channel Mixer       : {cfg.channel_mixer_type} (DeepSeek Expert Bias Load-Balancing)")
    print(f"Execution Engine    : CUDA Graphs = True | Muon Optimizer = True")
    print("-" * 95)

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
        eval_interval=100,
        eval_iters=20,
        grad_clip=1.0,
        device=device,
        use_cuda_graph=True,
        use_muon=True,
        muon_lr=0.02,
    )

    print("\nEvaluating Initial Pre-Training Zero-Shot Baseline...")
    init_eval = trainer.evaluate()
    print(f" >>> [INITIAL EVAL] Val Loss: {init_eval['val_loss']:.4f} | Val PPL: {init_eval['val_ppl']:>7.2f} (BPC: {init_eval['val_bpc']:.3f})\n")

    print(f"Starting {max_steps:,} Step Training Loop...\n")
    start_time = time.time()
    t_prev = start_time
    total_tokens_processed = 0

    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    for step in range(1, max_steps + 1):
        step_loss = trainer.train_step(step)
        total_tokens_processed += bytes_per_step

        if step % 50 == 0 or step == 1 or step == 10:
            t_now = time.time()
            elapsed_interval = t_now - t_prev
            steps_in_interval = step if step <= 10 else 50
            avg_step_ms = (elapsed_interval / steps_in_interval) * 1000.0
            tok_per_sec = (steps_in_interval * bytes_per_step) / max(1e-6, elapsed_interval)
            total_elapsed = t_now - start_time
            t_prev = t_now

            loss_val = float(step_loss) if not hasattr(step_loss, "item") else float(step_loss.item())
            ppl_val = math.exp(min(loss_val, 20.0))
            bpc_val = loss_val / math.log(2)

            print(
                f"Step {step:4d}/{max_steps} | "
                f"Loss: {loss_val:.4f} | "
                f"PPL: {ppl_val:>6.2f} | "
                f"BPC: {bpc_val:.3f} | "
                f"{avg_step_ms:>5.1f} ms/step | "
                f"{tok_per_sec:>7,.0f} tok/s | "
                f"Elapsed: {total_elapsed:>5.1f}s",
                flush=True
            )

        if step % trainer.eval_interval == 0 or step == max_steps:
            eval_metrics = trainer.evaluate()
            v_loss = eval_metrics["val_loss"]
            v_ppl = eval_metrics["val_ppl"]
            v_bpc = eval_metrics["val_bpc"]
            print(
                f" >>> [EVAL @ Step {step:4d}] "
                f"Val Loss: {v_loss:.4f} | "
                f"Val PPL: {v_ppl:>6.2f} | "
                f"Val BPC: {v_bpc:.3f}",
                flush=True
            )

    total_training_time = time.time() - start_time
    avg_speed_tok_s = total_tokens_processed / max(1e-6, total_training_time)
    avg_speed_step_ms = (total_training_time / max_steps) * 1000.0

    print("\n" + "=" * 95)
    print("      TRAINING COMPLETE - SUMMARY RESULTS")
    print("=" * 95)
    final_eval = trainer.evaluate()
    print(f"Total Steps Completed : {max_steps:,}")
    print(f"Total Tokens Trained  : {total_tokens_processed:,} ({total_tokens_processed/(1024*1024):.2f} M tokens)")
    print(f"Total Training Time   : {total_training_time:.2f} s ({total_training_time/60:.2f} min)")
    print(f"Average Throughput    : {avg_speed_tok_s:,.0f} tok/s ({avg_speed_step_ms:.1f} ms/step)")
    print(f"Initial Validation PPL: {init_eval['val_ppl']:.2f} (Loss: {init_eval['val_loss']:.4f})")
    print(f"Final Validation Loss : {final_eval['val_loss']:.4f}")
    print(f"Final Validation PPL  : {final_eval['val_ppl']:.2f}")
    print(f"Final Validation BPC  : {final_eval['val_bpc']:.3f}")
    print("-" * 95)

    save_pt = "models/toros_hybrid_700k_simplestories.pt"
    save_toros = "models/toros_hybrid_700k_simplestories.toros"
    torch.save(model.state_dict(), save_pt)
    print(f"Saved checkpoint to: {save_pt}")
    try:
        save_toros_model(model, save_toros)
        print(f"Saved compact toros artifact to: {save_toros}")
    except Exception as e:
        print(f"Note on toros format: {e}")
    print("=" * 95)


if __name__ == "__main__":
    main()
