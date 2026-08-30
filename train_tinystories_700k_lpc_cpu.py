import os
import time
import math
import torch
import numpy as np
from affine_ai import ASDAGLanguageModel, ASDAGTrainer

def main():
    print("=" * 85)
    print("      TRAINING 700k ASDAG MODEL ON 100MB TINYSTORIES (CPU, DEFAULT SETTINGS)")
    print("=" * 85)

    data_path = "data/tinystories_extracted.bin"
    if not os.path.exists(data_path):
        data_path = "data/tinystories_10mb.txt"

    print(f"Loading dataset from {data_path}...")
    with open(data_path, "rb") as f:
        raw_bytes = f.read()
    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    print(f"Dataset Size: {len(data):,d} bytes (~{len(data)/(1024*1024):.1f} MB)")

    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print("\nInitializing ASDAG Language Model (All Defaults)...")
    model = ASDAGLanguageModel(
        vocab_size=256,
        d_model=136,
        n_layers=6,
        n_heads=4,
        channel_mixer_type="ternary_swiglu", # Default
        use_blt=False,
        dtype=torch.float32
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Architecture  : d_model=136, n_layers=6, n_heads=4")
    print(f"Total Parameters    : {total_params:,d} parameters (~726k)")
    print(f"Channel Mixer       : Ternary BitLinear SwiGLU (1.58-bit MatMul-Free)")
    print(f"Time Mixer          : Monarch GLA Memory (Zero-MatMul)")
    print(f"Training Algorithm  : Local Predictive Coding (LPC, Forward-Only)")
    print(f"Optimizer           : Hybrid Muon (muon_lr=0.03, adamw_lr=3e-3)")
    print(f"Hardware Target     : CPU (Native C++ AVX2/AVX-512 SIMD Engine)")
    print("-" * 85)

    trainer = ASDAGTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        batch_size=16,
        seq_len=128,
        lr=3e-3,
        max_steps=1000,
        warmup_steps=100,
        eval_interval=100,
        eval_iters=20,
        grad_clip=1.0,
        device="cpu",
        use_lpc=True,
        use_muon=True,
        muon_lr=0.03
    )

    print("\nStarting 1,000 Step CPU Training Loop...\n")
    start_time = time.time()
    t_prev = start_time

    for step in range(1, 1001):
        step_loss = trainer.train_step(step)

        if step % 50 == 0 or step == 1:
            t_now = time.time()
            elapsed_interval = t_now - t_prev
            steps_in_interval = 50 if step > 1 else 1
            avg_step_ms = (elapsed_interval / steps_in_interval) * 1000
            tok_per_sec = (steps_in_interval * 16 * 128) / max(1e-6, elapsed_interval)
            total_elapsed = t_now - start_time
            t_prev = t_now

            print(f"Step {step:4d}/1000 | Train Loss: {step_loss:.4f} | {avg_step_ms:>6.1f} ms/step | {tok_per_sec:>7,.0f} tok/s | Elapsed: {total_elapsed:>5.1f}s")

        if step % trainer.eval_interval == 0 or step == 1000:
            eval_metrics = trainer.evaluate()
            val_loss = eval_metrics["val_loss"]
            val_ppl = eval_metrics["val_ppl"]
            print(f" >>> [EVAL @ Step {step:4d}] Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:>6.2f} (Bits/Byte: {eval_metrics['val_bpc']:.3f})")

    total_training_time = time.time() - start_time
    print("\n" + "=" * 85)
    print(f"Training Complete in {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)!")
    final_eval = trainer.evaluate()
    print(f"Final Validation Loss : {final_eval['val_loss']:.4f}")
    print(f"Final Perplexity (PPL): {final_eval['val_ppl']:.2f}")
    print("=" * 85)

if __name__ == "__main__":
    main()
