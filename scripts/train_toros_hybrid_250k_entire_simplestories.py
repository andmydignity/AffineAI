#!/usr/bin/env python3
"""
Train TorosHybrid (~250k params) using True Forward-Only LPC on ENTIRE SimpleStories (2.24 GB)
=============================================================================================
Runs 1 full epoch across all 2.13 GB of training bytes:
  - Architecture: TorosHybridLanguageModel (253,138 backbone params + layer-local LPC heads)
  - Training Mode: Forward-Only Local Predictive Coding (forward_lpc_step)
  - Defaults: channel_mixer_type="asdag_tree", dtype=torch.bfloat16, all TorosHybrid defaults
  - Size: dim=144, n_encoder_layers=6 (exact ~250k regime)
  - Batch size = 128, seq_len = 1024 (131,072 bytes / step)
  - 1 Epoch = 16,270 steps (~2.13 GB data seen)
  - Optimizers: Layer-Local HybridMuonAdamW (forward-only, no cross-layer backprop)
  - Saves:
      * Checkpoint: checkpoints/toros_hybrid_250k_lpc_latest.pt
      * Final PT: models/toros_hybrid_250k_simplestories_lpc.pt
      * Final TOROS: models/toros_hybrid_250k_simplestories_lpc.toros (compact 2-bit packed)
"""

import os
import sys
import math
import time
import signal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig
from affine_ai.core.format import save_toros_model


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.75,
    top_k: int = 40,
    top_p: float = 0.9,
    generator=None
) -> torch.Tensor:
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    scores = logits.float() / max(temperature, 1e-4)
    if top_k is not None and 0 < top_k < scores.size(-1):
        v, _ = torch.topk(scores, min(top_k, scores.size(-1)), dim=-1)
        scores = scores.masked_fill(scores < v[:, -1:], float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
        probs = F.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = (cumulative - probs) > top_p
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        scores = torch.full_like(scores, float("-inf")).scatter(-1, sorted_idx, sorted_scores)
    probs = F.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    data_path = "data/simplestories_eos.bin"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        sys.exit(1)

    raw_data = np.memmap(data_path, dtype=np.uint8, mode="r")
    total_bytes = len(raw_data)
    split = int(0.95 * total_bytes)
    train_data = raw_data[:split]
    val_data = raw_data[split:]

    batch_size = 128
    seq_len = 1024
    bytes_per_step = batch_size * seq_len
    num_train_chunks = (len(train_data) - 1) // seq_len
    total_steps = num_train_chunks // batch_size # Exactly 1 epoch

    print("=" * 105)
    print("   TRAINING TOROS-HYBRID 250K WITH FORWARD-ONLY LPC (ASDAG TREE + BF16) ON SIMPLESTORIES (2.24 GB)")
    print(f"   Device: {device} | Total Data: {total_bytes / (1024*1024):.1f} MB ({total_bytes:,} bytes)")
    print(f"   Train Set: {len(train_data) / (1024*1024):.1f} MB | Val Set: {len(val_data) / (1024*1024):.1f} MB")
    print(f"   Batch Size: {batch_size} | SeqLen: {seq_len} ({bytes_per_step:,} bytes/step)")
    print(f"   Target Steps: {total_steps:,} (1.00 Full Epoch = {total_steps * bytes_per_step / (1024*1024):.1f} MB)")
    print("   Algorithm: Local Predictive Coding (LPC, Forward-Only, Layer-Local Muon/AdamW)")
    print("=" * 105)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # TorosHybrid defaults everywhere except size params (dim=144, n_encoder_layers=6 -> 253k params)
    cfg = TorosHybridConfig(
        dim=144,
        n_encoder_layers=6,
        channel_mixer_type="asdag_tree", # default
        dtype=torch.bfloat16             # default
    )

    model = TorosHybridLanguageModel(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: TorosHybridLanguageModel | Backbone Parameters: {total_params:,}")
    print(f"Config: dim={cfg.dim}, layers={cfg.n_encoder_layers}, mixer={cfg.channel_mixer_type}, dtype={cfg.dtype}")
    print(f"Conv prefix: {cfg.use_conv_prefix} (k={cfg.conv_kernel_size}) | Dense readout: {cfg.use_dense_readout}")

    # Enable LPC local predictive heads and get layer-local optimizers
    model.enable_lpc(dtype=torch.bfloat16, device=device)
    muon_base_lr = 0.02
    adamw_base_lr = 3e-3
    optimizers = model.get_default_lpc_optimizers(
        lr=adamw_base_lr,
        weight_decay=0.01,
        use_muon=True,
        muon_lr=muon_base_lr
    )
    print(f"LPC Initialized: {len(optimizers)} layer-local optimizers configured (6 encoder layers + 1 decoder tail)")

    warmup_steps = 300
    def get_lr_scale(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    # Pre-select fixed validation batches
    val_rng = np.random.RandomState(42)
    val_num_chunks = (len(val_data) - 1) // seq_len
    val_offsets = val_rng.choice(val_num_chunks, size=min(48, val_num_chunks), replace=False) * seq_len
    val_x = torch.from_numpy(np.stack([val_data[i : i + seq_len] for i in val_offsets])).long().to(device)
    val_y = torch.from_numpy(np.stack([val_data[i + 1 : i + seq_len + 1] for i in val_offsets])).long().to(device)

    @torch.no_grad()
    def evaluate():
        model.eval()
        eval_bs = 16
        losses = []
        for i in range(0, val_x.shape[0], eval_bs):
            bx = val_x[i : i + eval_bs]
            by = val_y[i : i + eval_bs]
            logits, loss, _ = model(bx, targets=by)
            losses.append(loss.item())
        model.train()
        mean_loss = float(np.mean(losses))
        ppl = math.exp(min(mean_loss, 20.0))
        bpc = mean_loss / math.log(2.0)
        return mean_loss, ppl, bpc

    # Permutation data loader across all train chunks
    chunk_offsets = np.arange(num_train_chunks) * seq_len
    rng_data = np.random.RandomState(42)
    current_permutation = rng_data.permutation(chunk_offsets)
    perm_ptr = 0

    step_times = []
    total_data_bytes = 0

    def save_checkpoint(tag="latest", step_num=0):
        pt_path = f"checkpoints/toros_hybrid_250k_lpc_{tag}.pt"
        torch.save({
            "step": step_num,
            "config": cfg,
            "model_state_dict": model.state_dict(),
            "total_data_bytes": total_data_bytes,
        }, pt_path)
        toros_path = f"checkpoints/toros_hybrid_250k_lpc_{tag}.toros"
        try:
            save_toros_model(model, toros_path)
        except Exception as e:
            print(f"Toros packing note: {e}")

    # Handle graceful exit on interrupt
    interrupted = False
    def sig_handler(sig, frame):
        nonlocal interrupted
        print("\nInterrupt received! Saving checkpoint before exit...")
        interrupted = True
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Initial probe
    val_loss, val_ppl, val_bpc = evaluate()
    print(f"Step     0 / {total_steps} (   0.0 MB, 0.000 ep) | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:6.2f} | BPC: {val_bpc:.3f}")

    start_time = time.time()
    for step in range(1, total_steps + 1):
        if interrupted:
            break

        t0 = time.time()
        lr_scale = get_lr_scale(step)
        for opt in optimizers:
            for pg in opt.param_groups:
                if pg.get("use_muon", False):
                    pg["lr"] = muon_base_lr * lr_scale
                else:
                    pg["lr"] = adamw_base_lr * lr_scale

        if perm_ptr + batch_size > len(current_permutation):
            remaining = current_permutation[perm_ptr:]
            current_permutation = rng_data.permutation(chunk_offsets)
            needed = batch_size - len(remaining)
            batch_offsets = np.concatenate([remaining, current_permutation[:needed]])
            perm_ptr = needed
        else:
            batch_offsets = current_permutation[perm_ptr : perm_ptr + batch_size]
            perm_ptr += batch_size

        x = torch.from_numpy(np.stack([train_data[i : i + seq_len] for i in batch_offsets])).long().to(device)
        y = torch.from_numpy(np.stack([train_data[i + 1 : i + seq_len + 1] for i in batch_offsets])).long().to(device)

        # Forward-only LPC step: each layer computes loss_i, backward, clip, step locally!
        sync_loss = (step % 200 == 0 or step == total_steps)
        lpc_res = model.forward_lpc_step(x, y, optimizers=optimizers, sync_loss=sync_loss)
        t1 = time.time()

        step_times.append((t1 - t0) * 1000)
        total_data_bytes += bytes_per_step

        if step % 200 == 0 or step == total_steps:
            val_loss, val_ppl, val_bpc = evaluate()
            avg_ms = np.mean(step_times[-50:]) if len(step_times) >= 50 else np.mean(step_times)
            data_seen_mb = total_data_bytes / (1024 * 1024)
            epoch_frac = total_data_bytes / len(train_data)
            tok_per_sec = bytes_per_step / (avg_ms / 1000)
            elapsed_min = (time.time() - start_time) / 60.0
            eta_min = ((total_steps - step) * (avg_ms / 1000)) / 60.0
            cur_loss = lpc_res["loss"] if isinstance(lpc_res["loss"], float) else lpc_res["loss"].item()
            print(f"Step {step:5d}/{total_steps} ({data_seen_mb:7.1f}MB, {epoch_frac:5.3f}ep) | LPC Gen Loss: {cur_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:5.2f} | BPC: {val_bpc:.3f} | {tok_per_sec:7,.0f} tok/s | Elapsed: {elapsed_min:4.1f}m | ETA: {eta_min:4.1f}m")

        if step % 1000 == 0:
            save_checkpoint("latest", step_num=step)
            # In-distribution generational sample probe
            model.eval()
            prompt_bytes = bytes(val_data[:128].tolist())
            prompt_ids = torch.from_numpy(val_data[:128].astype(np.int64)).unsqueeze(0).to(device)
            rng_gen = torch.Generator(device=device)
            rng_gen.manual_seed(42)
            with torch.no_grad():
                curr_ids = prompt_ids
                for _ in range(96):
                    out_logits, _, _ = model(curr_ids)
                    next_token = sample_next_token(out_logits[:, -1, :], temperature=0.75, top_k=40, top_p=0.9, generator=rng_gen)
                    curr_ids = torch.cat([curr_ids, next_token], dim=1)
            gen_slice = bytes(curr_ids[0].tolist())[len(prompt_bytes):]
            sample_text = gen_slice.decode("utf-8", errors="replace").replace("\n", "\\n")
            print(f"    [LPC Sample Step {step}]: \"{sample_text}\"")
            model.train()

    # Final Evaluation & Saving
    final_val_loss, final_val_ppl, final_val_bpc = evaluate()
    print("\n" + "=" * 95)
    print(f"   TRAINING COMPLETE: 1 FULL EPOCH OVER ALL 2.13 GB SIMPLESTORIES WITH LPC ({total_steps:,} STEPS)")
    print(f"   Final Val Loss: {final_val_loss:.4f} | Final Val PPL: {final_val_ppl:.2f} | Final Val BPC: {final_val_bpc:.3f}")
    print("=" * 95)

    # Save final model
    pt_final = "models/toros_hybrid_250k_simplestories_lpc.pt"
    torch.save({
        "config": cfg,
        "model_state_dict": model.state_dict(),
        "final_val_loss": final_val_loss,
        "final_val_ppl": final_val_ppl,
        "final_val_bpc": final_val_bpc,
        "total_data_bytes": total_data_bytes,
        "total_steps": total_steps,
    }, pt_final)
    print(f"Saved PyTorch weights -> {pt_final}")

    toros_final = "models/toros_hybrid_250k_simplestories_lpc.toros"
    res = save_toros_model(model, toros_final)
    toros_size = os.path.getsize(toros_final)
    print(f"Saved Toros Binary format -> {toros_final} ({toros_size:,} bytes, {toros_size/1024:.1f} KB)")
    print(f"Toros Metadata: {res}")


if __name__ == "__main__":
    main()
