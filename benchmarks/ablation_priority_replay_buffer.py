#!/usr/bin/env python3
"""
A/B Benchmark: Standard Shuffle vs Dynamic Priority Replay Buffer (AXIOM Info-Gain Selection)
=============================================================================================
Evaluates whether prioritizing high-uncertainty / high-loss training chunks (75% fresh + 25% replay)
accelerates convergence and improves validation perplexity on 10MB SimpleStories:

Arms:
  - Arm 0: Standard Uniform Shuffle (Baseline, 100% fresh batches from random permutation)
  - Arm 1: Dynamic Priority Replay Buffer (75% fresh + 25% high-uncertainty replay, max 3 replays/chunk)

Environment:
  - Dataset: 10.0 MB slice of SimpleStories (9.5MB train / 0.5MB val)
  - Model: TorosHybridLanguageModel (dim=144, layers=6, asdag_tree, 253,138 params, BF16)
  - Steps: 800 steps, batch_size=64, seq_len=1024 (65,536 bytes/step)
  - Optimizer: Forward-Only LPC with layer-local HybridMuonAdamW
"""

import os
import sys
import math
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig
from affine_ai.optim.muon import HybridMuonAdamW
from affine_ai.core.priority_replay import DynamicPriorityReplayBuffer


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def run_experiment(
    arm_name: str,
    use_priority_replay: bool,
    train_data: np.memmap,
    val_data: np.memmap,
    steps: int = 800,
    batch_size: int = 64,
    seq_len: int = 1024,
    device: str = "cuda"
) -> Dict[str, Any]:
    set_seed(42)
    print("\n" + "=" * 105)
    print(f"   STARTING ARM: {arm_name.upper()}")
    print(f"   Mode: {'75% Fresh + 25% Priority Replay (Buffer)' if use_priority_replay else '100% Uniform Random Shuffle (Baseline)'}")
    print("=" * 105)

    cfg = TorosHybridConfig(
        dim=144,
        n_encoder_layers=6,
        channel_mixer_type="asdag_tree",
        dtype=torch.bfloat16
    )
    model = TorosHybridLanguageModel(cfg).to(device)
    model.enable_lpc(dtype=torch.bfloat16, device=device)
    optimizers = model.get_default_lpc_optimizers(
        lr=3e-3,
        weight_decay=0.01,
        use_muon=True,
        muon_lr=0.02
    )

    warmup_steps = 100
    def get_lr_scale(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    train_tensor = torch.from_numpy(np.asarray(train_data, dtype=np.int64))
    val_tensor = torch.from_numpy(np.asarray(val_data, dtype=np.int64))
    if device == "cuda":
        train_tensor = train_tensor.pin_memory()
        val_tensor = val_tensor.pin_memory()

    # Pre-select validation batches
    val_rng = np.random.RandomState(42)
    val_num_chunks = (len(val_data) - 1) // seq_len
    val_offsets = val_rng.choice(val_num_chunks, size=min(32, val_num_chunks), replace=False) * seq_len
    offsets = torch.arange(seq_len)
    val_idx = torch.tensor(val_offsets, dtype=torch.long).unsqueeze(1) + offsets.unsqueeze(0)
    val_x = val_tensor[val_idx].to(device, non_blocking=True)
    val_y = val_tensor[val_idx + 1].to(device, non_blocking=True)

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        eval_bs = 16
        for i in range(0, val_x.shape[0], eval_bs):
            bx = val_x[i : i + eval_bs]
            by = val_y[i : i + eval_bs]
            _, loss, _ = model(bx, targets=by, return_logits=False)
            losses.append(loss.item())
        model.train()
        m_loss = float(np.mean(losses))
        ppl = math.exp(min(m_loss, 20.0))
        bpc = m_loss / math.log(2.0)
        return m_loss, ppl, bpc

    init_loss, init_ppl, init_bpc = evaluate()
    print(f"[Init] Val Loss: {init_loss:.4f} | Val PPL: {init_ppl:.2f} | Val BPC: {init_bpc:.3f}\n")

    num_train_chunks = (len(train_data) - 1) // seq_len
    chunk_offsets = np.arange(num_train_chunks) * seq_len
    rng = np.random.RandomState(42)
    perm = rng.permutation(chunk_offsets)
    ptr = 0

    replay_buffer = DynamicPriorityReplayBuffer(max_capacity=2000, max_replays=3, max_loss_ceiling=4.5) if use_priority_replay else None

    history = []
    milestone_ppl_7 = None
    milestone_ppl_6_2 = None
    milestone_ppl_6_0 = None

    t0 = time.time()
    for step in range(steps):
        # LR schedule
        scale = get_lr_scale(step)
        for opt in optimizers:
            if hasattr(opt, 'adamw_base_lr'):
                for pg in opt.adamw_opt.param_groups:
                    pg['lr'] = opt.adamw_base_lr * scale
            if hasattr(opt, 'muon_base_lr') and opt.muon_opt is not None:
                for pg in opt.muon_opt.param_groups:
                    pg['lr'] = opt.muon_base_lr * scale

        # Batch assembly: Fresh vs Replay
        if use_priority_replay and replay_buffer is not None:
            # 75% fresh (48), 25% replay (16)
            n_replay = min(16, len(replay_buffer))
            n_fresh = batch_size - n_replay

            if ptr + n_fresh > len(perm):
                perm = rng.permutation(chunk_offsets)
                ptr = 0
            fresh_starts = perm[ptr : ptr + n_fresh].tolist()
            ptr += n_fresh

            replayed_starts = replay_buffer.sample(n_replay, rng) if n_replay > 0 else []
            batch_starts = fresh_starts + replayed_starts
        else:
            if ptr + batch_size > len(perm):
                perm = rng.permutation(chunk_offsets)
                ptr = 0
            fresh_starts = perm[ptr : ptr + batch_size].tolist()
            ptr += batch_size
            batch_starts = fresh_starts

        batch_idx = torch.tensor(batch_starts, dtype=torch.long).unsqueeze(1) + offsets.unsqueeze(0)
        bx = train_tensor[batch_idx].to(device, non_blocking=True)
        by = train_tensor[batch_idx + 1].to(device, non_blocking=True)

        sync = ((step + 1) % 100 == 0 or step == steps - 1)
        step_res = model.forward_lpc_step(
            byte_ids=bx,
            targets=by,
            optimizers=optimizers,
            use_async_pipelining=True,
            sync_loss=sync,
            return_sample_loss=use_priority_replay
        )

        # Update replay buffer with fresh chunks using core DynamicPriorityReplayBuffer
        if use_priority_replay and replay_buffer is not None and "sample_loss" in step_res and step_res["sample_loss"] is not None:
            replay_buffer.push_candidates(fresh_starts, step_res["sample_loss"][:len(fresh_starts)])

        # Logging & Milestones
        if (step + 1) % 100 == 0 or step == steps - 1:
            v_loss, v_ppl, v_bpc = evaluate()
            elapsed = time.time() - t0
            tok_per_sec = ((step + 1) * batch_size * seq_len) / elapsed
            
            t_loss = step_res['loss'].item() if hasattr(step_res['loss'], 'item') else float(step_res['loss'])
            buf_str = f" | Buffer: {len(replay_buffer):4d} active, {replay_buffer.total_replayed:4d} replays, {replay_buffer.total_retired:4d} retired" if use_priority_replay else ""
            print(f"Step {step+1:4d}/{steps} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val PPL: {v_ppl:.2f} | Val BPC: {v_bpc:.3f}{buf_str} | Speed: {tok_per_sec:,.0f} tok/s")

            history.append({
                "step": step + 1,
                "val_loss": v_loss,
                "val_ppl": v_ppl,
                "val_bpc": v_bpc,
                "tok_per_sec": tok_per_sec
            })

            if milestone_ppl_7 is None and v_ppl < 7.0:
                milestone_ppl_7 = step + 1
            if milestone_ppl_6_2 is None and v_ppl < 6.2:
                milestone_ppl_6_2 = step + 1
            if milestone_ppl_6_0 is None and v_ppl < 6.0:
                milestone_ppl_6_0 = step + 1

    total_time = time.time() - t0
    final_loss, final_ppl, final_bpc = evaluate()

    # Generation sample via core model incremental planning
    model.eval()
    prompt_text = "Once upon a time, there was a little"
    p_bytes = torch.tensor(list(prompt_text.encode("utf-8")), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        out_bytes = model.generate_with_latent_planning(
            prompt_bytes=p_bytes,
            max_new_bytes=120,
            temperature=0.75,
            top_k=40,
            top_p=0.9,
            eos_byte=None
        )
    gen_str = bytes(out_bytes[0].cpu().tolist()).decode("utf-8", errors="replace").replace("\n", "\\n")

    return {
        "arm_name": arm_name,
        "use_priority_replay": use_priority_replay,
        "total_time": total_time,
        "final_loss": final_loss,
        "final_ppl": final_ppl,
        "final_bpc": final_bpc,
        "milestone_ppl_7": milestone_ppl_7,
        "milestone_ppl_6_2": milestone_ppl_6_2,
        "milestone_ppl_6_0": milestone_ppl_6_0,
        "history": history,
        "sample": gen_str,
        "buffer_stats": {
            "enqueued": replay_buffer.total_enqueued,
            "replayed": replay_buffer.total_replayed,
            "retired": replay_buffer.total_retired
        } if replay_buffer else None
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "data/simplestories_eos.bin"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        sys.exit(1)

    raw_data = np.memmap(data_path, dtype=np.uint8, mode="r")
    slice_10mb_bytes = 10 * 1024 * 1024
    active_data = raw_data[:slice_10mb_bytes]
    split = int(0.95 * len(active_data))
    train_data = active_data[:split]
    val_data = active_data[split:]

    print("=" * 105)
    print("   A/B BENCHMARK: STANDARD SHUFFLE VS DYNAMIC PRIORITY REPLAY BUFFER")
    print(f"   Dataset: 10.0 MB SimpleStories ({len(active_data):,} bytes)")
    print(f"   Train Set: {len(train_data)/(1024*1024):.2f} MB | Val Set: {len(val_data)/(1024*1024):.2f} MB | Device: {device}")
    print(f"   Model: TorosHybrid 250k (dim=144, layers=6, asdag_tree, BF16)")
    print("=" * 105)

    # Arm 0: Baseline Standard Shuffle
    res_base = run_experiment(
        arm_name="Arm 0: Standard Shuffle (Baseline)",
        use_priority_replay=False,
        train_data=train_data,
        val_data=val_data,
        steps=800,
        batch_size=64,
        seq_len=1024,
        device=device
    )

    # Arm 1: Dynamic Priority Replay Buffer
    res_replay = run_experiment(
        arm_name="Arm 1: Dynamic Priority Replay (75% Fresh + 25% Replay)",
        use_priority_replay=True,
        train_data=train_data,
        val_data=val_data,
        steps=800,
        batch_size=64,
        seq_len=1024,
        device=device
    )

    # Comparison Summary Table
    print("\n" + "=" * 105)
    print("                       FINAL COMPARATIVE A/B RESULTS SUMMARY")
    print("=" * 105)
    header = f"{'Metric / Evaluation':<38} | {'Arm 0: Standard Shuffle':<28} | {'Arm 1: Dynamic Priority Replay':<30}"
    print(header)
    print("-" * 105)

    print(f"{'Final Validation Loss':<38} | {res_base['final_loss']:<28.4f} | {res_replay['final_loss']:<30.4f}")
    print(f"{'Final Validation Perplexity (PPL)':<38} | {res_base['final_ppl']:<28.2f} | {res_replay['final_ppl']:<30.2f}")
    print(f"{'Final Validation BPC':<38} | {res_base['final_bpc']:<28.3f} | {res_replay['final_bpc']:<30.3f}")
    print(f"{'Total Training Runtime':<38} | {res_base['total_time']:<26.1f}s | {res_replay['total_time']:<28.1f}s")
    
    m7_0 = f"Step {res_base['milestone_ppl_7']}" if res_base['milestone_ppl_7'] else "Did not reach"
    m7_1 = f"Step {res_replay['milestone_ppl_7']}" if res_replay['milestone_ppl_7'] else "Did not reach"
    print(f"{'Steps to PPL < 7.0':<38} | {m7_0:<28} | {m7_1:<30}")

    m62_0 = f"Step {res_base['milestone_ppl_6_2']}" if res_base['milestone_ppl_6_2'] else "Did not reach"
    m62_1 = f"Step {res_replay['milestone_ppl_6_2']}" if res_replay['milestone_ppl_6_2'] else "Did not reach"
    print(f"{'Steps to PPL < 6.2':<38} | {m62_0:<28} | {m62_1:<30}")

    m60_0 = f"Step {res_base['milestone_ppl_6_0']}" if res_base['milestone_ppl_6_0'] else "Did not reach"
    m60_1 = f"Step {res_replay['milestone_ppl_6_0']}" if res_replay['milestone_ppl_6_0'] else "Did not reach"
    print(f"{'Steps to PPL < 6.0':<38} | {m60_0:<28} | {m60_1:<30}")

    if res_replay['buffer_stats']:
        b = res_replay['buffer_stats']
        b_str = f"{b['enqueued']} / {b['replayed']} / {b['retired']}"
        print(f"{'Buffer Stats (Enq / Rep / Ret)':<38} | {'N/A':<28} | {b_str:<30}")

    print("-" * 105)
    print("STEP-BY-STEP VALIDATION PPL PROGRESSION:")
    print(f"{'Step':<10} | {'Arm 0 (Standard Shuffle)':<28} | {'Arm 1 (Dynamic Priority Replay)':<30} | {'PPL Delta':<15}")
    print("-" * 105)
    for h0, h1 in zip(res_base['history'], res_replay['history']):
        diff = h1['val_ppl'] - h0['val_ppl']
        diff_str = f"{diff:+.2f} ({diff/h0['val_ppl']*100:+.1f}%)"
        print(f"Step {h0['step']:<5d} | {h0['val_ppl']:<28.2f} | {h1['val_ppl']:<30.2f} | {diff_str:<15}")

    print("\n" + "=" * 105)
    print("                         GENERATION SAMPLE PROBES")
    print("=" * 105)
    print(f">>> Arm 0 (Standard Shuffle):\n    {res_base['sample'][:200]}...\n")
    print(f">>> Arm 1 (Priority Replay):\n    {res_replay['sample'][:200]}...\n")
    print("=" * 105)


if __name__ == "__main__":
    main()
