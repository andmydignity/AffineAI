#!/usr/bin/env python3
"""
Deep Function-Level Profiler & Bottleneck Analysis (100 Steps)
=============================================================
Runs 100 training steps on SimpleStories with TorosHybridLanguageModel + Forward-Only LPC + Muon.
Profiles:
  1. CUDA Event hardware timers for each pipeline phase (Data, Forward, Local Heads, Backward, Optimizer, Buffer).
  2. PyTorch Profiler: Self CUDA time, Self CPU time, Shapes, FLOPs, and Call Counts.
  3. Python function-level cProfile breakdown.
"""

import os
import sys
import math
import time
import cProfile
import pstats
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig
from affine_ai.core.priority_replay import DynamicPriorityReplayBuffer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA is required for deep GPU profiling.")
        return

    data_path = "data/simplestories_eos.bin"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        return

    raw_data = np.memmap(data_path, dtype=np.uint8, mode="r")
    slice_10mb = 10 * 1024 * 1024
    train_data = raw_data[:slice_10mb]

    batch_size = 64
    seq_len = 1024
    num_steps = 100
    warmup_steps = 10

    print("=" * 90)
    print("   DEEP FUNCTION-LEVEL BOTTLENECK PROFILER (100 STEPS)")
    print(f"   Model: TorosHybrid 250k (dim=144, 6 layers, asdag_tree, BF16)")
    print(f"   Batch: {batch_size} x {seq_len} = {batch_size * seq_len:,} tokens/step")
    print(f"   Steps: {num_steps} (warmup: {warmup_steps})")
    print("=" * 90)

    # Initialize model & optimizers
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

    # Pinned dataset buffer
    train_tensor = torch.from_numpy(np.asarray(train_data, dtype=np.int64)).pin_memory()
    num_train_chunks = (len(train_data) - 1) // seq_len
    chunk_offsets = np.arange(num_train_chunks) * seq_len
    rng = np.random.RandomState(42)
    perm = rng.permutation(chunk_offsets)
    ptr = 0

    replay_buffer = DynamicPriorityReplayBuffer(max_capacity=2000, max_replays=3, max_loss_ceiling=4.5)
    offsets = torch.arange(seq_len)

    # Section timing stats using CUDA events
    phase_times = defaultdict(list)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # We profile with PyTorch Profiler on steps 20..35 to avoid massive memory traces
    # while capturing steady-state kernel profiles
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True
    )
    prof_start_step = 20
    prof_end_step = 30

    print("\nStarting execution...")
    step_times = []
    t_global_start = time.time()

    # Enable cProfile across the entire 100 steps
    py_profiler = cProfile.Profile()
    py_profiler.enable()

    for step in range(num_steps):
        t_step_start = time.time()

        if step == prof_start_step:
            prof.start()

        # --- Phase 1: Batch Assembly & DMA Transfer ---
        e_data_start = torch.cuda.Event(enable_timing=True)
        e_data_end = torch.cuda.Event(enable_timing=True)
        e_data_start.record()

        n_replay = min(16, len(replay_buffer))
        n_fresh = batch_size - n_replay
        if ptr + n_fresh > len(perm):
            perm = rng.permutation(chunk_offsets)
            ptr = 0
        fresh_starts = perm[ptr : ptr + n_fresh].tolist()
        ptr += n_fresh

        replayed_starts = replay_buffer.sample(n_replay, rng) if n_replay > 0 else []
        batch_starts = fresh_starts + replayed_starts

        batch_idx = torch.tensor(batch_starts, dtype=torch.long).unsqueeze(1) + offsets.unsqueeze(0)
        bx = train_tensor[batch_idx].to(device, non_blocking=True)
        by = train_tensor[batch_idx + 1].to(device, non_blocking=True)

        e_data_end.record()

        # --- Phase 2: Forward-Only LPC Step ---
        e_train_start = torch.cuda.Event(enable_timing=True)
        e_train_end = torch.cuda.Event(enable_timing=True)
        e_train_start.record()

        sync = (step == num_steps - 1)
        with record_function("forward_lpc_step"):
            step_res = model.forward_lpc_step(
                byte_ids=bx,
                targets=by,
                optimizers=optimizers,
                use_async_pipelining=True,
                sync_loss=sync,
                return_sample_loss=True
            )

        e_train_end.record()

        # --- Phase 3: Priority Buffer Update ---
        e_buf_start = torch.cuda.Event(enable_timing=True)
        e_buf_end = torch.cuda.Event(enable_timing=True)
        e_buf_start.record()

        if "sample_loss" in step_res and step_res["sample_loss"] is not None:
            replay_buffer.push_candidates(fresh_starts, step_res["sample_loss"][:len(fresh_starts)])

        e_buf_end.record()

        torch.cuda.synchronize()
        if step >= warmup_steps:
            phase_times["data_dma"].append(e_data_start.elapsed_time(e_data_end))
            phase_times["train_step"].append(e_train_start.elapsed_time(e_train_end))
            phase_times["buffer_update"].append(e_buf_start.elapsed_time(e_buf_end))
            step_times.append(time.time() - t_step_start)

        if prof_start_step <= step < prof_end_step:
            prof.step()
        elif step == prof_end_step:
            prof.stop()

        if (step + 1) % 20 == 0:
            current_tok_s = (batch_size * seq_len) / np.mean(step_times[-20:])
            print(f"  Step {step + 1:3d}/100 complete | Current Speed: {current_tok_s:,.0f} tok/s")

    py_profiler.disable()
    t_global_total = time.time() - t_global_start

    print("\n" + "=" * 90)
    print("                    1. PIPELINE PHASE TIMING BREAKDOWN")
    print("=" * 90)
    total_avg_step_ms = np.mean(step_times) * 1000.0
    avg_data_ms = np.mean(phase_times["data_dma"])
    avg_train_ms = np.mean(phase_times["train_step"])
    avg_buf_ms = np.mean(phase_times["buffer_update"])
    overhead_ms = max(0.0, total_avg_step_ms - (avg_data_ms + avg_train_ms + avg_buf_ms))

    print(f"{'Pipeline Phase':<35} | {'Avg Time (ms)':<15} | {'Percentage':<12}")
    print("-" * 70)
    print(f"{'1. Data Slice & CUDA DMA':<35} | {avg_data_ms:<15.2f} | {avg_data_ms/total_avg_step_ms*100:<11.1f}%")
    print(f"{'2. Forward LPC + Optimizer Step':<35} | {avg_train_ms:<15.2f} | {avg_train_ms/total_avg_step_ms*100:<11.1f}%")
    print(f"{'3. Priority Replay Buffer Push':<35} | {avg_buf_ms:<15.2f} | {avg_buf_ms/total_avg_step_ms*100:<11.1f}%")
    print(f"{'4. Host Python / Kernel Launch Gap':<35} | {overhead_ms:<15.2f} | {overhead_ms/total_avg_step_ms*100:<11.1f}%")
    print("-" * 70)
    print(f"{'Total Step Time':<35} | {total_avg_step_ms:<15.2f} | 100.0%")
    print(f"Overall Throughput: {(batch_size * seq_len * num_steps) / t_global_total:,.0f} tokens/s")

    # PyTorch Profiler Detailed Tables
    if prof is not None:
        print("\n" + "=" * 90)
        print("          2. HIGH-LEVEL OPERATIONS & MODULES BY TOTAL CUDA TIME")
        print("=" * 90)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        print("\n" + "=" * 90)
        print("          3. TOP GPU KERNELS BY SELF CUDA TIME (HARDWARE BOTTLENECK)")
        print("=" * 90)
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))

        print("\n" + "=" * 90)
        print("          4. TOP CPU OPERATORS BY SELF CPU TIME (HOST OVERHEAD)")
        print("=" * 90)
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

    # Python cProfile: Top CPU Functions by Self Time
    print("\n" + "=" * 90)
    print("             5. TOP 20 PYTHON FUNCTIONS BY CUMULATIVE TIME (CPROFILE)")
    print("=" * 90)
    stats = pstats.Stats(py_profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(20)

    print("=" * 90)
    print("   PROFILING COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
