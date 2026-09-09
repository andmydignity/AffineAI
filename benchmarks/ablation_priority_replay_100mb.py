#!/usr/bin/env python3
"""
100MB Streaming A/B Benchmark: Standard Single-Pass vs Dynamic Priority Replay Buffer
====================================================================================
Tests Priority Replay in a strictly sub-epoch streaming regime (<1.0 epoch):
  - Dataset: 100.0 MB slice of SimpleStories (95.0 MB train / 5.0 MB val)
  - Training exposure: 1,000 steps x 64 x 1024 = 65.5 MB (0.655 epoch)
  - In Arm 0 (Baseline): Every chunk is strictly seen at most ONCE (zero natural repeats).
  - In Arm 1 (Priority Replay): 75% fresh exploration + 25% replay of high-uncertainty chunks.
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


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.75,
    top_k: int = 40,
    top_p: float = 0.9,
    generator: Optional[torch.Generator] = None
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


class DynamicPriorityReplayBuffer:
    def __init__(self, max_capacity: int = 4000, max_replays: int = 3, max_loss_ceiling: float = 4.5):
        self.max_capacity = max_capacity
        self.max_replays = max_replays
        self.max_loss_ceiling = max_loss_ceiling
        self.buffer: Dict[int, Dict[str, Any]] = {}
        self.total_enqueued = 0
        self.total_replayed = 0
        self.total_retired = 0

    def push_candidates(self, offsets: List[int], losses: np.ndarray, threshold: float):
        for off, l in zip(offsets, losses):
            if threshold <= l <= self.max_loss_ceiling:
                if off not in self.buffer:
                    if len(self.buffer) >= self.max_capacity:
                        min_k = min(self.buffer, key=lambda k: self.buffer[k]["loss"])
                        del self.buffer[min_k]
                    self.buffer[off] = {"loss": float(l), "replays": 0}
                    self.total_enqueued += 1

    def sample(self, n: int, rng: np.random.RandomState) -> List[int]:
        if not self.buffer or n <= 0:
            return []
        candidates = list(self.buffer.keys())
        losses = np.array([self.buffer[c]["loss"] for c in candidates])
        sum_l = np.sum(losses)
        if sum_l > 0:
            probs = losses / sum_l
        else:
            probs = np.ones(len(candidates)) / len(candidates)
        
        sample_n = min(n, len(candidates))
        chosen = rng.choice(candidates, size=sample_n, replace=False, p=probs).tolist()

        for c in chosen:
            self.buffer[c]["replays"] += 1
            self.total_replayed += 1
            if self.buffer[c]["replays"] >= self.max_replays:
                del self.buffer[c]
                self.total_retired += 1

        return chosen

    def __len__(self):
        return len(self.buffer)


def run_experiment(
    arm_name: str,
    use_priority_replay: bool,
    train_data: np.memmap,
    val_data: np.memmap,
    steps: int = 1000,
    batch_size: int = 64,
    seq_len: int = 1024,
    device: str = "cuda"
) -> Dict[str, Any]:
    set_seed(42)
    print("\n" + "=" * 105)
    print(f"   STARTING ARM: {arm_name.upper()}")
    print(f"   Mode: {'75% Fresh Exploration + 25% Priority Replay' if use_priority_replay else '100% Single-Pass Streaming (Baseline)'}")
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

    # Pre-select fixed validation batches
    val_rng = np.random.RandomState(42)
    val_num_chunks = (len(val_data) - 1) // seq_len
    val_offsets = val_rng.choice(val_num_chunks, size=min(48, val_num_chunks), replace=False) * seq_len
    val_x = torch.from_numpy(np.stack([val_data[i : i + seq_len] for i in val_offsets])).long().to(device)
    val_y = torch.from_numpy(np.stack([val_data[i + 1 : i + seq_len + 1] for i in val_offsets])).long().to(device)

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        eval_bs = 16
        for i in range(0, val_x.shape[0], eval_bs):
            bx = val_x[i : i + eval_bs]
            by = val_y[i : i + eval_bs]
            logits, loss, _ = model(bx, targets=by)
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
    # Strictly sequential permutation through the 100MB dataset
    perm = rng.permutation(chunk_offsets)
    ptr = 0

    replay_buffer = DynamicPriorityReplayBuffer(max_capacity=4000, max_replays=3, max_loss_ceiling=4.5) if use_priority_replay else None
    running_losses: List[float] = []

    history = []
    milestone_ppl_7 = None
    milestone_ppl_6_2 = None
    milestone_ppl_6_0 = None
    milestone_ppl_5_8 = None

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

        # Batch assembly: Fresh exploration vs Priority Replay
        if use_priority_replay and replay_buffer is not None:
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

        bx = torch.from_numpy(np.stack([train_data[i : i + seq_len] for i in batch_starts])).long().to(device)
        by = torch.from_numpy(np.stack([train_data[i + 1 : i + seq_len + 1] for i in batch_starts])).long().to(device)

        sync = ((step + 1) % 100 == 0 or step == steps - 1)
        step_res = model.forward_lpc_step(
            byte_ids=bx,
            targets=by,
            optimizers=optimizers,
            use_async_pipelining=True,
            sync_loss=sync,
            return_sample_loss=use_priority_replay
        )

        # Update replay buffer with fresh chunks
        if use_priority_replay and replay_buffer is not None and "sample_loss" in step_res and step_res["sample_loss"] is not None:
            s_losses = step_res["sample_loss"].float().cpu().numpy()
            fresh_losses = s_losses[:len(fresh_starts)]
            running_losses.extend(fresh_losses.tolist())
            if len(running_losses) > 1000:
                running_losses = running_losses[-1000:]
            
            thresh = float(np.percentile(running_losses, 65))
            replay_buffer.push_candidates(fresh_starts, fresh_losses, threshold=thresh)

        # Periodic evaluation & milestone check
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
            if milestone_ppl_5_8 is None and v_ppl < 5.8:
                milestone_ppl_5_8 = step + 1

    total_time = time.time() - t0
    final_loss, final_ppl, final_bpc = evaluate()

    # Generation sample
    model.eval()
    prompt_text = "Once upon a time, there was a little"
    p_bytes = torch.tensor(list(prompt_text.encode("utf-8")), dtype=torch.long, device=device).unsqueeze(0)
    gen_bytes = list(p_bytes[0].cpu().numpy())
    with torch.no_grad():
        curr_bytes = p_bytes
        for _ in range(120):
            logits, _, _ = model(curr_bytes)
            next_t = sample_next_token(logits[:, -1, :], temperature=0.75, top_k=40, top_p=0.9)
            gen_bytes.append(int(next_t.item()))
            curr_bytes = torch.tensor([gen_bytes], dtype=torch.long, device=device)
    gen_str = bytes(gen_bytes).decode("utf-8", errors="replace").replace("\n", "\\n")

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
        "milestone_ppl_5_8": milestone_ppl_5_8,
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
    slice_100mb_bytes = 100 * 1024 * 1024
    active_data = raw_data[:slice_100mb_bytes]
    split = int(0.95 * len(active_data))
    train_data = active_data[:split]
    val_data = active_data[split:]

    total_exposure = 1000 * 64 * 1024
    print("=" * 105)
    print("   100MB STREAMING A/B BENCHMARK: SINGLE-PASS VS DYNAMIC PRIORITY REPLAY BUFFER")
    print(f"   Corpus Size: 100.0 MB ({len(active_data):,} bytes)")
    print(f"   Train Set: {len(train_data)/(1024*1024):.2f} MB | Val Set: {len(val_data)/(1024*1024):.2f} MB | Device: {device}")
    print(f"   Steps: 1,000 steps x 64 BS x 1024 SeqLen = {total_exposure/(1024*1024):.2f} MB exposure ({total_exposure/len(train_data)*100:.1f}% of Train Set)")
    print("   Regime: Strictly sub-epoch streaming (each baseline chunk seen AT MOST ONCE)")
    print("=" * 105)

    # Arm 0: Baseline Standard Single-Pass Streaming
    res_base = run_experiment(
        arm_name="Arm 0: Single-Pass Streaming (Baseline)",
        use_priority_replay=False,
        train_data=train_data,
        val_data=val_data,
        steps=1000,
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
        steps=1000,
        batch_size=64,
        seq_len=1024,
        device=device
    )

    # Comparison Summary Table
    print("\n" + "=" * 105)
    print("                       FINAL COMPARATIVE 100MB STREAMING RESULTS")
    print("=" * 105)
    header = f"{'Metric / Evaluation':<38} | {'Arm 0: Single-Pass Baseline':<28} | {'Arm 1: Priority Replay (75/25)':<30}"
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

    m58_0 = f"Step {res_base['milestone_ppl_5_8']}" if res_base['milestone_ppl_5_8'] else "Did not reach"
    m58_1 = f"Step {res_replay['milestone_ppl_5_8']}" if res_replay['milestone_ppl_5_8'] else "Did not reach"
    print(f"{'Steps to PPL < 5.8':<38} | {m58_0:<28} | {m58_1:<30}")

    if res_replay['buffer_stats']:
        b = res_replay['buffer_stats']
        b_str = f"{b['enqueued']} / {b['replayed']} / {b['retired']}"
        print(f"{'Buffer Stats (Enq / Rep / Ret)':<38} | {'N/A':<28} | {b_str:<30}")

    print("-" * 105)
    print("STEP-BY-STEP VALIDATION PPL PROGRESSION:")
    print(f"{'Step':<10} | {'Arm 0 (Single-Pass Baseline)':<28} | {'Arm 1 (Priority Replay)':<30} | {'PPL Delta':<15}")
    print("-" * 105)
    for h0, h1 in zip(res_base['history'], res_replay['history']):
        diff = h1['val_ppl'] - h0['val_ppl']
        diff_str = f"{diff:+.2f} ({diff/h0['val_ppl']*100:+.1f}%)"
        print(f"Step {h0['step']:<5d} | {h0['val_ppl']:<28.2f} | {h1['val_ppl']:<30.2f} | {diff_str:<15}")

    print("\n" + "=" * 105)
    print("                         GENERATION SAMPLE PROBES")
    print("=" * 105)
    print(f">>> Arm 0 (Single-Pass Baseline):\n    {res_base['sample'][:200]}...\n")
    print(f">>> Arm 1 (Priority Replay):\n    {res_replay['sample'][:200]}...\n")
    print("=" * 105)


if __name__ == "__main__":
    main()
