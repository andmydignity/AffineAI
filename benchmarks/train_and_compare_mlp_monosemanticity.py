#!/usr/bin/env python3
"""
Classic MLP vs ASDAG Tree: Monosemanticity & Polysemanticity Benchmark on 10MB SimpleStories
=============================================================================================
Iso-parameter comparison (~250k parameters):
  - Model A: Classic Dense MLP (252,640 params, W1: 144->84 -> GELU -> W2: 84->144)
  - Model B: ASDAG Sparse Tree (253,138 params, 1:16 hierarchical sparse routing)
  - Model C: ASDAG Sparse Tree trained on full 2.24 GB SimpleStories

Protocol:
  1. Train Classic MLP on 10.0 MB slice for 800 steps using layer-local LPC (HybridMuonAdamW).
  2. Train ASDAG Tree on the identical 10.0 MB slice for 800 steps under the same setup.
  3. Load pre-existing 2.24 GB trained ASDAG model.
  4. Perform mechanistic interpretability & monosemanticity profiling on validation sequences:
     - Map every patch activation back to UTF-8 text string.
     - Profile 84 MLP neurons per layer vs 16 ASDAG tree leaves per layer.
     - Compute Semantic Purity %, Polysemanticity Rate (<50% purity), Shannon Entropy, and Sparsity.
"""

import os
import sys
import math
import time
from collections import Counter, defaultdict
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


# Comprehensive 6-class linguistic taxonomy
TAXONOMY = {
    "Dialogue & Punctuation": {
        '"', "'", "!", "?", ",", ".", ":", ";", "\n", "said", "asked", "replied", "shouted",
        "cried", "whispered", "told", "screamed", "spoke"
    },
    "Pronouns & Agents": {
        "he", "she", "they", "it", "i", "you", "we", "his", "her", "their", "my", "your",
        "our", "him", "them", "me", "us", "himself", "herself", "themselves", "someone"
    },
    "Entities & Nouns": {
        "cat", "dog", "tree", "bird", "garden", "house", "car", "ball", "flower", "sun",
        "water", "friend", "mom", "dad", "boy", "girl", "room", "door", "day", "night",
        "bed", "food", "toy", "box", "park", "forest", "sky", "star", "animal", "book",
        "baby", "bear", "fish", "rabbit", "kitty", "puppy", "lion", "duck", "apple"
    },
    "Actions & Verbs": {
        "jumped", "walked", "ran", "played", "looked", "saw", "found", "went", "came",
        "smiled", "laughed", "flew", "ate", "wanted", "took", "heard", "started", "stopped",
        "opened", "closed", "helped", "hugged", "kissed", "dropped", "picked", "pulled",
        "pushed", "called", "liked", "loved", "gave", "made", "lived", "felt"
    },
    "Grammar & Prepositions": {
        "the", "a", "an", "and", "to", "in", "of", "on", "with", "for", "at", "from",
        "but", "by", "up", "down", "out", "into", "over", "so", "that", "this", "then",
        "there", "here", "as", "if", "not", "no", "yes", "all", "very", "too", "also"
    },
    "Sensory & Adjectives": {
        "happy", "sad", "little", "big", "small", "tiny", "huge", "red", "blue", "green",
        "yellow", "bright", "soft", "warm", "cold", "pretty", "sweet", "beautiful", "good",
        "bad", "nice", "fun", "fast", "slow", "clean", "dirty", "funny", "scary", "fluffy"
    }
}


def classify_text(text: str) -> str:
    cleaned = text.strip().lower()
    if not cleaned:
        return "Whitespace / Delimiter"
    for cat, words in TAXONOMY.items():
        if cleaned in words or any(w in cleaned for w in words if len(w) >= 3):
            return cat
    if any(p in text for p in ['"', "'", "!", "?", ",", ".", ";"]):
        return "Dialogue & Punctuation"
    if cleaned.endswith("ed") or cleaned.endswith("ing"):
        return "Actions & Verbs"
    if cleaned.endswith("y") or cleaned.endswith("ful") or cleaned.endswith("ish"):
        return "Sensory & Adjectives"
    return "Other / Miscellaneous"


def compute_shannon_entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def compute_gini(values: np.ndarray) -> float:
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
    v = np.sort(np.abs(values))
    n = len(v)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)))


def train_model(
    model_name: str,
    cfg: TorosHybridConfig,
    train_data: np.memmap,
    val_data: np.memmap,
    steps: int = 800,
    batch_size: int = 64,
    seq_len: int = 1024,
    save_path: str = "",
    device: str = "cuda"
) -> TorosHybridLanguageModel:
    set_seed(42)
    print("=" * 105)
    print(f"   TRAINING {model_name.upper()} ON 10.0 MB SIMPLESTORIES (Steps: {steps}, BS: {batch_size}, Device: {device})")
    print("=" * 105)

    model = TorosHybridLanguageModel(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} | Backbone Parameters: {param_count:,}")

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

    # Pre-select validation batches
    val_rng = np.random.RandomState(42)
    val_num_chunks = (len(val_data) - 1) // seq_len
    val_offsets = val_rng.choice(val_num_chunks, size=min(32, val_num_chunks), replace=False) * seq_len
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
    perm = rng.permutation(chunk_offsets)
    ptr = 0

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

        # Batch assembly
        if ptr + batch_size > len(perm):
            perm = rng.permutation(chunk_offsets)
            ptr = 0
        batch_starts = perm[ptr : ptr + batch_size]
        ptr += batch_size

        bx = torch.from_numpy(np.stack([train_data[i : i + seq_len] for i in batch_starts])).long().to(device)
        by = torch.from_numpy(np.stack([train_data[i + 1 : i + seq_len + 1] for i in batch_starts])).long().to(device)

        step_res = model.forward_lpc_step(
            byte_ids=bx,
            targets=by,
            optimizers=optimizers,
            use_async_pipelining=True,
            sync_loss=(step % 200 == 0 or step == steps - 1)
        )

        if (step + 1) % 200 == 0 or step == steps - 1:
            v_loss, v_ppl, v_bpc = evaluate()
            elapsed = time.time() - t0
            tok_per_sec = ((step + 1) * batch_size * seq_len) / elapsed
            t_loss = step_res['loss'].item() if hasattr(step_res['loss'], 'item') else float(step_res['loss'])
            print(f"Step {step+1:4d}/{steps} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val PPL: {v_ppl:.2f} | Val BPC: {v_bpc:.3f} | Speed: {tok_per_sec:,.0f} tok/s")

    total_time = time.time() - t0
    final_loss, final_ppl, final_bpc = evaluate()
    print(f"\nTraining Complete in {total_time:.1f}s | Final PPL: {final_ppl:.2f} | Final BPC: {final_bpc:.3f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "final_ppl": final_ppl,
            "final_bpc": final_bpc
        }, save_path)
        print(f"Saved checkpoint to: {save_path}\n")

    return model


@torch.no_grad()
def profile_monosemanticity(
    model: TorosHybridLanguageModel,
    model_name: str,
    val_data: np.memmap,
    device: str = "cuda"
) -> Dict[str, Any]:
    model.eval()
    is_tree = (model.config.channel_mixer_type == "asdag_tree")
    num_layers = len(model.context_encoder.blocks)
    units_per_layer = 16 if is_tree else 84 # 16 leaves vs 84 neurons

    seq_len = 1024
    num_eval_chunks = 64
    eval_offsets = np.linspace(0, len(val_data) - seq_len - 1, num_eval_chunks, dtype=int)

    # Layer -> Unit -> list of (token_str, activation_value)
    unit_activations = {l: defaultdict(list) for l in range(num_layers)}

    for offset in eval_offsets:
        chunk_bytes = bytes(val_data[offset : offset + seq_len].tolist())
        byte_ids = torch.from_numpy(val_data[offset : offset + seq_len].astype(np.int64)).unsqueeze(0).to(device)

        # 1. Map patches to UTF-8 strings
        h_byte, boundary = model.context_encoder.byte_encoder(byte_ids)
        P = model.config.target_patch_size
        latent_patches, patch_assignments = model.context_encoder.patcher(
            h_byte, torch.zeros_like(boundary), fixed_patch_size=P
        )
        assignments = patch_assignments[0].cpu().numpy()
        M = latent_patches.shape[1]
        patch_texts = []
        for m in range(M):
            indices = np.where(assignments == m)[0]
            if len(indices) > 0:
                sub = chunk_bytes[indices[0] : indices[-1] + 1]
                s = sub.decode("utf-8", errors="replace").replace("\n", "\\n")
            else:
                s = ""
            patch_texts.append(s)

        # 2. Forward through layers and record unit activations
        curr_h = latent_patches
        for l_idx, block in enumerate(model.context_encoder.blocks):
            # Pre-norm and conv prefix
            h1 = block.norm1(curr_h)
            if block.use_conv_prefix and block.conv_prefix is not None:
                h1_pad = F.pad(h1, (0, 0, block.conv_kernel_size - 1, 0)).to(block.conv_prefix.weight.dtype)
                h1_conv = block.conv_prefix(h1_pad.transpose(1, 2)).transpose(1, 2).to(h1.dtype)
                h1 = block.conv_act(h1_conv)
            mix_out, _ = block.time_mixer(h1)
            curr_h = curr_h + mix_out

            h2 = block.norm2(curr_h)

            if is_tree:
                # ASDAG Leaf routing probabilities
                h2_flat = h2.reshape(-1, model.config.dim)
                probs, _ = block.channel_mixer.router.route_tokens(h2_flat)
                probs = probs.view(1, M, 16)
                top_vals, top_idx = torch.topk(probs, k=2, dim=-1)
                top1_leaf = top_idx[0, :, 0].cpu().numpy()
                top1_val = top_vals[0, :, 0].float().cpu().numpy()
                for m in range(M):
                    leaf_id = int(top1_leaf[m])
                    w = float(top1_val[m])
                    txt = patch_texts[m]
                    if txt.strip():
                        unit_activations[l_idx][leaf_id].append((txt, w))
            else:
                # Classic MLP hidden neuron activations: GELU(W1 * h2)
                # Shape: [1, M, 84]
                fc1 = block.channel_mixer.fc1
                act = block.channel_mixer.act
                h_mlp = act(fc1(h2))[0] # [M, 84]
                h_mlp_np = h_mlp.float().cpu().numpy()
                for m in range(M):
                    txt = patch_texts[m]
                    if not txt.strip():
                        continue
                    for u in range(84):
                        act_val = float(h_mlp_np[m, u])
                        unit_activations[l_idx][u].append((txt, act_val))

            curr_h = block(curr_h)

    # 3. Analyze Monosemanticity & Polysemanticity per unit
    layer_purities = []
    layer_entropies = []
    all_unit_purities = []
    all_unit_entropies = []
    unit_records = []
    high_purity_count = 0
    polysemantic_count = 0
    total_active_units = 0

    for l_idx in range(num_layers):
        l_purities = []
        l_entropies = []
        for u_id in range(units_per_layer):
            items = unit_activations[l_idx][u_id]
            if len(items) == 0:
                continue

            if is_tree:
                # For tree, all routed tokens are considered
                sample_tokens = [txt for txt, _ in items]
                raw_scores = [w for _, w in items]
            else:
                # For MLP neurons, sort by top activations (standard feature exemplar profiling)
                # Take top 15% highest activations for this neuron
                items.sort(key=lambda x: x[1], reverse=True)
                top_n = max(10, int(0.15 * len(items)))
                top_items = items[:top_n]
                sample_tokens = [txt for txt, _ in top_items]
                raw_scores = [val for _, val in top_items]

            # Categorical classification
            cat_counts = Counter([classify_text(txt) for txt in sample_tokens if classify_text(txt) != "Whitespace / Delimiter"])
            if not cat_counts:
                continue

            total_cat = sum(cat_counts.values())
            dominant_cat, dom_count = cat_counts.most_common(1)[0]
            purity = (dom_count / total_cat) * 100.0
            entropy = compute_shannon_entropy(cat_counts)

            l_purities.append(purity)
            l_entropies.append(entropy)
            all_unit_purities.append(purity)
            all_unit_entropies.append(entropy)
            total_active_units += 1

            if purity >= 70.0:
                high_purity_count += 1
            if purity < 50.0:
                polysemantic_count += 1

            # Get exemplar words
            word_freqs = Counter(sample_tokens)
            top_words = [w for w, _ in word_freqs.most_common(4)]
            exemplar_str = ", ".join([f'"{w}"' for w in top_words])

            unit_records.append({
                "layer": l_idx,
                "unit": u_id,
                "purity": purity,
                "entropy": entropy,
                "dominant_cat": dominant_cat,
                "exemplars": exemplar_str,
                "mean_act": float(np.mean(raw_scores)),
                "gini": compute_gini(np.array(raw_scores))
            })

        layer_purities.append(float(np.mean(l_purities)) if l_purities else 0.0)
        layer_entropies.append(float(np.mean(l_entropies)) if l_entropies else 0.0)

    mean_purity = float(np.mean(all_unit_purities)) if all_unit_purities else 0.0
    mean_entropy = float(np.mean(all_unit_entropies)) if all_unit_entropies else 0.0
    polysemantic_rate = (polysemantic_count / max(1, total_active_units)) * 100.0
    high_purity_rate = (high_purity_count / max(1, total_active_units)) * 100.0

    return {
        "model_name": model_name,
        "is_tree": is_tree,
        "total_active_units": total_active_units,
        "layer_purities": layer_purities,
        "layer_entropies": layer_entropies,
        "mean_purity": mean_purity,
        "mean_entropy": mean_entropy,
        "polysemantic_rate": polysemantic_rate,
        "high_purity_rate": high_purity_rate,
        "records": unit_records
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
    print("   ISO-SIZE MONOSEMANTICITY COMPARISON: CLASSIC DENSE MLP VS ASDAG SPARSE TREE")
    print(f"   Data Budget: 10.0 MB SimpleStories ({len(active_data):,} bytes)")
    print(f"   Train Set: {len(train_data)/(1024*1024):.2f} MB | Val Set: {len(val_data)/(1024*1024):.2f} MB | Device: {device}")
    print("=" * 105)

    # 1. Train Model A: Classic Dense MLP (252,640 params)
    cfg_mlp = TorosHybridConfig(
        dim=144,
        n_encoder_layers=6,
        channel_mixer_type="classic_mlp",
        mlp_hidden_dim=84,
        dtype=torch.bfloat16
    )
    mlp_model = train_model(
        model_name="Classic Dense MLP (252.6k)",
        cfg=cfg_mlp,
        train_data=train_data,
        val_data=val_data,
        steps=800,
        batch_size=64,
        seq_len=1024,
        save_path="checkpoints/toros_hybrid_250k_mlp_10mb.pt",
        device=device
    )

    # 2. Train Model B: ASDAG Tree on 10MB (253,138 params)
    cfg_tree_10mb = TorosHybridConfig(
        dim=144,
        n_encoder_layers=6,
        channel_mixer_type="asdag_tree",
        dtype=torch.bfloat16
    )
    asdag_10mb_model = train_model(
        model_name="ASDAG Sparse Tree (253.1k - 10MB)",
        cfg=cfg_tree_10mb,
        train_data=train_data,
        val_data=val_data,
        steps=800,
        batch_size=64,
        seq_len=1024,
        save_path="checkpoints/toros_hybrid_250k_asdag_10mb.pt",
        device=device
    )

    # 3. Load Model C: Pre-trained Full ASDAG Tree (2.24 GB)
    full_asdag_path = "models/toros_hybrid_250k_simplestories_lpc.pt"
    full_asdag_model = None
    if os.path.exists(full_asdag_path):
        print(f"Loading Full ASDAG Model (2.24 GB) from {full_asdag_path}...")
        ckpt = torch.load(full_asdag_path, map_location=device, weights_only=False)
        cfg_full = ckpt["config"]
        full_asdag_model = TorosHybridLanguageModel(cfg_full).to(device)
        full_asdag_model.enable_lpc(dtype=cfg_full.dtype, device=device)
        full_asdag_model.load_state_dict(ckpt["model_state_dict"])
        full_asdag_model.eval()

    # 4. Profile Monosemanticity
    print("\n" + "=" * 105)
    print("                 PROFILING MONOSEMANTICITY & MECHANISTIC INTERPRETABILITY")
    print("=" * 105)

    res_mlp = profile_monosemanticity(mlp_model, "Classic Dense MLP (10MB)", val_data, device=device)
    res_asdag_10mb = profile_monosemanticity(asdag_10mb_model, "ASDAG Sparse Tree (10MB)", val_data, device=device)
    res_asdag_full = profile_monosemanticity(full_asdag_model, "ASDAG Sparse Tree (2.24GB)", val_data, device=device) if full_asdag_model else None

    # 5. Print Results Table
    print("\n" + "=" * 105)
    print("                    COMPARATIVE MONOSEMANTICITY METRICS SUMMARY")
    print("=" * 105)
    header = f"{'Metric / Property':<35} | {'Classic MLP (10MB)':<22} | {'ASDAG Tree (10MB)':<22} | {'ASDAG Tree (2.24GB)':<22}"
    print(header)
    print("-" * 105)

    def f_val(res, key, fmt="{:.1f}%"):
        if res is None:
            return "N/A"
        val = res[key]
        return fmt.format(val)

    p_mlp = f_val(res_mlp, 'mean_purity')
    p_t10 = f_val(res_asdag_10mb, 'mean_purity')
    p_tfull = f_val(res_asdag_full, 'mean_purity') if res_asdag_full else "N/A"
    print(f"{'Mean Semantic Purity':<35} | {p_mlp:<22} | {p_t10:<22} | {p_tfull:<22}")

    h_mlp = f_val(res_mlp, 'high_purity_rate')
    h_t10 = f_val(res_asdag_10mb, 'high_purity_rate')
    h_tfull = f_val(res_asdag_full, 'high_purity_rate') if res_asdag_full else "N/A"
    print(f"{'High Purity Units (>= 70%)':<35} | {h_mlp:<22} | {h_t10:<22} | {h_tfull:<22}")

    poly_mlp = f_val(res_mlp, 'polysemantic_rate')
    poly_t10 = f_val(res_asdag_10mb, 'polysemantic_rate')
    poly_tfull = f_val(res_asdag_full, 'polysemantic_rate') if res_asdag_full else "N/A"
    print(f"{'Polysemantic Rate (< 50% purity)':<35} | {poly_mlp:<22} | {poly_t10:<22} | {poly_tfull:<22}")

    ent_mlp = f_val(res_mlp, 'mean_entropy', fmt="{:.2f} bits")
    ent_t10 = f_val(res_asdag_10mb, 'mean_entropy', fmt="{:.2f} bits")
    ent_tfull = f_val(res_asdag_full, 'mean_entropy', fmt="{:.2f} bits") if res_asdag_full else "N/A"
    print(f"{'Mean Shannon Entropy':<35} | {ent_mlp:<22} | {ent_t10:<22} | {ent_tfull:<22}")

    u_mlp = f"{res_mlp['total_active_units']} neurons"
    u_t10 = f"{res_asdag_10mb['total_active_units']} leaves"
    u_tfull = f"{res_asdag_full['total_active_units']} leaves" if res_asdag_full else "N/A"
    print(f"{'Total Evaluated Units':<35} | {u_mlp:<22} | {u_t10:<22} | {u_tfull:<22}")

    print("-" * 105)
    print("LAYER-BY-LAYER SEMANTIC PURITY (%):")
    for l in range(6):
        lp_mlp = f"{res_mlp['layer_purities'][l]:5.1f}%"
        lp_t10 = f"{res_asdag_10mb['layer_purities'][l]:5.1f}%"
        lp_tfull = f"{res_asdag_full['layer_purities'][l]:5.1f}%" if res_asdag_full else "N/A"
        print(f"  Layer {l}:{'':<25} | {lp_mlp:<22} | {lp_t10:<22} | {lp_tfull:<22}")

    print("\n" + "=" * 105)
    print("                    EXEMPLAR MONOSEMANTIC SPECIALISTS")
    print("=" * 105)
    print(">>> TOP MONOSEMANTIC LEAVES IN ASDAG TREE (10MB):")
    sorted_asdag = sorted(res_asdag_10mb["records"], key=lambda r: r["purity"], reverse=True)
    for r in sorted_asdag[:5]:
        print(f"  [L{r['layer']} Leaf {r['unit']:2d}] Purity: {r['purity']:5.1f}% | Entropy: {r['entropy']:.2f} | Category: {r['dominant_cat']:<25} | Exemplars: {r['exemplars']}")

    print("\n>>> TOP MONOSEMANTIC NEURONS IN CLASSIC MLP (10MB):")
    sorted_mlp = sorted(res_mlp["records"], key=lambda r: r["purity"], reverse=True)
    for r in sorted_mlp[:5]:
        print(f"  [L{r['layer']} Neuron {r['unit']:2d}] Purity: {r['purity']:5.1f}% | Entropy: {r['entropy']:.2f} | Category: {r['dominant_cat']:<25} | Exemplars: {r['exemplars']}")

    print("\n>>> MOST POLYSEMANTIC NEURONS IN CLASSIC MLP (SUPERPOSITION):")
    poly_mlp_records = sorted(res_mlp["records"], key=lambda r: r["purity"])
    for r in poly_mlp_records[:5]:
        print(f"  [L{r['layer']} Neuron {r['unit']:2d}] Purity: {r['purity']:5.1f}% | Entropy: {r['entropy']:.2f} | Category: {r['dominant_cat']:<25} | Exemplars: {r['exemplars']}")

    print("=" * 105)


if __name__ == "__main__":
    main()
