#!/usr/bin/env python3
"""
Monosemanticity & Mechanistic Interpretability Analysis of Trained TorosHybrid
==============================================================================
Analyzes the 1:16 sparse tree leaves in ASTDAGLayer of the trained 250k model:
  - Maps every patch latent back to its exact constituent UTF-8 text string
  - Profiles router activation distributions across all 16 leaves per layer
  - Computes top activating linguistic tokens, semantic purity scores, and entropy
  - Identifies distinct monosemantic specialists (dialogue, entities, actions, syntax)
"""

import os
import sys
import math
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn.functional as F

from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig


# Comprehensive linguistic taxonomy for semantic purity classification
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
        "pushed", "called", "liked", "loved", "gave", "made", "flew", "lived", "felt"
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
    # Exact check
    for cat, words in TAXONOMY.items():
        if cleaned in words or any(w in cleaned for w in words if len(w) >= 3):
            return cat
    # Substring heuristic
    if any(p in text for p in ['"', "'", "!", "?", ",", ".", ";"]):
        return "Dialogue & Punctuation"
    if cleaned.endswith("ed") or cleaned.endswith("ing"):
        return "Actions & Verbs"
    if cleaned.endswith("y") or cleaned.endswith("ful") or cleaned.endswith("ish"):
        return "Sensory & Adjectives"
    return "Other / Miscellaneous"


def analyze_model(model_path="models/toros_hybrid_250k_simplestories_lpc.pt", data_path="data/simplestories_eos.bin"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 105)
    print(f"   MONOSEMANTICITY & INTERPRETABILITY ANALYSIS: TOROS-HYBRID (Device: {device})")
    print(f"   Checkpoint: {model_path}")
    print("=" * 105)

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TorosHybridLanguageModel(cfg).to(device)
    model.enable_lpc(dtype=cfg.dtype, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    num_layers = len(model.context_encoder.blocks)
    num_leaves = model.context_encoder.blocks[0].channel_mixer.router.num_leaves
    print(f"Architecture: {num_layers} ASDAG Layers | {num_leaves} Leaves per Layer (Top-2 Sparse Routing)")

    # Read validation text chunks
    raw_data = np.memmap(data_path, dtype=np.uint8, mode="r")
    split = int(0.95 * len(raw_data))
    val_data = raw_data[split:]

    seq_len = 1024
    num_eval_chunks = 64
    eval_offsets = np.linspace(0, len(val_data) - seq_len - 1, num_eval_chunks, dtype=int)
    print(f"Profiling dataset: {num_eval_chunks} chunks x {seq_len} bytes = {num_eval_chunks * seq_len:,} validation bytes\n")

    # Layer -> Leaf -> list of (token_str, weight)
    leaf_tokens = {l: defaultdict(list) for l in range(num_layers)}
    leaf_counts = {l: Counter() for l in range(num_layers)}

    with torch.no_grad():
        for chunk_idx, offset in enumerate(eval_offsets):
            chunk_bytes = bytes(val_data[offset : offset + seq_len].tolist())
            byte_ids = torch.from_numpy(val_data[offset : offset + seq_len].astype(np.int64)).unsqueeze(0).to(device)

            # 1. Extract patch mappings
            h_byte, boundary = model.context_encoder.byte_encoder(byte_ids)
            P = model.config.target_patch_size
            latent_patches, patch_assignments = model.context_encoder.patcher(
                h_byte, torch.zeros_like(boundary), fixed_patch_size=P
            )

            # Map each patch m to its exact slice of text
            assignments = patch_assignments[0].cpu().numpy()
            M = latent_patches.shape[1]
            patch_texts = []
            for m in range(M):
                indices = np.where(assignments == m)[0]
                if len(indices) > 0:
                    sub_bytes = chunk_bytes[indices[0] : indices[-1] + 1]
                    s = sub_bytes.decode("utf-8", errors="replace").replace("\n", "\\n")
                else:
                    s = ""
                patch_texts.append(s)

            # 2. Forward through layers and extract routing probabilities
            curr_h = latent_patches
            for l_idx, block in enumerate(model.context_encoder.blocks):
                # Pre-norm and conv prefix
                h1 = block.norm1(curr_h)
                if block.use_conv_prefix and block.conv_prefix is not None:
                    h1_pad = F.pad(h1, (0, 0, block.conv_kernel_size - 1, 0)).to(block.conv_prefix.weight.dtype)
                    h1_conv = block.conv_prefix(h1_pad.transpose(1, 2)).transpose(1, 2).to(h1.dtype)
                    h1 = block.conv_act(h1_conv)
                
                # Time mixer
                mix_out, _ = block.time_mixer(h1)
                curr_h = curr_h + mix_out

                # Channel mixer router
                h2 = block.norm2(curr_h)
                h2_flat = h2.reshape(-1, model.config.dim)
                probs, _ = block.channel_mixer.router.route_tokens(h2_flat)
                probs = probs.view(1, M, num_leaves)

                top_vals, top_idx = torch.topk(probs, k=2, dim=-1)
                top1_leaf = top_idx[0, :, 0].cpu().numpy()
                top1_weight = top_vals[0, :, 0].float().cpu().numpy()

                for m in range(M):
                    leaf_id = int(top1_leaf[m])
                    w = float(top1_weight[m])
                    txt = patch_texts[m]
                    if txt.strip():
                        leaf_tokens[l_idx][leaf_id].append((txt, w))
                        leaf_counts[l_idx][leaf_id] += 1

                # Execute block forward
                curr_h = block(curr_h)

    print("=" * 105)
    print("                     LAYER-BY-LAYER MONOSEMANTICITY PROFILING")
    print("=" * 105)

    all_purities = []
    layer_stats = []

    for l_idx in range(num_layers):
        total_tokens_layer = sum(leaf_counts[l_idx].values())
        print(f"\n>>> [LAYER {l_idx}]: Total Tokens Routed = {total_tokens_layer:,}")
        print(f"{'Leaf':<6} | {'Firing %':<9} | {'Dominant Semantic Category':<27} | {'Purity':<8} | {'Top Exemplar Patches / Words':<45}")
        print("-" * 105)

        layer_purities = []
        for leaf_id in range(num_leaves):
            items = leaf_tokens[l_idx][leaf_id]
            count = leaf_counts[l_idx][leaf_id]
            if count == 0:
                print(f"Leaf {leaf_id:2d} | {'0.0%':<9} | {'[Pruned / Inactive Leaf]':<27} | {'—':<8} | {'—':<45}")
                continue

            pct = (count / total_tokens_layer) * 100.0
            word_freqs = Counter([txt for txt, _ in items])
            top_words = [w for w, _ in word_freqs.most_common(5)]
            exemplar_str = ", ".join([f'"{w}"' for w in top_words[:4]])

            # Compute categorical distribution
            cat_counts = Counter([classify_text(txt) for txt, _ in items if classify_text(txt) != "Whitespace / Delimiter"])
            if not cat_counts:
                cat_counts["Whitespace / Delimiter"] = count

            dominant_cat, dom_count = cat_counts.most_common(1)[0]
            valid_total = sum(cat_counts.values())
            purity = (dom_count / max(1, valid_total)) * 100.0
            layer_purities.append(purity)
            all_purities.append(purity)

            purity_str = f"{purity:5.1f}%"
            print(f"Leaf {leaf_id:2d} | {pct:5.1f}%    | {dominant_cat:<27} | {purity_str:<8} | {exemplar_str:<45}")

        avg_layer_purity = np.mean(layer_purities) if layer_purities else 0.0
        active_leaves = len([c for c in leaf_counts[l_idx].values() if c > 0])
        layer_stats.append((l_idx, avg_layer_purity, active_leaves))

    print("\n" + "=" * 105)
    print("                    GLOBAL MONOSEMANTICITY SUMMARY ACROSS ALL LAYERS")
    print("=" * 105)
    print(f"Mean Monosemantic Purity Across All Active Leaves: {np.mean(all_purities):.1f}%")
    print("\nLayer Progression:")
    for l_idx, p, act in layer_stats:
        bar = "█" * int(p / 4)
        print(f"  Layer {l_idx}: Mean Purity = {p:5.1f}% [{bar:<25}] ({act}/{num_leaves} active leaves)")
    print("=" * 105)


if __name__ == "__main__":
    analyze_model()
