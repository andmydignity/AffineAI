#!/usr/bin/env python3
"""
Test sentence generation with upcycled Qwen3.5 ASDAG model.
Loads layers sequentially with minimal memory footprint (< 8.5 GB RAM).
"""

import os
import gc
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from affine_ai.models.qwen35_asdag import Qwen35ASDAGConfig, Qwen35Block, Qwen35RMSNorm


def load_model(checkpoints_dir="checkpoints/qwen35_pot5"):
    print(f">>> Loading Qwen3.5 ASDAG Model from {checkpoints_dir}...")
    config = Qwen35ASDAGConfig()

    print("  Loading token embeddings...")
    token_embd = torch.load(os.path.join(checkpoints_dir, "token_embd.pt"), map_location="cpu", weights_only=True)

    print("  Loading output norm...")
    output_norm = Qwen35RMSNorm(config.dim, eps=config.rms_norm_eps)
    norm_st = torch.load(os.path.join(checkpoints_dir, "output_norm.pt"), map_location="cpu", weights_only=True)
    if isinstance(norm_st, torch.Tensor):
        output_norm.weight.data.copy_(norm_st)
    else:
        output_norm.load_state_dict(norm_st)
    del norm_st

    blocks = []
    print(f"  Loading {config.num_layers} blocks...")
    t0 = time.time()
    for i in range(config.num_layers):
        b = Qwen35Block(config, layer_idx=i)
        st = torch.load(os.path.join(checkpoints_dir, f"block_{i:02d}.pt"), map_location="cpu", weights_only=True)
        b.load_state_dict(st)
        del st
        b.eval()
        blocks.append(b)
        if (i + 1) % 8 == 0 or i == config.num_layers - 1:
            print(f"    Loaded block {i+1}/{config.num_layers} ({time.time()-t0:.1f}s)")

    gc.collect()
    print(">>> Model successfully loaded!\n")
    return config, token_embd, output_norm, blocks


@torch.no_grad()
def generate(prompt: str, tokenizer, config, token_embd, output_norm, blocks, max_new_tokens: int = 15, temperature: float = 0.7, top_p: float = 0.9):
    print(f"--- Prompt: \"{prompt}\" ---")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")  # [1, L]
    L = input_ids.shape[1]

    # 1. Prefill prompt
    print("Prefilling prompt...")
    t_start = time.time()
    x = F.embedding(input_ids, token_embd)  # [1, L, D]
    states = []

    for i, block in enumerate(blocks):
        x, st = block(x, state=None, pos=0)
        states.append(st)

    # Logits for last token
    h = output_norm(x[:, -1:])  # [1, 1, D]
    logits = F.linear(h.float(), token_embd.float()).squeeze(1)  # [1, vocab_size]

    # Predict first token
    top5_probs, top5_ids = torch.topk(F.softmax(logits, dim=-1), 5)
    print("Top 5 candidates for next token:")
    for p, idx in zip(top5_probs[0], top5_ids[0]):
        tok_str = tokenizer.decode([idx.item()])
        print(f"  {tok_str!r:15s} (p = {p.item()*100:.2f}%, id = {idx.item()})")

    next_token = top5_ids[0, 0].view(1, 1)
    generated_tokens = [next_token.item()]
    print(f"\nFirst generated token: {tokenizer.decode([next_token.item()])!r} (prefill time: {time.time()-t_start:.2f}s)")

    # 2. Autoregressive decoding
    curr_token = next_token
    curr_pos = L

    print("Generating autoregressively:")
    sys.stdout.write(prompt + tokenizer.decode([next_token.item()]))
    sys.stdout.flush()

    for step in range(max_new_tokens - 1):
        t0 = time.time()
        x = F.embedding(curr_token, token_embd)  # [1, 1, D]

        next_states = []
        for i, block in enumerate(blocks):
            x, st = block(x, state=states[i], pos=curr_pos)
            next_states.append(st)
        states = next_states

        h = output_norm(x[:, -1:])  # [1, 1, D]
        logits = F.linear(h.float(), token_embd.float()).squeeze(1)  # [1, vocab_size]

        # Greedy / sampling
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            # Top-p filtering
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            curr_token = torch.multinomial(probs, num_samples=1)
        else:
            curr_token = logits.argmax(dim=-1, keepdim=True)

        tok_id = curr_token.item()
        generated_tokens.append(tok_id)
        curr_pos += 1

        decoded_piece = tokenizer.decode([tok_id])
        sys.stdout.write(decoded_piece)
        sys.stdout.flush()

        if tok_id == tokenizer.eos_token_id:
            print("\n[EOS reached]")
            break

    full_output = tokenizer.decode(input_ids[0].tolist() + generated_tokens)
    print(f"\n\nFull Output:\n{full_output}\n")
    return full_output


if __name__ == "__main__":
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained("./")
    config, token_embd, output_norm, blocks = load_model()

    # Test prompt
    generate("The capital of France is", tokenizer, config, token_embd, output_norm, blocks, max_new_tokens=8, temperature=0.0)
