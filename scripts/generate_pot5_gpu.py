#!/usr/bin/env python3
"""
High-Speed GPU Generation Engine for 5-State POT Qwen3.5-4B ASDAG Model.
Uses 3-bitplane Triton kernels directly in GPU VRAM (1.35 GB total VRAM).
Zero disk I/O during generation.
"""

import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from affine_ai.models.qwen35_asdag import Qwen35ASDAGConfig, Qwen35RMSNorm
from affine_ai.kernels.triton_pot5 import pack_pot5_gpu_3bitplane, triton_pot5_bitpacked_linear


class FastBitpackedLinear(nn.Module):
    """3-bitplane POT5 Linear Layer on GPU."""
    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.in_features = w.shape[1]
        self.out_features = w.shape[0]
        # Pack on CPU to eliminate VRAM allocation spikes during loading
        w_nz, w_mag, w_sign, alpha, _ = pack_pot5_gpu_3bitplane(w.cpu())
        self.register_buffer("w_nz", w_nz.cuda())
        self.register_buffer("w_mag", w_mag.cuda())
        self.register_buffer("w_sign", w_sign.cuda())
        self.register_buffer("alpha", alpha.cuda())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_pot5_bitpacked_linear(
            x, self.w_nz, self.w_mag, self.w_sign, self.alpha, self.in_features
        )


class FastPOT5Block(nn.Module):
    """Complete Qwen 3.5 Block with 3-bitplane POT5 GPU Linear Layers."""
    def __init__(self, config: Qwen35ASDAGConfig, layer_idx: int, state_dict: dict):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_full_attention = (layer_idx == 32) or ((layer_idx + 1) % config.full_attn_interval == 0)

        # Norms
        self.attn_norm = Qwen35RMSNorm(config.dim, eps=config.rms_norm_eps).cuda()
        self.attn_norm.weight.data.copy_(state_dict["attn_norm.weight"].cuda())

        self.post_attention_norm = Qwen35RMSNorm(config.dim, eps=config.rms_norm_eps).cuda()
        self.post_attention_norm.weight.data.copy_(state_dict["post_attention_norm.weight"].cuda())

        # FFN: Pack all 3 projections into 3-bitplanes
        if "asdag_ffn.gate_proj.weight" in state_dict:
            gate_w = state_dict["asdag_ffn.gate_proj.weight"]
            up_w = state_dict["asdag_ffn.up_proj.weight"]
            down_w = state_dict["asdag_ffn.down_proj.weight"]
        else:
            num_leaves = config.num_leaves
            gate_w = torch.cat([state_dict[f"asdag_ffn.leaves.{i}.gate_proj.weight"] for i in range(num_leaves)], dim=0)
            up_w = torch.cat([state_dict[f"asdag_ffn.leaves.{i}.up_proj.weight"] for i in range(num_leaves)], dim=0)
            down_w = torch.cat([state_dict[f"asdag_ffn.leaves.{i}.down_proj.weight"] for i in range(num_leaves)], dim=1)
        self.gate_proj = FastBitpackedLinear(gate_w)
        self.up_proj = FastBitpackedLinear(up_w)
        self.down_proj = FastBitpackedLinear(down_w)

        # Time Mixer in native BF16 (unquantized)
        if self.is_full_attention:
            self.q_heads = config.attn_q_heads
            self.kv_heads = config.attn_kv_heads
            self.head_dim = config.attn_head_dim
            self.rope_dim = config.rope_dim
            self.out_dim = self.q_heads * self.head_dim

            self.attn_q = nn.Linear(config.dim, self.out_dim * 2, bias=False, dtype=config.dtype).cuda()
            self.attn_q.weight.data.copy_(state_dict["time_mixer.attn_q.weight"].cuda())
            self.attn_k = nn.Linear(config.dim, self.kv_heads * self.head_dim, bias=False, dtype=config.dtype).cuda()
            self.attn_k.weight.data.copy_(state_dict["time_mixer.attn_k.weight"].cuda())
            self.attn_v = nn.Linear(config.dim, self.kv_heads * self.head_dim, bias=False, dtype=config.dtype).cuda()
            self.attn_v.weight.data.copy_(state_dict["time_mixer.attn_v.weight"].cuda())
            self.attn_output = nn.Linear(self.out_dim, config.dim, bias=False, dtype=config.dtype).cuda()
            self.attn_output.weight.data.copy_(state_dict["time_mixer.attn_output.weight"].cuda())

            self.q_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps).cuda()
            self.q_norm.weight.data.copy_(state_dict["time_mixer.q_norm.weight"].cuda())
            self.k_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps).cuda()
            self.k_norm.weight.data.copy_(state_dict["time_mixer.k_norm.weight"].cuda())
        else:
            self.v_heads = config.ssm_v_heads
            self.qk_heads = config.ssm_qk_heads
            self.head_dim = config.ssm_head_dim
            self.out_dim = self.v_heads * self.head_dim

            self.qkv_proj = nn.Linear(config.dim, 8192, bias=False, dtype=config.dtype).cuda()
            self.qkv_proj.weight.data.copy_(state_dict["time_mixer.qkv_proj.weight"].cuda())
            self.attn_gate = nn.Linear(config.dim, self.out_dim, bias=False, dtype=config.dtype).cuda()
            self.attn_gate.weight.data.copy_(state_dict["time_mixer.attn_gate.weight"].cuda())
            self.ssm_out = nn.Linear(self.out_dim, config.dim, bias=False, dtype=config.dtype).cuda()
            self.ssm_out.weight.data.copy_(state_dict["time_mixer.ssm_out.weight"].cuda())

            # Conv1D and vector parameters
            self.conv1d = nn.Conv1d(
                in_channels=8192, out_channels=8192, kernel_size=4, groups=8192,
                padding=0, bias=False, dtype=config.dtype
            ).cuda()
            self.conv1d.weight.data.copy_(state_dict["time_mixer.conv1d.weight"].cuda())

            self.alpha_proj = nn.Linear(config.dim, self.v_heads, bias=False, dtype=torch.float32).cuda()
            self.alpha_proj.weight.data.copy_(state_dict["time_mixer.alpha_proj.weight"].cuda())

            self.beta_proj = nn.Linear(config.dim, self.v_heads, bias=False, dtype=torch.float32).cuda()
            self.beta_proj.weight.data.copy_(state_dict["time_mixer.beta_proj.weight"].cuda())

            self.ssm_a = nn.Parameter(state_dict["time_mixer.ssm_a"].cuda())
            self.ssm_dt_bias = nn.Parameter(state_dict["time_mixer.ssm_dt_bias"].cuda())

            self.norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps).cuda()
            self.norm.weight.data.copy_(state_dict["time_mixer.norm.weight"].cuda())

    def _apply_rope(self, x: torch.Tensor, pos: int = 0) -> torch.Tensor:
        B, T, H, D = x.shape
        x_rope = x[..., :self.rope_dim]
        x_pass = x[..., self.rope_dim:]
        positions = torch.arange(pos, pos + T, device=x.device, dtype=torch.float32)
        dim_idx = torch.arange(0, self.rope_dim, 2, device=x.device, dtype=torch.float32)
        inv_freq = 1.0 / (self.config.rope_theta ** (dim_idx / self.rope_dim))
        sinusoid = torch.outer(positions, inv_freq)
        sin = sinusoid.sin().repeat_interleave(2, dim=-1)[None, :, None, :]
        cos = sinusoid.cos().repeat_interleave(2, dim=-1)[None, :, None, :]
        x1 = x_rope[..., 0::2]
        x2 = x_rope[..., 1::2]
        x_rot = torch.cat([-x2, x1], dim=-1)
        x_rope_out = (x_rope.float() * cos + x_rot.float() * sin).to(x.dtype)
        return torch.cat([x_rope_out, x_pass], dim=-1)

    def forward(self, x: torch.Tensor, state=None, pos: int = 0):
        # 1. Time mixer
        res = x
        normed = self.attn_norm(x)
        B, T, D = normed.shape

        if self.is_full_attention:
            q_proj = self.attn_q(normed)
            q_reshaped = q_proj.view(B, T, self.q_heads, 2, self.head_dim)
            q = q_reshaped[:, :, :, 0, :]
            gate = q_reshaped[:, :, :, 1, :].reshape(B, T, self.out_dim)

            k = self.attn_k(normed).reshape(B, T, self.kv_heads, self.head_dim)
            v = self.attn_v(normed).reshape(B, T, self.kv_heads, self.head_dim)

            q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(B, T, self.q_heads, self.head_dim)
            k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(B, T, self.kv_heads, self.head_dim)

            q = self._apply_rope(q, pos=pos)
            k = self._apply_rope(k, pos=pos)

            if state is not None:
                past_k, past_v = state
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)
            next_state = (k.detach(), v.detach())

            k_rep = torch.repeat_interleave(k, self.q_heads // self.kv_heads, dim=2)
            v_rep = torch.repeat_interleave(v, self.q_heads // self.kv_heads, dim=2)

            q_t = q.transpose(1, 2)
            k_t = k_rep.transpose(1, 2)
            v_t = v_rep.transpose(1, 2)

            attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=(T > 1))
            attn_out = attn_out.transpose(1, 2).reshape(B, T, self.out_dim)
            tm_out = self.attn_output(attn_out * torch.sigmoid(gate))
        else:
            qkv = self.qkv_proj(normed)
            qkv_t = qkv.transpose(1, 2)

            conv_st = state[0] if state is not None else None
            ssm_st = state[1] if state is not None else None

            if conv_st is not None:
                conv_in = torch.cat([conv_st, qkv_t], dim=-1)
            else:
                conv_in = F.pad(qkv_t, (self.config.ssm_conv_kernel - 1, 0))

            next_conv_st = conv_in[:, :, -(self.config.ssm_conv_kernel - 1):].detach()
            conv_out = self.conv1d(conv_in)[:, :, :T]
            conv_act = F.silu(conv_out).transpose(1, 2)

            q = conv_act[:, :, :2048].reshape(B, T, 16, 128)
            k = conv_act[:, :, 2048:4096].reshape(B, T, 16, 128)
            v = conv_act[:, :, 4096:].reshape(B, T, 32, 128)

            q = q / (q.norm(dim=-1, keepdim=True) + self.config.rms_norm_eps)
            k = k / (k.norm(dim=-1, keepdim=True) + self.config.rms_norm_eps)
            q = q / math.sqrt(128.0)

            q = torch.repeat_interleave(q, 2, dim=2)
            k = torch.repeat_interleave(k, 2, dim=2)

            normed_f = normed.float()
            beta = torch.sigmoid(self.beta_proj(normed_f))
            alpha_biased = self.alpha_proj(normed_f) + self.ssm_dt_bias
            decay = torch.exp(F.softplus(alpha_biased) * self.ssm_a)

            if ssm_st is None:
                state_S = torch.zeros(B, 32, 128, 128, device=x.device, dtype=torch.float32)
            else:
                state_S = ssm_st.clone().float()

            q_f = q.float()
            k_f = k.float()
            v_f = v.float()

            outs = []
            for t in range(T):
                qt = q_f[:, t]
                kt = k_f[:, t]
                vt = v_f[:, t]
                dt = decay[:, t, :, None, None]
                bt = beta[:, t, :, None]

                state_S = state_S * dt
                sk = torch.matmul(kt.unsqueeze(-2), state_S).squeeze(-2)
                err = (vt - sk) * bt
                state_S = state_S + torch.matmul(kt.unsqueeze(-1), err.unsqueeze(-2))
                yt = torch.matmul(qt.unsqueeze(-2), state_S).squeeze(-2)
                outs.append(yt.unsqueeze(1))

            out_ssm = torch.cat(outs, dim=1)
            out_ssm = self.norm(out_ssm.reshape(-1, 128)).reshape(B, T, 4096).to(x.dtype)
            tm_out = self.ssm_out(out_ssm * F.silu(self.attn_gate(normed)))
            next_state = (next_conv_st, state_S)

        x = res + tm_out

        # 2. ASDAG FFN
        res_ffn = x
        normed2 = self.post_attention_norm(x)
        gate = self.gate_proj(normed2)
        up = self.up_proj(normed2)
        act = F.silu(gate) * up
        out_ffn = self.down_proj(act)
        x = res_ffn + out_ffn

        return x, next_state


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="5-State POT GPU High-Speed Generator")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?", help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling threshold")
    parser.add_argument("--repetition-penalty", type=float, default=1.15, help="Repetition penalty (> 1.0)")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/qwen35_pot5", help="Checkpoint directory")
    parser.add_argument("--raw", action="store_true", help="Do not wrap prompt in chat template")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("ERROR: CUDA GPU is required for high-speed bitpacked generation.")
        sys.exit(1)

    print("=================================================================")
    print("  AffineAI 5-State POT GPU Generation Engine (3-Bitplane Triton)  ")
    print("=================================================================")

    tokenizer = AutoTokenizer.from_pretrained("./")
    config = Qwen35ASDAGConfig()
    config.dtype = torch.bfloat16

    print(f"\n[1/3] Loading Token Embeddings to CPU...")
    token_embd = torch.load(os.path.join(args.ckpt_dir, "token_embd.pt"), map_location="cpu", weights_only=True).to(torch.bfloat16)
    # Pre-transpose for fast LM head GEMV
    token_embd_t = token_embd.float().T.contiguous()

    print(f"[2/3] Packing All 32 Blocks into GPU VRAM (3-bitplane format)...")
    t0_load = time.time()
    blocks = []
    for i in range(config.num_layers):
        st = torch.load(os.path.join(args.ckpt_dir, f"block_{i:02d}.pt"), map_location="cpu", weights_only=True)
        block = FastPOT5Block(config, layer_idx=i, state_dict=st)
        block.eval()
        blocks.append(block)
        del st
        if (i + 1) % 8 == 0 or i == config.num_layers - 1:
            torch.cuda.empty_cache()
            vram_mb = torch.cuda.memory_allocated() / 1e6
            print(f"  Loaded Block {i+1:02d}/32 -> GPU VRAM: {vram_mb:.1f} MB ({time.time()-t0_load:.1f}s)")

    output_norm = Qwen35RMSNorm(config.dim, eps=config.rms_norm_eps).cuda()
    norm_st = torch.load(os.path.join(args.ckpt_dir, "output_norm.pt"), map_location="cuda", weights_only=True)
    output_norm.weight.data.copy_(norm_st if isinstance(norm_st, torch.Tensor) else norm_st["weight"])
    output_norm.eval()

    total_vram = torch.cuda.memory_allocated() / 1e6
    print(f"\n[OK] Model Ready in GPU VRAM: {total_vram:.1f} MB (Total load time: {time.time()-t0_load:.1f}s)")

    # Prepare prompt
    if not args.raw:
        messages = [{"role": "user", "content": args.prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = args.prompt

    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")
    L = input_ids.shape[1]

    print(f"\n--- Prompt: {args.prompt!r} ({L} tokens) ---")
    print("\n--- Generating ---")

    # Prefill Prompt
    t_prefill = time.time()
    x = token_embd[input_ids[0]].unsqueeze(0).cuda()  # [1, L, D]
    states = []
    for b in blocks:
        x, st = b(x, state=None, pos=0)
        states.append(st)

    h = output_norm(x[:, -1:])
    h_cpu = h.squeeze(1).cpu().float()
    logits = torch.matmul(h_cpu, token_embd_t)

    prefill_time = time.time() - t_prefill
    prefill_tok_per_sec = L / max(1e-6, prefill_time)

    # First token (suppress immediate EOS)
    for eos_id in [tokenizer.eos_token_id, 151643, 151645]:
        if eos_id is not None:
            logits[0, eos_id] = -float('inf')

    if args.temperature <= 0.0:
        next_tok = logits.argmax(dim=-1).item()
    else:
        probs = F.softmax(logits / args.temperature, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1).item()

    generated_ids = [next_tok]
    piece = tokenizer.decode([next_tok])
    sys.stdout.write(piece)
    sys.stdout.flush()

    curr_token = next_tok
    curr_pos = L

    # Autoregressive Loop
    t_gen_start = time.time()
    tokens_generated = 1

    for step in range(args.max_new_tokens - 1):
        x = token_embd[curr_token].view(1, 1, -1).cuda()
        next_states = []
        for i, b in enumerate(blocks):
            x, st = b(x, state=states[i], pos=curr_pos)
            next_states.append(st)
        states = next_states

        h = output_norm(x[:, -1:])
        h_cpu = h.squeeze(1).cpu().float()
        logits = torch.matmul(h_cpu, token_embd_t)

        # Repetition penalty
        if args.repetition_penalty > 1.0:
            for prev_tok in set(input_ids[0].tolist() + generated_ids):
                if logits[0, prev_tok] > 0:
                    logits[0, prev_tok] /= args.repetition_penalty
                else:
                    logits[0, prev_tok] *= args.repetition_penalty

        if step < 12:
            for eos_id in [tokenizer.eos_token_id, 151643, 151645]:
                if eos_id is not None:
                    logits[0, eos_id] = -float('inf')

        if args.temperature <= 0.0:
            curr_token = logits.argmax(dim=-1).item()
        else:
            probs = F.softmax(logits / args.temperature, dim=-1)
            # Top-p filtering
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > args.top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = 0
            indices_to_remove = mask.scatter(1, sorted_indices, mask)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            curr_token = torch.multinomial(probs, num_samples=1).item()

        tokens_generated += 1
        curr_pos += 1
        generated_ids.append(curr_token)

        piece = tokenizer.decode([curr_token])
        sys.stdout.write(piece)
        sys.stdout.flush()

        if curr_token == tokenizer.eos_token_id or piece in ["<|im_end|>", "<|endoftext|>"]:
            break

    gen_time = time.time() - t_gen_start
    tok_per_sec = tokens_generated / max(1e-6, gen_time)

    print("\n\n-----------------------------------------------------------------")
    print(f"Speed: {tok_per_sec:.2f} tok/s | Latency: {1000/tok_per_sec:.1f} ms/token")
    print(f"Prefill: {L} tokens in {prefill_time:.3f}s ({prefill_tok_per_sec:.1f} tok/s)")
    print(f"Total Tokens Generated: {tokens_generated}")
    print("-----------------------------------------------------------------")


if __name__ == "__main__":
    main()
