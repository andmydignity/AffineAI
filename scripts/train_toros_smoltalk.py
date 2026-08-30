import os
import sys
import time
import math
import torch
import numpy as np
from affine_ai import TorosHybridLanguageModel, TorosHybridConfig

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print(f"          TOROS-HYBRID END-TO-END SMOLTALK TRAINING (Device: {device})")
    print("=" * 95)

    data_path = "data/smoltalk_eos.bin"
    file_size = os.path.getsize(data_path)
    print(f"Dataset File: {data_path} ({file_size:,d} bytes, {file_size / (1024*1024):.2f} MB)")

    data = np.memmap(data_path, dtype=np.uint8, mode="r")
    n_train = int(len(data) * 0.95)
    train_data = data[:n_train]
    val_data = data[n_train:]
    
    print(f"Train Set Size      : {len(train_data):,d} bytes ({len(train_data) / (1024*1024):.2f} MB)")
    print(f"Validation Set Size : {len(val_data):,d} bytes ({len(val_data) / (1024*1024):.2f} MB)")
    print("-" * 95)

    config = TorosHybridConfig(
        dim=136,
        d_byte=64,
        n_encoder_layers=6,
        n_predictor_layers=3,
        n_heads=4,
        target_patch_size=16,
        channel_mixer_type="ternary_swiglu",
        gen_loss_weight=1.0,
        dtype=torch.float32
    )

    model = TorosHybridLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model Architecture    : Toros-Hybrid (System 2 JEPA L=6 + System 1 BLT Decoder)")
    print(f"Total Pretraining Memory: {total_params:,d} parameters (~580k deployed inference)")
    print("-" * 95)

    batch_size = 16
    seq_len = 512
    target_shift = 64
    max_steps = 1500
    warmup_steps = 100
    eval_interval = 250
    base_lr = 3e-3

    def get_lr(step):
        if step < warmup_steps:
            return base_lr * (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses_gen, losses_jepa = [], []
        for _ in range(15):
            high = len(val_data) - seq_len - target_shift - 1
            ix = np.random.randint(0, high, size=batch_size)
            x = torch.from_numpy(np.stack([val_data[i : i + seq_len] for i in ix])).long().to(device)
            y = torch.from_numpy(np.stack([val_data[i + 1 : i + seq_len + 1] for i in ix])).long().to(device)
            _, _, metrics = model(x, targets=y, target_shift=target_shift)
            losses_gen.append(metrics["loss_gen"])
            losses_jepa.append(metrics["loss_jepa"])
        model.train()
        mean_gen = float(np.mean(losses_gen))
        mean_jepa = float(np.mean(losses_jepa))
        ppl = math.exp(min(mean_gen, 20.0))
        bpc = mean_gen / math.log(2)
        return mean_gen, mean_jepa, ppl, bpc

    best_val_ppl = 999.0
    save_path = "models/toros_hybrid_smoltalk_best_clean.pt"
    t0 = time.time()

    for step in range(1, max_steps + 1):
        step_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = step_lr

        high = len(train_data) - seq_len - target_shift - 1
        ix = np.random.randint(0, high, size=batch_size)
        x = torch.from_numpy(np.stack([train_data[i : i + seq_len] for i in ix])).long().to(device)
        y = torch.from_numpy(np.stack([train_data[i + 1 : i + seq_len + 1] for i in ix])).long().to(device)

        t_step_0 = time.time()
        optimizer.zero_grad()
        logits, loss, metrics = model(x, targets=y, target_shift=target_shift)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_step = time.time() - t_step_0

        if step % eval_interval == 0 or step == 1:
            val_gen, val_jepa, val_ppl, val_bpc = evaluate()
            elapsed = time.time() - t0
            speed = (batch_size * seq_len) / max(1e-6, t_step)
            print(f"Step {step:4d}/{max_steps} | Val Loss: {val_gen:.4f} | Val PPL: {val_ppl:>6.2f} (BPC: {val_bpc:.3f}) | JEPA Loss: {val_jepa:.4f} | {speed:>7,.0f} B/s | Elapsed: {elapsed:>5.1f}s")
            
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                model.save_inference_checkpoint(save_path, metadata={"best_val_ppl": best_val_ppl, "step": step})
                print(f"  --> Saved clean scaffolding-free checkpoint to {save_path} (Val PPL: {best_val_ppl:.2f})")

    total_time = time.time() - t0
    avg_throughput = (max_steps * batch_size * seq_len) / total_time
    print("=" * 95)
    print(f"Training Complete in {total_time:.2f}s ({avg_throughput:,.0f} bytes/sec on {device})!")
    print(f"Best Validation Perplexity (PPL): {best_val_ppl:.2f}")
    print("=" * 95)

    prompts = [
        "<|im_start|>user\nHello! How are you today?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nCan you give me advice on learning programming?<|im_end|>\n<|im_start|>assistant\n"
    ]

    print("\nSAMPLE CONVERSATIONAL CHAT GENERATIONS (Stopping at <|im_end|> / <|endoftext|>):")
    print("=" * 95)
    for prompt_text in prompts:
        prompt_bytes = torch.tensor([[ord(c) for c in prompt_text]], dtype=torch.long, device=device)
        out = model.generate_with_latent_planning(
            prompt_bytes, max_new_bytes=200, plan_steps=3, temperature=0.7, eos_byte=0
        )
        gen_bytes = out[0].tolist()
        
        decoded = ""
        for b in gen_bytes:
            if b == 0:
                decoded += " <|endoftext|>"
                break
            decoded += chr(b) if 32 <= b <= 126 or b in (10, 13) else f"\\x{b:02x}"
            
        print(f"Prompt:\n{prompt_text}")
        print(f"Response:\n{decoded[len(prompt_text):]}\n" + "-" * 80)

if __name__ == "__main__":
    main()
