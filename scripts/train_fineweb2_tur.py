#!/usr/bin/env python3
"""
FineWeb2-HQ Turkish (tur_Latn) PoC — byte-level ASDAG with live HF streaming.

Dataset: epfml/FineWeb2-HQ, subset tur_Latn (8.58M docs, ~100GB parquet).
No local download: streams directly via datasets + HFStreamDataLoader
(static [B,T] byte batches, CUDA-Graph-safe). No tokenizer needed.

Usage:
  python scripts/train_fineweb2_tur.py                          # defaults: 30M shakedown, L4/CUDA auto
  python scripts/train_fineweb2_tur.py --dim 768 --layers 10 --batch 32 --seq 1024 --steps 6000
  python scripts/train_fineweb2_tur.py --quality-min 0.5        # filter low quality_score docs
"""
import argparse
import math
import time
import torch
from datasets import load_dataset
from affine_ai import ASDAGLanguageModel
from affine_ai.data import HFStreamDataLoader
from affine_ai.training.trainer import ASDAGTrainer


def parse_args():
    p = argparse.ArgumentParser(description="FineWeb2-HQ tur_Latn byte-level ASDAG")
    p.add_argument("--dim", type=int, default=512, help="latent dim (512 shakedown, 768-1024 main)")
    p.add_argument("--layers", type=int, default=6, help="latent layers (6 shakedown, 10-12 main)")
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mixer", type=str, default="asdag_tree", choices=["asdag_tree", "ternary_swiglu"], help="channel mixer")
    p.add_argument("--batch", type=int, default=32, help="batch B (24GB fits 32-48 at T1024)")
    p.add_argument("--seq", type=int, default=1024, help="sequence length in bytes (patch P=16)")
    p.add_argument("--steps", type=int, default=3000, help="training steps (tokens = steps*B*T)")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--quality-min", type=float, default=None, help="drop docs with quality_score < threshold")
    p.add_argument("--max-rows", type=int, default=None, help="cap stream after N docs (debug)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", type=str, default="checkpoints/fineweb2_tur_best.pt")
    p.add_argument("--eval-interval", type=int, default=None, help="eval every N steps (default max(200, steps//10))")
    p.add_argument("--log-interval", type=int, default=1, help="print train loss/ppl/speed every N steps")
    p.add_argument("--bias-rate", "--balance-w", dest="bias_rate", type=float, default=1e-3, help="DeepSeek expert bias update rate (0 disables)")
    return p.parse_args()


def make_stream(quality_min=None, max_rows=None):
    # HF streaming: returns an IterableDataset; each item has fields including `text`, `quality_score`
    ds = load_dataset("epfml/FineWeb2-HQ", "tur_Latn", split="train", streaming=True)

    if quality_min is not None or max_rows is not None:
        def _gen():
            n = 0
            for ex in ds:
                if quality_min is not None and ex.get("quality_score", 1.0) < quality_min:
                    continue
                yield ex
                n += 1
                if max_rows is not None and n >= max_rows:
                    break
        return _gen()

    # cap for quick debug without full 8.5M scan
    if max_rows is not None:
        # fallback (already handled above) — kept for exhaustiveness
        pass
    return ds


def main():
    args = parse_args()
    device = args.device
    print(f"Device: {device} | dim={args.dim} L={args.layers} mixer={args.mixer}")
    print(f"Batch {args.batch} x seq {args.seq} x steps {args.steps} = {args.steps * args.batch * args.seq / 1e9:.2f}B bytes")
    print(f"Dataset: epfml/FineWeb2-HQ tur_Latn (streaming, text_field='text')")
    print(f"Context 1M, gen 20K, 1:16 sparsity, dynamic patching OFF (your spec)")

    # Two independent streams so train/val don't share iterator state
    train_src = make_stream(args.quality_min, args.max_rows)
    val_src = make_stream(args.quality_min, None)

    train_loader = HFStreamDataLoader(train_src, batch_size=args.batch, seq_len=args.seq, text_field="text", device=device, restart=True)
    val_loader = HFStreamDataLoader(val_src, batch_size=args.batch, seq_len=args.seq, text_field="text", device=device, restart=True)

    model = ASDAGLanguageModel(
        vocab_size=256, d_model=args.dim, n_layers=args.layers, n_heads=args.heads,
        channel_mixer_type=args.mixer, dtype=torch.bfloat16, use_blt=True, d_byte=min(128, args.dim),
        target_patch_size=16,
        sparsity_ratio=0.9375, num_leaves=16, top_k=2, leaf_mode="permutation",
    ).to(device)
    # your spec: 1M context / 20K gen — set on hybrid config (ASDAGLanguageModel doesn't take it in __init__)
    if hasattr(model, "hybrid") and model.hybrid is not None:
        model.hybrid.config.context_window = 1_000_000
        model.hybrid.config.max_seq_len = 1_000_000
    # disable dynamic "dictionary for hard tokens" machinery explicitly
    if hasattr(model, "hybrid") and model.hybrid is not None:
        for k in ("use_growth", "use_bmr", "use_info_gain", "use_type_codebook", "dynamic_patching"):
            if hasattr(model.hybrid.config, k):
                setattr(model.hybrid.config, k, False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params ({args.mixer})")
    if args.bias_rate and args.bias_rate > 0:
        from affine_ai.core.ast_dag import set_expert_bias_rate
        n_hit = set_expert_bias_rate(model, args.bias_rate)
        print(f"DeepSeek expert bias ON: rate={args.bias_rate} applied to {n_hit} ASTDAG layers")

    trainer = ASDAGTrainer(
        model=model,
        train_data=train_loader,
        val_data=val_loader,
        batch_size=args.batch,
        seq_len=args.seq,
        lr=args.lr,
        max_steps=args.steps,
        eval_interval=args.eval_interval if args.eval_interval is not None else max(200, args.steps // 10),
        eval_iters=20,
        use_lpc=True,
        use_muon=True,
        device=device,
    )

    # per-step logging + eval, screen-safe (flush every line)
    import sys as _sys
    best_val = float("inf")
    t0 = time.time()
    log_n = max(1, args.log_interval)
    for step in range(args.steps):
        t_step0 = time.time()
        train_loss = trainer.train_step(step)
        t_step = time.time() - t_step0
        tok_s = (args.batch * args.seq) / max(1e-6, t_step)
        # train ppl/bpc from loss (bf16-safe)
        try:
            tl = float(train_loss) if not hasattr(train_loss, "item") else float(train_loss.item())
        except Exception:
            tl = float(train_loss)
        ppl = math.exp(min(tl, 20.0))
        bpc = tl / math.log(2)
        if (step + 1) % log_n == 0 or step == 0:
            print(f"step {step+1:6d}/{args.steps} | loss {tl:.4f} ppl {ppl:6.1f} bpc {bpc:.3f} | {tok_s:,.0f} tok/s {t_step*1000:.0f}ms/step", flush=True)
        do_eval = (step + 1) % trainer.eval_interval == 0 or step == args.steps - 1
        if do_eval:
            em = trainer.evaluate()
            vl, vppl, vbpc = em["val_loss"], em["val_ppl"], em["val_bpc"]
            print(f"  -> eval val_loss {vl:.4f} ppl {vppl:.1f} bpc {vbpc:.3f} | elapsed {(time.time()-t0)/3600:.2f}h", flush=True)
            try:
                _rh = trainer.routing_health(n_batches=2)
                if _rh:
                    _wl = min(_rh, key=lambda li: _rh[li]["entropy"] / max(1e-9, _rh[li]["entropy_max"]))
                    _w = _rh[_wl]
                    print(f"  -> routing health worst=L{_wl} top1={_w['top1']:.2f} "
                          f"dead={int(_w['dead'])}/{int(_w['leaves'])} ent={_w['entropy']:.2f}/{_w['entropy_max']:.2f}", flush=True)
            except Exception:
                pass
            if vl < best_val:
                best_val = vl
                import os as _os
                _d = _os.path.dirname(args.save)
                if _d:
                    _os.makedirs(_d, exist_ok=True)
                torch.save(model.state_dict(), args.save)
                print(f"  ** saved {args.save} (best {best_val:.4f})", flush=True)
    stats = {"best_val_loss": best_val, "best_val_bpc": best_val / math.log(2), "best_val_ppl": math.exp(min(best_val, 20.0)), "total_time_seconds": time.time() - t0}
    print(f"Done. best_val_loss={stats['best_val_loss']:.4f} bpc={stats['best_val_bpc']:.4f} ppl={stats['best_val_ppl']:.2f}")

    # Quick Turkish sample (generation uses your 20K max from config)
    model.eval()
    prompt = "Merhaba, bugün hava çok güzel. "
    with torch.no_grad():
        ids = torch.tensor([[ord(c) % 256 for c in prompt]], dtype=torch.long, device=device)
        out = model.generate(ids, max_new_tokens=20_000)
        txt = "".join(chr(int(b) % 256) for b in out[0].tolist())
        print(f"\nPrompt: {prompt}\nSample: {txt[:400]}")


if __name__ == "__main__":
    main()
