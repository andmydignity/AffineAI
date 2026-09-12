import math
import numpy as np
import torch
from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig
from affine_ai.data.dataloader import PaddedDataLoader
from affine_ai.training.trainer import ASDAGTrainer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "models/toros_hybrid_700k_simplestories.pt"

    cfg = TorosHybridConfig(
        dim=288,
        n_encoder_layers=7,
        channel_mixer_type="asdag_tree",
        dtype=torch.bfloat16
    )
    model = TorosHybridLanguageModel(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)

    data_path = "data/simplestories_eos.bin"
    raw_data = np.memmap(data_path, dtype=np.uint8, mode="r")
    split = int(0.95 * len(raw_data))

    val_loader = PaddedDataLoader(raw_data[split:], batch_size=32, seq_len=512, device=device, as_stream=True)
    train_loader = PaddedDataLoader(raw_data[:split], batch_size=32, seq_len=512, device=device, as_stream=True)

    trainer = ASDAGTrainer(
        model=model,
        train_data=train_loader,
        val_data=val_loader,
        batch_size=32,
        seq_len=512,
        device=device
    )

    # Freeze expert bias during measurement to measure the checkpoint's routing state
    for blk in model.context_encoder.blocks:
        cm = getattr(blk, "channel_mixer", None) or getattr(blk, "asdag", None)
        if cm is not None:
            cm.expert_bias_rate = 0.0

    stats = trainer.routing_health(n_batches=5)

    print("=" * 95)
    print("      DEEPSEEK EXPERT BIAS & DEAD LEAF AUDIT REPORT")
    print(f"      Checkpoint: {checkpoint_path}")
    print("=" * 95)

    total_dead = 0
    total_leaves = 0

    for li, blk in enumerate(model.context_encoder.blocks):
        cm = getattr(blk, "channel_mixer", None) or getattr(blk, "asdag", None)
        if getattr(cm, "use_hierarchical_routing", False) and hasattr(cm, "router") and cm.router is not None:
            b = cm.router.biases.detach().cpu()
            b_type = "Node Biases"
        else:
            b = cm.expert_bias.detach().cpu()
            b_type = "Leaf Biases"
        s = stats[li]
        dead_cnt = int(s["dead"])
        leaf_cnt = int(s["leaves"])
        total_dead += dead_cnt
        total_leaves += leaf_cnt
        ent_pct = (s["entropy"] / max(1e-9, s["entropy_max"])) * 100.0

        print(f"Layer {li:2d} | Dead Leaves: {dead_cnt:2d}/{leaf_cnt:2d} ({dead_cnt/leaf_cnt*100:4.1f}%) | "
              f"Top-1 Conc: {s['top1']*100:5.1f}% | "
              f"Entropy: {s['entropy']:.2f}/{s['entropy_max']:.2f} ({ent_pct:5.1f}%)")
        print(f"         {b_type} Range: [{b.min():+.4f}, {b.max():+.4f}] | Std: {b.std():.4f} | "
              f"Biases: {[round(x, 3) for x in b.tolist()]}")

    print("-" * 95)
    print(f"OVERALL SUMMARY: {total_dead} dead leaves out of {total_leaves} total leaves across all layers ({total_dead/total_leaves*100:.1f}%)")
    print("=" * 95)

if __name__ == "__main__":
    main()
