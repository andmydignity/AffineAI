import os
import time
import torch
import torch.nn.functional as F

from affine_ai.models.hybrid import TorosHybridConfig, TorosHybridLanguageModel
from affine_ai.training.distributed import (
    spawn_multiprocess_training,
    get_rank,
    get_world_size,
    is_main_process,
    setup_distributed,
    cleanup_distributed,
    all_reduce_avg,
)

# -------------------------------------------------------------------
# Model Configuration: 1.08M TorosHybrid (Turing T4 FP16 optimized)
# -------------------------------------------------------------------
def get_1m_config() -> TorosHybridConfig:
    return TorosHybridConfig(
        dim=272,
        d_byte=136,
        n_encoder_layers=8,
        n_heads=4,
        target_patch_size=16,
        channel_mixer_type="asdag_tree",
        mlp_hidden_dim=84,
        decoder_channel_mixer="swiglu",
        time_mixer_rule="gla",
        swa_every_n=6,
        swa_window=256,
        use_csa=True,
        csa_every_n=4,
        csa_group_size=4,
        csa_window=256,
        csa_kv_quant="int4",
        use_tree_sga=True,
        gen_loss_weight=1.0,
        use_conv_prefix=True,
        conv_kernel_size=8,
        use_dense_readout=True,
        context_window=512,
        max_seq_len=512,
        lpc_chunk_size=4,
        dtype=torch.float16,  # T4 Turing SM_75 uses FP16 Tensor Cores
    )


# -------------------------------------------------------------------
# Benchmark 1: Single T4 Throughput
# -------------------------------------------------------------------
def bench_single_t4(batch_size: int = 96, seq_len: int = 512, iters: int = 40):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    cfg = get_1m_config()
    model = TorosHybridLanguageModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    dev_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"
    print("\n" + "=" * 70)
    print(f"PHASE 1: SINGLE DEVICE BENCHMARK ({dev_name})")
    print(f"Model Parameters: {n_params:,} (~1.08M)")
    print(f"Batch Size: {batch_size}, Sequence Length: {seq_len} ({batch_size * seq_len:,} tokens/step)")
    print("=" * 70)

    x = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long, device=device)
    targets = x.clone()

    # 1. Forward Inference
    for _ in range(5):  # Warmup
        with torch.no_grad():
            _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    fwd_time = (time.perf_counter() - t0) / iters
    fwd_tok_s = (batch_size * seq_len) / fwd_time

    print(f"[Inference Forward]  Latency: {fwd_time * 1000:>6.2f} ms | Throughput: {fwd_tok_s:>10,.0f} tok/s")

    # 2. LPC Training Step
    optimizers = model.get_default_lpc_optimizers(lr=1e-3, weight_decay=0.01)
    for _ in range(5):  # Warmup
        _ = model.forward_lpc_step(x, targets, optimizers=optimizers, use_cuda_graph=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model.forward_lpc_step(x, targets, optimizers=optimizers, use_cuda_graph=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    lpc_time = (time.perf_counter() - t0) / iters
    lpc_tok_s = (batch_size * seq_len) / lpc_time
    peak_vram = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if torch.cuda.is_available() else 0.0

    print(f"[LPC Training Step]  Latency: {lpc_time * 1000:>6.2f} ms | Throughput: {lpc_tok_s:>10,.0f} tok/s")
    if torch.cuda.is_available():
        print(f"[Memory Footprint]   Peak VRAM: {peak_vram:>6.1f} MB (out of 15,109 MB)")

    return {
        "fwd_tok_s": fwd_tok_s,
        "lpc_tok_s": lpc_tok_s,
        "lpc_latency_ms": lpc_time * 1000,
        "peak_vram_mb": peak_vram,
    }


# -------------------------------------------------------------------
# Benchmark 2: Dual T4 Multi-GPU Distributed LPC Step
# -------------------------------------------------------------------
def dual_t4_worker(batch_per_gpu: int = 96, seq_len: int = 512, iters: int = 40):
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{rank}")

    torch.cuda.reset_peak_memory_stats(device)

    cfg = get_1m_config()
    model = TorosHybridLanguageModel(cfg).to(device)
    optimizers = model.get_default_lpc_optimizers(lr=1e-3, weight_decay=0.01)

    x = torch.randint(0, 256, (batch_per_gpu, seq_len), dtype=torch.long, device=device)
    targets = x.clone()

    # Warmup
    for _ in range(5):
        _ = model.forward_lpc_step(x, targets, optimizers=optimizers, use_cuda_graph=False)
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model.forward_lpc_step(x, targets, optimizers=optimizers, use_cuda_graph=False)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    step_time = elapsed / iters

    total_tokens_per_step = batch_per_gpu * seq_len * world_size
    combined_tok_s = total_tokens_per_step / step_time
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "step_time_ms": step_time * 1000,
        "combined_tok_s": combined_tok_s,
        "total_tokens_per_step": total_tokens_per_step,
        "peak_vram_mb": peak_vram,
    }


def main():
    if not torch.cuda.is_available():
        print("Notice: CUDA is not available. Running CPU single-device baseline.")
        bench_single_t4(batch_size=4, seq_len=128, iters=5)
        return

    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} CUDA Device(s):")
    for i in range(n_gpus):
        print(f"  [GPU {i}] {torch.cuda.get_device_name(i)}")

    # 1. Single GPU Test
    single_res = bench_single_t4(batch_size=96, seq_len=512, iters=40)

    # 2. Dual GPU Test (if 2 GPUs available)
    if n_gpus >= 2:
        print("\n" + "=" * 70)
        print("PHASE 2: DUAL TESLA T4 DISTRIBUTED BENCHMARK (2x T4)")
        print(f"Per-GPU Batch: 96, Combined Global Batch: 192, Seq: 512 ({192 * 512:,} tok/step)")
        print("=" * 70)

        dual_res = spawn_multiprocess_training(
            dual_t4_worker,
            devices=[0, 1],
            batch_per_gpu=96,
            seq_len=512,
            iters=40,
        )

        speedup = dual_res["combined_tok_s"] / single_res["lpc_tok_s"]
        scaling_efficiency = (speedup / 2.0) * 100.0

        print(f"[Dual-GPU LPC Step] Latency: {dual_res['step_time_ms']:>6.2f} ms")
        print(f"[Dual-GPU LPC Step] Total Throughput: {dual_res['combined_tok_s']:>10,.0f} tok/s")
        print(f"[Peak VRAM per GPU] {dual_res['peak_vram_mb']:>6.1f} MB")
        print(f"[Multi-GPU Scaling] Speedup: {speedup:>5.2f}x ({scaling_efficiency:>5.1f}% linear efficiency)")

        print("\n" + "=" * 70)
        print("SUMMARY RESULTS: 1.08M TorosHybrid on Kaggle Dual T4")
        print("=" * 70)
        print(f"{'Metric':<25} | {'1x Tesla T4':<18} | {'2x Tesla T4':<18}")
        print("-" * 70)
        print(f"{'Global Batch Size':<25} | {96:<18} | {192:<18}")
        print(f"{'LPC Step Latency':<25} | {single_res['lpc_latency_ms']:>6.2f} ms          | {dual_res['step_time_ms']:>6.2f} ms")
        print(f"{'Training Throughput':<25} | {single_res['lpc_tok_s']:>9,.0f} tok/s        | {dual_res['combined_tok_s']:>9,.0f} tok/s")
        print(f"{'Peak VRAM / GPU':<25} | {single_res['peak_vram_mb']:>6.1f} MB          | {dual_res['peak_vram_mb']:>6.1f} MB")
        print(f"{'Multi-GPU Scaling':<25} | {'1.00x':<18} | {f'{speedup:.2f}x ({scaling_efficiency:.1f}%)':<18}")
        print("=" * 70)
    else:
        print("\n[Notice] Only 1 GPU detected. In Kaggle, switch Accelerator to 'GPU T4 x2' to run Dual-GPU.")


if __name__ == "__main__":
    main()
