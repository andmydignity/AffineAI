#!/usr/bin/env python3
"""
Benchmark: Triton Fused ASDAG Kernel vs Standard PyTorch Python Execution
==========================================================================
Measures forward throughput (tok/s) on GPU (RTX 3050).
"""

import time
import torch
import torch.nn.functional as F

from affine_ai.kernels.triton_asdag import fused_asdag_2d_triton
from affine_ai.core.ast_dag import ASTDAGLayer


def benchmark_asdag_speed():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA not available.")
        return

    B = 256
    T = 128
    D = 96
    K = 16
    M_max = 2

    print(f"Benchmarking ASDAG on {device}: Batch={B}, SeqLen={T} (Total={B*T:,} tokens), Dim={D}, Leaves={K}...")

    layer = ASTDAGLayer(dim=D, initial_branches=K, rank=None).to(device)
    x = torch.randn(B * T, D, device=device)

    # 1. PyTorch standard python loop execution
    layer.eval()
    # Warmup
    for _ in range(5):
        _ = layer(x)
    torch.cuda.synchronize()

    t0 = time.time()
    iters = 30
    for _ in range(iters):
        _ = layer(x)
    torch.cuda.synchronize()
    dt_pytorch = (time.time() - t0) / iters
    tok_s_pytorch = (B * T) / dt_pytorch
    print(f"  PyTorch Python Execution : {dt_pytorch*1000:.2f} ms | {tok_s_pytorch:>9,.0f} tok/s")

    # 2. PyTorch Batched Tensor Dispatch
    for _ in range(5):
        _ = layer.forward_batched_dispatch(x)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = layer.forward_batched_dispatch(x)
    torch.cuda.synchronize()
    dt_batched = (time.time() - t0) / iters
    tok_s_batched = (B * T) / dt_batched
    print(f"  PyTorch Batched Dispatch : {dt_batched*1000:.2f} ms | {tok_s_batched:>9,.0f} tok/s ({tok_s_batched/tok_s_pytorch:.1f}x speedup)")

    # 3. 2D Grid-Tiled Triton Kernel
    from affine_ai.kernels.triton_asdag import fused_asdag_2d_triton
    leaves = layer.leaves
    w_stack = torch.stack([leaf.latent_W_primary.data for leaf in leaves], dim=0)
    bias_stack = torch.stack([leaf.bias.data for leaf in leaves], dim=0)
    routing_probs = F.softmax(F.linear(x, layer.router_weights, layer.router_biases), dim=-1)
    context_gates = torch.zeros(K, M_max, D, device=device)
    norm_factors = torch.ones(K, device=device)

    for _ in range(5):
        _ = fused_asdag_2d_triton(x, w_stack, bias_stack, routing_probs, context_gates, norm_factors)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = fused_asdag_2d_triton(x, w_stack, bias_stack, routing_probs, context_gates, norm_factors)
    torch.cuda.synchronize()
    dt_triton = (time.time() - t0) / iters
    tok_s_triton = (B * T) / dt_triton
    print(f"  Triton 2D Grid-Tiled     : {dt_triton*1000:.2f} ms | {tok_s_triton:>9,.0f} tok/s ({tok_s_triton/tok_s_pytorch:.2f}x of PyTorch)")


if __name__ == "__main__":
    benchmark_asdag_speed()
