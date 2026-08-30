/**
 * ============================================================================
 * Full 32-Layer ASDAG 4B Inference Benchmark (Turbo 8-Row Engine)
 * ============================================================================
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>

#include "asdag_format.hpp"
#include "asdag_infer_engine.hpp"

using clk = std::chrono::high_resolution_clock;

int main() {
    std::cout << "====================================================================\n";
    std::cout << "  ASDAG Turbo 8-Row Multi-Accumulator 32-Layer Benchmark\n";
    std::cout << "====================================================================\n";

    ASDAGHeaderV2 header;
    header.d_model = 2560;
    header.n_layers = 32;
    header.num_leaves = 16;
    header.tree_depth = 4;
    header.nm_n = 1;
    header.nm_m = 16;  // 1:16 sparsity

    std::cout << "Model Architecture:\n";
    std::cout << "  - Layers               : " << header.n_layers << "\n";
    std::cout << "  - Hidden Dimension (d) : " << header.d_model << "\n";
    std::cout << "  - Leaves per Layer (K) : " << header.num_leaves << "\n";
    std::cout << "  - Vector Engine        : 8-Row Simultaneous Multi-Accumulator\n";
    std::cout << "  - Sign Arithmetic      : Pure Float Sign XOR (0 Branching)\n";
    std::cout << "  - Sparsity Structure   : 1:16 Nibble Packed (93.75% Zero Math)\n\n";

    asdag::ASDAGInferenceEngineTurbo engine(header);
    std::cout << "Initialized Turbo Engine on " << engine.physical_cores << " Physical CPU Cores.\n";

    size_t total_weight_bytes = 0;
    for (const auto& l : engine.layers) {
        total_weight_bytes += l.hyperplanes.size() * sizeof(int8_t);
        total_weight_bytes += l.router_biases.size() * sizeof(float);
        for (const auto& leaf : l.leaves) {
            total_weight_bytes += leaf.nibble_weights.size() * sizeof(uint8_t);
            total_weight_bytes += leaf.bias.size() * sizeof(float);
        }
    }
    printf("  Total Model RAM Footprint: %5.1f MB (%4.3f GB)\n\n",
           (float)total_weight_bytes / (1024.0f * 1024.0f),
           (float)total_weight_bytes / (1024.0f * 1024.0f * 1024.0f));

    std::mt19937 rng(42);
    for (auto& layer : engine.layers) {
        for (auto& p : layer.hyperplanes) p = (rng() % 3) - 1;
        for (auto& leaf : layer.leaves) {
            for (auto& b : leaf.nibble_weights) {
                uint8_t low_n = (rng() % 8) | ((rng() % 2 == 0) ? 0x00 : 0x08);
                uint8_t high_n = (rng() % 8) | ((rng() % 2 == 0) ? 0x00 : 0x08);
                b = (high_n << 4) | low_n;
            }
        }
    }

    std::vector<int32_t> test_tokens = {16, 64, 256, 1024};

    for (int32_t N : test_tokens) {
        std::vector<float> X(N * header.d_model, 1.0f);
        std::vector<float> Y(N * header.d_model, 0.0f);

        // Warmup
        for (int i = 0; i < 2; ++i) engine.forward_batch_parallel(X.data(), Y.data(), N);

        int32_t iters = 10;
        auto t0 = clk::now();
        for (int i = 0; i < iters; ++i) {
            engine.forward_batch_parallel(X.data(), Y.data(), N);
        }
        auto t1 = clk::now();

        double dt = std::chrono::duration<double>(t1 - t0).count() / iters;
        double tok_s = N / dt;
        double layer_latency_us = (dt * 1000.0 * 1000.0) / (header.n_layers * N);

        printf("--- Batch: %4d Tokens ---\n", N);
        printf("  Total 32-Layer Latency : %6.2f ms\n", dt * 1000.0);
        printf("  Per-Token Latency      : %6.2f ms / token (%.2f tok/s across full 32 layers)\n", 
               (dt * 1000.0) / N, tok_s);
        printf("  Per-Layer Math Latency : %6.2f µs / layer\n\n", layer_latency_us);
    }

    std::cout << "====================================================================\n";
    return 0;
}
