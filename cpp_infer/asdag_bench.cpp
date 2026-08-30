/**
 * ============================================================================
 * ASDAG C++20 Coroutine + 4-Way Register-Tiled Benchmark
 * ============================================================================
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include "asdag_engine.hpp"

using clk = std::chrono::high_resolution_clock;

void dense_mlp_forward(
    const float* X,
    const float* W1,
    const float* W2,
    float* Y,
    int32_t B,
    int32_t D,
    int32_t H
) {
#pragma omp parallel for schedule(static)
    for (int32_t b = 0; b < B; ++b) {
        const float* x_b = X + b * D;
        float* y_b = Y + b * D;

        std::vector<float> h(H, 0.0f);
        for (int32_t j = 0; j < H; ++j) {
            float s = 0.0f;
            for (int32_t i = 0; i < D; ++i) {
                s += x_b[i] * W1[j * D + i];
            }
            h[j] = 0.5f * s * (1.0f + std::tanh(0.79788456f * (s + 0.044715f * s * s * s)));
        }

        for (int32_t i = 0; i < D; ++i) {
            float s = 0.0f;
            for (int32_t j = 0; j < H; ++j) {
                s += h[j] * W2[i * H + j];
            }
            y_b[i] = s;
        }
    }
}

int main() {
    int32_t dim = 96;
    int32_t leaves = 16;
    int32_t hidden = 384;
    std::vector<int32_t> batch_sizes = {512, 2048, 8192, 32768};

    std::cout << "====================================================================\n";
    std::cout << "  ASDAG C++20 Coroutine + 4-Way SIMD Benchmark (AVX2 / OpenMP)\n";
    std::cout << "====================================================================\n";

    asdag::LayerConfig cfg;
    cfg.dim = dim;
    cfg.num_leaves = leaves;
    asdag::ASDAGLayerCPP asdag_layer(cfg);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist_ternary(-1, 1);
    std::uniform_real_distribution<float> dist_real(-1.0f, 1.0f);

    for (auto& w : asdag_layer.W_leaves) w = dist_ternary(rng);
    for (auto& p : asdag_layer.hyperplanes) p = dist_ternary(rng);
    for (auto& b : asdag_layer.bias_leaves) b = dist_real(rng);

    std::vector<float> W1(hidden * dim), W2(dim * hidden);
    for (auto& w : W1) w = dist_real(rng);
    for (auto& w : W2) w = dist_real(rng);

    for (int32_t B : batch_sizes) {
        std::vector<float> X(B * dim);
        std::vector<float> Y_asdag(B * dim, 0.0f);
        std::vector<float> Y_mlp(B * dim, 0.0f);
        for (auto& x : X) x = dist_real(rng);

        int32_t iters = 25;

        // Warmup
        for (int i = 0; i < 3; ++i) asdag_layer.forward_batch_coroutines(X.data(), Y_asdag.data(), B);

        auto t0 = clk::now();
        for (int i = 0; i < iters; ++i) {
            asdag_layer.forward_batch_coroutines(X.data(), Y_asdag.data(), B);
        }
        auto t1 = clk::now();
        double dt_asdag = std::chrono::duration<double>(t1 - t0).count() / iters;
        double tok_s_asdag = B / dt_asdag;

        for (int i = 0; i < 3; ++i) dense_mlp_forward(X.data(), W1.data(), W2.data(), Y_mlp.data(), B, dim, hidden);

        t0 = clk::now();
        for (int i = 0; i < iters; ++i) {
            dense_mlp_forward(X.data(), W1.data(), W2.data(), Y_mlp.data(), B, dim, hidden);
        }
        t1 = clk::now();
        double dt_mlp = std::chrono::duration<double>(t1 - t0).count() / iters;
        double tok_s_mlp = B / dt_mlp;

        printf("\n--- Tokens: %6d (Dim=%d) ---\n", B, dim);
        printf("  Dense FP32 MLP (C++ Multi-threaded): %6.2f ms | %10.0f tok/s\n", dt_mlp * 1000.0, tok_s_mlp);
        printf("  ASDAG C++20 Coroutine SIMD Engine  : %6.2f ms | %10.0f tok/s  [ %.2fx Faster! ]\n", 
               dt_asdag * 1000.0, tok_s_asdag, tok_s_asdag / tok_s_mlp);
    }

    std::cout << "\n====================================================================\n";
    return 0;
}
