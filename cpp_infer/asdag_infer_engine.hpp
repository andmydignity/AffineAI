#pragma once

/**
 * ============================================================================
 * ASDAG Ultra-Fast Turbo Engine (Max-Optimized Native CPU Engine)
 * ============================================================================
 * Key Architectures:
 * 1. Ping-pong zero-copy pointer swap between layers (buf_a <-> buf_b).
 * 2. Vectorized 2-pass RMSNorm: Fast AVX2/AVX-512 sum of squares + scale write.
 * 3. 8-Row Multi-Accumulator Tile with Register-Pressure-Free inner loop:
 *    - Uses single advancing base pointer x_ptr (no register spilling).
 *    - Sequential per-row unroll to keep all accumulators in YMM/XMM registers.
 * 4. Fused residual addition and ReLU6 activation directly during dst buffer write.
 * 5. Physical CPU core pinning (even CPU IDs) for SMT contention elimination.
 * 6. Supports both AVX2 (Zen 3 / 5900HX) and AVX-512 (EPYC / Zen 4+).
 * ============================================================================
 */

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <memory>
#include <pthread.h>
#include <sched.h>

#include "asdag_format.hpp"

#if defined(__GNUC__) || defined(__clang__)
    #define ASDAG_RESTRICT __restrict__
    #define ASDAG_INLINE   inline __attribute__((always_inline))
    #define ASDAG_HOT      __attribute__((hot))
    #define ASDAG_PREFETCH(p, rw, loc) __builtin_prefetch((p), (rw), (loc))
#elif defined(_MSC_VER)
    #define ASDAG_RESTRICT __declspec(restrict)
    #define ASDAG_INLINE   __forceinline
    #define ASDAG_HOT
    #define ASDAG_PREFETCH(p, rw, loc)
#else
    #define ASDAG_RESTRICT
    #define ASDAG_INLINE   inline
    #define ASDAG_HOT
    #define ASDAG_PREFETCH(p, rw, loc)
#endif

#if defined(__AVX512F__) && defined(__AVX512DQ__)
    #include <immintrin.h>
    #define ASDAG_SIMD_AVX512 1
#elif defined(__AVX2__)
    #include <immintrin.h>
    #define ASDAG_SIMD_AVX2 1
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #include <arm_neon.h>
    #define ASDAG_SIMD_NEON 1
#else
    #define ASDAG_SIMD_SCALAR 1
#endif

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace asdag {

// ─────────────────────────────────────────────────────────────────────────────
// Physical core detection & pinning
// ─────────────────────────────────────────────────────────────────────────────

inline int get_physical_core_count() {
    int total = (int)std::thread::hardware_concurrency();
    return (total >= 8) ? (total / 2) : std::max(1, total);
}

// Hard-pin this thread to physical core (skip SMT siblings: even CPU IDs on x86)
inline void pin_thread_to_physical_core(int thread_id) {
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int cpu_id = (thread_id * 2) % (int)std::thread::hardware_concurrency();
    CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// FP32 sign toggle via bitwise float sign XOR (zero branching)
// ─────────────────────────────────────────────────────────────────────────────
ASDAG_INLINE float apply_sign_bit(float x, uint8_t nibble) {
    union { float f; uint32_t u; } conv;
    conv.f = x;
    conv.u ^= ((uint32_t)(nibble & 0x08u) << 28);
    return conv.f;
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD Reductions
// ─────────────────────────────────────────────────────────────────────────────

#if defined(ASDAG_SIMD_AVX2) || defined(ASDAG_SIMD_AVX512)
ASDAG_INLINE float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
#endif

#if defined(ASDAG_SIMD_AVX512)
ASDAG_INLINE float hsum512(__m512 v) {
    __m256 lo  = _mm512_castps512_ps256(v);
    __m256 hi  = _mm512_extractf32x8_ps(v, 1);
    __m256 sum = _mm256_add_ps(lo, hi);
    return hsum256(sum);
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
// High-Speed Vectorized RMSNorm
// ─────────────────────────────────────────────────────────────────────────────
ASDAG_HOT ASDAG_INLINE void fast_rms_norm(
    const float* ASDAG_RESTRICT src,
    float* ASDAG_RESTRICT dst,
    const float* ASDAG_RESTRICT weight,
    int32_t dim,
    float eps = 1e-6f
) {
    float ss = 0.0f;
#if defined(ASDAG_SIMD_AVX512)
    __m512 acc512 = _mm512_setzero_ps();
    int32_t d = 0;
    for (; d + 16 <= dim; d += 16) {
        __m512 v = _mm512_loadu_ps(src + d);
        acc512 = _mm512_fmadd_ps(v, v, acc512);
    }
    ss = hsum512(acc512);
    for (; d < dim; ++d) ss += src[d] * src[d];
#elif defined(ASDAG_SIMD_AVX2)
    __m256 acc256 = _mm256_setzero_ps();
    int32_t d = 0;
    for (; d + 8 <= dim; d += 8) {
        __m256 v = _mm256_loadu_ps(src + d);
        acc256 = _mm256_fmadd_ps(v, v, acc256);
    }
    ss = hsum256(acc256);
    for (; d < dim; ++d) ss += src[d] * src[d];
#else
    for (int32_t d = 0; d < dim; ++d) ss += src[d] * src[d];
#endif

    float scale = 1.0f / std::sqrt((ss / (float)dim) + eps);

#if defined(ASDAG_SIMD_AVX512)
    __m512 sv512 = _mm512_set1_ps(scale);
    d = 0;
    for (; d + 16 <= dim; d += 16) {
        __m512 sv = _mm512_loadu_ps(src + d);
        __m512 wv = _mm512_loadu_ps(weight + d);
        _mm512_storeu_ps(dst + d, _mm512_mul_ps(_mm512_mul_ps(sv, sv512), wv));
    }
    for (; d < dim; ++d) dst[d] = src[d] * scale * weight[d];
#elif defined(ASDAG_SIMD_AVX2)
    __m256 sv256 = _mm256_set1_ps(scale);
    d = 0;
    for (; d + 8 <= dim; d += 8) {
        __m256 sv = _mm256_loadu_ps(src + d);
        __m256 wv = _mm256_loadu_ps(weight + d);
        _mm256_storeu_ps(dst + d, _mm256_mul_ps(_mm256_mul_ps(sv, sv256), wv));
    }
    for (; d < dim; ++d) dst[d] = src[d] * scale * weight[d];
#else
    for (int32_t d = 0; d < dim; ++d) dst[d] = src[d] * scale * weight[d];
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// 8-Row Multi-Accumulator Tile Contraction with Bitwise Sign XOR
// High GPR efficiency: process row-by-row inside the byte loop to prevent spills.
// ─────────────────────────────────────────────────────────────────────────────

ASDAG_HOT ASDAG_INLINE void sparse_1_16_tile_8x_xor(
    const uint8_t* ASDAG_RESTRICT nib0,
    const uint8_t* ASDAG_RESTRICT nib1,
    const uint8_t* ASDAG_RESTRICT nib2,
    const uint8_t* ASDAG_RESTRICT nib3,
    const uint8_t* ASDAG_RESTRICT nib4,
    const uint8_t* ASDAG_RESTRICT nib5,
    const uint8_t* ASDAG_RESTRICT nib6,
    const uint8_t* ASDAG_RESTRICT nib7,
    const float* ASDAG_RESTRICT x,
    float& ASDAG_RESTRICT acc0,
    float& ASDAG_RESTRICT acc1,
    float& ASDAG_RESTRICT acc2,
    float& ASDAG_RESTRICT acc3,
    float& ASDAG_RESTRICT acc4,
    float& ASDAG_RESTRICT acc5,
    float& ASDAG_RESTRICT acc6,
    float& ASDAG_RESTRICT acc7,
    int32_t num_bytes
) {
    float a0 = acc0, a1 = acc1, a2 = acc2, a3 = acc3;
    float a4 = acc4, a5 = acc5, a6 = acc6, a7 = acc7;

    for (int32_t b = 0; b < num_bytes; ++b) {
        const float* x_ptr = x + b * 16;

        // Row 0
        uint8_t byte0 = nib0[b];
        uint8_t n0_0 = byte0 & 0x0F, n0_1 = byte0 >> 4;
        a0 += apply_sign_bit(x_ptr[n0_0 & 0x07], n0_0);
        a0 += apply_sign_bit(x_ptr[8 + (n0_1 & 0x07)], n0_1);

        // Row 1
        uint8_t byte1 = nib1[b];
        uint8_t n1_0 = byte1 & 0x0F, n1_1 = byte1 >> 4;
        a1 += apply_sign_bit(x_ptr[n1_0 & 0x07], n1_0);
        a1 += apply_sign_bit(x_ptr[8 + (n1_1 & 0x07)], n1_1);

        // Row 2
        uint8_t byte2 = nib2[b];
        uint8_t n2_0 = byte2 & 0x0F, n2_1 = byte2 >> 4;
        a2 += apply_sign_bit(x_ptr[n2_0 & 0x07], n2_0);
        a2 += apply_sign_bit(x_ptr[8 + (n2_1 & 0x07)], n2_1);

        // Row 3
        uint8_t byte3 = nib3[b];
        uint8_t n3_0 = byte3 & 0x0F, n3_1 = byte3 >> 4;
        a3 += apply_sign_bit(x_ptr[n3_0 & 0x07], n3_0);
        a3 += apply_sign_bit(x_ptr[8 + (n3_1 & 0x07)], n3_1);

        // Row 4
        uint8_t byte4 = nib4[b];
        uint8_t n4_0 = byte4 & 0x0F, n4_1 = byte4 >> 4;
        a4 += apply_sign_bit(x_ptr[n4_0 & 0x07], n4_0);
        a4 += apply_sign_bit(x_ptr[8 + (n4_1 & 0x07)], n4_1);

        // Row 5
        uint8_t byte5 = nib5[b];
        uint8_t n5_0 = byte5 & 0x0F, n5_1 = byte5 >> 4;
        a5 += apply_sign_bit(x_ptr[n5_0 & 0x07], n5_0);
        a5 += apply_sign_bit(x_ptr[8 + (n5_1 & 0x07)], n5_1);

        // Row 6
        uint8_t byte6 = nib6[b];
        uint8_t n6_0 = byte6 & 0x0F, n6_1 = byte6 >> 4;
        a6 += apply_sign_bit(x_ptr[n6_0 & 0x07], n6_0);
        a6 += apply_sign_bit(x_ptr[8 + (n6_1 & 0x07)], n6_1);

        // Row 7
        uint8_t byte7 = nib7[b];
        uint8_t n7_0 = byte7 & 0x0F, n7_1 = byte7 >> 4;
        a7 += apply_sign_bit(x_ptr[n7_0 & 0x07], n7_0);
        a7 += apply_sign_bit(x_ptr[8 + (n7_1 & 0x07)], n7_1);
    }

    acc0 = a0; acc1 = a1; acc2 = a2; acc3 = a3;
    acc4 = a4; acc5 = a5; acc6 = a6; acc7 = a7;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fast Vectorized Hierarchical Router (AVX-512 / AVX2 / Scalar)
// ─────────────────────────────────────────────────────────────────────────────
ASDAG_HOT ASDAG_INLINE int32_t route_hierarchical_sign_fast(
    const float* ASDAG_RESTRICT x,
    const int8_t* ASDAG_RESTRICT hyperplanes,
    const float* ASDAG_RESTRICT biases,
    int32_t dim,
    int32_t tree_depth
) {
    int32_t curr_node = 0;
    for (int32_t depth = 0; depth < tree_depth; ++depth) {
        const int8_t* plane = hyperplanes + curr_node * dim;
        float dot = biases[curr_node];

#if defined(ASDAG_SIMD_AVX512)
        __m512 dot_v = _mm512_setzero_ps();
        int32_t d = 0;
        for (; d + 16 <= dim; d += 16) {
            __m512 xv = _mm512_loadu_ps(x + d);
            __m128i p8 = _mm_loadu_si128((const __m128i*)(plane + d));
            __m512 pv = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(p8));
            dot_v = _mm512_fmadd_ps(xv, pv, dot_v);
        }
        dot += hsum512(dot_v);
        for (; d < dim; ++d) {
            if (plane[d] == 1) dot += x[d];
            else if (plane[d] == -1) dot -= x[d];
        }
#elif defined(ASDAG_SIMD_AVX2)
        __m256 dot_v = _mm256_setzero_ps();
        int32_t d = 0;
        for (; d + 8 <= dim; d += 8) {
            __m256 xv = _mm256_loadu_ps(x + d);
            __m128i p8 = _mm_loadl_epi64((const __m128i*)(plane + d));
            __m256 pv = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(p8));
            dot_v = _mm256_fmadd_ps(xv, pv, dot_v);
        }
        dot += hsum256(dot_v);
        for (; d < dim; ++d) {
            if (plane[d] == 1) dot += x[d];
            else if (plane[d] == -1) dot -= x[d];
        }
#else
        for (int32_t d = 0; d < dim; ++d) {
            if (plane[d] == 1) dot += x[d];
            else if (plane[d] == -1) dot -= x[d];
        }
#endif
        int32_t step = (dot > 0.0f) ? 1 : 0;
        curr_node = 2 * curr_node + 1 + step;
    }
    int32_t leaf_idx = curr_node - ((1 << tree_depth) - 1);
    return std::max(0, leaf_idx);
}

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

struct ASDAGLeafNibbleTurbo {
    std::vector<uint8_t> nibble_weights; // (d_model * (d_model / 16)) bytes
    std::vector<float>   bias;           // (d_model) float32
    float                norm_factor = 1.0f;
};

struct ASDAGLayerDataTurbo {
    int32_t d_model;
    int32_t num_leaves;
    int32_t tree_depth;
    std::vector<int8_t>  hyperplanes;
    std::vector<float>   router_biases;
    std::vector<float>   norm_weight;    // Learnable RMSNorm scale vector
    std::vector<ASDAGLeafNibbleTurbo> leaves;
};

// ─────────────────────────────────────────────────────────────────────────────
// Turbo Inference Engine
// ─────────────────────────────────────────────────────────────────────────────

class ASDAGInferenceEngineTurbo {
public:
    ASDAGHeaderV2 header;
    std::vector<ASDAGLayerDataTurbo> layers;
    int32_t physical_cores;

    explicit ASDAGInferenceEngineTurbo(ASDAGHeaderV2 hdr) : header(hdr) {
        physical_cores = get_physical_core_count();
        layers.resize(header.n_layers);

        int32_t bytes_per_row = (int32_t)header.d_model / (int32_t)header.nm_m;

        for (uint32_t l = 0; l < header.n_layers; ++l) {
            layers[l].d_model = header.d_model;
            layers[l].num_leaves = header.num_leaves;
            layers[l].tree_depth = header.tree_depth;

            int32_t num_internal = (1 << header.tree_depth) - 1;
            layers[l].hyperplanes.resize(num_internal * header.d_model, 0);
            layers[l].router_biases.resize(num_internal, 0.0f);
            layers[l].norm_weight.resize(header.d_model, 1.0f);
            layers[l].leaves.resize(header.num_leaves);

            for (uint32_t k = 0; k < header.num_leaves; ++k) {
                layers[l].leaves[k].nibble_weights.resize(header.d_model * bytes_per_row, 0);
                layers[l].leaves[k].bias.resize(header.d_model, 0.0f);
                layers[l].leaves[k].norm_factor = 1.0f / std::sqrt((float)header.d_model);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 8-Row Tiled Layer Step with Fused Pre-RMSNorm and Output Residual Add
    // ─────────────────────────────────────────────────────────────────────────
    ASDAG_HOT void forward_layer_fused(
        int32_t layer_idx,
        const float* ASDAG_RESTRICT x_in,   // Input residual stream
        float* ASDAG_RESTRICT x_out,        // Output residual stream (x_in + act)
        float* ASDAG_RESTRICT x_normed      // Scratch buffer for RMSNorm
    ) const {
        const auto& layer = layers[layer_idx];
        const int32_t dim = layer.d_model;
        const int32_t num_bytes_row = dim / 16;

        // 1. Vectorized RMSNorm: x_in -> x_normed
        fast_rms_norm(x_in, x_normed, layer.norm_weight.data(), dim);

        // 2. Vectorized Hyperplane Routing
        int32_t leaf_idx = route_hierarchical_sign_fast(
            x_normed, layer.hyperplanes.data(), layer.router_biases.data(), dim, layer.tree_depth
        );
        if (leaf_idx >= layer.num_leaves) leaf_idx = layer.num_leaves - 1;

        const auto& leaf = layer.leaves[leaf_idx];
        const uint8_t* W_nibble = leaf.nibble_weights.data();
        const float* b_l = leaf.bias.data();
        const float norm_fac = leaf.norm_factor;

        // 3. 8-Row Tiled Sparse GEMV + Fused Residual ReLU6 Store
        int32_t r = 0;
        for (; r + 8 <= dim; r += 8) {
            float acc0 = b_l[r + 0], acc1 = b_l[r + 1], acc2 = b_l[r + 2], acc3 = b_l[r + 3];
            float acc4 = b_l[r + 4], acc5 = b_l[r + 5], acc6 = b_l[r + 6], acc7 = b_l[r + 7];

            ASDAG_PREFETCH(W_nibble + (r + 8) * num_bytes_row, 0, 1);

            sparse_1_16_tile_8x_xor(
                W_nibble + (r + 0) * num_bytes_row,
                W_nibble + (r + 1) * num_bytes_row,
                W_nibble + (r + 2) * num_bytes_row,
                W_nibble + (r + 3) * num_bytes_row,
                W_nibble + (r + 4) * num_bytes_row,
                W_nibble + (r + 5) * num_bytes_row,
                W_nibble + (r + 6) * num_bytes_row,
                W_nibble + (r + 7) * num_bytes_row,
                x_normed,
                acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
                num_bytes_row
            );

            // Fused ReLU6 activation + residual add directly to output stream
            x_out[r + 0] = x_in[r + 0] + std::min(std::max(acc0 * norm_fac, 0.0f), 6.0f);
            x_out[r + 1] = x_in[r + 1] + std::min(std::max(acc1 * norm_fac, 0.0f), 6.0f);
            x_out[r + 2] = x_in[r + 2] + std::min(std::max(acc2 * norm_fac, 0.0f), 6.0f);
            x_out[r + 3] = x_in[r + 3] + std::min(std::max(acc3 * norm_fac, 0.0f), 6.0f);
            x_out[r + 4] = x_in[r + 4] + std::min(std::max(acc4 * norm_fac, 0.0f), 6.0f);
            x_out[r + 5] = x_in[r + 5] + std::min(std::max(acc5 * norm_fac, 0.0f), 6.0f);
            x_out[r + 6] = x_in[r + 6] + std::min(std::max(acc6 * norm_fac, 0.0f), 6.0f);
            x_out[r + 7] = x_in[r + 7] + std::min(std::max(acc7 * norm_fac, 0.0f), 6.0f);
        }

        // Remainder loop
        for (; r < dim; ++r) {
            float acc = b_l[r];
            const uint8_t* row_nibbles = W_nibble + r * num_bytes_row;
            for (int32_t b = 0; b < num_bytes_row; ++b) {
                const float* x_ptr = x_normed + b * 16;
                uint8_t byte_val = row_nibbles[b];
                uint8_t low_n = byte_val & 0x0F, high_n = byte_val >> 4;
                acc += apply_sign_bit(x_ptr[low_n & 0x07], low_n);
                acc += apply_sign_bit(x_ptr[8 + (high_n & 0x07)], high_n);
            }
            x_out[r] = x_in[r] + std::min(std::max(acc * norm_fac, 0.0f), 6.0f);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Zero-Copy Ping-Pong Token Pipeline across all 32 layers
    // ─────────────────────────────────────────────────────────────────────────
    ASDAG_HOT void forward_token_direct(
        const float* ASDAG_RESTRICT x_in,
        float* ASDAG_RESTRICT y_out,
        float* ASDAG_RESTRICT buf_a,
        float* ASDAG_RESTRICT buf_b,
        float* ASDAG_RESTRICT buf_norm
    ) const {
        float* src = buf_a;
        float* dst = buf_b;

        std::memcpy(src, x_in, header.d_model * sizeof(float));

        for (uint32_t l = 0; l < header.n_layers; ++l) {
            forward_layer_fused(l, src, dst, buf_norm);
            std::swap(src, dst);
        }

        std::memcpy(y_out, src, header.d_model * sizeof(float));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Multi-Core Batch Parallel Execution
    // ─────────────────────────────────────────────────────────────────────────
    void forward_batch_parallel(
        const float* ASDAG_RESTRICT X,
        float* ASDAG_RESTRICT Y,
        int32_t num_tokens
    ) const {
        int32_t n_threads = physical_cores;
        int32_t dim = header.d_model;

#pragma omp parallel num_threads(n_threads)
        {
            // Explicit hard-pin to physical CPU ID (0, 2, 4, 6, 8, 10, 12, 14)
            int tid = omp_get_thread_num();
            pin_thread_to_physical_core(tid);

            // Fast aligned stack / heap buffers per thread
            std::vector<float> buf_a(dim);
            std::vector<float> buf_b(dim);
            std::vector<float> buf_norm(dim);

#pragma omp for schedule(static)
            for (int32_t t = 0; t < num_tokens; ++t) {
                const float* x_in = X + t * dim;
                float* y_out = Y + t * dim;
                if (t + 1 < num_tokens) {
                    ASDAG_PREFETCH(X + (t + 1) * dim, 0, 1);
                }
                forward_token_direct(x_in, y_out, buf_a.data(), buf_b.data(), buf_norm.data());
            }
        }
    }
};

} // namespace asdag
