#pragma once

/**
 * ============================================================================
 * ASDAG Layer Engine — Fixed & Optimized (Dense int8 weights, for benchmarking)
 * ============================================================================
 * Fixes applied:
 *   [FIX-1]  Replaced proc_bind(close) + coroutine no-op with pthread pinning
 *   [FIX-2]  Coroutine removed — was net-negative overhead; replaced with
 *            __builtin_prefetch with correct look-ahead distance
 *   [OPT-1]  schedule(static) for uniform token batches
 *   [OPT-2]  AVX-512 path in router (falls back to AVX2 / scalar)
 *   [OPT-3]  Tail loop in ternary_shift4_gemv_8x always runs (was #if guarded)
 * ============================================================================
 */

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <vector>
#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <pthread.h>
#include <sched.h>

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
// Physical core pinning (bypasses SMT siblings)
// ─────────────────────────────────────────────────────────────────────────────
inline const std::vector<int>& physical_cpu_list() {
    static std::vector<int> cpus = [] {
        std::vector<int> out;
#if defined(__linux__)
        auto parse_first = [](const char* path) {
            int first = -1;
            FILE* f = std::fopen(path, "r");
            if (f) {
                int a = -1, b = -1;
                char buf[256] = {0};
                if (std::fgets(buf, sizeof(buf), f)) {
                    if (std::sscanf(buf, "%d-%d", &a, &b) == 2) first = std::min(a, b);
                    else if (std::sscanf(buf, "%d", &a) == 1) first = a;
                }
                std::fclose(f);
            }
            return first;
        };
        int total = (int)std::thread::hardware_concurrency();
        if (total > 0) {
            std::vector<int> seen;
            for (int cpu = 0; cpu < total; ++cpu) {
                char path[128];
                std::snprintf(path, sizeof(path),
                    "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
                int first = parse_first(path);
                if (first < 0) { out.clear(); break; }
                bool known = false;
                for (int s : seen) if (s == first) { known = true; break; }
                if (!known) { seen.push_back(first); out.push_back(cpu); }
            }
        }
#endif
        if (out.empty()) {
            int total = (int)std::thread::hardware_concurrency();
            int n = (total >= 8) ? (total / 2) : std::max(1, total);
            for (int i = 0; i < n; ++i) out.push_back((i * 2) % std::max(1, total));
        }
        return out;
    }();
    return cpus;
}

inline int get_physical_cores() {
    return (int)physical_cpu_list().size();
}

inline void pin_thread_to_physical_core(int thread_id) {
#if defined(__linux__)
    const std::vector<int>& cpus = physical_cpu_list();
    if (cpus.empty()) return;
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(cpus[thread_id % (int)cpus.size()], &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cs);
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD helpers
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

ASDAG_INLINE int32_t hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

ASDAG_INLINE float hmax256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m128 = _mm_max_ps(lo, hi);
    __m128 m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
    __m128 m32 = _mm_max_ss(m64, _mm_shuffle_ps(m64, m64, 0x55));
    return _mm_cvtss_f32(m32);
}

ASDAG_INLINE __m256 exp256_ps(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.336544f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.722839f));

    const __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2_hi = _mm256_set1_ps(0.6931471805599453f);
    const __m256 ln2_lo = _mm256_set1_ps(2.3190468138462996e-17f);

    __m256 fx = _mm256_fmadd_ps(x, log2ef, _mm256_set1_ps(0.5f));
    __m256 fx_floor = _mm256_floor_ps(fx);
    __m256i emm0 = _mm256_cvttps_epi32(fx_floor);

    __m256 x_red = _mm256_fnmadd_ps(fx_floor, ln2_hi, x);
    x_red = _mm256_fnmadd_ps(fx_floor, ln2_lo, x_red);

    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(1.0f);
    const __m256 c2 = _mm256_set1_ps(0.5f);
    const __m256 c3 = _mm256_set1_ps(1.6666666666666666e-1f);
    const __m256 c4 = _mm256_set1_ps(4.1666666666666664e-2f);
    const __m256 c5 = _mm256_set1_ps(8.3333333333333332e-3f);

    __m256 y = _mm256_fmadd_ps(c5, x_red, c4);
    y = _mm256_fmadd_ps(y, x_red, c3);
    y = _mm256_fmadd_ps(y, x_red, c2);
    y = _mm256_fmadd_ps(y, x_red, c1);
    y = _mm256_fmadd_ps(y, x_red, c0);

    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
    emm0 = _mm256_slli_epi32(emm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(emm0);
    return _mm256_mul_ps(y, pow2n);
}

ASDAG_INLINE __m256 sigmoid256_ps(__m256 x) {
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_neg = exp256_ps(neg_x);
    __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg);
    return _mm256_div_ps(_mm256_set1_ps(1.0f), denom);
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
// LayerConfig
// ─────────────────────────────────────────────────────────────────────────────
struct LayerConfig {
    int32_t dim                    = 96;
    int32_t num_leaves             = 16;
    int32_t max_secondary          = 4;
    int32_t rank                   = 0;
    int32_t use_shift4             = 1;
    int32_t use_power_of_two_gates = 1;
    int32_t use_hierarchical_routing = 1;
    int32_t top_k                  = 2;
};

// ─────────────────────────────────────────────────────────────────────────────
// 8-way register-tiled ternary GEMV  (dense int8 layout)
// ─────────────────────────────────────────────────────────────────────────────
ASDAG_INLINE void ternary_shift4_gemv_8x(
    const int8_t* ASDAG_RESTRICT W,
    const float*  ASDAG_RESTRICT x,
    float*        ASDAG_RESTRICT y,
    int32_t out_dim,
    int32_t in_dim
) {
    int32_t i = 0;

#if defined(ASDAG_SIMD_AVX512)
    for (; i + 8 <= out_dim; i += 8) {
        const int8_t* w0 = W + (i+0) * in_dim;
        const int8_t* w1 = W + (i+1) * in_dim;
        const int8_t* w2 = W + (i+2) * in_dim;
        const int8_t* w3 = W + (i+3) * in_dim;
        const int8_t* w4 = W + (i+4) * in_dim;
        const int8_t* w5 = W + (i+5) * in_dim;
        const int8_t* w6 = W + (i+6) * in_dim;
        const int8_t* w7 = W + (i+7) * in_dim;

        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
        __m512 acc4 = _mm512_setzero_ps(), acc5 = _mm512_setzero_ps();
        __m512 acc6 = _mm512_setzero_ps(), acc7 = _mm512_setzero_ps();

        int32_t j = 0;
        for (; j + 16 <= in_dim; j += 16) {
            __m512 xv = _mm512_loadu_ps(x + j);
            __m128i raw0 = _mm_loadu_si128((const __m128i*)(w0+j));
            __m128i raw1 = _mm_loadu_si128((const __m128i*)(w1+j));
            __m128i raw2 = _mm_loadu_si128((const __m128i*)(w2+j));
            __m128i raw3 = _mm_loadu_si128((const __m128i*)(w3+j));
            __m128i raw4 = _mm_loadu_si128((const __m128i*)(w4+j));
            __m128i raw5 = _mm_loadu_si128((const __m128i*)(w5+j));
            __m128i raw6 = _mm_loadu_si128((const __m128i*)(w6+j));
            __m128i raw7 = _mm_loadu_si128((const __m128i*)(w7+j));
            acc0 = _mm512_fmadd_ps(xv, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(raw0)), acc0);
            acc1 = _mm512_fmadd_ps(xv, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(raw1)), acc1);
            acc2 = _mm512_fmadd_ps(xv, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(raw2)), acc2);
            acc3 = _mm512_fmadd_ps(xv, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(raw3)), acc3);
            acc4 = _mm512_fmadd_ps(xv, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(raw4)), acc4);
            acc5 = _mm512_fmadd_ps(xv, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(raw5)), acc5);
            acc6 = _mm512_fmadd_ps(xv, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(raw6)), acc6);
            acc7 = _mm512_fmadd_ps(xv, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(raw7)), acc7);
        }
        y[i+0] += hsum512(acc0);
        y[i+1] += hsum512(acc1);
        y[i+2] += hsum512(acc2);
        y[i+3] += hsum512(acc3);
        y[i+4] += hsum512(acc4);
        y[i+5] += hsum512(acc5);
        y[i+6] += hsum512(acc6);
        y[i+7] += hsum512(acc7);

        for (; j < in_dim; ++j) {
            float xj = x[j];
            if (w0[j]== 1) y[i+0]+=xj; else if (w0[j]==-1) y[i+0]-=xj;
            if (w1[j]== 1) y[i+1]+=xj; else if (w1[j]==-1) y[i+1]-=xj;
            if (w2[j]== 1) y[i+2]+=xj; else if (w2[j]==-1) y[i+2]-=xj;
            if (w3[j]== 1) y[i+3]+=xj; else if (w3[j]==-1) y[i+3]-=xj;
            if (w4[j]== 1) y[i+4]+=xj; else if (w4[j]==-1) y[i+4]-=xj;
            if (w5[j]== 1) y[i+5]+=xj; else if (w5[j]==-1) y[i+5]-=xj;
            if (w6[j]== 1) y[i+6]+=xj; else if (w6[j]==-1) y[i+6]-=xj;
            if (w7[j]== 1) y[i+7]+=xj; else if (w7[j]==-1) y[i+7]-=xj;
        }
    }
#elif defined(ASDAG_SIMD_AVX2)
    for (; i + 8 <= out_dim; i += 8) {
        const int8_t* w0 = W + (i+0) * in_dim;
        const int8_t* w1 = W + (i+1) * in_dim;
        const int8_t* w2 = W + (i+2) * in_dim;
        const int8_t* w3 = W + (i+3) * in_dim;
        const int8_t* w4 = W + (i+4) * in_dim;
        const int8_t* w5 = W + (i+5) * in_dim;
        const int8_t* w6 = W + (i+6) * in_dim;
        const int8_t* w7 = W + (i+7) * in_dim;

        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();

        int32_t j = 0;
        for (; j + 8 <= in_dim; j += 8) {
            __m256 xv = _mm256_loadu_ps(x + j);
            acc0 = _mm256_fmadd_ps(xv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(w0+j)))), acc0);
            acc1 = _mm256_fmadd_ps(xv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(w1+j)))), acc1);
            acc2 = _mm256_fmadd_ps(xv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(w2+j)))), acc2);
            acc3 = _mm256_fmadd_ps(xv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(w3+j)))), acc3);
            acc4 = _mm256_fmadd_ps(xv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(w4+j)))), acc4);
            acc5 = _mm256_fmadd_ps(xv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(w5+j)))), acc5);
            acc6 = _mm256_fmadd_ps(xv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(w6+j)))), acc6);
            acc7 = _mm256_fmadd_ps(xv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(w7+j)))), acc7);
        }
        y[i+0] += hsum256(acc0);
        y[i+1] += hsum256(acc1);
        y[i+2] += hsum256(acc2);
        y[i+3] += hsum256(acc3);
        y[i+4] += hsum256(acc4);
        y[i+5] += hsum256(acc5);
        y[i+6] += hsum256(acc6);
        y[i+7] += hsum256(acc7);

        for (; j < in_dim; ++j) {
            float xj = x[j];
            if (w0[j]== 1) y[i+0]+=xj; else if (w0[j]==-1) y[i+0]-=xj;
            if (w1[j]== 1) y[i+1]+=xj; else if (w1[j]==-1) y[i+1]-=xj;
            if (w2[j]== 1) y[i+2]+=xj; else if (w2[j]==-1) y[i+2]-=xj;
            if (w3[j]== 1) y[i+3]+=xj; else if (w3[j]==-1) y[i+3]-=xj;
            if (w4[j]== 1) y[i+4]+=xj; else if (w4[j]==-1) y[i+4]-=xj;
            if (w5[j]== 1) y[i+5]+=xj; else if (w5[j]==-1) y[i+5]-=xj;
            if (w6[j]== 1) y[i+6]+=xj; else if (w6[j]==-1) y[i+6]-=xj;
            if (w7[j]== 1) y[i+7]+=xj; else if (w7[j]==-1) y[i+7]-=xj;
        }
    }
#endif

    // Scalar remainder rows (always runs regardless of SIMD path)
    for (; i < out_dim; ++i) {
        const int8_t* w_row = W + i * in_dim;
        float acc = y[i];
        for (int32_t j = 0; j < in_dim; ++j) {
            if (w_row[j] ==  1) acc += x[j];
            else if (w_row[j] == -1) acc -= x[j];
        }
        y[i] = acc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hierarchical sign router  (AVX-512 / AVX2 / scalar)
// ─────────────────────────────────────────────────────────────────────────────
ASDAG_INLINE int32_t route_hierarchical_sign_simd(
    const float*  ASDAG_RESTRICT x,
    const int8_t* ASDAG_RESTRICT hyperplanes,
    const float*  ASDAG_RESTRICT biases,
    int32_t dim,
    int32_t tree_depth
) {
    int32_t node = 0;
    for (int32_t depth = 0; depth < tree_depth; ++depth) {
        const int8_t* plane = hyperplanes + node * dim;
        float dot = biases[node];

#if defined(ASDAG_SIMD_AVX512)
        __m512 acc = _mm512_setzero_ps();
        int32_t d = 0;
        for (; d + 16 <= dim; d += 16) {
            __m512  xv  = _mm512_loadu_ps(x + d);
            __m128i p8  = _mm_loadu_si128((const __m128i*)(plane + d));
            __m512  pv  = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(p8));
            acc = _mm512_fmadd_ps(xv, pv, acc);
        }
        dot += hsum512(acc);
        for (; d < dim; ++d) {
            if (plane[d] ==  1) dot += x[d];
            else if (plane[d] == -1) dot -= x[d];
        }
#elif defined(ASDAG_SIMD_AVX2)
        __m256 acc = _mm256_setzero_ps();
        int32_t d = 0;
        for (; d + 8 <= dim; d += 8) {
            __m256  xv  = _mm256_loadu_ps(x + d);
            __m128i p8  = _mm_loadl_epi64((const __m128i*)(plane + d));
            __m256  pv  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(p8));
            acc = _mm256_fmadd_ps(xv, pv, acc);
        }
        dot += hsum256(acc);
        for (; d < dim; ++d) {
            if (plane[d] ==  1) dot += x[d];
            else if (plane[d] == -1) dot -= x[d];
        }
#else
        for (int32_t d = 0; d < dim; ++d) {
            if (plane[d] ==  1) dot += x[d];
            else if (plane[d] == -1) dot -= x[d];
        }
#endif
        node = 2 * node + 1 + (dot > 0.0f ? 1 : 0);
    }
    return std::max(0, node - ((1 << tree_depth) - 1));
}

// ─────────────────────────────────────────────────────────────────────────────
// Branchless SIMD ReLU6 + norm activation (AVX2/scalar)
// ─────────────────────────────────────────────────────────────────────────────
ASDAG_INLINE void apply_norm_activation_simd(
    float* ASDAG_RESTRICT y,
    float  norm_factor,
    int32_t dim
) {
#if defined(ASDAG_SIMD_AVX2) || defined(ASDAG_SIMD_AVX512)
    __m256 nv   = _mm256_set1_ps(norm_factor);
    __m256 zero = _mm256_setzero_ps();
    __m256 six  = _mm256_set1_ps(6.0f);
    int32_t d = 0;
    for (; d + 8 <= dim; d += 8) {
        __m256 yv  = _mm256_loadu_ps(y + d);
        __m256 act = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(yv, nv), zero), six);
        _mm256_storeu_ps(y + d, act);
    }
    for (; d < dim; ++d) {
        float v = y[d] * norm_factor;
        y[d] = std::min(std::max(v, 0.0f), 6.0f);
    }
#else
    for (int32_t d = 0; d < dim; ++d) {
        float v = y[d] * norm_factor;
        y[d] = std::min(std::max(v, 0.0f), 6.0f);
    }
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// ASDAG Layer (dense int8 weights — microbenchmark / reference path)
// ─────────────────────────────────────────────────────────────────────────────
class ASDAGLayerCPP {
public:
    LayerConfig         cfg;
    std::vector<int8_t>  W_leaves;
    std::vector<float>   bias_leaves;
    std::vector<int8_t>  hyperplanes;
    std::vector<float>   router_biases;
    std::vector<float>   norm_factors;
    int32_t              tree_depth;
    int32_t              physical_cores;

    explicit ASDAGLayerCPP(LayerConfig config) : cfg(config) {
        tree_depth = std::max(1, (int32_t)std::ceil(std::log2((double)std::max(cfg.num_leaves, 2))));
        int32_t num_internal = (1 << tree_depth) - 1;
        physical_cores = get_physical_cores();

        W_leaves.resize(cfg.num_leaves * cfg.dim * cfg.dim, 0);
        bias_leaves.resize(cfg.num_leaves * cfg.dim, 0.0f);
        hyperplanes.resize(num_internal * cfg.dim, 0);
        router_biases.resize(num_internal, 0.0f);
        norm_factors.resize(cfg.num_leaves, 1.0f);
    }

    // [FIX-1+2] Replaced coroutine + proc_bind(close) with pthread pinning
    //           and a plain __builtin_prefetch with proper look-ahead distance.
    // [OPT-1]  schedule(static) — token batches are uniform work.
    void forward_batch(
        const float* ASDAG_RESTRICT X,
        float*       ASDAG_RESTRICT Y,
        int32_t B,
        int32_t prefetch_ahead = 4    // tokens to prefetch ahead
    ) const {
        int32_t n_threads = physical_cores;

#pragma omp parallel num_threads(n_threads)
        {
            // [FIX-1] Hard physical-core pinning (no SMT siblings)
            pin_thread_to_physical_core(omp_get_thread_num());

            // Per-thread scratch for bias init (avoids false sharing)
            std::vector<float> y_scratch(cfg.dim);

#pragma omp for schedule(static)           // [OPT-1]
            for (int32_t b = 0; b < B; ++b) {
                // [FIX-2] Plain prefetch with meaningful look-ahead
                if (b + prefetch_ahead < B) {
                    ASDAG_PREFETCH(X + (b + prefetch_ahead) * cfg.dim, 0, 2);
                }

                const float* x_b = X + b * cfg.dim;
                float*       y_b = Y + b * cfg.dim;

                int32_t leaf_idx = route_hierarchical_sign_simd(
                    x_b, hyperplanes.data(), router_biases.data(), cfg.dim, tree_depth);
                if (leaf_idx >= cfg.num_leaves) leaf_idx = cfg.num_leaves - 1;

                const int8_t* W_l = W_leaves.data() + leaf_idx * (cfg.dim * cfg.dim);
                const float*  b_l = bias_leaves.data() + leaf_idx * cfg.dim;
                float         nf  = norm_factors[leaf_idx];

                // Init y_scratch from bias
                std::memcpy(y_scratch.data(), b_l, cfg.dim * sizeof(float));

                ternary_shift4_gemv_8x(W_l, x_b, y_scratch.data(), cfg.dim, cfg.dim);
                apply_norm_activation_simd(y_scratch.data(), nf, cfg.dim);

                std::memcpy(y_b, y_scratch.data(), cfg.dim * sizeof(float));
            }
        }
    }

    // Keep old name as alias
    void forward_batch_coroutines(
        const float* ASDAG_RESTRICT X,
        float*       ASDAG_RESTRICT Y,
        int32_t B,
        int32_t /*tile_size*/ = 64
    ) const {
        forward_batch(X, Y, B);
    }
};

} // namespace asdag
