#include <torch/extension.h>
#include "../../cpp_infer/asdag_engine.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <atomic>
#include <functional>

#if defined(__AVX512BF16__)
    #include <immintrin.h>
    #define ASDAG_HAS_AVX512_BF16 1
#endif

namespace asdag_cpu {

// ─────────────────────────────────────────────────────────────────────────────
// Legacy Dense C++ Forward / Backward (for tests and comparison)
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor> asdag_forward_cpp(
    torch::Tensor x, torch::Tensor W_leaves, torch::Tensor biases, torch::Tensor routing_probs
) {
    x = x.contiguous().to(torch::kFloat32);
    W_leaves = W_leaves.contiguous().to(torch::kFloat32);
    biases = biases.contiguous().to(torch::kFloat32);
    routing_probs = routing_probs.contiguous().to(torch::kFloat32);

    int64_t B = x.size(0);
    int64_t dim = x.size(1);
    int64_t K = W_leaves.size(0);

    auto leaf_outs = torch::empty({B, K, dim}, x.options());
    auto composite_out = torch::zeros({B, dim}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* W_ptr = W_leaves.data_ptr<float>();
    const float* b_ptr = biases.data_ptr<float>();
    const float* r_ptr = routing_probs.data_ptr<float>();
    float* leaf_out_ptr = leaf_outs.data_ptr<float>();
    float* out_ptr = composite_out.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

#pragma omp parallel num_threads(n_threads)
    {
        asdag::pin_thread_to_physical_core(omp_get_thread_num());

#pragma omp for schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            const float* xb = x_ptr + b * dim;
            const float* rb = r_ptr + b * K;
            float* yb = out_ptr + b * dim;

            for (int64_t k = 0; k < K; ++k) {
                float* leaf_out_bk = leaf_out_ptr + (b * K + k) * dim;
                float prob_k = rb[k];

                if (prob_k <= 0.0f) {
                    std::memset(leaf_out_bk, 0, dim * sizeof(float));
                    continue;
                }

                const float* W_k = W_ptr + k * (dim * dim);
                const float* bias_k = b_ptr + k * dim;
                std::memcpy(leaf_out_bk, bias_k, dim * sizeof(float));

                for (int64_t i = 0; i < dim; ++i) {
                    const float* wr = W_k + i * dim;
                    float acc = 0.0f;
                    for (int64_t j = 0; j < dim; ++j) acc += wr[j] * xb[j];
                    leaf_out_bk[i] += acc;
                }

                for (int64_t d = 0; d < dim; ++d) {
                    float val = std::min(std::max(leaf_out_bk[d], 0.0f), 6.0f);
                    leaf_out_bk[d] = val;
                    yb[d] += prob_k * val;
                }
            }
        }
    }
    return std::make_tuple(composite_out, leaf_outs);
}

std::tuple<torch::Tensor, torch::Tensor> asdag_backward_cpp(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor W_leaves, torch::Tensor routing_probs, torch::Tensor leaf_outs
) {
    grad_output = grad_output.contiguous().to(torch::kFloat32);
    x = x.contiguous().to(torch::kFloat32);
    W_leaves = W_leaves.contiguous().to(torch::kFloat32);
    routing_probs = routing_probs.contiguous().to(torch::kFloat32);
    leaf_outs = leaf_outs.contiguous().to(torch::kFloat32);

    int64_t B = x.size(0);
    int64_t dim = x.size(1);
    int64_t K = W_leaves.size(0);

    auto grad_x = torch::zeros_like(x);
    auto grad_W = torch::zeros({K, dim, dim}, x.options());

    const float* go_ptr = grad_output.data_ptr<float>();
    const float* x_ptr = x.data_ptr<float>();
    const float* r_ptr = routing_probs.data_ptr<float>();
    const float* lo_ptr = leaf_outs.data_ptr<float>();
    const float* W_ptr = W_leaves.data_ptr<float>();
    float* gx_ptr = grad_x.data_ptr<float>();
    float* gw_ptr = grad_W.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

#pragma omp parallel num_threads(n_threads)
    {
        asdag::pin_thread_to_physical_core(omp_get_thread_num());

#pragma omp for schedule(static)
        for (int64_t k = 0; k < K; ++k) {
            float* gw_k = gw_ptr + k * (dim * dim);
            for (int64_t b = 0; b < B; ++b) {
                float prob_k = r_ptr[b * K + k];
                if (prob_k <= 0.0f) continue;
                const float* go_b = go_ptr + b * dim;
                const float* xb = x_ptr + b * dim;
                const float* lo_bk = lo_ptr + (b * K + k) * dim;
                for (int64_t d = 0; d < dim; ++d) {
                    float act_grad = (lo_bk[d] > 0.0f && lo_bk[d] < 6.0f) ? 1.0f : 0.0f;
                    float g_d = go_b[d] * prob_k * act_grad;
                    if (g_d != 0.0f) {
                        float* target_row = gw_k + d * dim;
                        for (int64_t i = 0; i < dim; ++i) target_row[i] += g_d * xb[i];
                    }
                }
            }
        }

#pragma omp for schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            const float* go_b = go_ptr + b * dim;
            float* gx_b = gx_ptr + b * dim;
            for (int64_t k = 0; k < K; ++k) {
                float prob_k = r_ptr[b * K + k];
                if (prob_k <= 0.0f) continue;
                const float* lo_bk = lo_ptr + (b * K + k) * dim;
                const float* W_k = W_ptr + k * (dim * dim);
                for (int64_t d = 0; d < dim; ++d) {
                    float act_grad = (lo_bk[d] > 0.0f && lo_bk[d] < 6.0f) ? 1.0f : 0.0f;
                    float g_d = go_b[d] * prob_k * act_grad;
                    if (g_d != 0.0f) {
                        const float* w_row = W_k + d * dim;
                        for (int64_t i = 0; i < dim; ++i) gx_b[i] += g_d * w_row[i];
                    }
                }
            }
        }
    }
    return std::make_tuple(grad_x, grad_W);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Fused Multi-Branch Permutation Projection (Q, K, V, Gate) Forward / Backward
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor asdag_fused_perm_proj_forward_cpp(
    torch::Tensor x,          // [B, dim] float32
    torch::Tensor w_fused,    // [M, P, dim] float32
    torch::Tensor perms,      // [P, dim] int32
    torch::Tensor biases      // [M, dim] float32
) {
    x = x.contiguous().to(torch::kFloat32);
    w_fused = w_fused.contiguous().to(torch::kFloat32);
    perms = perms.contiguous().to(torch::kInt32);
    biases = biases.contiguous().to(torch::kFloat32);

    int64_t B = x.size(0);
    int64_t dim = x.size(1);
    int64_t M = w_fused.size(0);
    int64_t P = w_fused.size(1);

    auto out = torch::empty({M, B, dim}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w_fused.data_ptr<float>();
    const int32_t* p_ptr = perms.data_ptr<int32_t>();
    const float* b_ptr = biases.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

#pragma omp parallel num_threads(n_threads)
    {
        asdag::pin_thread_to_physical_core(omp_get_thread_num());

#pragma omp for schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            const float* xb = x_ptr + b * dim;

            for (int64_t m = 0; m < M; ++m) {
                float* out_mb = out_ptr + (m * B + b) * dim;
                const float* bias_m = b_ptr + m * dim;
                std::memcpy(out_mb, bias_m, dim * sizeof(float));

                const float* wm = w_ptr + m * (P * dim);

                for (int64_t p_idx = 0; p_idx < P; ++p_idx) {
                    const float* wm_p = wm + p_idx * dim;
                    const int32_t* perm_p = p_ptr + p_idx * dim;

                    if (p_idx == 0) {
#if defined(ASDAG_SIMD_AVX512)
                        for (int64_t i = 0; i + 16 <= dim; i += 16) {
                            __m512 cur = _mm512_loadu_ps(out_mb + i);
                            __m512 wv = _mm512_loadu_ps(wm_p + i);
                            __m512 xv = _mm512_loadu_ps(xb + i);
                            _mm512_storeu_ps(out_mb + i, _mm512_fmadd_ps(wv, xv, cur));
                        }
#elif defined(ASDAG_SIMD_AVX2)
                        for (int64_t i = 0; i + 8 <= dim; i += 8) {
                            __m256 cur = _mm256_loadu_ps(out_mb + i);
                            __m256 wv = _mm256_loadu_ps(wm_p + i);
                            __m256 xv = _mm256_loadu_ps(xb + i);
                            _mm256_storeu_ps(out_mb + i, _mm256_fmadd_ps(wv, xv, cur));
                        }
#else
                        for (int64_t i = 0; i < dim; ++i) out_mb[i] += wm_p[i] * xb[i];
#endif
                    } else {
#if defined(ASDAG_SIMD_AVX512)
                        for (int64_t i = 0; i + 16 <= dim; i += 16) {
                            __m512 cur = _mm512_loadu_ps(out_mb + i);
                            __m512 wv = _mm512_loadu_ps(wm_p + i);
                            __m512i p_idx_v = _mm512_loadu_si512((const __m512i*)(perm_p + i));
                            __m512 xv = _mm512_i32gather_ps(p_idx_v, xb, 4);
                            _mm512_storeu_ps(out_mb + i, _mm512_fmadd_ps(wv, xv, cur));
                        }
#elif defined(ASDAG_SIMD_AVX2)
                        for (int64_t i = 0; i + 8 <= dim; i += 8) {
                            __m256 cur = _mm256_loadu_ps(out_mb + i);
                            __m256 wv = _mm256_loadu_ps(wm_p + i);
                            __m256i p_idx_v = _mm256_loadu_si256((const __m256i*)(perm_p + i));
                            __m256 xv = _mm256_i32gather_ps(xb, p_idx_v, 4);
                            _mm256_storeu_ps(out_mb + i, _mm256_fmadd_ps(wv, xv, cur));
                        }
#else
                        for (int64_t i = 0; i < dim; ++i) out_mb[i] += wm_p[i] * xb[perm_p[i]];
#endif
                    }
                }
            }
        }
    }

    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> asdag_fused_perm_proj_backward_cpp(
    torch::Tensor grad_output, // [M, B, dim] float32
    torch::Tensor x,           // [B, dim] float32
    torch::Tensor w_fused,     // [M, P, dim] float32
    torch::Tensor perms,       // [P, dim] int32
    torch::Tensor inv_perms    // [P, dim] int32
) {
    grad_output = grad_output.contiguous().to(torch::kFloat32);
    x = x.contiguous().to(torch::kFloat32);
    w_fused = w_fused.contiguous().to(torch::kFloat32);
    perms = perms.contiguous().to(torch::kInt32);
    inv_perms = inv_perms.contiguous().to(torch::kInt32);

    int64_t M = grad_output.size(0);
    int64_t B = grad_output.size(1);
    int64_t dim = grad_output.size(2);
    int64_t P = w_fused.size(1);

    auto grad_x = torch::zeros_like(x);
    auto grad_w = torch::zeros_like(w_fused);
    auto grad_bias = grad_output.sum(1); // [M, dim]

    const float* go_ptr = grad_output.data_ptr<float>();
    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w_fused.data_ptr<float>();
    const int32_t* p_ptr = perms.data_ptr<int32_t>();
    const int32_t* ip_ptr = inv_perms.data_ptr<int32_t>();
    float* gx_ptr = grad_x.data_ptr<float>();
    float* gw_ptr = grad_w.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

#pragma omp parallel num_threads(n_threads)
    {
        asdag::pin_thread_to_physical_core(omp_get_thread_num());

        // 1. Parallel Weight Gradients per branch
#pragma omp for schedule(static)
        for (int64_t m = 0; m < M; ++m) {
            float* gw_m = gw_ptr + m * (P * dim);
            const float* go_m = go_ptr + m * (B * dim);

            for (int64_t p_idx = 0; p_idx < P; ++p_idx) {
                float* gw_mp = gw_m + p_idx * dim;
                const int32_t* perm_p = p_ptr + p_idx * dim;

                for (int64_t b = 0; b < B; ++b) {
                    const float* go_mb = go_m + b * dim;
                    const float* xb = x_ptr + b * dim;

                    if (p_idx == 0) {
                        for (int64_t d = 0; d < dim; ++d) gw_mp[d] += go_mb[d] * xb[d];
                    } else {
                        for (int64_t d = 0; d < dim; ++d) gw_mp[d] += go_mb[d] * xb[perm_p[d]];
                    }
                }
            }
        }

        // 2. Parallel Input Gradient Accumulation
#pragma omp for schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            float* gx_b = gx_ptr + b * dim;

            for (int64_t m = 0; m < M; ++m) {
                const float* go_mb = go_ptr + (m * B + b) * dim;
                const float* wm = w_ptr + m * (P * dim);

                for (int64_t p_idx = 0; p_idx < P; ++p_idx) {
                    const float* wm_p = wm + p_idx * dim;
                    const int32_t* inv_perm_p = ip_ptr + p_idx * dim;

                    if (p_idx == 0) {
                        for (int64_t j = 0; j < dim; ++j) gx_b[j] += go_mb[j] * wm_p[j];
                    } else {
                        for (int64_t j = 0; j < dim; ++j) {
                            int32_t src_idx = inv_perm_p[j];
                            gx_b[j] += go_mb[src_idx] * wm_p[src_idx];
                        }
                    }
                }
            }
        }
    }

    return std::make_tuple(grad_x, grad_w, grad_bias);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Native Gated Linear Associative (GLA) Sequence Mixer SIMD Engine
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> asdag_gla_step_cpp(
    torch::Tensor q_t,     // [B, H, D] float32 (phi(q))
    torch::Tensor k_t,     // [B, H, D] float32 (phi(k))
    torch::Tensor v_t,     // [B, H, D] float32
    torch::Tensor gamma_t, // [B, H] float32 (decay in (0, 1))
    torch::Tensor state_S, // [B, H, D, D] float32
    torch::Tensor state_z, // [B, H, D] float32
    float eps
) {
    q_t = q_t.contiguous().to(torch::kFloat32);
    k_t = k_t.contiguous().to(torch::kFloat32);
    v_t = v_t.contiguous().to(torch::kFloat32);
    gamma_t = gamma_t.contiguous().to(torch::kFloat32);
    state_S = state_S.contiguous().to(torch::kFloat32);
    state_z = state_z.contiguous().to(torch::kFloat32);

    int64_t B = q_t.size(0);
    int64_t H = q_t.size(1);
    int64_t D = q_t.size(2);

    auto out_y = torch::empty({B, H, D}, q_t.options());

    const float* q_ptr = q_t.data_ptr<float>();
    const float* k_ptr = k_t.data_ptr<float>();
    const float* v_ptr = v_t.data_ptr<float>();
    const float* g_ptr = gamma_t.data_ptr<float>();
    float* S_ptr = state_S.data_ptr<float>();
    float* z_ptr = state_z.data_ptr<float>();
    float* y_ptr = out_y.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

#pragma omp parallel for collapse(2) num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            int64_t bh_offset_d = (b * H + h) * D;
            int64_t bh_offset_dd = (b * H + h) * (D * D);
            int64_t bh_idx = b * H + h;

            const float* q_bh = q_ptr + bh_offset_d;
            const float* k_bh = k_ptr + bh_offset_d;
            const float* v_bh = v_ptr + bh_offset_d;
            float gam = g_ptr[bh_idx];

            float* S_bh = S_ptr + bh_offset_dd;
            float* z_bh = z_ptr + bh_offset_d;
            float* y_bh = y_ptr + bh_offset_d;

            // 1. Update normalizer z: z = gamma * z + k
            float den = 0.0f;
            for (int64_t d = 0; d < D; ++d) {
                float z_new = gam * z_bh[d] + k_bh[d];
                z_bh[d] = z_new;
                den += q_bh[d] * z_new;
            }
            den = std::max(den, eps);
            float inv_den = 1.0f / den;

            // 2. Update memory matrix S: S = gamma * S + k^T * v
            // 3. Compute unnormalized output: num = q * S
            for (int64_t j = 0; j < D; ++j) {
                float num_j = 0.0f;
                float vj = v_bh[j];
                for (int64_t i = 0; i < D; ++i) {
                    float s_old = S_bh[i * D + j];
                    float s_new = gam * s_old + k_bh[i] * vj;
                    S_bh[i * D + j] = s_new;
                    num_j += q_bh[i] * s_new;
                }
                y_bh[j] = num_j * inv_den;
            }
        }
    }

    return std::make_tuple(out_y, state_S, state_z);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. C++ Monarch Permutation Chain Forward (AVX2 / AVX-512 SIMD)
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor asdag_monarch_chain_forward_cpp(
    torch::Tensor x,         // [B, dim] float32 or bfloat16
    torch::Tensor diagonals, // [L, dim] float32 or bfloat16
    torch::Tensor perms,     // [L-1, dim] int32 or int64
    torch::Tensor bias       // [dim] float32 or bfloat16
) {
    auto orig_dtype = x.scalar_type();
    x = x.contiguous().to(torch::kFloat32);
    diagonals = diagonals.contiguous().to(torch::kFloat32);
    perms = perms.contiguous().to(torch::kInt32);
    bias = bias.contiguous().to(torch::kFloat32);

    int64_t B = x.size(0);
    int64_t dim = x.size(1);
    int64_t L = diagonals.size(0);

    auto out = torch::empty({B, dim}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* d_ptr = diagonals.data_ptr<float>();
    const int32_t* p_ptr = perms.data_ptr<int32_t>();
    const float* b_ptr = bias.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

#pragma omp parallel num_threads(n_threads)
    {
        asdag::pin_thread_to_physical_core(omp_get_thread_num());
        std::vector<float> h_buf1(dim);
        std::vector<float> h_buf2(dim);

#pragma omp for schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            const float* xb = x_ptr + b * dim;
            float* yb = out_ptr + b * dim;
            float* h_cur = h_buf1.data();
            float* h_next = h_buf2.data();

            // Stage 0: h = x * D[0]
            const float* d0 = d_ptr;
            int64_t i = 0;
#if defined(ASDAG_SIMD_AVX512)
            for (; i + 16 <= dim; i += 16) {
                __m512 xv = _mm512_loadu_ps(xb + i);
                __m512 dv = _mm512_loadu_ps(d0 + i);
                _mm512_storeu_ps(h_cur + i, _mm512_mul_ps(xv, dv));
            }
#elif defined(ASDAG_SIMD_AVX2)
            for (; i + 8 <= dim; i += 8) {
                __m256 xv = _mm256_loadu_ps(xb + i);
                __m256 dv = _mm256_loadu_ps(d0 + i);
                _mm256_storeu_ps(h_cur + i, _mm256_mul_ps(xv, dv));
            }
#endif
            for (; i < dim; ++i) {
                h_cur[i] = xb[i] * d0[i];
            }

            // Stages 1 to L-1: h_next[i] = h_cur[perm[s-1, i]] * D[s, i]
            for (int64_t s = 0; s < L - 1; ++s) {
                const int32_t* perm_s = p_ptr + s * dim;
                const float* ds = d_ptr + (s + 1) * dim;

                for (int64_t j = 0; j < dim; ++j) {
                    h_next[j] = h_cur[perm_s[j]] * ds[j];
                }
                std::swap(h_cur, h_next);
            }

            // Add bias: yb[i] = h_cur[i] + b_ptr[i]
            i = 0;
#if defined(ASDAG_SIMD_AVX512)
            for (; i + 16 <= dim; i += 16) {
                __m512 hv = _mm512_loadu_ps(h_cur + i);
                __m512 bv = _mm512_loadu_ps(b_ptr + i);
                _mm512_storeu_ps(yb + i, _mm512_add_ps(hv, bv));
            }
#elif defined(ASDAG_SIMD_AVX2)
            for (; i + 8 <= dim; i += 8) {
                __m256 hv = _mm256_loadu_ps(h_cur + i);
                __m256 bv = _mm256_loadu_ps(b_ptr + i);
                _mm256_storeu_ps(yb + i, _mm256_add_ps(hv, bv));
            }
#endif
            for (; i < dim; ++i) {
                yb[i] = h_cur[i] + b_ptr[i];
            }
        }
    }

    return out.to(orig_dtype);
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. C++ Fused Monarch Permutation Chain Forward (AVX2 / AVX-512 SIMD)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<torch::Tensor> asdag_fused_monarch_chain_forward_cpp(
    torch::Tensor x,         // [B, dim]
    torch::Tensor diagonals, // [M, L, dim]
    torch::Tensor perms,     // [L-1, dim]
    torch::Tensor bias       // [M, dim]
) {
    auto orig_dtype = x.scalar_type();
    x = x.contiguous().to(torch::kFloat32);
    diagonals = diagonals.contiguous().to(torch::kFloat32);
    perms = perms.contiguous().to(torch::kInt32);
    bias = bias.contiguous().to(torch::kFloat32);

    int64_t B = x.size(0);
    int64_t dim = x.size(1);
    int64_t M = diagonals.size(0);
    int64_t L = diagonals.size(1);

    auto out_all = torch::empty({M, B, dim}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* d_ptr = diagonals.data_ptr<float>();
    const int32_t* p_ptr = perms.data_ptr<int32_t>();
    const float* b_ptr = bias.data_ptr<float>();
    float* out_ptr = out_all.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

#pragma omp parallel num_threads(n_threads)
    {
        asdag::pin_thread_to_physical_core(omp_get_thread_num());
        std::vector<float> h_buf1(dim);
        std::vector<float> h_buf2(dim);

#pragma omp for collapse(2) schedule(static)
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t b = 0; b < B; ++b) {
                const float* xb = x_ptr + b * dim;
                const float* d_m = d_ptr + m * (L * dim);
                const float* b_m = b_ptr + m * dim;
                float* y_mb = out_ptr + (m * B + b) * dim;

                float* h_cur = h_buf1.data();
                float* h_next = h_buf2.data();

                // Stage 0: h = x * D[m, 0]
                int64_t i = 0;
#if defined(ASDAG_SIMD_AVX512)
                for (; i + 16 <= dim; i += 16) {
                    __m512 xv = _mm512_loadu_ps(xb + i);
                    __m512 dv = _mm512_loadu_ps(d_m + i);
                    _mm512_storeu_ps(h_cur + i, _mm512_mul_ps(xv, dv));
                }
#elif defined(ASDAG_SIMD_AVX2)
                for (; i + 8 <= dim; i += 8) {
                    __m256 xv = _mm256_loadu_ps(xb + i);
                    __m256 dv = _mm256_loadu_ps(d_m + i);
                    _mm256_storeu_ps(h_cur + i, _mm256_mul_ps(xv, dv));
                }
#endif
                for (; i < dim; ++i) {
                    h_cur[i] = xb[i] * d_m[i];
                }

                // Stages 1 to L-1
                for (int64_t s = 0; s < L - 1; ++s) {
                    const int32_t* perm_s = p_ptr + s * dim;
                    const float* ds = d_m + (s + 1) * dim;

                    for (int64_t j = 0; j < dim; ++j) {
                        h_next[j] = h_cur[perm_s[j]] * ds[j];
                    }
                    std::swap(h_cur, h_next);
                }

                // Add bias
                i = 0;
#if defined(ASDAG_SIMD_AVX512)
                for (; i + 16 <= dim; i += 16) {
                    __m512 hv = _mm512_loadu_ps(h_cur + i);
                    __m512 bv = _mm512_loadu_ps(b_m + i);
                    _mm512_storeu_ps(y_mb + i, _mm512_add_ps(hv, bv));
                }
#elif defined(ASDAG_SIMD_AVX2)
                for (; i + 8 <= dim; i += 8) {
                    __m256 hv = _mm256_loadu_ps(h_cur + i);
                    __m256 bv = _mm256_loadu_ps(b_m + i);
                    _mm256_storeu_ps(y_mb + i, _mm256_add_ps(hv, bv));
                }
#endif
                for (; i < dim; ++i) {
                    y_mb[i] = h_cur[i] + b_m[i];
                }
            }
        }
    }

    std::vector<torch::Tensor> res;
    for (int64_t m = 0; m < M; ++m) {
        res.push_back(out_all[m].to(orig_dtype));
    }
    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. C++ Ternary BitLinear Forward Pass (Pure Integer Sign Accumulation)
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor asdag_bitlinear_forward_cpp(
    torch::Tensor x,         // [B, in_dim]
    torch::Tensor w_ternary, // [out_dim, in_dim] {-1, 0, +1}
    float gamma,             // weight scale
    torch::Tensor bias       // optional [out_dim]
) {
    auto orig_dtype = x.scalar_type();
    x = x.contiguous().to(torch::kFloat32);
    w_ternary = w_ternary.contiguous().to(torch::kFloat32);

    int64_t B = x.size(0);
    int64_t in_dim = x.size(1);
    int64_t out_dim = w_ternary.size(0);

    auto out = torch::empty({B, out_dim}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w_ternary.data_ptr<float>();
    const float* b_ptr = bias.defined() ? bias.contiguous().to(torch::kFloat32).data_ptr<float>() : nullptr;
    float* out_ptr = out.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

    auto scale_x = torch::empty({B}, torch::kFloat32);
    auto inv_scale_x = torch::empty({B}, torch::kFloat32);
    float* sx_ptr = scale_x.data_ptr<float>();
    float* isx_ptr = inv_scale_x.data_ptr<float>();
#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        const float* xb = x_ptr + b * in_dim;
        float max_val = 1e-5f;
        for (int64_t i = 0; i < in_dim; ++i) {
            max_val = std::max(max_val, std::abs(xb[i]));
        }
        sx_ptr[b] = 127.0f / max_val;
        isx_ptr[b] = 1.0f / sx_ptr[b];
    }
#pragma omp parallel for collapse(2) num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t o = 0; o < out_dim; ++o) {
            const float* xb = x_ptr + b * in_dim;
            const float* wo = w_ptr + o * in_dim;
            float sxx = sx_ptr[b];
            float inv_sxx = isx_ptr[b];

            float acc = 0.0f;
            int64_t i = 0;
#if defined(ASDAG_SIMD_AVX512)
            __m512 acc_v = _mm512_setzero_ps();
            __m512 sx_v = _mm512_set1_ps(sxx);
            for (; i + 16 <= in_dim; i += 16) {
                __m512 xv = _mm512_loadu_ps(xb + i);
                __m512 wv = _mm512_loadu_ps(wo + i);
                __m512 xq = _mm512_roundscale_ps(_mm512_mul_ps(xv, sx_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                acc_v = _mm512_fmadd_ps(xq, wv, acc_v);
            }
            acc += _mm512_reduce_add_ps(acc_v);
#elif defined(ASDAG_SIMD_AVX2)
            __m256 acc_v = _mm256_setzero_ps();
            __m256 sx_v = _mm256_set1_ps(sxx);
            for (; i + 8 <= in_dim; i += 8) {
                __m256 xv = _mm256_loadu_ps(xb + i);
                __m256 wv = _mm256_loadu_ps(wo + i);
                __m256 xq = _mm256_round_ps(_mm256_mul_ps(xv, sx_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                acc_v = _mm256_fmadd_ps(xq, wv, acc_v);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp, acc_v);
            for (int k = 0; k < 8; ++k) acc += tmp[k];
#endif
            for (; i < in_dim; ++i) {
                float xq = std::round(xb[i] * sxx);
                acc += xq * wo[i];
            }

            float y_val = acc * inv_sxx * gamma;
            if (b_ptr) y_val += b_ptr[o];
            out_ptr[b * out_dim + o] = y_val;
        }
    }

    return out.to(orig_dtype);
}

torch::Tensor asdag_bitlinear_twin_forward_cpp(
    torch::Tensor x,         // [B, in_dim]
    torch::Tensor w1_ternary,// [O, in_dim] {-1, 0, +1}
    float gamma1,
    torch::Tensor w2_ternary,// [O, in_dim] {-1, 0, +1}
    float gamma2,
    torch::Tensor bias       // optional [2*O]
) {
    auto orig_dtype = x.scalar_type();
    x = x.contiguous().to(torch::kFloat32);
    w1_ternary = w1_ternary.contiguous().to(torch::kFloat32);
    w2_ternary = w2_ternary.contiguous().to(torch::kFloat32);

    int64_t B = x.size(0);
    int64_t in_dim = x.size(1);
    int64_t O = w1_ternary.size(0);

    auto out = torch::empty({B, 2 * O}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* w1_ptr = w1_ternary.data_ptr<float>();
    const float* w2_ptr = w2_ternary.data_ptr<float>();
    const float* b_ptr = bias.defined() ? bias.contiguous().to(torch::kFloat32).data_ptr<float>() : nullptr;
    float* out_ptr = out.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

    auto scale_x = torch::empty({B}, torch::kFloat32);
    auto inv_scale_x = torch::empty({B}, torch::kFloat32);
    float* sx_ptr = scale_x.data_ptr<float>();
    float* isx_ptr = inv_scale_x.data_ptr<float>();
#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        const float* xb = x_ptr + b * in_dim;
        float max_val = 1e-5f;
        for (int64_t i = 0; i < in_dim; ++i) {
            max_val = std::max(max_val, std::abs(xb[i]));
        }
        sx_ptr[b] = 127.0f / max_val;
        isx_ptr[b] = 1.0f / sx_ptr[b];
    }
#pragma omp parallel for collapse(2) num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t o = 0; o < 2 * O; ++o) {
            const float* xb = x_ptr + b * in_dim;
            bool first = o < O;
            const float* wo = first ? (w1_ptr + o * in_dim) : (w2_ptr + (o - O) * in_dim);
            float gam = first ? gamma1 : gamma2;
            float sxx = sx_ptr[b];
            float inv_sxx = isx_ptr[b];

            float acc = 0.0f;
            int64_t i = 0;
#if defined(ASDAG_SIMD_AVX512)
            __m512 acc_v = _mm512_setzero_ps();
            __m512 sx_v = _mm512_set1_ps(sxx);
            for (; i + 16 <= in_dim; i += 16) {
                __m512 xv = _mm512_loadu_ps(xb + i);
                __m512 wv = _mm512_loadu_ps(wo + i);
                __m512 xq = _mm512_roundscale_ps(_mm512_mul_ps(xv, sx_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                acc_v = _mm512_fmadd_ps(xq, wv, acc_v);
            }
            acc += _mm512_reduce_add_ps(acc_v);
#elif defined(ASDAG_SIMD_AVX2)
            __m256 acc_v = _mm256_setzero_ps();
            __m256 sx_v = _mm256_set1_ps(sxx);
            for (; i + 8 <= in_dim; i += 8) {
                __m256 xv = _mm256_loadu_ps(xb + i);
                __m256 wv = _mm256_loadu_ps(wo + i);
                __m256 xq = _mm256_round_ps(_mm256_mul_ps(xv, sx_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                acc_v = _mm256_fmadd_ps(xq, wv, acc_v);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp, acc_v);
            for (int k = 0; k < 8; ++k) acc += tmp[k];
#endif
            for (; i < in_dim; ++i) {
                float xq = std::round(xb[i] * sxx);
                acc += xq * wo[i];
            }

            float y_val = acc * inv_sxx * gam;
            if (b_ptr) y_val += b_ptr[o];
            out_ptr[b * 2 * O + o] = y_val;
        }
    }

    return out.to(orig_dtype);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> asdag_bitlinear_twin_backward_cpp(
    torch::Tensor grad_output, // [B, 2*O]
    torch::Tensor x,           // [B, in_dim]
    torch::Tensor w1_ternary,  // [O, in_dim]
    float gamma1,
    torch::Tensor w2_ternary,  // [O, in_dim]
    float gamma2,
    bool has_bias
) {
    auto orig_dtype = grad_output.scalar_type();
    grad_output = grad_output.contiguous().to(torch::kFloat32);
    x = x.contiguous().to(torch::kFloat32);
    w1_ternary = w1_ternary.contiguous().to(torch::kFloat32);
    w2_ternary = w2_ternary.contiguous().to(torch::kFloat32);

    int64_t B = x.size(0);
    int64_t in_dim = x.size(1);
    int64_t O = w1_ternary.size(0);

    auto grad_x = torch::empty({B, in_dim}, x.options());
    auto grad_w1 = torch::zeros({O, in_dim}, x.options());
    auto grad_w2 = torch::zeros({O, in_dim}, x.options());
    auto grad_bias = has_bias ? torch::zeros({2 * O}, x.options()) : torch::tensor({}, x.options());

    const float* go_ptr = grad_output.data_ptr<float>();
    const float* x_ptr = x.data_ptr<float>();
    const float* w1_ptr = w1_ternary.data_ptr<float>();
    const float* w2_ptr = w2_ternary.data_ptr<float>();

    float* gx_ptr = grad_x.data_ptr<float>();
    float* gw1_ptr = grad_w1.data_ptr<float>();
    float* gw2_ptr = grad_w2.data_ptr<float>();
    float* gb_ptr = has_bias ? grad_bias.data_ptr<float>() : nullptr;

    int n_threads = asdag::get_physical_cores();

    auto x_quant = torch::empty({B, in_dim}, torch::kFloat32);
    float* xq_ptr = x_quant.data_ptr<float>();

#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        const float* xb = x_ptr + b * in_dim;
        float* xqb = xq_ptr + b * in_dim;

        float max_val = 1e-5f;
        for (int64_t i = 0; i < in_dim; ++i) max_val = std::max(max_val, std::abs(xb[i]));
        float scale_x = 127.0f / max_val;
        float inv_scale_x = 1.0f / scale_x;

        int64_t i = 0;
#if defined(ASDAG_SIMD_AVX512)
        __m512 sx_v = _mm512_set1_ps(scale_x);
        __m512 isx_v = _mm512_set1_ps(inv_scale_x);
        for (; i + 16 <= in_dim; i += 16) {
            __m512 xv = _mm512_loadu_ps(xb + i);
            __m512 qv = _mm512_roundscale_ps(_mm512_mul_ps(xv, sx_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm512_storeu_ps(xqb + i, _mm512_mul_ps(qv, isx_v));
        }
#elif defined(ASDAG_SIMD_AVX2)
        __m256 sx_v = _mm256_set1_ps(scale_x);
        __m256 isx_v = _mm256_set1_ps(inv_scale_x);
        for (; i + 8 <= in_dim; i += 8) {
            __m256 xv = _mm256_loadu_ps(xb + i);
            __m256 qv = _mm256_round_ps(_mm256_mul_ps(xv, sx_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm256_storeu_ps(xqb + i, _mm256_mul_ps(qv, isx_v));
        }
#endif
        for (; i < in_dim; ++i) {
            xqb[i] = std::round(xb[i] * scale_x) * inv_scale_x;
        }
    }

    std::vector<std::vector<float>> t_gw1(n_threads, std::vector<float>((size_t)O * in_dim, 0.f));
    std::vector<std::vector<float>> t_gw2(n_threads, std::vector<float>((size_t)O * in_dim, 0.f));
    std::vector<std::vector<float>> t_gb(n_threads, has_bias ? std::vector<float>(2 * (size_t)O, 0.f) : std::vector<float>());
#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        asdag::pin_thread_to_physical_core(tid);
#pragma omp for schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            const float* gob = go_ptr + b * 2 * O;
            const float* xqb = xq_ptr + b * in_dim;
            float* gxb = gx_ptr + b * in_dim;
            std::memset(gxb, 0, in_dim * sizeof(float));

            for (int64_t o = 0; o < 2 * O; ++o) {
                bool first = o < O;
                int64_t oo = first ? o : o - O;
                float go_scaled = gob[o] * (first ? gamma1 : gamma2);
                const float* wo = first ? (w1_ptr + oo * in_dim) : (w2_ptr + oo * in_dim);
                float* gw = first ? t_gw1[tid].data() : t_gw2[tid].data();

                int64_t i = 0;
#if defined(ASDAG_SIMD_AVX512)
                __m512 go_vec = _mm512_set1_ps(go_scaled);
                for (; i + 16 <= in_dim; i += 16) {
                    __m512 w_vec = _mm512_loadu_ps(wo + i);
                    __m512 gx_vec = _mm512_loadu_ps(gxb + i);
                    _mm512_storeu_ps(gxb + i, _mm512_fmadd_ps(go_vec, w_vec, gx_vec));
                }
#elif defined(ASDAG_SIMD_AVX2)
                __m256 go_vec = _mm256_set1_ps(go_scaled);
                for (; i + 8 <= in_dim; i += 8) {
                    __m256 w_vec = _mm256_loadu_ps(wo + i);
                    __m256 gx_vec = _mm256_loadu_ps(gxb + i);
                    _mm256_storeu_ps(gxb + i, _mm256_fmadd_ps(go_vec, w_vec, gx_vec));
                }
#endif
                for (; i < in_dim; ++i) gxb[i] += go_scaled * wo[i];
                float* gow = gw + oo * in_dim;
                for (int64_t k = 0; k < in_dim; ++k) gow[k] += gob[o] * xqb[k];
                if (has_bias) t_gb[tid][o] += gob[o];
            }
        }
    }
    for (int t = 0; t < n_threads; ++t) {
        float* a = gw1_ptr;
        float* b = gw2_ptr;
        const float* x1 = t_gw1[t].data();
        const float* x2 = t_gw2[t].data();
        for (int64_t i = 0; i < (int64_t)O * in_dim; ++i) { a[i] += x1[i]; b[i] += x2[i]; }
        if (has_bias) {
            float* g = gb_ptr;
            const float* xg = t_gb[t].data();
            for (int64_t i = 0; i < 2 * O; ++i) g[i] += xg[i];
        }
    }

    return std::make_tuple(
        grad_x.to(orig_dtype),
        grad_w1.to(orig_dtype),
        grad_w2.to(orig_dtype),
        grad_bias.to(orig_dtype)
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. C++ Monarch Permutation Chain Backward (AVX2 / AVX-512 SIMD)
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> asdag_monarch_chain_backward_cpp(
    torch::Tensor grad_output, // [B, dim]
    torch::Tensor x,           // [B, dim]
    torch::Tensor diagonals,   // [L, dim]
    torch::Tensor perms,       // [L-1, dim]
    torch::Tensor inv_perms    // [L-1, dim]
) {
    auto orig_dtype = grad_output.scalar_type();
    grad_output = grad_output.contiguous().to(torch::kFloat32);
    x = x.contiguous().to(torch::kFloat32);
    diagonals = diagonals.contiguous().to(torch::kFloat32);
    perms = perms.contiguous().to(torch::kInt32);
    inv_perms = inv_perms.contiguous().to(torch::kInt32);

    int64_t B = x.size(0);
    int64_t dim = x.size(1);
    int64_t L = diagonals.size(0);

    auto grad_x = torch::empty({B, dim}, x.options());
    auto grad_diagonals = torch::zeros({L, dim}, x.options());
    auto grad_bias = torch::zeros({dim}, x.options());

    const float* go_ptr = grad_output.data_ptr<float>();
    const float* x_ptr = x.data_ptr<float>();
    const float* d_ptr = diagonals.data_ptr<float>();
    const int32_t* p_ptr = perms.data_ptr<int32_t>();
    const int32_t* ip_ptr = inv_perms.data_ptr<int32_t>();

    float* gx_ptr = grad_x.data_ptr<float>();
    float* gd_ptr = grad_diagonals.data_ptr<float>();
    float* gb_ptr = grad_bias.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();
    std::vector<std::vector<float>> thread_gd(n_threads, std::vector<float>(L * dim, 0.0f));
    std::vector<std::vector<float>> thread_gb(n_threads, std::vector<float>(dim, 0.0f));

#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        asdag::pin_thread_to_physical_core(tid);
        float* local_gd = thread_gd[tid].data();
        float* local_gb = thread_gb[tid].data();

        std::vector<float> h_buf(L * dim);
        std::vector<float> g_cur(dim);
        std::vector<float> g_prev(dim);

#pragma omp for schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            const float* xb = x_ptr + b * dim;
            const float* gob = go_ptr + b * dim;
            float* gxb = gx_ptr + b * dim;

            // Bias grad accumulation
            for (int64_t i = 0; i < dim; ++i) local_gb[i] += gob[i];

            // 1. Forward stage recomputation
            float* h0 = h_buf.data();
            for (int64_t i = 0; i < dim; ++i) h0[i] = xb[i] * d_ptr[i];

            for (int64_t s = 0; s < L - 1; ++s) {
                const int32_t* perm_s = p_ptr + s * dim;
                const float* ds = d_ptr + (s + 1) * dim;
                const float* h_prev = h_buf.data() + s * dim;
                float* h_next = h_buf.data() + (s + 1) * dim;

                for (int64_t i = 0; i < dim; ++i) {
                    h_next[i] = h_prev[perm_s[i]] * ds[i];
                }
            }

            // 2. Backward propagation
            std::memcpy(g_cur.data(), gob, dim * sizeof(float));

            for (int64_t s = L - 1; s >= 1; --s) {
                const int32_t* perm_prev = p_ptr + (s - 1) * dim;
                const int32_t* inv_perm_prev = ip_ptr + (s - 1) * dim;
                const float* ds = d_ptr + s * dim;
                const float* h_prev = h_buf.data() + (s - 1) * dim;
                float* local_gd_s = local_gd + s * dim;

                for (int64_t i = 0; i < dim; ++i) {
                    local_gd_s[i] += g_cur[i] * h_prev[perm_prev[i]];
                }
                for (int64_t i = 0; i < dim; ++i) {
                    g_prev[inv_perm_prev[i]] = g_cur[i] * ds[i];
                }
                std::memcpy(g_cur.data(), g_prev.data(), dim * sizeof(float));
            }

            // Stage 0
            for (int64_t i = 0; i < dim; ++i) {
                local_gd[i] += g_cur[i] * xb[i];
                gxb[i] = g_cur[i] * d_ptr[i];
            }
        }
    }

    for (int t = 0; t < n_threads; ++t) {
        for (int64_t idx = 0; idx < L * dim; ++idx) gd_ptr[idx] += thread_gd[t][idx];
        for (int64_t idx = 0; idx < dim; ++idx) gb_ptr[idx] += thread_gb[t][idx];
    }

    return std::make_tuple(grad_x.to(orig_dtype), grad_diagonals.to(orig_dtype), grad_bias.to(orig_dtype));
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. C++ Fused Monarch Permutation Chain Backward (AVX2 / AVX-512 SIMD)
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> asdag_fused_monarch_chain_backward_cpp(
    torch::Tensor grad_output, // [M, B, dim]
    torch::Tensor x,           // [B, dim]
    torch::Tensor diagonals,   // [M, L, dim]
    torch::Tensor perms,       // [L-1, dim]
    torch::Tensor inv_perms    // [L-1, dim]
) {
    auto orig_dtype = grad_output.scalar_type();
    grad_output = grad_output.contiguous().to(torch::kFloat32);
    x = x.contiguous().to(torch::kFloat32);
    diagonals = diagonals.contiguous().to(torch::kFloat32);
    perms = perms.contiguous().to(torch::kInt32);
    inv_perms = inv_perms.contiguous().to(torch::kInt32);

    int64_t M = diagonals.size(0);
    int64_t B = x.size(0);
    int64_t dim = x.size(1);
    int64_t L = diagonals.size(1);

    auto grad_x = torch::zeros({B, dim}, x.options());
    auto grad_diagonals = torch::zeros({M, L, dim}, x.options());
    auto grad_bias = torch::zeros({M, dim}, x.options());

    const float* go_ptr = grad_output.data_ptr<float>();
    const float* x_ptr = x.data_ptr<float>();
    const float* d_ptr = diagonals.data_ptr<float>();
    const int32_t* p_ptr = perms.data_ptr<int32_t>();
    const int32_t* ip_ptr = inv_perms.data_ptr<int32_t>();

    float* gx_ptr = grad_x.data_ptr<float>();
    float* gd_ptr = grad_diagonals.data_ptr<float>();
    float* gb_ptr = grad_bias.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();
    std::vector<std::vector<float>> thread_gx(n_threads, std::vector<float>(B * dim, 0.0f));
    std::vector<std::vector<float>> thread_gd(n_threads, std::vector<float>(M * L * dim, 0.0f));
    std::vector<std::vector<float>> thread_gb(n_threads, std::vector<float>(M * dim, 0.0f));

#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        asdag::pin_thread_to_physical_core(tid);
        float* local_gx = thread_gx[tid].data();
        float* local_gd = thread_gd[tid].data();
        float* local_gb = thread_gb[tid].data();

        std::vector<float> h_buf(L * dim);
        std::vector<float> g_cur(dim);
        std::vector<float> g_prev(dim);

#pragma omp for collapse(2) schedule(static)
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t b = 0; b < B; ++b) {
                const float* xb = x_ptr + b * dim;
                const float* gob = go_ptr + (m * B + b) * dim;
                const float* d_m = d_ptr + m * (L * dim);
                float* local_gxb = local_gx + b * dim;
                float* local_gdm = local_gd + m * (L * dim);
                float* local_gbm = local_gb + m * dim;

                // Bias
                for (int64_t i = 0; i < dim; ++i) local_gbm[i] += gob[i];

                // Forward stages
                float* h0 = h_buf.data();
                for (int64_t i = 0; i < dim; ++i) h0[i] = xb[i] * d_m[i];

                for (int64_t s = 0; s < L - 1; ++s) {
                    const int32_t* perm_s = p_ptr + s * dim;
                    const float* ds = d_m + (s + 1) * dim;
                    const float* h_prev = h_buf.data() + s * dim;
                    float* h_next = h_buf.data() + (s + 1) * dim;
                    for (int64_t i = 0; i < dim; ++i) {
                        h_next[i] = h_prev[perm_s[i]] * ds[i];
                    }
                }

                // Backward stages
                std::memcpy(g_cur.data(), gob, dim * sizeof(float));

                for (int64_t s = L - 1; s >= 1; --s) {
                    const int32_t* perm_prev = p_ptr + (s - 1) * dim;
                    const int32_t* inv_perm_prev = ip_ptr + (s - 1) * dim;
                    const float* ds = d_m + s * dim;
                    const float* h_prev = h_buf.data() + (s - 1) * dim;
                    float* local_gd_s = local_gdm + s * dim;

                    for (int64_t i = 0; i < dim; ++i) {
                        local_gd_s[i] += g_cur[i] * h_prev[perm_prev[i]];
                    }
                    for (int64_t i = 0; i < dim; ++i) {
                        g_prev[inv_perm_prev[i]] = g_cur[i] * ds[i];
                    }
                    std::memcpy(g_cur.data(), g_prev.data(), dim * sizeof(float));
                }

                // Stage 0
                for (int64_t i = 0; i < dim; ++i) {
                    local_gdm[i] += g_cur[i] * xb[i];
                    local_gxb[i] += g_cur[i] * d_m[i];
                }
            }
        }
    }

    for (int t = 0; t < n_threads; ++t) {
        for (int64_t idx = 0; idx < B * dim; ++idx) gx_ptr[idx] += thread_gx[t][idx];
        for (int64_t idx = 0; idx < M * L * dim; ++idx) gd_ptr[idx] += thread_gd[t][idx];
        for (int64_t idx = 0; idx < M * dim; ++idx) gb_ptr[idx] += thread_gb[t][idx];
    }

    return std::make_tuple(grad_x.to(orig_dtype), grad_diagonals.to(orig_dtype), grad_bias.to(orig_dtype));
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. C++ BitLinear Backward Pass (Contiguous AVX2 / AVX-512 SIMD FMAs)
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> asdag_bitlinear_backward_cpp(
    torch::Tensor grad_output, // [B, out_dim]
    torch::Tensor x,           // [B, in_dim]
    torch::Tensor w_ternary,   // [out_dim, in_dim]
    float gamma,
    bool has_bias
) {
    auto orig_dtype = grad_output.scalar_type();
    grad_output = grad_output.contiguous().to(torch::kFloat32);
    x = x.contiguous().to(torch::kFloat32);
    w_ternary = w_ternary.contiguous().to(torch::kFloat32);

    int64_t B = x.size(0);
    int64_t in_dim = x.size(1);
    int64_t out_dim = w_ternary.size(0);

    auto grad_x = torch::empty({B, in_dim}, x.options());
    auto grad_w = torch::zeros({out_dim, in_dim}, x.options());
    auto grad_bias = has_bias ? torch::zeros({out_dim}, x.options()) : torch::tensor({}, x.options());

    const float* go_ptr = grad_output.data_ptr<float>();
    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w_ternary.data_ptr<float>();

    float* gx_ptr = grad_x.data_ptr<float>();
    float* gw_ptr = grad_w.data_ptr<float>();
    float* gb_ptr = has_bias ? grad_bias.data_ptr<float>() : nullptr;

    int n_threads = asdag::get_physical_cores();

    // 1. Parallel Token Quantization: X -> X_q
    auto x_quant = torch::empty({B, in_dim}, torch::kFloat32);
    float* xq_ptr = x_quant.data_ptr<float>();

#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        const float* xb = x_ptr + b * in_dim;
        float* xqb = xq_ptr + b * in_dim;

        float max_val = 1e-5f;
        for (int64_t i = 0; i < in_dim; ++i) max_val = std::max(max_val, std::abs(xb[i]));
        float scale_x = 127.0f / max_val;
        float inv_scale_x = 1.0f / scale_x;

        int64_t i = 0;
#if defined(ASDAG_SIMD_AVX512)
        __m512 sx_v = _mm512_set1_ps(scale_x);
        __m512 isx_v = _mm512_set1_ps(inv_scale_x);
        for (; i + 16 <= in_dim; i += 16) {
            __m512 xv = _mm512_loadu_ps(xb + i);
            __m512 qv = _mm512_roundscale_ps(_mm512_mul_ps(xv, sx_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm512_storeu_ps(xqb + i, _mm512_mul_ps(qv, isx_v));
        }
#elif defined(ASDAG_SIMD_AVX2)
        __m256 sx_v = _mm256_set1_ps(scale_x);
        __m256 isx_v = _mm256_set1_ps(inv_scale_x);
        for (; i + 8 <= in_dim; i += 8) {
            __m256 xv = _mm256_loadu_ps(xb + i);
            __m256 qv = _mm256_round_ps(_mm256_mul_ps(xv, sx_v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm256_storeu_ps(xqb + i, _mm256_mul_ps(qv, isx_v));
        }
#endif
        for (; i < in_dim; ++i) {
            xqb[i] = std::round(xb[i] * scale_x) * inv_scale_x;
        }
    }

    // 2. Parallel grad_x computation over tokens b in [0, B-1]
#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        const float* gob = go_ptr + b * out_dim;
        float* gxb = gx_ptr + b * in_dim;
        std::memset(gxb, 0, in_dim * sizeof(float));

        for (int64_t o = 0; o < out_dim; ++o) {
            float go_scaled = gob[o] * gamma;
            const float* wo = w_ptr + o * in_dim;

            int64_t i = 0;
#if defined(ASDAG_SIMD_AVX512)
            __m512 go_vec = _mm512_set1_ps(go_scaled);
            for (; i + 16 <= in_dim; i += 16) {
                __m512 w_vec = _mm512_loadu_ps(wo + i);
                __m512 gx_vec = _mm512_loadu_ps(gxb + i);
                _mm512_storeu_ps(gxb + i, _mm512_fmadd_ps(go_vec, w_vec, gx_vec));
            }
#elif defined(ASDAG_SIMD_AVX2)
            __m256 go_vec = _mm256_set1_ps(go_scaled);
            for (; i + 8 <= in_dim; i += 8) {
                __m256 w_vec = _mm256_loadu_ps(wo + i);
                __m256 gx_vec = _mm256_loadu_ps(gxb + i);
                _mm256_storeu_ps(gxb + i, _mm256_fmadd_ps(go_vec, w_vec, gx_vec));
            }
#endif
            for (; i < in_dim; ++i) {
                gxb[i] += go_scaled * wo[i];
            }
        }
    }

    // 3. Parallel grad_w computation over output rows o in [0, out_dim-1] (Zero Locks / Zero Merging)
#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t o = 0; o < out_dim; ++o) {
        float* gw_o = gw_ptr + o * in_dim;
        std::memset(gw_o, 0, in_dim * sizeof(float));

        float bias_acc = 0.0f;

        for (int64_t b = 0; b < B; ++b) {
            float go_val = go_ptr[b * out_dim + o];
            bias_acc += go_val;
            const float* xqb = xq_ptr + b * in_dim;

            int64_t i = 0;
#if defined(ASDAG_SIMD_AVX512)
            __m512 go_v = _mm512_set1_ps(go_val);
            for (; i + 16 <= in_dim; i += 16) {
                __m512 xq_v = _mm512_loadu_ps(xqb + i);
                __m512 gw_v = _mm512_loadu_ps(gw_o + i);
                _mm512_storeu_ps(gw_o + i, _mm512_fmadd_ps(go_v, xq_v, gw_v));
            }
#elif defined(ASDAG_SIMD_AVX2)
            __m256 go_v = _mm256_set1_ps(go_val);
            for (; i + 8 <= in_dim; i += 8) {
                __m256 xq_v = _mm256_loadu_ps(xqb + i);
                __m256 gw_v = _mm256_loadu_ps(gw_o + i);
                _mm256_storeu_ps(gw_o + i, _mm256_fmadd_ps(go_v, xq_v, gw_v));
            }
#endif
            for (; i < in_dim; ++i) {
                gw_o[i] += go_val * xqb[i];
            }
        }

        if (has_bias) {
            gb_ptr[o] = bias_acc;
        }
    }

    return std::make_tuple(grad_x.to(orig_dtype), grad_w.to(orig_dtype), grad_bias.to(orig_dtype));
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. C++ Fused GLA Associative Scan Forward (AVX2 / AVX-512 SIMD)
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> asdag_gla_scan_forward_cpp(
    torch::Tensor q,     // [B, H, T, D]
    torch::Tensor k,     // [B, H, T, D]
    torch::Tensor v,     // [B, H, T, D]
    torch::Tensor gamma  // [B, H, T]
) {
    auto orig_dtype = q.scalar_type();
    q = q.contiguous().to(torch::kFloat32);
    k = k.contiguous().to(torch::kFloat32);
    v = v.contiguous().to(torch::kFloat32);
    gamma = gamma.contiguous().to(torch::kFloat32);

    int64_t B = q.size(0);
    int64_t H = q.size(1);
    int64_t T = q.size(2);
    int64_t D = q.size(3);

    auto out_y = torch::empty({B, H, T, D}, q.options());
    auto S_all = torch::empty({B, H, T, D, D}, q.options());
    auto z_all = torch::empty({B, H, T, D}, q.options());

    const float* q_ptr = q.data_ptr<float>();
    const float* k_ptr = k.data_ptr<float>();
    const float* v_ptr = v.data_ptr<float>();
    const float* g_ptr = gamma.data_ptr<float>();

    float* y_ptr = out_y.data_ptr<float>();
    float* S_ptr = S_all.data_ptr<float>();
    float* z_ptr = z_all.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

#pragma omp parallel for collapse(2) num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            std::vector<float> S(D * D, 0.0f);
            std::vector<float> z(D, 0.0f);

            for (int64_t t = 0; t < T; ++t) {
                int64_t bht_idx = ((b * H + h) * T + t);
                const float* qt = q_ptr + bht_idx * D;
                const float* kt = k_ptr + bht_idx * D;
                const float* vt = v_ptr + bht_idx * D;
                float gam = g_ptr[(b * H + h) * T + t];

                float* yt = y_ptr + bht_idx * D;
                float* St_out = S_ptr + bht_idx * (D * D);
                float* zt_out = z_ptr + bht_idx * D;

                // 1. Update z: z = gamma * z + k
                float den = 0.0f;
                for (int64_t i = 0; i < D; ++i) {
                    float z_new = gam * z[i] + kt[i];
                    z[i] = z_new;
                    zt_out[i] = z_new;
                    den += qt[i] * z_new;
                }
                den = std::max(den, 1e-5f);
                float inv_den = 1.0f / den;

                // 2. Update S: S = gamma * S + k^T * v
                for (int64_t j = 0; j < D; ++j) {
                    float vj = vt[j];
                    float num_j = 0.0f;
                    for (int64_t i = 0; i < D; ++i) {
                        float s_new = gam * S[i * D + j] + kt[i] * vj;
                        S[i * D + j] = s_new;
                        St_out[i * D + j] = s_new;
                        num_j += qt[i] * s_new;
                    }
                    yt[j] = num_j * inv_den;
                }
            }
        }
    }

    return std::make_tuple(out_y.to(orig_dtype), S_all, z_all);
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. C++ Fused GLA Associative Scan Backward (AVX2 / AVX-512 SIMD)
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> asdag_gla_scan_backward_cpp(
    torch::Tensor grad_y, // [B, H, T, D]
    torch::Tensor q,      // [B, H, T, D]
    torch::Tensor k,      // [B, H, T, D]
    torch::Tensor v,      // [B, H, T, D]
    torch::Tensor gamma,  // [B, H, T]
    torch::Tensor S_all,  // [B, H, T, D, D]
    torch::Tensor z_all   // [B, H, T, D]
) {
    auto orig_dtype = grad_y.scalar_type();
    grad_y = grad_y.contiguous().to(torch::kFloat32);
    q = q.contiguous().to(torch::kFloat32);
    k = k.contiguous().to(torch::kFloat32);
    v = v.contiguous().to(torch::kFloat32);
    gamma = gamma.contiguous().to(torch::kFloat32);
    S_all = S_all.contiguous().to(torch::kFloat32);
    z_all = z_all.contiguous().to(torch::kFloat32);

    int64_t B = q.size(0);
    int64_t H = q.size(1);
    int64_t T = q.size(2);
    int64_t D = q.size(3);

    auto grad_q = torch::empty({B, H, T, D}, q.options());
    auto grad_k = torch::empty({B, H, T, D}, q.options());
    auto grad_v = torch::empty({B, H, T, D}, q.options());
    auto grad_gamma = torch::empty({B, H, T}, q.options());

    const float* gy_ptr = grad_y.data_ptr<float>();
    const float* q_ptr = q.data_ptr<float>();
    const float* k_ptr = k.data_ptr<float>();
    const float* v_ptr = v.data_ptr<float>();
    const float* g_ptr = gamma.data_ptr<float>();
    const float* S_ptr = S_all.data_ptr<float>();
    const float* z_ptr = z_all.data_ptr<float>();

    float* gq_ptr = grad_q.data_ptr<float>();
    float* gk_ptr = grad_k.data_ptr<float>();
    float* gv_ptr = grad_v.data_ptr<float>();
    float* gg_ptr = grad_gamma.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

#pragma omp parallel for collapse(2) num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            std::vector<float> dS(D * D, 0.0f);
            std::vector<float> dz(D, 0.0f);
            std::vector<float> num(D);
            std::vector<float> g_tilde(D);

            for (int64_t t = T - 1; t >= 0; --t) {
                int64_t bht_idx = ((b * H + h) * T + t);
                const float* gyt = gy_ptr + bht_idx * D;
                const float* qt = q_ptr + bht_idx * D;
                const float* kt = k_ptr + bht_idx * D;
                const float* vt = v_ptr + bht_idx * D;
                float gam_t = g_ptr[(b * H + h) * T + t];
                const float* St = S_ptr + bht_idx * (D * D);
                const float* zt = z_ptr + bht_idx * D;

                const float* S_prev = (t > 0) ? (S_ptr + ((b * H + h) * T + t - 1) * (D * D)) : nullptr;
                const float* z_prev = (t > 0) ? (z_ptr + ((b * H + h) * T + t - 1) * D) : nullptr;

                float* gqt = gq_ptr + bht_idx * D;
                float* gkt = gk_ptr + bht_idx * D;
                float* gvt = gv_ptr + bht_idx * D;
                float* ggt = gg_ptr + (b * H + h) * T + t;

                // Compute den = q . z
                float den = 0.0f;
                for (int64_t i = 0; i < D; ++i) den += qt[i] * zt[i];
                den = std::max(den, 1e-5f);
                float inv_den = 1.0f / den;

                // num = q . S
                for (int64_t j = 0; j < D; ++j) {
                    float dot = 0.0f;
                    for (int64_t i = 0; i < D; ++i) {
                        dot += qt[i] * St[i * D + j];
                    }
                    num[j] = dot;
                }

                // dot(gy, num) / den^2
                float gy_dot_y = 0.0f;
                for (int64_t j = 0; j < D; ++j) gy_dot_y += gyt[j] * (num[j] * inv_den);
                float d_den = -gy_dot_y * inv_den;

                // g_tilde = gyt / den
                for (int64_t j = 0; j < D; ++j) g_tilde[j] = gyt[j] * inv_den;

                // dS += q^T * g_tilde
                for (int64_t i = 0; i < D; ++i) {
                    float qi = qt[i];
                    for (int64_t j = 0; j < D; ++j) {
                        dS[i * D + j] += qi * g_tilde[j];
                    }
                }
                // dz += d_den * q
                for (int64_t i = 0; i < D; ++i) {
                    dz[i] += d_den * qt[i];
                }

                // grad_q = dS . S + dz . z
                for (int64_t i = 0; i < D; ++i) {
                    float acc = 0.0f;
                    for (int64_t j = 0; j < D; ++j) {
                        acc += St[i * D + j] * g_tilde[j];
                    }
                    gqt[i] = acc + d_den * zt[i];
                }

                // grad_k = dS . v + dz
                for (int64_t i = 0; i < D; ++i) {
                    float acc = 0.0f;
                    for (int64_t j = 0; j < D; ++j) {
                        acc += dS[i * D + j] * vt[j];
                    }
                    gkt[i] = acc + dz[i];
                }

                // grad_v = k . dS
                for (int64_t j = 0; j < D; ++j) {
                    float acc = 0.0f;
                    for (int64_t i = 0; i < D; ++i) {
                        acc += kt[i] * dS[i * D + j];
                    }
                    gvt[j] = acc;
                }

                // grad_gamma = dS : S_prev + dz . z_prev
                float dgam = 0.0f;
                if (S_prev != nullptr && z_prev != nullptr) {
                    for (int64_t idx = 0; idx < D * D; ++idx) {
                        dgam += dS[idx] * S_prev[idx];
                    }
                    for (int64_t i = 0; i < D; ++i) {
                        dgam += dz[i] * z_prev[i];
                    }
                }
                *ggt = dgam;

                // Propagate dS and dz to previous step: dS = gam_t * dS, dz = gam_t * dz
                for (int64_t idx = 0; idx < D * D; ++idx) dS[idx] *= gam_t;
                for (int64_t i = 0; i < D; ++i) dz[i] *= gam_t;
            }
        }
    }

    return std::make_tuple(grad_q.to(orig_dtype), grad_k.to(orig_dtype), grad_v.to(orig_dtype), grad_gamma);
}

// ─────────────────────────────────────────────────────────────────────────────
// 15. C++ SIMD-Block N:M Structured Sparse Tree Forward (AVX2 / AVX-512)
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor> asdag_sparse_tree_perm_forward_cpp(
    torch::Tensor x,            // [B, dim]
    torch::Tensor w_perm,       // [K, P, dim]
    torch::Tensor perms,        // [K, P, dim]
    torch::Tensor bias,         // [K, dim]
    torch::Tensor top_indices,  // [B, N] int32/int64
    torch::Tensor top_weights   // [B, N] float32
) {
    auto orig_dtype = x.scalar_type();
    x = x.contiguous().to(torch::kFloat32);
    w_perm = w_perm.contiguous().to(torch::kFloat32);
    perms = perms.contiguous().to(torch::kInt32);
    bias = bias.contiguous().to(torch::kFloat32);
    top_indices = top_indices.contiguous().to(torch::kInt32);
    top_weights = top_weights.contiguous().to(torch::kFloat32);

    int64_t B = x.size(0);
    int64_t dim = x.size(1);
    int64_t K = w_perm.size(0);
    int64_t P = w_perm.size(1);
    int64_t N = top_indices.size(1);

    auto out_y = torch::zeros({B, dim}, x.options());
    auto active_leaf_outs = torch::empty({B, N, dim}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w_perm.data_ptr<float>();
    const int32_t* p_ptr = perms.data_ptr<int32_t>();
    const float* b_ptr = bias.data_ptr<float>();
    const int32_t* top_idx_ptr = top_indices.data_ptr<int32_t>();
    const float* top_w_ptr = top_weights.data_ptr<float>();

    float* y_ptr = out_y.data_ptr<float>();
    float* lo_ptr = active_leaf_outs.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        const float* xb = x_ptr + b * dim;
        float* yb = y_ptr + b * dim;

        for (int64_t n = 0; n < N; ++n) {
            int32_t k = top_idx_ptr[b * N + n];
            float prob_k = top_w_ptr[b * N + n];
            float* leaf_out_bn = lo_ptr + (b * N + n) * dim;
            if (n + 1 < N) {
                int32_t k_next = top_idx_ptr[b * N + n + 1];
                __builtin_prefetch(w_ptr + k_next * (P * dim), 0, 1);
                __builtin_prefetch(p_ptr + k_next * (P * dim), 0, 1);
                __builtin_prefetch(b_ptr + k_next * dim, 0, 1);
            }
            if (prob_k < 1e-7f) {
                std::memset(leaf_out_bn, 0, dim * sizeof(float));
                continue;
            }

            // Load bias for leaf k
            const float* bk = b_ptr + k * dim;
            std::memcpy(leaf_out_bn, bk, dim * sizeof(float));

            const float* w_k = w_ptr + k * (P * dim);
            const int32_t* perm_k = p_ptr + k * (P * dim);

            for (int64_t p_idx = 0; p_idx < P; ++p_idx) {
                const float* w_kp = w_k + p_idx * dim;
                const int32_t* perm_kp = perm_k + p_idx * dim;

                if (p_idx == 0) {
                    // Identity permutation: direct AVX2/AVX-512 FMA
#if defined(ASDAG_SIMD_AVX512)
                    int64_t i = 0;
                    for (; i + 16 <= dim; i += 16) {
                        __m512 wv = _mm512_loadu_ps(w_kp + i);
                        __m512 xv = _mm512_loadu_ps(xb + i);
                        __m512 cur_out = _mm512_loadu_ps(leaf_out_bn + i);
                        _mm512_storeu_ps(leaf_out_bn + i, _mm512_fmadd_ps(wv, xv, cur_out));
                    }
                    for (; i < dim; ++i) leaf_out_bn[i] += w_kp[i] * xb[i];
#elif defined(ASDAG_SIMD_AVX2)
                    int64_t i = 0;
                    for (; i + 8 <= dim; i += 8) {
                        __m256 wv = _mm256_loadu_ps(w_kp + i);
                        __m256 xv = _mm256_loadu_ps(xb + i);
                        __m256 cur_out = _mm256_loadu_ps(leaf_out_bn + i);
                        _mm256_storeu_ps(leaf_out_bn + i, _mm256_fmadd_ps(wv, xv, cur_out));
                    }
                    for (; i < dim; ++i) leaf_out_bn[i] += w_kp[i] * xb[i];
#else
                    for (int64_t i = 0; i < dim; ++i) leaf_out_bn[i] += w_kp[i] * xb[i];
#endif
                } else {
                    // Permuted path: AVX2/AVX-512 Gather FMA
#if defined(ASDAG_SIMD_AVX512)
                    int64_t i = 0;
                    for (; i + 16 <= dim; i += 16) {
                        __m512 wv = _mm512_loadu_ps(w_kp + i);
                        __m512i p_indices = _mm512_loadu_si512((const __m512i*)(perm_kp + i));
                        __m512 xv = _mm512_i32gather_ps(p_indices, xb, 4);
                        __m512 cur_out = _mm512_loadu_ps(leaf_out_bn + i);
                        _mm512_storeu_ps(leaf_out_bn + i, _mm512_fmadd_ps(wv, xv, cur_out));
                    }
                    for (; i < dim; ++i) leaf_out_bn[i] += w_kp[i] * xb[perm_kp[i]];
#elif defined(ASDAG_SIMD_AVX2)
                    int64_t i = 0;
                    for (; i + 8 <= dim; i += 8) {
                        __m256 wv = _mm256_loadu_ps(w_kp + i);
                        __m256i p_indices = _mm256_loadu_si256((const __m256i*)(perm_kp + i));
                        __m256 xv = _mm256_i32gather_ps(xb, p_indices, 4);
                        __m256 cur_out = _mm256_loadu_ps(leaf_out_bn + i);
                        _mm256_storeu_ps(leaf_out_bn + i, _mm256_fmadd_ps(wv, xv, cur_out));
                    }
                    for (; i < dim; ++i) leaf_out_bn[i] += w_kp[i] * xb[perm_kp[i]];
#else
                    for (int64_t i = 0; i < dim; ++i) leaf_out_bn[i] += w_kp[i] * xb[perm_kp[i]];
#endif
                }
            }

            // ReLU6 + Routing weight accumulation
#if defined(ASDAG_SIMD_AVX512)
            __m512 pv = _mm512_set1_ps(prob_k);
            __m512 zero = _mm512_setzero_ps();
            __m512 six = _mm512_set1_ps(6.0f);
            int64_t d = 0;
            for (; d + 16 <= dim; d += 16) {
                __m512 val = _mm512_min_ps(_mm512_max_ps(_mm512_loadu_ps(leaf_out_bn + d), zero), six);
                _mm512_storeu_ps(leaf_out_bn + d, val);
                __m512 cur_y = _mm512_loadu_ps(yb + d);
                _mm512_storeu_ps(yb + d, _mm512_fmadd_ps(pv, val, cur_y));
            }
            for (; d < dim; ++d) {
                float val = std::min(std::max(leaf_out_bn[d], 0.0f), 6.0f);
                leaf_out_bn[d] = val;
                yb[d] += prob_k * val;
            }
#elif defined(ASDAG_SIMD_AVX2)
            __m256 pv = _mm256_set1_ps(prob_k);
            __m256 zero = _mm256_setzero_ps();
            __m256 six = _mm256_set1_ps(6.0f);
            int64_t d = 0;
            for (; d + 8 <= dim; d += 8) {
                __m256 val = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(leaf_out_bn + d), zero), six);
                _mm256_storeu_ps(leaf_out_bn + d, val);
                __m256 cur_y = _mm256_loadu_ps(yb + d);
                _mm256_storeu_ps(yb + d, _mm256_fmadd_ps(pv, val, cur_y));
            }
            for (; d < dim; ++d) {
                float val = std::min(std::max(leaf_out_bn[d], 0.0f), 6.0f);
                leaf_out_bn[d] = val;
                yb[d] += prob_k * val;
            }
#else
            for (int64_t d = 0; d < dim; ++d) {
                float val = std::min(std::max(leaf_out_bn[d], 0.0f), 6.0f);
                leaf_out_bn[d] = val;
                yb[d] += prob_k * val;
            }
#endif
        }
    }

    return std::make_tuple(out_y.to(orig_dtype), active_leaf_outs);
}

// ─────────────────────────────────────────────────────────────────────────────
// 16. C++ SIMD-Block N:M Structured Sparse Tree Backward (AVX2 / AVX-512)
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> asdag_sparse_tree_perm_backward_cpp(
    torch::Tensor grad_output,      // [B, dim]
    torch::Tensor x,                // [B, dim]
    torch::Tensor w_perm,           // [K, P, dim]
    torch::Tensor perms,            // [K, P, dim]
    torch::Tensor inv_perms,        // [K, P, dim]
    torch::Tensor bias,             // [K, dim]
    torch::Tensor top_indices,      // [B, N]
    torch::Tensor top_weights,      // [B, N]
    torch::Tensor active_leaf_outs  // [B, N, dim]
) {
    auto orig_dtype = grad_output.scalar_type();
    grad_output = grad_output.contiguous().to(torch::kFloat32);
    x = x.contiguous().to(torch::kFloat32);
    w_perm = w_perm.contiguous().to(torch::kFloat32);
    perms = perms.contiguous().to(torch::kInt32);
    inv_perms = inv_perms.contiguous().to(torch::kInt32);
    bias = bias.contiguous().to(torch::kFloat32);
    top_indices = top_indices.contiguous().to(torch::kInt32);
    top_weights = top_weights.contiguous().to(torch::kFloat32);
    active_leaf_outs = active_leaf_outs.contiguous().to(torch::kFloat32);

    int64_t B = x.size(0);
    int64_t dim = x.size(1);
    int64_t K = w_perm.size(0);
    int64_t P = w_perm.size(1);
    int64_t N = top_indices.size(1);

    auto grad_x = torch::zeros({B, dim}, x.options());
    auto grad_w = torch::zeros({K, P, dim}, w_perm.options());
    auto grad_bias = torch::zeros({K, dim}, bias.options());
    auto grad_top_weights = torch::empty({B, N}, top_weights.options());

    const float* go_ptr = grad_output.data_ptr<float>();
    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w_perm.data_ptr<float>();
    const int32_t* p_ptr = perms.data_ptr<int32_t>();
    const int32_t* ip_ptr = inv_perms.data_ptr<int32_t>();
    const int32_t* top_idx_ptr = top_indices.data_ptr<int32_t>();
    const float* top_w_ptr = top_weights.data_ptr<float>();
    const float* lo_ptr = active_leaf_outs.data_ptr<float>();

    float* gx_ptr = grad_x.data_ptr<float>();
    float* gw_ptr = grad_w.data_ptr<float>();
    float* gb_ptr = grad_bias.data_ptr<float>();
    float* gr_ptr = grad_top_weights.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();
    std::vector<std::vector<float>> thread_gw(n_threads, std::vector<float>(K * P * dim, 0.0f));
    std::vector<std::vector<float>> thread_gb(n_threads, std::vector<float>(K * dim, 0.0f));

#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        asdag::pin_thread_to_physical_core(tid);
        float* local_gw = thread_gw[tid].data();
        float* local_gb = thread_gb[tid].data();

#pragma omp for schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            const float* xb = x_ptr + b * dim;
            const float* gob = go_ptr + b * dim;
            float* gxb = gx_ptr + b * dim;

            for (int64_t n = 0; n < N; ++n) {
                int32_t k = top_idx_ptr[b * N + n];
                float prob_k = top_w_ptr[b * N + n];
                const float* lo_bn = lo_ptr + (b * N + n) * dim;
                const float* w_k = w_ptr + k * (P * dim);
                const int32_t* perm_k = p_ptr + k * (P * dim);
                const int32_t* inv_perm_k = ip_ptr + k * (P * dim);

                float* local_gw_k = local_gw + k * (P * dim);
                float* local_gb_k = local_gb + k * dim;

                // 1. Compute grad_top_weight = dot(gob, lo_bn)
                float d_prob = 0.0f;
                for (int64_t d = 0; d < dim; ++d) {
                    d_prob += gob[d] * lo_bn[d];
                }
                gr_ptr[b * N + n] = d_prob;

                // 2. Propagate gradient through ReLU6
                for (int64_t d = 0; d < dim; ++d) {
                    float act_grad = (lo_bn[d] > 0.0f && lo_bn[d] < 6.0f) ? 1.0f : 0.0f;
                    float g_d = gob[d] * prob_k * act_grad;

                    // Bias grad
                    local_gb_k[d] += g_d;

                    // Identity path (p_idx = 0)
                    local_gw_k[d] += g_d * xb[d];
                    gxb[d] += g_d * w_k[d];

                    // Permutation paths (p_idx > 0)
                    for (int64_t p_idx = 1; p_idx < P; ++p_idx) {
                        float* local_gw_kp = local_gw_k + p_idx * dim;
                        const float* w_kp = w_k + p_idx * dim;
                        const int32_t* perm_kp = perm_k + p_idx * dim;

                        local_gw_kp[d] += g_d * xb[perm_kp[d]];
                        gxb[perm_kp[d]] += g_d * w_kp[d];
                    }
                }
            }
        }
    }

    for (int t = 0; t < n_threads; ++t) {
        for (int64_t idx = 0; idx < K * P * dim; ++idx) gw_ptr[idx] += thread_gw[t][idx];
        for (int64_t idx = 0; idx < K * dim; ++idx) gb_ptr[idx] += thread_gb[t][idx];
    }

    return std::make_tuple(grad_x.to(orig_dtype), grad_w.to(orig_dtype), grad_bias.to(orig_dtype), grad_top_weights);
}

// ─────────────────────────────────────────────────────────────────────────────
// 17. 2-Bit Ternary Weight Packing & SIMD Unpacking Kernels
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor asdag_pack_ternary_2bit_cpp(torch::Tensor w_ternary) {
    auto orig_shape = w_ternary.sizes().vec();
    int64_t total_elements = w_ternary.numel();
    int64_t packed_bytes = (total_elements + 3) / 4;

    auto packed_tensor = torch::zeros({packed_bytes}, torch::kUInt8);
    const float* w_ptr = w_ternary.contiguous().data_ptr<float>();
    uint8_t* out_ptr = packed_tensor.data_ptr<uint8_t>();

#pragma omp parallel for schedule(static)
    for (int64_t byte_idx = 0; byte_idx < packed_bytes; ++byte_idx) {
        uint8_t byte_val = 0;
        for (int b = 0; b < 4; ++b) {
            int64_t elem_idx = byte_idx * 4 + b;
            if (elem_idx < total_elements) {
                float val = w_ptr[elem_idx];
                uint8_t code = 0;
                if (val > 0.5f) code = 1;      // +1 -> 01
                else if (val < -0.5f) code = 2; // -1 -> 10
                byte_val |= (code << (b * 2));
            }
        }
        out_ptr[byte_idx] = byte_val;
    }
    return packed_tensor;
}

torch::Tensor asdag_unpack_ternary_2bit_cpp(torch::Tensor packed_tensor, const std::vector<int64_t>& out_shape) {
    int64_t total_elements = 1;
    for (auto s : out_shape) total_elements *= s;
    int64_t packed_bytes = packed_tensor.numel();

    auto out_tensor = torch::zeros(out_shape, torch::kFloat32);
    const uint8_t* in_ptr = packed_tensor.contiguous().data_ptr<uint8_t>();
    float* out_ptr = out_tensor.data_ptr<float>();

    static const float LUT[4] = {0.0f, 1.0f, -1.0f, 0.0f};

#pragma omp parallel for schedule(static)
    for (int64_t byte_idx = 0; byte_idx < packed_bytes; ++byte_idx) {
        uint8_t byte_val = in_ptr[byte_idx];
        for (int b = 0; b < 4; ++b) {
            int64_t elem_idx = byte_idx * 4 + b;
            if (elem_idx < total_elements) {
                uint8_t code = (byte_val >> (b * 2)) & 3;
                out_ptr[elem_idx] = LUT[code];
            }
        }
    }
    return out_tensor;
}

// ─────────────────────────────────────────────────────────────────────────────
// 18. Multi-Stage Monarch Register-Fused Forward (Zero L1 Intermediate Writes)
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor asdag_monarch_reg_forward_cpp(
    torch::Tensor x,          // [B, dim]
    torch::Tensor diagonals,  // [L, dim]
    torch::Tensor perms,      // [L-1, dim]
    torch::Tensor bias        // [dim]
) {
    auto orig_dtype = x.scalar_type();
    x = x.contiguous().to(torch::kFloat32);
    diagonals = diagonals.contiguous().to(torch::kFloat32);
    perms = perms.contiguous().to(torch::kInt32);
    bias = bias.contiguous().to(torch::kFloat32);

    int64_t B = x.size(0);
    int64_t dim = x.size(1);
    int64_t L = diagonals.size(0);

    auto out = torch::empty({B, dim}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* d_ptr = diagonals.data_ptr<float>();
    const int32_t* p_ptr = perms.data_ptr<int32_t>();
    const float* b_ptr = bias.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        const float* xb = x_ptr + b * dim;
        float* yb = out_ptr + b * dim;

        // Load into stack cache
        std::vector<float> h0(dim);
        std::vector<float> h1(dim);

        // Stage 0: h = x * d[0]
        const float* d0 = d_ptr;
#if defined(ASDAG_SIMD_AVX512)
        int64_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 xv = _mm512_loadu_ps(xb + i);
            __m512 dv = _mm512_loadu_ps(d0 + i);
            _mm512_storeu_ps(h0.data() + i, _mm512_mul_ps(xv, dv));
        }
        for (; i < dim; ++i) h0[i] = xb[i] * d0[i];
#elif defined(ASDAG_SIMD_AVX2)
        int64_t i = 0;
        for (; i + 8 <= dim; i += 8) {
            __m256 xv = _mm256_loadu_ps(xb + i);
            __m256 dv = _mm256_loadu_ps(d0 + i);
            _mm256_storeu_ps(h0.data() + i, _mm256_mul_ps(xv, dv));
        }
        for (; i < dim; ++i) h0[i] = xb[i] * d0[i];
#else
        for (int64_t i = 0; i < dim; ++i) h0[i] = xb[i] * d0[i];
#endif

        float* cur_h = h0.data();
        float* next_h = h1.data();

        // Stages 1 .. L-1: Permutation + Diagonal Scaling
        for (int64_t s = 0; s < L - 1; ++s) {
            const int32_t* perm_s = p_ptr + s * dim;
            const float* d_s = d_ptr + (s + 1) * dim;

#if defined(ASDAG_SIMD_AVX512)
            int64_t d = 0;
            for (; d + 16 <= dim; d += 16) {
                __m512i idx = _mm512_loadu_si512((const __m512i*)(perm_s + d));
                __m512 hv = _mm512_i32gather_ps(idx, cur_h, 4);
                __m512 dv = _mm512_loadu_ps(d_s + d);
                _mm512_storeu_ps(next_h + d, _mm512_mul_ps(hv, dv));
            }
            for (; d < dim; ++d) next_h[d] = cur_h[perm_s[d]] * d_s[d];
#elif defined(ASDAG_SIMD_AVX2)
            int64_t d = 0;
            for (; d + 8 <= dim; d += 8) {
                __m256i idx = _mm256_loadu_si256((const __m256i*)(perm_s + d));
                __m256 hv = _mm256_i32gather_ps(cur_h, idx, 4);
                __m256 dv = _mm256_loadu_ps(d_s + d);
                _mm256_storeu_ps(next_h + d, _mm256_mul_ps(hv, dv));
            }
            for (; d < dim; ++d) next_h[d] = cur_h[perm_s[d]] * d_s[d];
#else
            for (int64_t d = 0; d < dim; ++d) next_h[d] = cur_h[perm_s[d]] * d_s[d];
#endif
            std::swap(cur_h, next_h);
        }

        // Add bias to output
#if defined(ASDAG_SIMD_AVX512)
        int64_t d = 0;
        for (; d + 16 <= dim; d += 16) {
            __m512 hv = _mm512_loadu_ps(cur_h + d);
            __m512 bv = _mm512_loadu_ps(b_ptr + d);
            _mm512_storeu_ps(yb + d, _mm512_add_ps(hv, bv));
        }
        for (; d < dim; ++d) yb[d] = cur_h[d] + b_ptr[d];
#elif defined(ASDAG_SIMD_AVX2)
        int64_t d = 0;
        for (; d + 8 <= dim; d += 8) {
            __m256 hv = _mm256_loadu_ps(cur_h + d);
            __m256 bv = _mm256_loadu_ps(b_ptr + d);
            _mm256_storeu_ps(yb + d, _mm256_add_ps(hv, bv));
        }
        for (; d < dim; ++d) yb[d] = cur_h[d] + b_ptr[d];
#else
        for (int64_t d = 0; d < dim; ++d) yb[d] = cur_h[d] + b_ptr[d];
#endif
    }

    return out.to(orig_dtype);
}

// ─────────────────────────────────────────────────────────────────────────────
static inline float dot_avx2_64(const float* a, const float* b);
static inline float dot_avx2_128(const float* a, const float* b);

// ─────────────────────────────────────────────────────────────────────────────
// 19c. Fused Native C++ SIMD Byte Local Encoder Forward
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor> asdag_byte_encoder_forward_cpp(
    torch::Tensor byte_ids,            // [B, T] int64
    torch::Tensor embed_weight,        // [V, d_byte] (256, 64)
    torch::Tensor conv_weight,         // [d_byte, 1, K] (64, 1, 4)
    torch::Tensor conv_bias,           // [d_byte] (64)
    torch::Tensor norm_scale,          // [d_byte] (64)
    torch::Tensor proj_weight,         // [d_byte, d_byte] (64, 64)
    torch::Tensor boundary_weight,     // [1, d_byte] (1, 64)
    torch::Tensor boundary_bias        // [1]
) {
    auto orig_dtype = embed_weight.scalar_type();
    byte_ids = byte_ids.contiguous().to(torch::kInt64);
    embed_weight = embed_weight.contiguous().to(torch::kFloat32);
    conv_weight = conv_weight.contiguous().to(torch::kFloat32);
    conv_bias = conv_bias.contiguous().to(torch::kFloat32);
    norm_scale = norm_scale.contiguous().to(torch::kFloat32);
    proj_weight = proj_weight.contiguous().to(torch::kFloat32);
    boundary_weight = boundary_weight.contiguous().to(torch::kFloat32);
    boundary_bias = boundary_bias.contiguous().to(torch::kFloat32);

    int64_t B = byte_ids.size(0);
    int64_t T = byte_ids.size(1);
    int64_t d_byte = embed_weight.size(1);
    int64_t K = conv_weight.size(2);

    auto h_byte = torch::empty({B, T, d_byte}, torch::kFloat32);
    auto boundary_logits = torch::empty({B, T}, torch::kFloat32);

    const int64_t* b_ptr = byte_ids.data_ptr<int64_t>();
    const float* emb_ptr = embed_weight.data_ptr<float>();
    const float* cw_ptr = conv_weight.data_ptr<float>();
    const float* cb_ptr = conv_bias.data_ptr<float>();
    const float* ns_ptr = norm_scale.data_ptr<float>();
    const float* pw_ptr = proj_weight.data_ptr<float>();
    const float* bw_ptr = boundary_weight.data_ptr<float>();
    float b_bias = boundary_bias.data_ptr<float>()[0];

    float* hb_ptr = h_byte.data_ptr<float>();
    float* bl_ptr = boundary_logits.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        std::vector<float> h_norm(d_byte);
        std::vector<float> h_raw(d_byte);

        for (int64_t t = 0; t < T; ++t) {
            int64_t cur_id = b_ptr[b * T + t];
            if (cur_id < 0 || cur_id >= 256) cur_id = 0;
            const float* cur_emb = emb_ptr + cur_id * d_byte;

            // Depthwise causal conv over K=4
            float sum_sq = 0.0f;
            for (int64_t i = 0; i < d_byte; ++i) {
                float conv_val = cb_ptr[i];
                for (int64_t k = 0; k < K; ++k) {
                    int64_t t_src = t - (K - 1) + k;
                    if (t_src >= 0) {
                        int64_t prev_id = b_ptr[b * T + t_src];
                        if (prev_id < 0 || prev_id >= 256) prev_id = 0;
                        conv_val += cw_ptr[i * K + k] * emb_ptr[prev_id * d_byte + i];
                    }
                }
                float h_val = cur_emb[i] + conv_val;
                h_raw[i] = h_val;
                sum_sq += h_val * h_val;
            }

            // RMSNorm
            float rms = 1.0f / std::sqrt((sum_sq / (float)d_byte) + 1e-5f);
            for (int64_t i = 0; i < d_byte; ++i) {
                h_norm[i] = h_raw[i] * rms * ns_ptr[i];
            }

            // Proj + SiLU
            float* cur_hb = hb_ptr + (b * T + t) * d_byte;
            float b_dot = b_bias;

            for (int64_t i = 0; i < d_byte; ++i) {
                const float* p_row = pw_ptr + i * d_byte;
                float dot = (d_byte == 64) ? dot_avx2_64(p_row, h_norm.data()) : 0.0f;
                if (d_byte != 64) {
                    for (int64_t k = 0; k < d_byte; ++k) dot += p_row[k] * h_norm[k];
                }
                float silu_val = dot / (1.0f + std::exp(-dot));
                cur_hb[i] = silu_val;
                b_dot += bw_ptr[i] * silu_val;
            }

            bl_ptr[b * T + t] = b_dot;
        }
    }

    return std::make_tuple(h_byte.to(orig_dtype), boundary_logits.to(orig_dtype));
}
static inline float dot_avx2_64(const float* a, const float* b) {
#if defined(ASDAG_SIMD_AVX2)
    __m256 acc0 = _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 48), _mm256_loadu_ps(b + 48), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 56), _mm256_loadu_ps(b + 56), acc0);

    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehl_ps(sum4, sum4);
    __m128 sum2 = _mm_add_ps(sum4, shuf);
    __m128 shuf2 = _mm_shuffle_ps(sum2, sum2, 1);
    return _mm_cvtss_f32(_mm_add_ss(sum2, shuf2));
#else
    float dot = 0.0f;
    for (int k = 0; k < 64; ++k) dot += a[k] * b[k];
    return dot;
#endif
}

static inline float dot_avx2_128(const float* a, const float* b) {
#if defined(ASDAG_SIMD_AVX2)
    __m256 acc0 = _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 48), _mm256_loadu_ps(b + 48), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 56), _mm256_loadu_ps(b + 56), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 64), _mm256_loadu_ps(b + 64), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 72), _mm256_loadu_ps(b + 72), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 80), _mm256_loadu_ps(b + 80), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 88), _mm256_loadu_ps(b + 88), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 96), _mm256_loadu_ps(b + 96), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 104), _mm256_loadu_ps(b + 104), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 112), _mm256_loadu_ps(b + 112), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 120), _mm256_loadu_ps(b + 120), acc0);

    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehl_ps(sum4, sum4);
    __m128 sum2 = _mm_add_ps(sum4, shuf);
    __m128 shuf2 = _mm_shuffle_ps(sum2, sum2, 1);
    return _mm_cvtss_f32(_mm_add_ss(sum2, shuf2));
#else
    float dot = 0.0f;
    for (int k = 0; k < 128; ++k) dot += a[k] * b[k];
    return dot;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// 19c-b. Fused Native C++ SIMD Byte Local Encoder Backward
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> asdag_byte_encoder_backward_cpp(
    torch::Tensor grad_hb,             // [B, T, d_byte]
    torch::Tensor grad_bl,             // [B, T]
    torch::Tensor byte_ids,            // [B, T] int64
    torch::Tensor embed_weight,        // [V, d_byte]
    torch::Tensor conv_weight,         // [d_byte, 1, K]
    torch::Tensor conv_bias,           // [d_byte]
    torch::Tensor norm_scale,          // [d_byte]
    torch::Tensor proj_weight,         // [d_byte, d_byte]
    torch::Tensor boundary_weight,     // [1, d_byte]
    torch::Tensor boundary_bias        // [1]
) {
    auto orig_dtype = embed_weight.scalar_type();
    grad_hb = grad_hb.contiguous().to(torch::kFloat32);
    grad_bl = grad_bl.contiguous().to(torch::kFloat32);
    byte_ids = byte_ids.contiguous().to(torch::kInt64);
    embed_weight = embed_weight.contiguous().to(torch::kFloat32);
    conv_weight = conv_weight.contiguous().to(torch::kFloat32);
    conv_bias = conv_bias.contiguous().to(torch::kFloat32);
    norm_scale = norm_scale.contiguous().to(torch::kFloat32);
    proj_weight = proj_weight.contiguous().to(torch::kFloat32);
    boundary_weight = boundary_weight.contiguous().to(torch::kFloat32);
    boundary_bias = boundary_bias.contiguous().to(torch::kFloat32);

    int64_t B = byte_ids.size(0);
    int64_t T = byte_ids.size(1);
    int64_t V = embed_weight.size(0);
    int64_t d_byte = embed_weight.size(1);
    int64_t K = conv_weight.size(2);

    auto grad_embed = torch::zeros({V, d_byte}, torch::kFloat32);
    auto grad_conv_w = torch::zeros({d_byte, 1, K}, torch::kFloat32);
    auto grad_conv_b = torch::zeros({d_byte}, torch::kFloat32);
    auto grad_norm_scale = torch::zeros({d_byte}, torch::kFloat32);
    auto grad_proj_w = torch::zeros({d_byte, d_byte}, torch::kFloat32);
    auto grad_boundary_w = torch::zeros({1, d_byte}, torch::kFloat32);
    auto grad_boundary_b = torch::zeros({1}, torch::kFloat32);

    const float* ghb_ptr = grad_hb.data_ptr<float>();
    const float* gbl_ptr = grad_bl.data_ptr<float>();
    const int64_t* b_ptr = byte_ids.data_ptr<int64_t>();
    const float* emb_ptr = embed_weight.data_ptr<float>();
    const float* cw_ptr = conv_weight.data_ptr<float>();
    const float* cb_ptr = conv_bias.data_ptr<float>();
    const float* ns_ptr = norm_scale.data_ptr<float>();
    const float* pw_ptr = proj_weight.data_ptr<float>();
    const float* bw_ptr = boundary_weight.data_ptr<float>();

    float* g_emb_ptr = grad_embed.data_ptr<float>();
    float* g_cw_ptr = grad_conv_w.data_ptr<float>();
    float* g_cb_ptr = grad_conv_b.data_ptr<float>();
    float* g_ns_ptr = grad_norm_scale.data_ptr<float>();
    float* g_pw_ptr = grad_proj_w.data_ptr<float>();
    float* g_bw_ptr = grad_boundary_w.data_ptr<float>();
    float* g_bb_ptr = grad_boundary_b.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();
    std::vector<std::vector<float>> t_emb(n_threads, std::vector<float>(V * d_byte, 0.0f));
    std::vector<std::vector<float>> t_cw(n_threads, std::vector<float>(d_byte * K, 0.0f));
    std::vector<std::vector<float>> t_cb(n_threads, std::vector<float>(d_byte, 0.0f));
    std::vector<std::vector<float>> t_ns(n_threads, std::vector<float>(d_byte, 0.0f));
    std::vector<std::vector<float>> t_pw(n_threads, std::vector<float>(d_byte * d_byte, 0.0f));
    std::vector<std::vector<float>> t_bw(n_threads, std::vector<float>(d_byte, 0.0f));
    std::vector<float> t_bb(n_threads, 0.0f);

#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        float* local_emb = t_emb[tid].data();
        float* local_cw = t_cw[tid].data();
        float* local_cb = t_cb[tid].data();
        float* local_ns = t_ns[tid].data();
        float* local_pw = t_pw[tid].data();
        float* local_bw = t_bw[tid].data();

        std::vector<float> h_raw(d_byte);
        std::vector<float> h_norm(d_byte);
        std::vector<float> u_raw(d_byte);
        std::vector<float> g_u(d_byte);
        std::vector<float> g_hnorm(d_byte);
        std::vector<float> g_hraw(d_byte);

#pragma omp for schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t t = 0; t < T; ++t) {
                int64_t cur_id = b_ptr[b * T + t];
                if (cur_id < 0 || cur_id >= 256) cur_id = 0;
                const float* cur_emb = emb_ptr + cur_id * d_byte;

                // 1. Forward recompute
                float sum_sq = 0.0f;
                for (int64_t i = 0; i < d_byte; ++i) {
                    float conv_val = cb_ptr[i];
                    for (int64_t k = 0; k < K; ++k) {
                        int64_t t_src = t - (K - 1) + k;
                        if (t_src >= 0) {
                            int64_t prev_id = b_ptr[b * T + t_src];
                            if (prev_id < 0 || prev_id >= 256) prev_id = 0;
                            conv_val += cw_ptr[i * K + k] * emb_ptr[prev_id * d_byte + i];
                        }
                    }
                    float h_val = cur_emb[i] + conv_val;
                    h_raw[i] = h_val;
                    sum_sq += h_val * h_val;
                }

                float rms = 1.0f / std::sqrt((sum_sq / (float)d_byte) + 1e-5f);
                for (int64_t i = 0; i < d_byte; ++i) {
                    h_norm[i] = h_raw[i] * rms * ns_ptr[i];
                }

                for (int64_t i = 0; i < d_byte; ++i) {
                    const float* p_row = pw_ptr + i * d_byte;
                    float dot = 0.0f;
                    for (int64_t k = 0; k < d_byte; ++k) dot += p_row[k] * h_norm[k];
                    u_raw[i] = dot;
                }

                // 2. Gradients from boundary predictor & SiLU
                float g_bl_val = gbl_ptr[b * T + t];
                t_bb[tid] += g_bl_val;
                const float* cur_ghb = ghb_ptr + (b * T + t) * d_byte;

                for (int64_t i = 0; i < d_byte; ++i) {
                    float ui = u_raw[i];
                    float sig_u = 1.0f / (1.0f + std::exp(-ui));
                    float silu_u = ui * sig_u;
                    float dsilu_u = sig_u * (1.0f + ui * (1.0f - sig_u));

                    local_bw[i] += g_bl_val * silu_u;
                    float g_hb_tot = cur_ghb[i] + g_bl_val * bw_ptr[i];
                    g_u[i] = g_hb_tot * dsilu_u;
                }

                // 3. Gradients through Proj Linear
                for (int64_t i = 0; i < d_byte; ++i) {
                    float gui = g_u[i];
                    float* pw_row = local_pw + i * d_byte;
                    for (int64_t k = 0; k < d_byte; ++k) {
                        pw_row[k] += gui * h_norm[k];
                    }
                }

                for (int64_t k = 0; k < d_byte; ++k) {
                    float acc = 0.0f;
                    for (int64_t i = 0; i < d_byte; ++i) {
                        acc += g_u[i] * pw_ptr[i * d_byte + k];
                    }
                    g_hnorm[k] = acc;
                }

                // 4. Gradients through RMSNorm
                float sum_gy_y = 0.0f;
                for (int64_t i = 0; i < d_byte; ++i) {
                    float g_pre = g_hnorm[i] * ns_ptr[i];
                    float y_norm = h_raw[i] * rms;
                    sum_gy_y += g_pre * y_norm;
                    local_ns[i] += g_hnorm[i] * y_norm;
                }

                for (int64_t i = 0; i < d_byte; ++i) {
                    float g_pre = g_hnorm[i] * ns_ptr[i];
                    float y_norm = h_raw[i] * rms;
                    g_hraw[i] = rms * (g_pre - y_norm * (sum_gy_y / (float)d_byte));
                }

                // 5. Gradients through Conv & Embedding
                for (int64_t i = 0; i < d_byte; ++i) {
                    float gh = g_hraw[i];
                    local_cb[i] += gh;
                    local_emb[cur_id * d_byte + i] += gh;

                    for (int64_t k = 0; k < K; ++k) {
                        int64_t t_src = t - (K - 1) + k;
                        if (t_src >= 0) {
                            int64_t prev_id = b_ptr[b * T + t_src];
                            if (prev_id < 0 || prev_id >= 256) prev_id = 0;
                            local_cw[i * K + k] += gh * emb_ptr[prev_id * d_byte + i];
                            local_emb[prev_id * d_byte + i] += gh * cw_ptr[i * K + k];
                        }
                    }
                }
            }
        }
    }

    // Reduction across threads
    for (int t = 0; t < n_threads; ++t) {
        for (int64_t idx = 0; idx < V * d_byte; ++idx) g_emb_ptr[idx] += t_emb[t][idx];
        for (int64_t idx = 0; idx < d_byte * K; ++idx) g_cw_ptr[idx] += t_cw[t][idx];
        for (int64_t idx = 0; idx < d_byte; ++idx) g_cb_ptr[idx] += t_cb[t][idx];
        for (int64_t idx = 0; idx < d_byte; ++idx) g_ns_ptr[idx] += t_ns[t][idx];
        for (int64_t idx = 0; idx < d_byte * d_byte; ++idx) g_pw_ptr[idx] += t_pw[t][idx];
        for (int64_t idx = 0; idx < d_byte; ++idx) g_bw_ptr[idx] += t_bw[t][idx];
        g_bb_ptr[0] += t_bb[t];
    }

    return std::make_tuple(
        grad_embed.to(orig_dtype),
        grad_conv_w.to(orig_dtype),
        grad_conv_b.to(orig_dtype),
        grad_norm_scale.to(orig_dtype),
        grad_proj_w.to(orig_dtype),
        grad_boundary_w.to(orig_dtype),
        grad_boundary_b.to(orig_dtype)
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 19d. 2-Layer Fused BLT Causal Byte Decoder Forward (AVX2 Multi-Threaded)
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor asdag_blt_2layer_decode_fused_cpp(
    torch::Tensor h_byte,                // [B, T, d_byte]
    torch::Tensor causal_latent_patches, // [B, M, d_model]
    torch::Tensor patch_to_byte_weight,  // [d_byte, d_model]
    torch::Tensor fusion_weight,         // [d_byte, 2*d_byte]
    torch::Tensor gate_weight,           // [d_byte, d_byte]
    torch::Tensor val_weight,            // [d_byte, d_byte]
    torch::Tensor down_weight,           // [d_byte, d_byte]
    torch::Tensor lm_head_weight,        // [V, d_byte]
    torch::Tensor patch_assignments      // [B, T] int64
) {
    auto orig_dtype = h_byte.scalar_type();
    h_byte = h_byte.contiguous().to(torch::kFloat32);
    causal_latent_patches = causal_latent_patches.contiguous().to(torch::kFloat32);
    patch_to_byte_weight = patch_to_byte_weight.contiguous().to(torch::kFloat32);
    fusion_weight = fusion_weight.contiguous().to(torch::kFloat32);
    gate_weight = gate_weight.contiguous().to(torch::kFloat32);
    val_weight = val_weight.contiguous().to(torch::kFloat32);
    down_weight = down_weight.contiguous().to(torch::kFloat32);
    lm_head_weight = lm_head_weight.contiguous().to(torch::kFloat32);
    patch_assignments = patch_assignments.contiguous().to(torch::kInt64);

    int64_t B = h_byte.size(0);
    int64_t T = h_byte.size(1);
    int64_t d_byte = h_byte.size(2);
    int64_t M = causal_latent_patches.size(1);
    int64_t d_model = causal_latent_patches.size(2);
    int64_t V = lm_head_weight.size(0);

    auto logits = torch::empty({B, T, V}, torch::kFloat32);

    const float* hb_ptr = h_byte.data_ptr<float>();
    const float* clp_ptr = causal_latent_patches.data_ptr<float>();
    const float* p2b_ptr = patch_to_byte_weight.data_ptr<float>();
    const float* fus_ptr = fusion_weight.data_ptr<float>();
    const float* gate_ptr = gate_weight.data_ptr<float>();
    const float* val_ptr = val_weight.data_ptr<float>();
    const float* down_ptr = down_weight.data_ptr<float>();
    const float* lm_ptr = lm_head_weight.data_ptr<float>();
    const int64_t* pa_ptr = patch_assignments.data_ptr<int64_t>();
    float* log_ptr = logits.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

    // 1. Hoisted patch_h_all [B, M, d_byte]
    auto patch_h_all = torch::empty({B, M, d_byte}, torch::kFloat32);
    float* ph_all_ptr = patch_h_all.data_ptr<float>();

#pragma omp parallel for collapse(2) num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t m = 0; m < M; ++m) {
            const float* cur_p = clp_ptr + (b * M + m) * d_model;
            float* ph = ph_all_ptr + (b * M + m) * d_byte;
            for (int64_t i = 0; i < d_byte; ++i) {
                const float* w_row = p2b_ptr + i * d_model;
                float dot = 0.0f;
#if defined(ASDAG_SIMD_AVX2)
                __m256 acc_v = _mm256_setzero_ps();
                int64_t k = 0;
                for (; k + 8 <= d_model; k += 8) {
                    acc_v = _mm256_fmadd_ps(_mm256_loadu_ps(w_row + k), _mm256_loadu_ps(cur_p + k), acc_v);
                }
                float tmp[8];
                _mm256_storeu_ps(tmp, acc_v);
                for (int e = 0; e < 8; ++e) dot += tmp[e];
                for (; k < d_model; ++k) dot += w_row[k] * cur_p[k];
#else
                for (int64_t k = 0; k < d_model; ++k) dot += w_row[k] * cur_p[k];
#endif
                ph[i] = dot;
            }
        }
    }

#pragma omp parallel num_threads(n_threads)
    {
        const bool is64 = (d_byte == 64);
        std::vector<float> cat_buf(2 * d_byte);
        std::vector<float> fused1(d_byte);
        std::vector<float> hact_buf(d_byte);
        std::vector<float> fused2(d_byte);

#pragma omp for collapse(2) schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t t = 0; t < T; ++t) {
                int64_t patch_idx = pa_ptr[b * T + t];
                if (patch_idx >= M) patch_idx = M - 1;
                if (patch_idx < 0) patch_idx = 0;

                const float* patch_h = ph_all_ptr + (b * M + patch_idx) * d_byte;
                const float* cur_hb = hb_ptr + (b * T + t) * d_byte;
                float* out_logits = log_ptr + (b * T + t) * V;

                std::memcpy(cat_buf.data(), cur_hb, d_byte * sizeof(float));
                std::memcpy(cat_buf.data() + d_byte, patch_h, d_byte * sizeof(float));

                float sum_sq1 = 0.0f;
                for (int64_t i = 0; i < d_byte; ++i) {
                    const float* w_row = fus_ptr + i * (2 * d_byte);
                    float dot = 0.0f;
                    if (is64) {
                        dot = dot_avx2_128(w_row, cat_buf.data());
                    } else {
                        for (int64_t k = 0; k < 2 * d_byte; ++k) dot += w_row[k] * cat_buf[k];
                    }
                    float s1 = dot / (1.0f + std::exp(-dot));
                    fused1[i] = s1;
                    sum_sq1 += s1 * s1;
                }
                float rms1 = 1.0f / std::sqrt((sum_sq1 / (float)d_byte) + 1e-5f);
                for (int64_t i = 0; i < d_byte; ++i) fused1[i] *= rms1;

                for (int64_t i = 0; i < d_byte; ++i) {
                    const float* g_row = gate_ptr + i * d_byte;
                    const float* v_row = val_ptr + i * d_byte;
                    float dot_g = 0.0f, dot_v = 0.0f;
                    if (is64) {
                        dot_g = dot_avx2_64(g_row, fused1.data());
                        dot_v = dot_avx2_64(v_row, fused1.data());
                    } else {
                        for (int64_t k = 0; k < d_byte; ++k) {
                            dot_g += g_row[k] * fused1[k];
                            dot_v += v_row[k] * fused1[k];
                        }
                    }
                    float silu_g = dot_g / (1.0f + std::exp(-dot_g));
                    hact_buf[i] = silu_g * dot_v;
                }

                float sum_sq2 = 0.0f;
                for (int64_t i = 0; i < d_byte; ++i) {
                    const float* w_row = down_ptr + i * d_byte;
                    float dot = 0.0f;
                    if (is64) {
                        dot = dot_avx2_64(w_row, hact_buf.data());
                    } else {
                        for (int64_t k = 0; k < d_byte; ++k) dot += w_row[k] * hact_buf[k];
                    }
                    float pre2 = fused1[i] + dot;
                    fused2[i] = pre2;
                    sum_sq2 += pre2 * pre2;
                }
                float rms2 = 1.0f / std::sqrt((sum_sq2 / (float)d_byte) + 1e-5f);
                for (int64_t i = 0; i < d_byte; ++i) fused2[i] *= rms2;

                for (int64_t v = 0; v < V; ++v) {
                    const float* w_row = lm_ptr + v * d_byte;
                    float dot = 0.0f;
                    if (is64) {
                        dot = dot_avx2_64(w_row, fused2.data());
                    } else {
                        for (int64_t k = 0; k < d_byte; ++k) dot += w_row[k] * fused2[k];
                    }
                    out_logits[v] = dot;
                }
            }
        }
    }

    return logits.to(orig_dtype);
}

// ─────────────────────────────────────────────────────────────────────────────
// 19e. 2-Layer Fused BLT Causal Decoder + Cross-Entropy Loss (Zero-Logits RAM)
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> asdag_blt_2layer_decode_loss_fused_cpp(
    torch::Tensor h_byte,                // [B, T, d_byte]
    torch::Tensor causal_latent_patches, // [B, M, d_model]
    torch::Tensor patch_to_byte_weight,  // [d_byte, d_model]
    torch::Tensor fusion_weight,         // [d_byte, 2*d_byte]
    torch::Tensor gate_weight,           // [d_byte, d_byte]
    torch::Tensor val_weight,            // [d_byte, d_byte]
    torch::Tensor down_weight,           // [d_byte, d_byte]
    torch::Tensor lm_head_weight,        // [V, d_byte]
    torch::Tensor patch_assignments,     // [B, T] int64
    torch::Tensor targets,               // [B, T] int64
    torch::Tensor norm1_scale,           // [d_byte]
    torch::Tensor norm2_scale            // [d_byte]
) {
    auto orig_dtype = h_byte.scalar_type();
    h_byte = h_byte.contiguous().to(torch::kFloat32);
    causal_latent_patches = causal_latent_patches.contiguous().to(torch::kFloat32);
    patch_to_byte_weight = patch_to_byte_weight.contiguous().to(torch::kFloat32);
    fusion_weight = fusion_weight.contiguous().to(torch::kFloat32);
    gate_weight = gate_weight.contiguous().to(torch::kFloat32);
    val_weight = val_weight.contiguous().to(torch::kFloat32);
    down_weight = down_weight.contiguous().to(torch::kFloat32);
    lm_head_weight = lm_head_weight.contiguous().to(torch::kFloat32);
    patch_assignments = patch_assignments.contiguous().to(torch::kInt64);
    targets = targets.contiguous().to(torch::kInt64);
    norm1_scale = norm1_scale.contiguous().to(torch::kFloat32);
    norm2_scale = norm2_scale.contiguous().to(torch::kFloat32);
    auto ternarize_w = [](torch::Tensor w) {
        float gamma = w.abs().mean().item<float>();
        if (gamma < 1e-5f) gamma = 1e-5f;
        return torch::clamp(torch::round(w / gamma), -1.f, 1.f);
    };
    auto w_p2b = ternarize_w(patch_to_byte_weight);
    auto w_fus = ternarize_w(fusion_weight);
    auto w_gate = ternarize_w(gate_weight);
    auto w_val = ternarize_w(val_weight);
    auto w_down = ternarize_w(down_weight);
    auto w_lm = ternarize_w(lm_head_weight);
    auto gamma_of = [](torch::Tensor w) {
        float g = w.abs().mean().item<float>();
        return g < 1e-5f ? 1e-5f : g;
    };
    float g_p2b = gamma_of(patch_to_byte_weight);
    float g_fus = gamma_of(fusion_weight);
    float g_down = gamma_of(down_weight);
    float g_lm = gamma_of(lm_head_weight);
    auto no_bias = torch::tensor({});

    int64_t B = h_byte.size(0);
    int64_t T = h_byte.size(1);
    int64_t d_byte = h_byte.size(2);
    int64_t M = causal_latent_patches.size(1);
    int64_t d_model = causal_latent_patches.size(2);
    int64_t V = lm_head_weight.size(0);
    int64_t N = B * T;

    // 1. Patch to byte context gathering
    auto clp_flat = causal_latent_patches.reshape({B * M, d_model});
    auto ph_p = asdag_bitlinear_forward_cpp(clp_flat, w_p2b, g_p2b, no_bias).reshape({B, M, d_byte});
    auto pa_exp = patch_assignments.unsqueeze(-1).expand({B, T, d_byte});
    auto patch_h = torch::gather(ph_p, 1, pa_exp).reshape({N, d_byte});

    // 2. Fusion layer: cat_h -> u1 -> s1 -> fused1
    auto hb_flat = h_byte.reshape({N, d_byte});
    auto cat_h = torch::cat({hb_flat, patch_h}, -1);
    auto u1 = asdag_bitlinear_forward_cpp(cat_h, w_fus, g_fus, no_bias);
    auto sig1 = torch::sigmoid(u1);
    auto s1 = u1 * sig1;
    auto rms1 = torch::rsqrt(s1.pow(2).mean(-1, true) + 1e-6f);
    auto fused1 = s1 * rms1 * norm1_scale.unsqueeze(0);

    // 3. Layer 2 SwiGLU: fused1 -> ug, uv -> hact -> f2_pre -> fused2
    // Twin gate+val: single activation quant instead of two.
    float gg1 = gate_weight.abs().mean().item<float>();
    if (gg1 < 1e-5f) gg1 = 1e-5f;
    float gg2 = val_weight.abs().mean().item<float>();
    if (gg2 < 1e-5f) gg2 = 1e-5f;
    auto gv = asdag_bitlinear_twin_forward_cpp(
        fused1, w_gate, gg1, w_val, gg2, torch::tensor({}));
    auto ug = gv.slice(1, 0, gv.size(1) / 2);
    auto uv = gv.slice(1, gv.size(1) / 2);
    auto sig_g = torch::sigmoid(ug);
    auto hact = (ug * sig_g) * uv;
    auto f2_pre = fused1 + asdag_bitlinear_forward_cpp(hact, w_down, g_down, no_bias);
    auto rms2 = torch::rsqrt(f2_pre.pow(2).mean(-1, true) + 1e-6f);
    auto fused2 = f2_pre * rms2 * norm2_scale.unsqueeze(0);

    // 4. Chunked Loss & Softmax Backprop (Zero-Logits RAM footprint)
    auto targets_flat = targets.reshape({N});
    auto total_loss = torch::zeros({1}, torch::kFloat32);
    auto g_fused2 = torch::empty({N, d_byte}, torch::kFloat32);
    auto grad_lm = torch::zeros_like(lm_head_weight);

    int64_t chunk_size = 4096;
    float scale = 1.0f / float(N);
    double loss_acc = 0.0;

    for (int64_t c_start = 0; c_start < N; c_start += chunk_size) {
        int64_t c_end = std::min(N, c_start + chunk_size);
        int64_t cur_c = c_end - c_start;

        auto f2_c = fused2.slice(0, c_start, c_end);
        auto tgt_c = targets_flat.slice(0, c_start, c_end);

        auto logits_c = asdag_bitlinear_forward_cpp(f2_c, w_lm, g_lm, no_bias);
        auto max_l = std::get<0>(logits_c.max(-1, true));
        auto exp_l = torch::exp(logits_c - max_l);
        auto sum_exp = exp_l.sum(-1, true);
        auto probs_c = exp_l / sum_exp;

        auto tgt_exp = tgt_c.unsqueeze(1);
        auto loss_c = -torch::log(probs_c.gather(1, tgt_exp)).sum();
        loss_acc += loss_c.item<double>();

        auto d_logits_c = probs_c * scale;
        d_logits_c.scatter_add_(1, tgt_exp, torch::full({cur_c, 1}, -scale, torch::kFloat32));

        auto [g_f2c, g_lm_c, g_lm_b] = asdag_bitlinear_backward_cpp(
            d_logits_c, f2_c, w_lm, g_lm, false
        );
        (void)g_lm_b;
        grad_lm += g_lm_c;
        g_fused2.slice(0, c_start, c_end) = g_f2c;
    }
    total_loss[0] = float(loss_acc * scale);

    // 5. Backprop through Layer 2 RMSNorm & SwiGLU
    auto yn2 = f2_pre * rms2;
    auto sum_g_f2 = (g_fused2 * yn2).sum(-1, true);
    auto g_f2_pre = rms2 * norm2_scale.unsqueeze(0) * (g_fused2 - yn2 * (sum_g_f2 / float(d_byte)));
    auto grad_n2 = (g_fused2 * yn2).sum(0);

    auto [g_hact, grad_down, grad_down_b] = asdag_bitlinear_backward_cpp(
        g_f2_pre, hact, w_down, g_down, false
    );
    (void)grad_down_b;

    auto dsilu_g = sig_g * (1.0f + ug * (1.0f - sig_g));
    auto g_ug = g_hact * uv * dsilu_g;
    auto g_uv = g_hact * (ug * sig_g);

    float hg1 = gate_weight.abs().mean().item<float>();
    if (hg1 < 1e-5f) hg1 = 1e-5f;
    float hg2 = val_weight.abs().mean().item<float>();
    if (hg2 < 1e-5f) hg2 = 1e-5f;
    auto g_gv = torch::cat({g_ug, g_uv}, 1);
    auto [g_fused1_add, grad_gate, grad_val, grad_gv_bias] = asdag_bitlinear_twin_backward_cpp(
        g_gv, fused1, w_gate, hg1, w_val, hg2, false
    );
    (void)grad_gv_bias;
    auto g_fused1 = g_f2_pre + g_fused1_add;

    // 6. Backprop through Layer 1 RMSNorm & Fusion
    auto yn1 = s1 * rms1;
    auto sum_g_f1 = (g_fused1 * yn1).sum(-1, true);
    auto g_s1 = rms1 * norm1_scale.unsqueeze(0) * (g_fused1 - yn1 * (sum_g_f1 / float(d_byte)));
    auto grad_n1 = (g_fused1 * yn1).sum(0);

    auto dsilu1 = sig1 * (1.0f + u1 * (1.0f - sig1));
    auto g_u1 = g_s1 * dsilu1;

    auto [g_cat, grad_fusion, grad_fusion_b] = asdag_bitlinear_backward_cpp(
        g_u1, cat_h, w_fus, g_fus, false
    );
    (void)grad_fusion_b;

    auto grad_h_byte = g_cat.slice(1, 0, d_byte).reshape({B, T, d_byte});
    auto g_patch_h = g_cat.slice(1, d_byte, 2 * d_byte).reshape({B, T, d_byte});

    // 7. Backprop through patch assignments & patch_to_byte_weight
    auto g_ph_p = torch::zeros({B, M, d_byte}, torch::kFloat32);
    g_ph_p.scatter_add_(1, pa_exp, g_patch_h);

    auto g_ph_flat = g_ph_p.reshape({B * M, d_byte});
    auto [g_clp_flat, grad_p2b, grad_p2b_b] = asdag_bitlinear_backward_cpp(
        g_ph_flat, clp_flat, w_p2b, g_p2b, false
    );
    (void)grad_p2b_b;
    auto grad_patches = g_clp_flat.reshape({B, M, d_model});

    return std::make_tuple(
        total_loss,
        grad_h_byte.to(orig_dtype),
        grad_patches.to(orig_dtype),
        grad_p2b.to(orig_dtype),
        grad_fusion.to(orig_dtype),
        grad_gate.to(orig_dtype),
        grad_val.to(orig_dtype),
        grad_down.to(orig_dtype),
        grad_lm.to(orig_dtype),
        grad_n1.to(orig_dtype),
        grad_n2.to(orig_dtype)
    );
}

// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor asdag_fused_rmsnorm_proj_cpp(
    torch::Tensor x,      // [B, dim]
    torch::Tensor weight, // [out_dim, dim]
    float eps
) {
    auto orig_dtype = x.scalar_type();
    x = x.contiguous().to(torch::kFloat32);
    weight = weight.contiguous().to(torch::kFloat32);

    int64_t B = x.size(0);
    int64_t in_dim = x.size(1);
    int64_t out_dim = weight.size(0);

    auto out = torch::empty({B, out_dim}, torch::kFloat32);

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        const float* xb = x_ptr + b * in_dim;
        float* yb = out_ptr + b * out_dim;

        // 1. Compute RMS scale
        float sum_sq = 0.0f;
#if defined(ASDAG_SIMD_AVX2)
        int64_t i = 0;
        __m256 acc = _mm256_setzero_ps();
        for (; i + 8 <= in_dim; i += 8) {
            __m256 xv = _mm256_loadu_ps(xb + i);
            acc = _mm256_fmadd_ps(xv, xv, acc);
        }
        sum_sq = asdag::hsum256(acc);
        for (; i < in_dim; ++i) sum_sq += xb[i] * xb[i];
#else
        for (int64_t i = 0; i < in_dim; ++i) sum_sq += xb[i] * xb[i];
#endif
        float rms_scale = 1.0f / std::sqrt((sum_sq / (float)in_dim) + eps);

        // 2. Fused projection: yb = weight * (xb * rms_scale)
        for (int64_t o = 0; o < out_dim; ++o) {
            const float* w_row = w_ptr + o * in_dim;
            float dot = 0.0f;
#if defined(ASDAG_SIMD_AVX2)
            int64_t k = 0;
            __m256 dot_acc = _mm256_setzero_ps();
            __m256 r_vec = _mm256_set1_ps(rms_scale);
            for (; k + 8 <= in_dim; k += 8) {
                __m256 wv = _mm256_loadu_ps(w_row + k);
                __m256 xv = _mm256_loadu_ps(xb + k);
                dot_acc = _mm256_fmadd_ps(wv, _mm256_mul_ps(xv, r_vec), dot_acc);
            }
            dot = asdag::hsum256(dot_acc);
            for (; k < in_dim; ++k) dot += w_row[k] * (xb[k] * rms_scale);
#else
            for (int64_t k = 0; k < in_dim; ++k) dot += w_row[k] * (xb[k] * rms_scale);
#endif
            yb[o] = dot;
        }
    }

    return out.to(orig_dtype);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> asdag_bitlinear_swiglu_ternary_int8_forward_cpp(
    torch::Tensor x,              // [N, D]
    torch::Tensor w_gate_val,     // [2*H, D]
    float gamma_gv,
    torch::Tensor w_down,         // [D, H]
    float gamma_down
) {
    auto orig_dtype = x.scalar_type();
    x = x.contiguous().to(torch::kFloat32);
    w_gate_val = w_gate_val.contiguous().to(torch::kFloat32);
    w_down = w_down.contiguous().to(torch::kFloat32);

    int64_t N = x.size(0);
    int64_t D = x.size(1);
    int64_t H = w_down.size(1);

    auto out = torch::empty({N, D}, torch::kFloat32);
    auto h_act = torch::empty({N, H}, torch::kFloat32);
    auto gate_raw = torch::empty({N, H}, torch::kFloat32);
    auto val_raw = torch::empty({N, H}, torch::kFloat32);

    const float* x_ptr = x.data_ptr<float>();
    const float* wgv_ptr = w_gate_val.data_ptr<float>();
    const float* wd_ptr = w_down.data_ptr<float>();

    float* out_ptr = out.data_ptr<float>();
    float* ha_ptr = h_act.data_ptr<float>();
    float* gr_ptr = gate_raw.data_ptr<float>();
    float* vr_ptr = val_raw.data_ptr<float>();

    std::vector<int16_t> wgv_i16(2 * H * D);
    std::vector<int16_t> wd_i16(D * H);
    for (int64_t i = 0; i < 2 * H * D; ++i) wgv_i16[i] = (int16_t)std::round(wgv_ptr[i]);
    for (int64_t i = 0; i < D * H; ++i) wd_i16[i] = (int16_t)std::round(wd_ptr[i]);

    int n_threads = asdag::get_physical_cores();

#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        const float* xn = x_ptr + n * D;
        float* han = ha_ptr + n * H;
        float* grn = gr_ptr + n * H;
        float* vrn = vr_ptr + n * H;
        float* outn = out_ptr + n * D;

        // 1. Dynamic Quantization of x to int16
        float max_x = 0.0f;
        for (int64_t d = 0; d < D; ++d) max_x = std::max(max_x, std::abs(xn[d]));
        float alpha_x = 127.0f / (max_x + 1e-5f);
        float inv_scale_gv = gamma_gv / alpha_x;

        std::vector<int16_t> x_i16(D);
        for (int64_t d = 0; d < D; ++d) {
            x_i16[d] = (int16_t)std::clamp((int)std::round(xn[d] * alpha_x), -128, 127);
        }

        // 2. Integer MatMul for w_gate and w_val
        for (int64_t h = 0; h < H; ++h) {
            const int16_t* wg_row = wgv_i16.data() + h * D;
            const int16_t* wv_row = wgv_i16.data() + (H + h) * D;

            int32_t dot_g_i = 0, dot_v_i = 0;
#if defined(ASDAG_SIMD_AVX2)
            int64_t d = 0;
            __m256i acc_g = _mm256_setzero_si256();
            __m256i acc_v = _mm256_setzero_si256();

            for (; d + 16 <= D; d += 16) {
                __m256i xv = _mm256_loadu_si256((const __m256i*)(x_i16.data() + d));
                __m256i gv = _mm256_loadu_si256((const __m256i*)(wg_row + d));
                __m256i vv = _mm256_loadu_si256((const __m256i*)(wv_row + d));
                acc_g = _mm256_add_epi32(acc_g, _mm256_madd_epi16(xv, gv));
                acc_v = _mm256_add_epi32(acc_v, _mm256_madd_epi16(xv, vv));
            }
            dot_g_i = asdag::hsum256_epi32(acc_g);
            dot_v_i = asdag::hsum256_epi32(acc_v);
            for (; d < D; ++d) {
                dot_g_i += x_i16[d] * wg_row[d];
                dot_v_i += x_i16[d] * wv_row[d];
            }
#else
            for (int64_t d = 0; d < D; ++d) {
                dot_g_i += x_i16[d] * wg_row[d];
                dot_v_i += x_i16[d] * wv_row[d];
            }
#endif
            float dot_g = dot_g_i * inv_scale_gv;
            float dot_v = dot_v_i * inv_scale_gv;
            grn[h] = dot_g;
            vrn[h] = dot_v;
            han[h] = (dot_g / (1.0f + std::exp(-dot_g))) * dot_v;
        }

        // 3. Dynamic Quantization of h_act to int16
        float max_h = 0.0f;
        for (int64_t h = 0; h < H; ++h) max_h = std::max(max_h, std::abs(han[h]));
        float alpha_h = 127.0f / (max_h + 1e-5f);
        float inv_scale_d = gamma_down / alpha_h;

        std::vector<int16_t> h_i16(H);
        for (int64_t h = 0; h < H; ++h) {
            h_i16[h] = (int16_t)std::clamp((int)std::round(han[h] * alpha_h), -128, 127);
        }

        // 4. Integer MatMul for w_down
        for (int64_t d = 0; d < D; ++d) {
            const int16_t* wd_row = wd_i16.data() + d * H;
            int32_t dot_d_i = 0;
#if defined(ASDAG_SIMD_AVX2)
            int64_t hk = 0;
            __m256i acc_d = _mm256_setzero_si256();
            for (; hk + 16 <= H; hk += 16) {
                __m256i hv = _mm256_loadu_si256((const __m256i*)(h_i16.data() + hk));
                __m256i wdv = _mm256_loadu_si256((const __m256i*)(wd_row + hk));
                acc_d = _mm256_add_epi32(acc_d, _mm256_madd_epi16(hv, wdv));
            }
            dot_d_i = asdag::hsum256_epi32(acc_d);
            for (; hk < H; ++hk) {
                dot_d_i += h_i16[hk] * wd_row[hk];
            }
#else
            for (int64_t hk = 0; hk < H; ++hk) {
                dot_d_i += h_i16[hk] * wd_row[hk];
            }
#endif
            outn[d] = dot_d_i * inv_scale_d;
        }
    }

    return std::make_tuple(out.to(orig_dtype), h_act, gate_raw, val_raw);
}


// ─────────────────────────────────────────────────────────────────────────────
// 22. C++ Accelerated 5th-Order Newton-Schulz Polar Decomposition (Muon Optimizer)
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor asdag_newton_schulz5_cpp(
    torch::Tensor G,
    int steps = 5,
    float eps = 1e-7f
) {
    auto orig_dtype = G.scalar_type();
    auto G_f = G.contiguous().to(torch::kFloat32);

    int64_t rows = G_f.size(0);
    int64_t cols = G_f.size(1);
    bool transposed = false;

    if (rows > cols) {
        G_f = G_f.t().contiguous();
        std::swap(rows, cols);
        transposed = true;
    }

    // Spectral norm normalization: X = G / (norm + eps)
    float norm = G_f.norm().item<float>() + eps;
    auto X = G_f / norm;

    const float a = 3.4445f;
    const float b = -4.7750f;
    const float c = 2.0315f;

    for (int s = 0; s < steps; ++s) {
        auto A = torch::mm(X, X.t());                     // [rows, rows]
        auto AA = torch::mm(A, A);                        // [rows, rows]
        auto B = b * A + c * AA;                          // [rows, rows]
        X = a * X + torch::mm(B, X);                      // [rows, cols]
    }

    if (transposed) {
        X = X.t().contiguous();
    }

    return X.to(orig_dtype);
}

// ─────────────────────────────────────────────────────────────────────────────
// 23. C++ Fused BitLinear SwiGLU Gated Channel Mixer (Forward & Backward)
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> asdag_bitlinear_swiglu_forward_cpp(
    torch::Tensor x,              // [N, D]
    torch::Tensor w_gate_val,     // [2*H, D]
    float gamma_gv,
    torch::Tensor w_down,         // [D, H]
    float gamma_down
) {
    auto orig_dtype = x.scalar_type();
    x = x.contiguous().to(torch::kFloat32);
    w_gate_val = w_gate_val.contiguous().to(torch::kFloat32);
    w_down = w_down.contiguous().to(torch::kFloat32);

    int64_t N = x.size(0);
    int64_t D = x.size(1);
    int64_t H = w_down.size(1);

    auto out = torch::empty({N, D}, torch::kFloat32);
    auto h_act = torch::empty({N, H}, torch::kFloat32);
    auto gate_raw = torch::empty({N, H}, torch::kFloat32);
    auto val_raw = torch::empty({N, H}, torch::kFloat32);

    const float* x_ptr = x.data_ptr<float>();
    const float* wgv_ptr = w_gate_val.data_ptr<float>();
    const float* wd_ptr = w_down.data_ptr<float>();

    float* out_ptr = out.data_ptr<float>();
    float* ha_ptr = h_act.data_ptr<float>();
    float* gr_ptr = gate_raw.data_ptr<float>();
    float* vr_ptr = val_raw.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        const float* xn = x_ptr + n * D;
        float* han = ha_ptr + n * H;
        float* grn = gr_ptr + n * H;
        float* vrn = vr_ptr + n * H;
        float* outn = out_ptr + n * D;

        int64_t h = 0;
        for (; h + 4 <= H; h += 4) {
            const float* wg0 = wgv_ptr + (h + 0) * D;
            const float* wv0 = wgv_ptr + (H + h + 0) * D;
            const float* wg1 = wgv_ptr + (h + 1) * D;
            const float* wv1 = wgv_ptr + (H + h + 1) * D;
            const float* wg2 = wgv_ptr + (h + 2) * D;
            const float* wv2 = wgv_ptr + (H + h + 2) * D;
            const float* wg3 = wgv_ptr + (h + 3) * D;
            const float* wv3 = wgv_ptr + (H + h + 3) * D;

            float dot_g0 = 0.0f, dot_v0 = 0.0f;
            float dot_g1 = 0.0f, dot_v1 = 0.0f;
            float dot_g2 = 0.0f, dot_v2 = 0.0f;
            float dot_g3 = 0.0f, dot_v3 = 0.0f;

#if defined(ASDAG_SIMD_AVX2)
            int64_t d = 0;
            __m256 acc_g0 = _mm256_setzero_ps();
            __m256 acc_v0 = _mm256_setzero_ps();
            __m256 acc_g1 = _mm256_setzero_ps();
            __m256 acc_v1 = _mm256_setzero_ps();
            __m256 acc_g2 = _mm256_setzero_ps();
            __m256 acc_v2 = _mm256_setzero_ps();
            __m256 acc_g3 = _mm256_setzero_ps();
            __m256 acc_v3 = _mm256_setzero_ps();

            for (; d + 8 <= D; d += 8) {
                __m256 xv = _mm256_loadu_ps(xn + d);
                acc_g0 = _mm256_fmadd_ps(_mm256_loadu_ps(wg0 + d), xv, acc_g0);
                acc_v0 = _mm256_fmadd_ps(_mm256_loadu_ps(wv0 + d), xv, acc_v0);
                acc_g1 = _mm256_fmadd_ps(_mm256_loadu_ps(wg1 + d), xv, acc_g1);
                acc_v1 = _mm256_fmadd_ps(_mm256_loadu_ps(wv1 + d), xv, acc_v1);
                acc_g2 = _mm256_fmadd_ps(_mm256_loadu_ps(wg2 + d), xv, acc_g2);
                acc_v2 = _mm256_fmadd_ps(_mm256_loadu_ps(wv2 + d), xv, acc_v2);
                acc_g3 = _mm256_fmadd_ps(_mm256_loadu_ps(wg3 + d), xv, acc_g3);
                acc_v3 = _mm256_fmadd_ps(_mm256_loadu_ps(wv3 + d), xv, acc_v3);
            }
            dot_g0 = asdag::hsum256(acc_g0); dot_v0 = asdag::hsum256(acc_v0);
            dot_g1 = asdag::hsum256(acc_g1); dot_v1 = asdag::hsum256(acc_v1);
            dot_g2 = asdag::hsum256(acc_g2); dot_v2 = asdag::hsum256(acc_v2);
            dot_g3 = asdag::hsum256(acc_g3); dot_v3 = asdag::hsum256(acc_v3);
            for (; d < D; ++d) {
                float x_val = xn[d];
                dot_g0 += wg0[d] * x_val; dot_v0 += wv0[d] * x_val;
                dot_g1 += wg1[d] * x_val; dot_v1 += wv1[d] * x_val;
                dot_g2 += wg2[d] * x_val; dot_v2 += wv2[d] * x_val;
                dot_g3 += wg3[d] * x_val; dot_v3 += wv3[d] * x_val;
            }
#else
            for (int64_t d = 0; d < D; ++d) {
                float x_val = xn[d];
                dot_g0 += wg0[d] * x_val; dot_v0 += wv0[d] * x_val;
                dot_g1 += wg1[d] * x_val; dot_v1 += wv1[d] * x_val;
                dot_g2 += wg2[d] * x_val; dot_v2 += wv2[d] * x_val;
                dot_g3 += wg3[d] * x_val; dot_v3 += wv3[d] * x_val;
            }
#endif
            dot_g0 *= gamma_gv; dot_v0 *= gamma_gv;
            dot_g1 *= gamma_gv; dot_v1 *= gamma_gv;
            dot_g2 *= gamma_gv; dot_v2 *= gamma_gv;
            dot_g3 *= gamma_gv; dot_v3 *= gamma_gv;

            grn[h + 0] = dot_g0; vrn[h + 0] = dot_v0;
            grn[h + 1] = dot_g1; vrn[h + 1] = dot_v1;
            grn[h + 2] = dot_g2; vrn[h + 2] = dot_v2;
            grn[h + 3] = dot_g3; vrn[h + 3] = dot_v3;

            han[h + 0] = (dot_g0 / (1.0f + std::exp(-dot_g0))) * dot_v0;
            han[h + 1] = (dot_g1 / (1.0f + std::exp(-dot_g1))) * dot_v1;
            han[h + 2] = (dot_g2 / (1.0f + std::exp(-dot_g2))) * dot_v2;
            han[h + 3] = (dot_g3 / (1.0f + std::exp(-dot_g3))) * dot_v3;
        }

        for (; h < H; ++h) {
            const float* w_gate = wgv_ptr + h * D;
            const float* w_val = wgv_ptr + (H + h) * D;
            float dot_g = 0.0f, dot_v = 0.0f;
            for (int64_t d = 0; d < D; ++d) {
                dot_g += w_gate[d] * xn[d];
                dot_v += w_val[d] * xn[d];
            }
            dot_g *= gamma_gv;
            dot_v *= gamma_gv;
            grn[h] = dot_g;
            vrn[h] = dot_v;
            han[h] = (dot_g / (1.0f + std::exp(-dot_g))) * dot_v;
        }

        int64_t d = 0;
        for (; d + 4 <= D; d += 4) {
            const float* wd0 = wd_ptr + (d + 0) * H;
            const float* wd1 = wd_ptr + (d + 1) * H;
            const float* wd2 = wd_ptr + (d + 2) * H;
            const float* wd3 = wd_ptr + (d + 3) * H;

            float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
#if defined(ASDAG_SIMD_AVX2)
            int64_t hk = 0;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (; hk + 8 <= H; hk += 8) {
                __m256 hv = _mm256_loadu_ps(han + hk);
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(wd0 + hk), hv, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(wd1 + hk), hv, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(wd2 + hk), hv, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(wd3 + hk), hv, acc3);
            }
            dot0 = asdag::hsum256(acc0);
            dot1 = asdag::hsum256(acc1);
            dot2 = asdag::hsum256(acc2);
            dot3 = asdag::hsum256(acc3);
            for (; hk < H; ++hk) {
                float h_val = han[hk];
                dot0 += wd0[hk] * h_val;
                dot1 += wd1[hk] * h_val;
                dot2 += wd2[hk] * h_val;
                dot3 += wd3[hk] * h_val;
            }
#else
            for (int64_t hk = 0; hk < H; ++hk) {
                float h_val = han[hk];
                dot0 += wd0[hk] * h_val;
                dot1 += wd1[hk] * h_val;
                dot2 += wd2[hk] * h_val;
                dot3 += wd3[hk] * h_val;
            }
#endif
            outn[d + 0] = dot0 * gamma_down;
            outn[d + 1] = dot1 * gamma_down;
            outn[d + 2] = dot2 * gamma_down;
            outn[d + 3] = dot3 * gamma_down;
        }

        for (; d < D; ++d) {
            const float* wd_row = wd_ptr + d * H;
            float dot = 0.0f;
            for (int64_t hk = 0; hk < H; ++hk) dot += wd_row[hk] * han[hk];
            outn[d] = dot * gamma_down;
        }
    }

    return std::make_tuple(out.to(orig_dtype), h_act, gate_raw, val_raw);
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> asdag_bitlinear_swiglu_backward_recompute_cpp(
    torch::Tensor grad_output,    // [N, D]
    torch::Tensor x,              // [N, D]
    torch::Tensor w_gate_val,     // [2*H, D]
    float gamma_gv,
    torch::Tensor w_down,         // [D, H]
    float gamma_down
) {
    auto orig_dtype = grad_output.scalar_type();
    grad_output = grad_output.contiguous().to(torch::kFloat32);
    x = x.contiguous().to(torch::kFloat32);
    w_gate_val = w_gate_val.contiguous().to(torch::kFloat32);
    w_down = w_down.contiguous().to(torch::kFloat32);

    int64_t N = x.size(0);
    int64_t D = x.size(1);
    int64_t H = w_down.size(1);

    const float* go_ptr = grad_output.data_ptr<float>();
    const float* x_ptr = x.data_ptr<float>();
    const float* wgv_ptr = w_gate_val.data_ptr<float>();
    const float* wd_ptr = w_down.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

    // 1. High-throughput BLAS GEMMs for upstream gradient and forward activation recomputation
    auto g_h_all = torch::mm(grad_output, w_down) * gamma_down;
    auto gv_raw_all = torch::mm(x, w_gate_val.t()) * gamma_gv;

    const float* gh_ptr = g_h_all.data_ptr<float>();
    const float* gv_raw_ptr = gv_raw_all.data_ptr<float>();

    auto g_gv_all = torch::empty({N, 2 * H}, torch::kFloat32);
    auto h_act_all = torch::empty({N, H}, torch::kFloat32);
    float* ggv_ptr = g_gv_all.data_ptr<float>();
    float* ha_ptr = h_act_all.data_ptr<float>();

#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        const float* ghn = gh_ptr + n * H;
        const float* grn = gv_raw_ptr + n * (2 * H);
        const float* vrn = grn + H;
        float* ggvn = ggv_ptr + n * (2 * H);
        float* han = ha_ptr + n * H;

#if defined(ASDAG_SIMD_AVX2)
        int64_t h = 0;
        const __m256 one_v = _mm256_set1_ps(1.0f);
        for (; h + 8 <= H; h += 8) {
            __m256 g_raw_v = _mm256_loadu_ps(grn + h);
            __m256 v_raw_v = _mm256_loadu_ps(vrn + h);
            __m256 g_h_v = _mm256_loadu_ps(ghn + h);

            __m256 sig_g_v = asdag::sigmoid256_ps(g_raw_v);
            __m256 silu_g_v = _mm256_mul_ps(g_raw_v, sig_g_v);
            __m256 dsilu_g_v = _mm256_mul_ps(sig_g_v, _mm256_fmadd_ps(g_raw_v, _mm256_sub_ps(one_v, sig_g_v), one_v));

            __m256 han_v = _mm256_mul_ps(silu_g_v, v_raw_v);
            _mm256_storeu_ps(han + h, han_v);

            __m256 g_gate_v = _mm256_mul_ps(_mm256_mul_ps(g_h_v, v_raw_v), dsilu_g_v);
            __m256 g_val_v = _mm256_mul_ps(g_h_v, silu_g_v);

            _mm256_storeu_ps(ggvn + h, g_gate_v);
            _mm256_storeu_ps(ggvn + H + h, g_val_v);
        }
        for (; h < H; ++h) {
            float g_raw = grn[h];
            float v_raw = vrn[h];
            float g_h = ghn[h];

            float sig_g = 1.0f / (1.0f + std::exp(-g_raw));
            float silu_g = g_raw * sig_g;
            float dsilu_g = sig_g * (1.0f + g_raw * (1.0f - sig_g));

            han[h] = silu_g * v_raw;
            ggvn[h] = g_h * v_raw * dsilu_g;
            ggvn[H + h] = g_h * silu_g;
        }
#else
        for (int64_t h = 0; h < H; ++h) {
            float g_raw = grn[h];
            float v_raw = vrn[h];
            float g_h = ghn[h];

            float sig_g = 1.0f / (1.0f + std::exp(-g_raw));
            float silu_g = g_raw * sig_g;
            float dsilu_g = sig_g * (1.0f + g_raw * (1.0f - sig_g));

            han[h] = silu_g * v_raw;
            ggvn[h] = g_h * v_raw * dsilu_g;
            ggvn[H + h] = g_h * silu_g;
        }
#endif
    }

    // 2. High-throughput GEMMs for grad_x, grad_w_gate_val, and grad_w_down
    auto grad_x = torch::mm(g_gv_all, w_gate_val) * gamma_gv;
    auto grad_wgv = torch::mm(g_gv_all.t(), x);
    auto grad_wd = torch::mm(grad_output.t(), h_act_all);

    return std::make_tuple(grad_x.to(orig_dtype), grad_wgv.to(orig_dtype), grad_wd.to(orig_dtype));
}

// ─────────────────────────────────────────────────────────────────────────────
// 24. Breakthrough 4: Sub-Byte SIMD Dynamic Entropy Patcher (AVX2 Fast Scan)
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor> asdag_blt_simd_patcher_cpp(
    torch::Tensor byte_embeddings,  // [B, T, D_byte]
    torch::Tensor byte_logits,      // [B, T, 256] or [B, T, 1] entropy scores
    int64_t target_patch_size,
    int64_t max_patches
) {
    auto orig_dtype = byte_embeddings.scalar_type();
    byte_embeddings = byte_embeddings.contiguous().to(torch::kFloat32);

    int64_t B = byte_embeddings.size(0);
    int64_t T = byte_embeddings.size(1);
    int64_t D_byte = byte_embeddings.size(2);
    int64_t M = max_patches > 0 ? max_patches : std::max<int64_t>(1, (T + target_patch_size - 1) / target_patch_size);

    auto patch_assignments = torch::empty({B, T}, torch::kInt64);
    auto pooled_patches = torch::zeros({B, M, D_byte}, torch::kFloat32);

    const float* be_ptr = byte_embeddings.data_ptr<float>();
    int64_t* pa_ptr = patch_assignments.data_ptr<int64_t>();
    float* pp_ptr = pooled_patches.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        int64_t p_idx = 0;
        int64_t patch_start = 0;

        for (int64_t t = 0; t < T; ++t) {
            pa_ptr[b * T + t] = p_idx;

            // Check boundary: close a patch at every target_patch_size boundary, or
            // at the final byte (flushing a possibly-short tail patch as a true mean).
            // The old `|| p_idx >= M - 1` early-close corrupted the tail: it re-accumulated
            // byte vectors into the same slot with inv_len = 1, turning the mean into a sum.
            bool is_boundary = ((t + 1) % target_patch_size == 0) || (t == T - 1);
            if (is_boundary) {
                // Mean pool slice [patch_start, t] into pooled_patches[b, p_idx]
                if (p_idx >= M) p_idx = M - 1;
                int64_t patch_len = t - patch_start + 1;
                float inv_len = 1.0f / (float)patch_len;
                float* cur_patch = pp_ptr + (b * M + p_idx) * D_byte;

                for (int64_t i = patch_start; i <= t; ++i) {
                    const float* byte_vec = be_ptr + (b * T + i) * D_byte;
                    int64_t d = 0;
#if defined(ASDAG_SIMD_AVX2)
                    for (; d + 8 <= D_byte; d += 8) {
                        __m256 pv = _mm256_loadu_ps(cur_patch + d);
                        __m256 bv = _mm256_loadu_ps(byte_vec + d);
                        _mm256_storeu_ps(cur_patch + d, _mm256_add_ps(pv, bv));
                    }
#endif
                    for (; d < D_byte; ++d) {
                        cur_patch[d] += byte_vec[d];
                    }
                }

                // Scale by inv_len
                int64_t d = 0;
#if defined(ASDAG_SIMD_AVX2)
                __m256 s_vec = _mm256_set1_ps(inv_len);
                for (; d + 8 <= D_byte; d += 8) {
                    __m256 pv = _mm256_loadu_ps(cur_patch + d);
                    _mm256_storeu_ps(cur_patch + d, _mm256_mul_ps(pv, s_vec));
                }
#endif
                for (; d < D_byte; ++d) {
                    cur_patch[d] *= inv_len;
                }

                p_idx = std::min(p_idx + 1, M - 1);
                patch_start = t + 1;
            }
        }
    }

    return std::make_tuple(pooled_patches.to(orig_dtype), patch_assignments);
}

// NOTE: incremental generation (TorosHybridLanguageModel.forward_incremental) mirrors
// this mean-pooling semantics patch-by-patch; keep them in sync if this changes.

// ─────────────────────────────────────────────────────────────────────────────
// 25. Breakthrough 2: 1-Cycle Pure Integer Ternary Add/Sub BitLinear Forward
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor asdag_bitlinear_ternary_int_forward_cpp(
    torch::Tensor x,              // [N, in_dim]
    torch::Tensor w_ternary,      // [out_dim, in_dim] in {-1.0, 0.0, +1.0}
    float gamma,
    torch::Tensor bias            // [out_dim] optional
) {
    auto orig_dtype = x.scalar_type();
    x = x.contiguous().to(torch::kFloat32);
    w_ternary = w_ternary.contiguous().to(torch::kFloat32);

    int64_t N = x.size(0);
    int64_t in_dim = x.size(1);
    int64_t out_dim = w_ternary.size(0);
    bool has_bias = bias.defined() && bias.numel() > 0;

    auto out = torch::empty({N, out_dim}, torch::kFloat32);

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w_ternary.data_ptr<float>();
    const float* b_ptr = has_bias ? bias.contiguous().to(torch::kFloat32).data_ptr<float>() : nullptr;
    float* out_ptr = out.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        const float* xn = x_ptr + n * in_dim;
        float* outn = out_ptr + n * out_dim;

        for (int64_t o = 0; o < out_dim; ++o) {
            const float* w_row = w_ptr + o * in_dim;
            float dot = 0.0f;

#if defined(ASDAG_SIMD_AVX2)
            int64_t i = 0;
            __m256 pos_acc = _mm256_setzero_ps();
            __m256 neg_acc = _mm256_setzero_ps();
            __m256 zero_v = _mm256_setzero_ps();

            for (; i + 8 <= in_dim; i += 8) {
                __m256 xv = _mm256_loadu_ps(xn + i);
                __m256 wv = _mm256_loadu_ps(w_row + i);

                __m256 is_pos = _mm256_cmp_ps(wv, zero_v, _CMP_GT_OQ);
                __m256 is_neg = _mm256_cmp_ps(wv, zero_v, _CMP_LT_OQ);

                __m256 pos_x = _mm256_and_ps(xv, is_pos);
                __m256 neg_x = _mm256_and_ps(xv, is_neg);

                pos_acc = _mm256_add_ps(pos_acc, pos_x);
                neg_acc = _mm256_add_ps(neg_acc, neg_x);
            }
            __m256 net_acc = _mm256_sub_ps(pos_acc, neg_acc);
            dot = asdag::hsum256(net_acc);

            for (; i < in_dim; ++i) {
                float w_val = w_row[i];
                if (w_val > 0.0f) dot += xn[i];
                else if (w_val < 0.0f) dot -= xn[i];
            }
#else
            for (int64_t i = 0; i < in_dim; ++i) {
                float w_val = w_row[i];
                if (w_val > 0.0f) dot += xn[i];
                else if (w_val < 0.0f) dot -= xn[i];
            }
#endif
            float val = dot * gamma;
            if (has_bias) val += b_ptr[o];
            outn[o] = val;
        }
    }

    return out.to(orig_dtype);
}

// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> asdag_fused_monarch_gla_forward_cpp(
    torch::Tensor x,              // [B, T, C]
    torch::Tensor qkvg_diagonals, // [4, S, C]
    torch::Tensor qkvg_perms,     // [S - 1, C]
    torch::Tensor qkvg_bias,      // [4, C]
    torch::Tensor q_norm_scale,   // [D]
    torch::Tensor k_norm_scale,   // [D]
    torch::Tensor w_decay,        // [H, C]
    torch::Tensor b_decay,        // [H]
    torch::Tensor out_diagonals,  // [S, C]
    torch::Tensor out_perms,      // [S - 1, C]
    torch::Tensor out_bias,       // [C]
    torch::Tensor reset_mask      // [B, T] (optional)
) {
    auto orig_dtype = x.scalar_type();
    x = x.contiguous().to(torch::kFloat32);
    qkvg_diagonals = qkvg_diagonals.contiguous().to(torch::kFloat32);
    qkvg_perms = qkvg_perms.contiguous().to(torch::kInt32);
    qkvg_bias = qkvg_bias.contiguous().to(torch::kFloat32);
    q_norm_scale = q_norm_scale.contiguous().to(torch::kFloat32);
    k_norm_scale = k_norm_scale.contiguous().to(torch::kFloat32);
    w_decay = w_decay.contiguous().to(torch::kFloat32);
    b_decay = b_decay.contiguous().to(torch::kFloat32);
    out_diagonals = out_diagonals.contiguous().to(torch::kFloat32);
    out_perms = out_perms.contiguous().to(torch::kInt32);
    out_bias = out_bias.contiguous().to(torch::kFloat32);

    int64_t B = x.size(0);
    int64_t T = x.size(1);
    int64_t C = x.size(2);
    int64_t H = w_decay.size(0);
    int64_t D = C / H;
    int64_t S_in = qkvg_diagonals.size(1);
    int64_t S_out = out_diagonals.size(0);
    int64_t K_chunk = 32;
    int64_t num_chunks = (T + K_chunk - 1) / K_chunk;

    bool has_reset = reset_mask.defined() && reset_mask.numel() > 0;
    auto reset_contig = has_reset ? reset_mask.contiguous().to(torch::kInt64) : torch::empty({0});
    const int64_t* rm_ptr = has_reset ? reset_contig.data_ptr<int64_t>() : nullptr;

    auto out_y = torch::empty({B, T, C}, x.options());
    auto qkvg_raw = torch::empty({4, B, T, C}, x.options());
    auto phi_q = torch::empty({B, H, T, D}, x.options());
    auto phi_k = torch::empty({B, H, T, D}, x.options());
    auto gamma_all = torch::empty({B, H, T}, x.options());
    auto S_checkpoints = torch::empty({B, H, num_chunks, D, D}, x.options());
    auto z_all = torch::empty({B, H, T, D}, x.options());
    auto y_mod = torch::empty({B, T, C}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* qkvg_d_ptr = qkvg_diagonals.data_ptr<float>();
    const int32_t* qkvg_p_ptr = qkvg_perms.data_ptr<int32_t>();
    const float* qkvg_b_ptr = qkvg_bias.data_ptr<float>();
    const float* q_scale_ptr = q_norm_scale.data_ptr<float>();
    const float* k_scale_ptr = k_norm_scale.data_ptr<float>();
    const float* wd_ptr = w_decay.data_ptr<float>();
    const float* bd_ptr = b_decay.data_ptr<float>();

    float* qkvg_raw_ptr = qkvg_raw.data_ptr<float>();
    float* phi_q_ptr = phi_q.data_ptr<float>();
    float* phi_k_ptr = phi_k.data_ptr<float>();
    float* gam_ptr = gamma_all.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

#pragma omp parallel num_threads(n_threads)
    {
        std::vector<float> h_buf1(C);
        std::vector<float> h_buf2(C);

#pragma omp for collapse(2) schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t t = 0; t < T; ++t) {
                const float* x_bt = x_ptr + (b * T + t) * C;

                // 1. QKVG Projections (4 branches)
                for (int64_t m = 0; m < 4; ++m) {
                    const float* d_m = qkvg_d_ptr + m * (S_in * C);
                    const float* b_m = qkvg_b_ptr + m * C;
                    float* raw_out = qkvg_raw_ptr + (m * (B * T) + (b * T + t)) * C;

                    float* h_cur = h_buf1.data();
                    float* h_next = h_buf2.data();

                    for (int64_t c = 0; c < C; ++c) h_cur[c] = x_bt[c] * d_m[c];

                    for (int64_t s = 0; s < S_in - 1; ++s) {
                        const int32_t* perm_s = qkvg_p_ptr + s * C;
                        const float* ds = d_m + (s + 1) * C;
                        for (int64_t c = 0; c < C; ++c) h_next[c] = h_cur[perm_s[c]] * ds[c];
                        std::swap(h_cur, h_next);
                    }

                    for (int64_t c = 0; c < C; ++c) raw_out[c] = h_cur[c] + b_m[c];
                }

                // Q/K RMSNorm + ELU(+)
                const float* q_raw = qkvg_raw_ptr + (0 * (B * T) + (b * T + t)) * C;
                const float* k_raw = qkvg_raw_ptr + (1 * (B * T) + (b * T + t)) * C;

                for (int64_t h = 0; h < H; ++h) {
                    const float* q_h = q_raw + h * D;
                    const float* k_h = k_raw + h * D;
                    float* phi_q_out = phi_q_ptr + ((b * H + h) * T + t) * D;
                    float* phi_k_out = phi_k_ptr + ((b * H + h) * T + t) * D;

                    float sum_sq_q = 0.0f;
                    float sum_sq_k = 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        sum_sq_q += q_h[d] * q_h[d];
                        sum_sq_k += k_h[d] * k_h[d];
                    }
                    float rms_q = 1.0f / std::sqrt((sum_sq_q / (float)D) + 1e-5f);
                    float rms_k = 1.0f / std::sqrt((sum_sq_k / (float)D) + 1e-5f);

                    for (int64_t d = 0; d < D; ++d) {
                        float q_norm = q_h[d] * rms_q * q_scale_ptr[d];
                        float k_norm = k_h[d] * rms_k * k_scale_ptr[d];
                        phi_q_out[d] = q_norm >= 0.0f ? (q_norm + 1.0f) : (std::exp(q_norm));
                        phi_k_out[d] = k_norm >= 0.0f ? (k_norm + 1.0f) : (std::exp(k_norm));
                    }

                    const float* wd_h = wd_ptr + h * C;
                    float dot_decay = bd_ptr[h];
                    for (int64_t c = 0; c < C; ++c) dot_decay += wd_h[c] * x_bt[c];
                    float gam_val = 1.0f / (1.0f + std::exp(-dot_decay));
                    if (rm_ptr && rm_ptr[b * T + t] != 0) gam_val = 0.0f;
                    gam_ptr[(b * H + h) * T + t] = gam_val;
                }
            }
        }
    }

    // 2. Fused Causal Associative Prefix Scan across Heads (L3 Chunk Checkpointing)
    float* S_chk_ptr = S_checkpoints.data_ptr<float>();
    float* z_ptr = z_all.data_ptr<float>();
    float* y_mod_ptr = y_mod.data_ptr<float>();
    const float* v_raw_ptr = qkvg_raw_ptr + (2 * (B * T)) * C;
    const float* g_raw_ptr = qkvg_raw_ptr + (3 * (B * T)) * C;

#pragma omp parallel for collapse(2) num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            std::vector<float> S(D * D, 0.0f);
            std::vector<float> z(D, 0.0f);

            for (int64_t t = 0; t < T; ++t) {
                if (t % K_chunk == 0) {
                    int64_t c_idx = t / K_chunk;
                    float* S_dst = S_chk_ptr + ((b * H + h) * num_chunks + c_idx) * (D * D);
                    std::memcpy(S_dst, S.data(), D * D * sizeof(float));
                }

                int64_t bht_idx = ((b * H + h) * T + t);
                const float* phi_qt = phi_q_ptr + bht_idx * D;
                const float* phi_kt = phi_k_ptr + bht_idx * D;
                const float* vt = v_raw_ptr + (b * T + t) * C + h * D;
                float gam = gam_ptr[bht_idx];

                float* zt_out = z_ptr + bht_idx * D;
                float* ym_bt = y_mod_ptr + (b * T + t) * C + h * D;
                const float* g_bt = g_raw_ptr + (b * T + t) * C + h * D;

                float den = 0.0f;
                for (int64_t d = 0; d < D; ++d) {
                    float z_new = gam * z[d] + phi_kt[d];
                    z[d] = z_new;
                    zt_out[d] = z_new;
                    den += phi_qt[d] * z_new;
                }
                den = std::max(den, 1e-5f);
                float inv_den = 1.0f / den;

                for (int64_t j = 0; j < D; ++j) {
                    float vj = vt[j];
                    float num_j = 0.0f;
                    for (int64_t i = 0; i < D; ++i) {
                        float s_new = gam * S[i * D + j] + phi_kt[i] * vj;
                        S[i * D + j] = s_new;
                        num_j += phi_qt[i] * s_new;
                    }
                    float y_scan_val = num_j * inv_den;
                    float gj = g_bt[j];
                    float silu_g = gj / (1.0f + std::exp(-gj));
                    ym_bt[j] = y_scan_val * silu_g;
                }
            }
        }
    }

    // 3. Out Monarch Permutation Chain Projection
    const float* out_d_ptr = out_diagonals.data_ptr<float>();
    const int32_t* out_p_ptr = out_perms.data_ptr<int32_t>();
    const float* out_b_ptr = out_bias.data_ptr<float>();
    float* out_ptr = out_y.data_ptr<float>();

#pragma omp parallel num_threads(n_threads)
    {
        std::vector<float> h_buf1(C);
        std::vector<float> h_buf2(C);

#pragma omp for collapse(2) schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t t = 0; t < T; ++t) {
                const float* ym_bt = y_mod_ptr + (b * T + t) * C;
                float* y_out = out_ptr + (b * T + t) * C;

                float* h_cur = h_buf1.data();
                float* h_next = h_buf2.data();

                for (int64_t c = 0; c < C; ++c) h_cur[c] = ym_bt[c] * out_d_ptr[c];

                for (int64_t s = 0; s < S_out - 1; ++s) {
                    const int32_t* perm_s = out_p_ptr + s * C;
                    const float* ds = out_d_ptr + (s + 1) * C;
                    for (int64_t c = 0; c < C; ++c) h_next[c] = h_cur[perm_s[c]] * ds[c];
                    std::swap(h_cur, h_next);
                }

                for (int64_t c = 0; c < C; ++c) y_out[c] = h_cur[c] + out_b_ptr[c];
            }
        }
    }

    return std::make_tuple(
        out_y.to(orig_dtype),
        qkvg_raw,
        phi_q,
        phi_k,
        gamma_all,
        S_checkpoints,
        z_all,
        y_mod
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// 31. C++ Fused Monarch GLA Block Backward (L3 Chunk Checkpointing Recompute)
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> asdag_fused_monarch_gla_backward_cpp(
    torch::Tensor grad_y,         // [B, T, C]
    torch::Tensor x,              // [B, T, C]
    torch::Tensor qkvg_diagonals, // [4, S, C]
    torch::Tensor qkvg_perms,     // [S - 1, C]
    torch::Tensor qkvg_inv_perms, // [S - 1, C]
    torch::Tensor qkvg_raw,       // [4, B, T, C]
    torch::Tensor phi_q,          // [B, H, T, D]
    torch::Tensor phi_k,          // [B, H, T, D]
    torch::Tensor gamma_all,      // [B, H, T]
    torch::Tensor S_checkpoints,  // [B, H, num_chunks, D, D]
    torch::Tensor z_all,          // [B, H, T, D]
    torch::Tensor y_mod,          // [B, T, C]
    torch::Tensor q_norm_scale,   // [D]
    torch::Tensor k_norm_scale,   // [D]
    torch::Tensor w_decay,        // [H, C]
    torch::Tensor b_decay,        // [H]
    torch::Tensor out_diagonals,  // [S, C]
    torch::Tensor out_perms,      // [S - 1, C]
    torch::Tensor out_inv_perms,  // [S - 1, C]
    torch::Tensor reset_mask      // [B, T] (optional)
) {
    auto orig_dtype = grad_y.scalar_type();
    grad_y = grad_y.contiguous().to(torch::kFloat32);
    x = x.contiguous().to(torch::kFloat32);
    qkvg_diagonals = qkvg_diagonals.contiguous().to(torch::kFloat32);
    qkvg_perms = qkvg_perms.contiguous().to(torch::kInt32);
    qkvg_inv_perms = qkvg_inv_perms.contiguous().to(torch::kInt32);
    qkvg_raw = qkvg_raw.contiguous().to(torch::kFloat32);
    phi_q = phi_q.contiguous().to(torch::kFloat32);
    phi_k = phi_k.contiguous().to(torch::kFloat32);
    gamma_all = gamma_all.contiguous().to(torch::kFloat32);
    S_checkpoints = S_checkpoints.contiguous().to(torch::kFloat32);
    z_all = z_all.contiguous().to(torch::kFloat32);
    y_mod = y_mod.contiguous().to(torch::kFloat32);
    q_norm_scale = q_norm_scale.contiguous().to(torch::kFloat32);
    k_norm_scale = k_norm_scale.contiguous().to(torch::kFloat32);
    w_decay = w_decay.contiguous().to(torch::kFloat32);
    b_decay = b_decay.contiguous().to(torch::kFloat32);
    out_diagonals = out_diagonals.contiguous().to(torch::kFloat32);
    out_perms = out_perms.contiguous().to(torch::kInt32);
    out_inv_perms = out_inv_perms.contiguous().to(torch::kInt32);

    int64_t B = x.size(0);
    int64_t T = x.size(1);
    int64_t C = x.size(2);
    int64_t H = w_decay.size(0);
    int64_t D = C / H;
    int64_t S_in = qkvg_diagonals.size(1);
    int64_t S_out = out_diagonals.size(0);
    int64_t K_chunk = 32;
    int64_t num_chunks = (T + K_chunk - 1) / K_chunk;

    bool has_reset = reset_mask.defined() && reset_mask.numel() > 0;
    auto reset_contig = has_reset ? reset_mask.contiguous().to(torch::kInt64) : torch::empty({0});
    const int64_t* rm_ptr = has_reset ? reset_contig.data_ptr<int64_t>() : nullptr;

    auto grad_x = torch::zeros({B, T, C}, x.options());
    auto grad_qkvg_d = torch::zeros({4, S_in, C}, x.options());
    auto grad_qkvg_b = torch::zeros({4, C}, x.options());
    auto grad_q_scale = torch::zeros({D}, x.options());
    auto grad_k_scale = torch::zeros({D}, x.options());
    auto grad_wd = torch::zeros({H, C}, x.options());
    auto grad_bd = torch::zeros({H}, x.options());
    auto grad_out_d = torch::zeros({S_out, C}, x.options());
    auto grad_out_b = torch::zeros({C}, x.options());

    auto grad_ym = torch::empty({B, T, C}, x.options());
    auto grad_qkvg_raw = torch::zeros({4, B, T, C}, x.options());

    // 1. Backward through Out Monarch Chain (Vectorized)
    {
        auto gy_flat = grad_y.reshape({B * T, C});
        auto ym_flat = y_mod.reshape({B * T, C});
        auto [g_ym, g_od, g_ob] = asdag_monarch_chain_backward_cpp(gy_flat, ym_flat, out_diagonals, out_perms, out_inv_perms);
        grad_ym = g_ym.reshape({B, T, C});
        grad_out_d.copy_(g_od);
        grad_out_b.copy_(g_ob);
    }

    // 2. Backward through SiLU Output Modulation + Causal Associative Scan
    int n_threads = asdag::get_physical_cores();

    const float* gym_ptr = grad_ym.data_ptr<float>();
    const float* phi_q_ptr = phi_q.data_ptr<float>();
    const float* phi_k_ptr = phi_k.data_ptr<float>();
    const float* gam_ptr = gamma_all.data_ptr<float>();
    const float* S_chk_ptr = S_checkpoints.data_ptr<float>();
    const float* z_ptr = z_all.data_ptr<float>();
    const float* qkvg_raw_ptr = qkvg_raw.data_ptr<float>();
    const float* v_raw_ptr = qkvg_raw_ptr + (2 * (B * T)) * C;
    const float* g_raw_ptr = qkvg_raw_ptr + (3 * (B * T)) * C;
    const float* q_raw_ptr = qkvg_raw_ptr + (0 * (B * T)) * C;
    const float* k_raw_ptr = qkvg_raw_ptr + (1 * (B * T)) * C;
    const float* q_scale_ptr = q_norm_scale.data_ptr<float>();
    const float* k_scale_ptr = k_norm_scale.data_ptr<float>();
    const float* wd_ptr = w_decay.data_ptr<float>();
    const float* x_ptr = x.data_ptr<float>();

    float* gx_ptr = grad_x.data_ptr<float>();
    float* g_qkvg_raw_ptr = grad_qkvg_raw.data_ptr<float>();
    float* g_q_raw_ptr = g_qkvg_raw_ptr + (0 * (B * T)) * C;
    float* g_k_raw_ptr = g_qkvg_raw_ptr + (1 * (B * T)) * C;
    float* g_v_raw_ptr = g_qkvg_raw_ptr + (2 * (B * T)) * C;
    float* g_g_raw_ptr = g_qkvg_raw_ptr + (3 * (B * T)) * C;

    float* g_qs_ptr = grad_q_scale.data_ptr<float>();
    float* g_ks_ptr = grad_k_scale.data_ptr<float>();
    float* g_wd_ptr = grad_wd.data_ptr<float>();
    float* g_bd_ptr = grad_bd.data_ptr<float>();

    auto d_decay_all = torch::empty({B, H, T}, torch::kFloat32);
    float* d_dec_ptr = d_decay_all.data_ptr<float>();

#pragma omp parallel num_threads(n_threads)
    {
        std::vector<float> dS(D * D, 0.0f);
        std::vector<float> dz(D, 0.0f);
        std::vector<float> num(D);
        std::vector<float> g_tilde(D);
        std::vector<float> g_phi_q(D);
        std::vector<float> g_phi_k(D);

        std::vector<float> t_gqs(D, 0.0f);
        std::vector<float> t_gks(D, 0.0f);
        std::vector<float> t_gwd(H * C, 0.0f);
        std::vector<float> t_gbd(H, 0.0f);
        std::vector<float> S_local((K_chunk + 1) * D * D);

#pragma omp for collapse(2) schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t h = 0; h < H; ++h) {
                std::fill(dS.begin(), dS.end(), 0.0f);
                std::fill(dz.begin(), dz.end(), 0.0f);

                for (int64_t c = num_chunks - 1; c >= 0; --c) {
                    int64_t t_start = c * K_chunk;
                    int64_t t_end = std::min(t_start + K_chunk, T);
                    int64_t chunk_len = t_end - t_start;

                    // 1. Recompute S forward within this chunk (lives 100% in L2 cache!)
                    const float* S_init = S_chk_ptr + ((b * H + h) * num_chunks + c) * (D * D);
                    std::memcpy(S_local.data(), S_init, D * D * sizeof(float));

                    for (int64_t step = 0; step < chunk_len; ++step) {
                        int64_t t = t_start + step;
                        int64_t bht_idx = ((b * H + h) * T + t);
                        const float* phi_kt = phi_k_ptr + bht_idx * D;
                        const float* vt = v_raw_ptr + (b * T + t) * C + h * D;
                        float gam = gam_ptr[bht_idx];

                        const float* S_prev = S_local.data() + step * (D * D);
                        float* S_curr = S_local.data() + (step + 1) * (D * D);

                        bool is_simd_aligned = (D % 8 == 0);
                        if (is_simd_aligned) {
#if defined(ASDAG_SIMD_AVX2)
                            for (int64_t i = 0; i < D; ++i) {
                                __m256 pki_v = _mm256_set1_ps(phi_kt[i]);
                                __m256 gam_v = _mm256_set1_ps(gam);
                                for (int64_t j = 0; j < D; j += 8) {
                                    __m256 sp_v = _mm256_loadu_ps(S_prev + i * D + j);
                                    __m256 vt_v = _mm256_loadu_ps(vt + j);
                                    __m256 sc_v = _mm256_fmadd_ps(gam_v, sp_v, _mm256_mul_ps(pki_v, vt_v));
                                    _mm256_storeu_ps(S_curr + i * D + j, sc_v);
                                }
                            }
#endif
                        } else {
                            for (int64_t i = 0; i < D; ++i) {
                                float pki = phi_kt[i];
                                for (int64_t j = 0; j < D; ++j) {
                                    S_curr[i * D + j] = gam * S_prev[i * D + j] + pki * vt[j];
                                }
                            }
                        }
                    }

                    // 2. Backward from t_end - 1 down to t_start
                    for (int64_t step = chunk_len - 1; step >= 0; --step) {
                        int64_t t = t_start + step;
                        int64_t bht_idx = ((b * H + h) * T + t);
                        int64_t bt_idx = (b * T + t);

                        const float* phi_qt = phi_q_ptr + bht_idx * D;
                        const float* phi_kt = phi_k_ptr + bht_idx * D;
                        const float* vt = v_raw_ptr + bt_idx * C + h * D;
                        const float* gt = g_raw_ptr + bt_idx * C + h * D;
                        const float* gym_t = gym_ptr + bt_idx * C + h * D;
                        float gam_t = gam_ptr[bht_idx];

                        const float* St = S_local.data() + (step + 1) * (D * D);
                        const float* zt = z_ptr + bht_idx * D;
                        const float* S_prev = S_local.data() + step * (D * D);
                        const float* z_prev = (t > 0) ? (z_ptr + ((b * H + h) * T + t - 1) * D) : nullptr;

                        float* gvt = g_v_raw_ptr + bt_idx * C + h * D;
                        float* ggt = g_g_raw_ptr + bt_idx * C + h * D;

                        bool is_simd_aligned = (D % 8 == 0);

                        // Compute den & num
                        float den = 0.0f;
                        if (is_simd_aligned) {
#if defined(ASDAG_SIMD_AVX2)
                            __m256 den_acc = _mm256_setzero_ps();
                            for (int64_t i = 0; i < D; i += 8) {
                                den_acc = _mm256_fmadd_ps(_mm256_loadu_ps(phi_qt + i), _mm256_loadu_ps(zt + i), den_acc);
                            }
                            den = asdag::hsum256(den_acc);
#endif
                        } else {
                            for (int64_t i = 0; i < D; ++i) den += phi_qt[i] * zt[i];
                        }
                        den = std::max(den, 1e-5f);
                        float inv_den = 1.0f / den;

                        if (is_simd_aligned) {
#if defined(ASDAG_SIMD_AVX2)
                            for (int64_t j = 0; j < D; j += 8) {
                                __m256 num_v = _mm256_setzero_ps();
                                for (int64_t i = 0; i < D; ++i) {
                                    __m256 qi_v = _mm256_set1_ps(phi_qt[i]);
                                    __m256 st_v = _mm256_loadu_ps(St + i * D + j);
                                    num_v = _mm256_fmadd_ps(qi_v, st_v, num_v);
                                }
                                _mm256_storeu_ps(num.data() + j, num_v);
                            }
#endif
                        } else {
                            for (int64_t j = 0; j < D; ++j) {
                                float dot = 0.0f;
                                for (int64_t i = 0; i < D; ++i) dot += phi_qt[i] * St[i * D + j];
                                num[j] = dot;
                            }
                        }

                        // SiLU gradient on gate g_raw
                        float gy_dot_y = 0.0f;
                        if (is_simd_aligned) {
#if defined(ASDAG_SIMD_AVX2)
                            const __m256 one_v = _mm256_set1_ps(1.0f);
                            __m256 inv_den_v = _mm256_set1_ps(inv_den);
                            __m256 gy_dot_acc = _mm256_setzero_ps();
                            for (int64_t j = 0; j < D; j += 8) {
                                __m256 gj_v = _mm256_loadu_ps(gt + j);
                                __m256 sig_v = asdag::sigmoid256_ps(gj_v);
                                __m256 silu_v = _mm256_mul_ps(gj_v, sig_v);
                                __m256 dsilu_v = _mm256_mul_ps(sig_v, _mm256_fmadd_ps(gj_v, _mm256_sub_ps(one_v, sig_v), one_v));

                                __m256 num_v = _mm256_loadu_ps(num.data() + j);
                                __m256 y_scan_v = _mm256_mul_ps(num_v, inv_den_v);
                                __m256 d_ym_v = _mm256_loadu_ps(gym_t + j);

                                __m256 ggt_v = _mm256_mul_ps(_mm256_mul_ps(d_ym_v, y_scan_v), dsilu_v);
                                _mm256_storeu_ps(ggt + j, ggt_v);

                                __m256 d_yscan_v = _mm256_mul_ps(d_ym_v, silu_v);
                                __m256 g_tilde_v = _mm256_mul_ps(d_yscan_v, inv_den_v);
                                _mm256_storeu_ps(g_tilde.data() + j, g_tilde_v);

                                gy_dot_acc = _mm256_fmadd_ps(_mm256_mul_ps(d_ym_v, silu_v), y_scan_v, gy_dot_acc);
                            }
                            gy_dot_y = asdag::hsum256(gy_dot_acc);
#endif
                        } else {
                            for (int64_t j = 0; j < D; ++j) {
                                float gj = gt[j];
                                float sig_g = 1.0f / (1.0f + std::exp(-gj));
                                float silu_g = gj * sig_g;
                                float dsilu_g = sig_g * (1.0f + gj * (1.0f - sig_g));

                                float y_scan_val = num[j] * inv_den;
                                float d_ym = gym_t[j];
                                ggt[j] = d_ym * y_scan_val * dsilu_g;

                                float d_yscan = d_ym * silu_g;
                                g_tilde[j] = d_yscan * inv_den;
                                gy_dot_y += (d_ym * silu_g) * y_scan_val;
                            }
                        }
                        float d_den = -gy_dot_y * inv_den;

                        // dS & dz update
                        if (is_simd_aligned) {
#if defined(ASDAG_SIMD_AVX2)
                            for (int64_t i = 0; i < D; ++i) {
                                __m256 qi_v = _mm256_set1_ps(phi_qt[i]);
                                for (int64_t j = 0; j < D; j += 8) {
                                    __m256 ds_v = _mm256_loadu_ps(dS.data() + i * D + j);
                                    __m256 gt_v = _mm256_loadu_ps(g_tilde.data() + j);
                                    _mm256_storeu_ps(dS.data() + i * D + j, _mm256_fmadd_ps(qi_v, gt_v, ds_v));
                                }
                                dz[i] += d_den * phi_qt[i];
                            }
#endif
                        } else {
                            for (int64_t i = 0; i < D; ++i) {
                                float qi = phi_qt[i];
                                for (int64_t j = 0; j < D; ++j) {
                                    dS[i * D + j] += qi * g_tilde[j];
                                }
                                dz[i] += d_den * qi;
                            }
                        }

                        // g_phi_q, g_phi_k, g_v_raw
                        if (is_simd_aligned) {
#if defined(ASDAG_SIMD_AVX2)
                            for (int64_t i = 0; i < D; ++i) {
                                __m256 acc_q = _mm256_setzero_ps();
                                __m256 acc_k = _mm256_setzero_ps();
                                for (int64_t j = 0; j < D; j += 8) {
                                    acc_q = _mm256_fmadd_ps(_mm256_loadu_ps(St + i * D + j), _mm256_loadu_ps(g_tilde.data() + j), acc_q);
                                    acc_k = _mm256_fmadd_ps(_mm256_loadu_ps(dS.data() + i * D + j), _mm256_loadu_ps(vt + j), acc_k);
                                }
                                g_phi_q[i] = asdag::hsum256(acc_q) + d_den * zt[i];
                                g_phi_k[i] = asdag::hsum256(acc_k) + dz[i];
                            }

                            for (int64_t j = 0; j < D; j += 8) {
                                __m256 acc_v = _mm256_setzero_ps();
                                for (int64_t i = 0; i < D; ++i) {
                                    __m256 pki_v = _mm256_set1_ps(phi_kt[i]);
                                    __m256 ds_v = _mm256_loadu_ps(dS.data() + i * D + j);
                                    acc_v = _mm256_fmadd_ps(pki_v, ds_v, acc_v);
                                }
                                _mm256_storeu_ps(gvt + j, acc_v);
                            }
#endif
                        } else {
                            for (int64_t i = 0; i < D; ++i) {
                                float acc_q = 0.0f;
                                float acc_k = 0.0f;
                                for (int64_t j = 0; j < D; ++j) {
                                    acc_q += St[i * D + j] * g_tilde[j];
                                    acc_k += dS[i * D + j] * vt[j];
                                }
                                g_phi_q[i] = acc_q + d_den * zt[i];
                                g_phi_k[i] = acc_k + dz[i];
                            }

                            for (int64_t j = 0; j < D; ++j) {
                                float acc = 0.0f;
                                for (int64_t i = 0; i < D; ++i) acc += phi_kt[i] * dS[i * D + j];
                                gvt[j] = acc;
                            }
                        }

                        // d_gamma
                        float dgam = 0.0f;
                        if (t > 0 && z_prev) {
                            if (is_simd_aligned) {
#if defined(ASDAG_SIMD_AVX2)
                                __m256 dgam_acc = _mm256_setzero_ps();
                                for (int64_t idx = 0; idx < D * D; idx += 8) {
                                    dgam_acc = _mm256_fmadd_ps(_mm256_loadu_ps(dS.data() + idx), _mm256_loadu_ps(S_prev + idx), dgam_acc);
                                }
                                dgam = asdag::hsum256(dgam_acc);
#endif
                            } else {
                                for (int64_t idx = 0; idx < D * D; ++idx) dgam += dS[idx] * S_prev[idx];
                            }
                            for (int64_t i = 0; i < D; ++i) dgam += dz[i] * z_prev[i];
                        }

                        // Decay linear layer gradients & record d_dot_decay
                        float d_dot_decay = dgam * gam_t * (1.0f - gam_t);
                        if (rm_ptr && rm_ptr[bt_idx] != 0) d_dot_decay = 0.0f;
                        d_dec_ptr[bht_idx] = d_dot_decay;

                        // Propagate dS & dz to previous step
                        if (is_simd_aligned) {
#if defined(ASDAG_SIMD_AVX2)
                            __m256 gam_tv = _mm256_set1_ps(gam_t);
                            for (int64_t idx = 0; idx < D * D; idx += 8) {
                                __m256 ds_v = _mm256_loadu_ps(dS.data() + idx);
                                _mm256_storeu_ps(dS.data() + idx, _mm256_mul_ps(ds_v, gam_tv));
                            }
                            for (int64_t i = 0; i < D; i += 8) {
                                __m256 dz_v = _mm256_loadu_ps(dz.data() + i);
                                _mm256_storeu_ps(dz.data() + i, _mm256_mul_ps(dz_v, gam_tv));
                            }
#endif
                        } else {
                            for (int64_t idx = 0; idx < D * D; ++idx) dS[idx] *= gam_t;
                            for (int64_t i = 0; i < D; ++i) dz[i] *= gam_t;
                        }

                        // Backward through Q/K RMSNorm & ELU
                        const float* q_raw_bt = q_raw_ptr + bt_idx * C + h * D;
                        const float* k_raw_bt = k_raw_ptr + bt_idx * C + h * D;
                        float* g_q_raw_bt = g_q_raw_ptr + bt_idx * C + h * D;
                        float* g_k_raw_bt = g_k_raw_ptr + bt_idx * C + h * D;

                        float sum_sq_q = 0.0f;
                        float sum_sq_k = 0.0f;
                        for (int64_t d = 0; d < D; ++d) {
                            sum_sq_q += q_raw_bt[d] * q_raw_bt[d];
                            sum_sq_k += k_raw_bt[d] * k_raw_bt[d];
                        }
                        float rms_q = 1.0f / std::sqrt((sum_sq_q / (float)D) + 1e-5f);
                        float rms_k = 1.0f / std::sqrt((sum_sq_k / (float)D) + 1e-5f);

                        float sum_gq_q = 0.0f;
                        float sum_gk_k = 0.0f;
                        for (int64_t d = 0; d < D; ++d) {
                            float q_norm = q_raw_bt[d] * rms_q * q_scale_ptr[d];
                            float k_norm = k_raw_bt[d] * rms_k * k_scale_ptr[d];

                            float delu_q = q_norm >= 0.0f ? 1.0f : std::exp(q_norm);
                            float delu_k = k_norm >= 0.0f ? 1.0f : std::exp(k_norm);

                            float dq_norm = g_phi_q[d] * delu_q;
                            float dk_norm = g_phi_k[d] * delu_k;

                            t_gqs[d] += dq_norm * (q_raw_bt[d] * rms_q);
                            t_gks[d] += dk_norm * (k_raw_bt[d] * rms_k);

                            float dq_scaled = dq_norm * q_scale_ptr[d];
                            float dk_scaled = dk_norm * k_scale_ptr[d];

                            sum_gq_q += dq_scaled * q_raw_bt[d];
                            sum_gk_k += dk_scaled * k_raw_bt[d];

                            g_q_raw_bt[d] = dq_scaled;
                            g_k_raw_bt[d] = dk_scaled;
                        }

                        for (int64_t d = 0; d < D; ++d) {
                            g_q_raw_bt[d] = (g_q_raw_bt[d] * rms_q) - (q_raw_bt[d] * sum_gq_q * (rms_q * rms_q * rms_q / (float)D));
                            g_k_raw_bt[d] = (g_k_raw_bt[d] * rms_k) - (k_raw_bt[d] * sum_gk_k * (rms_k * rms_k * rms_k / (float)D));
                        }
                    }
                }
            }
        }

#pragma omp critical
        {
            for (int64_t d = 0; d < D; ++d) {
                g_qs_ptr[d] += t_gqs[d];
                g_ks_ptr[d] += t_gks[d];
            }
            for (size_t i = 0; i < t_gwd.size(); ++i) g_wd_ptr[i] += t_gwd[i];
            for (int64_t h = 0; h < H; ++h) g_bd_ptr[h] += t_gbd[h];
        }
    }

    // Lock-free parallel accumulation into gx_ptr from d_decay_all
#pragma omp parallel for collapse(2) num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            float* gx_bt = gx_ptr + (b * T + t) * C;
            for (int64_t h = 0; h < H; ++h) {
                float d_decay = d_dec_ptr[((b * H + h) * T + t)];
                const float* wd_h = wd_ptr + h * C;
                for (int64_t c = 0; c < C; ++c) {
                    gx_bt[c] += d_decay * wd_h[c];
                }
            }
        }
    }

    // 3. Backward through QKVG Fused Monarch Chain
    {
        auto gqkvg_flat = grad_qkvg_raw.reshape({4, B * T, C});
        auto x_flat = x.reshape({B * T, C});
        auto [gx_m, g_qkvg_d, g_qkvg_b] = asdag_fused_monarch_chain_backward_cpp(gqkvg_flat, x_flat, qkvg_diagonals, qkvg_perms, qkvg_inv_perms);
        grad_x.add_(gx_m.reshape({B, T, C}));
        grad_qkvg_d.copy_(g_qkvg_d);
        grad_qkvg_b.copy_(g_qkvg_b);
    }

    return std::make_tuple(
        grad_x.to(orig_dtype),
        grad_qkvg_d,
        grad_qkvg_b,
        grad_q_scale,
        grad_k_scale,
        grad_wd,
        grad_bd,
        grad_out_d,
        grad_out_b
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 32. C++ Fused Full ASDAG Block Forward (RMSNorm1 + Monarch GLA + RMSNorm2 + Ternary SwiGLU)
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> asdag_fused_asdag_block_forward_cpp(
    torch::Tensor x,              // [B, T, C]
    torch::Tensor norm1_scale,    // [C]
    torch::Tensor qkvg_diagonals, // [4, S, C]
    torch::Tensor qkvg_perms,     // [S - 1, C]
    torch::Tensor qkvg_bias,      // [4, C]
    torch::Tensor q_norm_scale,   // [D]
    torch::Tensor k_norm_scale,   // [D]
    torch::Tensor w_decay,        // [H, C]
    torch::Tensor b_decay,        // [H]
    torch::Tensor out_diagonals,  // [S, C]
    torch::Tensor out_perms,      // [S - 1, C]
    torch::Tensor out_bias,       // [C]
    torch::Tensor norm2_scale,    // [C]
    torch::Tensor w_gate_val,     // [2*H_dim, C] (ternary)
    float gamma_gv,
    torch::Tensor w_down,         // [C, H_dim] (ternary)
    float gamma_down,
    torch::Tensor reset_mask      // [B, T] (optional)
) {
    auto orig_dtype = x.scalar_type();
    x = x.contiguous().to(torch::kFloat32);
    norm1_scale = norm1_scale.contiguous().to(torch::kFloat32);
    norm2_scale = norm2_scale.contiguous().to(torch::kFloat32);

    int64_t B = x.size(0);
    int64_t T = x.size(1);
    int64_t C = x.size(2);
    int64_t H_dim = w_down.size(1);

    auto x_norm1 = torch::empty({B, T, C}, torch::kFloat32);
    auto x1 = torch::empty({B, T, C}, torch::kFloat32);
    auto x_norm2 = torch::empty({B, T, C}, torch::kFloat32);
    auto out = torch::empty({B, T, C}, torch::kFloat32);

    const float* x_ptr = x.data_ptr<float>();
    const float* n1_scale_ptr = norm1_scale.data_ptr<float>();
    const float* n2_scale_ptr = norm2_scale.data_ptr<float>();
    float* xn1_ptr = x_norm1.data_ptr<float>();
    float* x1_ptr = x1.data_ptr<float>();
    float* xn2_ptr = x_norm2.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

    // 1. In-situ RMSNorm1
#pragma omp parallel for collapse(2) num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            const float* xb = x_ptr + (b * T + t) * C;
            float* xn1 = xn1_ptr + (b * T + t) * C;

            float sum_sq = 0.0f;
            for (int64_t c = 0; c < C; ++c) sum_sq += xb[c] * xb[c];
            float rms = 1.0f / std::sqrt((sum_sq / (float)C) + 1e-6f);

            for (int64_t c = 0; c < C; ++c) xn1[c] = xb[c] * rms * n1_scale_ptr[c];
        }
    }

    // 2. Fused Monarch GLA Forward
    auto [y_mixer, qkvg_raw, phi_q, phi_k, gamma_all, S_all, z_all, y_mod] = asdag_fused_monarch_gla_forward_cpp(
        x_norm1, qkvg_diagonals, qkvg_perms, qkvg_bias,
        q_norm_scale, k_norm_scale, w_decay, b_decay,
        out_diagonals, out_perms, out_bias, reset_mask
    );

    const float* ym_ptr = y_mixer.data_ptr<float>();

    // 3. In-situ Residual 1 + RMSNorm2
#pragma omp parallel for collapse(2) num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            const float* xb = x_ptr + (b * T + t) * C;
            const float* ymb = ym_ptr + (b * T + t) * C;
            float* x1b = x1_ptr + (b * T + t) * C;
            float* xn2 = xn2_ptr + (b * T + t) * C;

            float sum_sq = 0.0f;
            for (int64_t c = 0; c < C; ++c) {
                float v = xb[c] + ymb[c];
                x1b[c] = v;
                sum_sq += v * v;
            }
            float rms = 1.0f / std::sqrt((sum_sq / (float)C) + 1e-6f);
            for (int64_t c = 0; c < C; ++c) xn2[c] = x1b[c] * rms * n2_scale_ptr[c];
        }
    }

    // 4. Fused Ternary BitLinear SwiGLU Forward (AVX2 Int8 Ternary Engine)
    auto xn2_flat = x_norm2.reshape({B * T, C});
    auto [y_channel_flat, h_act_all, gate_raw_all, val_raw_all] = asdag_bitlinear_swiglu_ternary_int8_forward_cpp(
        xn2_flat, w_gate_val, gamma_gv, w_down, gamma_down
    );
    const float* yc_ptr = y_channel_flat.data_ptr<float>();

    // 5. In-situ Residual 2
#pragma omp parallel for collapse(2) num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            const float* x1b = x1_ptr + (b * T + t) * C;
            const float* ycb = yc_ptr + (b * T + t) * C;
            float* outb = out_ptr + (b * T + t) * C;

            for (int64_t c = 0; c < C; ++c) {
                outb[c] = x1b[c] + ycb[c];
            }
        }
    }

    return std::make_tuple(
        out.to(orig_dtype),
        x_norm1,
        x1,
        x_norm2,
        qkvg_raw,
        phi_q,
        phi_k,
        gamma_all,
        S_all,
        z_all,
        y_mod,
        h_act_all
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 33. C++ Fused Full ASDAG Block Backward (AVX2 / AVX-512 SIMD)
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> asdag_fused_asdag_block_backward_cpp(
    torch::Tensor grad_y,         // [B, T, C]
    torch::Tensor x,              // [B, T, C]
    torch::Tensor norm1_scale,    // [C]
    torch::Tensor x_norm1,        // [B, T, C]
    torch::Tensor x1,             // [B, T, C]
    torch::Tensor norm2_scale,    // [C]
    torch::Tensor x_norm2,        // [B, T, C]
    torch::Tensor qkvg_diagonals, // [4, S, C]
    torch::Tensor qkvg_perms,     // [S - 1, C]
    torch::Tensor qkvg_inv_perms, // [S - 1, C]
    torch::Tensor qkvg_bias,      // [4, C]
    torch::Tensor qkvg_raw,       // [4, B, T, C]
    torch::Tensor phi_q,          // [B, H, T, D]
    torch::Tensor phi_k,          // [B, H, T, D]
    torch::Tensor gamma_all,      // [B, H, T]
    torch::Tensor S_all,          // [B, H, T, D, D]
    torch::Tensor z_all,          // [B, H, T, D]
    torch::Tensor y_mod,          // [B, T, C]
    torch::Tensor q_norm_scale,   // [D]
    torch::Tensor k_norm_scale,   // [D]
    torch::Tensor w_decay,        // [H, C]
    torch::Tensor b_decay,        // [H]
    torch::Tensor out_diagonals,  // [S, C]
    torch::Tensor out_perms,      // [S - 1, C]
    torch::Tensor out_inv_perms,  // [S - 1, C]
    torch::Tensor out_bias,       // [C]
    torch::Tensor w_gate_val,     // [2*H_dim, C]
    float gamma_gv,
    torch::Tensor w_down,         // [C, H_dim]
    float gamma_down,
    torch::Tensor reset_mask      // [B, T]
) {
    auto orig_dtype = grad_y.scalar_type();
    grad_y = grad_y.contiguous().to(torch::kFloat32);
    x = x.contiguous().to(torch::kFloat32);
    norm1_scale = norm1_scale.contiguous().to(torch::kFloat32);
    x_norm1 = x_norm1.contiguous().to(torch::kFloat32);
    x1 = x1.contiguous().to(torch::kFloat32);
    norm2_scale = norm2_scale.contiguous().to(torch::kFloat32);
    x_norm2 = x_norm2.contiguous().to(torch::kFloat32);

    int64_t B = x.size(0);
    int64_t T = x.size(1);
    int64_t C = x.size(2);

    int n_threads = asdag::get_physical_cores();

    // 1. Backward through Ternary BitLinear SwiGLU: grad_output = grad_y
    auto gy_flat = grad_y.reshape({B * T, C});
    auto xn2_flat = x_norm2.reshape({B * T, C});
    auto [g_xn2_flat, g_wgv, g_wd] = asdag_bitlinear_swiglu_backward_recompute_cpp(
        gy_flat, xn2_flat, w_gate_val, gamma_gv, w_down, gamma_down
    );
    auto grad_xn2 = g_xn2_flat.reshape({B, T, C});

    // 2. Backward through RMSNorm2 & Accumulate into grad_x1 (grad_x1 = grad_y + dRMSNorm2)
    auto grad_x1 = grad_y.clone();
    auto grad_norm2_scale = torch::zeros({C}, torch::kFloat32);

    const float* gx2_ptr = grad_xn2.data_ptr<float>();
    const float* x1_ptr = x1.data_ptr<float>();
    const float* n2_scale_ptr = norm2_scale.data_ptr<float>();
    float* gx1_ptr = grad_x1.data_ptr<float>();
    float* gn2_ptr = grad_norm2_scale.data_ptr<float>();

#pragma omp parallel num_threads(n_threads)
    {
        std::vector<float> t_gn2(C, 0.0f);

#pragma omp for collapse(2) schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t t = 0; t < T; ++t) {
                const float* x1b = x1_ptr + (b * T + t) * C;
                const float* gx2b = gx2_ptr + (b * T + t) * C;
                float* gx1b = gx1_ptr + (b * T + t) * C;

                float sum_sq = 0.0f;
                for (int64_t c = 0; c < C; ++c) sum_sq += x1b[c] * x1b[c];
                float rms = 1.0f / std::sqrt((sum_sq / (float)C) + 1e-6f);

                float sum_gy_y = 0.0f;
                for (int64_t c = 0; c < C; ++c) {
                    float g_pre = gx2b[c] * n2_scale_ptr[c];
                    float y_norm = x1b[c] * rms;
                    sum_gy_y += g_pre * y_norm;
                    t_gn2[c] += gx2b[c] * y_norm;
                }

                for (int64_t c = 0; c < C; ++c) {
                    float g_pre = gx2b[c] * n2_scale_ptr[c];
                    float y_norm = x1b[c] * rms;
                    gx1b[c] += rms * (g_pre - y_norm * (sum_gy_y / (float)C));
                }
            }
        }

#pragma omp critical
        {
            for (int64_t c = 0; c < C; ++c) gn2_ptr[c] += t_gn2[c];
        }
    }

    // 3. Backward through Fused Monarch GLA (grad_y_mixer = grad_x1)
    auto [g_xn1, g_qkvg_d, g_qkvg_b, g_qs, g_ks, g_wd_gla, g_bd, g_od, g_ob] = asdag_fused_monarch_gla_backward_cpp(
        grad_x1, x_norm1, qkvg_diagonals, qkvg_perms, qkvg_inv_perms, qkvg_raw,
        phi_q, phi_k, gamma_all, S_all, z_all, y_mod,
        q_norm_scale, k_norm_scale, w_decay, b_decay,
        out_diagonals, out_perms, out_inv_perms, reset_mask
    );

    // 4. Backward through RMSNorm1 & Accumulate into grad_x (grad_x = grad_x1 + dRMSNorm1)
    auto grad_x = grad_x1.clone();
    auto grad_norm1_scale = torch::zeros({C}, torch::kFloat32);

    const float* gx1_norm_ptr = g_xn1.data_ptr<float>();
    const float* x_ptr = x.data_ptr<float>();
    const float* n1_scale_ptr = norm1_scale.data_ptr<float>();
    float* gx_ptr = grad_x.data_ptr<float>();
    float* gn1_ptr = grad_norm1_scale.data_ptr<float>();

#pragma omp parallel num_threads(n_threads)
    {
        std::vector<float> t_gn1(C, 0.0f);

#pragma omp for collapse(2) schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t t = 0; t < T; ++t) {
                const float* xb = x_ptr + (b * T + t) * C;
                const float* gx1nb = gx1_norm_ptr + (b * T + t) * C;
                float* gxb = gx_ptr + (b * T + t) * C;

                float sum_sq = 0.0f;
                for (int64_t c = 0; c < C; ++c) sum_sq += xb[c] * xb[c];
                float rms = 1.0f / std::sqrt((sum_sq / (float)C) + 1e-6f);

                float sum_gy_y = 0.0f;
                for (int64_t c = 0; c < C; ++c) {
                    float g_pre = gx1nb[c] * n1_scale_ptr[c];
                    float y_norm = xb[c] * rms;
                    sum_gy_y += g_pre * y_norm;
                    t_gn1[c] += gx1nb[c] * y_norm;
                }

                for (int64_t c = 0; c < C; ++c) {
                    float g_pre = gx1nb[c] * n1_scale_ptr[c];
                    float y_norm = xb[c] * rms;
                    gxb[c] += rms * (g_pre - y_norm * (sum_gy_y / (float)C));
                }
            }
        }

#pragma omp critical
        {
            for (int64_t c = 0; c < C; ++c) gn1_ptr[c] += t_gn1[c];
        }
    }

    return std::make_tuple(
        grad_x.to(orig_dtype),
        grad_norm1_scale,
        g_qkvg_d,
        g_qkvg_b,
        g_qs,
        g_ks,
        g_wd_gla,
        g_bd,
        g_od,
        g_ob,
        grad_norm2_scale,
        g_wgv,
        g_wd
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 27. Fused ASDAG Tree Block (RMSNorm1 + Monarch GLA + RMSNorm2 + Sparse Tree) - Hierarchical
// Helpers mirror ast_dag.py: _Log4ShiftSTE, _FP8HybridSTE (E4M3 fwd / E5M2 bwd
// grad), ternary_ste, HierarchicalSignRouter.route_tokens, ASTDAGNode perm
// root. All elementwise ops are float32; rounding uses ties-to-even.
// ─────────────────────────────────────────────────────────────────────────────
namespace asdag_fused_tree {
inline float shift4_fwd(float v, float scale) {
    float nx = v / scale;
    float s = (nx > 0.f) ? 1.f : ((nx < 0.f) ? -1.f : 0.f);
    float ax = std::fabs(nx);
    float axc = ax < 0.000030517578125f ? 0.000030517578125f : ax;
    float p = rintf(-log2f(axc));
    if (p < 0.f) p = 0.f;
    if (p > 7.f) p = 7.f;
    float q = (ax < 0.000043213918264f) ? 0.f : s * exp2f(-p);
    return q * scale;
}
inline bool shift4_pass(float v, float scale) {
    return std::fabs(v / scale) <= 1.5f;
}
inline float e4m3_round(float v) {
    if (!std::isfinite(v)) return v;
    uint32_t u;
    std::memcpy(&u, &v, 4);
    if ((u & 0x7FFFFFFFu) == 0) return v;
    float sgn = (u & 0x80000000u) ? -1.f : 1.f;
    float af = std::fabs(v);
    int32_t e2 = (int32_t)ilogbf(af);
    int32_t ef = e2 + 7;
    if (ef > 15) return std::numeric_limits<float>::quiet_NaN();
    if (ef == 15) {
        uint32_t m;
        std::memcpy(&m, &af, 4);
        m &= 0x7FFFFFu;
        uint32_t hi = m >> 20, lo = m & 0xFFFFFu;
        uint32_t q = hi + ((lo > 0x80000u) || (lo == 0x80000u && (hi & 1u)));
        if (q >= 8u) return std::numeric_limits<float>::quiet_NaN();
        if (q == 7u) return std::numeric_limits<float>::quiet_NaN();
        uint32_t bits = ((uint32_t)(15 + 120) << 23) | (q << 20);
        float o;
        std::memcpy(&o, &bits, 4);
        return sgn * o;
    }
    if (ef >= 1) {
        uint32_t m;
        std::memcpy(&m, &af, 4);
        m &= 0x7FFFFFu;
        uint32_t hi = m >> 20, lo = m & 0xFFFFFu;
        uint32_t q = hi + ((lo > 0x80000u) || (lo == 0x80000u && (hi & 1u)));
        if (q >= 8u) {
            q = 0;
            ef += 1;
            if (ef > 15) return std::numeric_limits<float>::quiet_NaN();
        }
        uint32_t bits = ((uint32_t)(ef + 120) << 23) | (q << 20);
        float o;
        std::memcpy(&o, &bits, 4);
        return sgn * o;
    }
    float qs = rintf(af * 512.f);
    if (qs >= 8.f) return sgn * 0.015625f;
    return sgn * qs * 0.001953125f;
}
inline float e5m2_round(float v) {
    if (!std::isfinite(v)) return v;
    uint32_t u;
    std::memcpy(&u, &v, 4);
    if ((u & 0x7FFFFFFFu) == 0) return v;
    float sgn = (u & 0x80000000u) ? -1.f : 1.f;
    float af = std::fabs(v);
    int32_t e2 = (int32_t)ilogbf(af);
    int32_t ef = e2 + 16;
    if (ef >= 31) return sgn * std::numeric_limits<float>::infinity();
    if (ef >= 1) {
        uint32_t m;
        std::memcpy(&m, &af, 4);
        m &= 0x7FFFFFu;
        uint32_t hi = m >> 21, lo = m & 0x1FFFFFu;
        uint32_t q = hi + ((lo > 0x100000u) || (lo == 0x100000u && (hi & 1u)));
        if (q >= 4u) { q = 0; ef += 1; if (ef >= 31) return sgn * std::numeric_limits<float>::infinity(); }
        uint32_t bits = ((uint32_t)(ef + 112) << 23) | (q << 21);
        float o;
        std::memcpy(&o, &bits, 4);
        return sgn * o;
    }
    float qs = rintf(af * 65536.f);
    if (qs >= 4.f) return sgn * 0.00006103515625f;
    return sgn * qs * 0.0000152587890625f;
}
inline float ternary_delta(const float* d, int64_t n, float frac) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += std::fabs((double)d[i]);
    return (float)(frac * s / (double)n);
}
inline float ternary_alpha_active(const float* d, const float* wq, int64_t n) {
    double s = 0.0;
    int64_t c = 0;
    for (int64_t i = 0; i < n; ++i) if (wq[i] != 0.f) { s += std::fabs((double)d[i]); ++c; }
    if (c == 0) return 0.f;
    return (float)(s / (double)c);
}
inline float sigmoid2(float z) {
    return 1.f / (1.f + expf(-2.f * z));
}
}  // namespace asdag_fused_tree
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
asdag_fused_asdag_tree_block_forward_cpp(
    torch::Tensor x,
    torch::Tensor norm1_scale,
    torch::Tensor qkvg_diagonals,
    torch::Tensor qkvg_perms,
    torch::Tensor qkvg_inv_perms,
    torch::Tensor qkvg_bias,
    torch::Tensor q_norm_scale,
    torch::Tensor k_norm_scale,
    torch::Tensor w_decay,
    torch::Tensor b_decay,
    torch::Tensor out_diagonals,
    torch::Tensor out_perms,
    torch::Tensor out_inv_perms,
    torch::Tensor out_bias,
    torch::Tensor norm2_scale,
    torch::Tensor w_perm,
    torch::Tensor perms,
    torch::Tensor inv_perms,
    torch::Tensor bias,
    torch::Tensor root_latent_w,
    torch::Tensor root_scale,
    torch::Tensor root_bias,
    torch::Tensor root_perms,
    torch::Tensor hyperplanes,
    torch::Tensor router_biases,
    torch::Tensor reset_mask
) {
    auto orig_dtype = x.scalar_type();
    x = x.contiguous().to(torch::kFloat32);
    norm1_scale = norm1_scale.contiguous().to(torch::kFloat32);
    norm2_scale = norm2_scale.contiguous().to(torch::kFloat32);
    int64_t B = x.size(0);
    int64_t T = x.size(1);
    int64_t C = x.size(2);
    auto x_norm1 = torch::empty({B, T, C}, torch::kFloat32);
    auto x1 = torch::empty({B, T, C}, torch::kFloat32);
    auto x_norm2 = torch::empty({B, T, C}, torch::kFloat32);
    auto out = torch::empty({B, T, C}, torch::kFloat32);
    const float* x_ptr = x.data_ptr<float>();
    const float* n1_ptr = norm1_scale.data_ptr<float>();
    const float* n2_ptr = norm2_scale.data_ptr<float>();
    float* xn1_ptr = x_norm1.data_ptr<float>();
    float* x1_ptr = x1.data_ptr<float>();
    float* xn2_ptr = x_norm2.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    int n_threads = asdag::get_physical_cores();
#pragma omp parallel for collapse(2) num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            const float* xb = x_ptr + (b * T + t) * C;
            float* xn1 = xn1_ptr + (b * T + t) * C;
            float sum_sq = 0.0f;
            for (int64_t c = 0; c < C; ++c) sum_sq += xb[c] * xb[c];
            float rms = 1.0f / std::sqrt((sum_sq / (float)C) + 1e-6f);
            for (int64_t c = 0; c < C; ++c) xn1[c] = xb[c] * rms * n1_ptr[c];
        }
    }
    auto [y_mixer, qkvg_raw, phi_q, phi_k, gamma_all, S_all, z_all, y_mod] = asdag_fused_monarch_gla_forward_cpp(
        x_norm1, qkvg_diagonals, qkvg_perms, qkvg_bias,
        q_norm_scale, k_norm_scale, w_decay, b_decay,
        out_diagonals, out_perms, out_bias, reset_mask
    );
    const float* ym_ptr = y_mixer.data_ptr<float>();
#pragma omp parallel for collapse(2) num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            const float* xb = x_ptr + (b * T + t) * C;
            const float* ymb = ym_ptr + (b * T + t) * C;
            float* x1b = x1_ptr + (b * T + t) * C;
            float* xn2 = xn2_ptr + (b * T + t) * C;
            float sum_sq = 0.0f;
            for (int64_t c = 0; c < C; ++c) { float v = xb[c] + ymb[c]; x1b[c] = v; sum_sq += v * v; }
            float rms = 1.0f / std::sqrt((sum_sq / (float)C) + 1e-6f);
            for (int64_t c = 0; c < C; ++c) xn2[c] = x1b[c] * rms * n2_ptr[c];
        }
    }
    using namespace asdag_fused_tree;
    auto x_norm2_c = x_norm2.contiguous().to(torch::kFloat32);
    root_latent_w = root_latent_w.contiguous().to(torch::kFloat32);
    root_scale = root_scale.contiguous().to(torch::kFloat32);
    root_bias = root_bias.contiguous().to(torch::kFloat32);
    root_perms = root_perms.contiguous().to(torch::kInt32);
    hyperplanes = hyperplanes.contiguous().to(torch::kFloat32);
    router_biases = router_biases.contiguous().to(torch::kFloat32);
    int64_t N = B * T;
    int64_t I = hyperplanes.size(0);
    int64_t K = w_perm.size(0);
    int64_t RP = root_latent_w.size(0);
    int depth = 1;
    while ((((int64_t)1) << depth) - 1 < I) ++depth;
    const float* hyp_ptr = hyperplanes.data_ptr<float>();
    const float* rb_ptr = router_biases.data_ptr<float>();
    const float* rlat_ptr = root_latent_w.data_ptr<float>();
    const float* rsc_ptr = root_scale.data_ptr<float>();
    const float* rbi_ptr = root_bias.data_ptr<float>();
    const int32_t* rpm_ptr = root_perms.data_ptr<int32_t>();
    std::vector<float> w_route(I * C);
    {
        float delta = ternary_delta(hyp_ptr, I * C, 0.7f);
        std::vector<float> sgn(I * C);
        for (int64_t i = 0; i < I * C; ++i) {
            float v = hyp_ptr[i];
            sgn[i] = (v > delta) ? 1.f : ((v < -delta) ? -1.f : 0.f);
        }
        float alpha = ternary_alpha_active(hyp_ptr, sgn.data(), I * C);
        for (int64_t i = 0; i < I * C; ++i) w_route[i] = sgn[i] * alpha;
    }
    std::vector<float> w_root(RP * C);
    std::vector<float> w_root_sgn(RP * C);
    {
        float delta = ternary_delta(rlat_ptr, RP * C, 0.7f);
        for (int64_t i = 0; i < RP * C; ++i) {
            float v = rlat_ptr[i];
            float s = (v > delta) ? 1.f : ((v < -delta) ? -1.f : 0.f);
            w_root_sgn[i] = s;
            w_root[i] = s * rsc_ptr[(i / C) % RP];
        }
    }
    auto r_in_flat = torch::empty({N, C}, torch::kFloat32);
    auto xq_save = torch::empty({N, C}, torch::kFloat32);
    auto root_out_save = torch::empty({N, C}, torch::kFloat32);
    auto root_preact = torch::empty({N, C}, torch::kFloat32);
    auto node_p = torch::empty({N, I}, torch::kFloat32);
    auto top_idx = torch::empty({N, 2}, torch::kInt32);
    auto top_w = torch::empty({N, 2}, torch::kFloat32);
    auto top_vals = torch::empty({N, 2}, torch::kFloat32);
    auto sc0 = torch::empty({N}, torch::kFloat32);
    auto scf = torch::empty({N}, torch::kFloat32);
    auto sc1 = torch::empty({N}, torch::kFloat32);
    const float* xn2c_ptr = x_norm2_c.data_ptr<float>();
    float* rin_ptr = r_in_flat.data_ptr<float>();
    float* xq_ptr = xq_save.data_ptr<float>();
    float* ro_ptr = root_out_save.data_ptr<float>();
    float* rp_ptr = root_preact.data_ptr<float>();
    float* np_ptr = node_p.data_ptr<float>();
    int32_t* ti_ptr = top_idx.data_ptr<int32_t>();
    float* tw_ptr = top_w.data_ptr<float>();
    float* tv_ptr = top_vals.data_ptr<float>();
    float* s0_ptr = sc0.data_ptr<float>();
    float* sf_ptr = scf.data_ptr<float>();
    float* s1_ptr = sc1.data_ptr<float>();
#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        const float* xn = xn2c_ptr + n * C;
        float* xq = xq_ptr + n * C;
        float* ro = ro_ptr + n * C;
        float* pre = rp_ptr + n * C;
        float* rin = rin_ptr + n * C;
        float am = 0.f;
        for (int64_t c = 0; c < C; ++c) { float a = std::fabs(xn[c]); if (a > am) am = a; }
        float scale0 = am < 1e-8f ? 1e-8f : am;
        s0_ptr[n] = scale0;
        for (int64_t c = 0; c < C; ++c) xq[c] = shift4_fwd(xn[c], scale0);
        float am1 = 0.f;
        for (int64_t c = 0; c < C; ++c) { float a = std::fabs(xq[c]); if (a > am1) am1 = a; }
        float fs = 240.f / (am1 < 1e-6f ? 1e-6f : am1);
        sf_ptr[n] = fs;
        for (int64_t c = 0; c < C; ++c) xq[c] = e4m3_round(xq[c] * fs) / fs;
        for (int64_t c = 0; c < C; ++c) {
            float h = rbi_ptr[c];
            for (int64_t p = 0; p < RP; ++p) h += w_root[p * C + c] * xq[rpm_ptr[p * C + c]];
            pre[c] = h;
            ro[c] = h < 0.f ? 0.f : (h > 6.f ? 6.f : h);
        }
        float amx = 0.f;
        for (int64_t c = 0; c < C; ++c) { float a = std::fabs(ro[c]); if (a > amx) amx = a; }
        float scale1 = amx < 1e-8f ? 1e-8f : amx;
        s1_ptr[n] = scale1;
        for (int64_t c = 0; c < C; ++c) rin[c] = shift4_fwd(ro[c], scale1);
        float logits[32];
        for (int64_t i = 0; i < I; ++i) {
            float s = rb_ptr[i];
            const float* w = &w_route[i * C];
            for (int64_t c = 0; c < C; ++c) s += w[c] * xn[c];
            logits[i] = s;
        }
        float* ndp = np_ptr + n * I;
        float cur[32], nxt[32];
        float pr0 = sigmoid2(logits[0]);
        ndp[0] = pr0;
        cur[0] = 1.f - pr0;
        cur[1] = pr0;
        int64_t cur_n = 2;
        for (int d = 1; d < depth; ++d) {
            int64_t start = (((int64_t)1) << d) - 1;
            int64_t idx = 0;
            for (int64_t j = 0; j < cur_n; ++j) {
                float sr = sigmoid2(logits[start + j]);
                ndp[start + j] = sr;
                nxt[idx++] = cur[j] * (1.f - sr);
                nxt[idx++] = cur[j] * sr;
            }
            cur_n *= 2;
            for (int64_t j = 0; j < cur_n; ++j) cur[j] = nxt[j];
        }
        float ssum = 0.f;
        for (int64_t k = 0; k < K; ++k) ssum += cur[k];
        if (ssum < 1e-8f) ssum = 1e-8f;
        int64_t i1 = 0, i2 = 1;
        float v1 = -1.f, v2 = -1.f;
        for (int64_t k = 0; k < K; ++k) {
            float v = cur[k] / ssum;
            if (v > v1) { v2 = v1; i2 = i1; v1 = v; i1 = k; }
            else if (v > v2) { v2 = v; i2 = k; }
        }
        float s = v1 + v2;
        if (s < 1e-8f) s = 1e-8f;
        ti_ptr[n * 2] = (int32_t)i1;
        ti_ptr[n * 2 + 1] = (int32_t)i2;
        tv_ptr[n * 2] = v1;
        tv_ptr[n * 2 + 1] = v2;
        tw_ptr[n * 2] = v1 / s;
        tw_ptr[n * 2 + 1] = v2 / s;
    }
    auto [y_channel_flat, active_leaf_outs] = asdag_sparse_tree_perm_forward_cpp(
        r_in_flat, w_perm, perms, bias, top_idx, top_w
    );
    const float* yc_ptr = y_channel_flat.data_ptr<float>();
#pragma omp parallel for collapse(2) num_threads(n_threads) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            const float* x1b = x1_ptr + (b * T + t) * C;
            const float* ycb = yc_ptr + (b * T + t) * C;
            float* outb = out_ptr + (b * T + t) * C;
            for (int64_t c = 0; c < C; ++c) outb[c] = x1b[c] + ycb[c];
        }
    }
    return std::make_tuple(out.to(orig_dtype), x_norm1, x1, x_norm2, qkvg_raw, phi_q, phi_k, gamma_all, S_all, z_all, y_mod, active_leaf_outs,
        r_in_flat, xq_save, root_out_save, root_preact, node_p, top_idx, top_w, top_vals, sc0, scf, sc1);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
asdag_fused_asdag_tree_block_backward_cpp(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor norm1_scale,
    torch::Tensor x_norm1,
    torch::Tensor x1,
    torch::Tensor norm2_scale,
    torch::Tensor x_norm2,
    torch::Tensor qkvg_diagonals,
    torch::Tensor qkvg_perms,
    torch::Tensor qkvg_inv_perms,
    torch::Tensor qkvg_bias,
    torch::Tensor qkvg_raw,
    torch::Tensor phi_q,
    torch::Tensor phi_k,
    torch::Tensor gamma_all,
    torch::Tensor S_all,
    torch::Tensor z_all,
    torch::Tensor y_mod,
    torch::Tensor q_norm_scale,
    torch::Tensor k_norm_scale,
    torch::Tensor w_decay,
    torch::Tensor b_decay,
    torch::Tensor out_diagonals,
    torch::Tensor out_perms,
    torch::Tensor out_inv_perms,
    torch::Tensor out_bias,
    torch::Tensor w_perm,
    torch::Tensor perms,
    torch::Tensor inv_perms,
    torch::Tensor bias,
    torch::Tensor root_latent_w,
    torch::Tensor root_scale,
    torch::Tensor root_bias,
    torch::Tensor root_perms,
    torch::Tensor hyperplanes,
    torch::Tensor router_biases,
    torch::Tensor r_in_flat,
    torch::Tensor xq_save,
    torch::Tensor root_out_save,
    torch::Tensor root_preact,
    torch::Tensor node_p,
    torch::Tensor top_idx,
    torch::Tensor top_vals,
    torch::Tensor active_leaf_outs,
    torch::Tensor sc0,
    torch::Tensor sc1,
    torch::Tensor reset_mask
) {
    grad_y = grad_y.contiguous().to(torch::kFloat32);
    x = x.contiguous().to(torch::kFloat32);
    norm1_scale = norm1_scale.contiguous().to(torch::kFloat32);
    x_norm1 = x_norm1.contiguous().to(torch::kFloat32);
    x1 = x1.contiguous().to(torch::kFloat32);
    norm2_scale = norm2_scale.contiguous().to(torch::kFloat32);
    x_norm2 = x_norm2.contiguous().to(torch::kFloat32);
    int64_t B = x.size(0);
    int64_t T = x.size(1);
    int64_t C = x.size(2);
    using namespace asdag_fused_tree;
    root_latent_w = root_latent_w.contiguous().to(torch::kFloat32);
    root_scale = root_scale.contiguous().to(torch::kFloat32);
    root_bias = root_bias.contiguous().to(torch::kFloat32);
    root_perms = root_perms.contiguous().to(torch::kInt32);
    hyperplanes = hyperplanes.contiguous().to(torch::kFloat32);
    router_biases = router_biases.contiguous().to(torch::kFloat32);
    int64_t N = B * T;
    int64_t I = hyperplanes.size(0);
    int64_t RP = root_latent_w.size(0);
    int depth = 1;
    while ((((int64_t)1) << depth) - 1 < I) ++depth;
    int n_threads = asdag::get_physical_cores();
    auto gy_flat = grad_y.reshape({B * T, C});
    auto top_w_re = torch::empty({N, 2}, torch::kFloat32);
    {
        const float* tv = top_vals.data_ptr<float>();
        float* tw = top_w_re.data_ptr<float>();
        for (int64_t n = 0; n < N; ++n) {
            float s = tv[n * 2] + tv[n * 2 + 1];
            if (s < 1e-8f) s = 1e-8f;
            tw[n * 2] = tv[n * 2] / s;
            tw[n * 2 + 1] = tv[n * 2 + 1] / s;
        }
    }
    auto active_flat = active_leaf_outs.reshape({B * T, 2, C});
    auto [g_rin_flat, g_w_perm, g_bias_tree, g_topw] = asdag_sparse_tree_perm_backward_cpp(
        gy_flat, r_in_flat, w_perm, perms, inv_perms, bias, top_idx, top_w_re, active_flat
    );
    const float* rlat_ptr = root_latent_w.data_ptr<float>();
    const float* rsc_ptr = root_scale.data_ptr<float>();
    const int32_t* rpm_ptr = root_perms.data_ptr<int32_t>();
    const float* hyp_ptr = hyperplanes.data_ptr<float>();
    std::vector<float> w_root(RP * C), w_root_sgn(RP * C), w_route(I * C);
    {
        float delta = ternary_delta(rlat_ptr, RP * C, 0.7f);
        for (int64_t i = 0; i < RP * C; ++i) {
            float v = rlat_ptr[i];
            float s = (v > delta) ? 1.f : ((v < -delta) ? -1.f : 0.f);
            w_root_sgn[i] = s;
            w_root[i] = s * rsc_ptr[(i / C) % RP];
        }
        float deltah = ternary_delta(hyp_ptr, I * C, 0.7f);
        std::vector<float> sg(I * C);
        for (int64_t i = 0; i < I * C; ++i) {
            float v = hyp_ptr[i];
            sg[i] = (v > deltah) ? 1.f : ((v < -deltah) ? -1.f : 0.f);
        }
        float alpha = ternary_alpha_active(hyp_ptr, sg.data(), I * C);
        for (int64_t i = 0; i < I * C; ++i) w_route[i] = sg[i] * alpha;
    }
    auto x_norm2_c = x_norm2.contiguous().to(torch::kFloat32);
    const float* xn2c_ptr = x_norm2_c.data_ptr<float>();
    const float* grin_ptr = g_rin_flat.data_ptr<float>();
    const float* gtw_ptr = g_topw.data_ptr<float>();
    const float* xq_ptr = xq_save.data_ptr<float>();
    const float* ro_ptr = root_out_save.data_ptr<float>();
    const float* pre_ptr = root_preact.data_ptr<float>();
    const float* ndp_ptr = node_p.data_ptr<float>();
    const int32_t* ti_ptr = top_idx.data_ptr<int32_t>();
    const float* tv_ptr = top_vals.data_ptr<float>();
    const float* s0_ptr = sc0.data_ptr<float>();
    const float* s1_ptr = sc1.data_ptr<float>();
    auto g_root_path = torch::zeros({N, C}, torch::kFloat32);
    auto g_route_path = torch::zeros({N, C}, torch::kFloat32);
    float* grp_ptr = g_root_path.data_ptr<float>();
    float* grt_ptr = g_route_path.data_ptr<float>();
    std::vector<std::vector<float>> t_gw(n_threads, std::vector<float>(RP * C, 0.f));
    std::vector<std::vector<float>> t_gs(n_threads, std::vector<float>(RP, 0.f));
    std::vector<std::vector<float>> t_gb(n_threads, std::vector<float>(C, 0.f));
    std::vector<std::vector<float>> t_gh(n_threads, std::vector<float>(I * C, 0.f));
    std::vector<std::vector<float>> t_grb(n_threads, std::vector<float>(I, 0.f));
#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        asdag::pin_thread_to_physical_core(tid);
        std::vector<float> gxq(C), dl(I), dn(16), dnp(16), lv(32);
#pragma omp for schedule(static)
        for (int64_t n = 0; n < N; ++n) {
            const float* grin = grin_ptr + n * C;
            const float* xq = xq_ptr + n * C;
            const float* ro = ro_ptr + n * C;
            const float* pre = pre_ptr + n * C;
            const float* xn = xn2c_ptr + n * C;
            float s1 = s1_ptr[n];
            float s0 = s0_ptr[n];
            for (int64_t c = 0; c < C; ++c) gxq[c] = 0.f;
            for (int64_t c = 0; c < C; ++c) {
                float gro = (std::fabs(ro[c] / s1) <= 1.5f) ? grin[c] : 0.f;
                float gh = (pre[c] > 0.f && pre[c] < 6.f) ? gro : 0.f;
                t_gb[tid][c] += gh;
                for (int64_t p = 0; p < RP; ++p) {
                    float g = gh * xq[rpm_ptr[p * C + c]];
                    t_gw[tid][p * C + c] += g;
                    t_gs[tid][p] += w_root_sgn[p * C + c] * g;
                    gxq[rpm_ptr[p * C + c]] += gh * w_root[p * C + c];
                }
            }
            float* grp = grp_ptr + n * C;
            for (int64_t c = 0; c < C; ++c) {
                float g = e5m2_round(gxq[c]);
                grp[c] = (std::fabs(xn[c] / s0) <= 1.5f) ? g : 0.f;
            }
            float v0 = tv_ptr[n * 2], v1 = tv_ptr[n * 2 + 1];
            float s = v0 + v1;
            if (s < 1e-8f) s = 1e-8f;
            float g0 = gtw_ptr[n * 2], g1 = gtw_ptr[n * 2 + 1];
            float dot = g0 * v0 + g1 * v1;
            float dv0 = (g0 * s - dot) / (s * s);
            float dv1 = (g1 * s - dot) / (s * s);
            for (int64_t k = 0; k < 16; ++k) dn[k] = 0.f;
            dn[ti_ptr[n * 2]] += dv0;
            dn[ti_ptr[n * 2 + 1]] += dv1;
            const float* ndp = ndp_ptr + n * I;
            lv[0] = 1.f - ndp[0];
            lv[1] = ndp[0];
            int64_t off = 0;
            for (int d = 1; d < depth; ++d) {
                int64_t start = (((int64_t)1) << d) - 1;
                int64_t idx = 0;
                for (int64_t j = 0; j < (((int64_t)1) << d); ++j) {
                    float sr = ndp[start + j];
                    lv[off + 2 + idx++] = lv[off + j] * (1.f - sr);
                    lv[off + 2 + idx++] = lv[off + j] * sr;
                }
                off += 2;
            }
            for (int d = depth - 1; d >= 1; --d) {
                int64_t start = (((int64_t)1) << d) - 1;
                int64_t npar = ((int64_t)1) << d;
                int64_t poff = off - 2;
                for (int64_t j = 0; j < npar; ++j) dnp[j] = 0.f;
                for (int64_t j = 0; j < npar; ++j) {
                    float sr = ndp[start + j];
                    float sl = 1.f - sr;
                    float pv = lv[poff + j];
                    dnp[j] = dn[2 * j] * sl + dn[2 * j + 1] * sr;
                    dl[start + j] = (dn[2 * j + 1] - dn[2 * j]) * pv * 2.f * sr * (1.f - sr);
                }
                for (int64_t j = 0; j < npar; ++j) dn[j] = dnp[j];
                off = poff;
            }
            float sr0 = ndp[0];
            dl[0] = (dn[1] - dn[0]) * 2.f * sr0 * (1.f - sr0);
            float* grt = grt_ptr + n * C;
            for (int64_t i = 0; i < I; ++i) {
                float dli = dl[i];
                t_grb[tid][i] += dli;
                const float* w = &w_route[i * C];
                float* gh = t_gh[tid].data() + i * C;
                for (int64_t c = 0; c < C; ++c) {
                    gh[c] += dli * xn[c];
                    grt[c] += dli * w[c];
                }
            }
        }
    }
    auto grad_xn2 = (g_root_path + g_route_path).reshape({B, T, C});
    auto g_root_w = torch::zeros({RP, C}, torch::kFloat32);
    auto g_root_scale = torch::zeros({RP, 1}, torch::kFloat32);
    auto g_root_b = torch::zeros({C}, torch::kFloat32);
    auto g_hyper = torch::zeros({I, C}, torch::kFloat32);
    auto g_router_b = torch::zeros({I}, torch::kFloat32);
    {
        float* a = g_root_w.data_ptr<float>();
        float* b = g_root_scale.data_ptr<float>();
        float* c = g_root_b.data_ptr<float>();
        float* d = g_hyper.data_ptr<float>();
        float* e = g_router_b.data_ptr<float>();
        for (int t = 0; t < n_threads; ++t) {
            const float* x1 = t_gw[t].data();
            const float* x2 = t_gs[t].data();
            const float* x3 = t_gb[t].data();
            const float* x4 = t_gh[t].data();
            const float* x5 = t_grb[t].data();
            for (int64_t i = 0; i < RP * C; ++i) a[i] += x1[i];
            for (int64_t i = 0; i < RP; ++i) b[i] += x2[i];
            for (int64_t i = 0; i < C; ++i) c[i] += x3[i];
            for (int64_t i = 0; i < I * C; ++i) d[i] += x4[i];
            for (int64_t i = 0; i < I; ++i) e[i] += x5[i];
        }
    }
    auto grad_x1 = grad_y.clone();
    auto grad_norm2_scale = torch::zeros({C}, torch::kFloat32);
    const float* gx2_ptr = grad_xn2.data_ptr<float>();
    const float* x1_ptr = x1.data_ptr<float>();
    const float* n2_ptr = norm2_scale.data_ptr<float>();
    float* gx1_ptr = grad_x1.data_ptr<float>();
    float* gn2_ptr = grad_norm2_scale.data_ptr<float>();
    std::vector<std::vector<float>> thread_gn2(n_threads, std::vector<float>(C, 0.0f));
#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        asdag::pin_thread_to_physical_core(tid);
        float* local_gn2 = thread_gn2[tid].data();
#pragma omp for collapse(2) schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t t = 0; t < T; ++t) {
                const float* gx2 = gx2_ptr + (b * T + t) * C;
                const float* x1b = x1_ptr + (b * T + t) * C;
                float* gx1b = gx1_ptr + (b * T + t) * C;
                float sum_gx2_x1 = 0.0f, sum_x1_sq = 0.0f;
                for (int64_t c = 0; c < C; ++c) { sum_gx2_x1 += gx2[c] * x1b[c]; sum_x1_sq += x1b[c] * x1b[c]; }
                float rms = 1.0f / std::sqrt((sum_x1_sq / (float)C) + 1e-6f);
                for (int64_t c = 0; c < C; ++c) {
                    float g = gx2[c] * rms * n2_ptr[c];
                    g -= x1b[c] * rms * n2_ptr[c] * sum_gx2_x1 / (float)C * rms * rms;
                    gx1b[c] += g;
                    local_gn2[c] += gx2[c] * x1b[c] * rms;
                }
            }
        }
    }
    for (int t = 0; t < n_threads; ++t) for (int64_t c = 0; c < C; ++c) gn2_ptr[c] += thread_gn2[t][c];
    auto [g_xn1, g_qkvg_d, g_qkvg_b, g_qs, g_ks, g_wd_gla, g_bd, g_od, g_ob] = asdag_fused_monarch_gla_backward_cpp(
        grad_x1, x_norm1, qkvg_diagonals, qkvg_perms, qkvg_inv_perms, qkvg_raw,
        phi_q, phi_k, gamma_all, S_all, z_all, y_mod,
        q_norm_scale, k_norm_scale, w_decay, b_decay,
        out_diagonals, out_perms, out_inv_perms, reset_mask
    );
    auto grad_x = g_xn1.clone();
    auto grad_norm1_scale = torch::zeros({C}, torch::kFloat32);
    const float* gx1g_ptr = g_xn1.data_ptr<float>();
    const float* x_ptr = x.data_ptr<float>();
    const float* n1_ptr = norm1_scale.data_ptr<float>();
    float* gx_ptr = grad_x.data_ptr<float>();
    float* gn1_ptr = grad_norm1_scale.data_ptr<float>();
    std::vector<std::vector<float>> thread_gn1(n_threads, std::vector<float>(C, 0.0f));
#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        asdag::pin_thread_to_physical_core(tid);
        float* local_gn1 = thread_gn1[tid].data();
#pragma omp for collapse(2) schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t t = 0; t < T; ++t) {
                const float* gx1g = gx1g_ptr + (b * T + t) * C;
                const float* xb = x_ptr + (b * T + t) * C;
                float* gxb = gx_ptr + (b * T + t) * C;
                float sum_sq = 0.0f, sum_gx = 0.0f;
                for (int64_t c = 0; c < C; ++c) sum_sq += xb[c]*xb[c];
                float rms = 1.0f / std::sqrt((sum_sq/(float)C)+1e-6f);
                for (int64_t c = 0; c < C; ++c) sum_gx += gx1g[c]*xb[c];
                for (int64_t c = 0; c < C; ++c) {
                    float g = gx1g[c]*rms*n1_ptr[c] - xb[c]*rms*n1_ptr[c]*sum_gx/(float)C * rms * rms;
                    gxb[c] = g;
                    local_gn1[c] += gx1g[c]*xb[c]*rms;
                }
            }
        }
    }
    for (int t = 0; t < n_threads; ++t) for (int64_t c = 0; c < C; ++c) gn1_ptr[c] += thread_gn1[t][c];
    return std::make_tuple(grad_x, grad_norm1_scale, g_qkvg_d, g_qkvg_b, g_qs, g_ks, g_wd_gla, g_bd, g_od, g_ob, grad_norm2_scale, g_w_perm, g_bias_tree,
        g_root_w.to(torch::kFloat32), g_root_scale.to(torch::kFloat32), g_root_b.to(torch::kFloat32), g_hyper.to(torch::kFloat32), g_router_b.to(torch::kFloat32));
}

// ─────────────────────────────────────────────────────────────────────────────
// 27. Native Fused CPU LPC Head (Zero Logit Materialization + AVX SIMD + OpenMP)
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> asdag_lpc_head_forward_backward_cpp(
    torch::Tensor H,        // [N, D] float32
    torch::Tensor W,        // [V, D] float32
    torch::Tensor targets,  // [N] int64
    int64_t ignore_index
) {
    auto orig_dtype = H.dtype();
    H = H.contiguous().to(torch::kFloat32);
    W = W.contiguous().to(torch::kFloat32);
    targets = targets.contiguous().to(torch::kInt64);

    int64_t N = H.size(0);
    int64_t D = H.size(1);
    int64_t V = W.size(0);

    auto grad_H = torch::zeros_like(H);
    auto grad_W = torch::zeros_like(W);
    auto loss_tensor = torch::zeros({1}, H.options());

    const int64_t* t_ptr = targets.data_ptr<int64_t>();

    // 1. Hardware-accelerated Forward GEMM: Logits = H @ W.T
    auto logits = at::mm(H, W.t());
    const float* logits_ptr = logits.data_ptr<float>();

    auto delta = torch::empty_like(logits);
    float* delta_ptr = delta.data_ptr<float>();

    int n_threads = asdag::get_physical_cores();

    // Count valid tokens
    int64_t valid_tokens = 0;
    for (int64_t i = 0; i < N; ++i) {
        if (t_ptr[i] != ignore_index && t_ptr[i] >= 0 && t_ptr[i] < V) {
            valid_tokens++;
        }
    }

    if (valid_tokens == 0) {
        return std::make_tuple(loss_tensor, torch::zeros_like(H).to(orig_dtype), torch::zeros_like(W).to(orig_dtype));
    }

    float inv_valid = 1.0f / (float)valid_tokens;
    double total_loss = 0.0;

    // 2. Fused OpenMP SIMD Log-Sum-Exp + Softmax Gradient Kernel
#pragma omp parallel for num_threads(n_threads) reduction(+:total_loss) schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        int64_t target_v = t_ptr[i];
        if (target_v == ignore_index || target_v < 0 || target_v >= V) {
            continue;
        }

        const float* z_i = logits_ptr + i * V;
        float* d_i = delta_ptr + i * V;

        // Find max
        float max_val = -1e30f;
        for (int64_t v = 0; v < V; ++v) {
            if (z_i[v] > max_val) max_val = z_i[v];
        }

        // Sum exp
        float sum_exp = 0.0f;
        for (int64_t v = 0; v < V; ++v) {
            float exp_val = std::exp(z_i[v] - max_val);
            d_i[v] = exp_val;
            sum_exp += exp_val;
        }

        float lse = max_val + std::log(std::max(sum_exp, 1e-12f));
        float inv_sum_exp = 1.0f / std::max(sum_exp, 1e-12f);
        float target_dot = z_i[target_v];
        total_loss += (double)(lse - target_dot);

        // Softmax gradient: (p_v - I) * inv_valid
        for (int64_t v = 0; v < V; ++v) {
            float p_v = d_i[v] * inv_sum_exp;
            d_i[v] = (p_v - (v == target_v ? 1.0f : 0.0f)) * inv_valid;
        }
    }

    loss_tensor[0] = (float)(total_loss * inv_valid);

    // 3. Hardware-accelerated Backward GEMMs (In-place MKL multi-threaded BLAS):
    // grad_H = delta @ W
    // grad_W = delta.T @ H
    at::mm_out(grad_H, delta, W);
    at::mm_out(grad_W, delta.t(), H);

    return std::make_tuple(loss_tensor, grad_H.to(orig_dtype), grad_W.to(orig_dtype));
}

} // namespace asdag_cpu

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_tree_perm_forward", &asdag_cpu::asdag_sparse_tree_perm_forward_cpp, "ASDAG CPU SIMD-Block N:M Sparse Tree Forward (AVX2/AVX-512)");
    m.def("sparse_tree_perm_backward", &asdag_cpu::asdag_sparse_tree_perm_backward_cpp, "ASDAG CPU SIMD-Block N:M Sparse Tree Backward (AVX2/AVX-512)");
    m.def("fused_perm_proj_forward", &asdag_cpu::asdag_fused_perm_proj_forward_cpp, "ASDAG CPU Fused Permutation Projection Forward (AVX2/AVX-512)");
    m.def("fused_perm_proj_backward", &asdag_cpu::asdag_fused_perm_proj_backward_cpp, "ASDAG CPU Fused Permutation Projection Backward (AVX2/AVX-512)");
    m.def("monarch_chain_forward", &asdag_cpu::asdag_monarch_chain_forward_cpp, "ASDAG CPU Monarch Permutation Chain Forward (AVX2/AVX-512)");
    m.def("monarch_chain_backward", &asdag_cpu::asdag_monarch_chain_backward_cpp, "ASDAG CPU Monarch Permutation Chain Backward (AVX2/AVX-512)");
    m.def("monarch_reg_forward", &asdag_cpu::asdag_monarch_reg_forward_cpp, "ASDAG CPU Register-Fused Multi-Stage Monarch Forward");
    m.def("fused_monarch_chain_forward", &asdag_cpu::asdag_fused_monarch_chain_forward_cpp, "ASDAG CPU Fused Monarch Chain Forward (AVX2/AVX-512)");
    m.def("fused_monarch_chain_backward", &asdag_cpu::asdag_fused_monarch_chain_backward_cpp, "ASDAG CPU Fused Monarch Chain Backward (AVX2/AVX-512)");
    m.def("bitlinear_forward", &asdag_cpu::asdag_bitlinear_forward_cpp, "ASDAG CPU BitLinear Ternary Forward (AVX2/AVX-512)");
    m.def("bitlinear_backward", &asdag_cpu::asdag_bitlinear_backward_cpp, "ASDAG CPU BitLinear Ternary Backward (AVX2/AVX-512)");
    m.def("bitlinear_twin_forward", &asdag_cpu::asdag_bitlinear_twin_forward_cpp, "ASDAG CPU BitLinear Twin Forward (AVX2/AVX-512)");
    m.def("bitlinear_twin_backward", &asdag_cpu::asdag_bitlinear_twin_backward_cpp, "ASDAG CPU BitLinear Twin Backward (AVX2/AVX-512)");
    m.def("bitlinear_ternary_int_forward", &asdag_cpu::asdag_bitlinear_ternary_int_forward_cpp, "ASDAG CPU 1-Cycle Pure Integer Ternary Add/Sub BitLinear Forward");
    m.def("bitlinear_swiglu_forward", &asdag_cpu::asdag_bitlinear_swiglu_forward_cpp, "ASDAG CPU Fused BitLinear SwiGLU Forward");
    m.def("bitlinear_swiglu_backward_recompute", &asdag_cpu::asdag_bitlinear_swiglu_backward_recompute_cpp, "ASDAG CPU Zero-RAM Rematerialization SwiGLU Backward");
    m.def("pack_ternary_2bit", &asdag_cpu::asdag_pack_ternary_2bit_cpp, "ASDAG CPU Pack Ternary Weights into 2-bit");
    m.def("unpack_ternary_2bit", &asdag_cpu::asdag_unpack_ternary_2bit_cpp, "ASDAG CPU Unpack 2-bit into Ternary Weights");
    m.def("blt_simd_patcher", &asdag_cpu::asdag_blt_simd_patcher_cpp, "ASDAG CPU Sub-Byte SIMD Dynamic Entropy Patcher");
    m.def("byte_encoder_forward", &asdag_cpu::asdag_byte_encoder_forward_cpp, "ASDAG CPU Fused Byte Encoder Forward");
    m.def("byte_encoder_backward", &asdag_cpu::asdag_byte_encoder_backward_cpp, "ASDAG CPU Fused Byte Encoder Backward");
    m.def("blt_2layer_decode_fused", &asdag_cpu::asdag_blt_2layer_decode_fused_cpp, "ASDAG CPU 2-Layer Fused BLT Causal Byte Decoder");
    m.def("blt_2layer_decode_loss_fused", &asdag_cpu::asdag_blt_2layer_decode_loss_fused_cpp, "ASDAG CPU 2-Layer Fused BLT Causal Decoder + Cross-Entropy Loss");
    m.def("fused_monarch_gla_forward", &asdag_cpu::asdag_fused_monarch_gla_forward_cpp, "ASDAG CPU Full-Layer Fused Monarch GLA Forward (AVX2/AVX-512)");
    m.def("fused_monarch_gla_backward", &asdag_cpu::asdag_fused_monarch_gla_backward_cpp, "ASDAG CPU Full-Layer Fused Monarch GLA Backward (AVX2/AVX-512)");
    m.def("fused_asdag_block_forward", &asdag_cpu::asdag_fused_asdag_block_forward_cpp, "ASDAG CPU Full-Block Fused ASDAG Layer Forward (AVX2/AVX-512)");
    m.def("fused_asdag_block_backward", &asdag_cpu::asdag_fused_asdag_block_backward_cpp, "ASDAG CPU Full-Block Fused ASDAG Layer Backward (AVX2/AVX-512)");
    m.def("fused_asdag_tree_block_forward", &asdag_cpu::asdag_fused_asdag_tree_block_forward_cpp, "ASDAG CPU Fused Tree Block Forward (AVX2/AVX-512)");
    m.def("fused_asdag_tree_block_backward", &asdag_cpu::asdag_fused_asdag_tree_block_backward_cpp, "ASDAG CPU Fused Tree Block Backward (AVX2/AVX-512)");
    m.def("fused_rmsnorm_proj", &asdag_cpu::asdag_fused_rmsnorm_proj_cpp, "ASDAG CPU Fused RMSNorm Projection");
    m.def("newton_schulz5", &asdag_cpu::asdag_newton_schulz5_cpp, "ASDAG CPU 5th-Order Newton-Schulz Optimizer Kernel");
    m.def("gla_scan_forward", &asdag_cpu::asdag_gla_scan_forward_cpp, "ASDAG CPU Fused GLA Associative Scan Forward (AVX2/AVX-512)");
    m.def("gla_scan_backward", &asdag_cpu::asdag_gla_scan_backward_cpp, "ASDAG CPU Fused GLA Associative Scan Backward (AVX2/AVX-512)");
    m.def("gla_step", &asdag_cpu::asdag_gla_step_cpp, "ASDAG CPU Native GLA O(1) State Space Step (AVX2/AVX-512)");
    m.def("lpc_head_forward_backward", &asdag_cpu::asdag_lpc_head_forward_backward_cpp, "ASDAG CPU Fused LPC Local Head (Zero Logits RAM + AVX SIMD)");
    m.def("forward", &asdag_cpu::asdag_forward_cpp, "ASDAG CPU Dense Forward (Legacy)");
    m.def("backward", &asdag_cpu::asdag_backward_cpp, "ASDAG CPU Dense Backward (Legacy)");
}
