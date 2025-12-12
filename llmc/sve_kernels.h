/*
 SVE-optimized kernels for A64FX (512-bit SVE)
 These kernels use ARM SVE intrinsics for vectorized operations.
 A64FX has 512-bit SVE vectors = 16 floats per vector.
*/
#ifndef SVE_KERNELS_H
#define SVE_KERNELS_H

#ifdef USE_SVE
#include <arm_sve.h>
#include <math.h>

// Maximum SVE vector length for float32 (512 bits = 16 floats on A64FX)
// Used for stack-allocated temporary arrays
#define SVE_MAX_FLOATS 16

// ----------------------------------------------------------------------------
// SVE-optimized matmul forward (blocked version)
// inp is (B*T, C), weight is (OC, C), bias is (OC), out is (B*T, OC)

static inline void matmul_forward_sve_blocked(float* out,
                                              const float* inp, const float* weight, const float* bias,
                                              int BT, int C, int OC) {
    #pragma omp parallel for collapse(2)
    for (int bt = 0; bt < BT; bt++) {
        for (int o = 0; o < OC; o++) {
            const float* inp_bt = inp + bt * C;
            float val = (bias != NULL) ? bias[o] : 0.0f;
            const float* wrow = weight + o * C;

            // Use SVE for vectorized dot product
            svbool_t pg = svptrue_b32();
            svfloat32_t acc = svdup_f32(0.0f);

            uint64_t vl = svcntw();  // number of floats per vector
            int i = 0;

            // Process C in SVE-sized chunks
            for (; i + (int)vl <= C; i += (int)vl) {
                svfloat32_t inp_vec = svld1_f32(pg, inp_bt + i);
                svfloat32_t w_vec = svld1_f32(pg, wrow + i);
                acc = svmla_f32_x(pg, acc, inp_vec, w_vec);
            }

            // Horizontal sum of accumulated values
            val += svaddv_f32(pg, acc);

            // Handle remaining elements with predicated load
            if (i < C) {
                svbool_t pg_tail = svwhilelt_b32((uint32_t)i, (uint32_t)C);
                svfloat32_t inp_vec = svld1_f32(pg_tail, inp_bt + i);
                svfloat32_t w_vec = svld1_f32(pg_tail, wrow + i);
                svfloat32_t prod = svmul_f32_x(pg_tail, inp_vec, w_vec);
                val += svaddv_f32(pg_tail, prod);
            }

            out[bt * OC + o] = val;
        }
    }
}

// ----------------------------------------------------------------------------
// SVE-optimized layernorm forward

static inline void layernorm_forward_sve(float* out, float* mean, float* rstd,
                                         const float* inp, const float* weight, const float* bias,
                                         int B, int T, int C) {
    const float eps = 1e-5f;

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* x = inp + b * T * C + t * C;
            float* out_bt = out + b * T * C + t * C;

            svbool_t pg = svptrue_b32();
            uint64_t vl = svcntw();

            // Calculate mean using SVE
            svfloat32_t sum_vec = svdup_f32(0.0f);
            int i = 0;

            for (; i + (int)vl <= C; i += (int)vl) {
                svfloat32_t x_vec = svld1_f32(pg, x + i);
                sum_vec = svadd_f32_x(pg, sum_vec, x_vec);
            }
            float m = svaddv_f32(pg, sum_vec);

            // Handle remaining with predicated load
            if (i < C) {
                svbool_t pg_tail = svwhilelt_b32((uint32_t)i, (uint32_t)C);
                svfloat32_t x_vec = svld1_f32(pg_tail, x + i);
                m += svaddv_f32(pg_tail, x_vec);
            }
            m = m / C;

            // Calculate variance using SVE
            svfloat32_t var_vec = svdup_f32(0.0f);
            svfloat32_t mean_vec = svdup_f32(m);

            i = 0;
            for (; i + (int)vl <= C; i += (int)vl) {
                svfloat32_t x_vec = svld1_f32(pg, x + i);
                svfloat32_t diff = svsub_f32_x(pg, x_vec, mean_vec);
                var_vec = svmla_f32_x(pg, var_vec, diff, diff);
            }
            float v = svaddv_f32(pg, var_vec);

            // Handle remaining
            if (i < C) {
                svbool_t pg_tail = svwhilelt_b32((uint32_t)i, (uint32_t)C);
                svfloat32_t x_vec = svld1_f32(pg_tail, x + i);
                svfloat32_t diff = svsub_f32_x(pg_tail, x_vec, mean_vec);
                svfloat32_t diff_sq = svmul_f32_x(pg_tail, diff, diff);
                v += svaddv_f32(pg_tail, diff_sq);
            }
            v = v / C;

            // Calculate rstd
            float s = 1.0f / sqrtf(v + eps);

            // Normalize, scale, and shift using SVE
            svfloat32_t rstd_vec = svdup_f32(s);

            i = 0;
            for (; i + (int)vl <= C; i += (int)vl) {
                svfloat32_t x_vec = svld1_f32(pg, x + i);
                svfloat32_t w_vec = svld1_f32(pg, weight + i);
                svfloat32_t b_vec = svld1_f32(pg, bias + i);

                // n = (x - mean) * rstd
                svfloat32_t n = svmul_f32_x(pg, svsub_f32_x(pg, x_vec, mean_vec), rstd_vec);
                // out = n * weight + bias
                svfloat32_t result = svmla_f32_x(pg, b_vec, n, w_vec);
                svst1_f32(pg, out_bt + i, result);
            }

            // Handle remaining
            if (i < C) {
                svbool_t pg_tail = svwhilelt_b32((uint32_t)i, (uint32_t)C);
                svfloat32_t x_vec = svld1_f32(pg_tail, x + i);
                svfloat32_t w_vec = svld1_f32(pg_tail, weight + i);
                svfloat32_t b_vec = svld1_f32(pg_tail, bias + i);

                svfloat32_t n = svmul_f32_x(pg_tail, svsub_f32_x(pg_tail, x_vec, mean_vec), rstd_vec);
                svfloat32_t result = svmla_f32_x(pg_tail, b_vec, n, w_vec);
                svst1_f32(pg_tail, out_bt + i, result);
            }

            // Store mean and rstd for backward pass
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

// ----------------------------------------------------------------------------
// SVE-optimized encoder forward (embedding lookup + positional)

static inline void encoder_forward_sve(float* out,
                                       const int* inp, const float* wte, const float* wpe,
                                       int B, int T, int C) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * C + t * C;
            int ix = inp[b * T + t];
            const float* wte_ix = wte + ix * C;
            const float* wpe_t = wpe + t * C;

            svbool_t pg = svptrue_b32();
            uint64_t vl = svcntw();
            int i = 0;

            // Add embeddings using SVE
            for (; i + (int)vl <= C; i += (int)vl) {
                svfloat32_t wte_vec = svld1_f32(pg, wte_ix + i);
                svfloat32_t wpe_vec = svld1_f32(pg, wpe_t + i);
                svfloat32_t result = svadd_f32_x(pg, wte_vec, wpe_vec);
                svst1_f32(pg, out_bt + i, result);
            }

            // Handle remaining with predicated operations
            if (i < C) {
                svbool_t pg_tail = svwhilelt_b32((uint32_t)i, (uint32_t)C);
                svfloat32_t wte_vec = svld1_f32(pg_tail, wte_ix + i);
                svfloat32_t wpe_vec = svld1_f32(pg_tail, wpe_t + i);
                svfloat32_t result = svadd_f32_x(pg_tail, wte_vec, wpe_vec);
                svst1_f32(pg_tail, out_bt + i, result);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// SVE-optimized GeLU forward
// Note: SVE doesn't have native tanh, so we use scalar fallback for that part

static inline void gelu_forward_sve(float* out, const float* inp, int N) {
    const float GELU_SCALING_FACTOR = 0.7978845608f;  // sqrt(2/pi)
    const float c = 0.044715f;

    #pragma omp parallel for
    for (int base = 0; base < N; base += SVE_MAX_FLOATS) {
        int remaining = N - base;
        int chunk = (remaining < SVE_MAX_FLOATS) ? remaining : SVE_MAX_FLOATS;

        // Process in chunks that fit in our temporary arrays
        float tanh_args[SVE_MAX_FLOATS];
        float x_vals[SVE_MAX_FLOATS];
        float results[SVE_MAX_FLOATS];

        // Load and compute tanh arguments using SVE where possible
        svbool_t pg = svwhilelt_b32((uint32_t)0, (uint32_t)chunk);
        svfloat32_t x = svld1_f32(pg, inp + base);

        // cube = c * x * x * x
        svfloat32_t x2 = svmul_f32_x(pg, x, x);
        svfloat32_t x3 = svmul_f32_x(pg, x2, x);
        svfloat32_t cube = svmul_f32_x(pg, x3, svdup_f32(c));

        // tanh_arg = GELU_SCALING_FACTOR * (x + cube)
        svfloat32_t sum = svadd_f32_x(pg, x, cube);
        svfloat32_t tanh_arg = svmul_f32_x(pg, sum, svdup_f32(GELU_SCALING_FACTOR));

        // Store to temporary arrays for scalar tanh computation
        svst1_f32(pg, tanh_args, tanh_arg);
        svst1_f32(pg, x_vals, x);

        // Compute tanh and final result in scalar (no SVE tanh available)
        for (int j = 0; j < chunk; j++) {
            float tanh_out = tanhf(tanh_args[j]);
            results[j] = 0.5f * x_vals[j] * (1.0f + tanh_out);
        }

        // Store results
        svfloat32_t result = svld1_f32(pg, results);
        svst1_f32(pg, out + base, result);
    }
}

// ----------------------------------------------------------------------------
// SVE-optimized residual forward

static inline void residual_forward_sve(float* out, const float* inp1, const float* inp2, int N) {
    svbool_t pg = svptrue_b32();
    uint64_t vl = svcntw();
    int i = 0;

    for (; i + (int)vl <= N; i += (int)vl) {
        svfloat32_t a = svld1_f32(pg, inp1 + i);
        svfloat32_t b = svld1_f32(pg, inp2 + i);
        svfloat32_t result = svadd_f32_x(pg, a, b);
        svst1_f32(pg, out + i, result);
    }

    // Handle remaining with predicated operations
    if (i < N) {
        svbool_t pg_tail = svwhilelt_b32((uint32_t)i, (uint32_t)N);
        svfloat32_t a = svld1_f32(pg_tail, inp1 + i);
        svfloat32_t b = svld1_f32(pg_tail, inp2 + i);
        svfloat32_t result = svadd_f32_x(pg_tail, a, b);
        svst1_f32(pg_tail, out + i, result);
    }
}

// ----------------------------------------------------------------------------
// SVE-optimized attention forward
// inp is (B, T, 3C) holding Q, K, V vectors
// preatt, att are (B, NH, T, T) for pre/post-attention scores
// out is (B, T, C)

static inline void attention_forward_sve(float* out, float* preatt, float* att,
                                          const float* inp,
                                          int B, int T, int C, int NH) {
    int C3 = C * 3;
    int hs = C / NH;  // head size
    float scale = 1.0f / sqrtf((float)hs);

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
                float* att_bth = att + b * NH * T * T + h * T * T + t * T;

                svbool_t pg = svptrue_b32();
                uint64_t vl = svcntw();

                // Pass 1: calculate query dot key and find maxval
                float maxval = -1e10f;

                for (int t2 = 0; t2 <= t; t2++) {
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;

                    // SVE vectorized dot product: query_t . key_t2
                    svfloat32_t acc = svdup_f32(0.0f);
                    int i = 0;

                    for (; i + (int)vl <= hs; i += (int)vl) {
                        svfloat32_t q_vec = svld1_f32(pg, query_t + i);
                        svfloat32_t k_vec = svld1_f32(pg, key_t2 + i);
                        acc = svmla_f32_x(pg, acc, q_vec, k_vec);
                    }
                    float val = svaddv_f32(pg, acc);

                    // Handle remaining elements with predicated load
                    if (i < hs) {
                        svbool_t pg_tail = svwhilelt_b32((uint32_t)i, (uint32_t)hs);
                        svfloat32_t q_vec = svld1_f32(pg_tail, query_t + i);
                        svfloat32_t k_vec = svld1_f32(pg_tail, key_t2 + i);
                        svfloat32_t prod = svmul_f32_x(pg_tail, q_vec, k_vec);
                        val += svaddv_f32(pg_tail, prod);
                    }

                    val *= scale;
                    preatt_bth[t2] = val;
                    if (val > maxval) {
                        maxval = val;
                    }
                }

                // Pass 2: calculate exp and sum (using vectorized libm expf)
                float expsum = 0.0f;

                // Process in SVE vector-sized chunks
                int t2 = 0;
                for (; t2 + (int)vl <= t + 1; t2 += (int)vl) {
                    svfloat32_t preatt_vec = svld1_f32(pg, preatt_bth + t2);
                    svfloat32_t shifted = svsub_f32_x(pg, preatt_vec, svdup_f32(maxval));

                    // Compute exp element-wise using libm (store, compute, load back)
                    float shifted_arr[SVE_MAX_FLOATS];
                    float exp_arr[SVE_MAX_FLOATS];
                    svst1_f32(pg, shifted_arr, shifted);

                    for (int k = 0; k < (int)vl; k++) {
                        exp_arr[k] = expf(shifted_arr[k]);
                        expsum += exp_arr[k];
                    }

                    svfloat32_t exp_vec = svld1_f32(pg, exp_arr);
                    svst1_f32(pg, att_bth + t2, exp_vec);
                }

                // Handle remaining elements
                if (t2 <= t) {
                    svbool_t pg_tail = svwhilelt_b32((uint32_t)t2, (uint32_t)(t + 1));
                    svfloat32_t preatt_vec = svld1_f32(pg_tail, preatt_bth + t2);
                    svfloat32_t shifted = svsub_f32_x(pg_tail, preatt_vec, svdup_f32(maxval));

                    float shifted_arr[SVE_MAX_FLOATS];
                    float exp_arr[SVE_MAX_FLOATS];
                    svst1_f32(pg_tail, shifted_arr, shifted);

                    for (int k = 0; k < t + 1 - t2; k++) {
                        exp_arr[k] = expf(shifted_arr[k]);
                        expsum += exp_arr[k];
                    }

                    svfloat32_t exp_vec = svld1_f32(pg_tail, exp_arr);
                    svst1_f32(pg_tail, att_bth + t2, exp_vec);
                }

                // Pass 3: normalize (softmax)
                float expsum_inv = (expsum == 0.0f) ? 0.0f : 1.0f / expsum;
                svfloat32_t inv_vec = svdup_f32(expsum_inv);

                t2 = 0;
                for (; t2 + (int)vl <= t + 1; t2 += (int)vl) {
                    svfloat32_t att_vec = svld1_f32(pg, att_bth + t2);
                    svfloat32_t norm_vec = svmul_f32_x(pg, att_vec, inv_vec);
                    svst1_f32(pg, att_bth + t2, norm_vec);
                }

                // Handle remaining
                if (t2 <= t) {
                    svbool_t pg_tail = svwhilelt_b32((uint32_t)t2, (uint32_t)(t + 1));
                    svfloat32_t att_vec = svld1_f32(pg_tail, att_bth + t2);
                    svfloat32_t norm_vec = svmul_f32_x(pg_tail, att_vec, inv_vec);
                    svst1_f32(pg_tail, att_bth + t2, norm_vec);
                }

                // Zero out causal mask positions (t2 > t)
                svfloat32_t zero_vec = svdup_f32(0.0f);
                for (t2 = t + 1; t2 + (int)vl <= T; t2 += (int)vl) {
                    svst1_f32(pg, att_bth + t2, zero_vec);
                }
                if (t2 < T) {
                    svbool_t pg_tail = svwhilelt_b32((uint32_t)t2, (uint32_t)T);
                    svst1_f32(pg_tail, att_bth + t2, zero_vec);
                }

                // Pass 4: accumulate weighted values into output
                float* out_bth = out + b * T * C + t * C + h * hs;

                // Initialize output to zero using SVE
                int i = 0;
                for (; i + (int)vl <= hs; i += (int)vl) {
                    svst1_f32(pg, out_bth + i, zero_vec);
                }
                if (i < hs) {
                    svbool_t pg_tail = svwhilelt_b32((uint32_t)i, (uint32_t)hs);
                    svst1_f32(pg_tail, out_bth + i, zero_vec);
                }

                // Weighted sum of values
                for (t2 = 0; t2 <= t; t2++) {
                    const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2;
                    float att_weight = att_bth[t2];
                    svfloat32_t att_scalar = svdup_f32(att_weight);

                    i = 0;
                    for (; i + (int)vl <= hs; i += (int)vl) {
                        svfloat32_t out_vec = svld1_f32(pg, out_bth + i);
                        svfloat32_t val_vec = svld1_f32(pg, value_t2 + i);
                        out_vec = svmla_f32_x(pg, out_vec, val_vec, att_scalar);
                        svst1_f32(pg, out_bth + i, out_vec);
                    }

                    // Handle remaining
                    if (i < hs) {
                        svbool_t pg_tail = svwhilelt_b32((uint32_t)i, (uint32_t)hs);
                        svfloat32_t out_vec = svld1_f32(pg_tail, out_bth + i);
                        svfloat32_t val_vec = svld1_f32(pg_tail, value_t2 + i);
                        out_vec = svmla_f32_x(pg_tail, out_vec, val_vec, att_scalar);
                        svst1_f32(pg_tail, out_bth + i, out_vec);
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// SVE-optimized softmax forward

static inline void softmax_forward_sve(float* probs, const float* logits, int B, int T, int V, int Vp) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* logits_bt = logits + b * T * Vp + t * Vp;
            float* probs_bt = probs + b * T * Vp + t * Vp;

            svbool_t pg = svptrue_b32();
            uint64_t vl = svcntw();

            // Find max using SVE
            svfloat32_t max_vec = svdup_f32(-1e10f);
            int i = 0;

            for (; i + (int)vl <= V; i += (int)vl) {
                svfloat32_t logits_vec = svld1_f32(pg, logits_bt + i);
                max_vec = svmax_f32_x(pg, max_vec, logits_vec);
            }
            float maxval = svmaxv_f32(pg, max_vec);

            // Handle remaining
            if (i < V) {
                svbool_t pg_tail = svwhilelt_b32((uint32_t)i, (uint32_t)V);
                svfloat32_t logits_vec = svld1_f32(pg_tail, logits_bt + i);
                float tail_max = svmaxv_f32(pg_tail, logits_vec);
                if (tail_max > maxval) maxval = tail_max;
            }

            // Compute exp and sum - need scalar exp fallback
            float sum = 0.0f;
            svfloat32_t max_scalar = svdup_f32(maxval);

            // Process in chunks for exp computation
            for (i = 0; i < V; i += SVE_MAX_FLOATS) {
                int remaining = V - i;
                int chunk = (remaining < SVE_MAX_FLOATS) ? remaining : SVE_MAX_FLOATS;

                float shifted_vals[SVE_MAX_FLOATS];
                float exp_vals[SVE_MAX_FLOATS];

                svbool_t pg_chunk = svwhilelt_b32((uint32_t)0, (uint32_t)chunk);
                svfloat32_t logits_vec = svld1_f32(pg_chunk, logits_bt + i);
                svfloat32_t shifted = svsub_f32_x(pg_chunk, logits_vec, max_scalar);
                svst1_f32(pg_chunk, shifted_vals, shifted);

                // Scalar exp computation
                for (int j = 0; j < chunk; j++) {
                    exp_vals[j] = expf(shifted_vals[j]);
                    sum += exp_vals[j];
                }

                // Store exp values to probs
                svfloat32_t exp_vec = svld1_f32(pg_chunk, exp_vals);
                svst1_f32(pg_chunk, probs_bt + i, exp_vec);
            }

            // Normalize
            float sum_inv = 1.0f / sum;
            svfloat32_t sum_inv_vec = svdup_f32(sum_inv);

            i = 0;
            for (; i + (int)vl <= V; i += (int)vl) {
                svfloat32_t p_vec = svld1_f32(pg, probs_bt + i);
                svfloat32_t result = svmul_f32_x(pg, p_vec, sum_inv_vec);
                svst1_f32(pg, probs_bt + i, result);
            }

            // Handle remaining normalization
            if (i < V) {
                svbool_t pg_tail = svwhilelt_b32((uint32_t)i, (uint32_t)V);
                svfloat32_t p_vec = svld1_f32(pg_tail, probs_bt + i);
                svfloat32_t result = svmul_f32_x(pg_tail, p_vec, sum_inv_vec);
                svst1_f32(pg_tail, probs_bt + i, result);
            }

            // Zero out padding
            for (i = V; i < Vp; i++) {
                probs_bt[i] = 0.0f;
            }
        }
    }
}

#endif // USE_SVE
#endif // SVE_KERNELS_H
