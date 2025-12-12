/*
 * SVE Kernels Validation and Performance Test
 * ============================================
 * Compares SVE-optimized kernels against reference (scalar) implementations.
 *
 * Build:
 *   make test_sve_kernels TARGET=a64fx_fcc        # SVE version
 *   make test_sve_kernels TARGET=a64fx_fcc SVE=0  # Reference version
 *
 * Usage:
 *   ./test_sve_kernels [options]
 *   Options:
 *     -v, --validate   Run validation mode (check correctness)
 *     -p, --perf       Run performance mode (benchmark)
 *     -a, --all        Run both validation and performance (default)
 *     -i <n>           Number of iterations for perf mode (default: 5)
 *     -B <n>           Batch size (default: 4)
 *     -T <n>           Sequence length (default: 64)
 *     -C <n>           Channels (default: 768)
 *     -H <n>           Number of heads (default: 12)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#ifdef OMP
#include <omp.h>
#endif

// ----------------------------------------------------------------------------
// Timer utilities

static inline double get_time_sec(void) {
#ifdef OMP
    return omp_get_wtime();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

// ----------------------------------------------------------------------------
// Reference implementations (scalar, no SVE)
// These are copied from train_gpt2.c without SVE

static void encoder_forward_ref(float* out,
                                const int* inp, const float* wte, const float* wpe,
                                int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * C + t * C;
            int ix = inp[b * T + t];
            const float* wte_ix = wte + ix * C;
            const float* wpe_t = wpe + t * C;
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

static void layernorm_forward_ref(float* out, float* mean, float* rstd,
                                  const float* inp, const float* weight, const float* bias,
                                  int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* x = inp + b * T * C + t * C;
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m / C;
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v / C;
            float s = 1.0f / sqrtf(v + eps);
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m));
                float o = n * weight[i] + bias[i];
                out_bt[i] = o;
            }
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

static void attention_forward_ref(float* out, float* preatt, float* att,
                                  const float* inp,
                                  int B, int T, int C, int NH) {
    int C3 = C * 3;
    int hs = C / NH;
    float scale = 1.0f / sqrtf((float)hs);

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
                float* att_bth = att + b * NH * T * T + h * T * T + t * T;

                float maxval = -10000.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }
                    preatt_bth[t2] = val;
                }

                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        att_bth[t2] = 0.0f;
                    }
                }

                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2;
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

static void matmul_forward_ref(float* out,
                               const float* inp, const float* weight, const float* bias,
                               int B, int T, int C, int OC) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            const float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                const float* wrow = weight + o * C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

static void gelu_forward_ref(float* out, const float* inp, int N) {
    const float gelu_scale = 0.7978845608f;  // sqrt(2/pi)
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(gelu_scale * (x + cube)));
    }
}

static void residual_forward_ref(float* out, const float* inp1, const float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

static void softmax_forward_ref(float* probs, const float* logits, int B, int T, int V) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* logits_bt = logits + b * T * V + t * V;
            float* probs_bt = probs + b * T * V + t * V;

            float maxval = -10000.0f;
            for (int i = 0; i < V; i++) {
                if (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            for (int i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// SVE implementations (included when USE_SVE is defined)

#ifdef USE_SVE
#include "llmc/sve_kernels.h"

// Wrapper functions to match reference API
static void encoder_forward_sve_wrapper(float* out,
                                        const int* inp, const float* wte, const float* wpe,
                                        int B, int T, int C) {
    encoder_forward_sve(out, inp, wte, wpe, B, T, C);
}

static void layernorm_forward_sve_wrapper(float* out, float* mean, float* rstd,
                                          const float* inp, const float* weight, const float* bias,
                                          int B, int T, int C) {
    layernorm_forward_sve(out, mean, rstd, inp, weight, bias, B, T, C);
}

static void attention_forward_sve_wrapper(float* out, float* preatt, float* att,
                                          const float* inp,
                                          int B, int T, int C, int NH) {
    attention_forward_sve(out, preatt, att, inp, B, T, C, NH);
}

static void matmul_forward_sve_wrapper(float* out,
                                       const float* inp, const float* weight, const float* bias,
                                       int B, int T, int C, int OC) {
    matmul_forward_sve_blocked(out, inp, weight, bias, B * T, C, OC);
}

static void gelu_forward_sve_wrapper(float* out, const float* inp, int N) {
    gelu_forward_sve(out, inp, N);
}

static void residual_forward_sve_wrapper(float* out, const float* inp1, const float* inp2, int N) {
    residual_forward_sve(out, inp1, inp2, N);
}

static void softmax_forward_sve_wrapper(float* probs, const float* logits, int B, int T, int V) {
    softmax_forward_sve(probs, logits, B, T, V, V);
}
#endif

// ----------------------------------------------------------------------------
// Random initialization

static void random_init(float* arr, int n, float scale) {
    for (int i = 0; i < n; i++) {
        arr[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

static void random_init_int(int* arr, int n, int max_val) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % max_val;
    }
}

// ----------------------------------------------------------------------------
// Validation utilities

static int validate_arrays(const float* ref, const float* test, int n,
                           float abs_tol, float rel_tol,
                           float* max_abs_err, float* max_rel_err) {
    *max_abs_err = 0.0f;
    *max_rel_err = 0.0f;
    int errors = 0;

    for (int i = 0; i < n; i++) {
        float abs_err = fabsf(ref[i] - test[i]);
        float rel_err = (fabsf(ref[i]) > 1e-6f) ? abs_err / fabsf(ref[i]) : abs_err;

        if (abs_err > *max_abs_err) *max_abs_err = abs_err;
        if (rel_err > *max_rel_err) *max_rel_err = rel_err;

        if (abs_err > abs_tol && rel_err > rel_tol) {
            if (errors < 5) {
                printf("  Mismatch at [%d]: ref=%.6f, test=%.6f, abs_err=%.2e, rel_err=%.2e\n",
                       i, ref[i], test[i], abs_err, rel_err);
            }
            errors++;
        }
    }
    return errors;
}

// ----------------------------------------------------------------------------
// Test functions

typedef struct {
    const char* name;
    int passed;
    double ref_time_ms;
    double sve_time_ms;
    double speedup;
    float max_abs_err;
    float max_rel_err;
} TestResult;

static TestResult test_encoder(int B, int T, int C, int V, int iterations, int validate) {
    TestResult result = {.name = "encoder_forward"};

    int inp_size = B * T;
    int wte_size = V * C;
    int wpe_size = T * C;
    int out_size = B * T * C;

    int* inp = (int*)malloc(inp_size * sizeof(int));
    float* wte = (float*)malloc(wte_size * sizeof(float));
    float* wpe = (float*)malloc(wpe_size * sizeof(float));
    float* out_ref = (float*)malloc(out_size * sizeof(float));
    float* out_sve = (float*)malloc(out_size * sizeof(float));

    random_init_int(inp, inp_size, V);
    random_init(wte, wte_size, 1.0f);
    random_init(wpe, wpe_size, 1.0f);

    // Warmup & reference
    encoder_forward_ref(out_ref, inp, wte, wpe, B, T, C);

    double t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        encoder_forward_ref(out_ref, inp, wte, wpe, B, T, C);
    }
    result.ref_time_ms = (get_time_sec() - t0) * 1000.0 / iterations;

#ifdef USE_SVE
    // Warmup & SVE
    encoder_forward_sve_wrapper(out_sve, inp, wte, wpe, B, T, C);

    t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        encoder_forward_sve_wrapper(out_sve, inp, wte, wpe, B, T, C);
    }
    result.sve_time_ms = (get_time_sec() - t0) * 1000.0 / iterations;
    result.speedup = result.ref_time_ms / result.sve_time_ms;

    if (validate) {
        int errors = validate_arrays(out_ref, out_sve, out_size, 1e-5f, 1e-4f,
                                     &result.max_abs_err, &result.max_rel_err);
        result.passed = (errors == 0);
    } else {
        result.passed = 1;
    }
#else
    result.sve_time_ms = result.ref_time_ms;
    result.speedup = 1.0;
    result.passed = 1;
#endif

    free(inp); free(wte); free(wpe); free(out_ref); free(out_sve);
    return result;
}

static TestResult test_layernorm(int B, int T, int C, int iterations, int validate) {
    TestResult result = {.name = "layernorm_forward"};

    int inp_size = B * T * C;
    int out_size = B * T * C;
    int stats_size = B * T;

    float* inp = (float*)malloc(inp_size * sizeof(float));
    float* weight = (float*)malloc(C * sizeof(float));
    float* bias = (float*)malloc(C * sizeof(float));
    float* out_ref = (float*)malloc(out_size * sizeof(float));
    float* out_sve = (float*)malloc(out_size * sizeof(float));
    float* mean_ref = (float*)malloc(stats_size * sizeof(float));
    float* mean_sve = (float*)malloc(stats_size * sizeof(float));
    float* rstd_ref = (float*)malloc(stats_size * sizeof(float));
    float* rstd_sve = (float*)malloc(stats_size * sizeof(float));

    random_init(inp, inp_size, 1.0f);
    random_init(weight, C, 1.0f);
    random_init(bias, C, 0.1f);

    // Warmup & reference
    layernorm_forward_ref(out_ref, mean_ref, rstd_ref, inp, weight, bias, B, T, C);

    double t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        layernorm_forward_ref(out_ref, mean_ref, rstd_ref, inp, weight, bias, B, T, C);
    }
    result.ref_time_ms = (get_time_sec() - t0) * 1000.0 / iterations;

#ifdef USE_SVE
    layernorm_forward_sve_wrapper(out_sve, mean_sve, rstd_sve, inp, weight, bias, B, T, C);

    t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        layernorm_forward_sve_wrapper(out_sve, mean_sve, rstd_sve, inp, weight, bias, B, T, C);
    }
    result.sve_time_ms = (get_time_sec() - t0) * 1000.0 / iterations;
    result.speedup = result.ref_time_ms / result.sve_time_ms;

    if (validate) {
        int errors = validate_arrays(out_ref, out_sve, out_size, 1e-4f, 1e-3f,
                                     &result.max_abs_err, &result.max_rel_err);
        result.passed = (errors == 0);
    } else {
        result.passed = 1;
    }
#else
    result.sve_time_ms = result.ref_time_ms;
    result.speedup = 1.0;
    result.passed = 1;
#endif

    free(inp); free(weight); free(bias);
    free(out_ref); free(out_sve);
    free(mean_ref); free(mean_sve); free(rstd_ref); free(rstd_sve);
    return result;
}

static TestResult test_attention(int B, int T, int C, int NH, int iterations, int validate) {
    TestResult result = {.name = "attention_forward"};

    int C3 = C * 3;
    int inp_size = B * T * C3;
    int att_size = B * NH * T * T;
    int out_size = B * T * C;

    float* inp = (float*)malloc(inp_size * sizeof(float));
    float* preatt_ref = (float*)malloc(att_size * sizeof(float));
    float* preatt_sve = (float*)malloc(att_size * sizeof(float));
    float* att_ref = (float*)malloc(att_size * sizeof(float));
    float* att_sve = (float*)malloc(att_size * sizeof(float));
    float* out_ref = (float*)malloc(out_size * sizeof(float));
    float* out_sve = (float*)malloc(out_size * sizeof(float));

    random_init(inp, inp_size, 0.5f);

    // Warmup & reference
    attention_forward_ref(out_ref, preatt_ref, att_ref, inp, B, T, C, NH);

    double t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        attention_forward_ref(out_ref, preatt_ref, att_ref, inp, B, T, C, NH);
    }
    result.ref_time_ms = (get_time_sec() - t0) * 1000.0 / iterations;

#ifdef USE_SVE
    attention_forward_sve_wrapper(out_sve, preatt_sve, att_sve, inp, B, T, C, NH);

    t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        attention_forward_sve_wrapper(out_sve, preatt_sve, att_sve, inp, B, T, C, NH);
    }
    result.sve_time_ms = (get_time_sec() - t0) * 1000.0 / iterations;
    result.speedup = result.ref_time_ms / result.sve_time_ms;

    if (validate) {
        int errors = validate_arrays(out_ref, out_sve, out_size, 1e-4f, 1e-3f,
                                     &result.max_abs_err, &result.max_rel_err);
        result.passed = (errors == 0);
    } else {
        result.passed = 1;
    }
#else
    result.sve_time_ms = result.ref_time_ms;
    result.speedup = 1.0;
    result.passed = 1;
#endif

    free(inp);
    free(preatt_ref); free(preatt_sve);
    free(att_ref); free(att_sve);
    free(out_ref); free(out_sve);
    return result;
}

static TestResult test_matmul(int B, int T, int C, int OC, int iterations, int validate) {
    TestResult result = {.name = "matmul_forward"};

    int inp_size = B * T * C;
    int weight_size = OC * C;
    int out_size = B * T * OC;

    float* inp = (float*)malloc(inp_size * sizeof(float));
    float* weight = (float*)malloc(weight_size * sizeof(float));
    float* bias = (float*)malloc(OC * sizeof(float));
    float* out_ref = (float*)malloc(out_size * sizeof(float));
    float* out_sve = (float*)malloc(out_size * sizeof(float));

    random_init(inp, inp_size, 1.0f);
    random_init(weight, weight_size, 0.1f);
    random_init(bias, OC, 0.1f);

    // Warmup & reference
    matmul_forward_ref(out_ref, inp, weight, bias, B, T, C, OC);

    double t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        matmul_forward_ref(out_ref, inp, weight, bias, B, T, C, OC);
    }
    result.ref_time_ms = (get_time_sec() - t0) * 1000.0 / iterations;

#ifdef USE_SVE
    matmul_forward_sve_wrapper(out_sve, inp, weight, bias, B, T, C, OC);

    t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        matmul_forward_sve_wrapper(out_sve, inp, weight, bias, B, T, C, OC);
    }
    result.sve_time_ms = (get_time_sec() - t0) * 1000.0 / iterations;
    result.speedup = result.ref_time_ms / result.sve_time_ms;

    if (validate) {
        int errors = validate_arrays(out_ref, out_sve, out_size, 1e-3f, 1e-2f,
                                     &result.max_abs_err, &result.max_rel_err);
        result.passed = (errors == 0);
    } else {
        result.passed = 1;
    }
#else
    result.sve_time_ms = result.ref_time_ms;
    result.speedup = 1.0;
    result.passed = 1;
#endif

    free(inp); free(weight); free(bias);
    free(out_ref); free(out_sve);
    return result;
}

static TestResult test_gelu(int N, int iterations, int validate) {
    TestResult result = {.name = "gelu_forward"};

    float* inp = (float*)malloc(N * sizeof(float));
    float* out_ref = (float*)malloc(N * sizeof(float));
    float* out_sve = (float*)malloc(N * sizeof(float));

    random_init(inp, N, 2.0f);

    // Warmup & reference
    gelu_forward_ref(out_ref, inp, N);

    double t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        gelu_forward_ref(out_ref, inp, N);
    }
    result.ref_time_ms = (get_time_sec() - t0) * 1000.0 / iterations;

#ifdef USE_SVE
    gelu_forward_sve_wrapper(out_sve, inp, N);

    t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        gelu_forward_sve_wrapper(out_sve, inp, N);
    }
    result.sve_time_ms = (get_time_sec() - t0) * 1000.0 / iterations;
    result.speedup = result.ref_time_ms / result.sve_time_ms;

    if (validate) {
        int errors = validate_arrays(out_ref, out_sve, N, 1e-4f, 1e-3f,
                                     &result.max_abs_err, &result.max_rel_err);
        result.passed = (errors == 0);
    } else {
        result.passed = 1;
    }
#else
    result.sve_time_ms = result.ref_time_ms;
    result.speedup = 1.0;
    result.passed = 1;
#endif

    free(inp); free(out_ref); free(out_sve);
    return result;
}

static TestResult test_residual(int N, int iterations, int validate) {
    TestResult result = {.name = "residual_forward"};

    float* inp1 = (float*)malloc(N * sizeof(float));
    float* inp2 = (float*)malloc(N * sizeof(float));
    float* out_ref = (float*)malloc(N * sizeof(float));
    float* out_sve = (float*)malloc(N * sizeof(float));

    random_init(inp1, N, 1.0f);
    random_init(inp2, N, 1.0f);

    // Warmup & reference
    residual_forward_ref(out_ref, inp1, inp2, N);

    double t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        residual_forward_ref(out_ref, inp1, inp2, N);
    }
    result.ref_time_ms = (get_time_sec() - t0) * 1000.0 / iterations;

#ifdef USE_SVE
    residual_forward_sve_wrapper(out_sve, inp1, inp2, N);

    t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        residual_forward_sve_wrapper(out_sve, inp1, inp2, N);
    }
    result.sve_time_ms = (get_time_sec() - t0) * 1000.0 / iterations;
    result.speedup = result.ref_time_ms / result.sve_time_ms;

    if (validate) {
        int errors = validate_arrays(out_ref, out_sve, N, 1e-6f, 1e-5f,
                                     &result.max_abs_err, &result.max_rel_err);
        result.passed = (errors == 0);
    } else {
        result.passed = 1;
    }
#else
    result.sve_time_ms = result.ref_time_ms;
    result.speedup = 1.0;
    result.passed = 1;
#endif

    free(inp1); free(inp2); free(out_ref); free(out_sve);
    return result;
}

static TestResult test_softmax(int B, int T, int V, int iterations, int validate) {
    TestResult result = {.name = "softmax_forward"};

    int size = B * T * V;

    float* logits = (float*)malloc(size * sizeof(float));
    float* probs_ref = (float*)malloc(size * sizeof(float));
    float* probs_sve = (float*)malloc(size * sizeof(float));

    random_init(logits, size, 2.0f);

    // Warmup & reference
    softmax_forward_ref(probs_ref, logits, B, T, V);

    double t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        softmax_forward_ref(probs_ref, logits, B, T, V);
    }
    result.ref_time_ms = (get_time_sec() - t0) * 1000.0 / iterations;

#ifdef USE_SVE
    softmax_forward_sve_wrapper(probs_sve, logits, B, T, V);

    t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        softmax_forward_sve_wrapper(probs_sve, logits, B, T, V);
    }
    result.sve_time_ms = (get_time_sec() - t0) * 1000.0 / iterations;
    result.speedup = result.ref_time_ms / result.sve_time_ms;

    if (validate) {
        int errors = validate_arrays(probs_ref, probs_sve, size, 1e-4f, 1e-3f,
                                     &result.max_abs_err, &result.max_rel_err);
        result.passed = (errors == 0);
    } else {
        result.passed = 1;
    }
#else
    result.sve_time_ms = result.ref_time_ms;
    result.speedup = 1.0;
    result.passed = 1;
#endif

    free(logits); free(probs_ref); free(probs_sve);
    return result;
}

// ----------------------------------------------------------------------------
// Main

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -v, --validate   Run validation mode (check correctness)\n");
    printf("  -p, --perf       Run performance mode (benchmark)\n");
    printf("  -a, --all        Run both validation and performance (default)\n");
    printf("  -i <n>           Number of iterations for perf mode (default: 5)\n");
    printf("  -B <n>           Batch size (default: 4)\n");
    printf("  -T <n>           Sequence length (default: 64)\n");
    printf("  -C <n>           Channels (default: 768)\n");
    printf("  -H <n>           Number of heads (default: 12)\n");
}

int main(int argc, char** argv) {
    // Default parameters
    int B = 4;      // batch size
    int T = 64;     // sequence length
    int C = 768;    // channels
    int NH = 12;    // number of heads
    int V = 50257;  // vocab size
    int iterations = 5;
    int run_validate = 1;
    int run_perf = 1;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--validate") == 0) {
            run_validate = 1;
            run_perf = 0;
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--perf") == 0) {
            run_validate = 0;
            run_perf = 1;
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--all") == 0) {
            run_validate = 1;
            run_perf = 1;
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-B") == 0 && i + 1 < argc) {
            B = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-T") == 0 && i + 1 < argc) {
            T = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-C") == 0 && i + 1 < argc) {
            C = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-H") == 0 && i + 1 < argc) {
            NH = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    srand(42);

    printf("SVE Kernels Validation & Performance Test\n");
    printf("==========================================\n");
#ifdef USE_SVE
    printf("Mode: SVE ENABLED\n");
#else
    printf("Mode: SVE DISABLED (reference only)\n");
#endif
    printf("\nConfiguration:\n");
    printf("  Batch size (B):     %d\n", B);
    printf("  Sequence length (T): %d\n", T);
    printf("  Channels (C):        %d\n", C);
    printf("  Heads (NH):          %d\n", NH);
    printf("  Vocab size (V):      %d\n", V);
    printf("  Iterations:          %d\n", iterations);
    printf("  Validate: %s, Perf: %s\n",
           run_validate ? "yes" : "no",
           run_perf ? "yes" : "no");
    printf("\n");

    // Run tests
    TestResult results[7];
    int num_results = 0;

    printf("Running tests...\n\n");

    results[num_results++] = test_encoder(B, T, C, V, iterations, run_validate);
    results[num_results++] = test_layernorm(B, T, C, iterations, run_validate);
    results[num_results++] = test_attention(B, T, C, NH, iterations, run_validate);
    results[num_results++] = test_matmul(B, T, C, C * 4, iterations, run_validate);
    results[num_results++] = test_gelu(B * T * C, iterations, run_validate);
    results[num_results++] = test_residual(B * T * C, iterations, run_validate);
    results[num_results++] = test_softmax(B, T, V, iterations, run_validate);

    // Print results
    printf("==========================================\n");
    printf("Results Summary\n");
    printf("==========================================\n\n");

    if (run_validate) {
        printf("Validation Results:\n");
        printf("%-20s %-8s %-12s %-12s\n", "Kernel", "Status", "Max Abs Err", "Max Rel Err");
        printf("------------------------------------------------------------\n");
        int all_passed = 1;
        for (int i = 0; i < num_results; i++) {
            printf("%-20s %-8s %.2e    %.2e\n",
                   results[i].name,
                   results[i].passed ? "PASS" : "FAIL",
                   results[i].max_abs_err,
                   results[i].max_rel_err);
            if (!results[i].passed) all_passed = 0;
        }
        printf("\nOverall: %s\n\n", all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    }

    if (run_perf) {
        printf("Performance Results:\n");
        printf("%-20s %10s %10s %10s\n", "Kernel", "Ref (ms)", "SVE (ms)", "Speedup");
        printf("------------------------------------------------------------\n");
        for (int i = 0; i < num_results; i++) {
            printf("%-20s %10.3f %10.3f %9.2fx\n",
                   results[i].name,
                   results[i].ref_time_ms,
                   results[i].sve_time_ms,
                   results[i].speedup);
        }
        printf("\n");
    }

    return 0;
}
