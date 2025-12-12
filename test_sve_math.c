/*
 Benchmark and accuracy test for SVE math functions on A64FX

 Compile:
   make test_sve_math TARGET=a64fx_native

 Run:
   OMP_NUM_THREADS=48 ./test_sve_math
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define SVE_MATH_PROFILE
#define USE_SVE
#include "llmc/sve_math.h"

// ----------------------------------------------------------------------------
// Reference implementations using standard libm

static void ref_exp_f32(float* out, const float* in, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = expf(in[i]);
    }
}

static void ref_log_f32(float* out, const float* in, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = logf(in[i]);
    }
}

static void ref_sin_f32(float* out, const float* in, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = sinf(in[i]);
    }
}

static void ref_cos_f32(float* out, const float* in, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = cosf(in[i]);
    }
}

static void ref_tan_f32(float* out, const float* in, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = tanf(in[i]);
    }
}

static void ref_tanh_f32(float* out, const float* in, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = tanhf(in[i]);
    }
}

// ----------------------------------------------------------------------------
// Utility functions

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void compute_error_stats(const float* ref, const float* test, int n,
                                 double* max_abs_err, double* max_rel_err,
                                 double* avg_abs_err, double* avg_rel_err) {
    *max_abs_err = 0.0;
    *max_rel_err = 0.0;
    *avg_abs_err = 0.0;
    *avg_rel_err = 0.0;

    int valid_count = 0;

    for (int i = 0; i < n; i++) {
        // Skip NaN/Inf comparisons
        if (!isfinite(ref[i]) || !isfinite(test[i])) {
            continue;
        }

        double abs_err = fabs((double)test[i] - (double)ref[i]);
        double rel_err = (fabs(ref[i]) > 1e-10) ? abs_err / fabs(ref[i]) : abs_err;

        if (abs_err > *max_abs_err) *max_abs_err = abs_err;
        if (rel_err > *max_rel_err) *max_rel_err = rel_err;

        *avg_abs_err += abs_err;
        *avg_rel_err += rel_err;
        valid_count++;
    }

    if (valid_count > 0) {
        *avg_abs_err /= valid_count;
        *avg_rel_err /= valid_count;
    }
}

static void fill_random(float* arr, int n, float min_val, float max_val) {
    for (int i = 0; i < n; i++) {
        arr[i] = min_val + (max_val - min_val) * ((float)rand() / RAND_MAX);
    }
}

static void fill_positive_random(float* arr, int n, float min_val, float max_val) {
    for (int i = 0; i < n; i++) {
        arr[i] = min_val + (max_val - min_val) * ((float)rand() / RAND_MAX);
        if (arr[i] <= 0) arr[i] = 0.001f;  // Ensure positive for log
    }
}

// ----------------------------------------------------------------------------
// Benchmark functions

typedef void (*math_func_t)(float*, const float*, int);

typedef struct {
    const char* name;
    math_func_t sve_func;
    math_func_t ref_func;
    float input_min;
    float input_max;
    int positive_only;  // For log
} benchmark_config_t;

static void run_benchmark(benchmark_config_t* config, int n, int warmup_iters, int bench_iters) {
    printf("\n========================================\n");
    printf("Benchmarking: %s\n", config->name);
    printf("N = %d, warmup = %d, iterations = %d\n", n, warmup_iters, bench_iters);
    printf("========================================\n");

    // Allocate arrays
    float* input = (float*)malloc(n * sizeof(float));
    float* output_sve = (float*)malloc(n * sizeof(float));
    float* output_ref = (float*)malloc(n * sizeof(float));

    if (!input || !output_sve || !output_ref) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Fill input
    if (config->positive_only) {
        fill_positive_random(input, n, config->input_min, config->input_max);
    } else {
        fill_random(input, n, config->input_min, config->input_max);
    }

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        config->sve_func(output_sve, input, n);
        config->ref_func(output_ref, input, n);
    }

    // Benchmark SVE
    sve_math_reset_stats();
    double sve_start = get_time_ms();
    for (int i = 0; i < bench_iters; i++) {
        config->sve_func(output_sve, input, n);
    }
    double sve_end = get_time_ms();
    double sve_time = sve_end - sve_start;

    // Benchmark reference
    double ref_start = get_time_ms();
    for (int i = 0; i < bench_iters; i++) {
        config->ref_func(output_ref, input, n);
    }
    double ref_end = get_time_ms();
    double ref_time = ref_end - ref_start;

    // Compute accuracy
    config->sve_func(output_sve, input, n);
    config->ref_func(output_ref, input, n);

    double max_abs_err, max_rel_err, avg_abs_err, avg_rel_err;
    compute_error_stats(output_ref, output_sve, n,
                        &max_abs_err, &max_rel_err, &avg_abs_err, &avg_rel_err);

    // Print results
    printf("\nPerformance:\n");
    printf("  SVE:       %8.2f ms total, %8.3f ns/elem\n",
           sve_time, sve_time * 1e6 / (bench_iters * (double)n));
    printf("  Reference: %8.2f ms total, %8.3f ns/elem\n",
           ref_time, ref_time * 1e6 / (bench_iters * (double)n));
    printf("  Speedup:   %.2fx\n", ref_time / sve_time);

    printf("\nAccuracy:\n");
    printf("  Max absolute error: %e\n", max_abs_err);
    printf("  Max relative error: %e\n", max_rel_err);
    printf("  Avg absolute error: %e\n", avg_abs_err);
    printf("  Avg relative error: %e\n", avg_rel_err);

    // Print SVE profiling stats
    printf("\nSVE Profiling:\n");
    sve_math_print_stats();

    free(input);
    free(output_sve);
    free(output_ref);
}

// ----------------------------------------------------------------------------
// Main

int main(int argc, char** argv) {
    printf("SVE Math Functions Benchmark\n");
    printf("============================\n\n");

    // Parse command line
    int n = 1000000;  // Default: 1M elements
    int warmup = 5;
    int iters = 100;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [-n elements] [-w warmup] [-i iterations]\n", argv[0]);
            return 0;
        }
    }

    printf("Configuration:\n");
    printf("  Elements:   %d\n", n);
    printf("  Warmup:     %d iterations\n", warmup);
    printf("  Benchmark:  %d iterations\n", iters);

    // Get SVE vector length
    uint64_t vl = svcntw();
    printf("  SVE VL:     %lu floats (%lu bits)\n", vl, vl * 32);

    srand(42);  // Fixed seed for reproducibility

    // Define benchmarks
    benchmark_config_t benchmarks[] = {
        {"exp",  sve_exp_f32_array,  ref_exp_f32,  -10.0f, 10.0f, 0},
        {"log",  sve_log_f32_array,  ref_log_f32,  0.001f, 1000.0f, 1},
        {"sin",  sve_sin_f32_array,  ref_sin_f32,  -10.0f, 10.0f, 0},
        {"cos",  sve_cos_f32_array,  ref_cos_f32,  -10.0f, 10.0f, 0},
        {"tan",  sve_tan_f32_array,  ref_tan_f32,  -1.5f, 1.5f, 0},  // Avoid poles
        {"tanh", sve_tanh_f32_array, ref_tanh_f32, -5.0f, 5.0f, 0},
    };

    int num_benchmarks = sizeof(benchmarks) / sizeof(benchmarks[0]);

    for (int i = 0; i < num_benchmarks; i++) {
        run_benchmark(&benchmarks[i], n, warmup, iters);
    }

    printf("\n========================================\n");
    printf("All benchmarks complete!\n");
    printf("========================================\n");

    return 0;
}
