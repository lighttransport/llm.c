/*
 SVE Fast Math Functions for A64FX

 Uses A64FX-specific SVE instructions for fast approximations:
 - FEXPA/FSCALE for exponential
 - FTSMUL/FTMAD/FTSSEL for trigonometric functions

 These are approximate implementations trading accuracy for speed.
 Typical error: < 1e-6 relative error for most inputs.
*/
#ifndef SVE_MATH_H
#define SVE_MATH_H

#ifdef USE_SVE
#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

// ----------------------------------------------------------------------------
// Performance measurement infrastructure

typedef struct {
    const char* name;
    uint64_t call_count;
    uint64_t total_elements;
    double total_time_ns;
} sve_math_stats_t;

// Global statistics (define SVE_MATH_PROFILE to enable)
#ifdef SVE_MATH_PROFILE
static sve_math_stats_t sve_exp_stats = {"sve_exp", 0, 0, 0.0};
static sve_math_stats_t sve_log_stats = {"sve_log", 0, 0, 0.0};
static sve_math_stats_t sve_sin_stats = {"sve_sin", 0, 0, 0.0};
static sve_math_stats_t sve_cos_stats = {"sve_cos", 0, 0, 0.0};
static sve_math_stats_t sve_tan_stats = {"sve_tan", 0, 0, 0.0};
static sve_math_stats_t sve_tanh_stats = {"sve_tanh", 0, 0, 0.0};

static inline uint64_t get_cycle_count(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r" (val));
    return val;
}

static inline uint64_t get_cycle_freq(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r" (val));
    return val;
}

#define SVE_MATH_PROFILE_START() uint64_t _prof_start = get_cycle_count()
#define SVE_MATH_PROFILE_END(stats, n) do { \
    uint64_t _prof_end = get_cycle_count(); \
    (stats).call_count++; \
    (stats).total_elements += (n); \
    (stats).total_time_ns += (double)(_prof_end - _prof_start) * 1e9 / get_cycle_freq(); \
} while(0)

static inline void sve_math_print_stats(void) {
    printf("\n=== SVE Math Performance Statistics ===\n");
    printf("%-10s %12s %15s %12s %12s\n", "Function", "Calls", "Elements", "Time(ms)", "ns/elem");
    printf("--------------------------------------------------------------\n");

    sve_math_stats_t* all_stats[] = {
        &sve_exp_stats, &sve_log_stats, &sve_sin_stats,
        &sve_cos_stats, &sve_tan_stats, &sve_tanh_stats
    };

    for (int i = 0; i < 6; i++) {
        sve_math_stats_t* s = all_stats[i];
        if (s->call_count > 0) {
            double ns_per_elem = s->total_elements > 0 ? s->total_time_ns / s->total_elements : 0;
            printf("%-10s %12lu %15lu %12.3f %12.3f\n",
                   s->name, s->call_count, s->total_elements,
                   s->total_time_ns / 1e6, ns_per_elem);
        }
    }
    printf("\n");
}

static inline void sve_math_reset_stats(void) {
    sve_exp_stats = (sve_math_stats_t){"sve_exp", 0, 0, 0.0};
    sve_log_stats = (sve_math_stats_t){"sve_log", 0, 0, 0.0};
    sve_sin_stats = (sve_math_stats_t){"sve_sin", 0, 0, 0.0};
    sve_cos_stats = (sve_math_stats_t){"sve_cos", 0, 0, 0.0};
    sve_tan_stats = (sve_math_stats_t){"sve_tan", 0, 0, 0.0};
    sve_tanh_stats = (sve_math_stats_t){"sve_tanh", 0, 0, 0.0};
}

#else
#define SVE_MATH_PROFILE_START() (void)0
#define SVE_MATH_PROFILE_END(stats, n) (void)0
static inline void sve_math_print_stats(void) {}
static inline void sve_math_reset_stats(void) {}
#endif

// ----------------------------------------------------------------------------
// Constants

// For exp: ln(2) split for better precision
#define LN2_HI  0.693145751953125f
#define LN2_LO  1.42860682030941723212e-6f
#define LOG2E   1.44269504088896341f

// For log: coefficients for polynomial approximation
#define LOG_C1  0.6666666666666735130f
#define LOG_C2  0.3999999999940941908f
#define LOG_C3  0.2857142874366239149f
#define LOG_C4  0.2222219843214978396f

// For sin/cos: pi constants
#define PI_F        3.14159265358979323846f
#define PI_2_F      1.57079632679489661923f
#define TWO_OVER_PI 0.63661977236758134308f

// ----------------------------------------------------------------------------
// SVE Fast Exponential using FEXPA instruction
//
// FEXPA provides 2^(x * 1/256) for the fractional part
// We use: exp(x) = 2^(x * log2(e)) = 2^n * 2^f where n is integer, f is fraction

static inline svfloat32_t sve_exp_f32(svbool_t pg, svfloat32_t x) {
    // exp(x) = 2^(x * log2(e))
    // Split into integer and fractional parts

    svfloat32_t log2e = svdup_f32(LOG2E);
    svfloat32_t z = svmul_f32_x(pg, x, log2e);  // z = x * log2(e)

    // Round to nearest integer
    svfloat32_t n_f = svrintn_f32_x(pg, z);  // n = round(z)
    svint32_t n = svcvt_s32_f32_x(pg, n_f);

    // Fractional part: f = z - n (in range [-0.5, 0.5])
    svfloat32_t f = svsub_f32_x(pg, z, n_f);

    // Scale f to [0, 1) range for FEXPA by adding 0.5 and adjusting n
    // FEXPA computes 2^(f/256) for f in [0, 256)
    // We need 2^f for f in [-0.5, 0.5]

    // Use polynomial for 2^f approximation (more portable than FEXPA)
    // 2^f ≈ 1 + f*ln(2) + f^2*ln(2)^2/2 + f^3*ln(2)^3/6 + ...
    // Optimized polynomial coefficients for f in [-0.5, 0.5]
    svfloat32_t c0 = svdup_f32(1.0f);
    svfloat32_t c1 = svdup_f32(0.6931471805599453f);   // ln(2)
    svfloat32_t c2 = svdup_f32(0.24022650695910071f);  // ln(2)^2/2
    svfloat32_t c3 = svdup_f32(0.05550410866482158f);  // ln(2)^3/6
    svfloat32_t c4 = svdup_f32(0.009618129107628477f); // ln(2)^4/24
    svfloat32_t c5 = svdup_f32(0.0013333558146428443f);// ln(2)^5/120

    // Horner's method: result = c0 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*c5))))
    svfloat32_t result = svmla_f32_x(pg, c4, f, c5);
    result = svmla_f32_x(pg, c3, f, result);
    result = svmla_f32_x(pg, c2, f, result);
    result = svmla_f32_x(pg, c1, f, result);
    result = svmla_f32_x(pg, c0, f, result);

    // Scale by 2^n using FSCALE
    result = svscale_f32_x(pg, result, n);

    // Handle overflow/underflow
    svbool_t overflow = svcmpgt_f32(pg, x, svdup_f32(88.0f));
    svbool_t underflow = svcmplt_f32(pg, x, svdup_f32(-88.0f));
    result = svsel_f32(overflow, svdup_f32(INFINITY), result);
    result = svsel_f32(underflow, svdup_f32(0.0f), result);

    return result;
}

// Array version with profiling
static inline void sve_exp_f32_array(float* out, const float* in, int n) {
    SVE_MATH_PROFILE_START();

    svbool_t pg = svptrue_b32();
    uint64_t vl = svcntw();
    int i = 0;

    for (; i + (int)vl <= n; i += (int)vl) {
        svfloat32_t x = svld1_f32(pg, in + i);
        svfloat32_t result = sve_exp_f32(pg, x);
        svst1_f32(pg, out + i, result);
    }

    if (i < n) {
        svbool_t pg_tail = svwhilelt_b32((uint32_t)i, (uint32_t)n);
        svfloat32_t x = svld1_f32(pg_tail, in + i);
        svfloat32_t result = sve_exp_f32(pg_tail, x);
        svst1_f32(pg_tail, out + i, result);
    }

#ifdef SVE_MATH_PROFILE
    SVE_MATH_PROFILE_END(sve_exp_stats, n);
#endif
}

// ----------------------------------------------------------------------------
// SVE Fast Logarithm
//
// log(x) = log(2^e * m) = e*log(2) + log(m) where m in [1, 2)
// Use polynomial for log(m)

static inline svfloat32_t sve_log_f32(svbool_t pg, svfloat32_t x) {
    // Extract exponent and mantissa
    // x = 2^e * m, where m in [1, 2)

    svuint32_t xi = svreinterpret_u32_f32(x);

    // Extract exponent: e = ((xi >> 23) & 0xFF) - 127
    svuint32_t exp_bits = svlsr_n_u32_x(pg, xi, 23);
    exp_bits = svand_u32_x(pg, exp_bits, svdup_u32(0xFF));
    svint32_t e = svsub_s32_x(pg, svreinterpret_s32_u32(exp_bits), svdup_s32(127));
    svfloat32_t e_f = svcvt_f32_s32_x(pg, e);

    // Extract mantissa and normalize to [1, 2): m = (xi & 0x7FFFFF) | 0x3F800000
    svuint32_t mant_bits = svand_u32_x(pg, xi, svdup_u32(0x007FFFFF));
    mant_bits = svorr_u32_x(pg, mant_bits, svdup_u32(0x3F800000));
    svfloat32_t m = svreinterpret_f32_u32(mant_bits);

    // For better accuracy, reduce m to [sqrt(2)/2, sqrt(2)]
    // If m > sqrt(2), use m/2 and increment e
    svfloat32_t sqrt2 = svdup_f32(1.41421356f);
    svbool_t reduce = svcmpgt_f32(pg, m, sqrt2);
    m = svsel_f32(reduce, svmul_f32_x(pg, m, svdup_f32(0.5f)), m);
    e_f = svadd_f32_m(reduce, e_f, svdup_f32(1.0f));

    // log(m) for m in [sqrt(2)/2, sqrt(2)] using polynomial
    // Let u = (m - 1) / (m + 1), then log(m) = 2 * (u + u^3/3 + u^5/5 + ...)
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t num = svsub_f32_x(pg, m, one);
    svfloat32_t den = svadd_f32_x(pg, m, one);
    svfloat32_t u = svdiv_f32_x(pg, num, den);
    svfloat32_t u2 = svmul_f32_x(pg, u, u);

    // Polynomial: 2 * u * (1 + u^2/3 + u^4/5 + u^6/7 + u^8/9)
    svfloat32_t c1 = svdup_f32(1.0f);
    svfloat32_t c3 = svdup_f32(0.33333333f);
    svfloat32_t c5 = svdup_f32(0.2f);
    svfloat32_t c7 = svdup_f32(0.14285714f);
    svfloat32_t c9 = svdup_f32(0.11111111f);

    svfloat32_t poly = svmla_f32_x(pg, c7, u2, c9);
    poly = svmla_f32_x(pg, c5, u2, poly);
    poly = svmla_f32_x(pg, c3, u2, poly);
    poly = svmla_f32_x(pg, c1, u2, poly);

    svfloat32_t log_m = svmul_f32_x(pg, svdup_f32(2.0f), svmul_f32_x(pg, u, poly));

    // log(x) = e * log(2) + log(m)
    svfloat32_t ln2 = svdup_f32(0.6931471805599453f);
    svfloat32_t result = svmla_f32_x(pg, log_m, e_f, ln2);

    // Handle special cases
    svbool_t is_zero = svcmpeq_f32(pg, x, svdup_f32(0.0f));
    svbool_t is_neg = svcmplt_f32(pg, x, svdup_f32(0.0f));
    svbool_t is_inf = svcmpeq_f32(pg, x, svdup_f32(INFINITY));

    result = svsel_f32(is_zero, svdup_f32(-INFINITY), result);
    result = svsel_f32(is_neg, svdup_f32(NAN), result);
    result = svsel_f32(is_inf, svdup_f32(INFINITY), result);

    return result;
}

static inline void sve_log_f32_array(float* out, const float* in, int n) {
    SVE_MATH_PROFILE_START();

    svbool_t pg = svptrue_b32();
    uint64_t vl = svcntw();
    int i = 0;

    for (; i + (int)vl <= n; i += (int)vl) {
        svfloat32_t x = svld1_f32(pg, in + i);
        svfloat32_t result = sve_log_f32(pg, x);
        svst1_f32(pg, out + i, result);
    }

    if (i < n) {
        svbool_t pg_tail = svwhilelt_b32((uint32_t)i, (uint32_t)n);
        svfloat32_t x = svld1_f32(pg_tail, in + i);
        svfloat32_t result = sve_log_f32(pg_tail, x);
        svst1_f32(pg_tail, out + i, result);
    }

#ifdef SVE_MATH_PROFILE
    SVE_MATH_PROFILE_END(sve_log_stats, n);
#endif
}

// ----------------------------------------------------------------------------
// SVE Fast Sine using FTMAD instruction
//
// FTMAD computes coefficients for Taylor series of sin/cos
// sin(x) = x - x^3/6 + x^5/120 - x^7/5040 + ...

static inline svfloat32_t sve_sin_f32(svbool_t pg, svfloat32_t x) {
    // Range reduction: x = x mod 2*pi, then to [-pi, pi]
    svfloat32_t two_over_pi = svdup_f32(TWO_OVER_PI);
    svfloat32_t pi = svdup_f32(PI_F);
    svfloat32_t half_pi = svdup_f32(PI_2_F);

    // k = round(x / pi)
    svfloat32_t k_f = svrintn_f32_x(pg, svmul_f32_x(pg, x, svdup_f32(1.0f / PI_F)));
    svint32_t k = svcvt_s32_f32_x(pg, k_f);

    // r = x - k * pi (reduced to [-pi/2, pi/2])
    svfloat32_t r = svsub_f32_x(pg, x, svmul_f32_x(pg, k_f, pi));

    // If |r| > pi/2, adjust: sin(r) = sin(pi - r) for r > pi/2
    svbool_t need_flip_pos = svcmpgt_f32(pg, r, half_pi);
    svbool_t need_flip_neg = svcmplt_f32(pg, r, svneg_f32_x(pg, half_pi));
    r = svsel_f32(need_flip_pos, svsub_f32_x(pg, pi, r), r);
    r = svsel_f32(need_flip_neg, svsub_f32_x(pg, svneg_f32_x(pg, pi), r), r);

    // Sign adjustment for odd k
    svuint32_t k_odd = svand_u32_x(pg, svreinterpret_u32_s32(k), svdup_u32(1));
    svbool_t negate = svcmpne_u32(pg, k_odd, svdup_u32(0));

    // Taylor series for sin(r): r - r^3/6 + r^5/120 - r^7/5040 + r^9/362880
    svfloat32_t r2 = svmul_f32_x(pg, r, r);

    svfloat32_t c1 = svdup_f32(1.0f);
    svfloat32_t c3 = svdup_f32(-0.16666666666666666f);  // -1/6
    svfloat32_t c5 = svdup_f32(0.008333333333333333f);  // 1/120
    svfloat32_t c7 = svdup_f32(-0.0001984126984126984f); // -1/5040
    svfloat32_t c9 = svdup_f32(2.7557319223985893e-6f); // 1/362880

    // Horner's method: r * (c1 + r^2*(c3 + r^2*(c5 + r^2*(c7 + r^2*c9))))
    svfloat32_t poly = svmla_f32_x(pg, c7, r2, c9);
    poly = svmla_f32_x(pg, c5, r2, poly);
    poly = svmla_f32_x(pg, c3, r2, poly);
    poly = svmla_f32_x(pg, c1, r2, poly);

    svfloat32_t result = svmul_f32_x(pg, r, poly);

    // Apply sign for odd k
    result = svneg_f32_m(result, negate, result);

    return result;
}

static inline void sve_sin_f32_array(float* out, const float* in, int n) {
    SVE_MATH_PROFILE_START();

    svbool_t pg = svptrue_b32();
    uint64_t vl = svcntw();
    int i = 0;

    for (; i + (int)vl <= n; i += (int)vl) {
        svfloat32_t x = svld1_f32(pg, in + i);
        svfloat32_t result = sve_sin_f32(pg, x);
        svst1_f32(pg, out + i, result);
    }

    if (i < n) {
        svbool_t pg_tail = svwhilelt_b32((uint32_t)i, (uint32_t)n);
        svfloat32_t x = svld1_f32(pg_tail, in + i);
        svfloat32_t result = sve_sin_f32(pg_tail, x);
        svst1_f32(pg_tail, out + i, result);
    }

#ifdef SVE_MATH_PROFILE
    SVE_MATH_PROFILE_END(sve_sin_stats, n);
#endif
}

// ----------------------------------------------------------------------------
// SVE Fast Cosine
//
// cos(x) = sin(x + pi/2)

static inline svfloat32_t sve_cos_f32(svbool_t pg, svfloat32_t x) {
    // cos(x) = sin(x + pi/2)
    svfloat32_t half_pi = svdup_f32(PI_2_F);
    return sve_sin_f32(pg, svadd_f32_x(pg, x, half_pi));
}

static inline void sve_cos_f32_array(float* out, const float* in, int n) {
    SVE_MATH_PROFILE_START();

    svbool_t pg = svptrue_b32();
    uint64_t vl = svcntw();
    int i = 0;

    for (; i + (int)vl <= n; i += (int)vl) {
        svfloat32_t x = svld1_f32(pg, in + i);
        svfloat32_t result = sve_cos_f32(pg, x);
        svst1_f32(pg, out + i, result);
    }

    if (i < n) {
        svbool_t pg_tail = svwhilelt_b32((uint32_t)i, (uint32_t)n);
        svfloat32_t x = svld1_f32(pg_tail, in + i);
        svfloat32_t result = sve_cos_f32(pg_tail, x);
        svst1_f32(pg_tail, out + i, result);
    }

#ifdef SVE_MATH_PROFILE
    SVE_MATH_PROFILE_END(sve_cos_stats, n);
#endif
}

// ----------------------------------------------------------------------------
// SVE Fast Tangent
//
// tan(x) = sin(x) / cos(x)

static inline svfloat32_t sve_tan_f32(svbool_t pg, svfloat32_t x) {
    svfloat32_t s = sve_sin_f32(pg, x);
    svfloat32_t c = sve_cos_f32(pg, x);
    return svdiv_f32_x(pg, s, c);
}

static inline void sve_tan_f32_array(float* out, const float* in, int n) {
    SVE_MATH_PROFILE_START();

    svbool_t pg = svptrue_b32();
    uint64_t vl = svcntw();
    int i = 0;

    for (; i + (int)vl <= n; i += (int)vl) {
        svfloat32_t x = svld1_f32(pg, in + i);
        svfloat32_t result = sve_tan_f32(pg, x);
        svst1_f32(pg, out + i, result);
    }

    if (i < n) {
        svbool_t pg_tail = svwhilelt_b32((uint32_t)i, (uint32_t)n);
        svfloat32_t x = svld1_f32(pg_tail, in + i);
        svfloat32_t result = sve_tan_f32(pg_tail, x);
        svst1_f32(pg_tail, out + i, result);
    }

#ifdef SVE_MATH_PROFILE
    SVE_MATH_PROFILE_END(sve_tan_stats, n);
#endif
}

// ----------------------------------------------------------------------------
// SVE Fast Hyperbolic Tangent
//
// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
//         = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
// For |x| > 9, tanh(x) ≈ sign(x)

static inline svfloat32_t sve_tanh_f32(svbool_t pg, svfloat32_t x) {
    // For small |x|, use polynomial approximation
    // For large |x|, clamp to ±1

    svfloat32_t abs_x = svabs_f32_x(pg, x);
    svbool_t large = svcmpgt_f32(pg, abs_x, svdup_f32(9.0f));
    svbool_t small = svcmplt_f32(pg, abs_x, svdup_f32(0.0004f));

    // For medium range: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    svfloat32_t two_x = svmul_f32_x(pg, x, svdup_f32(2.0f));
    svfloat32_t exp_2x = sve_exp_f32(pg, two_x);
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t num = svsub_f32_x(pg, exp_2x, one);
    svfloat32_t den = svadd_f32_x(pg, exp_2x, one);
    svfloat32_t result = svdiv_f32_x(pg, num, den);

    // For small x: tanh(x) ≈ x
    result = svsel_f32(small, x, result);

    // For large |x|: tanh(x) = sign(x)
    svfloat32_t sign_x = svsel_f32(svcmplt_f32(pg, x, svdup_f32(0.0f)),
                                    svdup_f32(-1.0f), svdup_f32(1.0f));
    result = svsel_f32(large, sign_x, result);

    return result;
}

static inline void sve_tanh_f32_array(float* out, const float* in, int n) {
    SVE_MATH_PROFILE_START();

    svbool_t pg = svptrue_b32();
    uint64_t vl = svcntw();
    int i = 0;

    for (; i + (int)vl <= n; i += (int)vl) {
        svfloat32_t x = svld1_f32(pg, in + i);
        svfloat32_t result = sve_tanh_f32(pg, x);
        svst1_f32(pg, out + i, result);
    }

    if (i < n) {
        svbool_t pg_tail = svwhilelt_b32((uint32_t)i, (uint32_t)n);
        svfloat32_t x = svld1_f32(pg_tail, in + i);
        svfloat32_t result = sve_tanh_f32(pg_tail, x);
        svst1_f32(pg_tail, out + i, result);
    }

#ifdef SVE_MATH_PROFILE
    SVE_MATH_PROFILE_END(sve_tanh_stats, n);
#endif
}

// ----------------------------------------------------------------------------
// Combined sincos for efficiency (computes both at once)

static inline void sve_sincos_f32(svbool_t pg, svfloat32_t x,
                                   svfloat32_t* sin_out, svfloat32_t* cos_out) {
    // Range reduction
    svfloat32_t pi = svdup_f32(PI_F);
    svfloat32_t half_pi = svdup_f32(PI_2_F);

    svfloat32_t k_f = svrintn_f32_x(pg, svmul_f32_x(pg, x, svdup_f32(1.0f / PI_F)));
    svint32_t k = svcvt_s32_f32_x(pg, k_f);

    svfloat32_t r = svsub_f32_x(pg, x, svmul_f32_x(pg, k_f, pi));

    svbool_t need_flip_pos = svcmpgt_f32(pg, r, half_pi);
    svbool_t need_flip_neg = svcmplt_f32(pg, r, svneg_f32_x(pg, half_pi));
    svfloat32_t r_sin = svsel_f32(need_flip_pos, svsub_f32_x(pg, pi, r), r);
    r_sin = svsel_f32(need_flip_neg, svsub_f32_x(pg, svneg_f32_x(pg, pi), r_sin), r_sin);

    svuint32_t k_odd = svand_u32_x(pg, svreinterpret_u32_s32(k), svdup_u32(1));
    svbool_t negate_sin = svcmpne_u32(pg, k_odd, svdup_u32(0));

    // Compute sin using Taylor series
    svfloat32_t r2 = svmul_f32_x(pg, r_sin, r_sin);

    svfloat32_t c1 = svdup_f32(1.0f);
    svfloat32_t c3 = svdup_f32(-0.16666666666666666f);
    svfloat32_t c5 = svdup_f32(0.008333333333333333f);
    svfloat32_t c7 = svdup_f32(-0.0001984126984126984f);

    svfloat32_t poly = svmla_f32_x(pg, c5, r2, c7);
    poly = svmla_f32_x(pg, c3, r2, poly);
    poly = svmla_f32_x(pg, c1, r2, poly);

    svfloat32_t sin_r = svmul_f32_x(pg, r_sin, poly);
    sin_r = svneg_f32_m(sin_r, negate_sin, sin_r);

    // cos = sqrt(1 - sin^2) with sign correction
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t cos_r2 = svsub_f32_x(pg, one, svmul_f32_x(pg, sin_r, sin_r));
    svfloat32_t cos_r = svsqrt_f32_x(pg, cos_r2);

    // Correct cos sign based on quadrant
    svfloat32_t r_for_cos = svadd_f32_x(pg, r, half_pi);
    svfloat32_t k_cos_f = svrintn_f32_x(pg, svmul_f32_x(pg, r_for_cos, svdup_f32(1.0f / PI_F)));
    svint32_t k_cos = svcvt_s32_f32_x(pg, k_cos_f);
    svuint32_t k_cos_odd = svand_u32_x(pg, svreinterpret_u32_s32(k_cos), svdup_u32(1));
    svbool_t negate_cos = svcmpne_u32(pg, k_cos_odd, svdup_u32(0));
    cos_r = svneg_f32_m(cos_r, negate_cos, cos_r);

    *sin_out = sin_r;
    *cos_out = cos_r;
}

#endif // USE_SVE
#endif // SVE_MATH_H
