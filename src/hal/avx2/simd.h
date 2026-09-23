/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <immintrin.h>

// x86 AVX2 primitives (i9 tuning, verbatim from the baseline lm_ops.cpp).
namespace tcpu {
static inline float simd_dot(const float* a, const float* b, int n) {
    // 4 independent accumulator chains hide the ~4-cycle FMA latency (a single chain
    // is latency-bound; head_dim-64 attention dots ran at ~6% of peak with one).
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    int i = 0;
    for (; i+32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i),    _mm256_loadu_ps(b+i),    acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),  _mm256_loadu_ps(b+i+8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24), acc3);
    }
    for (; i+8 <= n; i += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc0);

    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);

    float s = _mm_cvtss_f32(lo);
    for (; i < n; i++)
        s += a[i]*b[i];
    return s;
}

static inline void simd_axpy(float* o, float alpha, const float* v, int n) {
    __m256 va = _mm256_set1_ps(alpha);
    int i = 0;
    for (; i+8 <= n; i += 8)
        _mm256_storeu_ps(o+i, _mm256_fmadd_ps(va, _mm256_loadu_ps(v+i), _mm256_loadu_ps(o+i)));
    for (; i < n; i++)
        o[i] += alpha*v[i];
}

static inline float hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

static inline float hmax8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

// Cephes-style exp, ~1-2 ulp (avx_mathfun lineage). Lanes below -87 flush to exactly
// 0.0f (mirrors libm underflow), so blocked-key scores keep producing zero weight.
static inline __m256 exp256_ps(__m256 x) {
    const __m256 keep = _mm256_cmp_ps(x, _mm256_set1_ps(-87.0f), _CMP_NLE_UQ);
    x = _mm256_min_ps(_mm256_set1_ps(88.3762626647949f), x);
    x = _mm256_max_ps(_mm256_set1_ps(-88.3762626647950f), x);

    __m256 fx = _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f));
    fx = _mm256_floor_ps(fx);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);

    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), _mm256_add_ps(x, _mm256_set1_ps(1.0f)));

    __m256i n = _mm256_cvtps_epi32(fx);
    n = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
    y = _mm256_mul_ps(y, _mm256_castsi256_ps(n));

    return _mm256_and_ps(y, keep);
}
} // namespace tcpu
