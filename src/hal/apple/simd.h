/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <arm_neon.h>

// Apple-Silicon NEON primitives (M4 tuning, verbatim from the m4 branch).
namespace tcpu {

// NEON mirrors of the AVX2 primitives (128-bit vectors, 32 registers). Same
// accumulation structure per op class -> same fp32 reorder tolerance story.
static inline float simd_dot(const float* a, const float* b, int n) {
    // 8 independent chains: M4 P-cores have 4 FMA pipes x ~3-cycle latency, so a
    // single 4-wide chain is latency-bound just like the one-chain AVX2 version was.
    float32x4_t c0 = vdupq_n_f32(0);
    float32x4_t c1 = vdupq_n_f32(0);
    float32x4_t c2 = vdupq_n_f32(0);
    float32x4_t c3 = vdupq_n_f32(0);
    float32x4_t c4 = vdupq_n_f32(0);
    float32x4_t c5 = vdupq_n_f32(0);
    float32x4_t c6 = vdupq_n_f32(0);
    float32x4_t c7 = vdupq_n_f32(0);

    int i = 0;
    for (; i+32<=n; i+=32) {
        c0 = vfmaq_f32(c0, vld1q_f32(a+i),    vld1q_f32(b+i));
        c1 = vfmaq_f32(c1, vld1q_f32(a+i+4),  vld1q_f32(b+i+4));
        c2 = vfmaq_f32(c2, vld1q_f32(a+i+8),  vld1q_f32(b+i+8));
        c3 = vfmaq_f32(c3, vld1q_f32(a+i+12), vld1q_f32(b+i+12));
        c4 = vfmaq_f32(c4, vld1q_f32(a+i+16), vld1q_f32(b+i+16));
        c5 = vfmaq_f32(c5, vld1q_f32(a+i+20), vld1q_f32(b+i+20));
        c6 = vfmaq_f32(c6, vld1q_f32(a+i+24), vld1q_f32(b+i+24));
        c7 = vfmaq_f32(c7, vld1q_f32(a+i+28), vld1q_f32(b+i+28));
    }
    for (; i+4<=n; i+=4)
        c0 = vfmaq_f32(c0, vld1q_f32(a+i), vld1q_f32(b+i));

    c0 = vaddq_f32(vaddq_f32(vaddq_f32(c0, c1), vaddq_f32(c2, c3)),
                   vaddq_f32(vaddq_f32(c4, c5), vaddq_f32(c6, c7)));
    float s = vaddvq_f32(c0);

    for (; i<n; i++)
        s += a[i]*b[i];
    return s;
}

static inline void simd_axpy(float* o, float alpha, const float* v, int n) {
    const float32x4_t va = vdupq_n_f32(alpha);
    int i = 0;

    for (; i+4<=n; i+=4)
        vst1q_f32(o+i, vfmaq_f32(vld1q_f32(o+i), va, vld1q_f32(v+i)));

    for (; i<n; i++)
        o[i] += alpha*v[i];
}

// Cephes-style exp, same coefficients as the AVX2 exp256_ps; lanes below -87 flush
// to exactly 0.0f so blocked-key scores keep producing zero weight.
static inline float32x4_t exp_ps(float32x4_t x) {
    const uint32x4_t keep = vmvnq_u32(vcleq_f32(x, vdupq_n_f32(-87.0f)));
    x = vminq_f32(x, vdupq_n_f32(88.3762626647949f));
    x = vmaxq_f32(x, vdupq_n_f32(-88.3762626647950f));

    float32x4_t fx = vfmaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(1.44269504088896341f));
    fx = vrndmq_f32(fx);   // floor
    x = vfmsq_f32(x, fx, vdupq_n_f32(0.693359375f));
    x = vfmsq_f32(x, fx, vdupq_n_f32(-2.12194440e-4f));

    float32x4_t y = vdupq_n_f32(1.9875691500e-4f);
    y = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), y, x);
    y = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), y, x);
    y = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), y, x);
    y = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), y, x);
    y = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), y, x);
    y = vfmaq_f32(vaddq_f32(x, vdupq_n_f32(1.0f)), y, vmulq_f32(x, x));

    int32x4_t n = vcvtq_s32_f32(fx);
    n = vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
    y = vmulq_f32(y, vreinterpretq_f32_s32(n));

    return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(y), keep));
}
} // namespace tcpu
