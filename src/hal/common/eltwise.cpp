/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// Elementwise activations. gelu_tanh has a SIMD path per backend (same sigmoid
// identity and Cephes exp everywhere - one exp per vector instead of per-lane
// tanhf; tolerance class). relu is OMP-parallel only on the Apple backend
// (measured win there; the other branches shipped it serial).

#include "../arch.h"
#include "../simd.h"
#include "env.h"
#include "../../ops/lm_ops.h"
#include <cmath>
#include <cstddef>
using std::size_t;

namespace tcpu {

// silu(g)*u = g/(1+exp(-g)) * u. Same treatment as gelu_tanh: std::exp is an
// opaque libm call, so the loop cannot auto-vectorize (measured: one expf call
// per element survives at -O3). The SIMD paths use the backend's Cephes exp -
// one exp per vector - which puts silu in the same poly-exp tolerance class the
// gelu path already ships (measured max rel 2.7e-07 on the SmolVLA shapes).
// TCPU_SIMD_SILU=0 restores the scalar loop. silu() below is deliberately left
// scalar: its only callers are Octo's diffusion head (64-1024 elements), where
// there is nothing to win and the numerics are golden-verified as they are.
void silu_gate(float* out, const float* g, const float* u, int n) {
#if TCPU_ISA_X86 || TCPU_HAL_APPLE || TCPU_HAL_NEON
    if (hal::env::simd_silu()) {
        const int lanes = TCPU_ISA_X86 ? 8 : 4;
        const int nv = n - (n % lanes);
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) if(n > hal::env::omp_min())
#endif
        for (int i=0; i<nv; i += lanes) {
#if TCPU_ISA_X86
            const __m256 x = _mm256_loadu_ps(g+i);
            const __m256 e = exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), x));
            const __m256 s = _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0f), e));
            _mm256_storeu_ps(out+i, _mm256_mul_ps(s, _mm256_loadu_ps(u+i)));
#elif TCPU_HAL_APPLE
            const float32x4_t x = vld1q_f32(g+i);
            const float32x4_t e = exp_ps(vnegq_f32(x));
            const float32x4_t s = vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.0f), e));
            vst1q_f32(out+i, vmulq_f32(s, vld1q_f32(u+i)));
#else
            const float32x4_t x = vld1q_f32(g+i);
            const float32x4_t e = exp_ps_neon(vnegq_f32(x));
            const float32x4_t s = vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.0f), e));
            vst1q_f32(out+i, vmulq_f32(s, vld1q_f32(u+i)));
#endif
        }
        for (int i=nv; i<n; i++) {
            float x = g[i];
            out[i] = (x/(1.0f+std::exp(-x)))*u[i];
        }
        return;
    }
#endif
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(n > hal::env::omp_min())
#endif
    for (int i=0; i<n; i++) {
        float x = g[i];
        out[i] = (x/(1.0f+std::exp(-x)))*u[i];
    }
}

void silu(float* x, int n) {
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(n > hal::env::omp_min())
#endif
    for (int i=0; i<n; i++)
        x[i] = x[i]/(1.0f+std::exp(-x[i]));
}

void gelu_tanh(float* x, int n) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
#if TCPU_ISA_X86
    // 0.5*(1+tanh(y)) == sigmoid(2y), so gelu = v * sigmoid(2c*(v + 0.044715 v^3));
    // one exp256_ps per 8 elements instead of 8 scalar tanhf (tolerance class).
    const int nv = n & ~7;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(n > hal::env::omp_min())
#endif
    for (int i=0; i<nv; i += 8) {
        const __m256 v = _mm256_loadu_ps(x+i);
        const __m256 v3 = _mm256_mul_ps(_mm256_mul_ps(v, v), v);
        const __m256 y = _mm256_mul_ps(_mm256_set1_ps(2.0f*c),
                                       _mm256_fmadd_ps(_mm256_set1_ps(0.044715f), v3, v));
        const __m256 e = exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), y));
        const __m256 s = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(_mm256_set1_ps(1.0f), e));
        _mm256_storeu_ps(x+i, _mm256_mul_ps(v, s));
    }
    for (int i=nv; i<n; i++) {
        float v = x[i];
        x[i] = 0.5f*v*(1.0f+std::tanh(c*(v+0.044715f*v*v*v)));
    }
#elif TCPU_HAL_APPLE
    // same sigmoid identity as the AVX2 path: one exp_ps per 4 elements
    const int nv = n & ~3;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(n > hal::env::omp_min())
#endif
    for (int i=0; i<nv; i += 4) {
        const float32x4_t v = vld1q_f32(x+i);
        const float32x4_t v3 = vmulq_f32(vmulq_f32(v, v), v);
        const float32x4_t y = vmulq_n_f32(vfmaq_n_f32(v, v3, 0.044715f), 2.0f*c);
        const float32x4_t e = exp_ps(vnegq_f32(y));
        const float32x4_t s = vdivq_f32(vdupq_n_f32(1.0f), vaddq_f32(vdupq_n_f32(1.0f), e));
        vst1q_f32(x+i, vmulq_f32(v, s));
    }
    for (int i=nv; i<n; i++) {
        float v = x[i];
        x[i] = 0.5f*v*(1.0f+std::tanh(c*(v+0.044715f*v*v*v)));
    }
#elif TCPU_HAL_NEON
    // Same sigmoid identity as the AVX2 path: gelu = v * sigmoid(2c*(v+0.044715 v^3)),
    // one exp_ps_neon per 4 elements instead of 4 scalar tanhf (11.7M calls across the
    // Octo MLP). Matches the AVX2 gelu numerics (tolerance class).
    const int nv = n & ~3;
    const float32x4_t vc2   = vdupq_n_f32(2.0f*c);
    const float32x4_t vcoef = vdupq_n_f32(0.044715f);
    const float32x4_t one   = vdupq_n_f32(1.0f);
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(n > hal::env::omp_min())
#endif
    for (int i=0; i<nv; i += 4) {
        const float32x4_t v = vld1q_f32(x+i);
        const float32x4_t v3 = vmulq_f32(vmulq_f32(v, v), v);
        const float32x4_t y = vmulq_f32(vc2, vfmaq_f32(v, vcoef, v3));
        const float32x4_t e = exp_ps_neon(vnegq_f32(y));
        const float32x4_t s = vdivq_f32(one, vaddq_f32(one, e));
        vst1q_f32(x+i, vmulq_f32(v, s));
    }
    for (int i=nv; i<n; i++) {
        float v = x[i];
        x[i] = 0.5f*v*(1.0f+std::tanh(c*(v+0.044715f*v*v*v)));
    }
#else
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(n > hal::env::omp_min())
#endif
    for (int i=0; i<n; i++) {
        float v = x[i];
        x[i] = 0.5f*v*(1.0f+std::tanh(c*(v+0.044715f*v*v*v)));
    }
#endif
}

// Exact GELU: 0.5*x*(1+erf(x/sqrt(2))). std::erf is an opaque libm call, so the
// scalar loop cannot vectorize - one erff per element, and this activation runs
// 19M times per TurboVLA inference (12 DINOv3 layers x 261 tokens x 3072, twice).
// It measured 21 ms of the tower's 125 ms per view, so the SIMD path evaluates
// erf itself: Abramowitz-Stegun 7.1.26,
//
//   erf(z) = 1 - (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5) e^(-z^2),
//   t = 1/(1 + p z),  z >= 0,  odd symmetry for z < 0
//
// which is 1.5e-07 absolute on erf and so 0.5*|x|*1.5e-07 on the output - under
// 1e-06 over the range an activation reaches, the same poly-exp tolerance class
// as gelu_tanh and silu_gate. TCPU_SIMD_ERF=0 restores the libm loop.
// gelu_tanh is NOT a substitute for either path: see the header.
void gelu_erf(float* x, int n) {
    const float inv_sqrt2 = 0.70710678118654752f;
#if TCPU_ISA_X86 || TCPU_HAL_APPLE || TCPU_HAL_NEON
    if (hal::env::simd_erf()) {
        const float p  =  0.3275911f;
        const float a1 =  0.254829592f,  a2 = -0.284496736f, a3 = 1.421413741f;
        const float a4 = -1.453152027f,  a5 =  1.061405429f;
        const int lanes = TCPU_ISA_X86 ? 8 : 4;
        const int nv = n - (n % lanes);
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) if(n > hal::env::omp_min())
#endif
        for (int i=0; i<nv; i += lanes) {
#if TCPU_ISA_X86
            const __m256 v = _mm256_loadu_ps(x+i);
            const __m256 z = _mm256_mul_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), v),
                                           _mm256_set1_ps(inv_sqrt2));
            const __m256 t = _mm256_div_ps(_mm256_set1_ps(1.0f),
                                           _mm256_fmadd_ps(_mm256_set1_ps(p), z,
                                                           _mm256_set1_ps(1.0f)));
            __m256 poly = _mm256_fmadd_ps(_mm256_set1_ps(a5), t, _mm256_set1_ps(a4));
            poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(a3));
            poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(a2));
            poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(a1));
            poly = _mm256_mul_ps(poly, t);
            const __m256 e = exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(z, z)));
            // erf(|x|/sqrt2) with the sign of x put back, then 0.5*x*(1+erf)
            const __m256 erf_abs = _mm256_fnmadd_ps(poly, e, _mm256_set1_ps(1.0f));
            const __m256 sign = _mm256_and_ps(v, _mm256_set1_ps(-0.0f));
            const __m256 erf = _mm256_or_ps(erf_abs, sign);
            _mm256_storeu_ps(x+i, _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), v),
                                                _mm256_add_ps(_mm256_set1_ps(1.0f), erf)));
#else
            const float32x4_t v = vld1q_f32(x+i);
            const float32x4_t z = vmulq_n_f32(vabsq_f32(v), inv_sqrt2);
            const float32x4_t t = vdivq_f32(vdupq_n_f32(1.0f),
                                            vfmaq_n_f32(vdupq_n_f32(1.0f), z, p));
            float32x4_t poly = vfmaq_n_f32(vdupq_n_f32(a4), t, a5);
            poly = vfmaq_f32(vdupq_n_f32(a3), poly, t);
            poly = vfmaq_f32(vdupq_n_f32(a2), poly, t);
            poly = vfmaq_f32(vdupq_n_f32(a1), poly, t);
            poly = vmulq_f32(poly, t);
#if TCPU_HAL_APPLE
            const float32x4_t e = exp_ps(vnegq_f32(vmulq_f32(z, z)));
#else
            const float32x4_t e = exp_ps_neon(vnegq_f32(vmulq_f32(z, z)));
#endif
            const float32x4_t erf_abs = vfmsq_f32(vdupq_n_f32(1.0f), poly, e);
            const uint32x4_t sign = vandq_u32(vreinterpretq_u32_f32(v), vdupq_n_u32(0x80000000u));
            const float32x4_t erf = vreinterpretq_f32_u32(
                vorrq_u32(vreinterpretq_u32_f32(erf_abs), sign));
            vst1q_f32(x+i, vmulq_f32(vmulq_n_f32(v, 0.5f), vaddq_f32(vdupq_n_f32(1.0f), erf)));
#endif
        }
        for (int i=nv; i<n; i++) {
            const float v = x[i];
            x[i] = 0.5f*v*(1.0f+std::erf(v*inv_sqrt2));
        }
        return;
    }
#endif
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(n > hal::env::omp_min())
#endif
    for (int i=0; i<n; i++) {
        const float v = x[i];
        x[i] = 0.5f*v*(1.0f+std::erf(v*inv_sqrt2));
    }
}

void relu(float* x, int n) {
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(n > hal::env::omp_min())
#endif
    for (int i=0; i<n; i++)
        if (x[i] < 0.0f) x[i] = 0.0f;
}

void mish(float* x, int n) {
    // softplus(v) = log1p(exp(v)) saturates to v once exp(v) overflows fp32.
    // torch switches to the identity at 20 (its Softplus threshold) and so do we:
    // above that, tanh(softplus(v)) is 1 to well inside fp32, and the branch also
    // keeps log1p(exp(v)) from returning inf for the large activations the UNet's
    // wide channel blocks do produce.
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(n > hal::env::omp_min())
#endif
    for (int i=0; i<n; i++) {
        const float v  = x[i];
        const float sp = v > 20.0f ? v : std::log1p(std::exp(v));
        x[i] = v*std::tanh(sp);
    }
}

} // namespace tcpu
