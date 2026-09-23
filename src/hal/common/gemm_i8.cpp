/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// Backend-independent half of the W8A8 GEMM: the offline packer, the activation
// quantizer, and - for backends with no int8 kernel of their own - a portable
// reference implementation. The NEON dotprod kernel overrides the two hot
// entry points (src/hal/neon/gemm_i8.cpp).

#include "../arch.h"
#include "../../ops/quant_ops.h"
#include "env.h"
#if TCPU_ISA_ARM
#include <arm_neon.h>
#elif TCPU_ISA_X86
#include <immintrin.h>
#endif
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstddef>
using std::size_t;

namespace tcpu {

int i8_kpad(int K) { return (K+3) & ~3; }

// panels, then one int32 per output row. N*Kp is a multiple of 4 (Kp is), so the
// table lands 4-byte aligned.
static size_t i8_panel_bytes(int N, int K) { return (size_t)N*i8_kpad(K); }

size_t packed_i8_floats(int N, int K) {
    return i8_panel_bytes(N, K) + (size_t)N*sizeof(int32_t);
}

size_t packed_i8_words(int N, int K) {
    return packed_i8_floats(N, K)/sizeof(int32_t);   // panel bytes are a multiple of 4
}

const int32_t* i8_rowsums(const int8_t* Wq, int N, int K) {
    return (const int32_t*)(Wq + i8_panel_bytes(N, K));
}

// Round-to-nearest-even: one instruction on every backend (cvtps_epi32 /
// vcvtnq_s32_f32), so packer, kernel and scalar tail agree bit for bit. The
// half-away emulation trunc(x + copysign(0.5,x)) does not - at 0x3EFFFFFF it
// returns 1 where lround returns 0.
static inline int8_t q8(float v) {
    long r = std::lrint(v);
    if (r >  127) r =  127;
    if (r < -127) r = -127;
    return (int8_t)r;
}

void pack_weights_i8(const float* W, int8_t* Wq, float* wscale, int N, int K) {
    // The panel index below is b = n/16, so a partial trailing block writes past
    // the buffer packed_i8_floats sized. Both callers check this
    // (nn::Linear::init_int8, nn::Conv2d::init_int8); assert so a third cannot
    // discover it as heap corruption.
    assert(N % 16 == 0 && "pack_weights_i8 requires N % 16 == 0");
    const int Kp = i8_kpad(K);
    const int KG = Kp/4;
    std::memset(Wq, 0, i8_panel_bytes(N, K));
    int32_t* rowsum = (int32_t*)(Wq + i8_panel_bytes(N, K));

    for (int n=0; n<N; n++) {
        const float* wr = W+(size_t)n*K;
        float amax = 0.0f;
        for (int k=0; k<K; k++) {
            const float a = std::fabs(wr[k]);
            if (a > amax) amax = a;
        }
        // An all-zero row would divide by zero; its quantized row is all zeros
        // anyway, so any nonzero scale works and 1 keeps the epilogue finite.
        const float s = amax > 0.0f ? amax/127.0f : 1.0f;
        wscale[n] = s;

        const int b = n/16, j = n%16;
        const float inv = 1.0f/s;
        int32_t sum = 0;
        for (int k=0; k<K; k++) {
            const int8_t q = q8(wr[k]*inv);
            Wq[((size_t)(b*KG + k/4)*16 + j)*4 + (k%4)] = q;
            sum += q;
        }
        rowsum[n] = sum;   // padding lanes are zero, so they do not contribute
    }
}

#if TCPU_ISA_ARM
// NEON absmax + convert. vcvtnq_s32_f32 (FCVTNS) is round-to-nearest-even.
static inline void quantize_row_neon(const float* xt, int8_t* q, float* scale, int K, int Kp) {
    float32x4_t m0 = vdupq_n_f32(0), m1 = vdupq_n_f32(0);
    int k = 0;
    for (; k+8<=K; k+=8) {
        m0 = vmaxq_f32(m0, vabsq_f32(vld1q_f32(xt+k)));
        m1 = vmaxq_f32(m1, vabsq_f32(vld1q_f32(xt+k+4)));
    }
    float amax = vmaxvq_f32(vmaxq_f32(m0, m1));
    for (; k<K; k++) {
        const float a = std::fabs(xt[k]);
        if (a > amax) amax = a;
    }

    const float s = amax > 0.0f ? amax/127.0f : 1.0f;
    *scale = s;
    const float32x4_t inv = vdupq_n_f32(1.0f/s);

    k = 0;
    for (; k+16<=K; k+=16) {
        const int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xt+k),    inv));
        const int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xt+k+4),  inv));
        const int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xt+k+8),  inv));
        const int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xt+k+12), inv));
        // saturating narrows clamp to [-128,127]; -128 cannot occur here because
        // |x|/s <= 127 by construction, so this matches the scalar clamp
        const int16x8_t s0 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        const int16x8_t s1 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
        vst1q_s8(q+k, vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1)));
    }
    const float invs = 1.0f/s;
    for (; k<K; k++) q[k] = q8(xt[k]*invs);
    for (; k<Kp; k++) q[k] = 0;
}
#endif

#if TCPU_ISA_X86
// AVX2 absmax + convert, the x86 twin of quantize_row_neon. Without it the int8
// GEMM is a net loss: a 1024x3072 ViT activation is 3.1M scalar calls per layer,
// more than the 4x the integer kernel wins.
static inline void quantize_row_avx2(const float* xt, int8_t* q, float* scale, int K, int Kp) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 m0 = _mm256_setzero_ps(), m1 = _mm256_setzero_ps();
    int k = 0;
    for (; k+16<=K; k+=16) {
        m0 = _mm256_max_ps(m0, _mm256_and_ps(_mm256_loadu_ps(xt+k),   absmask));
        m1 = _mm256_max_ps(m1, _mm256_and_ps(_mm256_loadu_ps(xt+k+8), absmask));
    }
    __m128 h = _mm_max_ps(_mm256_castps256_ps128(_mm256_max_ps(m0, m1)),
                          _mm256_extractf128_ps(_mm256_max_ps(m0, m1), 1));
    h = _mm_max_ps(h, _mm_movehl_ps(h, h));
    h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 0x55));
    float amax = _mm_cvtss_f32(h);
    for (; k<K; k++) {
        const float a = std::fabs(xt[k]);
        if (a > amax) amax = a;
    }

    const float s = amax > 0.0f ? amax/127.0f : 1.0f;
    *scale = s;
    const __m256 inv  = _mm256_set1_ps(1.0f/s);
    const __m256i ord = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    k = 0;
    for (; k+32<=K; k+=32) {
        __m256i c[4];
        for (int j = 0; j < 4; j++)
            c[j] = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xt+k+8*j), inv));
        // pack 32 int32 -> 32 int8. Both packs saturate to [-128,127], which
        // |x|/s <= 127 never reaches, and both work within 128-bit lanes; the
        // final permute undoes that interleaving.
        const __m256i a = _mm256_packs_epi32(c[0], c[1]);
        const __m256i b = _mm256_packs_epi32(c[2], c[3]);
        const __m256i p = _mm256_packs_epi16(a, b);
        _mm256_storeu_si256((__m256i*)(q+k), _mm256_permutevar8x32_epi32(p, ord));
    }
    const float invs = 1.0f/s;
    for (; k<K; k++) q[k] = q8(xt[k]*invs);
    for (; k<Kp; k++) q[k] = 0;
}
#endif

// The row quantizers need no clamp: |x|/s <= 127. Conv scales by
// TCPU_I8_CLIP <= 1, so x*inv can exceed 127 and the saturating narrows land on
// -128 where q8 gives -127. Clamp in fp32 first.
void quantize_span_i8(const float* x, int8_t* q, size_t n, float inv) {
    size_t i = 0;
#if TCPU_ISA_ARM
    const float32x4_t vi = vdupq_n_f32(inv);
    const float32x4_t lo = vdupq_n_f32(-127.0f), hi = vdupq_n_f32(127.0f);
    auto cvt = [&](const float* p) {
        return vcvtnq_s32_f32(vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(p), vi), lo), hi));
    };
    for (; i+16<=n; i+=16) {
        const int16x8_t s0 = vcombine_s16(vqmovn_s32(cvt(x+i)),   vqmovn_s32(cvt(x+i+4)));
        const int16x8_t s1 = vcombine_s16(vqmovn_s32(cvt(x+i+8)), vqmovn_s32(cvt(x+i+12)));
        vst1q_s8(q+i, vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1)));
    }
#elif TCPU_ISA_X86
    const __m256 vi = _mm256_set1_ps(inv);
    const __m256 lo = _mm256_set1_ps(-127.0f), hi = _mm256_set1_ps(127.0f);
    const __m256i ord = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (; i+32<=n; i+=32) {
        __m256i c[4];
        for (int j = 0; j < 4; j++) {
            const __m256 v = _mm256_mul_ps(_mm256_loadu_ps(x+i+8*j), vi);
            c[j] = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(v, lo), hi));
        }
        const __m256i a = _mm256_packs_epi32(c[0], c[1]);
        const __m256i b = _mm256_packs_epi32(c[2], c[3]);
        _mm256_storeu_si256((__m256i*)(q+i),
                            _mm256_permutevar8x32_epi32(_mm256_packs_epi16(a, b), ord));
    }
#endif
    for (; i<n; i++) q[i] = q8(x[i]*inv);
}

void quantize_act_i8(const float* x, int8_t* xq, float* ascale, int seq, int K) {
    const int Kp = i8_kpad(K);
#if TCPU_ISA_ARM
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if((size_t)seq*K > (size_t)hal::env::omp_min())
#endif
    for (int t=0; t<seq; t++)
        quantize_row_neon(x+(size_t)t*K, xq+(size_t)t*Kp, ascale+t, K, Kp);
    return;
#elif TCPU_ISA_X86
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if((size_t)seq*K > (size_t)hal::env::omp_min())
#endif
    for (int t=0; t<seq; t++)
        quantize_row_avx2(x+(size_t)t*K, xq+(size_t)t*Kp, ascale+t, K, Kp);
    return;
#endif
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if((size_t)seq*K > (size_t)hal::env::omp_min())
#endif
    for (int t=0; t<seq; t++) {
        const float* xt = x+(size_t)t*K;
        float amax = 0.0f;
        for (int k=0; k<K; k++) {
            const float a = std::fabs(xt[k]);
            if (a > amax) amax = a;
        }
        const float s = amax > 0.0f ? amax/127.0f : 1.0f;
        ascale[t] = s;

        int8_t* q = xq+(size_t)t*Kp;
        const float inv = 1.0f/s;
        for (int k=0; k<K; k++) q[k] = q8(xt[k]*inv);
        for (int k=K; k<Kp; k++) q[k] = 0;
    }
}

void dense_linear_i8(float* out, const float* x, const int8_t* Wq, const float* wscale,
                     const float* bias, int seq, int N, int K,
                     int8_t* xq_scratch, float* ascale_scratch) {
    quantize_act_i8(x, xq_scratch, ascale_scratch, seq, K);
    // The x86 kernel is VNNI-only and SIGILLs without the cpuid check; the
    // nn/ callers gate on it, this entry point has to gate too.
    if (!int8_gemm_available())
        dense_linear_i8_ref(out, xq_scratch, ascale_scratch, Wq, wscale, bias, seq, N, K);
    else
        dense_linear_i8_pre(out, xq_scratch, ascale_scratch, Wq, wscale, bias, seq, N, K);
}

// Portable reference: same arithmetic, no SIMD. Compiled on every backend so the
// tests always have an oracle - the NEON sdot and x86 vpdpbusd kernels must
// agree with it EXACTLY, not approximately, because integer accumulation has no
// reorder freedom.
void dense_linear_i8_ref(float* out, const int8_t* xq, const float* ascale,
                         const int8_t* Wq, const float* wscale, const float* bias,
                         int seq, int N, int K) {
    const int Kp = i8_kpad(K);
    const int KG = Kp/4;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int n=0; n<N; n++) {
        const int b = n/16, j = n%16;
        for (int t=0; t<seq; t++) {
            const int8_t* q = xq+(size_t)t*Kp;
            int32_t acc = 0;
            for (int g=0; g<KG; g++)
                for (int c=0; c<4; c++)
                    acc += (int32_t)Wq[((size_t)(b*KG+g)*16 + j)*4 + c] * (int32_t)q[g*4+c];
            out[(size_t)t*N+n] = std::fma((float)acc*wscale[n], ascale[t], bias ? bias[n] : 0.0f);
        }
    }
}

#if !TCPU_ISA_ARM && !TCPU_ISA_X86
// No vector int8 on this backend: callers keep their fp32 path, and the
// reference stands in for anything that calls the entry point anyway.
bool int8_gemm_available() { return false; }

void dense_linear_i8_pre(float* out, const int8_t* xq, const float* ascale,
                         const int8_t* Wq, const float* wscale, const float* bias,
                         int seq, int N, int K) {
    dense_linear_i8_ref(out, xq, ascale, Wq, wscale, bias, seq, N, K);
}
#endif // !TCPU_ISA_ARM && !TCPU_ISA_X86

} // namespace tcpu
