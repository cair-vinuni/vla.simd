/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// x86 AVX2 GEMM backend (i9 tuning). Packed-panel 6x16 C-resident
// micro-kernel, in-register K^T transpose epilogue, static schedule.
//
// Shared by BOTH x86 backends (TCPU_HAL_X86), Intel and AMD Zen: the 6x16 tile
// is FMA-bound and both uarchs retire 2x256-bit FMA per cycle, so the Intel
// tuning already lands at 93% of a Ryzen 5 5500's pinned FMA peak (measured:
// 713 GF/s on 768x768x1024, 625-645 GF/s on the 3072-wide MLP shapes). Nothing
// Zen-specific was found to add here.

#include "../arch.h"
#if TCPU_HAL_X86

#include "../simd.h"
#include "../../ops/lm_ops.h"
#include <cstdint>
#include <cstring>
#include <vector>
#include <cstddef>
using std::size_t;

namespace tcpu {

// 4-output-row micro-kernel: for one block of 4 weight rows (w0..w3, each length K, fp32),
// compute out[t, n0..n0+3] for all t. 4 rows x 2 accumulators (K unrolled by 2) = 8
// independent FMA chains -> hides FMA latency. Weights come either straight from a fp32
// tensor (dense_linear) or from a dequantized bf16 scratch (dense_linear_bf16).
static inline void mm_block4(float* out, const float* x,
                             const float* w0, const float* w1, const float* w2, const float* w3,
                             int seq, int N, int K, const float* bias, int n0) {
    for (int t=0; t<seq; t++) {
        const float* xt = x+(size_t)t*K;
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps();
        __m256 b0 = _mm256_setzero_ps();
        __m256 b1 = _mm256_setzero_ps();
        __m256 b2 = _mm256_setzero_ps();
        __m256 b3 = _mm256_setzero_ps();

        int k = 0;
        for (; k+16 <= K; k += 16) {
            __m256 x0 = _mm256_loadu_ps(xt+k);
            __m256 x1 = _mm256_loadu_ps(xt+k+8);
            a0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w0+k),   a0);
            b0 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(w0+k+8), b0);
            a1 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w1+k),   a1);
            b1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(w1+k+8), b1);
            a2 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w2+k),   a2);
            b2 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(w2+k+8), b2);
            a3 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w3+k),   a3);
            b3 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(w3+k+8), b3);
        }
        for (; k+8 <= K; k += 8) {
            __m256 xv = _mm256_loadu_ps(xt+k);
            a0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w0+k), a0);
            a1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w1+k), a1);
            a2 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w2+k), a2);
            a3 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w3+k), a3);
        }

        float s0 = hsum8(_mm256_add_ps(a0, b0));
        float s1 = hsum8(_mm256_add_ps(a1, b1));
        float s2 = hsum8(_mm256_add_ps(a2, b2));
        float s3 = hsum8(_mm256_add_ps(a3, b3));
        for (; k < K; k++) {
            float xk = xt[k];
            s0 += xk*w0[k];
            s1 += xk*w1[k];
            s2 += xk*w2[k];
            s3 += xk*w3[k];
        }

        float* o = out+(size_t)t*N+n0;
        o[0] = bias ? s0+bias[n0+0] : s0;
        o[1] = bias ? s1+bias[n0+1] : s1;
        o[2] = bias ? s2+bias[n0+2] : s2;
        o[3] = bias ? s3+bias[n0+3] : s3;
    }
}
// bf16 (top 16 bits of fp32) -> fp32 row.
static inline void bf16_to_f32(const uint16_t* src, float* dst, int n) {
    int i = 0;
    for (; i+8 <= n; i += 8) {
        __m256i w = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(src+i)));
        _mm256_storeu_ps(dst+i, _mm256_castsi256_ps(_mm256_slli_epi32(w, 16)));
    }
    for (; i < n; i++)
        dst[i] = bf16_f32(src[i]);
}

void dense_linear(float* out, const float* x, const float* W, const float* bias,
                  int seq, int N, int K) {
    // Register-blocked micro-kernel; threaded over row-blocks. Sums are reordered vs a plain
    // dot -> within the fp32 noise floor (SIMD FMA ops may reorder sums).
    constexpr int NR = 4;
    const int nblocks = (N+NR-1)/NR;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int b=0; b<nblocks; b++) {
        const int n0 = b*NR;
        if (n0+NR <= N) {
            mm_block4(out, x, W+(size_t)n0*K, W+(size_t)(n0+1)*K, W+(size_t)(n0+2)*K, W+(size_t)(n0+3)*K,
                      seq, N, K, bias, n0);
        } else {
            for (int j=0; n0+j < N; j++) {
                const float* wr = W+(size_t)(n0+j)*K;
                for (int t=0; t<seq; t++) {
                    float s = simd_dot(x+(size_t)t*K, wr, K);
                    out[(size_t)t*N+n0+j] = bias ? s+bias[n0+j] : s;
                }
            }
        }
    }
}

// gelu_tanh on 8 lanes, same op sequence as the eltwise gelu_tanh vector path
// (hal/common/eltwise.cpp) -> bit-identical values.
static inline __m256 gelu_tanh8(__m256 v) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    const __m256 v3 = _mm256_mul_ps(_mm256_mul_ps(v, v), v);
    const __m256 y = _mm256_mul_ps(_mm256_set1_ps(2.0f*c),
                                   _mm256_fmadd_ps(_mm256_set1_ps(0.044715f), v3, v));
    const __m256 e = exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), y));
    const __m256 s = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(_mm256_set1_ps(1.0f), e));
    return _mm256_mul_ps(v, s);
}

// ROWSx16 C-resident micro-kernel: up to 12 accumulators + 2 B-panel loads + 1
// broadcast = 15 YMM. Per k: ROWS+2 load-port uops vs 2*ROWS FMAs -> FMA-bound (the
// row-blocked kernel is load-bound). ROWS is a template constant so the accumulator
// array fully unrolls into registers (a runtime bound spills it to stack).
// GELU applies gelu_tanh8 in the epilogue (bit-identical to a separate gelu pass).
// ADD accumulates into out instead of overwriting (residual fused into the store;
// the same x + (c+bias) add the separate residual pass performed).
template <int ROWS, bool GELU, bool ADD>
static inline void mm_pack_rows16(float* out, const float* x, const float* wp,
                                  const float* bias, int N, int K, int n0) {
    __m256 c0[ROWS];
    __m256 c1[ROWS];
    for (int i=0; i<ROWS; i++) {
        c0[i] = _mm256_setzero_ps();
        c1[i] = _mm256_setzero_ps();
    }

    for (int k=0; k<K; k++) {
        const __m256 b0 = _mm256_loadu_ps(wp+(size_t)k*16);
        const __m256 b1 = _mm256_loadu_ps(wp+(size_t)k*16+8);
        for (int i=0; i<ROWS; i++) {
            const __m256 a = _mm256_set1_ps(x[(size_t)i*K+k]);
            c0[i] = _mm256_fmadd_ps(a, b0, c0[i]);
            c1[i] = _mm256_fmadd_ps(a, b1, c1[i]);
        }
    }

    const __m256 bb0 = bias ? _mm256_loadu_ps(bias+n0)   : _mm256_setzero_ps();
    const __m256 bb1 = bias ? _mm256_loadu_ps(bias+n0+8) : _mm256_setzero_ps();
    for (int i=0; i<ROWS; i++) {
        __m256 r0 = _mm256_add_ps(c0[i], bb0);
        __m256 r1 = _mm256_add_ps(c1[i], bb1);
        if (GELU) {
            r0 = gelu_tanh8(r0);
            r1 = gelu_tanh8(r1);
        }
        if (ADD) {
            r0 = _mm256_add_ps(_mm256_loadu_ps(out+(size_t)i*N+n0),   r0);
            r1 = _mm256_add_ps(_mm256_loadu_ps(out+(size_t)i*N+n0+8), r1);
        }
        _mm256_storeu_ps(out+(size_t)i*N+n0,   r0);
        _mm256_storeu_ps(out+(size_t)i*N+n0+8, r1);
    }
}

template <bool GELU, bool ADD>
static inline void mm_pack6x16(float* out, const float* x, const float* wp,
                               const float* bias, int rows, int N, int K, int n0) {
    switch (rows) {
        case 6: mm_pack_rows16<6, GELU, ADD>(out, x, wp, bias, N, K, n0); break;
        case 5: mm_pack_rows16<5, GELU, ADD>(out, x, wp, bias, N, K, n0); break;
        case 4: mm_pack_rows16<4, GELU, ADD>(out, x, wp, bias, N, K, n0); break;
        case 3: mm_pack_rows16<3, GELU, ADD>(out, x, wp, bias, N, K, n0); break;
        case 2: mm_pack_rows16<2, GELU, ADD>(out, x, wp, bias, N, K, n0); break;
        default: mm_pack_rows16<1, GELU, ADD>(out, x, wp, bias, N, K, n0); break;
    }
}

// standard AVX2 8x8 fp32 transpose
static inline void transpose8(__m256 r[8]) {
    __m256 t0 = _mm256_unpacklo_ps(r[0], r[1]);
    __m256 t1 = _mm256_unpackhi_ps(r[0], r[1]);
    __m256 t2 = _mm256_unpacklo_ps(r[2], r[3]);
    __m256 t3 = _mm256_unpackhi_ps(r[2], r[3]);
    __m256 t4 = _mm256_unpacklo_ps(r[4], r[5]);
    __m256 t5 = _mm256_unpackhi_ps(r[4], r[5]);
    __m256 t6 = _mm256_unpacklo_ps(r[6], r[7]);
    __m256 t7 = _mm256_unpackhi_ps(r[6], r[7]);

    __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
    __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
    __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
    __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
    __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    r[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
    r[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
    r[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
    r[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
    r[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
    r[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
    r[6] = _mm256_permute2f128_ps(s2, s6, 0x31);
    r[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
}

// as mm_pack_rows16, but the C tile is transposed in registers and stored to the
// [n][t] layout (bias added per n-row after the transpose; same value per element,
// same k accumulation order -> values identical to the token-major kernel).
template <int ROWS>
static inline void mm_pack_rows16_kt(float* out_t, const float* x, const float* wp,
                                     const float* bias, int K, int n0, int t0, int ldo) {
    __m256 c0[ROWS];
    __m256 c1[ROWS];
    for (int i=0; i<ROWS; i++) {
        c0[i] = _mm256_setzero_ps();
        c1[i] = _mm256_setzero_ps();
    }

    for (int k=0; k<K; k++) {
        const __m256 b0 = _mm256_loadu_ps(wp+(size_t)k*16);
        const __m256 b1 = _mm256_loadu_ps(wp+(size_t)k*16+8);
        for (int i=0; i<ROWS; i++) {
            const __m256 a = _mm256_set1_ps(x[(size_t)i*K+k]);
            c0[i] = _mm256_fmadd_ps(a, b0, c0[i]);
            c1[i] = _mm256_fmadd_ps(a, b1, c1[i]);
        }
    }

    __m256 r[8];
    for (int half=0; half<2; half++) {
        __m256* c = half ? c1 : c0;
        for (int i=0; i<ROWS; i++)
            r[i] = c[i];
        for (int i=ROWS; i<8; i++)
            r[i] = _mm256_setzero_ps();
        transpose8(r);

        for (int j=0; j<8; j++) {
            const int n = n0+half*8+j;
            const __m256 row = bias ? _mm256_add_ps(r[j], _mm256_set1_ps(bias[n])) : r[j];
            float* dst = out_t+(size_t)n*ldo+t0;
            const __m128 lo = _mm256_castps256_ps128(row);
            const __m128 hi = _mm256_extractf128_ps(row, 1);
            if constexpr (ROWS >= 4) {
                _mm_storeu_ps(dst, lo);
                std::memcpy(dst+4, &hi, (ROWS-4)*sizeof(float));
            } else {
                std::memcpy(dst, &lo, ROWS*sizeof(float));
            }
        }
    }
}

void dense_linear_packed_kt(float* out_t, const float* x, const float* Wp, const float* bias,
                            int seq, int N, int K, int ldo) {
    constexpr int MR = 6;
    const int nblocks = N/16;
    const int mtiles = (seq+MR-1)/MR;
    auto tile = [&](int b, int m) {
        const int t0 = m*MR;
        const int rows = seq-t0 < MR ? seq-t0 : MR;
        const float* xa = x+(size_t)t0*K;
        const float* wp = Wp+(size_t)b*K*16;
        switch (rows) {
            case 6: mm_pack_rows16_kt<6>(out_t, xa, wp, bias, K, b*16, t0, ldo); break;
            case 5: mm_pack_rows16_kt<5>(out_t, xa, wp, bias, K, b*16, t0, ldo); break;
            case 4: mm_pack_rows16_kt<4>(out_t, xa, wp, bias, K, b*16, t0, ldo); break;
            case 3: mm_pack_rows16_kt<3>(out_t, xa, wp, bias, K, b*16, t0, ldo); break;
            case 2: mm_pack_rows16_kt<2>(out_t, xa, wp, bias, K, b*16, t0, ldo); break;
            default: mm_pack_rows16_kt<1>(out_t, xa, wp, bias, K, b*16, t0, ldo); break;
        }
    };
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int b=0; b<nblocks; b++) {
        for (int m=0; m<mtiles; m++)
            tile(b, m);
    }
}

template <bool GELU, bool ADD>
static void dense_linear_packed_impl(float* out, const float* x, const float* Wp, const float* bias,
                                     int seq, int N, int K) {
    constexpr int MR = 6;
    const int nblocks = N/16;
    const int mtiles = (seq+MR-1)/MR;
    auto tile = [&](int b, int m) {
        const int t0 = m*MR;
        const int rows = seq-t0 < MR ? seq-t0 : MR;
        mm_pack6x16<GELU, ADD>(out+(size_t)t0*N, x+(size_t)t0*K, Wp+(size_t)b*K*16,
                               bias, rows, N, K, b*16);
    };
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int b=0; b<nblocks; b++) {
        for (int m=0; m<mtiles; m++)
            tile(b, m);
    }
}

void dense_linear_packed(float* out, const float* x, const float* Wp, const float* bias,
                         int seq, int N, int K) {
    dense_linear_packed_impl<false, false>(out, x, Wp, bias, seq, N, K);
}

void dense_linear_packed_gelu(float* out, const float* x, const float* Wp, const float* bias,
                              int seq, int N, int K) {
    dense_linear_packed_impl<true, false>(out, x, Wp, bias, seq, N, K);
}

void dense_linear_packed_add(float* out, const float* x, const float* Wp, const float* bias,
                             int seq, int N, int K) {
    dense_linear_packed_impl<false, true>(out, x, Wp, bias, seq, N, K);
}

void dense_linear_bf16(float* out, const float* x, const uint16_t* W, const float* bias,
                       int seq, int N, int K) {
    constexpr int NR = 4;
    const int nblocks = (N+NR-1)/NR;
#if defined(_OPENMP)
    #pragma omp parallel
#endif
  {
    std::vector<float> wf((size_t)NR*K);   // per-thread dequant scratch (one block)
#if defined(_OPENMP)
    #pragma omp for schedule(static)
#endif
    for (int b=0; b<nblocks; b++) {
        const int n0 = b*NR;
        if (n0+NR <= N) {
            for (int j=0; j<NR; j++)
                bf16_to_f32(W+(size_t)(n0+j)*K, wf.data()+(size_t)j*K, K);
            mm_block4(out, x, wf.data(), wf.data()+K, wf.data()+2*(size_t)K, wf.data()+3*(size_t)K,
                      seq, N, K, bias, n0);
        } else {
            for (int j=0; n0+j < N; j++) {
                float* wr = wf.data();
                bf16_to_f32(W+(size_t)(n0+j)*K, wr, K);
                for (int t=0; t<seq; t++) {
                    float s = simd_dot(x+(size_t)t*K, wr, K);
                    out[(size_t)t*N+n0+j] = bias ? s+bias[n0+j] : s;
                }
            }
        }
    }
  }
}

} // namespace tcpu

#endif // TCPU_HAL_X86
