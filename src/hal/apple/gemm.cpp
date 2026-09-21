/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// Apple-Silicon NEON GEMM backend (M4 tuning, verbatim from the m4 branch).
// These are the hand kernels used when Accelerate routing is off (TCPU_ACCEL=0)
// or for shapes the model does not send to BLAS; with TCPU_ACCEL on (default),
// Gemm/Mlp-role linears go to the AMX units via hal/blas instead. 6x16
// C-resident packed micro-kernel, static schedule.

#include "../arch.h"
#if TCPU_HAL_APPLE

#include "../simd.h"
#include "../common/env.h"
#include "../../ops/lm_ops.h"
#include <cstdint>
#include <cstring>
#include <vector>
#include <cstddef>
using std::size_t;

namespace tcpu {
using hal::env::gemm_threads;
#if defined(_OPENMP)
namespace {
inline int gemm_chunk4() {
    const int c = hal::env::gemm_chunk();
    return c > 0 ? c : 4;
}
}
#endif

// NEON mirror of the AVX2 4-row micro-kernel: 4 rows x 2 chains (K unrolled by 8)
// = 8 independent FMA chains at width 4.
static inline void mm_block4(float* out, const float* x,
                             const float* w0, const float* w1, const float* w2, const float* w3,
                             int seq, int N, int K, const float* bias, int n0) {
    for (int t=0; t<seq; t++) {
        const float* xt = x+(size_t)t*K;
        float32x4_t a0 = vdupq_n_f32(0);
        float32x4_t a1 = vdupq_n_f32(0);
        float32x4_t a2 = vdupq_n_f32(0);
        float32x4_t a3 = vdupq_n_f32(0);
        float32x4_t b0 = vdupq_n_f32(0);
        float32x4_t b1 = vdupq_n_f32(0);
        float32x4_t b2 = vdupq_n_f32(0);
        float32x4_t b3 = vdupq_n_f32(0);

        int k = 0;
        for (; k+8<=K; k+=8) {
            const float32x4_t x0 = vld1q_f32(xt+k);
            const float32x4_t x1 = vld1q_f32(xt+k+4);
            a0 = vfmaq_f32(a0, x0, vld1q_f32(w0+k));
            b0 = vfmaq_f32(b0, x1, vld1q_f32(w0+k+4));
            a1 = vfmaq_f32(a1, x0, vld1q_f32(w1+k));
            b1 = vfmaq_f32(b1, x1, vld1q_f32(w1+k+4));
            a2 = vfmaq_f32(a2, x0, vld1q_f32(w2+k));
            b2 = vfmaq_f32(b2, x1, vld1q_f32(w2+k+4));
            a3 = vfmaq_f32(a3, x0, vld1q_f32(w3+k));
            b3 = vfmaq_f32(b3, x1, vld1q_f32(w3+k+4));
        }
        for (; k+4<=K; k+=4) {
            const float32x4_t xv = vld1q_f32(xt+k);
            a0 = vfmaq_f32(a0, xv, vld1q_f32(w0+k));
            a1 = vfmaq_f32(a1, xv, vld1q_f32(w1+k));
            a2 = vfmaq_f32(a2, xv, vld1q_f32(w2+k));
            a3 = vfmaq_f32(a3, xv, vld1q_f32(w3+k));
        }

        float s0 = vaddvq_f32(vaddq_f32(a0, b0));
        float s1 = vaddvq_f32(vaddq_f32(a1, b1));
        float s2 = vaddvq_f32(vaddq_f32(a2, b2));
        float s3 = vaddvq_f32(vaddq_f32(a3, b3));
        for (; k<K; k++) {
            float xk = xt[k];
            s0 += xk*w0[k];
            s1 += xk*w1[k];
            s2 += xk*w2[k];
            s3 += xk*w3[k];
        }

        float* o = out + (size_t)t*N + n0;
        o[0] = bias ? s0+bias[n0+0] : s0;
        o[1] = bias ? s1+bias[n0+1] : s1;
        o[2] = bias ? s2+bias[n0+2] : s2;
        o[3] = bias ? s3+bias[n0+3] : s3;
    }
}

void dense_linear(float* out, const float* x, const float* W, const float* bias,
                  int seq, int N, int K) {
    // Register-blocked micro-kernel; threaded over row-blocks. Sums are reordered vs a plain
    // dot -> within the fp32 noise floor (see CONVENTIONS: SIMD FMA ops may reorder sums).
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

// NEON ROWSx16 C-resident micro-kernel: the B panel is 4 vectors per k, the C tile
// is ROWS x 4 accumulators (24 for ROWS=6; +4 panel loads + 1 broadcast = 29 of the
// 32 NEON registers). Per k: ROWS scalar loads + 4 vector loads vs 4*ROWS FMAs ->
// FMA-bound. Same template-ROWS discipline as the AVX2 kernel (a runtime row bound
// spills the accumulator array to stack).
template <int ROWS>
static inline void mm_pack_rows16(float* out, const float* x, const float* wp,
                                  const float* bias, int N, int K, int n0) {
    float32x4_t c[ROWS][4];
    for (int i=0; i<ROWS; i++)
        for (int j=0; j<4; j++) c[i][j] = vdupq_n_f32(0);

    for (int k=0; k<K; k++) {
        const float* w = wp+(size_t)k*16;
        const float32x4_t b0 = vld1q_f32(w);
        const float32x4_t b1 = vld1q_f32(w+4);
        const float32x4_t b2 = vld1q_f32(w+8);
        const float32x4_t b3 = vld1q_f32(w+12);
        for (int i=0; i<ROWS; i++) {
            const float a = x[(size_t)i*K+k];
            c[i][0] = vfmaq_n_f32(c[i][0], b0, a);
            c[i][1] = vfmaq_n_f32(c[i][1], b1, a);
            c[i][2] = vfmaq_n_f32(c[i][2], b2, a);
            c[i][3] = vfmaq_n_f32(c[i][3], b3, a);
        }
    }

    float32x4_t bb[4];
    for (int j=0; j<4; j++)
        bb[j] = bias ? vld1q_f32(bias+n0+4*j) : vdupq_n_f32(0);
    for (int i=0; i<ROWS; i++)
        for (int j=0; j<4; j++)
            vst1q_f32(out+(size_t)i*N+n0+4*j, vaddq_f32(c[i][j], bb[j]));
}

static inline void mm_pack6x16(float* out, const float* x, const float* wp,
                               const float* bias, int rows, int N, int K, int n0) {
    switch (rows) {
        case 6: mm_pack_rows16<6>(out, x, wp, bias, N, K, n0); break;
        case 5: mm_pack_rows16<5>(out, x, wp, bias, N, K, n0); break;
        case 4: mm_pack_rows16<4>(out, x, wp, bias, N, K, n0); break;
        case 3: mm_pack_rows16<3>(out, x, wp, bias, N, K, n0); break;
        case 2: mm_pack_rows16<2>(out, x, wp, bias, N, K, n0); break;
        default: mm_pack_rows16<1>(out, x, wp, bias, N, K, n0); break;
    }
}

void dense_linear_packed(float* out, const float* x, const float* Wp, const float* bias,
                         int seq, int N, int K) {
    constexpr int MR = 6;
    const int nblocks = N/16;
    const int mtiles  = (seq+MR-1)/MR;
    auto tile = [&](int b, int m) {
        const int t0   = m*MR;
        const int rows = seq-t0 < MR ? seq-t0 : MR;
        mm_pack6x16(out+(size_t)t0*N, x+(size_t)t0*K, Wp+(size_t)b*K*16,
                    bias, rows, N, K, b*16);
    };
#if defined(_OPENMP)
    const int gt = gemm_threads();
    if (gt > 0) {
        const int chunk = gemm_chunk4();
        #pragma omp parallel for schedule(dynamic, chunk) collapse(2) num_threads(gt)
        for (int b=0; b<nblocks; b++)
            for (int m=0; m<mtiles; m++) tile(b, m);
        return;
    }
    #pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int b=0; b<nblocks; b++) {
        for (int m=0; m<mtiles; m++) tile(b, m);
    }
}

} // namespace tcpu

#endif // TCPU_HAL_APPLE
