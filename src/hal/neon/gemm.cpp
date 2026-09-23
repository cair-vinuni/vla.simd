/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pi / generic ARM NEON GEMM backend (raspi tuning, verbatim from the raspi
// branch; Cortex-A72-class cores). MR=4 lane-FMLA packed micro-kernel (no
// spills in 32 regs), dynamic whole-panel scheduling by default
// (TCPU_GEMM_SCHED=static reverts), opt-in bf16 packed kernel for
// memory-bound MLPs (TCPU_BF16_MLP).

#include "../arch.h"
#if TCPU_HAL_NEON

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
using hal::env::gemm_chunk;
using hal::env::gemm_force_static;

// 4-output-row micro-kernel (NEON analogue of mm_block4): one block of 4 weight
// rows, 4 independent FMLA chains -> hides FMLA latency; sums reordered vs a plain
// dot (within the fp32 noise floor, see CONVENTIONS).
static inline void mm_block4_neon(float* out, const float* x,
                                  const float* w0, const float* w1, const float* w2, const float* w3,
                                  int seq, int N, int K, const float* bias, int n0) {
    for (int t=0; t<seq; t++) {
        const float* xt = x+(size_t)t*K;
        float32x4_t a0 = vdupq_n_f32(0);
        float32x4_t a1 = vdupq_n_f32(0);
        float32x4_t a2 = vdupq_n_f32(0);
        float32x4_t a3 = vdupq_n_f32(0);

        int k = 0;
        for (; k+4<=K; k+=4) {
            const float32x4_t xv = vld1q_f32(xt+k);
            a0 = vfmaq_f32(a0, xv, vld1q_f32(w0+k));
            a1 = vfmaq_f32(a1, xv, vld1q_f32(w1+k));
            a2 = vfmaq_f32(a2, xv, vld1q_f32(w2+k));
            a3 = vfmaq_f32(a3, xv, vld1q_f32(w3+k));
        }

        float s0 = vaddvq_f32(a0);
        float s1 = vaddvq_f32(a1);
        float s2 = vaddvq_f32(a2);
        float s3 = vaddvq_f32(a3);
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
// ROWSx16 C-resident packed micro-kernel (NEON analogue of mm_pack_rows16): the
// 16-wide weight column of block b for input k is contiguous (4 NEON vectors).
// Per k: 4 weight loads + ROWS broadcasts + 4*ROWS FMLAs -> more rows amortise the
// weight loads (A72 is load-limited). ROWS is a template constant so the 4*ROWS
// accumulators unroll into registers (32 NEON regs; ROWS<=6 => 24 acc + 5 fits).
template <int ROWS>
static inline void mm_pack_rows16_neon(float* out, const float* x, const float* wp,
                                       const float* bias, int N, int K, int n0) {
    float32x4_t c0[ROWS];
    float32x4_t c1[ROWS];
    float32x4_t c2[ROWS];
    float32x4_t c3[ROWS];
    for (int i=0; i<ROWS; i++) {
        c0[i] = vdupq_n_f32(0);
        c1[i] = vdupq_n_f32(0);
        c2[i] = vdupq_n_f32(0);
        c3[i] = vdupq_n_f32(0);
    }
    // Load 4 k-values of each x row into a vector, then FMLA-by-lane (vfmaq_laneq):
    // the x operand comes from a register lane instead of a memory broadcast (LD1R),
    // cutting x-operand load-unit uops ~4x on the load-limited A72. 4 weight vectors
    // (16 cols) are shared across all ROWS rows, so more rows amortise the w loads.
    // Same k order and same c += w * x[k] per lane -> bit-identical to the scalar path.
    int k = 0;
#define PACK_FMA_LANE(WOFF, LANE) do {                                        \
        const float32x4_t b0 = vld1q_f32(w + (WOFF)),      b1 = vld1q_f32(w + (WOFF) + 4);  \
        const float32x4_t b2 = vld1q_f32(w + (WOFF) + 8),  b3 = vld1q_f32(w + (WOFF) + 12); \
        for (int i = 0; i < ROWS; i++) {                                      \
            c0[i] = vfmaq_laneq_f32(c0[i], b0, xr[i], (LANE));                \
            c1[i] = vfmaq_laneq_f32(c1[i], b1, xr[i], (LANE));                \
            c2[i] = vfmaq_laneq_f32(c2[i], b2, xr[i], (LANE));                \
            c3[i] = vfmaq_laneq_f32(c3[i], b3, xr[i], (LANE));                \
        }                                                                     \
    } while (0)
    for (; k+4<=K; k+=4) {
        float32x4_t xr[ROWS];
        for (int i=0; i<ROWS; i++) xr[i] = vld1q_f32(x+(size_t)i*K+k);
        const float* w = wp+(size_t)k*16;
        PACK_FMA_LANE(0,  0);
        PACK_FMA_LANE(16, 1);
        PACK_FMA_LANE(32, 2);
        PACK_FMA_LANE(48, 3);
    }
#undef PACK_FMA_LANE
    for (; k<K; k++) {   // K tail (K % 4 != 0, e.g. conv im2col strips)
        const float32x4_t b0 = vld1q_f32(wp+(size_t)k*16);
        const float32x4_t b1 = vld1q_f32(wp+(size_t)k*16+4);
        const float32x4_t b2 = vld1q_f32(wp+(size_t)k*16+8);
        const float32x4_t b3 = vld1q_f32(wp+(size_t)k*16+12);
        for (int i=0; i<ROWS; i++) {
            const float32x4_t a = vdupq_n_f32(x[(size_t)i*K+k]);
            c0[i] = vfmaq_f32(c0[i], a, b0);
            c1[i] = vfmaq_f32(c1[i], a, b1);
            c2[i] = vfmaq_f32(c2[i], a, b2);
            c3[i] = vfmaq_f32(c3[i], a, b3);
        }
    }
    const float32x4_t bb0 = bias ? vld1q_f32(bias+n0)    : vdupq_n_f32(0);
    const float32x4_t bb1 = bias ? vld1q_f32(bias+n0+4)  : vdupq_n_f32(0);
    const float32x4_t bb2 = bias ? vld1q_f32(bias+n0+8)  : vdupq_n_f32(0);
    const float32x4_t bb3 = bias ? vld1q_f32(bias+n0+12) : vdupq_n_f32(0);
    for (int i=0; i<ROWS; i++) {
        float* o = out + (size_t)i*N + n0;
        vst1q_f32(o,    vaddq_f32(c0[i], bb0));
        vst1q_f32(o+4,  vaddq_f32(c1[i], bb1));
        vst1q_f32(o+8,  vaddq_f32(c2[i], bb2));
        vst1q_f32(o+12, vaddq_f32(c3[i], bb3));
    }
}
static inline void mm_pack6x16_neon(float* out, const float* x, const float* wp,
                                    const float* bias, int rows, int N, int K, int n0) {
    switch (rows) {
        case 6: mm_pack_rows16_neon<6>(out, x, wp, bias, N, K, n0); break;
        case 5: mm_pack_rows16_neon<5>(out, x, wp, bias, N, K, n0); break;
        case 4: mm_pack_rows16_neon<4>(out, x, wp, bias, N, K, n0); break;
        case 3: mm_pack_rows16_neon<3>(out, x, wp, bias, N, K, n0); break;
        case 2: mm_pack_rows16_neon<2>(out, x, wp, bias, N, K, n0); break;
        default: mm_pack_rows16_neon<1>(out, x, wp, bias, N, K, n0); break;
    }
}
// bf16 weights variant of mm_pack_rows16_neon: weights are the top 16 bits of the
// fp32, so widening is just a shift-left-16 (vshll_n_u16) into the fp32 bit pattern.
// Halves the weight bytes read - a win only when the GEMM is memory-hierarchy-bound
// (the Octo MLP is: it does not scale past 2 threads). Adds ~4 vshll per k vs fp32.
template <int ROWS>
static inline void mm_pack_rows16_bf16_neon(float* out, const float* x, const uint16_t* wp,
                                            const float* bias, int N, int K, int n0) {
    float32x4_t c0[ROWS];
    float32x4_t c1[ROWS];
    float32x4_t c2[ROWS];
    float32x4_t c3[ROWS];
    for (int i=0; i<ROWS; i++) {
        c0[i] = vdupq_n_f32(0);
        c1[i] = vdupq_n_f32(0);
        c2[i] = vdupq_n_f32(0);
        c3[i] = vdupq_n_f32(0);
    }
    int k = 0;
#define PACK_FMA_LANE_BF16(WOFF, LANE) do {                                     \
        const uint16x8_t wlo = vld1q_u16(w + (WOFF));                           \
        const uint16x8_t whi = vld1q_u16(w + (WOFF) + 8);                       \
        const float32x4_t b0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(wlo),  16)); \
        const float32x4_t b1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(wlo), 16)); \
        const float32x4_t b2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(whi),  16)); \
        const float32x4_t b3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(whi), 16)); \
        for (int i = 0; i < ROWS; i++) {                                        \
            c0[i] = vfmaq_laneq_f32(c0[i], b0, xr[i], (LANE));                  \
            c1[i] = vfmaq_laneq_f32(c1[i], b1, xr[i], (LANE));                  \
            c2[i] = vfmaq_laneq_f32(c2[i], b2, xr[i], (LANE));                  \
            c3[i] = vfmaq_laneq_f32(c3[i], b3, xr[i], (LANE));                  \
        }                                                                       \
    } while (0)
    for (; k+4<=K; k+=4) {
        float32x4_t xr[ROWS];
        for (int i=0; i<ROWS; i++) xr[i] = vld1q_f32(x+(size_t)i*K+k);
        const uint16_t* w = wp+(size_t)k*16;
        PACK_FMA_LANE_BF16(0,  0);
        PACK_FMA_LANE_BF16(16, 1);
        PACK_FMA_LANE_BF16(32, 2);
        PACK_FMA_LANE_BF16(48, 3);
    }
#undef PACK_FMA_LANE_BF16
    for (; k<K; k++) {
        const uint16x8_t wlo = vld1q_u16(wp+(size_t)k*16);
        const uint16x8_t whi = vld1q_u16(wp+(size_t)k*16+8);
        const float32x4_t b0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(wlo),  16));
        const float32x4_t b1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(wlo), 16));
        const float32x4_t b2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(whi),  16));
        const float32x4_t b3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(whi), 16));
        for (int i=0; i<ROWS; i++) {
            const float32x4_t a = vdupq_n_f32(x[(size_t)i*K+k]);
            c0[i] = vfmaq_f32(c0[i], a, b0);
            c1[i] = vfmaq_f32(c1[i], a, b1);
            c2[i] = vfmaq_f32(c2[i], a, b2);
            c3[i] = vfmaq_f32(c3[i], a, b3);
        }
    }
    const float32x4_t bb0 = bias ? vld1q_f32(bias+n0)    : vdupq_n_f32(0);
    const float32x4_t bb1 = bias ? vld1q_f32(bias+n0+4)  : vdupq_n_f32(0);
    const float32x4_t bb2 = bias ? vld1q_f32(bias+n0+8)  : vdupq_n_f32(0);
    const float32x4_t bb3 = bias ? vld1q_f32(bias+n0+12) : vdupq_n_f32(0);
    for (int i=0; i<ROWS; i++) {
        float* o = out + (size_t)i*N + n0;
        vst1q_f32(o,    vaddq_f32(c0[i], bb0));
        vst1q_f32(o+4,  vaddq_f32(c1[i], bb1));
        vst1q_f32(o+8,  vaddq_f32(c2[i], bb2));
        vst1q_f32(o+12, vaddq_f32(c3[i], bb3));
    }
}

void dense_linear(float* out, const float* x, const float* W, const float* bias,
                  int seq, int N, int K) {
    constexpr int NR = 4;
    const int nblocks = (N+NR-1)/NR;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int b=0; b<nblocks; b++) {
        const int n0 = b*NR;
        if (n0+NR <= N) {
            mm_block4_neon(out, x, W+(size_t)n0*K, W+(size_t)(n0+1)*K, W+(size_t)(n0+2)*K, W+(size_t)(n0+3)*K,
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

void dense_linear_packed(float* out, const float* x, const float* Wp, const float* bias,
                         int seq, int N, int K) {
    // NEON C-resident kernel over 16-wide weight blocks x MR token rows. Packing
    // (pack_weights16) only produces N/16 full blocks, so - like the AVX2 path -
    // callers guarantee N % 16 == 0 (octo: D=384, mlp=1536, stem_dim=512).
    // MR=4 measured fastest on the A72: live regs = 4*ROWS acc + ROWS xr + 4 weight
    // = 5*ROWS+4 = 24, well under the 32 NEON regs so nothing spills. MR=5 (29) and
    // MR=6 (34, spills) both benchmarked slower despite better weight-load amortisation.
    constexpr int MR = 4;
    const int nblocks = N/16;
    const int mtiles  = (seq+MR-1)/MR;
#if defined(_OPENMP)
    // Default schedule on ARM is dynamic with WHOLE-PANEL chunks (one 16-wide weight
    // panel = mtiles iterations per dequeue): on the Pi's 4 homogeneous but
    // OS-jittery cores, idle threads steal panels from a preempted straggler - won
    // 8/8 interleaved A/B pairs vs static, -8% end-to-end, bit-exact (scheduling
    // does not touch per-tile math). Small chunks lose B-panel locality (measured
    // worse, same as x86). On hybrid x86 dynamic lost (E-core imbalance), so the
    // AVX2 path above keeps static. TCPU_GEMM_SCHED=static reverts (A/B hook);
    // TCPU_GEMM_THREADS/CHUNK override the team size / chunk as on x86.
    // Bodies are duplicated, NOT a shared lambda: a [&] lambda in this loop cost
    // ~100 ms end-to-end (capture struct blocks inlining in the outlined region).
    if (!gemm_force_static()) {
        const int ce = gemm_chunk();
        const int chunk = (ce > 0 && ce < mtiles) ? ce : mtiles;
        const int gt = gemm_threads();
        if (gt > 0) {
            #pragma omp parallel for schedule(dynamic, chunk) collapse(2) num_threads(gt)
            for (int b=0; b<nblocks; b++) {
                for (int m=0; m<mtiles; m++) {
                    const int t0   = m*MR;
                    const int rows = seq-t0 < MR ? seq-t0 : MR;
                    mm_pack6x16_neon(out+(size_t)t0*N, x+(size_t)t0*K,
                                     Wp+(size_t)b*K*16, bias, rows, N, K, b*16);
                }
            }
        } else {
            #pragma omp parallel for schedule(dynamic, chunk) collapse(2)
            for (int b=0; b<nblocks; b++) {
                for (int m=0; m<mtiles; m++) {
                    const int t0   = m*MR;
                    const int rows = seq-t0 < MR ? seq-t0 : MR;
                    mm_pack6x16_neon(out+(size_t)t0*N, x+(size_t)t0*K,
                                     Wp+(size_t)b*K*16, bias, rows, N, K, b*16);
                }
            }
        }
        return;
    }
    #pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int b=0; b<nblocks; b++) {
        for (int m=0; m<mtiles; m++) {
            const int t0   = m*MR;
            const int rows = seq-t0 < MR ? seq-t0 : MR;
            mm_pack6x16_neon(out+(size_t)t0*N, x+(size_t)t0*K, Wp+(size_t)b*K*16,
                             bias, rows, N, K, b*16);
        }
    }
}

void dense_linear_packed_bf16(float* out, const float* x, const uint16_t* Wp, const float* bias,
                              int seq, int N, int K) {
    constexpr int MR = 4;
    const int nblocks = N/16;
    const int mtiles  = (seq+MR-1)/MR;
#if defined(_OPENMP)
    // dynamic whole-panel schedule by default, same rationale + hooks as the fp32
    // NEON path in dense_linear_packed (bodies duplicated, not a lambda).
    if (!gemm_force_static()) {
        const int ce = gemm_chunk();
        const int chunk = (ce > 0 && ce < mtiles) ? ce : mtiles;
        #pragma omp parallel for schedule(dynamic, chunk) collapse(2)
        for (int b=0; b<nblocks; b++) {
            for (int m=0; m<mtiles; m++) {
                const int t0   = m*MR;
                const int rows = seq-t0 < MR ? seq-t0 : MR;

                float* o           = out+(size_t)t0*N;
                const float* xa    = x+(size_t)t0*K;
                const uint16_t* wp = Wp+(size_t)b*K*16;

                switch (rows) {
                    case 4: mm_pack_rows16_bf16_neon<4>(o, xa, wp, bias, N, K, b*16); break;
                    case 3: mm_pack_rows16_bf16_neon<3>(o, xa, wp, bias, N, K, b*16); break;
                    case 2: mm_pack_rows16_bf16_neon<2>(o, xa, wp, bias, N, K, b*16); break;
                    default: mm_pack_rows16_bf16_neon<1>(o, xa, wp, bias, N, K, b*16); break;
                }
            }
        }
        return;
    }
    #pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int b=0; b<nblocks; b++) {
        for (int m=0; m<mtiles; m++) {
            const int t0   = m*MR;
            const int rows = seq-t0 < MR ? seq-t0 : MR;

            float* o           = out+(size_t)t0*N;
            const float* xa    = x+(size_t)t0*K;
            const uint16_t* wp = Wp+(size_t)b*K*16;

            switch (rows) {
                case 4: mm_pack_rows16_bf16_neon<4>(o, xa, wp, bias, N, K, b*16); break;
                case 3: mm_pack_rows16_bf16_neon<3>(o, xa, wp, bias, N, K, b*16); break;
                case 2: mm_pack_rows16_bf16_neon<2>(o, xa, wp, bias, N, K, b*16); break;
                default: mm_pack_rows16_bf16_neon<1>(o, xa, wp, bias, N, K, b*16); break;
            }
        }
    }
}

} // namespace tcpu

#endif // TCPU_HAL_NEON
