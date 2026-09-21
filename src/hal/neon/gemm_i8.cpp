/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// NEON dotprod (ARMv8.2 `asimddp`) W8A8 GEMM - the int8 counterpart of
// dense_linear_packed's fp32 panel kernel, same panel/tile/schedule structure.
//
// Why this exists: a Cortex-A76 (Pi 5) does 2 NEON FMLA/cycle = 16 fp32
// MAC/cycle/core, but `sdot` retires 16 int8 MACs in ONE instruction, so the
// integer path has 4x the arithmetic ceiling AND streams a quarter of the
// weight bytes. On a Cortex-A72 (Pi 4) there is no `asimddp` and the widening
// vmull/vpadal chain eats the win - which is why the earlier Pi port stopped at
// bf16. Availability is a runtime check, not an assumption.
//
// The k-group of 4 is what makes the layout work: `vdotq_s32(acc, w, a)` folds
// 4 int8 products into each of 4 int32 lanes, so one 64-byte weight group holds
// 16 output rows x 4 k, its 4 lanes ARE 4 output rows, and the activation
// operand is the same 4 bytes broadcast into every lane. Per group: 4 weight
// loads + ROWS activation loads + 4*ROWS sdot - against 16 loads + 16*ROWS fmla
// for the same work in fp32.
//
// Built with the baseline -march: the two kernels take +dotprod through a
// function-level target pragma instead, because -mcpu=cortex-a76 measured 8%
// SLOWER end-to-end on this part (GCC's A76 cost model pessimizes the
// hand-written fp32 NEON kernels). See the note in CMakeLists.txt.

#include "../arch.h"
#if TCPU_ISA_ARM

#include "../common/env.h"
#include "../../ops/quant_ops.h"
#include <arm_neon.h>
#include <cstdint>
#include <cstring>
#include <cstddef>
#if defined(__linux__)
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif
using std::size_t;

namespace tcpu {

using hal::env::gemm_chunk;
using hal::env::gemm_force_static;
using hal::env::gemm_threads;

#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif

bool int8_gemm_available() {
    static const bool v = [] {
#if defined(__linux__)
        return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#elif defined(__APPLE__)
        // Every Apple silicon part has FEAT_DotProd (it is mandatory from ARMv8.4
        // and the M1 is v8.5), but ask rather than assume: this is the one check
        // standing between a wrong answer and a SIGILL on some future part.
        int has = 0;
        size_t sz = sizeof(has);
        if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &has, &sz, nullptr, 0) != 0)
            return false;
        return has != 0;
#else
        return false;
#endif
    }();
    return v;
}

#if !TCPU_HAL_APPLE
#pragma GCC push_options
#pragma GCC target ("arch=armv8.2-a+dotprod")
#endif

// ROWS tokens x one 16-row weight panel, C-resident. Live registers:
// 4*ROWS int32 accumulators + 4 weight vectors + 1 activation vector, so
// ROWS=4 -> 21 and ROWS=6 -> 29 of the 32 NEON registers (no spill either way).
template <int ROWS>
static inline void mm_i8_rows16(float* out, const int8_t* xq, const float* ascale,
                                const int8_t* wq, const float* wscale, const float* bias,
                                int N, int Kp, int n0) {
    int32x4_t c0[ROWS], c1[ROWS], c2[ROWS], c3[ROWS];
    for (int i=0; i<ROWS; i++) {
        c0[i] = vdupq_n_s32(0);
        c1[i] = vdupq_n_s32(0);
        c2[i] = vdupq_n_s32(0);
        c3[i] = vdupq_n_s32(0);
    }

    const int KG = Kp/4;
    for (int g=0; g<KG; g++) {
        const int8_t* w = wq+(size_t)g*64;
        const int8x16_t w0 = vld1q_s8(w);
        const int8x16_t w1 = vld1q_s8(w+16);
        const int8x16_t w2 = vld1q_s8(w+32);
        const int8x16_t w3 = vld1q_s8(w+48);
        for (int i=0; i<ROWS; i++) {
            // the group's 4 activation bytes, broadcast into all 4 lanes: each
            // lane then dots them against a different output row's 4 weights
            int32_t a4;
            std::memcpy(&a4, xq+(size_t)i*Kp+(size_t)g*4, 4);
            const int8x16_t a = vreinterpretq_s8_s32(vdupq_n_s32(a4));
            c0[i] = vdotq_s32(c0[i], w0, a);
            c1[i] = vdotq_s32(c1[i], w1, a);
            c2[i] = vdotq_s32(c2[i], w2, a);
            c3[i] = vdotq_s32(c3[i], w3, a);
        }
    }

    // delayed scaling: one (s_a * s_w) per output, never per element
    const float32x4_t ws0 = vld1q_f32(wscale+n0);
    const float32x4_t ws1 = vld1q_f32(wscale+n0+4);
    const float32x4_t ws2 = vld1q_f32(wscale+n0+8);
    const float32x4_t ws3 = vld1q_f32(wscale+n0+12);
    const float32x4_t bb0 = bias ? vld1q_f32(bias+n0)    : vdupq_n_f32(0);
    const float32x4_t bb1 = bias ? vld1q_f32(bias+n0+4)  : vdupq_n_f32(0);
    const float32x4_t bb2 = bias ? vld1q_f32(bias+n0+8)  : vdupq_n_f32(0);
    const float32x4_t bb3 = bias ? vld1q_f32(bias+n0+12) : vdupq_n_f32(0);
    for (int i=0; i<ROWS; i++) {
        const float32x4_t sa = vdupq_n_f32(ascale[i]);
        float* o = out+(size_t)i*N+n0;
        vst1q_f32(o,    vfmaq_f32(bb0, vmulq_f32(vcvtq_f32_s32(c0[i]), ws0), sa));
        vst1q_f32(o+4,  vfmaq_f32(bb1, vmulq_f32(vcvtq_f32_s32(c1[i]), ws1), sa));
        vst1q_f32(o+8,  vfmaq_f32(bb2, vmulq_f32(vcvtq_f32_s32(c2[i]), ws2), sa));
        vst1q_f32(o+12, vfmaq_f32(bb3, vmulq_f32(vcvtq_f32_s32(c3[i]), ws3), sa));
    }
}

static inline void mm_i8_tile(float* out, const int8_t* xq, const float* ascale,
                              const int8_t* wq, const float* wscale, const float* bias,
                              int rows, int N, int Kp, int n0) {
    switch (rows) {
        case 6: mm_i8_rows16<6>(out, xq, ascale, wq, wscale, bias, N, Kp, n0); break;
        case 5: mm_i8_rows16<5>(out, xq, ascale, wq, wscale, bias, N, Kp, n0); break;
        case 4: mm_i8_rows16<4>(out, xq, ascale, wq, wscale, bias, N, Kp, n0); break;
        case 3: mm_i8_rows16<3>(out, xq, ascale, wq, wscale, bias, N, Kp, n0); break;
        case 2: mm_i8_rows16<2>(out, xq, ascale, wq, wscale, bias, N, Kp, n0); break;
        default: mm_i8_rows16<1>(out, xq, ascale, wq, wscale, bias, N, Kp, n0); break;
    }
}

#if !TCPU_HAL_APPLE
#pragma GCC pop_options
#endif

void dense_linear_i8_pre(float* out, const int8_t* xq, const float* ascale,
                         const int8_t* Wq, const float* wscale, const float* bias,
                         int seq, int N, int K) {
    const int Kp = i8_kpad(K);
    const int MR = hal::env::i8_mr();
    const int nblocks = N/16;
    const size_t pstride = (size_t)(Kp/4)*64;   // bytes of one 16-row weight panel

    // Cache-block the token axis. Without this the (panel, tile) loop sweeps the
    // WHOLE activation matrix once per weight panel: ACT's 602x3200 down-
    // projection re-reads 1.9 MB for each of 32 panels = 61 MB per GEMM, which
    // is what capped the int8 kernel at ~2.2x fp32 instead of the ~4x the
    // instruction ratio allows. Holding a block of tokens resident while every
    // panel sweeps it trades that for (seq/MB) passes over the much smaller
    // weights: 8 MB + 1.9 MB at MB=128. TCPU_I8_MBLOCK=0 disables (A/B hook).
    const int mb = hal::env::i8_mblock();
    if (mb > 0 && seq > mb) {
        for (int t00=0; t00<seq; t00+=mb) {
            const int rows_here = seq-t00 < mb ? seq-t00 : mb;
            dense_linear_i8_pre(out+(size_t)t00*N, xq+(size_t)t00*Kp, ascale+t00,
                                Wq, wscale, bias, rows_here, N, K);
        }
        return;
    }
    const int mtiles = (seq+MR-1)/MR;

#if defined(_OPENMP)
    // Same schedule policy as the fp32 packed GEMM: dynamic with whole-panel
    // chunks, which won on these homogeneous-but-jittery cores. Bodies are
    // duplicated rather than shared through a lambda for the inlining reason
    // documented in dense_linear_packed.
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
                    mm_i8_tile(out+(size_t)t0*N, xq+(size_t)t0*Kp, ascale+t0,
                               Wq+(size_t)b*pstride, wscale, bias, rows, N, Kp, b*16);
                }
            }
        } else {
            #pragma omp parallel for schedule(dynamic, chunk) collapse(2)
            for (int b=0; b<nblocks; b++) {
                for (int m=0; m<mtiles; m++) {
                    const int t0   = m*MR;
                    const int rows = seq-t0 < MR ? seq-t0 : MR;
                    mm_i8_tile(out+(size_t)t0*N, xq+(size_t)t0*Kp, ascale+t0,
                               Wq+(size_t)b*pstride, wscale, bias, rows, N, Kp, b*16);
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
            mm_i8_tile(out+(size_t)t0*N, xq+(size_t)t0*Kp, ascale+t0,
                       Wq+(size_t)b*pstride, wscale, bias, rows, N, Kp, b*16);
        }
    }
}

} // namespace tcpu

#endif // TCPU_ISA_ARM
