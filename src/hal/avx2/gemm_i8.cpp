/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// AVX-VNNI (`vpdpbusd`) W8A8 GEMM: the x86 counterpart of neon/gemm_i8.cpp over
// the same packed layout [N/16][Kp/4][16][4].
//
// AVX2 does 16 fp32 MAC/cycle; one vpdpbusd retires 32 int8 MACs and two issue
// per cycle, so 4x the arithmetic ceiling on a quarter of the weight bytes. A
// 64-byte k-group (16 rows x 4 k) is two YMM of 8 dwords, so one group feeds two
// vpdpbusd whose 8 lanes are 8 output rows.
//
// Signedness differs from NEON: `sdot` is signed x signed, `vpdpbusd` is
// unsigned x signed. Activations are biased to unsigned and corrected with the
// weight row sums (i8_rowsums), precomputed at pack time:
//
//     x_u = x_q + 128                        (byte-wise XOR 0x80)
//     sum_k w*x_q = vpdpbusd(x_u, w) - 128 * sum_k w
//
// One subtract per output in the epilogue, nothing in the k-loop. Both sides
// still span 8 bits, so accuracy is unchanged.
//
// Range: |acc| <= K*127*255, |128*rowsum| <= K*127*128. At the widest K here
// (SmolVLA's 12288 modality projection) the corrected value stays under 6e8.
//
// AVX-VNNI arrived on Alder Lake / Zen 4, so it is taken per-function through a
// target attribute behind a cpuid check, not as a global flag.

#include "../arch.h"
#if TCPU_ISA_X86

#include "../common/env.h"
#include "../../ops/quant_ops.h"
#include <immintrin.h>
#include <cpuid.h>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstddef>
using std::size_t;

namespace tcpu {

bool int8_gemm_available() {
    static const bool v = [] {
        unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
        // CPUID.(EAX=7, ECX=1):EAX[4] = AVX-VNNI (the VEX-encoded form).
        if (!__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx)) return false;
        return (eax & (1u << 4)) != 0;
    }();
    return v;
}

#define VLA_VNNI __attribute__((target("avx2,avxvnni")))

// ROWS tokens x one 16-row weight panel, C-resident.
template <int ROWS>
VLA_VNNI static void mm_i8_rows16(float* out, const int8_t* xq, const float* ascale,
                                  const int8_t* wq, const float* wscale, const float* bias,
                                  const int32_t* rowsum, int N, int Kp, int n0) {
    __v8si c0[ROWS], c1[ROWS];
    for (int i=0; i<ROWS; i++) {
        c0[i] = (__v8si)_mm256_setzero_si256();
        c1[i] = (__v8si)_mm256_setzero_si256();
    }

    const int KG = Kp/4;
    const __m256i k80 = _mm256_set1_epi32((int)0x80808080u);
#pragma GCC unroll 1
    for (int g=0; g<KG; g++) {
        const int8_t* w = wq+(size_t)g*64;
        const __m256i w0 = _mm256_loadu_si256((const __m256i*)w);        // rows 0..7
        const __m256i w1 = _mm256_loadu_si256((const __m256i*)(w+32));   // rows 8..15
        for (int i=0; i<ROWS; i++) {
            // the group's 4 activation bytes, biased to unsigned (XOR 0x80 per
            // byte) and broadcast into all 8 dwords; each dword dots them
            // against a different output row's 4 weights
            uint32_t a4;
            std::memcpy(&a4, xq+(size_t)i*Kp+(size_t)g*4, 4);
            const __m256i a = _mm256_xor_si256(_mm256_set1_epi32((int)a4), k80);
            c0[i] = (__v8si)_mm256_dpbusd_avx_epi32((__m256i)c0[i], a, w0);
            c1[i] = (__v8si)_mm256_dpbusd_avx_epi32((__m256i)c1[i], a, w1);
        }
    }

    // delayed scaling: one (s_a * s_w) per output, after the 128*rowsum
    // correction that undoes the activation bias
    const __m256i r0 = _mm256_loadu_si256((const __m256i*)(rowsum+n0));
    const __m256i r1 = _mm256_loadu_si256((const __m256i*)(rowsum+n0+8));
    const __m256i b0 = _mm256_slli_epi32(r0, 7);
    const __m256i b1 = _mm256_slli_epi32(r1, 7);
    const __m256 ws0 = _mm256_loadu_ps(wscale+n0);
    const __m256 ws1 = _mm256_loadu_ps(wscale+n0+8);
    const __m256 bb0 = bias ? _mm256_loadu_ps(bias+n0)   : _mm256_setzero_ps();
    const __m256 bb1 = bias ? _mm256_loadu_ps(bias+n0+8) : _mm256_setzero_ps();
    for (int i=0; i<ROWS; i++) {
        const __m256 sa = _mm256_set1_ps(ascale[i]);
        float* o = out+(size_t)i*N+n0;
        const __m256 v0 = _mm256_cvtepi32_ps(_mm256_sub_epi32((__m256i)c0[i], b0));
        const __m256 v1 = _mm256_cvtepi32_ps(_mm256_sub_epi32((__m256i)c1[i], b1));
        _mm256_storeu_ps(o,   _mm256_fmadd_ps(_mm256_mul_ps(v0, ws0), sa, bb0));
        _mm256_storeu_ps(o+8, _mm256_fmadd_ps(_mm256_mul_ps(v1, ws1), sa, bb1));
    }
}

VLA_VNNI static void mm_i8_tile(float* out, const int8_t* xq, const float* ascale,
                                const int8_t* wq, const float* wscale, const float* bias,
                                const int32_t* rowsum, int rows, int N, int Kp, int n0) {
    switch (rows) {
        case 6: mm_i8_rows16<6>(out, xq, ascale, wq, wscale, bias, rowsum, N, Kp, n0); break;
        case 5: mm_i8_rows16<5>(out, xq, ascale, wq, wscale, bias, rowsum, N, Kp, n0); break;
        case 4: mm_i8_rows16<4>(out, xq, ascale, wq, wscale, bias, rowsum, N, Kp, n0); break;
        case 3: mm_i8_rows16<3>(out, xq, ascale, wq, wscale, bias, rowsum, N, Kp, n0); break;
        case 2: mm_i8_rows16<2>(out, xq, ascale, wq, wscale, bias, rowsum, N, Kp, n0); break;
        default: mm_i8_rows16<1>(out, xq, ascale, wq, wscale, bias, rowsum, N, Kp, n0); break;
    }
}

#undef VLA_VNNI

void dense_linear_i8_pre(float* out, const int8_t* xq, const float* ascale,
                         const int8_t* Wq, const float* wscale, const float* bias,
                         int seq, int N, int K) {
    // vpdpbusd is per-function, so calling this without the cpuid check SIGILLs.
    assert(int8_gemm_available() && "dense_linear_i8_pre needs AVX-VNNI");
    const int Kp = i8_kpad(K);
    const int MR = hal::env::i8_mr();
    const int nblocks = N/16;
    const size_t pstride = (size_t)(Kp/4)*64;   // bytes of one 16-row weight panel
    const int32_t* rowsum = i8_rowsums(Wq, N, K);

    // Cache-block the token axis, as in the NEON kernel: otherwise the
    // (panel, tile) loop sweeps the whole activation matrix once per weight
    // panel. TCPU_I8_MBLOCK=0 disables.
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
    #pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int b=0; b<nblocks; b++) {
        for (int m=0; m<mtiles; m++) {
            const int t0   = m*MR;
            const int rows = seq-t0 < MR ? seq-t0 : MR;
            mm_i8_tile(out+(size_t)t0*N, xq+(size_t)t0*Kp, ascale+t0,
                       Wq+(size_t)b*pstride, wscale, bias, rowsum, rows, N, Kp, b*16);
        }
    }
}

} // namespace tcpu

#endif // TCPU_ISA_X86
