/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// Portable GEMM pieces shared by every backend: the 16-wide weight packer and
// the reference fallbacks for capabilities a backend does not implement
// natively (K^T-output packed GEMM, bf16 kernels).

#include "../arch.h"
#include "../../ops/lm_ops.h"
#include <cstdint>
#include <cstring>
#include <vector>
#include <cstddef>
using std::size_t;

namespace tcpu {

void pack_weights16(const float* W, float* Wp, int N, int K) {
    for (int b=0; b<N/16; b++)
        for (int k=0; k<K; k++)
            for (int j=0; j<16; j++)
                Wp[((size_t)b*K+k)*16+j] = W[(size_t)(b*16+j)*K+k];
}

#if !TCPU_ISA_X86
// K^T-output packed GEMM: only the AVX2 backend has the in-register transpose
// epilogue; elsewhere compute token-major and copy (callers on these backends
// only take this path when forced - see nn::Linear::kt_native).
void dense_linear_packed_kt(float* out_t, const float* x, const float* Wp, const float* bias,
                            int seq, int N, int K, int ldo) {
    std::vector<float> tmp((size_t)seq*N);
    dense_linear_packed(tmp.data(), x, Wp, bias, seq, N, K);
    for (int t=0; t<seq; t++)
        for (int n=0; n<N; n++)
            out_t[(size_t)n*ldo+t] = tmp[(size_t)t*N+n];
}

// bf16-weight row GEMM: AVX2 has the dequant-block kernel; scalar elsewhere.
void dense_linear_bf16(float* out, const float* x, const uint16_t* W, const float* bias,
                       int seq, int N, int K) {
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int nn=0; nn<N; nn++) {
        const uint16_t* wr = W+(size_t)nn*K;
        for (int t=0; t<seq; t++) {
            const float* xt = x+(size_t)t*K;

            float s = 0.0f;
            for (int k=0; k<K; k++) {
                uint32_t u = (uint32_t)wr[k] << 16;
                float f;
                std::memcpy(&f, &u, 4);
                s += xt[k]*f;
            }

            out[(size_t)t*N+nn] = bias ? s+bias[nn] : s;
        }
    }
}
#endif // !TCPU_ISA_X86

#if !TCPU_HAL_NEON
// bf16 packed-panel GEMM: native only on the Pi NEON backend (the one place it
// was measured to win); reference loop elsewhere.
void dense_linear_packed_bf16(float* out, const float* x, const uint16_t* Wp, const float* bias,
                              int seq, int N, int K) {
    for (int n=0; n<N; n++) {
        const int b = n/16;
        const int j = n%16;
        for (int t=0; t<seq; t++) {
            float s = 0.0f;
            for (int k=0; k<K; k++) {
                uint32_t u = (uint32_t)Wp[((size_t)b*K+k)*16+j] << 16;
                float f;
                std::memcpy(&f, &u, 4);
                s += x[(size_t)t*K+k]*f;
            }

            out[(size_t)t*N+n] = bias ? s+bias[n] : s;
        }
    }
}
#endif // !TCPU_HAL_NEON

} // namespace tcpu
