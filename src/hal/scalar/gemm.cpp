/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// Scalar GEMM backend: the reference loops every SIMD kernel is validated
// against (and the -U__ARM_NEON / non-SIMD control build).

#include "../arch.h"
#if TCPU_HAL_SCALAR

#include "../../ops/lm_ops.h"
#include <cstdint>
#include <cstring>
#include <cstddef>
using std::size_t;

namespace tcpu {

void dense_linear(float* out, const float* x, const float* W, const float* bias,
                  int seq, int N, int K) {
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int nn=0; nn<N; nn++) {
        const float* wr = W+(size_t)nn*K;
        for (int t=0; t<seq; t++) {
            const float* xt = x+(size_t)t*K;

            float s = 0.0f;
            for (int k=0; k<K; k++)
                s += xt[k]*wr[k];

            out[(size_t)t*N+nn] = bias ? s+bias[nn] : s;
        }
    }
}

void dense_linear_packed(float* out, const float* x, const float* Wp, const float* bias,
                         int seq, int N, int K) {
    // scalar fallback: unpack block layout on the fly
    for (int n=0; n<N; n++) {
        const int b = n/16;
        const int j = n%16;
        for (int t=0; t<seq; t++) {
            float s = 0.0f;
            for (int k=0; k<K; k++)
                s += x[(size_t)t*K+k]*Wp[((size_t)b*K+k)*16+j];

            out[(size_t)t*N+n] = bias ? s+bias[n] : s;
        }
    }
}

} // namespace tcpu

#endif // TCPU_HAL_SCALAR
