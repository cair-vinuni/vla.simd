/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "blas.h"
#include "env.h"
#include "../../ops/lm_ops.h"
#include <cstddef>
#include <vector>
using std::size_t;

#if defined(TCPU_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

namespace tcpu {

namespace hal {
bool accel_on() { return accel_available() && env::accel(); }
} // namespace hal

#if defined(TCPU_ACCELERATE)
bool accel_available() { return true; }
void dense_linear_blas(float* out, const float* x, const float* W, const float* bias,
                       int seq, int N, int K) {
    // out[seq,N] = x[seq,K] * W[N,K]^T on the AMX units via Accelerate.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, seq, N, K,
                1.0f, x, K, W, K, 0.0f, out, N);
    if (bias) {
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int t=0; t<seq; t++) {
            float* o = out+(size_t)t*N;
            for (int n=0; n<N; n++)
                o[n] += bias[n];
        }
    }
}
void dense_linear_blas_kt(float* out_t, const float* x, const float* W, const float* bias,
                          int seq, int N, int K, int ldo) {
    // out_t[N, seq] = W[N,K] * x[seq,K]^T straight into the K^T layout.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, seq, K,
                1.0f, W, K, x, K, 0.0f, out_t, ldo);
    if (bias) {
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int n=0; n<N; n++) {
            float* o = out_t+(size_t)n*ldo;
            const float b = bias[n];
            for (int t=0; t<seq; t++)
                o[t] += b;
        }
    }
}
#else
bool accel_available() { return false; }
void dense_linear_blas(float* out, const float* x, const float* W, const float* bias,
                       int seq, int N, int K) {
    dense_linear(out, x, W, bias, seq, N, K);
}
void dense_linear_blas_kt(float* out_t, const float* x, const float* W, const float* bias,
                          int seq, int N, int K, int ldo) {
    std::vector<float> tmp((size_t)seq*N);
    dense_linear(tmp.data(), x, W, bias, seq, N, K);
    for (int t=0; t<seq; t++)
        for (int n=0; n<N; n++)
            out_t[(size_t)n*ldo+t] = tmp[(size_t)t*N+n];
}
#endif

} // namespace tcpu
