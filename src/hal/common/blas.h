/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Vendor-BLAS GEMM routing (Apple Accelerate -> AMX matrix units). Compiled to
// real sgemm calls only when TCPU_ACCELERATE is defined (Apple targets); the
// fallbacks keep callers link-clean elsewhere. fp32 reorder class (BLAS
// accumulation order). Callers gate on accel_available() + env::accel().

namespace tcpu {

// True when a vendor BLAS is linked (Accelerate on Apple).
bool accel_available();

namespace hal {
// Linked AND enabled (TCPU_ACCEL) - the routing predicate callers use.
bool accel_on();
} // namespace hal

// out[seq,N] = x[seq,K] * W[N,K]^T (+bias). Same signature/layout as dense_linear.
void dense_linear_blas(float* out, const float* x, const float* W, const float* bias,
                       int seq, int N, int K);

// Same GEMM, output stored TRANSPOSED: out_t[n*ldo + t] (attention's K^T layout).
// One sgemm call computes W x^T, no separate transpose pass. Columns [seq, ldo)
// are left untouched (callers keep the buffer zeroed).
void dense_linear_blas_kt(float* out_t, const float* x, const float* W, const float* bias,
                          int seq, int N, int K, int ldo);

} // namespace tcpu
