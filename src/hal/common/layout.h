/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "../arch.h"
#include <cstddef>
#include <limits>

// Buffer geometry the backend wants. Pure functions of the shape, so producer
// (nn::Linear::forward_kt) and consumer (gqa_attention_masked) derive the same
// value independently - no extra parameter on the op API.

namespace tcpu {
namespace hal {

// Leading dimension of the pre-transposed key buffer K^T [n_kv*head_dim][ldk]
// (rows are head-dim components, columns are keys). Callers allocate
// n_kv*head_dim*kt_stride(seq_k) and leave columns [seq_k, ldk) zeroed.
//
// The QK micro-kernel walks one column group down all head_dim rows, i.e. it
// strides by ldk floats. At seq 1024 the natural stride is 4096 B, which on
// Zen 3 (32 KB, 8-way, 64 sets) maps ALL 64 head-dim rows onto the same L1 set:
// only 8 of the 64 lines it needs can be resident, so the tile re-misses every
// pass. Padding the stride by one vector breaks the aliasing - measured on a
// Ryzen 5 5500 at seq 1024, 6 threads: QK 389 -> 528 GF/s (4.14 -> 3.05 ms).
// Intel's 48 KB / 12-way L1 absorbs the same pattern, so every other backend
// keeps the exact stride it shipped with.
inline int kt_stride(int seq_k) {
    const int skp = (seq_k+7) & ~7;
#if TCPU_HAL_AMD
    return (skp & 511) == 0 ? skp+8 : skp;
#else
    return skp;
#endif
}

inline void transpose_kt(float* kt, const float* K, int seq_k, int n_kv, int head_dim, int ldk) {
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int jb=0; jb<ldk; jb += 8) {
        const int rem = seq_k-jb;
        const int je  = rem < 0 ? 0 : (rem < 8 ? rem : 8);
        const int jz  = ldk-jb < 8 ? ldk-jb : 8;
        for (int kv=0; kv<n_kv; kv++)
            for (int d=0; d<head_dim; d++) {
                float* dst = kt+((std::size_t)kv*head_dim+d)*ldk+jb;
                for (int j=0; j<je; j++)
                    dst[j] = K[((std::size_t)(jb+j)*n_kv+kv)*head_dim+d];
                for (int j=je; j<jz; j++)
                    dst[j] = 0.0f;
            }
    }
}

inline int row_bound(const float* mrow, int seq_k) {
    while (seq_k > 0 && mrow[seq_k-1] == std::numeric_limits<float>::lowest())
        seq_k--;
    return seq_k;
}

} // namespace hal
} // namespace tcpu
