/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// Attention ops shared by every backend: unmasked GQA and T5's per-head-bias
// MHA (dot/axpy over the backend primitives; HAVE_SIMD falls back to plain
// loops on scalar), plus the scalar reference for the masked op.

#include "../arch.h"
#include "../simd.h"
#include "../../ops/lm_ops.h"
#include <cmath>
#include <limits>
#include <vector>
#include <cstddef>
using std::size_t;

namespace tcpu {

#if TCPU_HAL_X86
void gqa_attention_dense(float* out, const float* Q, const float* K, const float* V,
                         int seq_q, int seq_k, int n_q, int n_kv, int head_dim,
                         float scale, const float* K_pre) {
    gqa_attention_masked(out, Q, K, V, seq_q, seq_k, n_q, n_kv, head_dim,
                         scale, nullptr, K_pre);
}
#elif !TCPU_HAL_NEON
// Backends without a dedicated dense kernel keep their masked path exactly as it
// was: the zero mask they used to be handed is now built here instead of by the
// caller, so the arithmetic and the measured numbers are unchanged.
void gqa_attention_dense(float* out, const float* Q, const float* K, const float* V,
                         int seq_q, int seq_k, int n_q, int n_kv, int head_dim,
                         float scale, const float* K_pre) {
    static thread_local std::vector<float> zero;
    if (zero.size() < (size_t)seq_q*seq_k) zero.assign((size_t)seq_q*seq_k, 0.0f);
    gqa_attention_masked(out, Q, K, V, seq_q, seq_k, n_q, n_kv, head_dim,
                         scale, zero.data(), K_pre);
}
#endif

void mha_attention_bias(float* out, const float* Q, const float* K, const float* V,
                        int seq_q, int seq_k, int n_heads, int head_dim,
                        float scale, const float* bias) {
    const float BLOCK = std::numeric_limits<float>::lowest();   // see gqa_attention_masked
    const float NINF  = -std::numeric_limits<float>::infinity();
#if defined(_OPENMP)
    #pragma omp parallel
#endif
  {
    std::vector<float> scores(seq_k);   // per-thread scratch (reused across queries)
#if defined(_OPENMP)
    #pragma omp for schedule(static)
#endif
    for (int qi=0; qi<n_heads*seq_q; qi++) {
        const int h  = qi/seq_q;
        const int t1 = qi%seq_q;
        const float* q = Q+((size_t)t1*n_heads+h)*head_dim;
        const float* brow = bias+((size_t)h*seq_q+t1)*seq_k;

        float maxs = NINF;
        for (int t2=0; t2<seq_k; t2++) {
            if (brow[t2] == BLOCK) {
                scores[t2] = NINF;
                continue;
            }
            const float* k = K+((size_t)t2*n_heads+h)*head_dim;
#if defined(HAVE_SIMD)
            float dot = simd_dot(q, k, head_dim)*scale + brow[t2];
#else
            float dot = 0.0f;
            for (int d=0; d<head_dim; d++)
                dot += q[d]*k[d];
            dot = dot*scale + brow[t2];
#endif
            scores[t2] = dot;
            if (dot > maxs) maxs = dot;
        }

        float sum = 0.0f;
        if (maxs == NINF) {
            for (int t2=0; t2<seq_k; t2++)
                scores[t2] = 1.0f;
            sum = (float)seq_k;
        } else {
            for (int t2=0; t2<seq_k; t2++) {
                float s = scores[t2] == NINF ? 0.0f : std::exp(scores[t2]-maxs);
                scores[t2] = s;
                sum += s;
            }
        }

        float inv = 1.0f/sum;
        float* o = out+((size_t)t1*n_heads+h)*head_dim;
        for (int d=0; d<head_dim; d++)
            o[d] = 0.0f;

        for (int t2=0; t2<seq_k; t2++) {
            if (scores[t2] == 0.0f) continue;
            const float* vv = V+((size_t)t2*n_heads+h)*head_dim;
#if defined(HAVE_SIMD)
            simd_axpy(o, scores[t2]*inv, vv, head_dim);
#else
            float a = scores[t2]*inv;
            for (int d=0; d<head_dim; d++)
                o[d] += a*vv[d];
#endif
        }
    }
  }
}

} // namespace tcpu
