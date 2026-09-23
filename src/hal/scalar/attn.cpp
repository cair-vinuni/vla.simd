/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// Scalar masked attention: the reference loop every SIMD engine is validated
// against (per-key mask skip, libm exp, plain dots).

#include "../arch.h"
#if TCPU_HAL_SCALAR

#include "../../ops/lm_ops.h"
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>
using std::size_t;

namespace tcpu {

void gqa_attention_masked(float* out, const float* Q, const float* K, const float* V,
                          int seq_q, int seq_k, int n_q, int n_kv, int head_dim,
                          float scale, const float* mask, const float*) {
    const int group = n_q/n_kv;
    const float NINF  = -std::numeric_limits<float>::infinity();
    const float BLOCK = std::numeric_limits<float>::lowest();
#if defined(_OPENMP)
    #pragma omp parallel
#endif
  {
    std::vector<float> scores(seq_k);   // per-thread scratch (reused across queries)
#if defined(_OPENMP)
    #pragma omp for schedule(static)
#endif
    for (int qi=0; qi<n_q*seq_q; qi++) {
        const int h  = qi/seq_q;
        const int t1 = qi%seq_q;
        const int kv = h/group;
        const float* q = Q+((size_t)t1*n_q+h)*head_dim;
        const float* mrow = mask+(size_t)t1*seq_k;

        float maxs = NINF;
        for (int t2=0; t2<seq_k; t2++) {
            if (mrow[t2] == BLOCK) {
                scores[t2] = NINF;
                continue;
            }
            const float* k = K+((size_t)t2*n_kv+kv)*head_dim;

            float dot = 0.0f;
            for (int d=0; d<head_dim; d++)
                dot += q[d]*k[d];
            dot = dot*scale + mrow[t2];

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
        float* o = out+((size_t)t1*n_q+h)*head_dim;
        for (int d=0; d<head_dim; d++)
            o[d] = 0.0f;

        for (int t2=0; t2<seq_k; t2++) {
            if (scores[t2] == 0.0f) continue;
            const float* vv = V+((size_t)t2*n_kv+kv)*head_dim;
            float a = scores[t2]*inv;
            for (int d=0; d<head_dim; d++)
                o[d] += a*vv[d];
        }
    }
  }
}

} // namespace tcpu

#endif // TCPU_HAL_SCALAR
