/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// Token-wise norms + RoPE. Portable scalar+OMP bodies, identical on every
// backend (per-token double accumulation; no SIMD reorder to worry about).

#include <vector>
#include "env.h"
#include "../../ops/lm_ops.h"
#include <cmath>
#include <cstddef>
using std::size_t;

namespace tcpu {

void rmsnorm(float* out, const float* x, const float* w, int seq, int hidden, float eps) {
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if((size_t)seq*hidden > (size_t)hal::env::omp_min())
#endif
    for (int t=0; t<seq; t++) {
        const float* xt = x+(size_t)t*hidden;
        float* ot = out+(size_t)t*hidden;

        double ss = 0.0;
        for (int h=0; h<hidden; h++)
            ss += (double)xt[h]*xt[h];

        float inv = (float)(1.0/std::sqrt(ss/hidden + eps));
        for (int h=0; h<hidden; h++)
            ot[h] = xt[h]*inv*w[h];
    }
}

void rope_neox(float* x, const int* pos, int seq, int n_heads, int head_dim, float base) {
    const int half = head_dim/2;
    const double theta_scale = std::pow((double)base, -2.0/head_dim);
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if((size_t)seq*n_heads*head_dim > (size_t)hal::env::omp_min())
#endif
    for (int t=0; t<seq; t++) {
        // theta depends on (t, i) only: one table per token, reused across heads.
        // Declared inside the loop so each OMP worker gets its own, not the master's.
        static thread_local std::vector<double> cs;
        if ((int)cs.size() < 2*half) cs.resize(2*half);
        double* cst = cs.data();

        double theta = (double)pos[t];
        for (int i=0; i<half; i++) {
            cst[2*i]   = std::cos(theta);
            cst[2*i+1] = std::sin(theta);
            theta *= theta_scale;
        }

        for (int hh=0; hh<n_heads; hh++) {
            float* v = x+((size_t)t*n_heads+hh)*head_dim;
            for (int i=0; i<half; i++) {
                const double c = cst[2*i], s = cst[2*i+1];
                const float x0 = v[i], x1 = v[i+half];
                v[i]      = (float)(x0*c - x1*s);
                v[i+half] = (float)(x0*s + x1*c);
            }
        }
    }
}

void layernorm(float* out, const float* x, const float* w, const float* b, int seq, int hidden, float eps) {
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if((size_t)seq*hidden > (size_t)hal::env::omp_min())
#endif
    for (int t=0; t<seq; t++) {
        const float* xt = x+(size_t)t*hidden;
        float* ot = out+(size_t)t*hidden;

        double mean = 0.0;
        for (int h=0; h<hidden; h++)
            mean += xt[h];
        mean /= hidden;

        double var = 0.0;
        for (int h=0; h<hidden; h++) {
            double d = xt[h]-mean;
            var += d*d;
        }

        float inv = (float)(1.0/std::sqrt(var/hidden + eps));
        if (b) {
            for (int h=0; h<hidden; h++)
                ot[h] = ((float)(xt[h]-mean))*inv*w[h] + b[h];
        } else {
            for (int h=0; h<hidden; h++)
                ot[h] = ((float)(xt[h]-mean))*inv*w[h];
        }
    }
}

} // namespace tcpu
