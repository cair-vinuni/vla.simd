/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// Apple-Silicon masked attention (M4 tuning).
// Two engines: per-head sgemm on the AMX units (TCPU_ATTN_BLAS, needs the
// pre-transposed K panel) and the key-vectorized NEON path with dynamic
// query-row scheduling (TCPU_ATTN_DYNAMIC; hybrid P/E cores).

#include "../arch.h"
#if TCPU_HAL_APPLE

#include "../simd.h"
#include "../common/env.h"
#include "../common/layout.h"
#include "../../ops/lm_ops.h"
#include <cmath>
#include <limits>
#include <vector>
#include <cstddef>
using std::size_t;

#if defined(TCPU_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

namespace tcpu {

void gqa_attention_masked(float* out, const float* Q, const float* K, const float* V,
                          int seq_q, int seq_k, int n_q, int n_kv, int head_dim,
                          float scale, const float* mask, const float* K_pre) {
    const int group = n_q / n_kv;
    const float NINF = -std::numeric_limits<float>::infinity();
#if defined(TCPU_ACCELERATE)
    // TCPU_ATTN_BLAS=1: QK and AV as per-head AMX sgemm (strided A/B/C views into
    // the [t][h][d] tensors), with the masked softmax vectorized in between.
    // Needs K_pre ([kv][d][key], ld = skp). fp32 reorder class.
    const bool attn_blas = hal::env::attn_blas();
    if (attn_blas && K_pre) {
        const int skp0 = (seq_k+7) & ~7;
        static thread_local std::vector<float> S;
        static thread_local std::vector<float> sums;
        if (S.size() < (size_t)seq_q*seq_k) S.resize((size_t)seq_q*seq_k);
        if (sums.size() < (size_t)seq_q) sums.resize(seq_q);
        float* const Sp   = S.data();
        float* const sump = sums.data();

        // jmax depends only on the mask row, not the head: scan once per call
        static thread_local std::vector<int> jmx;
        if (jmx.size() < (size_t)seq_q) jmx.resize(seq_q);
        int* const jmp = jmx.data();
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int t1=0; t1<seq_q; t1++)
            jmp[t1] = hal::row_bound(mask+(size_t)t1*seq_k, seq_k);

        for (int h=0; h<n_q; h++) {
            const int kv = h/group;
            // scores = scale * Q_h [seq_q, d] x K^T_h [d, seq_k]
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, seq_q, seq_k, head_dim,
                        scale, Q+(size_t)h*head_dim, n_q*head_dim,
                        K_pre+(size_t)kv*head_dim*skp0, skp0,
                        0.0f, Sp, seq_k);
#if defined(_OPENMP)
            #pragma omp parallel for schedule(dynamic, 8)
#endif
            for (int t1=0; t1<seq_q; t1++) {
                float* srow = Sp+(size_t)t1*seq_k;
                const float* mrow = mask+(size_t)t1*seq_k;
                const int jmax = jmp[t1];
                if (jmax == 0) {   // fully-masked row -> uniform (dense semantics)
                    for (int j=0; j<seq_k; j++) srow[j] = 1.0f;
                    sump[t1] = (float)seq_k;
                    continue;
                }

                float maxs = NINF;
                int j = 0;
#if defined(__ARM_NEON)
                const int jv = jmax & ~3;
                float32x4_t vmax = vdupq_n_f32(NINF);
                for (; j<jv; j+=4) {
                    const float32x4_t s = vaddq_f32(vld1q_f32(srow+j), vld1q_f32(mrow+j));
                    vst1q_f32(srow+j, s);
                    vmax = vmaxq_f32(vmax, s);
                }
                maxs = vmaxvq_f32(vmax);
#endif
                for (; j<jmax; j++) {
                    srow[j] += mrow[j];
                    if (srow[j] > maxs) maxs = srow[j];
                }

                float sum = 0.0f;
                if (maxs <= std::numeric_limits<float>::lowest()) {
                    for (j=0; j<seq_k; j++) srow[j] = 1.0f;
                    sum = (float)seq_k;
                    sump[t1] = sum;
                    continue;
                }
                j = 0;
#if defined(__ARM_NEON)
                const float32x4_t vm = vdupq_n_f32(maxs);
                float32x4_t vsum = vdupq_n_f32(0);
                for (; j<jv; j+=4) {
                    const float32x4_t e = exp_ps(vsubq_f32(vld1q_f32(srow+j), vm));
                    vst1q_f32(srow+j, e);
                    vsum = vaddq_f32(vsum, e);
                }
                sum = vaddvq_f32(vsum);
#endif
                for (; j<jmax; j++) {
                    const float e = std::exp(srow[j]-maxs);
                    srow[j] = e;
                    sum += e;
                }
                for (j=jmax; j<seq_k; j++) srow[j] = 0.0f;   // blocked tail
                sump[t1] = sum;
            }

            // out_h = P [seq_q, seq_k] x V_h [seq_k, d]; rows scaled by 1/sum after
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, seq_q, head_dim, seq_k,
                        1.0f, Sp, seq_k, V+(size_t)kv*head_dim, n_kv*head_dim,
                        0.0f, out+(size_t)h*head_dim, n_q*head_dim);
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static)
#endif
            for (int t1=0; t1<seq_q; t1++) {
                float* o = out+((size_t)t1*n_q+h)*head_dim;
                const float inv = 1.0f/sump[t1];
                for (int d=0; d<head_dim; d++) o[d] *= inv;
            }
        }
        return;
    }
#endif
    // Key-vectorized path at NEON width: K^T [kv][d][key] built once (or taken
    // from K_pre), scores for 16 keys per pass with the q broadcast coming from a
    // lane of one 4-float load (vfmaq_laneq_f32), softmax max/exp/sum vectorized
    // with exp_ps.
    const int skp = (seq_k+7) & ~7;
    const float* KTp = K_pre;
    if (!KTp) {
        // NOTE: grab the pointer BEFORE the parallel regions - inside them each
        // OpenMP worker would see its own (empty) thread_local instance.
        static thread_local std::vector<float> KT;
        if (KT.size() < (size_t)n_kv*head_dim*skp) KT.resize((size_t)n_kv*head_dim*skp);
        hal::transpose_kt(KT.data(), K, seq_k, n_kv, head_dim, skp);
        KTp = KT.data();
    }
    // TCPU_ATTN_DYNAMIC=N runs the query-row loop schedule(dynamic, N): rows have
    // uneven work (block-causal jmax) and hybrid P/E cores amplify the static-chunk
    // straggler. 0 / unset = static.
    const int attn_dyn = hal::env::attn_dynamic();   // 0 = static; default 8 (measured best on M4 4P+6E)
#if defined(_OPENMP)
    #pragma omp parallel
#endif
  {
    std::vector<float> scores(skp);   // per-thread scratch (reused across queries)
    auto row = [&](int qi) {
        const int h  = qi/seq_q;
        const int t1 = qi%seq_q;
        const int kv = h/group;

        const float* q    = Q+((size_t)t1*n_q+h)*head_dim;
        const float* mrow = mask+(size_t)t1*seq_k;
        const float* kt   = KTp+(size_t)kv*head_dim*skp;

        // bound the QK/softmax loops at the last allowed key (bit-exact)
        const int jmax = hal::row_bound(mrow, seq_k);
        if (jmax == 0) {   // fully-masked row -> uniform over all keys (dense semantics)
            const float u = 1.0f/(float)seq_k;
            float* o = out+((size_t)t1*n_q+h)*head_dim;
            for (int d=0; d<head_dim; d++) o[d] = 0.0f;
            for (int t2=0; t2<seq_k; t2++)
                simd_axpy(o, u, V+((size_t)t2*n_kv+kv)*head_dim, head_dim);
            return;
        }
        const int skq = (jmax+7) & ~7;

        // scores = (q . K^T) * scale, 16 keys per pass (4 accumulator chains)
        int j = 0;
        for (; j+16<=skq; j+=16) {
            float32x4_t a0 = vdupq_n_f32(0);
            float32x4_t a1 = vdupq_n_f32(0);
            float32x4_t a2 = vdupq_n_f32(0);
            float32x4_t a3 = vdupq_n_f32(0);

            int d = 0;
            for (; d+4<=head_dim; d+=4) {
                const float32x4_t qv = vld1q_f32(q+d);
                const float* kr0 = kt+(size_t)d*skp+j;
                const float* kr1 = kr0+skp;
                const float* kr2 = kr1+skp;
                const float* kr3 = kr2+skp;
                a0 = vfmaq_laneq_f32(a0, vld1q_f32(kr0),    qv, 0);
                a1 = vfmaq_laneq_f32(a1, vld1q_f32(kr0+4),  qv, 0);
                a2 = vfmaq_laneq_f32(a2, vld1q_f32(kr0+8),  qv, 0);
                a3 = vfmaq_laneq_f32(a3, vld1q_f32(kr0+12), qv, 0);
                a0 = vfmaq_laneq_f32(a0, vld1q_f32(kr1),    qv, 1);
                a1 = vfmaq_laneq_f32(a1, vld1q_f32(kr1+4),  qv, 1);
                a2 = vfmaq_laneq_f32(a2, vld1q_f32(kr1+8),  qv, 1);
                a3 = vfmaq_laneq_f32(a3, vld1q_f32(kr1+12), qv, 1);
                a0 = vfmaq_laneq_f32(a0, vld1q_f32(kr2),    qv, 2);
                a1 = vfmaq_laneq_f32(a1, vld1q_f32(kr2+4),  qv, 2);
                a2 = vfmaq_laneq_f32(a2, vld1q_f32(kr2+8),  qv, 2);
                a3 = vfmaq_laneq_f32(a3, vld1q_f32(kr2+12), qv, 2);
                a0 = vfmaq_laneq_f32(a0, vld1q_f32(kr3),    qv, 3);
                a1 = vfmaq_laneq_f32(a1, vld1q_f32(kr3+4),  qv, 3);
                a2 = vfmaq_laneq_f32(a2, vld1q_f32(kr3+8),  qv, 3);
                a3 = vfmaq_laneq_f32(a3, vld1q_f32(kr3+12), qv, 3);
            }
            for (; d<head_dim; d++) {
                const float* kr = kt+(size_t)d*skp+j;
                a0 = vfmaq_n_f32(a0, vld1q_f32(kr),    q[d]);
                a1 = vfmaq_n_f32(a1, vld1q_f32(kr+4),  q[d]);
                a2 = vfmaq_n_f32(a2, vld1q_f32(kr+8),  q[d]);
                a3 = vfmaq_n_f32(a3, vld1q_f32(kr+12), q[d]);
            }
            vst1q_f32(scores.data()+j,    vmulq_n_f32(a0, scale));
            vst1q_f32(scores.data()+j+4,  vmulq_n_f32(a1, scale));
            vst1q_f32(scores.data()+j+8,  vmulq_n_f32(a2, scale));
            vst1q_f32(scores.data()+j+12, vmulq_n_f32(a3, scale));
        }
        for (; j+4<=skq; j+=4) {
            float32x4_t a0 = vdupq_n_f32(0);
            for (int d=0; d<head_dim; d++)
                a0 = vfmaq_n_f32(a0, vld1q_f32(kt+(size_t)d*skp+j), q[d]);
            vst1q_f32(scores.data()+j, vmulq_n_f32(a0, scale));
        }
        // add the mask row; lanes past jmax (still < seq_k) -> -inf
        const int me = skq < seq_k ? skq : seq_k;
        for (j=0; j+4<=me; j+=4)
            vst1q_f32(scores.data()+j,
                      vaddq_f32(vld1q_f32(scores.data()+j), vld1q_f32(mrow+j)));
        for (; j<me; j++) scores[j] += mrow[j];
        for (j=me; j<skq; j++) scores[j] = NINF;

        float32x4_t vmax = vdupq_n_f32(NINF);
        for (j=0; j<skq; j+=4)
            vmax = vmaxq_f32(vmax, vld1q_f32(scores.data()+j));
        const float maxs = vmaxvq_f32(vmax);

        float sum;
        if (maxs <= std::numeric_limits<float>::lowest()) {
            // all remaining scores collapsed to finfo.min -> uniform (dense semantics)
            for (j=0; j<seq_k; j++) scores[j] = 1.0f;
            sum = (float)seq_k;
        } else {
            const float32x4_t vm = vdupq_n_f32(maxs);
            float32x4_t vsum = vdupq_n_f32(0);
            for (j=0; j<skq; j+=4) {
                const float32x4_t e = exp_ps(vsubq_f32(vld1q_f32(scores.data()+j), vm));
                vst1q_f32(scores.data()+j, e);
                vsum = vaddq_f32(vsum, e);
            }
            sum = vaddvq_f32(vsum);
        }

        const int av_end = maxs <= std::numeric_limits<float>::lowest() ? seq_k : jmax;
        const float inv = 1.0f/sum;
        float* o = out+((size_t)t1*n_q+h)*head_dim;
        if (head_dim == 64) {
            // AV with the output resident in 16 accumulators: per key 1 broadcast +
            // 16 V loads + 16 FMAs, no o read-modify-write. Same key order ->
            // bit-exact vs the axpy path.
            float32x4_t av[16];
            for (int v=0; v<16; v++) av[v] = vdupq_n_f32(0);

            for (int t2=0; t2<av_end; t2++) {
                if (scores[t2] == 0.0f) continue;
                const float* vv = V+((size_t)t2*n_kv+kv)*64;
                const float a = scores[t2]*inv;
                for (int v=0; v<16; v++)
                    av[v] = vfmaq_n_f32(av[v], vld1q_f32(vv+4*v), a);
            }
            for (int v=0; v<16; v++) vst1q_f32(o+4*v, av[v]);
        } else {
            for (int d=0; d<head_dim; d++) o[d] = 0.0f;
            for (int t2=0; t2<av_end; t2++) {
                if (scores[t2] == 0.0f) continue;
                const float* vv = V+((size_t)t2*n_kv+kv)*head_dim;
                simd_axpy(o, scores[t2]*inv, vv, head_dim);
            }
        }
    };
    if (attn_dyn > 0) {
#if defined(_OPENMP)
        #pragma omp for schedule(dynamic, attn_dyn)
#endif
        for (int qi=0; qi<n_q*seq_q; qi++) row(qi);
    } else {
#if defined(_OPENMP)
        #pragma omp for schedule(static)
#endif
        for (int qi=0; qi<n_q*seq_q; qi++) row(qi);
    }
  }
}

} // namespace tcpu

#endif // TCPU_HAL_APPLE
