/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pi / generic ARM NEON masked attention (raspi tuning, verbatim). Mask-density
// dispatch: sparse rows (history padding) take the per-key SKIP loop
// (bit-exact vs scalar); dense rows take the 4-query-tiled kernel that cuts
// K/V traffic 4x (the op is L2-bandwidth-bound on the A72). TCPU_ATTN forces
// one path.

#include "../arch.h"
#if TCPU_HAL_NEON

#include "../simd.h"
#include "../common/env.h"
#include "../common/layout.h"
#include "../../ops/lm_ops.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <cstddef>
using std::size_t;

namespace tcpu {

// One masked-attention query row, per-key mask-SKIP form (reads K directly, no
// transpose): blocked keys cost one compare each and their dot/exp/AV work is
// skipped entirely. Fastest when most keys are blocked (e.g. octo's {0,1}
// history-padding mask). Scalar-order dots/exp -> bit-identical to the scalar
// reference. sc is a seq_k float scratch.
static void attn_row_skip_neon(float* o, const float* q, const float* mrow,
                               const float* K, const float* V, int kv, int n_kv,
                               int head_dim, int seq_k, float scale, float* sc) {
    const float BLOCK = std::numeric_limits<float>::lowest();
    const float NINF  = -std::numeric_limits<float>::infinity();

    float maxs = NINF;
    for (int t2=0; t2<seq_k; t2++) {
        if (mrow[t2] == BLOCK) {
            sc[t2] = NINF;
            continue;
        }
        const float* k = K+((size_t)t2*n_kv+kv)*head_dim;
        float dot = simd_dot(q, k, head_dim)*scale + mrow[t2];
        sc[t2] = dot;
        if (dot > maxs) maxs = dot;
    }

    float sum = 0.0f;
    if (maxs == NINF) {
        for (int t2=0; t2<seq_k; t2++) sc[t2] = 1.0f;
        sum = (float)seq_k;
    } else {
        for (int t2=0; t2<seq_k; t2++) {
            float e = sc[t2] == NINF ? 0.0f : std::exp(sc[t2]-maxs);
            sc[t2] = e;
            sum += e;
        }
    }

    const float inv = 1.0f/sum;
    if (head_dim == 64) {
        // AV with the output row resident in 16 NEON accumulators
        float32x4_t acc[16];
        for (int v=0; v<16; v++) acc[v] = vdupq_n_f32(0);

        for (int t2=0; t2<seq_k; t2++) {
            if (sc[t2] == 0.0f) continue;
            const float* vv = V+((size_t)t2*n_kv+kv)*64;
            const float32x4_t a = vdupq_n_f32(sc[t2]*inv);
            for (int v=0; v<16; v++)
                acc[v] = vfmaq_f32(acc[v], a, vld1q_f32(vv+4*v));
        }
        for (int v=0; v<16; v++) vst1q_f32(o+4*v, acc[v]);
    } else {
        for (int d=0; d<head_dim; d++) o[d] = 0.0f;
        for (int t2=0; t2<seq_k; t2++) {
            if (sc[t2] == 0.0f) continue;
            simd_axpy(o, sc[t2]*inv, V+((size_t)t2*n_kv+kv)*head_dim, head_dim);
        }
    }
}

// One masked-attention query row against a per-head transposed K panel
// kt = [head_dim][skp] (key direction contiguous). Vectorized QK (lane-FMLA),
// softmax (exp_ps_neon; blocked keys flush to weight exactly 0), jmax row bound,
// register-resident AV for head_dim 64. Used for tile tails and head_dim != 64;
// the 4-query tiled path in gqa_attention_masked computes identical floats.
static void attn_row_kt_neon(float* o, const float* q, const float* mrow,
                             const float* kt, const float* V, int kv, int n_kv,
                             int head_dim, int seq_k, int skp, float scale,
                             float* sc) {
    const float NINF = -std::numeric_limits<float>::infinity();

    int jmax = seq_k;
    while (jmax > 0 && mrow[jmax-1] == std::numeric_limits<float>::lowest()) jmax--;
    if (jmax == 0) {   // fully-masked row -> uniform over all keys (dense semantics)
        const float u = 1.0f/(float)seq_k;
        for (int d=0; d<head_dim; d++) o[d] = 0.0f;
        for (int t2=0; t2<seq_k; t2++)
            simd_axpy(o, u, V+((size_t)t2*n_kv+kv)*head_dim, head_dim);
        return;
    }
    const int skq = (jmax+3) & ~3;

    // scores = (q . K^T) * scale, 16 keys/pass; per 4 dims: one q load + 16
    // contiguous K loads + 16 lane-FMLAs (no per-dot horizontal reduction)
    const float32x4_t vscale4 = vdupq_n_f32(scale);
    int j = 0;
    for (; j+16<=skq; j+=16) {
        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = a0;
        float32x4_t a2 = a0;
        float32x4_t a3 = a0;

        int d = 0;
        for (; d+4<=head_dim; d+=4) {
            const float32x4_t qv = vld1q_f32(q+d);
            const float* kr = kt+(size_t)d*skp+j;
            a0 = vfmaq_laneq_f32(a0, vld1q_f32(kr),    qv, 0);
            a1 = vfmaq_laneq_f32(a1, vld1q_f32(kr+4),  qv, 0);
            a2 = vfmaq_laneq_f32(a2, vld1q_f32(kr+8),  qv, 0);
            a3 = vfmaq_laneq_f32(a3, vld1q_f32(kr+12), qv, 0);
            kr += skp;
            a0 = vfmaq_laneq_f32(a0, vld1q_f32(kr),    qv, 1);
            a1 = vfmaq_laneq_f32(a1, vld1q_f32(kr+4),  qv, 1);
            a2 = vfmaq_laneq_f32(a2, vld1q_f32(kr+8),  qv, 1);
            a3 = vfmaq_laneq_f32(a3, vld1q_f32(kr+12), qv, 1);
            kr += skp;
            a0 = vfmaq_laneq_f32(a0, vld1q_f32(kr),    qv, 2);
            a1 = vfmaq_laneq_f32(a1, vld1q_f32(kr+4),  qv, 2);
            a2 = vfmaq_laneq_f32(a2, vld1q_f32(kr+8),  qv, 2);
            a3 = vfmaq_laneq_f32(a3, vld1q_f32(kr+12), qv, 2);
            kr += skp;
            a0 = vfmaq_laneq_f32(a0, vld1q_f32(kr),    qv, 3);
            a1 = vfmaq_laneq_f32(a1, vld1q_f32(kr+4),  qv, 3);
            a2 = vfmaq_laneq_f32(a2, vld1q_f32(kr+8),  qv, 3);
            a3 = vfmaq_laneq_f32(a3, vld1q_f32(kr+12), qv, 3);
        }
        for (; d<head_dim; d++) {   // head_dim % 4 != 0 tail
            const float32x4_t qd = vdupq_n_f32(q[d]);
            const float* kr = kt+(size_t)d*skp+j;
            a0 = vfmaq_f32(a0, qd, vld1q_f32(kr));
            a1 = vfmaq_f32(a1, qd, vld1q_f32(kr+4));
            a2 = vfmaq_f32(a2, qd, vld1q_f32(kr+8));
            a3 = vfmaq_f32(a3, qd, vld1q_f32(kr+12));
        }
        vst1q_f32(sc+j,    vmulq_f32(a0, vscale4));
        vst1q_f32(sc+j+4,  vmulq_f32(a1, vscale4));
        vst1q_f32(sc+j+8,  vmulq_f32(a2, vscale4));
        vst1q_f32(sc+j+12, vmulq_f32(a3, vscale4));
    }
    for (; j+4<=skq; j+=4) {
        float32x4_t a0 = vdupq_n_f32(0.0f);
        int d = 0;
        for (; d+4<=head_dim; d+=4) {
            const float32x4_t qv = vld1q_f32(q+d);
            const float* kr = kt+(size_t)d*skp+j;
            a0 = vfmaq_laneq_f32(a0, vld1q_f32(kr),               qv, 0);
            a0 = vfmaq_laneq_f32(a0, vld1q_f32(kr+skp),           qv, 1);
            a0 = vfmaq_laneq_f32(a0, vld1q_f32(kr+2*(size_t)skp), qv, 2);
            a0 = vfmaq_laneq_f32(a0, vld1q_f32(kr+3*(size_t)skp), qv, 3);
        }
        for (; d<head_dim; d++)
            a0 = vfmaq_f32(a0, vdupq_n_f32(q[d]), vld1q_f32(kt+(size_t)d*skp+j));
        vst1q_f32(sc+j, vmulq_f32(a0, vscale4));
    }

    // add the mask row; lanes past jmax (still < seq_k) -> -inf
    const int me = skq < seq_k ? skq : seq_k;
    for (j=0; j+4<=me; j+=4)
        vst1q_f32(sc+j, vaddq_f32(vld1q_f32(sc+j), vld1q_f32(mrow+j)));
    for (; j<me; j++) sc[j] += mrow[j];
    for (j=me; j<skq; j++) sc[j] = NINF;

    float32x4_t vmax4 = vdupq_n_f32(NINF);
    for (j=0; j<skq; j+=4) vmax4 = vmaxq_f32(vmax4, vld1q_f32(sc+j));
    const float maxs = vmaxvq_f32(vmax4);

    float sum;
    if (maxs <= std::numeric_limits<float>::lowest()) {
        // all remaining scores collapsed to finfo.min -> uniform (dense semantics)
        for (j=0; j<seq_k; j++) sc[j] = 1.0f;
        sum = (float)seq_k;
    } else {
        const float32x4_t vm = vdupq_n_f32(maxs);
        float32x4_t vsum = vdupq_n_f32(0.0f);
        for (j=0; j<skq; j+=4) {
            const float32x4_t e = exp_ps_neon(vsubq_f32(vld1q_f32(sc+j), vm));
            vst1q_f32(sc+j, e);
            vsum = vaddq_f32(vsum, e);
        }
        sum = vaddvq_f32(vsum);
    }

    const int av_end = maxs <= std::numeric_limits<float>::lowest() ? seq_k : jmax;
    const float inv = 1.0f/sum;
    if (head_dim == 64) {
        // AV with the output row resident in 16 NEON accumulators
        float32x4_t acc[16];
        for (int v=0; v<16; v++) acc[v] = vdupq_n_f32(0);

        for (int t2=0; t2<av_end; t2++) {
            if (sc[t2] == 0.0f) continue;
            const float* vv = V+((size_t)t2*n_kv+kv)*64;
            const float32x4_t a = vdupq_n_f32(sc[t2]*inv);
            for (int v=0; v<16; v++)
                acc[v] = vfmaq_f32(acc[v], a, vld1q_f32(vv+4*v));
        }
        for (int v=0; v<16; v++) vst1q_f32(o+4*v, acc[v]);
    } else {
        for (int d=0; d<head_dim; d++) o[d] = 0.0f;
        for (int t2=0; t2<av_end; t2++) {
            if (sc[t2] == 0.0f) continue;
            simd_axpy(o, sc[t2]*inv, V+((size_t)t2*n_kv+kv)*head_dim, head_dim);
        }
    }
}

void gqa_attention_masked(float* out, const float* Q, const float* K, const float* V,
                          int seq_q, int seq_k, int n_q, int n_kv, int head_dim,
                          float scale, const float* mask,
                          [[maybe_unused]] const float* K_pre) {
    // ponytail: this path transposes K itself; K_pre is an AVX2-only shortcut.
    const int group = n_q / n_kv;
    const float NINF = -std::numeric_limits<float>::infinity();
    // Key-vectorized path, NEON port of the AVX2 block above. K is transposed once
    // per call to [kv][d][key] (thread_local arena) so the key direction is
    // contiguous; scores for 16 keys/pass via lane-FMLA from a loaded q vector;
    // softmax max/exp/sum vectorized (exp_ps_neon's deep-negative clamp flushes
    // blocked keys to weight exactly 0, so the AV zero-skip still works).
    //
    // HISTORY: an earlier 4-keys/pass variant WITHOUT the transpose read K at
    // stride n_kv*head_dim and was ~1200 ms SLOWER than the per-key mask-skip loop
    // on the pad-mask demo ({0,1} history, most keys blocked). But on REAL history
    // ({1,1}, control loop) most keys are allowed and the skip loop collapsed to
    // 3.7 s/inference in attention alone. This transposed + jmax-bounded port is
    // the Intel 6d design: contiguous key loads, no per-dot reduction, and the row
    // bound still caps the work on sparse rows. Numerics: poly exp + reordered
    // sums - same tolerance class as the AVX2 path and the SIMD gelu.
    // Density dispatch: the two regimes want different algorithms. Sparse masks
    // (octo's {0,1} history padding, ~26% allowed) win with the per-key SKIP loop
    // (blocked keys cost one compare; no transpose). Dense masks (real {1,1}
    // history ~74%, SmolVLA cross-attn ~100%) win with the tiled path below -
    // the skip loop collapsed to 3.7 s/inference in attention there. Sample ~3%
    // of mask rows to pick; both paths produce the documented tolerance classes
    // (skip = scalar bit-exact, tiled = poly-exp class like the AVX2 path).
    {
        // TCPU_ATTN=skip|tiled forces one path (A/B + numeric cross-check hook)
        const int force = hal::env::attn_force();   // TCPU_ATTN=skip|tiled
        const float BLOCKV = std::numeric_limits<float>::lowest();

        size_t allowed = 0;
        size_t sampled = 0;
        if (!force)
            for (int t1=0; t1<seq_q; t1+=33) {
                const float* mrow = mask+(size_t)t1*seq_k;
                for (int t2=0; t2<seq_k; t2++) allowed += (mrow[t2] != BLOCKV);
                sampled += (size_t)seq_k;
            }

        if (force == 1 || (!force && allowed*2 < sampled)) {   // < 50% allowed -> skip path
#if defined(_OPENMP)
            #pragma omp parallel
#endif
          {
            std::vector<float> sc(seq_k);
#if defined(_OPENMP)
            #pragma omp for schedule(static)
#endif
            for (int qi=0; qi<n_q*seq_q; qi++) {
                const int h  = qi/seq_q;
                const int t1 = qi%seq_q;
                attn_row_skip_neon(out+((size_t)t1*n_q+h)*head_dim,
                                   Q+((size_t)t1*n_q+h)*head_dim,
                                   mask+(size_t)t1*seq_k,
                                   K, V, h/group, n_kv, head_dim, seq_k, scale,
                                   sc.data());
            }
          }
            return;
        }
    }

    const int skp = (seq_k+3) & ~3;
    static thread_local std::vector<float> KT;
    if (KT.size() < (size_t)n_kv*head_dim*skp) KT.resize((size_t)n_kv*head_dim*skp);
    float* const KTw = KT.data();   // hoisted BEFORE the parallel regions (see AVX2 note)
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int jb=0; jb<skp; jb+=4) {
        const int je = seq_k-jb < 4 ? seq_k-jb : 4;
        for (int kv=0; kv<n_kv; kv++)
            for (int d=0; d<head_dim; d++) {
                float* dst = KTw+((size_t)kv*head_dim+d)*skp+jb;
                for (int j=0; j<je; j++)
                    dst[j] = K[((size_t)(jb+j)*n_kv+kv)*head_dim+d];
                for (int j=je; j<4; j++) dst[j] = 0.0f;
            }
    }

    // 4-query tiles: attention here is L2-bandwidth-bound on the A72 (each query
    // streams its head's whole K^T/V panels, which exceed L1), so the win comes
    // from TRAFFIC, not per-op SIMD: 4 queries per pass read each K/V line once
    // instead of 4 times (a GEMM-style micro-kernel; the plain per-query dense
    // port measured ~2x SLOWER than the mask-skip loop on sparse pad-mask rows
    // precisely because it only vectorized, without cutting traffic).
    const int mtq = (seq_q+3)/4;
#if defined(_OPENMP)
    #pragma omp parallel
#endif
  {
    // per-thread scratch: 4 score rows + the transposed 4-query tile
    std::vector<float> scores ((size_t)4*skp);
    std::vector<float> qtile  ((size_t)head_dim*4);
#if defined(_OPENMP)
    #pragma omp for schedule(static)
#endif
    for (int ti=0; ti<n_q*mtq; ti++) {
        const int h     = ti/mtq;
        const int tq0   = (ti%mtq)*4;
        const int nrows = seq_q-tq0 < 4 ? seq_q-tq0 : 4;
        const int kv    = h/group;

        const float* kt = KTw+(size_t)kv*head_dim*skp;
        if (nrows < 4 || head_dim != 64) {   // tile tail / generic head_dim
            for (int r=0; r<nrows; r++)
                attn_row_kt_neon(out+((size_t)(tq0+r)*n_q+h)*head_dim,
                                 Q+((size_t)(tq0+r)*n_q+h)*head_dim,
                                 mask+(size_t)(tq0+r)*seq_k,
                                 kt, V, kv, n_kv, head_dim, seq_k, skp, scale,
                                 scores.data());
            continue;
        }

        const float* qr[4];
        const float* mr[4];
        float* sc[4];
        int jmax[4];
        int jmax_t = 0;
        for (int r=0; r<4; r++) {
            qr[r] = Q+((size_t)(tq0+r)*n_q+h)*64;
            mr[r] = mask+(size_t)(tq0+r)*seq_k;
            sc[r] = scores.data()+(size_t)r*skp;

            int jm = seq_k;   // per-row bound at the last allowed key
            while (jm > 0 && mr[r][jm-1] == std::numeric_limits<float>::lowest()) jm--;
            jmax[r] = jm;
            if (jm > jmax_t) jmax_t = jm;
        }
        if (jmax_t == 0) {   // all four rows fully masked -> uniform fallback each
            for (int r=0; r<4; r++)
                attn_row_kt_neon(out+((size_t)(tq0+r)*n_q+h)*64, qr[r],
                                 mr[r], kt, V, kv, n_kv, 64, seq_k, skp, scale, sc[r]);
            continue;
        }
        const int skq = (jmax_t+3) & ~3;
        // rows with jmax[r] == 0 fall through: their mask row collapses every score
        // and the per-row uniform fallback below restores dense semantics.

        // transpose the 4 q vectors -> qtile[d][4]: one load feeds 4 lane-FMLAs
        float* const qt = qtile.data();
        for (int d=0; d<64; d++)
            for (int r=0; r<4; r++) qt[d*4+r] = qr[r][d];

        // QK: 4 queries x 16 keys per pass; per dim: 1 q load + 4 K loads + 16
        // lane-FMLAs (16 accumulators + 4 K vectors + 1 q vector = 21 live regs)
        const float32x4_t vscale4 = vdupq_n_f32(scale);
        int j = 0;
        for (; j+16<=skq; j+=16) {
            float32x4_t a00 = vdupq_n_f32(0.0f);
            float32x4_t a01 = a00;
            float32x4_t a02 = a00;
            float32x4_t a03 = a00;
            float32x4_t a10 = a00;
            float32x4_t a11 = a00;
            float32x4_t a12 = a00;
            float32x4_t a13 = a00;
            float32x4_t a20 = a00;
            float32x4_t a21 = a00;
            float32x4_t a22 = a00;
            float32x4_t a23 = a00;
            float32x4_t a30 = a00;
            float32x4_t a31 = a00;
            float32x4_t a32 = a00;
            float32x4_t a33 = a00;

            for (int d=0; d<64; d++) {
                const float32x4_t qv = vld1q_f32(qt+d*4);
                const float* kr = kt+(size_t)d*skp+j;
                const float32x4_t k0 = vld1q_f32(kr);
                const float32x4_t k1 = vld1q_f32(kr+4);
                const float32x4_t k2 = vld1q_f32(kr+8);
                const float32x4_t k3 = vld1q_f32(kr+12);
                a00 = vfmaq_laneq_f32(a00, k0, qv, 0);
                a01 = vfmaq_laneq_f32(a01, k1, qv, 0);
                a02 = vfmaq_laneq_f32(a02, k2, qv, 0);
                a03 = vfmaq_laneq_f32(a03, k3, qv, 0);
                a10 = vfmaq_laneq_f32(a10, k0, qv, 1);
                a11 = vfmaq_laneq_f32(a11, k1, qv, 1);
                a12 = vfmaq_laneq_f32(a12, k2, qv, 1);
                a13 = vfmaq_laneq_f32(a13, k3, qv, 1);
                a20 = vfmaq_laneq_f32(a20, k0, qv, 2);
                a21 = vfmaq_laneq_f32(a21, k1, qv, 2);
                a22 = vfmaq_laneq_f32(a22, k2, qv, 2);
                a23 = vfmaq_laneq_f32(a23, k3, qv, 2);
                a30 = vfmaq_laneq_f32(a30, k0, qv, 3);
                a31 = vfmaq_laneq_f32(a31, k1, qv, 3);
                a32 = vfmaq_laneq_f32(a32, k2, qv, 3);
                a33 = vfmaq_laneq_f32(a33, k3, qv, 3);
            }
            vst1q_f32(sc[0]+j,    vmulq_f32(a00, vscale4));
            vst1q_f32(sc[0]+j+4,  vmulq_f32(a01, vscale4));
            vst1q_f32(sc[0]+j+8,  vmulq_f32(a02, vscale4));
            vst1q_f32(sc[0]+j+12, vmulq_f32(a03, vscale4));
            vst1q_f32(sc[1]+j,    vmulq_f32(a10, vscale4));
            vst1q_f32(sc[1]+j+4,  vmulq_f32(a11, vscale4));
            vst1q_f32(sc[1]+j+8,  vmulq_f32(a12, vscale4));
            vst1q_f32(sc[1]+j+12, vmulq_f32(a13, vscale4));
            vst1q_f32(sc[2]+j,    vmulq_f32(a20, vscale4));
            vst1q_f32(sc[2]+j+4,  vmulq_f32(a21, vscale4));
            vst1q_f32(sc[2]+j+8,  vmulq_f32(a22, vscale4));
            vst1q_f32(sc[2]+j+12, vmulq_f32(a23, vscale4));
            vst1q_f32(sc[3]+j,    vmulq_f32(a30, vscale4));
            vst1q_f32(sc[3]+j+4,  vmulq_f32(a31, vscale4));
            vst1q_f32(sc[3]+j+8,  vmulq_f32(a32, vscale4));
            vst1q_f32(sc[3]+j+12, vmulq_f32(a33, vscale4));
        }
        for (; j+4<=skq; j+=4) {   // 4-key tail
            float32x4_t a0 = vdupq_n_f32(0.0f);
            float32x4_t a1 = a0;
            float32x4_t a2 = a0;
            float32x4_t a3 = a0;
            for (int d=0; d<64; d++) {
                const float32x4_t qv = vld1q_f32(qt+d*4);
                const float32x4_t k0 = vld1q_f32(kt+(size_t)d*skp+j);
                a0 = vfmaq_laneq_f32(a0, k0, qv, 0);
                a1 = vfmaq_laneq_f32(a1, k0, qv, 1);
                a2 = vfmaq_laneq_f32(a2, k0, qv, 2);
                a3 = vfmaq_laneq_f32(a3, k0, qv, 3);
            }
            vst1q_f32(sc[0]+j, vmulq_f32(a0, vscale4));
            vst1q_f32(sc[1]+j, vmulq_f32(a1, vscale4));
            vst1q_f32(sc[2]+j, vmulq_f32(a2, vscale4));
            vst1q_f32(sc[3]+j, vmulq_f32(a3, vscale4));
        }

        // per-row mask + softmax + prescale by 1/sum (weights land in sc[r])
        int av_end[4];
        int av_end_t = 0;
        bool uni = false;
        for (int r=0; r<4; r++) {
            float* s = sc[r];
            const int me = skq < seq_k ? skq : seq_k;
            int jj = 0;
            for (; jj+4<=me; jj+=4)
                vst1q_f32(s+jj, vaddq_f32(vld1q_f32(s+jj), vld1q_f32(mr[r]+jj)));
            for (; jj<me; jj++) s[jj] += mr[r][jj];
            for (jj=me; jj<skq; jj++) s[jj] = NINF;

            float32x4_t vmax4 = vdupq_n_f32(NINF);
            for (jj=0; jj<skq; jj+=4) vmax4 = vmaxq_f32(vmax4, vld1q_f32(s+jj));
            const float maxs = vmaxvq_f32(vmax4);

            if (maxs <= std::numeric_limits<float>::lowest()) {
                // row collapsed -> uniform over all keys (dense semantics)
                const float u = 1.0f/(float)seq_k;
                for (jj=0; jj<seq_k; jj++) s[jj] = u;
                av_end[r] = seq_k;
                uni = true;
            } else {
                const float32x4_t vm = vdupq_n_f32(maxs);
                float32x4_t vsum = vdupq_n_f32(0.0f);
                for (jj=0; jj<skq; jj+=4) {
                    const float32x4_t e = exp_ps_neon(vsubq_f32(vld1q_f32(s+jj), vm));
                    vst1q_f32(s+jj, e);
                    vsum = vaddq_f32(vsum, e);
                }
                const float32x4_t vi = vdupq_n_f32(1.0f/vaddvq_f32(vsum));
                for (jj=0; jj<skq; jj+=4)
                    vst1q_f32(s+jj, vmulq_f32(vld1q_f32(s+jj), vi));
                av_end[r] = jmax[r];
            }
            if (av_end[r] > av_end_t) av_end_t = av_end[r];
        }
        if (uni && av_end_t > skq)   // a uniform row widened the AV range: zero the
            for (int r=0; r<4; r++)   // other rows' uninitialized weight lanes
                if (av_end[r] < av_end_t)
                    std::memset(sc[r]+skq, 0, sizeof(float)*(av_end_t-skq));

        // AV: 4 queries x 16-dim chunks; per key one V line per chunk is shared by
        // all 4 rows (16 accumulators + 4 V vectors + 1 weight vector = 21 regs).
        // Keys where all 4 weights are 0 are skipped; a 0-weight FMA adds exact
        // zeros, so results match the per-row zero-skip path bit-for-bit.
        float* o[4];
        for (int r=0; r<4; r++) o[r] = out+((size_t)(tq0+r)*n_q+h)*64;

        for (int c=0; c<64; c+=16) {
            float32x4_t b00 = vdupq_n_f32(0.0f);
            float32x4_t b01 = b00;
            float32x4_t b02 = b00;
            float32x4_t b03 = b00;
            float32x4_t b10 = b00;
            float32x4_t b11 = b00;
            float32x4_t b12 = b00;
            float32x4_t b13 = b00;
            float32x4_t b20 = b00;
            float32x4_t b21 = b00;
            float32x4_t b22 = b00;
            float32x4_t b23 = b00;
            float32x4_t b30 = b00;
            float32x4_t b31 = b00;
            float32x4_t b32 = b00;
            float32x4_t b33 = b00;

            for (int t2=0; t2<av_end_t; t2++) {
                float32x4_t pv = vdupq_n_f32(sc[0][t2]);
                pv = vsetq_lane_f32(sc[1][t2], pv, 1);
                pv = vsetq_lane_f32(sc[2][t2], pv, 2);
                pv = vsetq_lane_f32(sc[3][t2], pv, 3);
                if (vmaxvq_f32(pv) == 0.0f) continue;   // weights are >= 0

                const float* vv = V+((size_t)t2*n_kv+kv)*64+c;
                const float32x4_t v0 = vld1q_f32(vv);
                const float32x4_t v1 = vld1q_f32(vv+4);
                const float32x4_t v2 = vld1q_f32(vv+8);
                const float32x4_t v3 = vld1q_f32(vv+12);
                b00 = vfmaq_laneq_f32(b00, v0, pv, 0);
                b01 = vfmaq_laneq_f32(b01, v1, pv, 0);
                b02 = vfmaq_laneq_f32(b02, v2, pv, 0);
                b03 = vfmaq_laneq_f32(b03, v3, pv, 0);
                b10 = vfmaq_laneq_f32(b10, v0, pv, 1);
                b11 = vfmaq_laneq_f32(b11, v1, pv, 1);
                b12 = vfmaq_laneq_f32(b12, v2, pv, 1);
                b13 = vfmaq_laneq_f32(b13, v3, pv, 1);
                b20 = vfmaq_laneq_f32(b20, v0, pv, 2);
                b21 = vfmaq_laneq_f32(b21, v1, pv, 2);
                b22 = vfmaq_laneq_f32(b22, v2, pv, 2);
                b23 = vfmaq_laneq_f32(b23, v3, pv, 2);
                b30 = vfmaq_laneq_f32(b30, v0, pv, 3);
                b31 = vfmaq_laneq_f32(b31, v1, pv, 3);
                b32 = vfmaq_laneq_f32(b32, v2, pv, 3);
                b33 = vfmaq_laneq_f32(b33, v3, pv, 3);
            }
            vst1q_f32(o[0]+c,    b00);
            vst1q_f32(o[0]+c+4,  b01);
            vst1q_f32(o[0]+c+8,  b02);
            vst1q_f32(o[0]+c+12, b03);
            vst1q_f32(o[1]+c,    b10);
            vst1q_f32(o[1]+c+4,  b11);
            vst1q_f32(o[1]+c+8,  b12);
            vst1q_f32(o[1]+c+12, b13);
            vst1q_f32(o[2]+c,    b20);
            vst1q_f32(o[2]+c+4,  b21);
            vst1q_f32(o[2]+c+8,  b22);
            vst1q_f32(o[2]+c+12, b23);
            vst1q_f32(o[3]+c,    b30);
            vst1q_f32(o[3]+c+4,  b31);
            vst1q_f32(o[3]+c+8,  b32);
            vst1q_f32(o[3]+c+12, b33);
        }
    }
  }
}

// ---------------------------------------------------------------------------
// Dense (unmasked) attention: the ViT case, where every query attends every key.
//
// The masked path above walks 4-query tiles, so each tile re-streams its head's
// whole K^T and V panels. At SigLIP's shape (seq 1024, head_dim 64) a panel pair
// is 512 KB and there are 256 tiles per head: 128 MB of DRAM traffic per head,
// per layer, per view. That is what made SigLIP's attention 2.0 s/view on a Pi 5
// while its GEMMs ran at 78 GFLOP/s - the op is not compute-bound, it is bound
// on re-reading K and V.
//
// So block it. A block of QB queries makes one pass over each panel:
//   QK  runs keys-outer, so kt's 16-key slice serves every micro-tile in the
//       block before the next slice is touched;
//   AV  runs 16-dim-column-outer, so V's column slice (seq_k*16*4 B) stays in L2
//       across the block's micro-tiles.
// Panel traffic drops by QB/4 - 16x at the default QB=64.
//
// The micro-kernels are the masked path's, with the mask, the jmax row bound and
// the zero-weight skip removed. With an all-zero mask each of those is a no-op
// (adding 0.0, a bound of seq_k, and skipping FMAs that add exact zeros), so the
// FMA order is unchanged and this path is bit-identical to calling
// gqa_attention_masked with a zero mask - it just does not pay for the traffic.
static void attn_dense_qblock(float* out, const float* Q, const float* vp,
                              const float* kt, int q0, int nq, int n_q, int h,
                              int seq_k, int skp, float scale,
                              float* scores, float* qt) {
    const int nt  = (nq+3)/4;              // 4-query micro-tiles in this block
    const int skq = (seq_k+3) & ~3;
    const float NINF = -std::numeric_limits<float>::infinity();

    // queries -> per-tile [64][4] so one load feeds four lane-FMLAs. Tail rows of
    // a short block are zeroed; their scores are computed and then not stored.
    for (int t=0; t<nt; t++)
        for (int d=0; d<64; d++)
            for (int r=0; r<4; r++) {
                const int qr = t*4+r;
                qt[((size_t)t*64+d)*4+r] =
                    qr < nq ? Q[((size_t)(q0+qr)*n_q+h)*64+d] : 0.0f;
            }

    // ---- QK, keys outer: kt's 16-key slice is loaded once for the whole block
    const float32x4_t vscale4 = vdupq_n_f32(scale);
    int j = 0;
    for (; j+16<=skq; j+=16) {
        for (int t=0; t<nt; t++) {
            const float* qtile = qt+(size_t)t*64*4;
            float32x4_t a00 = vdupq_n_f32(0.0f);
            float32x4_t a01 = a00, a02 = a00, a03 = a00;
            float32x4_t a10 = a00, a11 = a00, a12 = a00, a13 = a00;
            float32x4_t a20 = a00, a21 = a00, a22 = a00, a23 = a00;
            float32x4_t a30 = a00, a31 = a00, a32 = a00, a33 = a00;

            for (int d=0; d<64; d++) {
                const float32x4_t qv = vld1q_f32(qtile+d*4);
                const float* kr = kt+(size_t)d*skp+j;
                const float32x4_t k0 = vld1q_f32(kr);
                const float32x4_t k1 = vld1q_f32(kr+4);
                const float32x4_t k2 = vld1q_f32(kr+8);
                const float32x4_t k3 = vld1q_f32(kr+12);
                a00 = vfmaq_laneq_f32(a00, k0, qv, 0);
                a01 = vfmaq_laneq_f32(a01, k1, qv, 0);
                a02 = vfmaq_laneq_f32(a02, k2, qv, 0);
                a03 = vfmaq_laneq_f32(a03, k3, qv, 0);
                a10 = vfmaq_laneq_f32(a10, k0, qv, 1);
                a11 = vfmaq_laneq_f32(a11, k1, qv, 1);
                a12 = vfmaq_laneq_f32(a12, k2, qv, 1);
                a13 = vfmaq_laneq_f32(a13, k3, qv, 1);
                a20 = vfmaq_laneq_f32(a20, k0, qv, 2);
                a21 = vfmaq_laneq_f32(a21, k1, qv, 2);
                a22 = vfmaq_laneq_f32(a22, k2, qv, 2);
                a23 = vfmaq_laneq_f32(a23, k3, qv, 2);
                a30 = vfmaq_laneq_f32(a30, k0, qv, 3);
                a31 = vfmaq_laneq_f32(a31, k1, qv, 3);
                a32 = vfmaq_laneq_f32(a32, k2, qv, 3);
                a33 = vfmaq_laneq_f32(a33, k3, qv, 3);
            }
            float* s0 = scores+(size_t)(t*4+0)*skp+j;
            float* s1 = scores+(size_t)(t*4+1)*skp+j;
            float* s2 = scores+(size_t)(t*4+2)*skp+j;
            float* s3 = scores+(size_t)(t*4+3)*skp+j;
            vst1q_f32(s0,    vmulq_f32(a00, vscale4));
            vst1q_f32(s0+4,  vmulq_f32(a01, vscale4));
            vst1q_f32(s0+8,  vmulq_f32(a02, vscale4));
            vst1q_f32(s0+12, vmulq_f32(a03, vscale4));
            vst1q_f32(s1,    vmulq_f32(a10, vscale4));
            vst1q_f32(s1+4,  vmulq_f32(a11, vscale4));
            vst1q_f32(s1+8,  vmulq_f32(a12, vscale4));
            vst1q_f32(s1+12, vmulq_f32(a13, vscale4));
            vst1q_f32(s2,    vmulq_f32(a20, vscale4));
            vst1q_f32(s2+4,  vmulq_f32(a21, vscale4));
            vst1q_f32(s2+8,  vmulq_f32(a22, vscale4));
            vst1q_f32(s2+12, vmulq_f32(a23, vscale4));
            vst1q_f32(s3,    vmulq_f32(a30, vscale4));
            vst1q_f32(s3+4,  vmulq_f32(a31, vscale4));
            vst1q_f32(s3+8,  vmulq_f32(a32, vscale4));
            vst1q_f32(s3+12, vmulq_f32(a33, vscale4));
        }
    }
    for (; j+4<=skq; j+=4) {   // 4-key tail
        for (int t=0; t<nt; t++) {
            const float* qtile = qt+(size_t)t*64*4;
            float32x4_t a0 = vdupq_n_f32(0.0f);
            float32x4_t a1 = a0, a2 = a0, a3 = a0;
            for (int d=0; d<64; d++) {
                const float32x4_t qv = vld1q_f32(qtile+d*4);
                const float32x4_t k0 = vld1q_f32(kt+(size_t)d*skp+j);
                a0 = vfmaq_laneq_f32(a0, k0, qv, 0);
                a1 = vfmaq_laneq_f32(a1, k0, qv, 1);
                a2 = vfmaq_laneq_f32(a2, k0, qv, 2);
                a3 = vfmaq_laneq_f32(a3, k0, qv, 3);
            }
            vst1q_f32(scores+(size_t)(t*4+0)*skp+j, vmulq_f32(a0, vscale4));
            vst1q_f32(scores+(size_t)(t*4+1)*skp+j, vmulq_f32(a1, vscale4));
            vst1q_f32(scores+(size_t)(t*4+2)*skp+j, vmulq_f32(a2, vscale4));
            vst1q_f32(scores+(size_t)(t*4+3)*skp+j, vmulq_f32(a3, vscale4));
        }
    }

    // ---- softmax, per row, prescaled by 1/sum (same order as the masked path)
    for (int r=0; r<nq; r++) {
        float* s = scores+(size_t)r*skp;
        for (int jj=seq_k; jj<skq; jj++) s[jj] = NINF;   // padding lanes of kt

        float32x4_t vmax4 = vdupq_n_f32(NINF);
        for (int jj=0; jj<skq; jj+=4) vmax4 = vmaxq_f32(vmax4, vld1q_f32(s+jj));
        const float32x4_t vm = vdupq_n_f32(vmaxvq_f32(vmax4));

        float32x4_t vsum = vdupq_n_f32(0.0f);
        for (int jj=0; jj<skq; jj+=4) {
            const float32x4_t e = exp_ps_neon(vsubq_f32(vld1q_f32(s+jj), vm));
            vst1q_f32(s+jj, e);
            vsum = vaddq_f32(vsum, e);
        }
        const float32x4_t vi = vdupq_n_f32(1.0f/vaddvq_f32(vsum));
        for (int jj=0; jj<skq; jj+=4)
            vst1q_f32(s+jj, vmulq_f32(vld1q_f32(s+jj), vi));
    }

    // ---- AV, 16-dim column outer: V's column slice serves every micro-tile
    for (int c=0; c<64; c+=16) {
        for (int t=0; t<nt; t++) {
            const int rows = nq-t*4 < 4 ? nq-t*4 : 4;
            const float* w0 = scores+(size_t)(t*4+0)*skp;
            const float* w1 = scores+(size_t)(t*4+1)*skp;
            const float* w2 = scores+(size_t)(t*4+2)*skp;
            const float* w3 = scores+(size_t)(t*4+3)*skp;

            float32x4_t b00 = vdupq_n_f32(0.0f);
            float32x4_t b01 = b00, b02 = b00, b03 = b00;
            float32x4_t b10 = b00, b11 = b00, b12 = b00, b13 = b00;
            float32x4_t b20 = b00, b21 = b00, b22 = b00, b23 = b00;
            float32x4_t b30 = b00, b31 = b00, b32 = b00, b33 = b00;

            for (int t2=0; t2<seq_k; t2++) {
                float32x4_t pv = vdupq_n_f32(w0[t2]);
                pv = vsetq_lane_f32(w1[t2], pv, 1);
                pv = vsetq_lane_f32(w2[t2], pv, 2);
                pv = vsetq_lane_f32(w3[t2], pv, 3);

                const float* vv = vp+(size_t)t2*64+c;
                const float32x4_t v0 = vld1q_f32(vv);
                const float32x4_t v1 = vld1q_f32(vv+4);
                const float32x4_t v2 = vld1q_f32(vv+8);
                const float32x4_t v3 = vld1q_f32(vv+12);
                b00 = vfmaq_laneq_f32(b00, v0, pv, 0);
                b01 = vfmaq_laneq_f32(b01, v1, pv, 0);
                b02 = vfmaq_laneq_f32(b02, v2, pv, 0);
                b03 = vfmaq_laneq_f32(b03, v3, pv, 0);
                b10 = vfmaq_laneq_f32(b10, v0, pv, 1);
                b11 = vfmaq_laneq_f32(b11, v1, pv, 1);
                b12 = vfmaq_laneq_f32(b12, v2, pv, 1);
                b13 = vfmaq_laneq_f32(b13, v3, pv, 1);
                b20 = vfmaq_laneq_f32(b20, v0, pv, 2);
                b21 = vfmaq_laneq_f32(b21, v1, pv, 2);
                b22 = vfmaq_laneq_f32(b22, v2, pv, 2);
                b23 = vfmaq_laneq_f32(b23, v3, pv, 2);
                b30 = vfmaq_laneq_f32(b30, v0, pv, 3);
                b31 = vfmaq_laneq_f32(b31, v1, pv, 3);
                b32 = vfmaq_laneq_f32(b32, v2, pv, 3);
                b33 = vfmaq_laneq_f32(b33, v3, pv, 3);
            }

            float* o0 = out+((size_t)(q0+t*4+0)*n_q+h)*64+c;
            if (rows > 0) { vst1q_f32(o0, b00); vst1q_f32(o0+4, b01);
                            vst1q_f32(o0+8, b02); vst1q_f32(o0+12, b03); }
            if (rows > 1) { float* o = out+((size_t)(q0+t*4+1)*n_q+h)*64+c;
                            vst1q_f32(o, b10); vst1q_f32(o+4, b11);
                            vst1q_f32(o+8, b12); vst1q_f32(o+12, b13); }
            if (rows > 2) { float* o = out+((size_t)(q0+t*4+2)*n_q+h)*64+c;
                            vst1q_f32(o, b20); vst1q_f32(o+4, b21);
                            vst1q_f32(o+8, b22); vst1q_f32(o+12, b23); }
            if (rows > 3) { float* o = out+((size_t)(q0+t*4+3)*n_q+h)*64+c;
                            vst1q_f32(o, b30); vst1q_f32(o+4, b31);
                            vst1q_f32(o+8, b32); vst1q_f32(o+12, b33); }
        }
    }
}

void gqa_attention_dense(float* out, const float* Q, const float* K, const float* V,
                         int seq_q, int seq_k, int n_q, int n_kv, int head_dim,
                         float scale, const float* K_pre) {
    const int group = n_q/n_kv;
    const int skp   = K_pre ? hal::kt_stride(seq_k) : ((seq_k+3) & ~3);

    if (!hal::env::attn_dense()) {   // A/B: the pre-dense-path behaviour
        static thread_local std::vector<float> zero;
        if (zero.size() < (size_t)seq_q*seq_k) zero.assign((size_t)seq_q*seq_k, 0.0f);
        gqa_attention_masked(out, Q, K, V, seq_q, seq_k, n_q, n_kv, head_dim,
                             scale, zero.data(), K_pre);
        return;
    }

    // head_dim != 64 has no register-blocked micro-kernel; run the generic row
    // path against a single zero mask row (dense semantics, no [q,k] mask array).
    if (head_dim != 64) {
        static thread_local std::vector<float> zrow;
        if ((int)zrow.size() < seq_k) zrow.assign(seq_k, 0.0f);
        const float* zr = zrow.data();
        std::vector<float> KTv;
        const float* KTw = K_pre;
        if (!KTw) {
            KTv.resize((size_t)n_kv*head_dim*skp, 0.0f);
            for (int t2=0; t2<seq_k; t2++)
                for (int kv=0; kv<n_kv; kv++)
                    for (int d=0; d<head_dim; d++)
                        KTv[((size_t)kv*head_dim+d)*skp+t2] = K[((size_t)t2*n_kv+kv)*head_dim+d];
            KTw = KTv.data();
        }
#if defined(_OPENMP)
        #pragma omp parallel
#endif
      {
        std::vector<float> sc(skp);
#if defined(_OPENMP)
        #pragma omp for schedule(static)
#endif
        for (int qi=0; qi<n_q*seq_q; qi++) {
            const int h = qi/seq_q, t1 = qi%seq_q, kv = h/group;
            attn_row_kt_neon(out+((size_t)t1*n_q+h)*head_dim,
                             Q+((size_t)t1*n_q+h)*head_dim, zr,
                             KTw+(size_t)kv*head_dim*skp, V, kv, n_kv,
                             head_dim, seq_k, skp, scale, sc.data());
        }
      }
        return;
    }

    // K^T [kv][d][key], built once (the masked path's transpose, verbatim)
    static thread_local std::vector<float> KT;
    const float* KTr = K_pre;
    if (!KTr) {
        if (KT.size() < (size_t)n_kv*head_dim*skp) KT.resize((size_t)n_kv*head_dim*skp);
        float* const KTw = KT.data();
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int jb=0; jb<skp; jb+=4) {
            const int je = seq_k-jb < 4 ? seq_k-jb : 4;
            for (int kv=0; kv<n_kv; kv++)
                for (int d=0; d<head_dim; d++) {
                    float* dst = KTw+((size_t)kv*head_dim+d)*skp+jb;
                    for (int j=0; j<je; j++)
                        dst[j] = K[((size_t)(jb+j)*n_kv+kv)*head_dim+d];
                    for (int j=je; j<4; j++) dst[j] = 0.0f;
                }
        }
        KTr = KT.data();
    }

    static thread_local std::vector<float> VP;
    if (VP.size() < (size_t)seq_k*head_dim) VP.resize((size_t)seq_k*head_dim);
    float* const VPw = VP.data();   // hoisted before the parallel region, like KT

    const int QB  = hal::env::attn_qblock();          // TCPU_ATTN_QB, default 16
    const int nqb = (seq_q+QB-1)/QB;

    // Heads run outermost and sequentially so the whole team shares one head's
    // K^T/V panels; the query blocks inside are what the threads split.
#if defined(_OPENMP)
    #pragma omp parallel
#endif
  {
    std::vector<float> scores((size_t)QB*skp);
    std::vector<float> qt((size_t)((QB+3)/4)*64*4);
    int built = -1;
    for (int h=0; h<n_q; h++) {
        const int kv = h/group;
        const float* kt = KTr+(size_t)kv*head_dim*skp;
        // Gather this kv head's V rows into a contiguous [seq_k][head_dim] panel.
        // In the packed layout one head's rows sit n_kv*head_dim floats apart
        // (3 KB at the ViT's shape), so AV touches one 64-byte line per key and
        // walks a 3 MB span; from the panel it walks 256 KB sequentially. Same
        // values in a different place, same FMA order - measured 2x on the AV
        // phase (30.8 -> 15.5 ms at seq 1024, 12 heads, Cortex-A76). Rebuilt only
        // when the kv head changes, so a GQA group pays for it once.
        if (kv != built) {
#if defined(_OPENMP)
            #pragma omp for schedule(static)
#endif
            for (int t2=0; t2<seq_k; t2++)
                std::memcpy(VPw+(size_t)t2*head_dim,
                            V+((size_t)t2*n_kv+kv)*head_dim,
                            (size_t)head_dim*sizeof(float));
            built = kv;
        }
#if defined(_OPENMP)
        #pragma omp for schedule(static)
#endif
        for (int b=0; b<nqb; b++) {
            const int q0 = b*QB;
            const int nq = seq_q-q0 < QB ? seq_q-q0 : QB;
            attn_dense_qblock(out, Q, VPw, kt, q0, nq, n_q, h,
                              seq_k, skp, scale, scores.data(), qt.data());
        }
    }
  }
}

} // namespace tcpu

#endif // TCPU_HAL_NEON
