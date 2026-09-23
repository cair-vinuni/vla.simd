/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// x86 AVX2 masked attention: K transposed once to [kv][d][key] (or taken
// pre-transposed from the fused K projection), vectorized softmax, jmax row
// bound.
//
// Both halves of the query-tiled path share one 6-row tile. QK holds 6 queries x
// 16 keys and A*V holds 6 queries x 16 output dims, each 12 accumulators + 2
// loads + 1 broadcast = 15 YMM, the same budget as the packed GEMM. A*V used to
// finish one query row at a time, so the six rows that had just shared a K^T
// stream each re-walked the head's whole V slice; tiling it drops V reads per
// tile from six passes to four (one per head_dim/16 block). Measured end to end
// on an i5-12400F at 6 threads: IMPACT 153.2 -> 146.7 ms, ACT 124.5 -> 120.4 ms,
// Octo neutral (its block-causal mask already bounded most rows).
//
// Bit-exactness vs the per-row form:
//   - scores are prescaled by 1/sum in the softmax epilogue, which is the same
//     fp32 product `p[t2]*inv` the per-row A*V computed at use time;
//   - a tile softmaxes over the tile-wide bound instead of each row's own: the
//     extra lanes are -inf -> exp() gives exactly 0.0f, so they change neither
//     the max, nor the (vector-lane) sum, nor the A*V accumulation (acc + 0*v);
//   - the per-row zero-weight skip is a pure optimisation (0*v adds nothing), so
//     dropping it inside a tile is value-preserving;
//   - degenerate rows (fully-masked -> uniform weights) keep the per-row path.

#include "../arch.h"
#if TCPU_HAL_X86

#include "../simd.h"
#include "../common/env.h"
#include "../common/layout.h"
#include "../../ops/lm_ops.h"
#include <limits>
#include <vector>
#include <cstddef>
using std::size_t;

namespace tcpu {

// fully-masked row -> uniform over all keys (dense semantics)
static inline void attn_uniform_row(float* o, const float* V, int seq_k,
                                    int kv, int n_kv, int head_dim) {
    const float u = 1.0f/(float)seq_k;
    for (int d=0; d<head_dim; d++)
        o[d] = 0.0f;
    for (int t2=0; t2<seq_k; t2++)
        simd_axpy(o, u, V+((size_t)t2*n_kv+kv)*head_dim, head_dim);
}

// Mask + softmax for one row, over the tile-wide bound `sbound` (a multiple of
// 8, >= this row's own bound). Leaves scores prescaled by 1/sum, i.e. exactly
// the values the per-row A*V used to form as `p[t2]*inv`. Returns false for the
// degenerate all-masked case, which the caller finishes per row.
static inline bool attn_softmax_row(float* scores, const float* mrow,
                                    int seq_k, int jmax, int sbound) {
    const float NINF = -std::numeric_limits<float>::infinity();

    // add the mask row; lanes past jmax (still < seq_k) -> -inf
    const int me = jmax < seq_k ? jmax : seq_k;
    int j;
    if (mrow) {
        for (j=0; j+8 <= me; j += 8)
            _mm256_storeu_ps(scores+j,
                             _mm256_add_ps(_mm256_loadu_ps(scores+j),
                                           _mm256_loadu_ps(mrow+j)));
        for (; j < me; j++)
            scores[j] += mrow[j];
    }
    for (j=me; j<sbound; j++)
        scores[j] = NINF;

    __m256 vmax = _mm256_set1_ps(NINF);
    for (j=0; j<sbound; j += 8)
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(scores+j));
    const float maxs = hmax8(vmax);
    if (maxs <= std::numeric_limits<float>::lowest())
        return false;   // every remaining score collapsed -> uniform (dense semantics)

    const __m256 vm = _mm256_set1_ps(maxs);
    __m256 vsum = _mm256_setzero_ps();
    for (j=0; j<sbound; j += 8) {
        const __m256 e = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scores+j), vm));
        _mm256_storeu_ps(scores+j, e);
        vsum = _mm256_add_ps(vsum, e);
    }
    const __m256 vinv = _mm256_set1_ps(1.0f/hsum8(vsum));
    for (j=0; j<sbound; j += 8)
        _mm256_storeu_ps(scores+j, _mm256_mul_ps(_mm256_loadu_ps(scores+j), vinv));
    return true;
}

// Per-row A*V (the Intel form), for tiles that contain a degenerate row. Scores
// are already prescaled, so this is a plain weighted sum in key order.
static inline void attn_av_row(const float* p, float* o, const float* V, int av_end,
                               int kv, int n_kv, int head_dim) {
    if (head_dim == 64) {
        __m256 av[8];
        for (int v=0; v<8; v++)
            av[v] = _mm256_setzero_ps();
        for (int t2=0; t2<av_end; t2++) {
            if (p[t2] == 0.0f) continue;
            const float* vv = V+((size_t)t2*n_kv+kv)*64;
            const __m256 a = _mm256_set1_ps(p[t2]);
            for (int v=0; v<8; v++)
                av[v] = _mm256_fmadd_ps(a, _mm256_loadu_ps(vv+8*v), av[v]);
        }
        for (int v=0; v<8; v++)
            _mm256_storeu_ps(o+8*v, av[v]);
        return;
    }
    for (int d=0; d<head_dim; d++)
        o[d] = 0.0f;
    for (int t2=0; t2<av_end; t2++) {
        if (p[t2] == 0.0f) continue;
        simd_axpy(o, p[t2], V+((size_t)t2*n_kv+kv)*head_dim, head_dim);
    }
}

// ROWSx16 A*V micro-kernel: the query rows of one QK tile share every V load.
// Per key: 2 V loads + ROWS score broadcasts vs 2*ROWS FMAs -> FMA-bound on
// Zen 3's 2 load ports at ROWS=6, where the per-row form is load/traffic-bound.
// Each output element accumulates keys in ascending order, as before.
//
// V is read in place at the model's [key][kv][d] stride: the 16 lanes one pass
// needs are contiguous within a key, so no repack is required. The Zen backend
// does pre-pack (vpack) because its L2 set-aliasing at that stride costs more
// than the copy; on Intel the pack measured as a net loss - a full V copy per
// call against a cache that absorbs the stride. Same values either way: each
// output element still accumulates keys in ascending order.
template <int ROWS>
static inline void av_tile16(const float* p, size_t lds, float* o, size_t ldo,
                             const float* vb, size_t ldt, size_t ldblk, int av_end,
                             int head_dim) {
    for (int d0=0; d0<head_dim; d0 += 16) {
        __m256 c0[ROWS];
        __m256 c1[ROWS];
        for (int i=0; i<ROWS; i++) {
            c0[i] = _mm256_setzero_ps();
            c1[i] = _mm256_setzero_ps();
        }

        const float* vv = vb+(size_t)(d0/16)*ldblk;
        for (int t2=0; t2<av_end; t2++, vv += ldt) {
            const __m256 b0 = _mm256_loadu_ps(vv);
            const __m256 b1 = _mm256_loadu_ps(vv+8);
            for (int i=0; i<ROWS; i++) {
                const __m256 a = _mm256_set1_ps(p[(size_t)i*lds+t2]);
                c0[i] = _mm256_fmadd_ps(a, b0, c0[i]);
                c1[i] = _mm256_fmadd_ps(a, b1, c1[i]);
            }
        }

        for (int i=0; i<ROWS; i++) {
            _mm256_storeu_ps(o+(size_t)i*ldo+d0,   c0[i]);
            _mm256_storeu_ps(o+(size_t)i*ldo+d0+8, c1[i]);
        }
    }
}

static inline void av_tile16_rows(int rows, const float* p, size_t lds, float* o, size_t ldo,
                                  const float* vb, size_t ldt, size_t ldblk, int av_end,
                                  int head_dim) {
    switch (rows) {
        case 6: av_tile16<6>(p, lds, o, ldo, vb, ldt, ldblk, av_end, head_dim); break;
        case 5: av_tile16<5>(p, lds, o, ldo, vb, ldt, ldblk, av_end, head_dim); break;
        case 4: av_tile16<4>(p, lds, o, ldo, vb, ldt, ldblk, av_end, head_dim); break;
        case 3: av_tile16<3>(p, lds, o, ldo, vb, ldt, ldblk, av_end, head_dim); break;
        case 2: av_tile16<2>(p, lds, o, ldo, vb, ldt, ldblk, av_end, head_dim); break;
        default: av_tile16<1>(p, lds, o, ldo, vb, ldt, ldblk, av_end, head_dim); break;
    }
}

// V comes pre-packed as [kv][d/16][key][16], so each of the four head_dim passes
// streams one contiguous run instead of walking the model's [key][kv][d] layout
// at an n_kv*head_dim stride. That stride is the same L1/L2 set-aliasing trap as
// the K^T one: at the ViT shape it is 3072 B, which folds the head's 256 KB V
// slice onto 64 of the L2's 1024 sets (16 lines deep in an 8-way cache), so a
// "resident" slice re-misses every pass. Measured A*V alone at seq 1024, 6
// threads: 411 -> 676 GF/s, 1.49x including the pack.
//
// V [key][kv][d] -> VP [kv][d/16][key][16]. Pure copy, so the A*V above sees the
// same values in the same key order as the per-row form.
static void vpack(float* VP, const float* V, int seq_k, int n_kv, int head_dim) {
    const int nb = head_dim/16;
    const size_t ldv = (size_t)n_kv*head_dim;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int kv=0; kv<n_kv; kv++) {
        for (int db=0; db<nb; db++) {
            float* dst = VP+((size_t)kv*nb+db)*(size_t)seq_k*16;
            const float* src = V+(size_t)kv*head_dim+db*16;
            for (int t=0; t<seq_k; t++, dst += 16, src += ldv) {
                _mm256_storeu_ps(dst,   _mm256_loadu_ps(src));
                _mm256_storeu_ps(dst+8, _mm256_loadu_ps(src+8));
            }
        }
    }
}

// ROWSx16 QK micro-kernel over transposed keys: ROWS query rows share every K
// panel load (up to 12 accumulators + 2 K loads + 1 broadcast = 15 YMM, the
// packed-GEMM budget; the per-query pass re-streams K^T for every query). Each
// 8-key lane group accumulates sequentially over d - the same chain as the
// per-query pass, so score values are bit-identical.
template <int ROWS>
static inline void qk_rows16(float* scores, size_t lds, const float* q0, size_t ldq,
                             const float* kt, size_t ldk, int head_dim, float scale,
                             int skt) {
    const __m256 vscale = _mm256_set1_ps(scale);
    int j = 0;
    for (; j+16 <= skt; j += 16) {
        __m256 c0[ROWS];
        __m256 c1[ROWS];
        for (int i=0; i<ROWS; i++) {
            c0[i] = _mm256_setzero_ps();
            c1[i] = _mm256_setzero_ps();
        }
#pragma GCC unroll 1
        for (int d=0; d<head_dim; d++) {
            const float* kr = kt+(size_t)d*ldk+j;
            const __m256 b0 = _mm256_loadu_ps(kr);
            const __m256 b1 = _mm256_loadu_ps(kr+8);
            for (int i=0; i<ROWS; i++) {
                const __m256 qd = _mm256_set1_ps(q0[(size_t)i*ldq+d]);
                c0[i] = _mm256_fmadd_ps(qd, b0, c0[i]);
                c1[i] = _mm256_fmadd_ps(qd, b1, c1[i]);
            }
        }
        for (int i=0; i<ROWS; i++) {
            _mm256_storeu_ps(scores+(size_t)i*lds+j,   _mm256_mul_ps(c0[i], vscale));
            _mm256_storeu_ps(scores+(size_t)i*lds+j+8, _mm256_mul_ps(c1[i], vscale));
        }
    }
    for (; j+8 <= skt; j += 8) {
        __m256 c[ROWS];
        for (int i=0; i<ROWS; i++)
            c[i] = _mm256_setzero_ps();
        for (int d=0; d<head_dim; d++) {
            const __m256 b0 = _mm256_loadu_ps(kt+(size_t)d*ldk+j);
            for (int i=0; i<ROWS; i++)
                c[i] = _mm256_fmadd_ps(_mm256_set1_ps(q0[(size_t)i*ldq+d]), b0, c[i]);
        }
        for (int i=0; i<ROWS; i++)
            _mm256_storeu_ps(scores+(size_t)i*lds+j, _mm256_mul_ps(c[i], vscale));
    }
}

static inline void qk_rows16_rows(int rows, float* scores, size_t lds, const float* q0,
                                  size_t ldq, const float* kt, size_t ldk, int head_dim,
                                  float scale, int skt) {
    switch (rows) {
        case 6: qk_rows16<6>(scores, lds, q0, ldq, kt, ldk, head_dim, scale, skt); break;
        case 5: qk_rows16<5>(scores, lds, q0, ldq, kt, ldk, head_dim, scale, skt); break;
        case 4: qk_rows16<4>(scores, lds, q0, ldq, kt, ldk, head_dim, scale, skt); break;
        case 3: qk_rows16<3>(scores, lds, q0, ldq, kt, ldk, head_dim, scale, skt); break;
        case 2: qk_rows16<2>(scores, lds, q0, ldq, kt, ldk, head_dim, scale, skt); break;
        default: qk_rows16<1>(scores, lds, q0, ldq, kt, ldk, head_dim, scale, skt); break;
    }
}

void gqa_attention_masked(float* out, const float* Q, const float* K, const float* V,
                          int seq_q, int seq_k, int n_q, int n_kv, int head_dim,
                          float scale, const float* mask, const float* K_pre) {
    const int group = n_q/n_kv;
    // Key-vectorized path: transpose K to [kv][d][key] once (or take K_pre straight
    // from dense_linear_packed_kt), then each query's scores for 8 keys come from one
    // fmadd stream (no per-dot hsum), and softmax max/exp/sum are vectorized
    // (exp256_ps). Blocked keys (mask == finfo.min) flush to weight exactly 0.0f via
    // the exp clamp, so the AV pass still skips them.
    const int skp = hal::kt_stride(seq_k);   // K^T leading dimension (padded on Zen)
    const bool zen = hal::env::zen();
    const int sds = zen ? ((seq_k+7) & ~7) + 8 : skp;   // scores leading dimension (same aliasing)
    const float* KTp = K_pre;
    if (!KTp) {
        // reused across calls (the op is entered from one thread; parallelism is
        // inside) - avoids re-faulting ~1 MB per layer. Pad columns zeroed below.
        // NOTE: grab the pointer BEFORE the parallel regions - inside them each
        // OpenMP worker would see its own (empty) thread_local instance.
        static thread_local std::vector<float> KT;
        if (KT.size() < (size_t)n_kv*head_dim*skp)
            KT.resize((size_t)n_kv*head_dim*skp);
        hal::transpose_kt(KT.data(), K, seq_k, n_kv, head_dim, skp);
        KTp = KT.data();
    }

    if (hal::env::attn_qtile() && seq_q >= 6 && head_dim%16 == 0) {
        // Query-tiled path: 6-row tiles amortize the K^T streams (QK) and, since
        // the tile softmaxes over a shared bound, the V streams as well (A*V).
        // The row bound is hoisted (computed once per query row instead of once
        // per (head, query)). Tiles spanning rows with different bounds compute
        // extra score lanes; those lanes are -inf -> exp gives exactly 0.0f, so
        // they change neither the max, the sum, nor the A*V accumulation.
        constexpr int QR = 6;
        const int ntiles = (seq_q+QR-1)/QR;

        // V repacked once per call for the tiled A*V (same reuse rule as KT:
        // grab the pointer before any parallel region).
        const float* VB = V;
        size_t ldkv = head_dim, ldt = (size_t)n_kv*head_dim, ldblk = 16;
        if (zen) {
            static thread_local std::vector<float> VPbuf;
            if (VPbuf.size() < (size_t)n_kv*head_dim*seq_k)
                VPbuf.resize((size_t)n_kv*head_dim*seq_k);
            vpack(VPbuf.data(), V, seq_k, n_kv, head_dim);
            VB = VPbuf.data();
            ldkv = (size_t)head_dim*seq_k;
            ldt = 16;
            ldblk = (size_t)seq_k*16;
        }

        std::vector<int> jmaxv(seq_q, seq_k);
        if (mask) {
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static)
#endif
            for (int t1=0; t1<seq_q; t1++)
                jmaxv[t1] = hal::row_bound(mask+(size_t)t1*seq_k, seq_k);
        }

#if defined(_OPENMP)
        #pragma omp parallel
#endif
      {
        std::vector<float> scores((size_t)QR*sds);   // per-thread scratch
#if defined(_OPENMP)
        #pragma omp for schedule(static)
#endif
        for (int u=0; u<n_q*ntiles; u++) {
            const int h    = u/ntiles;
            const int t0   = (u%ntiles)*QR;
            const int rows = seq_q-t0 < QR ? seq_q-t0 : QR;
            const int kv   = h/group;
            const float* kt = KTp+(size_t)kv*head_dim*skp;
            const float* q0 = Q+((size_t)t0*n_q+h)*head_dim;
            const size_t ldq = (size_t)n_q*head_dim;
            float* o0 = out+((size_t)t0*n_q+h)*head_dim;

            int tjm = 0;
            for (int i=0; i<rows; i++)
                if (jmaxv[t0+i] > tjm) tjm = jmaxv[t0+i];
            if (tjm == 0) {   // whole tile fully masked
                for (int i=0; i<rows; i++)
                    attn_uniform_row(o0+(size_t)i*ldq, V, seq_k, kv, n_kv, head_dim);
                continue;
            }

            const int skt = (tjm+7) & ~7;
            qk_rows16_rows(rows, scores.data(), sds, q0, ldq, kt, skp, head_dim, scale, skt);

            bool ok[QR];
            bool all_ok = true;
            for (int i=0; i<rows; i++) {
                const int jmax = jmaxv[t0+i];
                ok[i] = jmax > 0 &&
                        attn_softmax_row(scores.data()+(size_t)i*sds,
                                         mask ? mask+(size_t)(t0+i)*seq_k : nullptr,
                                         seq_k, jmax, skt);
                all_ok &= ok[i];
            }

            if (all_ok) {
                av_tile16_rows(rows, scores.data(), sds, o0, ldq, VB+(size_t)kv*ldkv, ldt, ldblk,
                               tjm, head_dim);
                continue;
            }

            // rare: a degenerate (fully-masked) row in the tile - finish per row
            for (int i=0; i<rows; i++) {
                float* o = o0+(size_t)i*ldq;
                if (ok[i]) attn_av_row(scores.data()+(size_t)i*sds, o, V, jmaxv[t0+i],
                                       kv, n_kv, head_dim);
                else       attn_uniform_row(o, V, seq_k, kv, n_kv, head_dim);
            }
        }
      }
        return;
    }

    // Per-query path (short queries: decode steps, cross-attention with seq_q < 6).
#if defined(_OPENMP)
    #pragma omp parallel
#endif
  {
    std::vector<float> scores(sds);   // per-thread scratch (reused across queries)
#if defined(_OPENMP)
    #pragma omp for schedule(static)
#endif
    for (int qi=0; qi<n_q*seq_q; qi++) {
        const int h  = qi/seq_q;
        const int t1 = qi%seq_q;
        const int kv = h/group;
        const float* q = Q+((size_t)t1*n_q+h)*head_dim;
        const float* mrow = mask ? mask+(size_t)t1*seq_k : nullptr;
        const float* kt = KTp+(size_t)kv*head_dim*skp;

        // block-causal masks end each row with a blocked tail: bound the QK/softmax
        // loops at the last allowed key (bit-exact; blocked keys had weight 0 anyway)
        const int jmax = mrow ? hal::row_bound(mrow, seq_k) : seq_k;

        float* o = out+((size_t)t1*n_q+h)*head_dim;
        if (jmax == 0) {
            attn_uniform_row(o, V, seq_k, kv, n_kv, head_dim);
            continue;
        }
        const int skq = (jmax+7) & ~7;

        // scores = (q . K^T) * scale, 32 keys per pass (4 accumulator chains)
        const __m256 vscale = _mm256_set1_ps(scale);
        int j = 0;
        for (; j+32 <= skq; j += 32) {
            __m256 a0 = _mm256_setzero_ps();
            __m256 a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps();
            __m256 a3 = _mm256_setzero_ps();
            for (int d=0; d<head_dim; d++) {
                const __m256 qd = _mm256_set1_ps(q[d]);
                const float* kr = kt+(size_t)d*skp+j;
                a0 = _mm256_fmadd_ps(qd, _mm256_loadu_ps(kr),    a0);
                a1 = _mm256_fmadd_ps(qd, _mm256_loadu_ps(kr+8),  a1);
                a2 = _mm256_fmadd_ps(qd, _mm256_loadu_ps(kr+16), a2);
                a3 = _mm256_fmadd_ps(qd, _mm256_loadu_ps(kr+24), a3);
            }
            _mm256_storeu_ps(scores.data()+j,    _mm256_mul_ps(a0, vscale));
            _mm256_storeu_ps(scores.data()+j+8,  _mm256_mul_ps(a1, vscale));
            _mm256_storeu_ps(scores.data()+j+16, _mm256_mul_ps(a2, vscale));
            _mm256_storeu_ps(scores.data()+j+24, _mm256_mul_ps(a3, vscale));
        }
        for (; j+8 <= skq; j += 8) {
            __m256 a0 = _mm256_setzero_ps();
            for (int d=0; d<head_dim; d++)
                a0 = _mm256_fmadd_ps(_mm256_set1_ps(q[d]),
                                     _mm256_loadu_ps(kt+(size_t)d*skp+j), a0);
            _mm256_storeu_ps(scores.data()+j, _mm256_mul_ps(a0, vscale));
        }

        if (attn_softmax_row(scores.data(), mrow, seq_k, jmax, skq))
            attn_av_row(scores.data(), o, V, jmax, kv, n_kv, head_dim);
        else
            attn_uniform_row(o, V, seq_k, kv, n_kv, head_dim);
    }
  }
}

} // namespace tcpu

#endif // TCPU_HAL_X86
