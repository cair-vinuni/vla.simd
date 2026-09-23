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
// The structure is ported from amd/attn.cpp with its V pre-pack left out: the 16
// lanes one A*V pass needs are already contiguous within a key, and on Intel the
// pack measured as a net loss (a full V copy per call against a cache that
// absorbs the stride). Zen keeps it because its L2 set-aliasing costs more than
// the copy. Values are identical either way - see attn_softmax_row for why the
// tile-wide softmax bound is value-preserving.

#include "../arch.h"
#if TCPU_HAL_X86

#include "../simd.h"
#include "../common/env.h"
#include "../../ops/lm_ops.h"
#include <cmath>
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

// Post-QK per-row pass (mask add, softmax, AV), shared by the per-query and
// query-tiled paths: identical op order on identical score values -> the two
// paths are bit-exact vs each other.
static inline void attn_softmax_av_row(float* scores, const float* mrow, float* o,
                                       const float* V, int seq_k, int jmax,
                                       int kv, int n_kv, int head_dim) {
    const float NINF = -std::numeric_limits<float>::infinity();
    const int skq = (jmax+7) & ~7;

    // add the mask row; lanes past jmax (still < seq_k) -> -inf
    const int me = skq < seq_k ? skq : seq_k;
    int j;
    if (mrow) {
        for (j=0; j+8 <= me; j += 8)
            _mm256_storeu_ps(scores+j,
                             _mm256_add_ps(_mm256_loadu_ps(scores+j),
                                           _mm256_loadu_ps(mrow+j)));
        for (; j < me; j++)
            scores[j] += mrow[j];
    }
    for (j=me; j<skq; j++)
        scores[j] = NINF;

    __m256 vmax = _mm256_set1_ps(NINF);
    for (j=0; j<skq; j += 8)
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(scores+j));
    const float maxs = hmax8(vmax);

    float sum;
    if (maxs <= std::numeric_limits<float>::lowest()) {
        // all remaining scores collapsed to finfo.min -> uniform (dense semantics)
        for (j=0; j<seq_k; j++)
            scores[j] = 1.0f;
        sum = (float)seq_k;
    } else {
        const __m256 vm = _mm256_set1_ps(maxs);
        __m256 vsum = _mm256_setzero_ps();
        for (j=0; j<skq; j += 8) {
            const __m256 e = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scores+j), vm));
            _mm256_storeu_ps(scores+j, e);
            vsum = _mm256_add_ps(vsum, e);
        }
        sum = hsum8(vsum);
    }

    const int av_end = maxs <= std::numeric_limits<float>::lowest() ? seq_k : jmax;
    const float inv = 1.0f/sum;
    if (head_dim == 64) {
        // AV with the output resident in 8 accumulators: per key 1 broadcast +
        // 8 V loads + 8 FMAs, no o read-modify-write. Same key order -> bit-exact
        // vs the axpy path.
        __m256 av[8];
        for (int v=0; v<8; v++)
            av[v] = _mm256_setzero_ps();
        for (int t2=0; t2<av_end; t2++) {
            if (scores[t2] == 0.0f) continue;
            const float* vv = V+((size_t)t2*n_kv+kv)*64;
            const __m256 a = _mm256_set1_ps(scores[t2]*inv);
            for (int v=0; v<8; v++)
                av[v] = _mm256_fmadd_ps(a, _mm256_loadu_ps(vv+8*v), av[v]);
        }
        for (int v=0; v<8; v++)
            _mm256_storeu_ps(o+8*v, av[v]);
    } else {
        for (int d=0; d<head_dim; d++)
            o[d] = 0.0f;
        for (int t2=0; t2<av_end; t2++) {
            if (scores[t2] == 0.0f) continue;
            const float* vv = V+((size_t)t2*n_kv+kv)*head_dim;
            simd_axpy(o, scores[t2]*inv, vv, head_dim);
        }
    }
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
// does pre-pack (amd/attn.cpp, vpack) because its L2 set-aliasing at that stride
// costs more than the copy; on Intel the pack measured as a net loss - a full V
// copy per call against a cache that absorbs the stride. Same values either way:
// each output element still accumulates keys in ascending order.
template <int ROWS>
static inline void av_tile16(const float* p, size_t lds, float* o, size_t ldo,
                             const float* V, int av_end, int kv, int n_kv, int head_dim) {
    const size_t ldv = (size_t)n_kv*head_dim;
    for (int d0=0; d0<head_dim; d0 += 16) {
        __m256 c0[ROWS];
        __m256 c1[ROWS];
        for (int i=0; i<ROWS; i++) {
            c0[i] = _mm256_setzero_ps();
            c1[i] = _mm256_setzero_ps();
        }

        const float* vv = V+(size_t)kv*head_dim+d0;
        for (int t2=0; t2<av_end; t2++, vv += ldv) {
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
                                  const float* V, int av_end, int kv, int n_kv, int head_dim) {
    switch (rows) {
        case 6: av_tile16<6>(p, lds, o, ldo, V, av_end, kv, n_kv, head_dim); break;
        case 5: av_tile16<5>(p, lds, o, ldo, V, av_end, kv, n_kv, head_dim); break;
        case 4: av_tile16<4>(p, lds, o, ldo, V, av_end, kv, n_kv, head_dim); break;
        case 3: av_tile16<3>(p, lds, o, ldo, V, av_end, kv, n_kv, head_dim); break;
        case 2: av_tile16<2>(p, lds, o, ldo, V, av_end, kv, n_kv, head_dim); break;
        default: av_tile16<1>(p, lds, o, ldo, V, av_end, kv, n_kv, head_dim); break;
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
    const int skp = (seq_k+7) & ~7;
    const float* KTp = K_pre;
    if (!KTp) {
        // reused across calls (the op is entered from one thread; parallelism is
        // inside) - avoids re-faulting ~1 MB per layer. Pad columns zeroed below.
        // NOTE: grab the pointer BEFORE the parallel regions - inside them each
        // OpenMP worker would see its own (empty) thread_local instance.
        static thread_local std::vector<float> KT;
        if (KT.size() < (size_t)n_kv*head_dim*skp)
            KT.resize((size_t)n_kv*head_dim*skp);
        float* const KTw = KT.data();
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int jb=0; jb<skp; jb += 8) {
            const int je = seq_k-jb < 8 ? seq_k-jb : 8;
            for (int kv=0; kv<n_kv; kv++)
                for (int d=0; d<head_dim; d++) {
                    float* dst = KTw+((size_t)kv*head_dim+d)*skp+jb;
                    for (int j=0; j<je; j++)
                        dst[j] = K[((size_t)(jb+j)*n_kv+kv)*head_dim+d];
                    for (int j=je; j<8; j++)
                        dst[j] = 0.0f;
                }
        }
        KTp = KTw;
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


        std::vector<int> jmaxv(seq_q, seq_k);
        if (mask) {
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static)
#endif
            for (int t1=0; t1<seq_q; t1++) {
                const float* mrow = mask+(size_t)t1*seq_k;
                int jm = seq_k;
                while (jm > 0 && mrow[jm-1] == std::numeric_limits<float>::lowest())
                    jm--;
                jmaxv[t1] = jm;
            }
        }

#if defined(_OPENMP)
        #pragma omp parallel
#endif
      {
        std::vector<float> scores((size_t)QR*skp);   // per-thread scratch
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
            qk_rows16_rows(rows, scores.data(), skp, q0, ldq, kt, skp, head_dim, scale, skt);

            bool ok[QR];
            bool all_ok = true;
            for (int i=0; i<rows; i++) {
                const int jmax = jmaxv[t0+i];
                ok[i] = jmax > 0 &&
                        attn_softmax_row(scores.data()+(size_t)i*skp,
                                         mask ? mask+(size_t)(t0+i)*seq_k : nullptr,
                                         seq_k, jmax, skt);
                all_ok &= ok[i];
            }

            if (all_ok) {
                av_tile16_rows(rows, scores.data(), skp, o0, ldq, V, tjm, kv, n_kv, head_dim);
                continue;
            }

            // rare: a degenerate (fully-masked) row in the tile - finish per row
            for (int i=0; i<rows; i++) {
                float* o = o0+(size_t)i*ldq;
                if (ok[i]) attn_av_row(scores.data()+(size_t)i*skp, o, V, jmaxv[t0+i],
                                       kv, n_kv, head_dim);
                else       attn_uniform_row(o, V, seq_k, kv, n_kv, head_dim);
            }
        }
      }
        return;
    }

#if defined(_OPENMP)
    #pragma omp parallel
#endif
  {
    std::vector<float> scores(skp);   // per-thread scratch (reused across queries)
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
        int jmax = seq_k;
        while (mrow && jmax > 0 && mrow[jmax-1] == std::numeric_limits<float>::lowest())
            jmax--;

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

        attn_softmax_av_row(scores.data(), mrow, o, V, seq_k, jmax, kv, n_kv, head_dim);
    }
  }
}

} // namespace tcpu

#endif // TCPU_HAL_X86
