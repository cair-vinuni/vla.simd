/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "attention.h"
#include "encoder.h"
#include "../hal/common/layout.h"
#include "../ops/lm_ops.h"
#include <cmath>
#include <cstddef>
using std::size_t;

namespace tcpu {
namespace nn {

void MhaMasked::set_shape(int heads_, int head_dim_) {
    heads    = heads_;
    head_dim = head_dim_;
    scale    = 1.0f/std::sqrt((float)head_dim_);
}

void MhaMasked::forward(float* out, const float* x_norm, int total, const float* mask,
                        int q_row, Scratch& s, Prof* prof, bool add_out) const {
    const int D        = heads*head_dim;
    const int seq_q    = q_row >= 0 ? 1 : total;
    const bool fuse_kt = wk.kt_ok();
    // K^T leading dimension: the backend's, so producer and consumer agree
    // (identical to (total+7)&~7 on every backend but AMD - see hal/common/layout.h)
    const int skp      = hal::kt_stride(total);

    if (s.q.size()   < (size_t)seq_q*D) s.q.resize  ((size_t)seq_q*D);
    if (s.v.size()   < (size_t)total*D) s.v.resize  ((size_t)total*D);
    if (s.att.size() < (size_t)seq_q*D) s.att.resize((size_t)seq_q*D);

    // grow-only zeroing: the K^T producers leave columns [total, skp) untouched
    if (fuse_kt  && s.kt.size() < (size_t)D*skp  ) s.kt.assign((size_t)D*skp, 0.0f);
    if (!fuse_kt && s.k.size()  < (size_t)total*D) s.k.resize ((size_t)total*D);

    if (prof) prof->tic();
    wq.forward(s.q.data(), q_row >= 0 ? x_norm+(size_t)q_row*D : x_norm, seq_q);
    const float* ktp = nullptr;
    if (fuse_kt) {
        wk.forward_kt(s.kt.data(), x_norm, total, skp);
        ktp = s.kt.data();
    } else {
        wk.forward(s.k.data(), x_norm, total);
    }
    wv.forward(s.v.data(), x_norm, total);
    if (prof) prof->toc(prof->qkv);

    if (prof) prof->tic();
    if (mask) {
        gqa_attention_masked(s.att.data(), s.q.data(), s.k.data(), s.v.data(),
                             seq_q, total, heads, heads, head_dim, scale,
                             mask+(q_row >= 0 ? (size_t)q_row*total : 0), ktp);
    } else {
        // full attention: no [seq, seq] zero array to build, add or stream
        gqa_attention_dense(s.att.data(), s.q.data(), s.k.data(), s.v.data(),
                            seq_q, total, heads, heads, head_dim, scale, ktp);
    }
    if (prof) prof->toc(prof->attn);

    if (prof) prof->tic();
    if (add_out) wo.forward_add(out, s.att.data(), seq_q);
    else         wo.forward(out, s.att.data(), seq_q);
    if (prof) prof->toc(prof->proj);
}

void MhaQKV::set_shape(int heads_, int head_dim_) {
    heads    = heads_;
    head_dim = head_dim_;
    scale    = 1.0f/std::sqrt((float)head_dim_);
}

void MhaQKV::forward(float* out, const float* xq, const float* xk, const float* xv,
                     int seq_q, int seq_kv, Scratch& s, bool add_out,
                     const float* mask) const {
    const int D        = heads*head_dim;
    const bool fuse_kt = wk.kt_ok();
    const int skp      = hal::kt_stride(seq_kv);

    if (s.q.size()    < (size_t)seq_q*D       ) s.q.resize   ((size_t)seq_q*D);
    if (s.v.size()    < (size_t)seq_kv*D      ) s.v.resize   ((size_t)seq_kv*D);
    if (s.att.size()  < (size_t)seq_q*D       ) s.att.resize ((size_t)seq_q*D);

    if (fuse_kt  && s.kt.size() < (size_t)D*skp   ) s.kt.assign((size_t)D*skp, 0.0f);
    if (!fuse_kt && s.k.size()  < (size_t)seq_kv*D) s.k.resize ((size_t)seq_kv*D);

    wq.forward(s.q.data(), xq, seq_q);
    const float* ktp = nullptr;
    if (fuse_kt) {
        wk.forward_kt(s.kt.data(), xk, seq_kv, skp);
        ktp = s.kt.data();
    } else {
        wk.forward(s.k.data(), xk, seq_kv);
    }
    wv.forward(s.v.data(), xv, seq_kv);

    // Unmasked is the common case: take the dense op rather than build, stream
    // and add a [seq_q, seq_kv] zero mask.
    if (mask) {
        gqa_attention_masked(s.att.data(), s.q.data(), s.k.data(), s.v.data(),
                             seq_q, seq_kv, heads, heads, head_dim, scale, mask, ktp);
    } else {
        gqa_attention_dense(s.att.data(), s.q.data(), s.k.data(), s.v.data(),
                            seq_q, seq_kv, heads, heads, head_dim, scale, ktp);
    }

    if (add_out) wo.forward_add(out, s.att.data(), seq_q);
    else         wo.forward(out, s.att.data(), seq_q);
}

} // namespace nn
} // namespace tcpu
