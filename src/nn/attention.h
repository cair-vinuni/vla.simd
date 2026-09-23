/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "linear.h"
#include <vector>

namespace tcpu {
namespace nn {

// Reusable activation scratch, owned by the model (persistent across layers and
// calls, like the branches' buffers). Every consumed region is fully written
// before it is read, so reuse never changes values. kt is grow-only zeroed:
// the K^T producers leave the padding columns untouched.
struct Scratch {
    std::vector<float> h, q, k, v, att, ff, kt;
};

// Per-op time buckets (the OCTO_PROFILE_TF breakdown). Layers run serially and
// the parallelism is inside each op, so plain accumulators are race-free.
struct Prof;

// Masked multi-head self-attention with per-backend K^T plumbing. Computes
// q/k/v from x_norm, runs the HAL masked-attention engine, and writes the
// output projection. When the backend produces K^T natively from the K
// projection (Linear::kt_native), the transpose pass disappears; otherwise the
// engine gets plain K and handles layout itself.
struct MhaMasked {
    Linear wq, wk, wv, wo;   // init with Role::Gemm
    int heads = 0, head_dim = 0;
    float scale = 0.0f;

    void set_shape(int heads, int head_dim);

    // x_norm [total, D] -> out [seq_q, D] (attention + output projection).
    // mask [total, total] additive rows. q_row >= 0 computes only that query
    // row (readout path, seq_q = 1). out may alias x_norm: x_norm is consumed
    // (q/k/v) before out is written. prof may be null. add_out accumulates the
    // projection into out (residual fused into the GEMM epilogue; call only
    // when Linear::add_native()).
    void forward(float* out, const float* x_norm, int total, const float* mask,
                 int q_row, Scratch& s, Prof* prof, bool add_out = false) const;
};

// Multi-head attention over three independent sources, no mask. DETR-style
// blocks add the positional embedding to the query and key inputs but not to
// the value, and their cross-attention reads keys/values from another module's
// output, so q/k/v cannot come from one buffer like MhaMasked does.
struct MhaQKV {
    Linear wq, wk, wv, wo;   // init with Role::Gemm
    int heads = 0, head_dim = 0;
    float scale = 0.0f;

    void set_shape(int heads, int head_dim);

    // xq [seq_q, D], xk / xv [seq_kv, D] -> out [seq_q, D] (attention + output
    // projection). out may alias xv (consumed before the projection is written),
    // not xq/xk. add_out accumulates the projection into out (residual fused into
    // the GEMM epilogue; call only when Linear::add_native()). mask, when given,
    // is [seq_q, seq_kv] additive rows (0 = keep) - what a key-padding mask on a
    // cross-attention looks like once broadcast; null takes the dense path.
    void forward(float* out, const float* xq, const float* xk, const float* xv,
                 int seq_q, int seq_kv, Scratch& s, bool add_out = false,
                 const float* mask = nullptr) const;
};

} // namespace nn
} // namespace tcpu
