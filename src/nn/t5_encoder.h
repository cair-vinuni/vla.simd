/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <string>
#include <vector>

// T5 encoder (the frozen language tower of Octo and of IMPACT). Pre-RMSNorm
// blocks with relative-position-bias attention (no 1/sqrt(dk) scale - T5 folds
// it into the initialization), ReLU FFN, no linear biases, final RMSNorm.
//
// Every dimension is a config field, so the same code is T5-base (Octo:
// t5.meta/t5.bin from tools/octo/convert_octo.py) and T5-small (IMPACT:
// text.meta/text.bin from tools/impact/convert_impact.py) - hence the
// `stem` argument to load().
//
// `vocab` is whatever the embedding table in the arena has. A checkpoint may
// prune it to the tokens its instruction corpus uses, so the caller must remap
// ids through the checkpoint's vocab_map before calling encode(); an id outside
// `vocab` is reported and zeroed rather than read out of bounds.

namespace tcpu {
namespace nn {

struct T5Config {
    int d_model = 768, n_layers = 12, n_heads = 12, d_kv = 64, d_ff = 3072;
    int vocab = 32128, n_buckets = 32, max_dist = 128, n_tokens = 16;
    float eps = 1e-6f;
};

struct T5LayerW {
    const float *ln1, *wq, *wk, *wv, *wo, *ln2, *wi, *wo2;
};

struct T5Encoder {
    T5Config cfg;
    std::vector<float> data;
    const float* emb = nullptr;   // [vocab, d_model]
    const float* rel = nullptr;   // [n_buckets, n_heads]
    const float* final_ln = nullptr;
    std::vector<T5LayerW> layers;

    // Reads <dir>/<stem>.meta and <dir>/<stem>.bin. Anything the arena holds
    // past the encoder (MicroVLA's text projection) is left for the caller:
    // `tail` receives the offset just past what this encoder consumed.
    bool load(const std::string& dir, const std::string& stem = "t5", size_t* tail = nullptr);

    // ids/attn_mask [seq] -> out [seq, d_model]. attn_mask: 1 = real token, 0 = pad
    // (pad positions still produce outputs; they are just not attended to).
    void encode(const int* ids, const int* attn_mask, int seq, float* out) const;
};

} // namespace nn
} // namespace tcpu
