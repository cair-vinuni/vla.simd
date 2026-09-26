/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "t5_encoder.h"
#include "models/arena.h"
#include "ops/lm_ops.h"
#include "io/files.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>

namespace tcpu {
namespace nn {

bool T5Encoder::load(const std::string& dir, const std::string& stem, size_t* tail) {
    io::InFile meta(dir + "/" + stem + ".meta");
    if (!meta) return false;
    std::string key;
    double val;
    while (meta >> key >> val) {
        if      (key == "d_model"  ) cfg.d_model = (int)val;
        else if (key == "n_layers" ) cfg.n_layers = (int)val;
        else if (key == "n_heads"  ) cfg.n_heads = (int)val;
        else if (key == "d_kv"     ) cfg.d_kv = (int)val;
        else if (key == "d_ff"     ) cfg.d_ff = (int)val;
        else if (key == "vocab"    ) cfg.vocab = (int)val;
        else if (key == "n_buckets") cfg.n_buckets = (int)val;
        else if (key == "max_dist" ) cfg.max_dist = (int)val;
        else if (key == "eps"      ) cfg.eps = (float)val;
    }
    if (cfg.n_heads < 1 || cfg.d_kv < 1 || (long long)cfg.n_heads*cfg.d_kv != cfg.d_model ||
        cfg.n_buckets < 4 || cfg.max_dist <= cfg.n_buckets/4) return false;

    if (!read_arena(dir + "/" + stem + ".bin", data)) return false;

    const int D  = cfg.d_model;
    const int FF = cfg.d_ff;
    ArenaCursor<float> take{data};
    emb = take((size_t)cfg.vocab*D);
    rel = take((size_t)cfg.n_buckets*cfg.n_heads);

    layers.resize(cfg.n_layers);
    for (auto& L : layers) {
        L.ln1 = take(D);
        L.wq  = take((size_t)D*D);
        L.wk  = take((size_t)D*D);
        L.wv  = take((size_t)D*D);
        L.wo  = take((size_t)D*D);
        L.ln2 = take(D);
        L.wi  = take((size_t)FF*D);
        L.wo2 = take((size_t)D*FF);
    }
    final_ln = take(D);
    // Only the "nothing left over" case is a shape error, and only when the
    // caller did not ask for the remainder.
    if (tail) *tail = take.off;
    return tail ? take.ok : take.done();
}

// HF T5 _relative_position_bucket, bidirectional. rp = key_pos - query_pos.
static int rel_bucket(int rp, int num_buckets, int max_dist) {
    const int nb = num_buckets/2;
    int ret = rp > 0 ? nb : 0;
    int arp = rp < 0 ? -rp : rp;
    const int max_exact = nb/2;
    if (arp < max_exact) return ret+arp;

    int large = max_exact+(int)(std::log((double)arp/max_exact) /
                                std::log((double)max_dist/max_exact) * (nb-max_exact));
    if (large > nb-1) large = nb-1;
    return ret+large;
}

void T5Encoder::encode(const int* ids, const int* attn_mask, int seq, float* out) const {
    const int D  = cfg.d_model;
    const int NH = cfg.n_heads;
    const int HD = cfg.d_kv;
    const int FF = cfg.d_ff;
    const float NEG = std::numeric_limits<float>::lowest();

    // per-head additive bias: relative position bias + key-side pad mask
    std::vector<float> bias((size_t)NH*seq*seq);
    for (int h=0; h<NH; h++)
        for (int i=0; i<seq; i++)
            for (int j=0; j<seq; j++)
                bias[((size_t)h*seq+i)*seq+j] =
                    attn_mask[j] ? rel[(size_t)rel_bucket(j-i, cfg.n_buckets, cfg.max_dist)*NH+h] : NEG;

    std::vector<float> x   ((size_t)seq*D);
    std::vector<float> h   ((size_t)seq*D);
    std::vector<float> q   ((size_t)seq*D);
    std::vector<float> k   ((size_t)seq*D);
    std::vector<float> v   ((size_t)seq*D);
    std::vector<float> att ((size_t)seq*D);
    std::vector<float> ff  ((size_t)seq*FF);
    // an id past this vocab reads other weights and returns wrong features
    for (int t=0; t<seq; t++) {
        const int id = ids[t];
        if (id < 0 || id >= cfg.vocab) {
            std::fprintf(stderr, "t5: token id %d at %d outside vocab %d\n", id, t, cfg.vocab);
            std::memset(x.data()+(size_t)t*D, 0, sizeof(float)*D);
            continue;
        }
        std::memcpy(x.data()+(size_t)t*D, emb+(size_t)id*D, sizeof(float)*D);
    }

    for (const auto& L : layers) {
        rmsnorm(h.data(), x.data(), L.ln1, seq, D, cfg.eps);
        dense_linear(q.data(), h.data(), L.wq, nullptr, seq, D, D);
        dense_linear(k.data(), h.data(), L.wk, nullptr, seq, D, D);
        dense_linear(v.data(), h.data(), L.wv, nullptr, seq, D, D);
        mha_attention_bias(att.data(), q.data(), k.data(), v.data(), seq, seq, NH, HD, 1.0f, bias.data());
        dense_linear(h.data(), att.data(), L.wo, nullptr, seq, D, D);
        for (size_t i=0; i<x.size(); i++)
            x[i] += h[i];

        rmsnorm(h.data(), x.data(), L.ln2, seq, D, cfg.eps);
        dense_linear(ff.data(), h.data(), L.wi, nullptr, seq, FF, D);
        relu(ff.data(), seq*FF);
        dense_linear(h.data(), ff.data(), L.wo2, nullptr, seq, D, FF);
        for (size_t i=0; i<x.size(); i++)
            x[i] += h[i];
    }
    rmsnorm(out, x.data(), final_ln, seq, D, cfg.eps);
}

} // namespace nn
} // namespace tcpu
