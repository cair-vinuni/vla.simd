/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bert_text.h"
#include "models/arena.h"
#include "ops/lm_ops.h"
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
using std::size_t;

namespace tcpu {

bool BertText::load(const std::string& dir, int fusion_hidden) {
    std::ifstream meta(dir + "/text.meta");
    if (!meta) { std::fprintf(stderr, "turbovla: cannot open %s/text.meta\n", dir.c_str()); return false; }
    std::string key; float val;
    while (meta >> key >> val) {
        if      (key == "hidden"    ) cfg.hidden     = (int)val;
        else if (key == "n_heads"   ) cfg.n_heads    = (int)val;
        else if (key == "head_dim"  ) cfg.head_dim   = (int)val;
        else if (key == "inter"     ) cfg.inter      = (int)val;
        else if (key == "n_layers"  ) cfg.n_layers   = (int)val;
        else if (key == "vocab"     ) cfg.vocab      = (int)val;
        else if (key == "max_pos"   ) cfg.max_pos    = (int)val;
        else if (key == "type_vocab") cfg.type_vocab = (int)val;
        else if (key == "ln_eps"    ) cfg.ln_eps     = val;
    }
    out_dim = fusion_hidden;

    const bool bad = cfg.hidden < 1 || cfg.n_heads < 1 || cfg.head_dim < 1 || cfg.inter < 1 ||
                     cfg.n_layers < 1 || cfg.vocab < 1 || cfg.max_pos < 1 ||
                     cfg.type_vocab < 1 || out_dim < 1 ||
                     cfg.n_heads*cfg.head_dim != cfg.hidden;
    if (bad) {
        std::fprintf(stderr, "turbovla: %s/text.meta shapes do not close (hidden %d heads %d "
                     "head_dim %d inter %d layers %d vocab %d max_pos %d types %d out %d)\n",
                     dir.c_str(), cfg.hidden, cfg.n_heads, cfg.head_dim, cfg.inter,
                     cfg.n_layers, cfg.vocab, cfg.max_pos, cfg.type_vocab, out_dim);
        return false;
    }

    if (!read_arena(dir + "/text.bin", data)) {
        std::fprintf(stderr, "turbovla: cannot read %s/text.bin\n", dir.c_str());
        return false;
    }
    const size_t H = (size_t)cfg.hidden, I = (size_t)cfg.inter;
    const size_t want = (size_t)cfg.vocab*H + (size_t)cfg.max_pos*H
                      + (size_t)cfg.type_vocab*H + 2*H
                      + (size_t)cfg.n_layers*(4*(H*H + H)   // q, k, v, attn out dense
                                              + 2*H         // attention LayerNorm
                                              + I*H + I     // intermediate
                                              + H*I + H     // output dense
                                              + 2*H)        // output LayerNorm
                      + (size_t)out_dim*H + (size_t)out_dim;
    if (data.size() != want) {
        std::fprintf(stderr, "turbovla: %s/text.bin has %zu floats, expected %zu\n",
                     dir.c_str(), data.size(), want);
        return false;
    }

    ArenaCursor<float> take{data};
    using Role = nn::Linear::Role;

    word_emb = take((size_t)cfg.vocab*H);
    pos_emb  = take((size_t)cfg.max_pos*H);
    type_emb = take((size_t)cfg.type_vocab*H);
    emb_ln_w = take(H); emb_ln_b = take(H);

    layers.resize((size_t)cfg.n_layers);
    for (int L = 0; L < cfg.n_layers; L++) {
        BertLayer& l = layers[(size_t)L];
        const float* qw = take(H*H); const float* qb = take(H);
        const float* kw = take(H*H); const float* kb = take(H);
        const float* vw = take(H*H); const float* vb = take(H);
        const float* ow = take(H*H); const float* ob = take(H);
        l.ln1_w = take(H); l.ln1_b = take(H);
        const float* uw = take(I*H); const float* ub = take(I);
        const float* dw = take(H*I); const float* db = take(H);
        l.ln2_w = take(H); l.ln2_b = take(H);
        l.attn.wq.init(qw, qb, cfg.hidden, cfg.hidden, Role::Gemm);
        l.attn.wk.init(kw, kb, cfg.hidden, cfg.hidden, Role::Gemm);
        l.attn.wv.init(vw, vb, cfg.hidden, cfg.hidden, Role::Gemm);
        l.attn.wo.init(ow, ob, cfg.hidden, cfg.hidden, Role::Gemm);
        l.attn.set_shape(cfg.n_heads, cfg.head_dim);
        l.up.init(uw, ub, cfg.inter, cfg.hidden, Role::Mlp);
        l.down.init(dw, db, cfg.hidden, cfg.inter, Role::Mlp);
    }

    const float* tw = take((size_t)out_dim*H);
    const float* tb = take((size_t)out_dim);
    text_proj.init(tw, tb, out_dim, cfg.hidden, Role::Gemm);
    return true;
}

void BertText::build_masks(const int* ids, int n, const TurboVlaConfig& tc,
                           int* position_ids, float* mask) const {
    const float blocked = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; i++) {
        position_ids[i] = 0;
        for (int j = 0; j < n; j++) mask[(size_t)i*n + j] = (i == j) ? 0.0f : blocked;
    }
    // Walk the special tokens left to right, opening one all-to-all block per
    // span. A special token in column 0 or n-1 only sets its own diagonal, which
    // is why a sequence whose [SEP] lands in the last column stays on the
    // identity - reproduced, not corrected.
    int previous = 0;
    for (int col = 0; col < n; col++) {
        const int id = ids[col];
        if (id != tc.cls_id && id != tc.sep_id && id != tc.dot_id && id != tc.question_id)
            continue;
        if (col == 0 || col == n-1) {
            mask[(size_t)col*n + col] = 0.0f;
            position_ids[col] = 0;
        } else {
            for (int i = previous+1; i <= col; i++) {
                for (int j = previous+1; j <= col; j++) mask[(size_t)i*n + j] = 0.0f;
                position_ids[i] = i - previous - 1;
            }
        }
        previous = col;
    }
}

void BertText::encode(const int* ids, const int* position_ids, const float* mask, int n,
                      float* hidden) const {
    const int H = cfg.hidden, I = cfg.inter;
    const size_t NH = (size_t)n*H;
    if (x.size()  < NH) x.resize(NH);
    if (h.size()  < NH) h.resize(NH);
    if (ff.size() < (size_t)n*I) ff.resize((size_t)n*I);

    // word + token_type(0) + position, then LayerNorm. Ids are clamped rather
    // than trusted: a caller-supplied id past the vocabulary would read weights
    // out of the arena.
    for (int t = 0; t < n; t++) {
        const int id = ids[t] >= 0 && ids[t] < cfg.vocab ? ids[t] : 0;
        const int p  = position_ids[t] >= 0 && position_ids[t] < cfg.max_pos ? position_ids[t] : 0;
        const float* w = word_emb + (size_t)id*H;
        const float* pe = pos_emb + (size_t)p*H;
        float* dst = x.data() + (size_t)t*H;
        for (int i = 0; i < H; i++) dst[i] = w[i] + type_emb[i] + pe[i];
    }
    layernorm(x.data(), x.data(), emb_ln_w, emb_ln_b, n, H, cfg.ln_eps);

    for (int L = 0; L < cfg.n_layers; L++) {
        const BertLayer& l = layers[(size_t)L];
        l.attn.forward(h.data(), x.data(), n, mask, -1, sc, nullptr);
        for (size_t i = 0; i < NH; i++) h[i] += x[i];
        layernorm(x.data(), h.data(), l.ln1_w, l.ln1_b, n, H, cfg.ln_eps);

        l.up.forward(ff.data(), x.data(), n);
        gelu_erf(ff.data(), n*I);
        l.down.forward(h.data(), ff.data(), n);
        for (size_t i = 0; i < NH; i++) h[i] += x[i];
        layernorm(x.data(), h.data(), l.ln2_w, l.ln2_b, n, H, cfg.ln_eps);
    }
    std::memcpy(hidden, x.data(), NH*sizeof(float));
}

void BertText::project(const float* hidden, int n, float* tokens) const {
    text_proj.forward(tokens, hidden, n);
}

} // namespace tcpu
