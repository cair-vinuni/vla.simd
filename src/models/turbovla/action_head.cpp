/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "action_head.h"
#include "models/arena.h"
#include "ops/lm_ops.h"
#include "io/files.h"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
using std::size_t;

namespace tcpu {

bool TurboActionHead::load(const std::string& dir) {
    io::InFile meta(dir + "/head.meta");
    if (!meta) { std::fprintf(stderr, "turbovla: cannot open %s/head.meta\n", dir.c_str()); return false; }
    std::string key; float val;
    while (meta >> key >> val) {
        if      (key == "hidden"      ) cfg.hidden       = (int)val;
        else if (key == "n_layers"    ) cfg.n_layers     = (int)val;
        else if (key == "n_heads"     ) cfg.n_heads      = (int)val;
        else if (key == "ff"          ) cfg.ff           = (int)val;
        else if (key == "chunk"       ) cfg.chunk        = (int)val;
        else if (key == "action_dim"  ) cfg.action_dim   = (int)val;
        else if (key == "state_dim"   ) cfg.state_dim    = (int)val;
        else if (key == "state_tokens") cfg.state_tokens = (int)val;
        else if (key == "state_hidden") cfg.state_hidden = (int)val;
        else if (key == "mlp_hidden"  ) cfg.mlp_hidden   = (int)val;
        else if (key == "mlp_layers"  ) cfg.mlp_layers   = (int)val;
        else if (key == "ln_eps"      ) cfg.ln_eps       = val;
    }

    const bool bad = cfg.hidden < 1 || cfg.n_layers < 1 || cfg.n_heads < 1 || cfg.ff < 1 ||
                     cfg.chunk < 1 || cfg.action_dim < 1 || cfg.state_dim < 1 ||
                     cfg.state_tokens < 1 || cfg.state_hidden < 1 || cfg.mlp_hidden < 1 ||
                     cfg.mlp_layers < 2 || cfg.hidden % cfg.n_heads != 0;
    if (bad) {
        std::fprintf(stderr, "turbovla: %s/head.meta shapes do not close (hidden %d layers %d "
                     "heads %d ff %d chunk %d action %d state %d tokens %d shidden %d "
                     "mlp %d/%d)\n", dir.c_str(), cfg.hidden, cfg.n_layers, cfg.n_heads,
                     cfg.ff, cfg.chunk, cfg.action_dim, cfg.state_dim, cfg.state_tokens,
                     cfg.state_hidden, cfg.mlp_hidden, cfg.mlp_layers);
        return false;
    }

    if (!read_arena(dir + "/head.bin", data)) {
        std::fprintf(stderr, "turbovla: cannot read %s/head.bin\n", dir.c_str());
        return false;
    }
    const size_t D = (size_t)cfg.hidden, S = (size_t)cfg.state_dim;
    const size_t SH = (size_t)cfg.state_hidden, ST = (size_t)cfg.state_tokens;
    const size_t F = (size_t)cfg.ff, MH = (size_t)cfg.mlp_hidden, A = (size_t)cfg.action_dim;
    size_t mlp_floats = (MH*D + MH);                            // first layer
    for (int i = 1; i < cfg.mlp_layers-1; i++) mlp_floats += MH*MH + MH;
    mlp_floats += A*MH + A;                                     // last layer
    const size_t want = 2*S + (SH*S + SH) + (ST*D*SH + ST*D) + ST*D + 2*D   // state projection
                      + (size_t)cfg.chunk*D                                  // action queries
                      + (size_t)cfg.n_layers*(4*(D*D + D) + 2*D              // self-attn + norm1
                                              + 4*(D*D + D) + 2*D            // cross-attn + norm2
                                              + (F*D + F) + (D*F + D) + 2*D) // ff + norm3
                      + mlp_floats;
    if (data.size() != want) {
        std::fprintf(stderr, "turbovla: %s/head.bin has %zu floats, expected %zu\n",
                     dir.c_str(), data.size(), want);
        return false;
    }

    ArenaCursor<float> take{data};
    using Role = nn::Linear::Role;

    state_ln_w = take(S); state_ln_b = take(S);
    const float* s1w = take(SH*S); const float* s1b = take(SH);
    const float* s2w = take(ST*D*SH); const float* s2b = take(ST*D);
    state_pos = take(ST*D);
    state_norm_w = take(D); state_norm_b = take(D);
    sp1.init(s1w, s1b, cfg.state_hidden, cfg.state_dim, Role::Generic);
    sp2.init(s2w, s2b, cfg.state_tokens*cfg.hidden, cfg.state_hidden, Role::Generic);

    queries = take((size_t)cfg.chunk*D);
    layers.resize((size_t)cfg.n_layers);
    for (int L = 0; L < cfg.n_layers; L++) {
        TurboDecoderLayer& l = layers[(size_t)L];
        auto mha = [&](nn::Linear& wq, nn::Linear& wk, nn::Linear& wv, nn::Linear& wo) {
            const float* qw = take(D*D); const float* qb = take(D);
            const float* kw = take(D*D); const float* kb = take(D);
            const float* vw = take(D*D); const float* vb = take(D);
            const float* ow = take(D*D); const float* ob = take(D);
            wq.init(qw, qb, cfg.hidden, cfg.hidden, Role::Gemm);
            wk.init(kw, kb, cfg.hidden, cfg.hidden, Role::Gemm);
            wv.init(vw, vb, cfg.hidden, cfg.hidden, Role::Gemm);
            wo.init(ow, ob, cfg.hidden, cfg.hidden, Role::Gemm);
        };
        mha(l.self.wq, l.self.wk, l.self.wv, l.self.wo);
        l.self.set_shape(cfg.n_heads, cfg.head_dim());
        l.n1_w = take(D); l.n1_b = take(D);
        mha(l.cross.wq, l.cross.wk, l.cross.wv, l.cross.wo);
        l.cross.set_shape(cfg.n_heads, cfg.head_dim());
        l.n2_w = take(D); l.n2_b = take(D);
        const float* f1w = take(F*D); const float* f1b = take(F);
        const float* f2w = take(D*F); const float* f2b = take(D);
        l.ff1.init(f1w, f1b, cfg.ff, cfg.hidden, Role::Mlp);
        l.ff2.init(f2w, f2b, cfg.hidden, cfg.ff, Role::Mlp);
        l.n3_w = take(D); l.n3_b = take(D);
    }

    mlp.resize((size_t)cfg.mlp_layers);
    for (int i = 0; i < cfg.mlp_layers; i++) {
        const int in  = i == 0 ? cfg.hidden : cfg.mlp_hidden;
        const int out = i == cfg.mlp_layers-1 ? cfg.action_dim : cfg.mlp_hidden;
        const float* w = take((size_t)out*in);
        const float* b = take((size_t)out);
        mlp[(size_t)i].init(w, b, out, in, Role::Generic);
    }
    return true;
}

void TurboActionHead::state_tokens(const float* state_norm, float* tokens) const {
    const int D = cfg.hidden, ST = cfg.state_tokens;
    std::vector<float> s((size_t)cfg.state_dim), mid((size_t)cfg.state_hidden);
    layernorm(s.data(), state_norm, state_ln_w, state_ln_b, 1, cfg.state_dim, cfg.ln_eps);
    sp1.forward(mid.data(), s.data(), 1);
    gelu_erf(mid.data(), cfg.state_hidden);
    sp2.forward(tokens, mid.data(), 1);
    for (size_t i = 0; i < (size_t)ST*D; i++) tokens[i] += state_pos[i];
    layernorm(tokens, tokens, state_norm_w, state_norm_b, ST, D, cfg.ln_eps);
}

void TurboActionHead::decode(const float* memory, int n_mem, float* actions_norm) const {
    const int D = cfg.hidden, C = cfg.chunk, F = cfg.ff;
    const size_t CD = (size_t)C*D;
    if (x.size()   < CD) x.resize(CD);
    if (h.size()   < CD) h.resize(CD);
    if (res.size() < CD) res.resize(CD);
    if (ff.size()  < (size_t)C*F) ff.resize((size_t)C*F);

    std::memcpy(x.data(), queries, CD*sizeof(float));
    for (int L = 0; L < cfg.n_layers; L++) {
        const TurboDecoderLayer& l = layers[(size_t)L];

        layernorm(h.data(), x.data(), l.n1_w, l.n1_b, C, D, cfg.ln_eps);
        l.self.forward(res.data(), h.data(), C, nullptr, -1, sc, nullptr);
        for (size_t i = 0; i < CD; i++) x[i] += res[i];

        layernorm(h.data(), x.data(), l.n2_w, l.n2_b, C, D, cfg.ln_eps);
        l.cross.forward(res.data(), h.data(), memory, memory, C, n_mem, sc);
        for (size_t i = 0; i < CD; i++) x[i] += res[i];

        layernorm(h.data(), x.data(), l.n3_w, l.n3_b, C, D, cfg.ln_eps);
        l.ff1.forward(ff.data(), h.data(), C);
        relu(ff.data(), C*F);
        l.ff2.forward(res.data(), ff.data(), C);
        for (size_t i = 0; i < CD; i++) x[i] += res[i];
    }
    // nn.TransformerDecoder(norm=None): no final LayerNorm.

    // Ping-pong buffers: nn::Linear::forward must not read and write the same
    // memory, and the hidden layers of this MLP are all mlp_hidden wide.
    std::vector<float> buf0((size_t)C*cfg.mlp_hidden), buf1((size_t)C*cfg.mlp_hidden);
    const float* in = x.data();
    for (int i = 0; i < cfg.mlp_layers; i++) {
        const bool last = i == cfg.mlp_layers-1;
        float* out = last ? actions_norm : (i % 2 == 0 ? buf0.data() : buf1.data());
        mlp[(size_t)i].forward(out, in, C);
        if (!last) {
            relu(out, C*cfg.mlp_hidden);
            in = out;
        }
    }
    for (int i = 0; i < C*cfg.action_dim; i++) actions_norm[i] = std::tanh(actions_norm[i]);
}

} // namespace tcpu
