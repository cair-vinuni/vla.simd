/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "t5_text.h"
#include "models/arena.h"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
using std::size_t;

namespace tcpu {

bool ImpactText::load(const std::string& dir, int vocab_full, int unk_id) {
    size_t enc_end = 0;
    if (!t5.load(dir, "text", &enc_end)) {
        std::fprintf(stderr, "impact: cannot load %s/text.{meta,bin}\n", dir.c_str());
        return false;
    }

    std::ifstream meta(dir + "/text.meta");
    std::string key; double val;
    while (meta >> key >> val) {
        if      (key == "proj_dim"      ) cfg.proj_dim  = (int)val;
        else if (key == "n_text"        ) cfg.n_text    = (int)val;
        else if (key == "n_film"        ) cfg.n_film    = (int)val;
        else if (key == "film_total"    ) cfg.film_total= (int)val;
        else if (key == "encoder_floats") cfg.encoder_floats = (long)val;
    }
    if (cfg.encoder_floats > 0 && (size_t)cfg.encoder_floats != enc_end) {
        std::fprintf(stderr, "impact: text.bin encoder ends at %zu floats, meta says %ld\n",
                     enc_end, cfg.encoder_floats);
        return false;
    }
    if (cfg.proj_dim < 1 || cfg.n_text < 1 || cfg.film_total < 1) {
        std::fprintf(stderr, "impact: text.meta is missing proj_dim / n_text / film_total\n");
        return false;
    }

    // Tail layout, in this order: projection, text positions, FiLM head.
    const size_t D = (size_t)t5.cfg.d_model;
    const size_t P = (size_t)cfg.proj_dim;
    const size_t T = (size_t)cfg.n_text;
    const size_t F = (size_t)cfg.film_total;
    const size_t want = enc_end + (P*D + P) + T*P + (2*F*D + 2*F);
    if (t5.data.size() != want) {
        std::fprintf(stderr, "impact: %s/text.bin has %zu floats, expected %zu "
                     "(encoder %zu + proj %zu + text_pos %zu + film %zu)\n",
                     dir.c_str(), t5.data.size(), want, enc_end,
                     P*D+P, T*P, 2*F*D+2*F);
        return false;
    }

    const float* p = t5.data.data() + enc_end;
    proj.init(p, p + P*D, cfg.proj_dim, t5.cfg.d_model, nn::Linear::Role::Generic);
    p += P*D + P;
    text_pos = p;
    p += T*P;
    film.init(p, p + 2*F*D, 2*(int)F, t5.cfg.d_model, nn::Linear::Role::Generic);

    // vocab_map.bin: int32 per full-vocabulary id. Its length is the tell that
    // the tables belong together, so it is checked rather than trusted.
    std::ifstream vm(dir + "/vocab_map.bin", std::ios::binary);
    if (!vm) { std::fprintf(stderr, "impact: cannot open %s/vocab_map.bin\n", dir.c_str()); return false; }
    vm.seekg(0, std::ios::end);
    const std::streamoff bytes = vm.tellg();
    vm.seekg(0);
    if (bytes != (std::streamoff)sizeof(int32_t)*vocab_full) {
        std::fprintf(stderr, "impact: %s/vocab_map.bin is %lld bytes, expected %zu\n",
                     dir.c_str(), (long long)bytes, sizeof(int32_t)*(size_t)vocab_full);
        return false;
    }
    vocab_map.resize((size_t)vocab_full);
    vm.read((char*)vocab_map.data(), bytes);
    if (!vm) { std::fprintf(stderr, "impact: short read on %s/vocab_map.bin\n", dir.c_str()); return false; }

    if (unk_id < 0 || unk_id >= vocab_full || vocab_map[(size_t)unk_id] < 0) {
        std::fprintf(stderr, "impact: <unk> (id %d) is not in the pruned vocabulary; "
                     "out-of-vocabulary words would have nowhere to go\n", unk_id);
        return false;
    }
    unk_compact = vocab_map[(size_t)unk_id];
    return true;
}

int ImpactText::remap(const int* ids, int seq, int* compact) const {
    int oov = 0;
    for (int t = 0; t < seq; t++) {
        const int id = ids[t];
        const int c = (id >= 0 && id < (int)vocab_map.size()) ? vocab_map[(size_t)id] : -1;
        if (c < 0) { compact[t] = unk_compact; oov++; }
        else       { compact[t] = c; }
    }
    if (oov && !warned_oov) {
        warned_oov = true;
        std::fprintf(stderr,
            "impact: instruction contains %d token(s) outside the pruned vocabulary; "
            "substituting <unk>. The language pathway is blind to those words - this "
            "checkpoint's %zu-token table covers its training corpus and nothing else.\n",
            oov, (size_t)t5.cfg.vocab);
    }
    return oov;
}

void ImpactText::encode(const int* compact, const int* attn_mask, int seq,
                        float* tokens, float* gamma, float* beta, float* hidden) const {
    const size_t D = (size_t)t5.cfg.d_model;
    const size_t F = (size_t)cfg.film_total;

    if (h.size() < (size_t)seq*D) h.resize((size_t)seq*D);
    t5.encode(compact, attn_mask, seq, h.data());
    if (hidden) std::copy(h.begin(), h.begin() + (size_t)seq*D, hidden);

    // Text tokens: the projection only. The learned text position lives in the
    // transformer's pos array, not in the token, because ACT's DETR blocks add
    // the position to attention queries and keys and never to the values - the
    // visual tokens obey that and so must these.
    proj.forward(tokens, h.data(), seq);

    // FiLM reads a mask-aware mean-pool: a padded position holds a real T5 output
    // and would otherwise drag gamma/beta toward whatever the pad rows encode.
    if (pooled.size() < D) pooled.resize(D);
    std::fill(pooled.begin(), pooled.begin()+D, 0.0f);
    int live = 0;
    for (int t = 0; t < seq; t++) {
        if (attn_mask && !attn_mask[t]) continue;
        const float* row = h.data() + (size_t)t*D;
        for (size_t i = 0; i < D; i++) pooled[i] += row[i];
        live++;
    }
    if (live > 0) {
        const float inv = 1.0f/(float)live;
        for (size_t i = 0; i < D; i++) pooled[i] *= inv;
    }

    if (gb.size() < 2*F) gb.resize(2*F);
    film.forward(gb.data(), pooled.data(), 1);
    std::memcpy(gamma, gb.data(),   sizeof(float)*F);
    std::memcpy(beta,  gb.data()+F, sizeof(float)*F);
}

} // namespace tcpu
