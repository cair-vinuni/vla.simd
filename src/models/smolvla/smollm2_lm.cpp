/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "smollm2_lm.h"
#include "ops/lm_ops.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace tcpu {

bool SmollmVlm::load(const std::string& dir) {
    std::ifstream meta(dir + "/vlm.meta");
    if (!meta) { std::fprintf(stderr, "smolvla: cannot open %s/vlm.meta\n", dir.c_str()); return false; }
    std::string key; float val;
    while (meta >> key >> val) {
        if      (key == "hidden"   ) cfg.hidden = (int)val;
        else if (key == "n_q"      ) cfg.n_q = (int)val;
        else if (key == "n_kv"     ) cfg.n_kv = (int)val;
        else if (key == "head_dim" ) cfg.head_dim = (int)val;
        else if (key == "ffn"      ) cfg.ffn = (int)val;
        else if (key == "eps"      ) cfg.rms_eps = val;
        else if (key == "rope_base") cfg.rope_base = val;
        else if (key == "n_layers" ) cfg.n_layers = (int)val;
    }

    const int H = cfg.hidden, QF = cfg.q_full(), KV = cfg.kv_full(), F = cfg.ffn, NL = cfg.n_layers;
    const size_t fcount = (size_t)2*H*NL + (size_t)H;   // per-layer ln_in+ln_post, then out_norm
    const size_t per_layer_w = (size_t)QF*H + (size_t)KV*H*2 + (size_t)H*QF
                             + (size_t)F*H*2 + (size_t)H*F;
    const size_t wcount = per_layer_w*NL;

    std::ifstream bin(dir + "/vlm.bin", std::ios::binary);
    if (!bin) { std::fprintf(stderr, "smolvla: cannot open %s/vlm.bin\n", dir.c_str()); return false; }
    fnorms.resize(fcount);
    bin.read(reinterpret_cast<char*>(fnorms.data()), fcount*sizeof(float));
    wbf.resize(wcount);
    bin.read(reinterpret_cast<char*>(wbf.data()), wcount*sizeof(uint16_t));
    if (!bin) { std::fprintf(stderr, "smolvla: short read on vlm.bin\n"); return false; }

    layers.resize(NL);
    size_t fo = 0, wo = 0;
    auto tf = [&](size_t n) { const float* p = fnorms.data()+fo; fo += n; return p; };
    auto tw = [&](size_t n) { const uint16_t* p = wbf.data()+wo; wo += n; return p; };
    for (int L = 0; L < NL; L++) {
        layers[L].ln_in   = tf(H);
        layers[L].ln_post = tf(H);
    }
    out_norm = tf(H);

    using Role = nn::Linear::Role;
    for (int L = 0; L < NL; L++) {
        VlmLayerW& w = layers[L];
        w.q   .init_bf16(tw((size_t)QF*H), nullptr, QF, H, Role::Gemm);
        w.k   .init_bf16(tw((size_t)KV*H), nullptr, KV, H, Role::Gemm);
        w.v   .init_bf16(tw((size_t)KV*H), nullptr, KV, H, Role::Gemm);
        w.o   .init_bf16(tw((size_t)H*QF), nullptr, H, QF, Role::Gemm);
        w.gate.init_bf16(tw((size_t)F*H),  nullptr, F, H,  Role::Mlp);
        w.up  .init_bf16(tw((size_t)F*H),  nullptr, F, H,  Role::Mlp);
        w.down.init_bf16(tw((size_t)H*F),  nullptr, H, F,  Role::Mlp);
    }

    if (!nn::Linear::bf16_keeps_raw()) {
        wbf.clear();
        wbf.shrink_to_fit();
    }
    return true;
}

void SmollmVlm::prefix_forward(const float* embs, const float* mask, const int* pos, int seq,
                               float* out, std::vector<VlmKV>& kv_out) const {
    const int H = cfg.hidden, QF = cfg.q_full(), KV = cfg.kv_full(), F = cfg.ffn;
    const float scale = 1.0f/std::sqrt((float)cfg.head_dim);

    std::vector<float> h(embs, embs + (size_t)seq*H);
    std::vector<float> xn((size_t)seq*H), q((size_t)seq*QF), attn((size_t)seq*QF);
    std::vector<float> o((size_t)seq*H), h2((size_t)seq*H);
    std::vector<float> g((size_t)seq*F), u((size_t)seq*F), gu((size_t)seq*F), dn((size_t)seq*H);

    kv_out.resize(cfg.n_layers);

    for (int L = 0; L < cfg.n_layers; L++) {
        const VlmLayerW& w = layers[L];
        VlmKV& kv = kv_out[L];
        kv.k.resize((size_t)seq*KV);
        kv.v.resize((size_t)seq*KV);

        // attention
        rmsnorm(xn.data(), h.data(), w.ln_in, seq, H, cfg.rms_eps);
        w.q.forward(q.data(),    xn.data(), seq);
        w.k.forward(kv.k.data(), xn.data(), seq);
        w.v.forward(kv.v.data(), xn.data(), seq);
        rope_neox(q.data(),    pos, seq, cfg.n_q,  cfg.head_dim, cfg.rope_base);
        rope_neox(kv.k.data(), pos, seq, cfg.n_kv, cfg.head_dim, cfg.rope_base);
        gqa_attention_masked(attn.data(), q.data(), kv.k.data(), kv.v.data(),
                             seq, seq, cfg.n_q, cfg.n_kv, cfg.head_dim, scale, mask);
        w.o.forward(o.data(), attn.data(), seq);
        for (size_t i = 0; i < (size_t)seq*H; i++) h[i] += o[i];

        // SwiGLU MLP
        rmsnorm(h2.data(), h.data(), w.ln_post, seq, H, cfg.rms_eps);
        w.gate.forward(g.data(), h2.data(), seq);
        w.up  .forward(u.data(), h2.data(), seq);
        silu_gate(gu.data(), g.data(), u.data(), seq*F);
        w.down.forward(dn.data(), gu.data(), seq);
        for (size_t i = 0; i < (size_t)seq*H; i++) h[i] += dn[i];
    }

    rmsnorm(out, h.data(), out_norm, seq, H, cfg.rms_eps);
}

} // namespace tcpu
