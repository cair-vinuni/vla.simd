/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../arena.h"
#include "octo_transformer.h"
#include "ops/lm_ops.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>

namespace tcpu {

bool OctoTransformer::load(const std::string& dir) {
    std::ifstream meta(dir + "/octo.meta");
    if (!meta) return false;
    std::string key;
    double val;
    while (meta >> key >> val) {
        if      (key == "d"          ) cfg.d = (int)val;
        else if (key == "n_layers"   ) cfg.n_layers = (int)val;
        else if (key == "heads"      ) cfg.heads = (int)val;
        else if (key == "head_dim"   ) cfg.head_dim = (int)val;
        else if (key == "mlp"        ) cfg.mlp = (int)val;
        else if (key == "max_horizon") cfg.max_horizon = (int)val;
        else if (key == "n_task"     ) cfg.n_task = (int)val;
        else if (key == "tok_primary") cfg.tok_primary = (int)val;
        else if (key == "tok_wrist"  ) cfg.tok_wrist = (int)val;
        else if (key == "n_readout"  ) cfg.n_readout = (int)val;
        else if (key == "t5_dim"     ) cfg.t5_dim = (int)val;
        else if (key == "stem_dim"   ) cfg.stem_dim = (int)val;
        else if (key == "ln_eps"     ) cfg.ln_eps = (float)val;
        else if (key == "gelu_erf"   ) cfg.gelu_erf = val != 0;
    }

    if (!read_arena(dir + "/octo.bin", data)) return false;

    const int D = cfg.d;
    // Shapes come from the .meta, the buffer size from the .bin. Linear::init
    // packs immediately, so a short bin has to be caught before the init rather
    // than by the off == data.size() check at the end of the walk.
    size_t off = 0;
    bool ok = true;
    auto take = [&](size_t n) -> const float* {
        if (n > data.size() - off) { ok = false; return nullptr; }
        const float* p = data.data()+off;
        off += n;
        return p;
    };
    using Role = nn::Linear::Role;
    auto take_linear = [&](nn::Linear& lin, int N, int K, Role role) {
        const float* w = take((size_t)N*K);
        const float* b = take(N);
        if (!ok) return;
        lin.init(w, b, N, K, role);
    };

    take_linear(proj_task,  D, cfg.t5_dim,   Role::Gemm);
    take_linear(proj_prim,  D, cfg.stem_dim, Role::Gemm);
    take_linear(proj_wrist, D, cfg.stem_dim, Role::Gemm);

    pos_task    = take((size_t)cfg.n_task*D);
    pos_prim    = take((size_t)cfg.max_horizon*cfg.tok_primary*D);
    pos_wrist   = take((size_t)cfg.max_horizon*cfg.tok_wrist  *D);
    pos_readout = take((size_t)cfg.max_horizon*cfg.n_readout  *D);

    layers.resize(cfg.n_layers);
    for (auto& L : layers) {
        L.d      = D;
        L.mlp    = cfg.mlp;
        L.ln_eps = cfg.ln_eps;
        L.gelu = cfg.gelu_erf ? nn::EncoderLayer::Gelu::Erf
                              : nn::EncoderLayer::Gelu::Tanh;
        L.attn.set_shape(cfg.heads, cfg.head_dim);

        L.ln1_s = take(D);
        L.ln1_b = take(D);

        take_linear(L.attn.wq, D, D, Role::Gemm);
        take_linear(L.attn.wk, D, D, Role::Gemm);
        take_linear(L.attn.wv, D, D, Role::Gemm);
        take_linear(L.attn.wo, D, D, Role::Gemm);

        L.ln2_s = take(D);
        L.ln2_b = take(D);

        take_linear(L.w1, cfg.mlp, D, Role::Mlp);
        take_linear(L.w2, D, cfg.mlp, Role::Mlp);
        if (!ok) return false;
    }

    final_s = take(D);
    final_b = take(D);
    return ok && off == data.size();
}

// groups: 0 task_language (prefix, timestep -1), 1 obs_primary, 2 obs_wrist,
// 3 obs_language (repeated task), 4 readout_action
static bool attends(int gi, int ti, int gj, int tj) {
    if (gi == 0) return gj == 0;                       // task -> task only
    if (gj == 0) return true;                          // anyone -> task (tj=-1 <= ti)
    if (gj <= 3) return tj <= ti;                      // -> obs_* causal
    return gi == 4 && gj == 4 && tj <= ti;             // readout -> own readout only
}

void OctoTransformer::build_mask(int wnd, const uint8_t* timestep_mask, bool wrist, uint8_t* keep) const {
    const int per   = tokens_per_step();
    const int total = total_tokens(wnd);

    std::vector<int> grp(total);
    std::vector<int> ts(total);
    std::vector<uint8_t> pad(total);
    for (int i=0; i<cfg.n_task; i++) {
        grp[i]  = 0;
        ts[i]   = -1;
        pad[i]  = 1;
    }

    const int b1 = cfg.tok_primary;
    const int b2 = b1+cfg.tok_wrist;
    const int b3 = b2+cfg.n_task;
    for (int t=0; t<wnd; t++)
        for (int i=0; i<per; i++) {
            const int idx = cfg.n_task+t*per+i;
            grp[idx] = i < b1 ? 1 : i < b2 ? 2 : i < b3 ? 3 : 4;
            ts[idx]  = t;
            // obs_primary/wrist keys are masked at padded timesteps; the repeated
            // task tokens and readouts are not (matches octo_module.py).
            pad[idx] = grp[idx] == 1 ? timestep_mask[t] : grp[idx] == 2 ? wrist && timestep_mask[t] : 1;
        }

    for (int i=0; i<total; i++)
        for (int j=0; j<total; j++)
            keep[(size_t)i*total+j] = attends(grp[i], ts[i], grp[j], ts[j]) && pad[j];
}

void OctoTransformer::forward(const float* t5_out, const float* stem_p, const float* stem_w,
                              int wnd, const uint8_t* timestep_mask, float* out,
                              bool last_token_only) const {
    const int D     = cfg.d;
    const int per   = tokens_per_step();
    const int total = total_tokens(wnd);

    nn::Prof prof;
    prof.on = std::getenv("OCTO_PROFILE_TF") != nullptr;
    auto wall0 = std::chrono::steady_clock::now();

    // assemble input tokens
    prof.tic();
    std::vector<float> x    ((size_t)total*D);
    std::vector<float> task ((size_t)cfg.n_task*D);
    proj_task.forward(task.data(), t5_out, cfg.n_task);

    for (size_t i=0; i<task.size(); i++)
        task[i] += pos_task[i];

    std::memcpy(x.data(), task.data(), task.size()*sizeof(float));
    for (int t = 0; t<wnd; t++) {
        float* row = x.data()+((size_t)cfg.n_task+(size_t)t*per)*D;
        proj_prim.forward(row, stem_p+(size_t)t*cfg.tok_primary*cfg.stem_dim, cfg.tok_primary);

        const float* pp = pos_prim+(size_t)t*cfg.tok_primary*D;
        for (size_t i = 0; i < (size_t)cfg.tok_primary*D; i++)
            row[i] += pp[i];

        row += (size_t)cfg.tok_primary*D;
        if (stem_w) {
            proj_wrist.forward(row, stem_w+(size_t)t*cfg.tok_wrist*cfg.stem_dim, cfg.tok_wrist);
            const float* pw = pos_wrist+(size_t)t*cfg.tok_wrist*D;
            for (size_t i = 0; i < (size_t)cfg.tok_wrist*D; i++)
                row[i] += pw[i];
        }

        row += (size_t)cfg.tok_wrist*D;
        std::memcpy(row, task.data(), task.size()*sizeof(float));   // repeated task tokens
        row += task.size();
        std::memcpy(row, pos_readout+(size_t)t*cfg.n_readout*D,
                    (size_t)cfg.n_readout*D*sizeof(float));       // readout = pos emb only
    }
    prof.toc(prof.assemble);

    // additive attention mask: static per (wnd, timestep_mask), built once and cached
    prof.tic();
    int mask_key = wnd;
    for (int t=0; t<wnd; t++)
        mask_key = mask_key*2+(timestep_mask[t] ? 1 : 0);
    mask_key = mask_key*2+(stem_w ? 1 : 0);

    std::vector<float>& mask = mask_cache[mask_key];
    if (mask.empty()) {
        std::vector<uint8_t> keep((size_t)total*total);
        build_mask(wnd, timestep_mask, stem_w != nullptr, keep.data());

        const float NEG = std::numeric_limits<float>::lowest();
        mask.resize((size_t)total*total);
        for (size_t i=0; i<mask.size(); i++)
            mask[i] = keep[i] ? 0.0f : NEG;
    }
    prof.toc(prof.mask);

    // encoder blocks (pre-LN, MHA with qkv/out bias, gelu-tanh MLP)
    const int ro = readout_index(wnd, wnd-1);
    for (int li=0; li<cfg.n_layers; li++) {
        if (last_token_only && li == cfg.n_layers-1) {
            // final layer: K/V need every token, but only the readout row is consumed
            layers[li].forward_last_row(x.data(), ro, total, mask.data(), scratch);
            layernorm(out+(size_t)ro*D, x.data()+(size_t)ro*D,
                      final_s, final_b, 1, D, cfg.ln_eps);
            if (prof.on) {
                prof.wall = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now()-wall0).count();
                prof.report();
            }
            return;
        }
        layers[li].forward(x.data(), total, mask.data(), scratch, &prof);
    }
    prof.tic();
    layernorm(out, x.data(), final_s, final_b, total, D, cfg.ln_eps);
    prof.toc(prof.ln);

    if (prof.on) {
        prof.wall = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now()-wall0).count();
        prof.report();
    }
}

} // namespace tcpu
