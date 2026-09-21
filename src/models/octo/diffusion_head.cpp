/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../arena.h"
#include "diffusion_head.h"
#include "hal/common/env.h"
#include "hal/common/threads.h"
#include "ops/lm_ops.h"
#include <cmath>
#include <cstring>
#include <fstream>

namespace tcpu {

bool DiffusionHead::load(const std::string& dir) {
    std::ifstream meta(dir + "/head.meta");
    if (!meta) return false;
    std::string key;
    double val;
    while (meta >> key >> val) {
        if      (key == "emb"       ) cfg.emb = (int)val;
        else if (key == "action_dim") cfg.action_dim = (int)val;
        else if (key == "horizon"   ) cfg.horizon = (int)val;
        else if (key == "time_dim"  ) cfg.time_dim = (int)val;
        else if (key == "num_blocks") cfg.num_blocks = (int)val;
        else if (key == "hidden"    ) cfg.hidden = (int)val;
        else if (key == "steps"     ) cfg.steps = (int)val;
        else if (key == "max_action") cfg.max_action = (float)val;
    }

    if (!read_arena(dir + "/head.bin", data)) return false;

    const int TD = cfg.time_dim;
    const int H  = cfg.hidden;
    const int IN = TD+cfg.emb+cfg.flat();
    size_t off = 0;
    bool ok = true;
    auto take = [&](size_t n) -> const float* {
        if (!ok || n > data.size() - off) { ok = false; return nullptr; }
        const float* p = data.data()+off;
        off += n;
        return p;
    };
    using Role = nn::Linear::Role;

    fourier_w = take(TD/2);

    {
        const float* w = take((size_t)2*TD*TD);
        const float* b = take(2*TD);
        if (!ok) return false;
        cond0.init(w, b, 2*TD, TD, Role::Generic);
    }

    {
        const float* w = take((size_t)TD*2*TD);
        const float* b = take(TD);
        if (!ok) return false;
        cond1.init(w, b, TD, 2*TD, Role::Generic);
    }

    net.hidden = H;
    net.ln_eps = 1e-6f;

    {
        const float* w = take((size_t)H*IN);
        const float* b = take(H);
        if (!ok) return false;
        net.in_proj.init(w, b, H, IN, Role::Generic);
    }

    net.blocks.resize(cfg.num_blocks);
    for (auto& B : net.blocks) {
        B.ln_s = take(H);
        B.ln_b = take(H);

        {
            const float* w = take((size_t)4*H*H);
            const float* b = take(4*H);
            if (!ok) return false;
            B.d0.init(w, b, 4*H, H, Role::Generic);
        }

        {
            const float* w = take((size_t)H*4*H);
            const float* b = take(H);
            if (!ok) return false;
            B.d1.init(w, b, H, 4*H, Role::Generic);
        }
    }

    {
        const float* w = take((size_t)cfg.flat()*H);
        const float* b = take(cfg.flat());
        if (!ok) return false;
        net.out_proj.init(w, b, cfg.flat(), H, Role::Generic);
    }

    betas      = take(cfg.steps);
    alphas     = take(cfg.steps);
    alpha_hats = take(cfg.steps);
    if (!ok || off != data.size()) return false;

    // the sampling loop only uses integer times 0..steps-1: precompute their conditioning
    cond_table.resize((size_t)cfg.steps*cfg.time_dim);
    for (int t=0; t<cfg.steps; t++)
        time_cond((float)t, cond_table.data()+(size_t)t*cfg.time_dim);

    return true;
}

void DiffusionHead::time_cond(float t, float* cond) const {
    const int TD = cfg.time_dim;
    const float two_pi = 6.283185307179586f;

    // learnable Fourier features -> cond MLP (Dense 2*TD -> swish -> Dense TD)
    std::vector<float> tff(TD);
    for (int i=0; i<TD/2; i++) {
        float f = two_pi*t*fourier_w[i];
        tff[i]      = std::cos(f);
        tff[TD/2+i] = std::sin(f);
    }

    std::vector<float> c1(2*TD);
    cond0.forward(c1.data(), tff.data(), 1);
    silu(c1.data(), 2*TD);
    cond1.forward(cond, c1.data(), 1);
}

void DiffusionHead::eps(const float* emb, const float* x, float t, float* out) const {
    const int TD   = cfg.time_dim;
    const int FLAT = cfg.flat();
    const int IN   = TD+cfg.emb+FLAT;

    const int ti = (int)t;
    std::vector<float> cond_buf;
    const float* cond;
    if (t == (float)ti && ti >= 0 && ti < cfg.steps && !cond_table.empty()) {
        cond = cond_table.data()+(size_t)ti*TD;
    } else {
        cond_buf.resize(TD);
        time_cond(t, cond_buf.data());
        cond = cond_buf.data();
    }

    // reverse network on concat[cond, emb, x]
    std::vector<float> in(IN);
    std::memcpy(in.data(), cond, TD*sizeof(float));
    std::memcpy(in.data()+TD, emb, cfg.emb*sizeof(float));
    std::memcpy(in.data()+TD+cfg.emb, x, FLAT*sizeof(float));
    net.forward(out, in.data());
}

void DiffusionHead::denoise(const float* emb, const float* noise, const float* z, float* actions) const {
    const int FLAT = cfg.flat();
    std::vector<float> x(noise, noise+FLAT);
    std::vector<float> e(FLAT);

    // the score net's ops are tiny (seq=1, N<=1024): past ~4 threads the OMP
    // fork/join tax of 20 steps x ~10 regions dominates. The clamp default is
    // per-backend (hal::env::head_threads; on where it measured a win, off
    // elsewhere) - threading only, never math.
    hal::ScopedTeamClamp clamp(hal::env::head_threads());
    for (int s=0; s<cfg.steps; s++) {
        const int t = cfg.steps-1-s;
        eps(emb, x.data(), (float)t, e.data());

        const float a1 = 1.0f/std::sqrt(alphas[t]);
        const float a2 = (1.0f-alphas[t])/std::sqrt(1.0f-alpha_hats[t]);
        const float sb = t > 0 ? std::sqrt(betas[t]) : 0.0f;
        const float* zs = z+(size_t)s*FLAT;

        for (int i=0; i<FLAT; i++) {
            float v = a1*(x[i]-a2*e[i])+sb*zs[i];
            x[i] = v < -cfg.max_action ? -cfg.max_action : v > cfg.max_action ? cfg.max_action : v;
        }
    }
    std::memcpy(actions, x.data(), FLAT*sizeof(float));
}

} // namespace tcpu
