/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "action_expert.h"
#include "hal/common/env.h"
#include "hal/common/threads.h"
#include "ops/lm_ops.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

namespace tcpu {

// Per-op attribution inside the denoise loop (SMOLVLA_PROFILE_EX=1). The loop is
// 10 steps x 32 layers of seq-50 ops, so it is the phase where OMP region count,
// not FLOPs, can dominate - this is what tells the two apart.
namespace {
struct ExpertProf {
    bool on = std::getenv("SMOLVLA_PROFILE_EX") != nullptr;
    double embed = 0, norm = 0, qkv = 0, rope = 0, concat = 0, attn = 0;
    double proj = 0, mlp = 0, silu = 0, res = 0, out = 0, wall = 0;
    std::chrono::steady_clock::time_point t0;

    void tic() { if (on) t0 = std::chrono::steady_clock::now(); }
    void toc(double& acc) {
        if (!on) return;
        acc += std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();
    }
    void report() const {
        if (!on) return;
        double sum = embed+norm+qkv+rope+concat+attn+proj+mlp+silu+res+out;
        std::fprintf(stderr,
            "  [expert] embed %5.1f  norm %5.1f  qkv %5.1f  rope %5.1f  concat %5.1f  attn %5.1f\n"
            "           proj %5.1f  mlp %5.1f  silu %5.1f  res %5.1f  out %5.1f | other %5.1f | wall %6.1f ms\n",
            embed, norm, qkv, rope, concat, attn, proj, mlp, silu, res, out, wall-sum, wall);
    }
};

// denoise() drives denoise_step() sequentially (all parallelism is inside the
// ops), so one instance keeps the public signatures unchanged. thread_local: the
// reset and the wall write run whether or not profiling is on.
ExpertProf& prof() { static thread_local ExpertProf p; return p; }
} // namespace

bool ActionExpert::load(const std::string& dir) {
    std::ifstream meta(dir + "/aex.meta");
    if (!meta) { std::fprintf(stderr, "smolvla: cannot open %s/aex.meta\n", dir.c_str()); return false; }
    std::string k; double v;
    int san = cfg.self_attn_every_n;
    while (meta >> k >> v) {
        if      (k == "expert_h"         ) cfg.expert_h = (int)v;
        else if (k == "expert_ffn"       ) cfg.expert_ffn = (int)v;
        else if (k == "n_q"              ) cfg.n_q = (int)v;
        else if (k == "n_kv"             ) cfg.n_kv = (int)v;
        else if (k == "head_dim"         ) cfg.head_dim = (int)v;
        else if (k == "eps"              ) cfg.rms_eps = v;
        else if (k == "rope_base"        ) cfg.rope_base = v;
        else if (k == "n_layers"         ) cfg.n_layers = (int)v;
        else if (k == "self_attn_every_n") san = (int)v;
        else if (k == "chunk"            ) cfg.chunk = (int)v;
        else if (k == "num_steps"        ) cfg.num_steps = (int)v;
        else if (k == "max_action_dim"   ) cfg.max_action_dim = (int)v;
        else if (k == "min_period"       ) cfg.min_period = v;
        else if (k == "max_period"       ) cfg.max_period = v;
    }
    cfg.self_attn_every_n = san;

    // SMOLVLA_NUM_STEPS overrides the checkpoint's flow-matching step count. The
    // Euler integrator is exact in the limit and the chunk moves smoothly as the
    // count drops, so this is the one knob that trades action accuracy for
    // latency directly (~50 ms/step on a Pi 5). Serving default is the
    // checkpoint's; see docs/11-smolvla-raspi.md for the measured curve.
    if (const char* e = std::getenv("SMOLVLA_NUM_STEPS")) {
        const int n = std::atoi(e);
        if (n > 0) cfg.num_steps = n;
    }

    const int EH = cfg.expert_h, EF = cfg.expert_ffn, QF = cfg.q_full(), KV = cfg.kv_full();
    const int NL = cfg.n_layers, MAD = cfg.max_action_dim;

    // total float count
    size_t total = 0;
    auto layer_floats = [&](bool self) {
        size_t kvw = self ? (size_t)KV*EH : (size_t)KV*KV;
        return (size_t)EH + (size_t)QF*EH + kvw*2 + (size_t)EH*QF
             + (size_t)EH + (size_t)EF*EH*2 + (size_t)EH*EF;
    };
    for (int L = 0; L < NL; L++) total += layer_floats(cfg.self_attn_every_n > 0 && L % cfg.self_attn_every_n == 0);
    total += (size_t)EH;                                   // out_norm
    total += (size_t)EH*MAD + EH;                          // action_in_proj
    total += (size_t)EH*(2*EH) + EH;                       // action_time_mlp_in
    total += (size_t)EH*EH + EH;                           // action_time_mlp_out
    total += (size_t)MAD*EH + MAD;                         // action_out_proj

    std::ifstream bin(dir + "/aex.bin", std::ios::binary);
    if (!bin) { std::fprintf(stderr, "smolvla: cannot open %s/aex.bin\n", dir.c_str()); return false; }
    blob.resize(total);
    bin.read(reinterpret_cast<char*>(blob.data()), total*sizeof(float));
    if (!bin || bin.peek() != EOF) {
        std::fprintf(stderr, "smolvla: %s/aex.bin size does not match aex.meta (need %zu floats)\n", dir.c_str(), total);
        return false;
    }

    using Role = nn::Linear::Role;
    layers.resize(NL);
    size_t off = 0;
    auto take = [&](size_t n) { const float* p = blob.data()+off; off += n; return p; };
    for (int L = 0; L < NL; L++) {
        ExpertLayerW& w = layers[L];
        w.is_self_attn = cfg.self_attn_every_n > 0 && L % cfg.self_attn_every_n == 0;
        const int kk = w.is_self_attn ? EH : KV;
        w.ln_in = take(EH);
        w.q   .init(take((size_t)QF*EH), nullptr, QF, EH, Role::Gemm);
        w.k   .init(take((size_t)KV*kk), nullptr, KV, kk, Role::Gemm);
        w.v   .init(take((size_t)KV*kk), nullptr, KV, kk, Role::Gemm);
        w.o   .init(take((size_t)EH*QF), nullptr, EH, QF, Role::Gemm);
        w.ln_post = take(EH);
        w.gate.init(take((size_t)EF*EH), nullptr, EF, EH, Role::Mlp);
        w.up  .init(take((size_t)EF*EH), nullptr, EF, EH, Role::Mlp);
        w.down.init(take((size_t)EH*EF), nullptr, EH, EF, Role::Mlp);
    }
    out_norm     = take(EH);
    flow.ain_w   = take((size_t)EH*MAD);   flow.ain_b  = take(EH);
    flow.at1_w   = take((size_t)EH*2*EH);  flow.at1_b  = take(EH);
    flow.at2_w   = take((size_t)EH*EH);    flow.at2_b  = take(EH);
    flow.aout_w  = take((size_t)MAD*EH);   flow.aout_b = take(MAD);
    return true;
}

void ActionExpert::embed_suffix(const float* x_t, float time, float* suffix) const {
    const int EH = cfg.expert_h, MAD = cfg.max_action_dim, C = cfg.chunk, half = EH/2;
    auto grow = [](std::vector<float>& v, size_t n) { if (v.size() < n) v.resize(n); };
    grow(ds.aemb, (size_t)C*EH); grow(ds.te, EH);
    grow(ds.at, (size_t)C*2*EH); grow(ds.mlp1, (size_t)C*EH);
    std::vector<float>& action_emb = ds.aemb;
    std::vector<float>& te = ds.te;

    dense_linear(action_emb.data(), x_t, flow.ain_w, flow.ain_b, C, EH, MAD);

    // sinusoidal time embedding (double math, matches create_sinusoidal_pos_embedding)
    for (int i = 0; i < half; i++) {
        double frac = (half == 1) ? 0.0 : (double)i/(double)(half-1);
        double period = cfg.min_period*std::pow((double)cfg.max_period/cfg.min_period, frac);
        double s = (2.0*M_PI/period)*(double)time;
        te[i]      = (float)std::sin(s);
        te[half+i] = (float)std::cos(s);
    }

    // action_time = concat(action_emb, time_emb) -> [C, 2*EH]
    std::vector<float>& at = ds.at;
    for (int t = 0; t < C; t++) {
        std::memcpy(at.data() + (size_t)t*2*EH, action_emb.data() + (size_t)t*EH, EH*sizeof(float));
        std::memcpy(at.data() + (size_t)t*2*EH + EH, te.data(), EH*sizeof(float));
    }

    std::vector<float>& mlp1 = ds.mlp1;
    dense_linear(mlp1.data(), at.data(), flow.at1_w, flow.at1_b, C, EH, 2*EH);
    for (size_t i = 0; i < (size_t)C*EH; i++) { float x = mlp1[i]; mlp1[i] = x/(1.0f + std::exp(-x)); }
    dense_linear(suffix, mlp1.data(), flow.at2_w, flow.at2_b, C, EH, EH);
}

void ActionExpert::prepare_denoise(const std::vector<VlmKV>& kv, int n_prefix) const {
    const int EH = cfg.expert_h, EF = cfg.expert_ffn, QF = cfg.q_full(), KV = cfg.kv_full();
    const int C = cfg.chunk, SK = n_prefix + C;
    auto grow = [](std::vector<float>& v, size_t n) { if (v.size() < n) v.resize(n); };

    grow(ds.h,  (size_t)C*EH); grow(ds.hn, (size_t)C*EH); grow(ds.q,   (size_t)C*QF);
    grow(ds.attn,(size_t)C*QF); grow(ds.o, (size_t)C*EH); grow(ds.hn2, (size_t)C*EH);
    grow(ds.g,  (size_t)C*EF); grow(ds.u,  (size_t)C*EF); grow(ds.gu,  (size_t)C*EF);
    grow(ds.dn, (size_t)C*EH); grow(ds.ks, (size_t)C*KV); grow(ds.vs,  (size_t)C*KV);

    ds.Kf.resize(cfg.n_layers);
    ds.Vf.resize(cfg.n_layers);
    for (int L = 0; L < cfg.n_layers; L++) {
        if (!layers[L].is_self_attn) continue;
        ds.Kf[L].resize((size_t)SK*KV);
        ds.Vf[L].resize((size_t)SK*KV);
        // the prefix half is the VLM cache, constant across the denoise loop
        std::memcpy(ds.Kf[L].data(), kv[L].k.data(), (size_t)n_prefix*KV*sizeof(float));
        std::memcpy(ds.Vf[L].data(), kv[L].v.data(), (size_t)n_prefix*KV*sizeof(float));
    }
}

void ActionExpert::denoise_step(const std::vector<VlmKV>& kv, int n_prefix, const float* x_t, float time,
                                const float* mask_full, const float* mask_prefix,
                                const int* pos_full, const int* pos_rebased, float* v_t,
                                const std::vector<std::vector<float>>& cK,
                                const std::vector<std::vector<float>>& cV) const {
    const int EH = cfg.expert_h, EF = cfg.expert_ffn, KV = cfg.kv_full();
    const int C = cfg.chunk, HD = cfg.head_dim, MAD = cfg.max_action_dim;
    const float scale = 1.0f/std::sqrt((float)HD);
    const int SK = n_prefix + C;   // self-attn key length

    ExpertProf& P = prof();

    // Grow-only members, see DenoiseScratch.
    std::vector<float>& h = ds.h;
    P.tic();
    embed_suffix(x_t, time, h.data());
    P.toc(P.embed);

    std::vector<float>&hn=ds.hn; std::vector<float>&q=ds.q; std::vector<float>&attn=ds.attn;
    std::vector<float>&o=ds.o;   std::vector<float>&hn2=ds.hn2; std::vector<float>&g=ds.g;
    std::vector<float>&u=ds.u;   std::vector<float>&gu=ds.gu;  std::vector<float>&dn=ds.dn;
    std::vector<float>&ks=ds.ks; std::vector<float>&vs=ds.vs;

    for (int L = 0; L < cfg.n_layers; L++) {
        const ExpertLayerW& w = layers[L];
        P.tic();
        rmsnorm(hn.data(), h.data(), w.ln_in, C, EH, cfg.rms_eps);
        P.toc(P.norm);
        P.tic();
        w.q.forward(q.data(), hn.data(), C);
        P.toc(P.qkv);

        if (w.is_self_attn) {
            P.tic();
            w.k.forward(ks.data(), hn.data(), C);
            w.v.forward(vs.data(), hn.data(), C);
            P.toc(P.qkv);
            P.tic();
            rope_neox(q.data(),  pos_full, C, cfg.n_q,  HD, cfg.rope_base);
            rope_neox(ks.data(), pos_full, C, cfg.n_kv, HD, cfg.rope_base);
            P.toc(P.rope);
            P.tic();
            // only the suffix tail changes between steps; the [0, n_prefix)
            // half is the VLM cache, written once by prepare_denoise()
            std::vector<float>& Kf = ds.Kf[L];
            std::vector<float>& Vf = ds.Vf[L];
            std::memcpy(Kf.data() + (size_t)n_prefix*KV, ks.data(), (size_t)C*KV*sizeof(float));
            std::memcpy(Vf.data() + (size_t)n_prefix*KV, vs.data(), (size_t)C*KV*sizeof(float));
            P.toc(P.concat);
            P.tic();
            gqa_attention_masked(attn.data(), q.data(), Kf.data(), Vf.data(),
                                 C, SK, cfg.n_q, cfg.n_kv, HD, scale, mask_full);
            P.toc(P.attn);
        } else {
            // cross-attn: K/V are the reprojected VLM cache (precomputed once, constant across
            // denoise steps). Q gets rebased RoPE; K is not re-RoPE'd.
            const float *Kx, *Vx;
            std::vector<float> Kr, Vr;
            if (L < (int)cK.size() && !cK[L].empty()) {
                Kx = cK[L].data();
                Vx = cV[L].data();
            } else {   // fallback: compute locally (e.g. single-step call in tests)
                Kr.resize((size_t)n_prefix*KV);
                Vr.resize((size_t)n_prefix*KV);
                w.k.forward(Kr.data(), kv[L].k.data(), n_prefix);
                w.v.forward(Vr.data(), kv[L].v.data(), n_prefix);
                Kx = Kr.data();
                Vx = Vr.data();
            }
            P.tic();
            rope_neox(q.data(), pos_rebased, C, cfg.n_q, HD, cfg.rope_base);
            P.toc(P.rope);
            P.tic();
            gqa_attention_masked(attn.data(), q.data(), Kx, Vx,
                                 C, n_prefix, cfg.n_q, cfg.n_kv, HD, scale, mask_prefix);
            P.toc(P.attn);
        }
        P.tic();
        w.o.forward(o.data(), attn.data(), C);
        P.toc(P.proj);
        P.tic();
        for (size_t i = 0; i < (size_t)C*EH; i++) h[i] += o[i];
        P.toc(P.res);

        P.tic();
        rmsnorm(hn2.data(), h.data(), w.ln_post, C, EH, cfg.rms_eps);
        P.toc(P.norm);
        P.tic();
        w.gate.forward(g.data(), hn2.data(), C);
        w.up  .forward(u.data(), hn2.data(), C);
        P.toc(P.mlp);
        P.tic();
        silu_gate(gu.data(), g.data(), u.data(), C*EF);
        P.toc(P.silu);
        P.tic();
        w.down.forward(dn.data(), gu.data(), C);
        P.toc(P.mlp);
        P.tic();
        for (size_t i = 0; i < (size_t)C*EH; i++) h[i] += dn[i];
        P.toc(P.res);
    }

    if (ds.hf.size() < (size_t)C*EH) ds.hf.resize((size_t)C*EH);
    std::vector<float>& hf = ds.hf;
    P.tic();
    rmsnorm(hf.data(), h.data(), out_norm, C, EH, cfg.rms_eps);
    dense_linear(v_t, hf.data(), flow.aout_w, flow.aout_b, C, MAD, EH);
    P.toc(P.out);
}

void ActionExpert::denoise(const std::vector<VlmKV>& kv, int n_prefix, const float* noise,
                           const float* mask_full, const int* pos_full, float* out) const {
    const int C = cfg.chunk, MAD = cfg.max_action_dim, SK = n_prefix + C;

    // The whole loop is seq-50 ops repeated 10 x 32 times, so its cost is part
    // FLOPs and part OMP region overhead; TCPU_EXPERT_THREADS caps the team for
    // the loop only (threading, never math). 0 = inherit the global team.
    hal::ScopedTeamClamp clamp(hal::env::expert_threads());

    ExpertProf& P = prof();
    P = ExpertProf{};
    const auto wall0 = std::chrono::steady_clock::now();

    // derive cross-attn mask (prefix columns) and rebased positions
    std::vector<float> mask_prefix((size_t)C*n_prefix);
    for (int i = 0; i < C; i++)
        std::memcpy(mask_prefix.data() + (size_t)i*n_prefix, mask_full + (size_t)i*SK, n_prefix*sizeof(float));
    std::vector<int> pos_rebased(C);
    int pmin = pos_full[0];
    for (int i = 1; i < C; i++) if (pos_full[i] < pmin) pmin = pos_full[i];
    for (int i = 0; i < C; i++) pos_rebased[i] = pos_full[i] - pmin;

    // Precompute cross-attn K/V reprojections once: they depend only on the VLM cache,
    // which is constant across denoise steps (was redundantly recomputed 10x before).
    const int KV = cfg.kv_full();
    std::vector<std::vector<float>> cK(cfg.n_layers), cV(cfg.n_layers);
    for (int L = 0; L < cfg.n_layers; L++) {
        if (layers[L].is_self_attn) continue;
        cK[L].resize((size_t)n_prefix*KV);
        cV[L].resize((size_t)n_prefix*KV);
        layers[L].k.forward(cK[L].data(), kv[L].k.data(), n_prefix);
        layers[L].v.forward(cV[L].data(), kv[L].v.data(), n_prefix);
    }

    prepare_denoise(kv, n_prefix);

    std::vector<float> x((size_t)C*MAD, 0.0f);
    std::memcpy(x.data(), noise, (size_t)C*MAD*sizeof(float));
    std::vector<float> v_t((size_t)C*MAD);
    const float dt = -1.0f/(float)cfg.num_steps;

    for (int step = 0; step < cfg.num_steps; step++) {
        const float time = (float)(1.0 + step*(-1.0/cfg.num_steps));
        denoise_step(kv, n_prefix, x.data(), time, mask_full, mask_prefix.data(),
                     pos_full, pos_rebased.data(), v_t.data(), cK, cV);
        for (size_t i = 0; i < x.size(); i++) x[i] += dt*v_t[i];
    }
    std::memcpy(out, x.data(), (size_t)C*MAD*sizeof(float));

    P.wall = std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - wall0).count();
    P.report();
}

} // namespace tcpu
