/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "models/diffusion/unet1d.h"
#include "models/arena.h"
#include "ops/conv_ops.h"
#include "ops/lm_ops.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

namespace tcpu {

// Weight order in <name>.bin, and the converter writes exactly this. Every shape
// is derivable from DPConfig, so the UNet needs no meta of its own -- which also
// means a config/checkpoint mismatch shows up as the arena not being consumed
// exactly, checked at the end rather than read past.
//
//   step1.W [4D, D]  step1.b [4D]   step2.W [D, 4D]  step2.b [D]
//   down i:  res(r1) res(r2) [ds.W [C, 3C]  ds.b [C]]
//   mid:     res     res
//   up   i:  res(r1) res(r2) [up.W [Cin, 4, Cout]  up.b [Cout]]
//   final:   conv.W [C0, k*C0]  conv.b [C0]  gn_s [C0]  gn_b [C0]
//            final.W [A, C0]  final.b [A]
//
//   res(cin -> cout):
//     conv1.W [cout, k*cin]  conv1.b  gn_s  gn_b
//     cond.W  [cond_ch, cond_dim]  cond.b        (cond_ch = 2*cout with FiLM scale)
//     conv2.W [cout, k*cout]  conv2.b  gn_s  gn_b
//     res.W   [cout, cin]  res.b                 (only when cin != cout)

namespace {
struct Take {
    const std::vector<float>& d;
    size_t off = 0;
    bool ok = true;
    const float* operator()(size_t n) {
        if (!ok || n > d.size() - off) { ok = false; return nullptr; }
        const float* p = d.data() + off;
        off += n;
        return p;
    }
};
} // namespace

bool DPUNet1d::load(const std::string& dir, const std::string& name, const DPConfig& c) {
    cfg = c;
    if (cfg.down_dims.empty()) return false;
    if (!read_arena(dir + "/" + name + ".bin", data)) return false;

    const int A  = cfg.action_dim;
    const int D  = cfg.step_embed_dim;
    const int CD = cfg.cond_dim();
    const int k  = cfg.kernel_size;
    Take take{data};

    auto lin = [&](nn::Linear& L, int N, int K, nn::Linear::Role r) {
        const float* w = take((size_t)N*K);
        const float* b = take(N);
        if (take.ok) L.init(w, b, N, K, r);
    };

    auto conv_block = [&](DPConvBlock& cb, int cin, int cout, int kk) {
        cb.cin = cin; cb.cout = cout; cb.k = kk;
        lin(cb.conv, cout, kk*cin, nn::Linear::Role::Gemm);
        cb.gn_s = take(cout);
        cb.gn_b = take(cout);
    };

    auto res_block = [&](DPResBlock& rb, int cin, int cout) {
        rb.cin = cin; rb.cout = cout; rb.film_scale = cfg.film_scale;
        conv_block(rb.conv1, cin, cout, k);
        lin(rb.cond, cfg.film_scale ? 2*cout : cout, CD, nn::Linear::Role::Generic);
        conv_block(rb.conv2, cout, cout, k);
        rb.has_res = (cin != cout);
        if (rb.has_res) lin(rb.res, cout, cin, nn::Linear::Role::Gemm);
    };

    lin(step1, 4*D, D, nn::Linear::Role::Generic);
    lin(step2, D, 4*D, nn::Linear::Role::Generic);

    // in_out mirrors the reference: (action_dim -> down_dims[0]), then successive
    // down_dims pairs.
    std::vector<std::pair<int,int>> in_out;
    in_out.emplace_back(A, cfg.down_dims[0]);
    for (size_t i=0; i+1<cfg.down_dims.size(); i++)
        in_out.emplace_back(cfg.down_dims[i], cfg.down_dims[i+1]);

    down.clear();
    down.resize(in_out.size());
    for (size_t i=0; i<in_out.size(); i++) {
        Down& d = down[i];
        res_block(d.r1, in_out[i].first, in_out[i].second);
        res_block(d.r2, in_out[i].second, in_out[i].second);
        d.has = (i + 1 < in_out.size());
        if (d.has) {
            d.dc = in_out[i].second;
            lin(d.ds, d.dc, 3*d.dc, nn::Linear::Role::Gemm);
        }
    }

    const int mid = cfg.down_dims.back();
    res_block(mid1, mid, mid);
    res_block(mid2, mid, mid);

    // Populated in place, never built as a local and pushed: nn::Linear caches a
    // pointer into its own packed-panel buffer, so copying an initialised one
    // leaves that pointer aimed at the source's freed memory.
    up.clear();
    up.resize(in_out.size() - 1);
    for (size_t ind=0; ind+1<in_out.size(); ind++) {
        // reversed(in_out[1:]) yields (dim_out, dim_in) pairs.
        const std::pair<int,int>& p = in_out[in_out.size()-1-ind];
        const int dim_out = p.first, dim_in = p.second;
        Up& u = up[ind];
        res_block(u.r1, dim_in*2, dim_out);
        res_block(u.r2, dim_out, dim_out);
        // is_last is `ind >= len(in_out) - 1`, which the stock 3-stage config
        // never reaches -- both up blocks upsample and the decoder returns to
        // the full horizon. Kept as the reference's condition, not simplified
        // to "always", so a 2-stage config still loads correctly.
        u.has = (ind < in_out.size() - 1);
        if (u.has) {
            u.uc = dim_out;
            u.uw = take((size_t)dim_out*4*dim_out);
            u.ub = take(dim_out);
        }
    }

    conv_block(final_block, cfg.down_dims[0], cfg.down_dims[0], k);
    lin(final_conv, A, cfg.down_dims[0], nn::Linear::Role::Generic);

    return take.ok && take.off == data.size();
}

void DPConvBlock::forward(float* out, const float* x, int T, int groups, float eps,
                          std::vector<float>& col) const {
    im2col1d(col, x, T, cin, k, /*stride*/1, /*pad*/k/2, T);
    conv.forward(out, col.data(), T);
    groupnorm(out, out, gn_s, gn_b, T, cout, groups, eps);
    mish(out, T*cout);
}

void DPResBlock::forward(float* out, const float* x, const float* gm, int T,
                         int groups, float eps, std::vector<float>& col,
                         std::vector<float>& scratch) const {
    scratch.resize((size_t)T*cout + (size_t)cout*2);
    float* h  = scratch.data();
    float* ce = h + (size_t)T*cout;

    conv1.forward(h, x, T, groups, eps, col);

    // FiLM: one [cond_dim] vector per step, broadcast over the sequence.
    cond.forward(ce, gm, 1);
    if (film_scale) {
        const float* sc = ce;
        const float* bi = ce + cout;
        for (int t=0; t<T; t++) {
            float* r = h + (size_t)t*cout;
            for (int c=0; c<cout; c++) r[c] = sc[c]*r[c] + bi[c];
        }
    } else {
        for (int t=0; t<T; t++) {
            float* r = h + (size_t)t*cout;
            for (int c=0; c<cout; c++) r[c] += ce[c];
        }
    }

    conv2.forward(out, h, T, groups, eps, col);

    // Residual: 1x1 conv when the channel count changes, identity otherwise.
    if (has_res) {
        std::vector<float> r((size_t)T*cout);
        res.forward(r.data(), x, T);
        for (size_t i=0; i<(size_t)T*cout; i++) out[i] += r[i];
    } else {
        for (size_t i=0; i<(size_t)T*cout; i++) out[i] += x[i];
    }
}

void DPUNet1d::step_embedding(float* out, int t) const {
    // DiffusionSinusoidalPosEmb: half the dim is sin, half cos, with the
    // frequency ladder exp(-i * log(10000)/(half-1)).
    const int dim  = cfg.step_embed_dim;
    const int half = dim/2;
    const float sc = std::log(10000.0f)/(float)(half - 1);
    for (int i=0; i<half; i++) {
        const float f = (float)t*std::exp(-(float)i*sc);
        out[i]        = std::sin(f);
        out[half + i] = std::cos(f);
    }
}

void DPUNet1d::forward(float* eps_out, const float* sample, const float* gc, int t) const {
    const int A    = cfg.action_dim;
    const int H    = cfg.horizon;
    const int G    = cfg.n_groups;
    const float ge = cfg.gn_eps;
    const int D    = cfg.step_embed_dim;
    const int CD   = cfg.cond_dim();

    // Global feature = [step_encoder(t), global_cond], then Mish once for every
    // block's cond_encoder (see DPResBlock::forward).
    std::vector<float> emb(D), wide(D*4), gfeat(CD);
    step_embedding(emb.data(), t);
    step1.forward(wide.data(), emb.data(), 1);
    mish(wide.data(), D*4);
    step2.forward(gfeat.data(), wide.data(), 1);
    std::memcpy(gfeat.data() + D, gc, sizeof(float)*(size_t)(CD - D));
    std::vector<float> gm(CD);
    std::memcpy(gm.data(), gfeat.data(), sizeof(float)*(size_t)CD);
    mish(gm.data(), CD);

    std::vector<float> col, scratch;
    std::vector<float> x((size_t)H*A);
    std::memcpy(x.data(), sample, sizeof(float)*(size_t)H*A);

    int T = H, C = A;
    std::vector<std::vector<float>> skips;
    std::vector<int> skipT, skipC;

    for (const Down& d : down) {
        std::vector<float> a((size_t)T*d.r1.cout);
        d.r1.forward(a.data(), x.data(), gm.data(), T, G, ge, col, scratch);
        std::vector<float> b((size_t)T*d.r2.cout);
        d.r2.forward(b.data(), a.data(), gm.data(), T, G, ge, col, scratch);
        C = d.r2.cout;

        skips.push_back(b);
        skipT.push_back(T);
        skipC.push_back(C);

        if (d.has) {
            const int Tout = (T + 2*1 - 3)/2 + 1;
            im2col1d(col, b.data(), T, C, /*k*/3, /*stride*/2, /*pad*/1, Tout);
            x.assign((size_t)Tout*C, 0.f);
            d.ds.forward(x.data(), col.data(), Tout);
            T = Tout;
        } else {
            x = b;
        }
    }

    {
        std::vector<float> a((size_t)T*C);
        mid1.forward(a.data(), x.data(), gm.data(), T, G, ge, col, scratch);
        std::vector<float> b((size_t)T*C);
        mid2.forward(b.data(), a.data(), gm.data(), T, G, ge, col, scratch);
        x = b;
    }

    for (const Up& u : up) {
        // Channel-concat with the matching encoder skip. Both are [T, C] so the
        // concat is a per-timestep splice, not a memcpy of two halves.
        const int sC = skipC.back(), sT = skipT.back();
        std::vector<float> cat((size_t)T*(C + sC));
        for (int tt=0; tt<T; tt++) {
            std::memcpy(cat.data() + (size_t)tt*(C+sC), x.data() + (size_t)tt*C,
                        sizeof(float)*(size_t)C);
            std::memcpy(cat.data() + (size_t)tt*(C+sC) + C,
                        skips.back().data() + (size_t)tt*sC, sizeof(float)*(size_t)sC);
        }
        (void)sT;
        skips.pop_back(); skipT.pop_back(); skipC.pop_back();

        std::vector<float> a((size_t)T*u.r1.cout);
        u.r1.forward(a.data(), cat.data(), gm.data(), T, G, ge, col, scratch);
        std::vector<float> b((size_t)T*u.r2.cout);
        u.r2.forward(b.data(), a.data(), gm.data(), T, G, ge, col, scratch);
        C = u.r2.cout;

        if (u.has) {
            const int Tout = (T-1)*2 - 2*1 + 4;
            x.assign((size_t)Tout*u.uc, 0.f);
            conv_transpose1d(x.data(), b.data(), u.uw, u.ub, T, C, u.uc, 4, 2, 1);
            T = Tout;
            C = u.uc;
        } else {
            x = b;
        }
    }

    std::vector<float> f((size_t)T*final_block.cout);
    final_block.forward(f.data(), x.data(), T, 8, ge, col);
    final_conv.forward(eps_out, f.data(), T);
}

} // namespace tcpu
