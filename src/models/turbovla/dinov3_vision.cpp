/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dinov3_vision.h"
#include "models/arena.h"
#include "ops/lm_ops.h"
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
using std::size_t;

namespace tcpu {

bool Dinov3Vision::load(const std::string& dir, int img_size) {
    std::ifstream meta(dir + "/vision.meta");
    if (!meta) { std::fprintf(stderr, "turbovla: cannot open %s/vision.meta\n", dir.c_str()); return false; }
    std::string key; float val;
    while (meta >> key >> val) {
        if      (key == "hidden"    ) cfg.hidden     = (int)val;
        else if (key == "n_heads"   ) cfg.n_heads    = (int)val;
        else if (key == "head_dim"  ) cfg.head_dim   = (int)val;
        else if (key == "inter"     ) cfg.inter      = (int)val;
        else if (key == "n_layers"  ) cfg.n_layers   = (int)val;
        else if (key == "patch"     ) cfg.patch      = (int)val;
        else if (key == "prefix"    ) cfg.prefix     = (int)val;
        else if (key == "rope_theta") cfg.rope_theta = val;
        else if (key == "ln_eps"    ) cfg.ln_eps     = val;
    }

    grid = cfg.patch > 0 ? img_size/cfg.patch : 0;
    const bool bad = cfg.hidden < 1 || cfg.n_heads < 1 || cfg.head_dim < 1 || cfg.inter < 1 ||
                     cfg.n_layers < 1 || cfg.patch < 1 || cfg.prefix < 1 ||
                     cfg.n_heads*cfg.head_dim != cfg.hidden ||
                     cfg.head_dim % 4 != 0 ||       // RoPE splits head_dim into 4 quarters
                     img_size < cfg.patch || img_size % cfg.patch != 0 || grid < 1 ||
                     !(cfg.rope_theta > 0.0f);
    if (bad) {
        std::fprintf(stderr, "turbovla: %s/vision.meta shapes do not close (hidden %d heads %d "
                     "head_dim %d inter %d layers %d patch %d prefix %d img %d)\n", dir.c_str(),
                     cfg.hidden, cfg.n_heads, cfg.head_dim, cfg.inter, cfg.n_layers,
                     cfg.patch, cfg.prefix, img_size);
        return false;
    }

    if (!read_arena(dir + "/vision.bin", data)) {
        std::fprintf(stderr, "turbovla: cannot read %s/vision.bin\n", dir.c_str());
        return false;
    }
    const size_t H = (size_t)cfg.hidden, I = (size_t)cfg.inter, PD = (size_t)cfg.patch_dim();
    const size_t want = H*(size_t)cfg.prefix                       // cls + registers
                      + H*PD + H                                   // patch embed
                      + (size_t)cfg.n_layers*(2*H                  // ln1
                                              + 3*H*H + 2*H        // q(+b) k(no b) v(+b)
                                              + H*H + H            // o(+b)
                                              + 2*H                // ln2
                                              + I*H + I            // up
                                              + H*I + H)           // down
                      + 2*H;
    if (data.size() != want) {
        std::fprintf(stderr, "turbovla: %s/vision.bin has %zu floats, expected %zu\n",
                     dir.c_str(), data.size(), want);
        return false;
    }

    size_t off = 0;
    auto take = [&](size_t n) { const float* p = data.data()+off; off += n; return p; };
    using Role = nn::Linear::Role;

    cls_token  = take(H);
    reg_tokens = take(H*(size_t)(cfg.prefix-1));
    const float* pw = take(H*PD);
    const float* pb = take(H);
    patch.init(pw, pb, cfg.hidden, cfg.patch_dim(), Role::StemGemm);

    layers.resize((size_t)cfg.n_layers);
    for (int L = 0; L < cfg.n_layers; L++) {
        Dinov3Layer& l = layers[(size_t)L];
        l.ln1_w = take(H); l.ln1_b = take(H);
        const float* qw = take(H*H); const float* qb = take(H);
        const float* kw = take(H*H);
        const float* vw = take(H*H); const float* vb = take(H);
        const float* ow = take(H*H); const float* ob = take(H);
        l.ln2_w = take(H); l.ln2_b = take(H);
        const float* uw = take(I*H); const float* ub = take(I);
        const float* dw = take(H*I); const float* db = take(H);
        l.wq.init(qw, qb, cfg.hidden, cfg.hidden, Role::Gemm);
        l.wk.init(kw, nullptr, cfg.hidden, cfg.hidden, Role::Gemm);
        l.wv.init(vw, vb, cfg.hidden, cfg.hidden, Role::Gemm);
        l.wo.init(ow, ob, cfg.hidden, cfg.hidden, Role::Gemm);
        l.up.init(uw, ub, cfg.inter, cfg.hidden, Role::Mlp);
        l.down.init(dw, db, cfg.hidden, cfg.inter, Role::Mlp);
    }
    norm_w = take(H);
    norm_b = take(H);

    // RoPE table for the fixed grid. Patch centers are normalized to [-1, 1]:
    // the model was trained with random rescale, so transformers recomputes this
    // from the pixel grid at every forward - here the grid never changes.
    const int hd = cfg.head_dim, quarter = hd/4;
    const int NP = n_patches();
    rope_cos.assign((size_t)NP*hd, 0.0f);
    rope_sin.assign((size_t)NP*hd, 0.0f);
    std::vector<double> inv_freq((size_t)quarter);
    for (int i = 0; i < quarter; i++)
        inv_freq[(size_t)i] = 1.0/std::pow((double)cfg.rope_theta, (4.0*i)/hd);
    const double two_pi = 6.283185307179586476925286766559;
    for (int ph = 0; ph < grid; ph++)
        for (int pw2 = 0; pw2 < grid; pw2++) {
            const double cy = 2.0*((ph + 0.5)/grid) - 1.0;
            const double cx = 2.0*((pw2 + 0.5)/grid) - 1.0;
            float* c = rope_cos.data() + (size_t)(ph*grid+pw2)*hd;
            float* s = rope_sin.data() + (size_t)(ph*grid+pw2)*hd;
            for (int i = 0; i < quarter; i++) {
                // angles = [y*inv_freq | x*inv_freq] (head_dim/2), tiled twice
                const double ay = two_pi*cy*inv_freq[(size_t)i];
                const double ax = two_pi*cx*inv_freq[(size_t)i];
                c[i] = c[i + hd/2] = (float)std::cos(ay);
                s[i] = s[i + hd/2] = (float)std::sin(ay);
                c[quarter+i] = c[quarter+i + hd/2] = (float)std::cos(ax);
                s[quarter+i] = s[quarter+i + hd/2] = (float)std::sin(ax);
            }
        }
    return true;
}

// [n_patches, patch_dim] with per-patch order (ic, kh, kw), matching the conv
// weight reshape the converter writes.
static void extract_patches(const float* pixels, int img, int patch, float* out) {
    const int g = img/patch, pd = 3*patch*patch;
    for (int ph = 0; ph < g; ph++)
        for (int pw = 0; pw < g; pw++) {
            float* dst = out + (size_t)(ph*g+pw)*pd;
            for (int ic = 0; ic < 3; ic++)
                for (int kh = 0; kh < patch; kh++)
                    for (int kw = 0; kw < patch; kw++)
                        dst[ic*patch*patch + kh*patch + kw] =
                            pixels[((size_t)ic*img + (ph*patch+kh))*img + (pw*patch+kw)];
        }
}

// In-place RoPE over the patch rows of x [T, n_heads, head_dim]: rows
// [prefix, T) get the 2D rotation, the cls and register rows are left alone.
static void apply_rope(float* x, int prefix, int n_patches, int n_heads, int hd,
                       const float* cos_t, const float* sin_t) {
    const int half = hd/2;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int p = 0; p < n_patches; p++) {
        const float* c = cos_t + (size_t)p*hd;
        const float* s = sin_t + (size_t)p*hd;
        for (int h = 0; h < n_heads; h++) {
            float* r = x + ((size_t)(prefix+p)*n_heads + h)*hd;
            for (int d = 0; d < half; d++) {
                const float a = r[d], b = r[d+half];
                r[d]      = a*c[d]      - b*s[d];
                r[d+half] = b*c[d+half] + a*s[d+half];
            }
        }
    }
}

// TURBOVLA_PROFILE_VIT=1: per-op attribution inside one encode() (stderr). The
// tower is ~90% of a TurboVLA inference, so this is where an optimization pass
// starts. Layers run serially and the parallelism is inside each op, so plain
// accumulators are race-free.
namespace {
struct VitProf {
    bool on = std::getenv("TURBOVLA_PROFILE_VIT") != nullptr;
    double stem = 0, ln = 0, qkv = 0, rope = 0, attn = 0, proj = 0, mlp = 0, gelu = 0, res = 0;
    double t0 = 0;
    static double now_ms() {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    void tic() { if (on) t0 = now_ms(); }
    void toc(double& acc) { if (on) acc += now_ms() - t0; }
};
} // namespace

void Dinov3Vision::encode(const float* pixels, int n_views, float* out) const {
    const int H = cfg.hidden, I = cfg.inter, NP = n_patches(), T = n_tokens();
    const int img = grid*cfg.patch, PD = cfg.patch_dim();
    const int B = n_views > 0 ? n_views : 1;         // rows in the batch = B*T
    const size_t TH = (size_t)B*T*H;
    VitProf vp;
    const double t_start = VitProf::now_ms();

    if (patches.size() < (size_t)B*NP*PD) patches.resize((size_t)B*NP*PD);
    if (tok.size() < TH) tok.resize(TH);
    if (h.size()   < TH) h.resize(TH);
    if (q.size()   < TH) q.resize(TH);
    if (k.size()   < TH) k.resize(TH);
    if (v.size()   < TH) v.resize(TH);
    if (att.size() < TH) att.resize(TH);
    if (ff.size()  < (size_t)B*T*I) ff.resize((size_t)B*T*I);

    // Per view: [cls | registers | patch embeddings]. The patch GEMM runs over
    // every view's patches at once; the prefix rows are copied in around them.
    vp.tic();
    for (int w = 0; w < B; w++)
        extract_patches(pixels + (size_t)w*3*img*img, img, cfg.patch,
                        patches.data() + (size_t)w*NP*PD);
    for (int w = 0; w < B; w++) {
        float* base = tok.data() + (size_t)w*T*H;
        std::memcpy(base, cls_token, (size_t)H*sizeof(float));
        std::memcpy(base+H, reg_tokens, (size_t)(cfg.prefix-1)*H*sizeof(float));
    }
    if (B == 1) {
        patch.forward(tok.data()+(size_t)cfg.prefix*H, patches.data(), NP);
    } else {
        // One GEMM over B*NP patches, then scattered into the per-view token
        // blocks (the prefix rows break the stride).
        if (pemb.size() < (size_t)B*NP*H) pemb.resize((size_t)B*NP*H);
        patch.forward(pemb.data(), patches.data(), B*NP);
        for (int w = 0; w < B; w++)
            std::memcpy(tok.data() + (size_t)w*T*H + (size_t)cfg.prefix*H,
                        pemb.data() + (size_t)w*NP*H, (size_t)NP*H*sizeof(float));
    }
    vp.toc(vp.stem);

    const float scale = 1.0f/std::sqrt((float)cfg.head_dim);
    for (int L = 0; L < cfg.n_layers; L++) {
        const Dinov3Layer& l = layers[(size_t)L];

        vp.tic();
        layernorm(h.data(), tok.data(), l.ln1_w, l.ln1_b, B*T, H, cfg.ln_eps);
        vp.toc(vp.ln);
        vp.tic();
        l.wq.forward(q.data(), h.data(), B*T);
        l.wk.forward(k.data(), h.data(), B*T);
        l.wv.forward(v.data(), h.data(), B*T);
        vp.toc(vp.qkv);
        vp.tic();
        for (int w = 0; w < B; w++) {
            apply_rope(q.data() + (size_t)w*T*H, cfg.prefix, NP, cfg.n_heads, cfg.head_dim,
                       rope_cos.data(), rope_sin.data());
            apply_rope(k.data() + (size_t)w*T*H, cfg.prefix, NP, cfg.n_heads, cfg.head_dim,
                       rope_cos.data(), rope_sin.data());
        }
        vp.toc(vp.rope);
        // Per view, and no mask: a ViT has no padding, the prefix tokens are
        // real tokens, and one view never attends another.
        vp.tic();
        for (int w = 0; w < B; w++) {
            const size_t o = (size_t)w*T*H;
            gqa_attention_dense(att.data()+o, q.data()+o, k.data()+o, v.data()+o, T, T,
                                cfg.n_heads, cfg.n_heads, cfg.head_dim, scale);
        }
        vp.toc(vp.attn);
        vp.tic();
        if (l.wo.add_ok()) {
            l.wo.forward_add(tok.data(), att.data(), B*T);
            vp.toc(vp.proj);
        } else {
            l.wo.forward(h.data(), att.data(), B*T);
            vp.toc(vp.proj);
            vp.tic();
            for (size_t i = 0; i < TH; i++) tok[i] += h[i];
            vp.toc(vp.res);
        }

        vp.tic();
        layernorm(h.data(), tok.data(), l.ln2_w, l.ln2_b, B*T, H, cfg.ln_eps);
        vp.toc(vp.ln);
        vp.tic();
        l.up.forward(ff.data(), h.data(), B*T);
        vp.toc(vp.mlp);
        vp.tic();
        gelu_erf(ff.data(), B*T*I);
        vp.toc(vp.gelu);
        vp.tic();
        if (l.down.add_ok()) {
            l.down.forward_add(tok.data(), ff.data(), B*T);
            vp.toc(vp.mlp);
        } else {
            l.down.forward(h.data(), ff.data(), B*T);
            vp.toc(vp.mlp);
            vp.tic();
            for (size_t i = 0; i < TH; i++) tok[i] += h[i];
            vp.toc(vp.res);
        }
    }

    for (int w = 0; w < B; w++)
        layernorm(out + (size_t)w*NP*H, tok.data() + ((size_t)w*T + cfg.prefix)*H,
                  norm_w, norm_b, NP, H, cfg.ln_eps);

    if (vp.on) {
        const double wall = VitProf::now_ms() - t_start;
        const double acc = vp.stem+vp.ln+vp.qkv+vp.rope+vp.attn+vp.proj+vp.mlp+vp.gelu+vp.res;
        std::fprintf(stderr,
            "  [dinov3] x%d views  stem %5.1f | x%d: ln %5.1f  qkv %6.1f  rope %5.1f  attn %6.1f"
            "  proj %6.1f  mlp %6.1f  gelu %5.1f  res %5.1f | other %5.1f | wall %6.1f ms\n",
            B, vp.stem, cfg.n_layers, vp.ln, vp.qkv, vp.rope, vp.attn, vp.proj, vp.mlp,
            vp.gelu, vp.res, wall-acc, wall);
    }
}

} // namespace tcpu
