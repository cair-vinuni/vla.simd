/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "siglip_vision.h"
#include "ops/lm_ops.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

namespace tcpu {

bool SiglipVision::load(const std::string& dir) {
    std::ifstream meta(dir + "/vit.meta");
    if (!meta) { std::fprintf(stderr, "smolvla: cannot open %s/vit.meta\n", dir.c_str()); return false; }
    std::string k; float v;
    while (meta >> k >> v) {
        if      (k == "hidden"      ) cfg.hidden = (int)v;
        else if (k == "n_heads"     ) cfg.n_heads = (int)v;
        else if (k == "head_dim"    ) cfg.head_dim = (int)v;
        else if (k == "inter"       ) cfg.inter = (int)v;
        else if (k == "n_layers"    ) cfg.n_layers = (int)v;
        else if (k == "patch"       ) cfg.patch = (int)v;
        else if (k == "img"         ) cfg.img = (int)v;
        else if (k == "n_patches"   ) cfg.n_patches = (int)v;
        else if (k == "ln_eps"      ) cfg.ln_eps = v;
        else if (k == "scale_factor") cfg.scale_factor = (int)v;
        else if (k == "mm_out"      ) cfg.mm_out = (int)v;
        else if (k == "n_img_tok"   ) cfg.n_img_tok = (int)v;
    }

    // The patch extractor and the pixel shuffle derive their grids from these, so
    // a meta that does not close writes past both buffers.
    {
        const int side = (int)std::lround(std::sqrt((double)cfg.n_patches));
        const bool bad = cfg.hidden < 1 || cfg.n_heads < 1 || cfg.head_dim < 1 ||
                         cfg.inter < 1 || cfg.n_layers < 1 || cfg.patch < 1 ||
                         cfg.img < 1 || cfg.mm_out < 1 || cfg.scale_factor < 1 ||
                         cfg.n_heads*cfg.head_dim != cfg.hidden ||
                         side*side != cfg.n_patches ||
                         cfg.img % cfg.patch != 0 ||
                         (cfg.img/cfg.patch)*(cfg.img/cfg.patch) != cfg.n_patches ||
                         side % cfg.scale_factor != 0 ||
                         (side/cfg.scale_factor)*(side/cfg.scale_factor) != cfg.n_img_tok;
        if (bad) {
            std::fprintf(stderr, "smolvla: %s/vit.meta shapes do not close "
                         "(img %d patch %d n_patches %d sf %d n_img_tok %d hidden %d "
                         "heads %d head_dim %d)\n", dir.c_str(), cfg.img, cfg.patch,
                         cfg.n_patches, cfg.scale_factor, cfg.n_img_tok, cfg.hidden,
                         cfg.n_heads, cfg.head_dim);
            return false;
        }
    }

    const int H = cfg.hidden, I = cfg.inter, PD = cfg.patch_dim(), NP = cfg.n_patches;
    // fp32 region: patch_b, pos_emb, per-layer(ln1 w/b, bq,bk,bv,bo, ln2 w/b, fc1_b, fc2_b), post_ln w/b
    const size_t fcount = (size_t)H + (size_t)NP*H
                        + (size_t)cfg.n_layers*(2*H + 4*H + 2*H + I + H) + (size_t)2*H;
    // bf16 region: patch_w, per-layer(Wq,Wk,Wv,Wo, fc1_w, fc2_w), mm_proj
    const size_t wcount = (size_t)H*PD
                        + (size_t)cfg.n_layers*(4*(size_t)H*H + (size_t)I*H + (size_t)H*I)
                        + (size_t)cfg.mm_out*cfg.shuffled_dim();

    std::ifstream bin(dir + "/vit.bin", std::ios::binary);
    if (!bin) { std::fprintf(stderr, "smolvla: cannot open %s/vit.bin\n", dir.c_str()); return false; }
    fnorms.resize(fcount);
    bin.read(reinterpret_cast<char*>(fnorms.data()), fcount*sizeof(float));
    wbf.resize(wcount);
    bin.read(reinterpret_cast<char*>(wbf.data()), wcount*sizeof(uint16_t));
    if (!bin || bin.peek() != EOF) {
        std::fprintf(stderr, "smolvla: %s/vit.bin size does not match vit.meta\n", dir.c_str());
        return false;
    }

    size_t fo = 0, wo = 0;
    auto tf = [&](size_t n) { const float* p = fnorms.data()+fo; fo += n; return p; };
    auto tw = [&](size_t n) { const uint16_t* p = wbf.data()+wo; wo += n; return p; };

    const float* patch_b = tf(H);
    pos_emb = tf((size_t)NP*H);

    struct LayerF { const float *ln1_w, *ln1_b, *bq, *bk, *bv, *bo, *ln2_w, *ln2_b, *fc1_b, *fc2_b; };
    std::vector<LayerF> lf(cfg.n_layers);
    for (int L = 0; L < cfg.n_layers; L++) {
        LayerF& f = lf[L];
        f.ln1_w = tf(H); f.ln1_b = tf(H);
        f.bq = tf(H); f.bk = tf(H); f.bv = tf(H); f.bo = tf(H);
        f.ln2_w = tf(H); f.ln2_b = tf(H);
        f.fc1_b = tf(I); f.fc2_b = tf(H);
    }
    post_ln_w = tf(H); post_ln_b = tf(H);

    using Role = nn::Linear::Role;
    patch_lin.init_bf16(tw((size_t)H*PD), patch_b, H, PD, Role::StemGemm);

    layers.resize(cfg.n_layers);
    for (int L = 0; L < cfg.n_layers; L++) {
        nn::EncoderLayer& e = layers[L];
        const LayerF& f = lf[L];
        e.ln1_s = f.ln1_w; e.ln1_b = f.ln1_b;
        e.ln2_s = f.ln2_w; e.ln2_b = f.ln2_b;
        e.attn.wq.init_bf16(tw((size_t)H*H), f.bq, H, H, Role::Gemm);
        e.attn.wk.init_bf16(tw((size_t)H*H), f.bk, H, H, Role::Gemm);
        e.attn.wv.init_bf16(tw((size_t)H*H), f.bv, H, H, Role::Gemm);
        e.attn.wo.init_bf16(tw((size_t)H*H), f.bo, H, H, Role::Gemm);
        e.attn.set_shape(cfg.n_heads, cfg.head_dim);
        e.w1.init_bf16(tw((size_t)I*H), f.fc1_b, I, H, Role::Mlp);
        e.w2.init_bf16(tw((size_t)H*I), f.fc2_b, H, I, Role::Mlp);
        e.d = H; e.mlp = I;
        e.ln_eps = cfg.ln_eps;
    }
    mm_proj.init_bf16(tw((size_t)cfg.mm_out*cfg.shuffled_dim()), nullptr,
                      cfg.mm_out, cfg.shuffled_dim(), Role::Gemm);

    if (!nn::Linear::bf16_keeps_raw()) {
        wbf.clear();
        wbf.shrink_to_fit();
    }

    return true;
}

// extract [n_patches, patch_dim] with per-patch order (ic, kh, kw), matching the conv
// weight reshape [hidden, ic*patch*patch + kh*patch + kw].
static void extract_patches(const float* pixels, int img, int patch, float* out) {
    const int grid = img/patch, pd = 3*patch*patch;
    for (int ph = 0; ph < grid; ph++)
        for (int pw = 0; pw < grid; pw++) {
            float* dst = out + ((size_t)(ph*grid+pw))*pd;
            for (int ic = 0; ic < 3; ic++)
                for (int kh = 0; kh < patch; kh++)
                    for (int kw = 0; kw < patch; kw++)
                        dst[ic*patch*patch + kh*patch + kw] =
                            pixels[((size_t)ic*img + (ph*patch+kh))*img + (pw*patch+kw)];
        }
}

void SiglipVision::encode(const float* pixels, float* out) const {
    encode(pixels, out, scratch);
}

// SMOLVLA_PROFILE_VIT=1: per-op attribution inside one encode() (stderr). The
// encoder-layer buckets (ln/qkv/attn/proj/mlp/res) come from nn::Prof; the
// non-layer stages are timed here.
namespace {
struct VitProf {
    bool on = std::getenv("SMOLVLA_PROFILE_VIT") != nullptr;
    double patches = 0, stem = 0, pos = 0, postln = 0, shuffle = 0, mmproj = 0, wall = 0;
    double t0 = 0;

    static double now_ms() {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    void tic() { if (on) t0 = now_ms(); }
    void toc(double& acc) { if (on) acc += now_ms()-t0; }
};
} // namespace

void SiglipVision::encode(const float* pixels, float* out, nn::Scratch& sc) const {
    const int H = cfg.hidden, NP = cfg.n_patches, PD = cfg.patch_dim();

    VitProf vp;
    nn::Prof lp;
    lp.on = vp.on;
    const double t_start = VitProf::now_ms();

    vp.tic();
    std::vector<float> patches((size_t)NP*PD);
    extract_patches(pixels, cfg.img, cfg.patch, patches.data());
    vp.toc(vp.patches);

    vp.tic();
    std::vector<float> h((size_t)NP*H);
    patch_lin.forward(h.data(), patches.data(), NP);
    vp.toc(vp.stem);

    vp.tic();
    for (size_t i = 0; i < (size_t)NP*H; i++) h[i] += pos_emb[i];   // position ids 0..NP-1
    vp.toc(vp.pos);

    // nullptr mask = full attention: SigLIP has no padding, and a 1024x1024 zero
    // array is 4 MB the dense kernel would otherwise stream once per head.
    for (int L = 0; L < cfg.n_layers; L++)
        layers[L].forward(h.data(), NP, nullptr, sc, vp.on ? &lp : nullptr);

    vp.tic();
    layernorm(h.data(), h.data(), post_ln_w, post_ln_b, NP, H, cfg.ln_eps);
    vp.toc(vp.postln);

    // pixel shuffle (scale_factor sf): 32x32 patches -> 8x8 tokens of dim H*sf*sf.
    // out_tok t=(h2*g2 + w2); its dim = concat over (dh,dw) in [0,sf) of patch[h2*sf+dh, w2*sf+dw].
    const int sf = cfg.scale_factor, side = (int)std::lround(std::sqrt((double)NP));
    const int g2 = side/sf, SD = cfg.shuffled_dim();
    vp.tic();
    std::vector<float> shuf((size_t)cfg.n_img_tok*SD);
    for (int h2 = 0; h2 < g2; h2++)
        for (int w2 = 0; w2 < g2; w2++) {
            float* dst = shuf.data() + ((size_t)(h2*g2+w2))*SD;
            for (int dh = 0; dh < sf; dh++)
                for (int dw = 0; dw < sf; dw++) {
                    int src_patch = (h2*sf+dh)*side + (w2*sf+dw);
                    std::memcpy(dst + (size_t)(dh*sf+dw)*H, h.data() + (size_t)src_patch*H, H*sizeof(float));
                }
        }
    vp.toc(vp.shuffle);

    vp.tic();
    mm_proj.forward(out, shuf.data(), cfg.n_img_tok);
    vp.toc(vp.mmproj);

    if (vp.on) {
        vp.wall = VitProf::now_ms() - t_start;
        const double layers_sum = lp.ln+lp.qkv+lp.attn+lp.proj+lp.mlp+lp.res;
        const double acc = vp.patches+vp.stem+vp.pos+layers_sum+vp.postln+vp.shuffle+vp.mmproj;
        std::fprintf(stderr,
            "  [vit] patches %6.1f  stem %6.1f  pos %5.1f | x%d: ln %6.1f  qkv %6.1f"
            "  attn %6.1f  proj %6.1f  mlp %6.1f  res %5.1f | postln %5.1f  shuffle %5.1f"
            "  mmproj %5.1f | other %5.1f | wall %6.1f ms\n",
            vp.patches, vp.stem, vp.pos, cfg.n_layers,
            lp.ln, lp.qkv, lp.attn, lp.proj, lp.mlp, lp.res,
            vp.postln, vp.shuffle, vp.mmproj, vp.wall-acc, vp.wall);
    }
}

} // namespace tcpu
