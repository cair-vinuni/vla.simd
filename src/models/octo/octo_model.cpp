/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "octo_model.h"
#include "models/arena.h"
#include "hal/common/env.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

namespace tcpu {

// OCTO_PROFILE=1 prints per-module latency
struct Prof {
    bool on;
    std::chrono::steady_clock::time_point t;
    Prof() : on(std::getenv("OCTO_PROFILE") != nullptr), t(std::chrono::steady_clock::now()) {}
    void tick(const char* name) {
        if (!on) return;
        auto now = std::chrono::steady_clock::now();
        std::printf("  [octo] %-12s %7.1f ms\n", name,
                    std::chrono::duration<double, std::milli>(now-t).count());
        t = now;
    }
};

// OCTO_INT8 -> route matmul weights through the symmetric W8A8 kernel
// (ops/quant_ops.h). A bitmask, so a group can be A/B'd against fp32 on its own
// and each carries its own accuracy argument:
//
//   1  transformer attention projections (q/k/v/o, all layers)
//   2  transformer MLP w1
//   4  transformer MLP w2
//   8  token projections (task/primary/wrist) + both stem embeds
//   16 stem convolutions (both towers, stem conv included)
//   32 diffusion score net (in/out projections, residual blocks)
//   63 = all of it
//
// The groups mirror SMOLVLA_INT8's layout so the W8A8 ablation reads across
// models, with two deliberate differences.
//
// **The T5 encoder is not a group.** nn::T5Encoder holds raw weight pointers
// rather than nn::Linear objects, so there is nothing here to call init_int8()
// on - but the reason not to add it is stronger than the inconvenience: the T5
// output is a pure function of the instruction string and lang_encode() caches
// it, so in a control loop the text tower runs once per episode and never again.
// Quantizing it would trade accuracy for a saving that is amortized to nothing.
//
// **The diffusion head is one group, not three.** Its score net runs `steps`
// times per query - the only module in Octo on the inner loop of a loop - so the
// interesting question is whether it survives quantization at all, not which of
// its projections costs most. If bit 32 turns out lossy, split it then.
namespace {
enum : int { I8_TF_ATTN = 1, I8_TF_W1 = 2, I8_TF_W2 = 4, I8_PROJ = 8,
             I8_STEM_CONV = 16, I8_HEAD = 32 };
} // namespace

bool OctoModel::load(const std::string& dir, const std::string& tok_dir) {
    if (!tok.load(tok_dir)) return false;
    if (!t5.load(dir)) return false;
    if (!stem_primary.load(dir, "stem_primary")) return false;
    if (!stem_wrist.load(dir, "stem_wrist")) return false;
    if (!tf.load(dir)) return false;
    if (!head.load(dir)) return false;

    auto grid = [](const SmallStemConfig& c, int s) {
        for (int i=0; i<c.n_layers; i++) {
            s = (s+2*c.pad-c.k)/c.stride+1;
            if (s < 1) return -1;
        }
        return s*s;
    };
    if (t5.cfg.d_model != tf.cfg.t5_dim || head.cfg.emb != tf.cfg.d || tf.cfg.heads*tf.cfg.head_dim != tf.cfg.d ||
        stem_primary.cfg.embed_dim != tf.cfg.stem_dim || stem_wrist.cfg.embed_dim != tf.cfg.stem_dim ||
        grid(stem_primary.cfg, 256) != tf.cfg.tok_primary || grid(stem_wrist.cfg, 128) != tf.cfg.tok_wrist)
        return false;

    const size_t AD = head.cfg.action_dim;
    if (!read_floats(dir + "/stats_action_mean.bin", act_mean, AD)) return false;
    if (!read_floats(dir + "/stats_action_std.bin",  act_std,  AD)) return false;
    if (!read_floats(dir + "/stats_action_mask.bin", act_mask, AD)) return false;

    apply_int8();
    return true;
}

// Quantize the groups OCTO_INT8 selects. Runs after every sub-module has loaded,
// so one place decides and one line reports it. Layers whose shape the kernel
// cannot take (N % 16 != 0) stay fp32 silently, and on a CPU with no int8 kernel
// every call returns false - so a caller may set the mask unconditionally.
void OctoModel::apply_int8() {
    static const int mask = hal::env::int_env("OCTO_INT8", 0);
    if (!mask) return;

    int n = 0;
    for (nn::EncoderLayer& l : tf.layers) {
        if (mask & I8_TF_ATTN) {
            n += l.attn.wq.init_int8();
            n += l.attn.wk.init_int8();
            n += l.attn.wv.init_int8();
            n += l.attn.wo.init_int8();
        }
        if (mask & I8_TF_W1) n += l.w1.init_int8();
        if (mask & I8_TF_W2) n += l.w2.init_int8();
    }
    if (mask & I8_PROJ) {
        n += tf.proj_task.init_int8();
        n += tf.proj_prim.init_int8();
        n += tf.proj_wrist.init_int8();
        n += stem_primary.embed.init_int8();
        n += stem_wrist.embed.init_int8();
    }
    if (mask & I8_STEM_CONV) {
        for (SmallStem* st : {&stem_primary, &stem_wrist})
            for (StemLayer& l : st->layers)
                n += l.conv.init_int8();
    }
    if (mask & I8_HEAD) {
        n += head.net.in_proj.init_int8();
        n += head.net.out_proj.init_int8();
        for (nn::DiffusionMlp::Block& b : head.net.blocks) {
            n += b.d0.init_int8();
            n += b.d1.init_int8();
        }
    }
    std::fprintf(stderr, "[octo] int8 GEMMs: %d (OCTO_INT8=%d)%s\n", n, mask,
                 n ? "" : " - no int8 kernel on this CPU, staying fp32");
}

// T5 output is a static factor of the instruction: encode once, then fetch from RAM.
const float* OctoModel::lang_encode(const std::string& instruction) const {
    auto it = lang_cache.find(instruction);
    if (it != lang_cache.end()) return it->second.data();

    const int NT = tf.cfg.n_task;
    std::vector<int> ids = tok.encode(instruction, NT);
    std::vector<int> am(NT);
    for (int i=0; i<NT; i++)
        am[i] = ids[i] != tok.pad_id;

    std::vector<float> out((size_t)NT*t5.cfg.d_model);
    t5.encode(ids.data(), am.data(), NT, out.data());
    if (lang_cache.size() >= 64) lang_cache.clear();
    return lang_cache.emplace(instruction, std::move(out)).first->second.data();
}

void OctoModel::predict(const uint8_t* primary, const uint8_t* wrist, int wnd,
                        const uint8_t* timestep_mask, const std::string& instruction,
                        const float* noise, const float* z, uint64_t seed,
                        bool unnormalize, float* actions) const {
    const int D = tf.cfg.d;
    Prof prof;

    // vision (goal images absent -> zeros, language-conditioned)
    const size_t PB = (size_t)256*256*3, WB = (size_t)128*128*3;
    const size_t SP = (size_t)tf.cfg.tok_primary*tf.cfg.stem_dim, SW = (size_t)tf.cfg.tok_wrist*tf.cfg.stem_dim;
    std::vector<float> sp((size_t)wnd*SP), sw(wrist ? (size_t)wnd*SW : 0);
    auto stems = [wnd](const SmallStem& st, const uint8_t* img, int hw, size_t nb, size_t ns, float* out,
                       const std::vector<uint8_t>& prev, const std::vector<float>& prev_out) {
        for (int t=0; t<wnd; t++) {
            const uint8_t* f = img+(size_t)t*nb;
            const float* hit = nullptr;
            for (int u=0; u<t && !hit; u++)
                if (!std::memcmp(f, img+(size_t)u*nb, nb)) hit = out+(size_t)u*ns;
            for (size_t u=0; u<prev.size()/nb && !hit; u++)
                if (!std::memcmp(f, prev.data()+u*nb, nb)) hit = prev_out.data()+u*ns;
            if (hit) std::memcpy(out+(size_t)t*ns, hit, ns*sizeof(float));
            else     st.encode(f, hw, hw, out+(size_t)t*ns);
        }
    };
    stems(stem_primary, primary, 256, PB, SP, sp.data(), win_p, win_sp);
    if (wrist) stems(stem_wrist, wrist, 128, WB, SW, sw.data(), win_w, win_sw);
    win_p.clear();
    win_w.clear();
    win_sp.swap(sp);
    win_sw.swap(sw);
    win_p.assign(primary, primary+(size_t)wnd*PB);
    if (wrist) win_w.assign(wrist, wrist+(size_t)wnd*WB);
    prof.tick("stems");

    const float* t5_out = lang_encode(instruction);
    prof.tick("t5");

    // transformer -> readout embedding of the last timestep (fast path: the final
    // layer only computes the readout row; the rest of `out` is unused here)
    std::vector<float> out((size_t)tf.total_tokens(wnd)*D);
    tf.forward(t5_out, win_sp.data(), wrist ? win_sw.data() : nullptr, wnd, timestep_mask,
               out.data());
    prof.tick("transformer");
    const float* emb = out.data()+(size_t)tf.readout_index(wnd, wnd-1)*D;

    // DDPM
    const int FLAT = head.cfg.flat();
    std::vector<float> nbuf, zbuf;
    if (!noise || !z) {
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> N(0.0f, 1.0f);
        nbuf.resize(FLAT);
        zbuf.resize((size_t)head.cfg.steps*FLAT);
        for (auto& v : nbuf) v = N(rng);
        for (auto& v : zbuf) v = N(rng);
        noise = nbuf.data();
        z     = zbuf.data();
    }
    head.denoise(emb, noise, z, actions);
    prof.tick("head");

    if (unnormalize)
        for (int h=0; h<head.cfg.horizon; h++)
            for (int a=0; a<head.cfg.action_dim; a++) {
                float* v = actions+(size_t)h*head.cfg.action_dim+a;
                if (act_mask[a] > 0.0f) *v = *v*act_std[a]+act_mean[a];
            }
}

} // namespace tcpu
