/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "smolvla_model.h"
#include "hal/common/env.h"
#include "ops/lm_ops.h"
#include <chrono>
#include <exception>
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>

namespace tcpu {

// SMOLVLA_INT8 -> route matmul weights through the symmetric W8A8 kernel
// (ops/quant_ops.h). A bitmask, so a group can be A/B'd against fp32 on its own:
//
//   1  vision attention projections (q/k/v/o, 12 layers)
//   2  vision MLP fc1
//   4  vision MLP fc2
//   8  vision patch embed + the connector's modality projection
//   16 VLM text tower (SmolLM2 prefix)
//   32 action expert (the denoise loop)
//   63 = all of it
//
// The vision tower is 88% of a query on a Pi 5 and 77% of the tower is these
// GEMMs, so 7 is where nearly all of the win is. Measured on a Pi 5 against
// tools/ref_smolvla.py, RMS over 8 observations, in the arm's own units
// (mean |action| 64.6 deg): 8 -> 0.091, 32 -> 0.104, 16 -> 0.148, 1 -> 0.227,
// 2 -> 0.238, 4 -> 0.907. **fc2 (bit 4) carries most of the error** - its input
// is the GELU output, whose long positive tail drags the per-token absmax and
// coarsens the rest of the row - so 59 (everything but fc2) is 0.69% of the
// arm's range where 63 is 1.64%, for about three quarters of the saving.
// Unlike ACT's decoder, the action expert quantizes cleanly and is worth having.
//
// Left out entirely: attention itself (scores and A*V are activation x
// activation, a different quantization problem), state_proj and the flow
// matching in/out projections (K and N of 32 or less, far too small to pay for
// quantizing, and action_out_proj writes the joint commands).
namespace {
enum : int { I8_VIT_ATTN = 1, I8_VIT_W1 = 2, I8_VIT_W2 = 4, I8_VIT_STEM = 8,
             I8_LM = 16, I8_EXPERT = 32 };

int int8_mask() {
    static const int v = [] {
        const char* e = std::getenv("SMOLVLA_INT8");
        return e ? std::atoi(e) : 0;
    }();
    return v;
}
} // namespace

static bool read_floats(const std::string& path, std::vector<float>& out, size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "smolvla: cannot open %s\n", path.c_str()); return false; }
    out.resize(n);
    f.read(reinterpret_cast<char*>(out.data()), n * sizeof(float));
    return (bool)f;
}

bool SmolvlaModel::load(const std::string& dir) {
    if (!vit.load(dir) || !vlm.load(dir) || !aex.load(dir)) return false;

    std::ifstream meta(dir + "/heads.meta");
    if (!meta) { std::fprintf(stderr, "smolvla: cannot open %s/heads.meta\n", dir.c_str()); return false; }
    std::string k; float v;
    while (meta >> k >> v) {
        if (k == "vocab") vocab = (int)v;
        else if (k == "hidden") hidden = (int)v;
        else if (k == "max_state_dim") max_state_dim = (int)v;
        else if (k == "real_state_dim") real_state_dim = (int)v;
        else if (k == "real_action_dim") real_action_dim = (int)v;
        else if (k == "n_views") n_views = (int)v;
    }
    // A renamed key or a non-numeric value ends the loop early and leaves these
    // at 0; every read below then asks for 0 bytes, succeeds, and returns a live
    // handle whose first predict indexes an empty embedding table.
    if (vocab <= 0 || hidden <= 0 || max_state_dim <= 0 || n_views <= 0 ||
        real_state_dim <= 0 || real_state_dim > max_state_dim ||
        real_action_dim <= 0 || real_action_dim > aex.cfg.max_action_dim ||
        hidden != vlm.cfg.hidden || hidden != vit.cfg.mm_out ||
        aex.cfg.kv_full() != vlm.cfg.kv_full() || aex.cfg.n_layers > vlm.cfg.n_layers) {
        std::fprintf(stderr, "smolvla: %s/heads.meta is missing or malformed "
                     "(vocab=%d hidden=%d state=%d/%d action=%d/%d views=%d)\n",
                     dir.c_str(), vocab, hidden, real_state_dim, max_state_dim,
                     real_action_dim, aex.cfg.max_action_dim, n_views);
        return false;
    }

    { std::ifstream f(dir + "/emb.bin", std::ios::binary);
      if (!f) { std::fprintf(stderr, "smolvla: cannot open %s/emb.bin\n", dir.c_str()); return false; }
      emb.resize((size_t)vocab * hidden);
      f.read(reinterpret_cast<char*>(emb.data()), emb.size() * sizeof(uint16_t));
      if (!f) return false; }
    { std::ifstream f(dir + "/heads.bin", std::ios::binary);
      if (!f) { std::fprintf(stderr, "smolvla: cannot open %s/heads.bin\n", dir.c_str()); return false; }
      state_w.resize((size_t)hidden * max_state_dim); state_b.resize(hidden);
      f.read(reinterpret_cast<char*>(state_w.data()), state_w.size() * sizeof(float));
      f.read(reinterpret_cast<char*>(state_b.data()), state_b.size() * sizeof(float));
      if (!f) return false; }

    // Unchecked, these fail two ways and both reach the robot: a missing file
    // leaves the vector empty and predict indexes it; a truncated one leaves it
    // zero and the state is scaled by 1/norm_eps = 1e8.
    if (!read_floats(dir + "/stats_state_mean.bin",  state_mean,  real_state_dim)  ||
        !read_floats(dir + "/stats_state_std.bin",   state_std,   real_state_dim)  ||
        !read_floats(dir + "/stats_action_mean.bin", action_mean, real_action_dim) ||
        !read_floats(dir + "/stats_action_std.bin",  action_std,  real_action_dim))
        return false;

    apply_int8();
    return true;
}

// Quantize the groups SMOLVLA_INT8 selects. Runs after every sub-module has
// loaded, so one place decides and one line reports it. Layers whose shape the
// kernel cannot take (N % 16 != 0) stay fp32 silently, and on a CPU with no int8
// kernel every call returns false - so a caller may set the mask unconditionally.
void SmolvlaModel::apply_int8() {
    const int mask = int8_mask();
    if (!mask) return;

    int n = 0;
    for (nn::EncoderLayer& l : vit.layers) {
        if (mask & I8_VIT_ATTN) {
            n += l.attn.wq.init_int8();
            n += l.attn.wk.init_int8();
            n += l.attn.wv.init_int8();
            n += l.attn.wo.init_int8();
        }
        if (mask & I8_VIT_W1) n += l.w1.init_int8();
        if (mask & I8_VIT_W2) n += l.w2.init_int8();
    }
    if (mask & I8_VIT_STEM) {
        n += vit.patch_lin.init_int8();
        n += vit.mm_proj.init_int8();
    }
    if (mask & I8_LM) {
        for (VlmLayerW& w : vlm.layers) {
            n += w.q.init_int8();
            n += w.k.init_int8();
            n += w.v.init_int8();
            n += w.o.init_int8();
            n += w.gate.init_int8();
            n += w.up.init_int8();
            n += w.down.init_int8();
        }
    }
    if (mask & I8_EXPERT) {
        for (ExpertLayerW& w : aex.layers) {
            n += w.q.init_int8();
            n += w.k.init_int8();
            n += w.v.init_int8();
            n += w.o.init_int8();
            n += w.gate.init_int8();
            n += w.up.init_int8();
            n += w.down.init_int8();
        }
    }
    std::fprintf(stderr, "[smolvla] int8 GEMMs: %d (SMOLVLA_INT8=%d)%s\n", n, mask,
                 n ? "" : " - no int8 kernel on this CPU, staying fp32");
}

std::vector<float> SmolvlaModel::predict_normalized(
        const float* pixels_all, int nv,
        const int32_t* lang_tokens, const int32_t* lang_mask, int n_lang,
        const float* state, const float* noise) const {
    const int H = hidden, TOK = vit.cfg.n_img_tok;
    const int n_img = nv * TOK, n_prefix = n_img + n_lang + 1;
    const int C = aex.cfg.chunk, MAD = aex.cfg.max_action_dim;
    const float sq = std::sqrt((float)H);
    const float BIG_NEG = std::numeric_limits<float>::lowest();
    const size_t per_view = (size_t)3 * vit.cfg.img * vit.cfg.img;

    // Token ids cross the C ABI raw and index the embedding table below. A
    // tokenizer directory whose vocab is wider than this checkpoint's reads past
    // it and returns plausible, wrong actions, so refuse rather than clamp.
    for (int t = 0; t < n_lang; t++) {
        if (lang_tokens[t] < 0 || lang_tokens[t] >= vocab) {
            std::fprintf(stderr, "smolvla: token id %d at %d is outside the "
                         "checkpoint's vocab of %d\n", lang_tokens[t], t, vocab);
            return {};
        }
    }

    const bool prof = std::getenv("SMOLVLA_PROFILE") != nullptr;
    using clk = std::chrono::high_resolution_clock;
    auto tic = clk::now();
    auto lap = [&](const char* name) {
        if (!prof) return;
        auto now = clk::now();
        std::fprintf(stderr, "  [profile] %-10s %7.1f ms\n", name,
                     std::chrono::duration<double, std::milli>(now - tic).count());
        tic = now;
    };

    // ---- assemble prefix embeddings [n_prefix, H] ----
    // encode() writes [n_img_tok, mm_out] = exactly this view's prefix slice, so
    // the views land in place and the sqrt(H) scaling is one pass afterwards.
    std::vector<float> prefix((size_t)n_prefix * H);
    const int vt = hal::env::view_threads();
    if (vt > 0 && nv > 1) {
        // Views are independent: run them concurrently on vt threads each. Each
        // worker needs its own scratch (the member one is shared). Same math.
        std::vector<nn::Scratch> vsc(nv);
        std::vector<std::exception_ptr> err(nv);
        std::vector<std::thread> workers;
        workers.reserve(nv);
        // An exception out of a thread body is std::terminate, and so is unwinding
        // past a joinable thread: catch per worker, join, then rethrow.
        try {
            for (int v = 0; v < nv; v++) {
                workers.emplace_back([&, v] {
#if defined(_OPENMP)
                    omp_set_num_threads(vt);
#endif
                    try {
                        vit.encode(pixels_all + (size_t)v * per_view,
                                   prefix.data() + (size_t)v * TOK * H, vsc[v]);
                    } catch (...) { err[v] = std::current_exception(); }
                });
            }
        } catch (...) {
            for (std::thread& w : workers) w.join();
            throw;
        }
        for (std::thread& w : workers) w.join();
        for (std::exception_ptr& e : err) if (e) std::rethrow_exception(e);
    } else {
        for (int v = 0; v < nv; v++)
            vit.encode(pixels_all + (size_t)v * per_view, prefix.data() + (size_t)v * TOK * H);
    }
    for (size_t i = 0; i < (size_t)n_img * H; i++)
        prefix[i] *= sq;
    lap("vision");
    for (int t = 0; t < n_lang; t++) {
        const uint16_t* e = emb.data() + (size_t)lang_tokens[t] * H;   // bf16 token embedding
        float* dst = prefix.data() + (size_t)(n_img + t) * H;
        for (int j = 0; j < H; j++) {
            uint32_t u = (uint32_t)e[j] << 16; float f; std::memcpy(&f, &u, 4);
            dst[j] = f * sq;
        }
    }
    dense_linear(prefix.data() + (size_t)(n_prefix - 1) * H, state, state_w.data(), state_b.data(),
                 1, H, max_state_dim);

    // ---- pad + prefix-LM mask + positions ----
    std::vector<int> pad(n_prefix);
    for (int j = 0; j < n_prefix; j++)
        pad[j] = (j < n_img) ? 1 : (j < n_img + n_lang) ? lang_mask[j - n_img] : 1;
    const int state_idx = n_prefix - 1;
    auto att = [&](int i) { return i >= state_idx ? 1 : 0; };   // cumsum: only state has att=1 (last)

    std::vector<float> pmask((size_t)n_prefix * n_prefix);
    for (int i = 0; i < n_prefix; i++)
        for (int j = 0; j < n_prefix; j++)
            pmask[(size_t)i * n_prefix + j] =
                (att(j) <= att(i) && pad[i] && pad[j]) ? 0.0f : BIG_NEG;

    std::vector<int> pos(n_prefix);
    { int c = 0; for (int i = 0; i < n_prefix; i++) { c += pad[i]; pos[i] = c - 1; } }

    std::vector<float> vlm_out((size_t)n_prefix * H);
    std::vector<VlmKV> kv;
    vlm.prefix_forward(prefix.data(), pmask.data(), pos.data(), n_prefix, vlm_out.data(), kv);
    lap("vlm_prefill");

    // ---- denoise mask + positions ----
    const int SK = n_prefix + C;
    int prefix_offset = 0; for (int j = 0; j < n_prefix; j++) prefix_offset += pad[j];
    std::vector<float> dmask((size_t)C * SK);
    std::vector<int> pos_full(C);
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < n_prefix; j++) dmask[(size_t)i * SK + j] = pad[j] ? 0.0f : BIG_NEG;
        for (int j = 0; j < C; j++) dmask[(size_t)i * SK + n_prefix + j] = (j <= i) ? 0.0f : BIG_NEG;
        pos_full[i] = prefix_offset + i;
    }

    std::vector<float> actions((size_t)C * MAD);
    aex.denoise(kv, n_prefix, noise, dmask.data(), pos_full.data(), actions.data());
    lap("denoise");
    return actions;
}

std::vector<float> SmolvlaModel::predict(const float* pixels_all, int nv,
                                         const int32_t* lang_tokens, const int32_t* lang_mask, int n_lang,
                                         const float* raw_state, const float* noise) const {
    const int C = aex.cfg.chunk, MAD = aex.cfg.max_action_dim;

    // normalize state (first real_state_dim dims), pad rest with 0
    std::vector<float> state(max_state_dim, 0.0f);
    for (int i = 0; i < real_state_dim; i++)
        state[i] = (raw_state[i] - state_mean[i]) / (state_std[i] + norm_eps);

    std::vector<float> noise_buf;
    if (!noise) {
        noise_buf.resize((size_t)C * MAD);
        std::mt19937 rng(0);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& x : noise_buf) x = nd(rng);
        noise = noise_buf.data();
    }

    std::vector<float> a = predict_normalized(pixels_all, nv, lang_tokens, lang_mask, n_lang, state.data(), noise);
    if (a.empty()) return a;   // rejected input; the C ABI turns this into an error

    // un-normalize actions (first real_action_dim dims of each chunk row)
    for (int r = 0; r < C; r++)
        for (int j = 0; j < real_action_dim; j++)
            a[(size_t)r * MAD + j] = a[(size_t)r * MAD + j] * (action_std[j] + norm_eps) + action_mean[j];
    return a;
}

} // namespace tcpu
