/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// C ABI for SmolvlaModel, for ctypes/FFI callers (e.g. a policy server that feeds
// raw camera frames). Built as the shared lib vla_simd_smolvla.
// Contract: include/vla_simd.h.
//
// The caller hands over native-resolution uint8 RGB frames; resize-with-pad to
// the checkpoint's side and the [0,1]->[-1,1] map happen here, in
// tcpu::resize_with_pad (src/preprocess/image.cpp).

#include <memory>
#include "vla_simd.h"
#include "models/smolvla/smolvla_model.h"
#include "preprocess/image.h"
#include "tokenizer/tokenizer.h"
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using tcpu::SmolvlaModel;
using tcpu::Tokenizer;

namespace {

struct Handle {
    SmolvlaModel m;
    Tokenizer    tok;
    int tok_maxlen = 48;
    int pad_id     = 2;
};

// Optional <model_dir>/config.txt keys the model loader itself does not read.
//
// Parsed a line at a time, splitting on the FIRST space. It used to read
// whitespace-separated token *pairs*, which desynchronizes on the very first
// line - `instruction <free text>` - for any instruction with an odd word count,
// after which no key ever matched and these settings silently kept their
// defaults. The Python readers (serve/, tools/) already split this way.
void read_config(const std::string& dir, Handle* hh) {
    std::ifstream f(dir + "/config.txt");
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        const std::string k = line.substr(0, sp);
        const std::string v = line.substr(sp + 1);
        try {
            if      (k == "tokenizer_max_length") hh->tok_maxlen = std::stoi(v);
            else if (k == "pad_token_id"        ) hh->pad_id     = std::stoi(v);
        } catch (...) {
            // a hand-edited non-numeric value keeps the default rather than
            // unwinding out of the load
        }
    }
    if (hh->tok_maxlen < 1) hh->tok_maxlen = 48;
}

} // namespace

extern "C" {

void* vla_smolvla_load(const char* model_dir, const char* tok_dir) try {
    if (!model_dir || !tok_dir) return nullptr;
    auto hh = std::make_unique<Handle>();
    if (!hh->m.load(model_dir) || !hh->tok.load(tok_dir)) return nullptr;
    read_config(model_dir, hh.get());
    return hh.release();
} catch (...) {
    return nullptr;
}

void vla_smolvla_free(void* h) { delete static_cast<Handle*>(h); }

int32_t vla_smolvla_chunk(void* h)      { return h ? static_cast<Handle*>(h)->m.aex.cfg.chunk : 0; }
int32_t vla_smolvla_action_dim(void* h) { return h ? static_cast<Handle*>(h)->m.real_action_dim : 0; }
int32_t vla_smolvla_state_dim(void* h)  { return h ? static_cast<Handle*>(h)->m.real_state_dim : 0; }
int32_t vla_smolvla_n_views(void* h)    { return h ? static_cast<Handle*>(h)->m.n_views : 0; }
int32_t vla_smolvla_img_size(void* h)   { return h ? static_cast<Handle*>(h)->m.vit.cfg.img : 0; }
int32_t vla_smolvla_tok_maxlen(void* h) { return h ? static_cast<Handle*>(h)->tok_maxlen : 0; }

int32_t vla_smolvla_tokenize(void* h, const char* text, int32_t* ids, int32_t* mask) try {
    if (!h || !text || !ids || !mask) return VLA_ERR_ARG;
    auto* hh = static_cast<Handle*>(h);
    // lerobot's pipeline appends a newline before tokenizing. Doing it here
    // rather than in every caller means no entry point can get it wrong; the
    // rstrip makes it idempotent for callers that already added one.
    std::string s(text);
    if (s.size() > 65536) s.erase(0, s.size() - 65536);
    while (!s.empty() && s.back() == '\n') s.pop_back();
    s.push_back('\n');

    auto enc = hh->tok.encode(s, false);
    if ((int)enc.size() > hh->tok_maxlen) enc.erase(enc.begin(), enc.end() - hh->tok_maxlen);
    for (int i = 0; i < hh->tok_maxlen; i++) {
        ids[i]  = i < (int)enc.size() ? enc[i] : hh->pad_id;
        mask[i] = i < (int)enc.size() ? 1 : 0;
    }
    return (int32_t)enc.size();
} catch (...) {
    return VLA_ERR_EXCEPTION;
}

int32_t vla_smolvla_predict(void* h, const uint8_t* frames, int32_t n_views,
                            int32_t height, int32_t width,
                            const int32_t* lang_tokens, const int32_t* lang_mask, int32_t n_lang,
                            const float* state, const float* noise, uint64_t seed,
                            float* actions) try {
    if (!h || !frames || !lang_tokens || !lang_mask || !state || !actions) return VLA_ERR_ARG;
    // A camera driver that fails can hand back a (0,0,3) frame; resize_with_pad
    // would then divide by zero and the model would infer on a black image.
    if (height <= 0 || width <= 0 || n_lang < 0) return VLA_ERR_ARG;

    auto* hh = static_cast<Handle*>(h);
    const SmolvlaModel& m = hh->m;
    if (n_views != m.n_views) return VLA_ERR_SHAPE;

    const int S = m.vit.cfg.img, C = m.aex.cfg.chunk, MAD = m.aex.cfg.max_action_dim;
    const size_t per_view = (size_t)3*S*S;

    std::vector<float> pixels((size_t)n_views*per_view);
    for (int v = 0; v < n_views; v++)
        tcpu::resize_with_pad(frames + (size_t)v*height*width*3, height, width, S,
                              pixels.data() + (size_t)v*per_view);

    std::vector<float> noise_buf;
    if (!noise) {
        noise_buf.resize((size_t)C*MAD);
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& x : noise_buf) x = nd(rng);
        noise = noise_buf.data();
    }

    auto out = m.predict(pixels.data(), n_views, lang_tokens, lang_mask, n_lang, state, noise);
    if (out.empty()) return VLA_ERR_SHAPE;

    const int adim = m.real_action_dim;
    for (int t = 0; t < C; t++)
        std::memcpy(actions + (size_t)t*adim, out.data() + (size_t)t*MAD, adim*sizeof(float));
    return VLA_OK;
} catch (...) {
    return VLA_ERR_EXCEPTION;
}

} // extern "C"
