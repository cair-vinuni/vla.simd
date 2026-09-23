/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "config.h"
#include "nn/attention.h"
#include "nn/linear.h"
#include <string>
#include <vector>

// ACT transformer (DETR-style, post-norm): token assembly + encoder + decoder +
// action head. Token sequence, per camera feature map of fh*fw tokens:
//   [latent(1) | state(1) | cam0(fh*fw) | cam1(fh*fw) ... | text(n_text)]
// The positional embedding is added to the attention queries and keys, never to
// the values or to the tokens themselves. Decoder queries start at zero and are
// identified only by their learned position embedding.
// Weights: act.meta/act.bin from tools/convert_act.py.
//
// The padded text slots hold real T5 outputs, not zeros, so they must never be
// attended to. They used to be emitted and then blocked as keys in two places -
// the encoder self-attention and the decoder's cross-attention over the memory -
// with an additive [T,T] mask. They are now simply not emitted: the sequence ends
// at the last real text token. That enforces the same invariant structurally
// (there is no padded column left to forget to mask), and it takes the attention
// ops' dense path, which on the NEON backend streams each head's K^T/V panels
// once per 16 queries instead of once per 4. Measured 75 ms of 1202 on a Pi 5 for
// the campaign's 9-token instruction.

namespace tcpu {

struct ActEncoderLayer {
    nn::MhaQKV attn;
    nn::Linear w1, w2;
    const float *n1s = nullptr, *n1b = nullptr;
    const float *n2s = nullptr, *n2b = nullptr;
};

struct ActDecoderLayer {
    nn::MhaQKV self, cross;
    nn::Linear w1, w2;
    const float *n1s = nullptr, *n1b = nullptr;
    const float *n2s = nullptr, *n2b = nullptr;
    const float *n3s = nullptr, *n3b = nullptr;
};

struct ActTransformer {
    ActConfig cfg;
    std::vector<float> data;
    nn::Linear img_proj;      // 1x1 conv over the feature map == linear over tokens
    nn::Linear state_proj;
    nn::Linear head;          // [action_dim, dim]
    const float* latent_tok = nullptr;   // [dim] latent proj of zeros == its bias
    const float* pos1d = nullptr;        // [n_1d, dim]
    const float* dec_pos = nullptr;      // [chunk, dim]
    const float *dec_ns = nullptr, *dec_nb = nullptr;
    std::vector<ActEncoderLayer> enc;
    std::vector<ActDecoderLayer> dec;

    mutable nn::Scratch scratch;                     // attention buffers
    mutable std::vector<float> xq, kpos, ff, resid;  // block scratch
    mutable std::vector<float> dec_x;                // decoder queries, reused per call
    mutable std::vector<float> tokens, pos;          // token/pos arena, reused per call

    // The camera position embedding depends only on (fh, fw, dim), so it is the
    // same tensor on every frame a given camera produces: 150k sin/cos in a
    // single-threaded loop, recomputed per control step for nothing. Built once
    // and reused; cam_pos_fh/fw record what it was built for.
    mutable std::vector<float> cam_pos;
    mutable int cam_pos_fh = -1, cam_pos_fw = -1;

    const char* tag = "act";
    int prof = 0;

    bool load(const std::string& dir, int img_ch, int int8_mask, bool int8_state = false);

    int clamp_text(int n) const { return n < 0 ? 0 : n > cfg.n_text ? cfg.n_text : n; }

    // n_text_real of the cfg.n_text text slots carry a real token; the padded tail
    // is not emitted, so the sequence length depends on the instruction.
    int n_tokens(int n_cams, int fh, int fw, int n_text_real) const {
        return cfg.n_1d + n_cams*fh*fw + clamp_text(n_text_real);
    }

    // 2D sinusoidal camera position embedding (ACTSinusoidalPositionEmbedding2d):
    // out [fh*fw, dim], the y half followed by the x half.
    static void sinusoid_pos_2d(int fh, int fw, int dim, float* out);

    // feats: n_cams pointers to [fh*fw, dim] backbone feature maps (NHWC tokens);
    // state_norm [state_dim] -> tokens / pos [n_tokens(...), dim].
    // out_tokens/out_pos are named apart from the same-purpose members below,
    // which is what predict() passes in. text [n_text, dim] projected text tokens,
    // of which the first n_text_real are real and the rest are dropped; text_pos
    // [n_text, dim] their learned position.
    void build_tokens(const float* const* feats, int n_cams, int fh, int fw,
                      const float* state_norm, float* out_tokens, float* out_pos,
                      const float* text, const float* text_pos, int n_text_real) const;

    // x [T, dim] in place through the encoder blocks.
    void encode(float* x, const float* tok_pos, int T) const;

    // encoder output + its pos -> actions_norm [chunk, action_dim].
    void decode(const float* enc_out, const float* tok_pos, int T, float* actions_norm) const;

    // build_tokens + encode + decode, using the model's own scratch.
    void forward(const float* const* feats, int n_cams, int fh, int fw,
                 const float* state_norm, float* actions_norm,
                 const float* text = nullptr, const float* text_pos = nullptr,
                 int n_text_real = 0) const;
};

} // namespace tcpu
