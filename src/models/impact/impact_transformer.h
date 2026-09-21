/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "config.h"
#include "nn/attention.h"
#include "nn/linear.h"
#include <string>
#include <vector>

// IMPACT transformer (DETR-style, post-norm): token assembly + encoder + decoder +
// action head. ACT's transformer with a text tail. Token sequence, per camera
// feature map of fh*fw tokens:
//   [latent(1) | state(1) | cam0(fh*fw) | cam1(fh*fw) ... | text(n_text)]
//
// Text goes last so the visual block layout is byte-identical to ACT's - the 2D
// sinusoid is built the same way over the same offsets - and so the padding mask
// is a contiguous tail.
//
// As in ACT the positional embedding is added to the attention queries and keys,
// never to the values or to the tokens themselves; the text tokens obey that too,
// which is why their learned table arrives as `text_pos` for the pos array rather
// than pre-added by the text tower. Decoder queries start at zero and are
// identified only by their learned position embedding.
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
//
// Weights: impact.meta/impact.bin from tools/impact/convert_impact.py.

namespace tcpu {

struct ImpactEncoderLayer {
    nn::MhaQKV attn;
    nn::Linear w1, w2;
    const float *n1s = nullptr, *n1b = nullptr;
    const float *n2s = nullptr, *n2b = nullptr;
};

struct ImpactDecoderLayer {
    nn::MhaQKV self, cross;
    nn::Linear w1, w2;
    const float *n1s = nullptr, *n1b = nullptr;
    const float *n2s = nullptr, *n2b = nullptr;
    const float *n3s = nullptr, *n3b = nullptr;
};

struct ImpactTransformer {
    ImpactConfig cfg;
    std::vector<float> data;
    nn::Linear img_proj;      // 1x1 conv over the feature map == linear over tokens
    nn::Linear state_proj;
    nn::Linear head;          // [action_dim, dim]
    const float* latent_tok = nullptr;   // [dim] latent proj of zeros == its bias
    const float* pos1d = nullptr;        // [n_1d, dim]
    const float* dec_pos = nullptr;      // [chunk, dim]
    const float *dec_ns = nullptr, *dec_nb = nullptr;
    std::vector<ImpactEncoderLayer> enc;
    std::vector<ImpactDecoderLayer> dec;

    mutable nn::Scratch scratch;                     // attention buffers
    mutable std::vector<float> xq, kpos, ff, resid;  // block scratch
    mutable std::vector<float> dec_x;                // decoder queries, reused per call
    mutable std::vector<float> tokens, pos;          // token/pos arena, reused per call

    // The camera position embedding depends only on (fh, fw, dim), so it is the
    // same tensor on every frame a given camera produces. Built once and reused.
    mutable std::vector<float> cam_pos;
    mutable int cam_pos_fh = -1, cam_pos_fw = -1;

    bool load(const std::string& dir);

    // n_text_real of the cfg.n_text text slots carry a real token; the padded tail
    // is not emitted, so the sequence length depends on the instruction.
    int n_tokens(int n_cams, int fh, int fw, int n_text_real) const {
        return cfg.n_1d + n_cams*fh*fw + clamp_text(n_text_real);
    }

    int clamp_text(int n_text_real) const {
        return n_text_real < 0 ? 0 : n_text_real > cfg.n_text ? cfg.n_text : n_text_real;
    }

    // 2D sinusoidal camera position embedding (ACTSinusoidalPositionEmbedding2d):
    // out [fh*fw, dim], the y half followed by the x half.
    static void sinusoid_pos_2d(int fh, int fw, int dim, float* out);

    // feats: n_cams pointers to [fh*fw, dim] backbone feature maps (NHWC tokens);
    // state_norm [state_dim]; text [n_text, dim] projected text tokens, of which
    // the first n_text_real are real and the rest are dropped; text_pos [n_text,
    // dim] their learned position -> tokens / pos [n_tokens(...), dim].
    void build_tokens(const float* const* feats, int n_cams, int fh, int fw,
                      const float* state_norm, const float* text, const float* text_pos,
                      int n_text_real, float* out_tokens, float* out_pos) const;

    // x [T, dim] in place through the encoder blocks.
    void encode(float* x, const float* tok_pos, int T) const;

    // encoder output + its pos -> actions_norm [chunk, action_dim]. dec_out
    // (optional) receives the [chunk, dim] decoder output after its final norm.
    void decode(const float* enc_out, const float* tok_pos, int T,
                float* actions_norm, float* dec_out = nullptr) const;

    // build_tokens + encode + decode, using the model's own scratch.
    void forward(const float* const* feats, int n_cams, int fh, int fw,
                 const float* state_norm, const float* text, const float* text_pos,
                 int n_real_text, float* actions_norm) const;
};

} // namespace tcpu
