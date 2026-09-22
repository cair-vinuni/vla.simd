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

// ACT transformer (DETR-style, post-norm): token assembly + encoder + decoder +
// action head. Token sequence, per camera feature map of fh*fw tokens:
//   [latent(1) | state(1) | cam0(fh*fw) | cam1(fh*fw) ...]
// The positional embedding is added to the attention queries and keys, never to
// the values or to the tokens themselves. Decoder queries start at zero and are
// identified only by their learned position embedding.
// Weights: act.meta/act.bin from tools/convert_act.py.

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

    bool load(const std::string& dir);

    int n_tokens(int n_cams, int fh, int fw) const { return cfg.n_1d + n_cams*fh*fw; }

    // 2D sinusoidal camera position embedding (ACTSinusoidalPositionEmbedding2d):
    // out [fh*fw, dim], the y half followed by the x half.
    static void sinusoid_pos_2d(int fh, int fw, int dim, float* out);

    // feats: n_cams pointers to [fh*fw, dim] backbone feature maps (NHWC tokens);
    // state_norm [state_dim] -> tokens / pos [n_tokens(...), dim].
    // out_tokens/out_pos are named apart from the same-purpose members below,
    // which is what predict() passes in.
    void build_tokens(const float* const* feats, int n_cams, int fh, int fw,
                      const float* state_norm, float* out_tokens, float* out_pos) const;

    // x [T, dim] in place through the encoder blocks.
    void encode(float* x, const float* tok_pos, int T) const;

    // encoder output + its pos -> actions_norm [chunk, action_dim].
    // dec_out (optional) receives the [chunk, dim] decoder output after its final norm.
    void decode(const float* enc_out, const float* tok_pos, int T,
                float* actions_norm, float* dec_out = nullptr) const;

    // build_tokens + encode + decode, using the model's own scratch.
    void forward(const float* const* feats, int n_cams, int fh, int fw,
                 const float* state_norm, float* actions_norm) const;
};

} // namespace tcpu
