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

// TurboVLA action head: state projection + a pre-norm ACT decoder + the action MLP.
//
// State: LN -> 256 -gelu-> state_tokens*hidden, reshaped to [state_tokens,
// hidden], + a learned position, + LayerNorm. Those tokens are appended to the
// condition, so the decoder's memory is [visual | text | state].
//
// Decoder (torch nn.TransformerDecoderLayer, norm_first=True, ReLU, no final
// norm), 3 layers over `chunk` learned action queries:
//   x = x + self_attn(norm1(x))
//   x = x + cross_attn(norm2(x), memory)
//   x = x + linear2(relu(linear1(norm3(x))))
//
// Head: a 3-layer ReLU MLP to action_dim, then tanh. The output is in the
// normalized action space; TurboVlaModel maps it to env units.

namespace tcpu {

struct TurboDecoderLayer {
    nn::MhaMasked self;              // queries only, no mask
    const float *n1_w = nullptr, *n1_b = nullptr;
    nn::MhaQKV cross;                // queries x memory
    const float *n2_w = nullptr, *n2_b = nullptr;
    nn::Linear ff1, ff2;
    const float *n3_w = nullptr, *n3_b = nullptr;
};

struct TurboActionHead {
    TurboHeadConfig cfg;
    std::vector<float> data;
    const float *state_ln_w = nullptr, *state_ln_b = nullptr;
    nn::Linear sp1, sp2;
    const float* state_pos = nullptr;        // [state_tokens, hidden]
    const float *state_norm_w = nullptr, *state_norm_b = nullptr;
    const float* queries = nullptr;          // [chunk, hidden]
    std::vector<TurboDecoderLayer> layers;
    std::vector<nn::Linear> mlp;             // action_projection

    bool load(const std::string& dir);

    // state_norm [state_dim] -> tokens [state_tokens, hidden].
    void state_tokens(const float* state_norm, float* tokens) const;

    // memory [n_mem, hidden] -> actions_norm [chunk, action_dim] (post-tanh).
    void decode(const float* memory, int n_mem, float* actions_norm) const;

  private:
    mutable nn::Scratch sc;
    mutable std::vector<float> x, h, ff, res;
};

} // namespace tcpu
