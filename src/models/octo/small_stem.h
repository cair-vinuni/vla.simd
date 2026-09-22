/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "nn/conv.h"
#include "nn/linear.h"
#include <cstdint>
#include <string>
#include <vector>

// Octo SmallStem16 image tokenizer: 4x (3x3 stride-2 conv + GroupNorm + ReLU), then
// a 1x1 "embedding" conv to embed_dim. Input is the observation RGB stacked with the
// goal RGB channel-wise (6ch); weight standardization is folded into the dumped conv
// weights. Weights: stem_{primary,wrist}.meta/.bin from tools/convert_octo.py.

namespace tcpu {

struct SmallStemConfig {
    static constexpr int MAX_LAYERS = 8;   // the bound on `features`, enforced in load()
    int in_ch = 6, n_layers = 4, k = 3, stride = 2, pad = 1;
    int features[MAX_LAYERS] = {32, 96, 192, 384};
    int embed_dim = 512, gn_groups = 32;
    float gn_eps = 1e-6f;
};

struct StemLayer {
    nn::Conv2d conv;
    const float *gn_scale = nullptr, *gn_bias = nullptr;
};

struct SmallStem {
    SmallStemConfig cfg;
    std::vector<float> data;
    std::vector<StemLayer> layers;
    nn::Linear embed;   // [embed_dim, features[last]]

    bool load(const std::string& dir, const std::string& name);

    // obs [H,W,3] uint8 HWC; goal same or nullptr (absent goal = zeros, like
    // create_tasks). out [(H/16)*(W/16), embed_dim].
    void encode(const uint8_t* obs, const uint8_t* goal, int H, int W, float* out) const;
};

} // namespace tcpu
