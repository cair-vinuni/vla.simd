/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "models/act/resnet_backbone.h"
#include "models/diffusion/config.h"
#include "nn/linear.h"
#include <string>
#include <vector>

// Diffusion Policy's DiffusionRgbEncoder: ResNet-18 truncated at layer4, a 1x1
// conv to `num_keypoints` channels, spatial soft-argmax to 2 coordinates per
// keypoint, then Linear + ReLU. Output is 2*num_keypoints per camera per frame.
//
// The ResNet is the same one ACT and IMPACT use, so the backbone code and its
// BN folding are shared rather than duplicated. Note the difference from ACT:
// DP pools the feature map to keypoints instead of feeding it to a transformer,
// so the whole image collapses to 64 numbers before the action model sees it.

namespace tcpu {

struct DPRgbEncoder {
    DPConfig cfg;
    ResNetBackbone backbone;

    // 1x1 conv over the [fh*fw, C] feature map == a plain linear over channels.
    nn::Linear to_keypoints;   // [C -> num_keypoints]
    nn::Linear out;            // [2*num_keypoints -> 2*num_keypoints]

    std::vector<float> data;   // owns to_keypoints/out weights
    std::vector<float> grid;   // [fh*fw, 2] normalized coordinates
    int fh = 0, fw = 0;

    // `name` distinguishes the per-camera encoders (rgb_encoder0, rgb_encoder1).
    bool load(const std::string& dir, const std::string& name, const DPConfig& c);

    // x [crop_h, crop_w, 3] normalized image (NHWC) -> feat [2*num_keypoints].
    void forward(const float* x, BackboneScratch& s, float* feat) const;

    int feature_dim() const { return cfg.num_keypoints*2; }
};

} // namespace tcpu
