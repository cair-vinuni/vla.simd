/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "nn/conv.h"
#include <string>
#include <vector>

// ResNet-18 truncated at layer4, with FiLM. Adapted from
// src/models/act/resnet_backbone.cpp - see docs/14-impact-design.md section 4 for
// why it is a copy rather than a shared module.
//
// Every BatchNorm is frozen in the checkpoint and folded into the conv in front
// of it, so what runs here is conv + bias + relu. FiLM cannot be folded the same
// way: its scale/shift come from the instruction at runtime. It is applied at the
// output of a stage, after the residual add and its ReLU, as
//
//     x = (1 + gamma_c) * x + beta_c
//
// per channel. The `1 +` is what makes a zero-initialized FiLM head the identity,
// so an untrained head leaves the pretrained backbone alone; writing the plain
// product instead zeroes the feature map.
//
// `film_after` lists the block indices a FiLM point follows (for ResNet-18 with
// 8 basic blocks: 1, 3, 5, 7 - the end of each stage). gamma/beta arrive as one
// flat buffer, the points concatenated in that order.
//
// Weights: vision.meta/.bin from tools/convert_impact.py.

namespace tcpu {

struct ImpactResNetConfig {
    int in_ch = 3, stem_out = 64;
    int stem_k = 7, stem_stride = 2, stem_pad = 3;
    int pool_k = 3, pool_stride = 2, pool_pad = 1;
    int block_k = 3;   // basic block (resnet18/34); bottlenecks are not supported
};

struct ImpactBasicBlock {
    nn::Conv2d conv1, conv2, down;
    int cin = 0, cout = 0, stride = 1;
    bool has_down = false;
};

// Activation buffers, owned by the caller so two cameras can run concurrently
// (and so the frames of a control loop reuse one allocation).
struct ImpactBackboneScratch {
    std::vector<float> a, b, c, res;
};

struct ResNetFilm {
    ImpactResNetConfig cfg;
    std::vector<float> data;
    nn::Conv2d stem;
    std::vector<ImpactBasicBlock> blocks;
    std::vector<int> film_after;   // block indices a FiLM point follows

    bool load(const std::string& dir, const std::string& name = "vision");

    int out_channels() const { return blocks.empty() ? cfg.stem_out : blocks.back().cout; }
    void feat_size(int H, int W, int* fh, int* fw) const;

    // Channels the FiLM points modulate, in order. Empty when film_after is.
    std::vector<int> film_channels() const;
    int film_total() const;

    // x [H, W, in_ch] normalized image (NHWC) -> out [fh*fw, out_channels()].
    // gamma/beta are [film_total()] each, the points concatenated; pass null for
    // both to run the backbone unmodulated (which is what an ablation wants).
    void forward(const float* x, int H, int W, ImpactBackboneScratch& s, float* out,
                 const float* gamma = nullptr, const float* beta = nullptr) const;
};

} // namespace tcpu
