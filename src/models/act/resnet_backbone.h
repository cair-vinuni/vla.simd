/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "nn/conv.h"
#include <string>
#include <vector>

// ResNet-18 image backbone truncated at layer4 (torchvision IntermediateLayerGetter,
// as ACT uses it): 7x7/s2 stem + 3x3/s2 maxpool, then 4 stages of 2 basic blocks.
// Every BatchNorm is frozen in the checkpoint, so the converter folds it into the
// conv in front of it - what runs here is conv + bias + relu only.
// Weights: backbone.meta/.bin from tools/act/convert_act.py.

namespace tcpu {

struct ResNetConfig {
    int in_ch = 3, stem_out = 64;
    int stem_k = 7, stem_stride = 2, stem_pad = 3;
    int pool_k = 3, pool_stride = 2, pool_pad = 1;
    int block_k = 3;   // basic block (resnet18/34); bottlenecks are not supported
};

struct BasicBlock {
    nn::Conv2d conv1, conv2, down;
    int cin = 0, cout = 0, stride = 1;
    bool has_down = false;
};

// Activation buffers, owned by the caller so two cameras can run concurrently
// (and so the frames of a control loop reuse one allocation).
struct BackboneScratch {
    std::vector<float> a, b, c, res;
};

struct ResNetBackbone {
    ResNetConfig cfg;
    std::vector<float> data;
    nn::Conv2d stem;
    std::vector<BasicBlock> blocks;

    bool load(const std::string& dir, const std::string& name = "backbone");

    int out_channels() const { return blocks.empty() ? cfg.stem_out : blocks.back().cout; }
    void feat_size(int H, int W, int* fh, int* fw) const;

    // x [H, W, in_ch] normalized image (NHWC) -> out [fh*fw, out_channels()].
    void forward(const float* x, int H, int W, BackboneScratch& s, float* out) const;
};

} // namespace tcpu
