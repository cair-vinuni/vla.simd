/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "models/diffusion/config.h"
#include "nn/linear.h"
#include <string>
#include <vector>

// DiffusionConditionalUnet1d: the noise predictor, and the whole of the per-step
// cost. It runs once per denoising step, so a DDPM query at 100 train steps runs
// this UNet 100 times -- which is why the step count is a first-class row in the
// results rather than a footnote.
//
// Layout is [T, C] channel-last throughout, matching ops/conv1d and letting
// groupnorm() apply with n_pixels = T. The reference works in torch's [C, T];
// the converter transposes the conv weights once so no transpose runs per step.

namespace tcpu {

// Conv1d -> GroupNorm -> Mish. The conv is an nn::Linear over the im2col'd
// taps, so it routes to the same packed-panel GEMM every other layer uses.
struct DPConvBlock {
    nn::Linear conv;                             // N = cout, K = k*cin
    const float *gn_s = nullptr, *gn_b = nullptr;
    int cin = 0, cout = 0, k = 0;

    // x [T, cin] -> out [T, cout] (stride 1, pad k/2 keeps T)
    void forward(float* out, const float* x, int T, int groups, float eps,
                 std::vector<float>& col) const;
};

// ResNet-style block with FiLM conditioning from the global feature vector.
struct DPResBlock {
    DPConvBlock conv1, conv2;
    nn::Linear cond;                             // cond_dim -> cout (or 2*cout with scale)
    nn::Linear res;                              // 1x1 residual conv; unused when cin==cout
    bool has_res = false;
    int cin = 0, cout = 0;
    bool film_scale = false;

    // x [T, cin] -> out [T, cout]. `gm` is mish(global_feature), already applied:
    // the reference's cond_encoder is Mish followed by Linear and the global
    // feature is the same tensor for every block in the UNet, so the Mish is
    // hoisted out of the block and run once per step instead of eighteen times.
    void forward(float* out, const float* x, const float* gm, int T,
                 int groups, float eps, std::vector<float>& col,
                 std::vector<float>& scratch) const;
};

struct DPUNet1d {
    DPConfig cfg;
    std::vector<float> data;

    // diffusion timestep encoder: sinusoidal -> Linear -> Mish -> Linear
    nn::Linear step1, step2;

    // Downsample is a strided Conv1d (k3 s2 p1); upsample a ConvTranspose1d
    // (k4 s2 p1). The last down block and, in the stock 3-stage config, neither
    // up block, is an Identity -- `has` records which.
    struct Down { DPResBlock r1, r2; nn::Linear ds; bool has = false; int dc = 0; };
    struct Up   { DPResBlock r1, r2; const float *uw = nullptr, *ub = nullptr;
                  bool has = false; int uc = 0; };
    std::vector<Down> down;
    DPResBlock mid1, mid2;
    std::vector<Up> up;
    DPConvBlock final_block;
    nn::Linear final_conv;                                 // 1x1 conv to action_dim

    bool load(const std::string& dir, const std::string& name, const DPConfig& c);

    // sample [horizon, action_dim] -> eps [horizon, action_dim]; gc is the
    // global conditioning vector [global_cond_dim], t the timestep being denoised.
    void forward(float* eps, const float* sample, const float* gc, int t) const;

    // Sinusoidal timestep embedding, exposed so the golden test can check it
    // in isolation -- it is the one part of the UNet with no learned weights
    // and so the one a shape bug hides in most easily.
    void step_embedding(float* out, int t) const;
};

} // namespace tcpu
