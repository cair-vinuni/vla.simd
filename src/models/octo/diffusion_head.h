/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "nn/diffusion.h"
#include "nn/linear.h"
#include <string>
#include <vector>

// Octo DiffusionActionHead: MLPResNet score network + 20-step DDPM sampling with a
// cosine beta schedule. Input embedding = the readout_action token of the last
// timestep. Weights: head.meta/head.bin from tools/convert_octo.py.

namespace tcpu {

struct DiffusionHeadConfig {
    int emb = 384, action_dim = 7, horizon = 4, time_dim = 32, num_blocks = 3, hidden = 256, steps = 20;
    float max_action = 5.0f;
    int flat() const { return action_dim*horizon; }
};

struct DiffusionHead {
    DiffusionHeadConfig cfg;
    std::vector<float> data;
    const float *fourier_w = nullptr;                       // [time_dim/2]
    nn::DiffusionMlp net;   // score network: in -> N residual blocks -> out
    nn::Linear cond0, cond1;
    const float *betas = nullptr, *alphas = nullptr, *alpha_hats = nullptr;  // [steps]
    std::vector<float> cond_table;   // [steps, time_dim] precomputed time conditioning

    bool load(const std::string& dir);

    // One score-net eval: eps [flat] for embedding emb [emb], noisy actions x [flat],
    // diffusion time t.
    void eps(const float* emb, const float* x, int t, float* out) const;

    // Time conditioning (Fourier features + cond MLP) for diffusion time t; the
    // sampling loop only ever uses t = 0..steps-1, precomputed in cond_table.
    void time_cond(float t, float* cond) const;

    // Full DDPM loop. noise [flat] = initial x; z [steps, flat] = per-step gaussian
    // (step s corresponds to t = steps-1-s; z at t=0 is unused). actions [flat].
    void denoise(const float* emb, const float* noise, const float* z, float* actions) const;
};

} // namespace tcpu
