/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <string>
#include <vector>

// Diffusion Policy (Chi et al. 2023), as lerobot implements it in
// `lerobot/policies/diffusion`. Shapes come from the checkpoint's own config;
// nothing here is defaulted to the paper, because a converted checkpoint that
// disagrees with these fields is a converter bug we want to see rather than
// silently absorb.

namespace tcpu {

// Which reverse process the sampler runs. Both are first-class: DDPM at its
// trained step count is the configuration the reference ships, DDIM at a short
// step count is the one anybody actually deploys, and the two differ by enough
// in control rate that collapsing them into one number would hide the result.
enum class DPScheduler { DDPM, DDIM };

struct DPConfig {
    // observation
    int n_obs_steps = 2;        // frames AND states per query; DP is not single-frame
    int n_cams      = 2;
    int img_h       = 480, img_w = 640;
    int crop_h      = 0, crop_w = 0;   // 0 = no crop (center crop at eval when set)
    int resize_h    = 0, resize_w = 0; // 0 = no resize
    int state_dim   = 6;

    // action
    int action_dim     = 6;
    int horizon        = 64;    // what the UNet denoises
    int n_action_steps = 32;    // what the robot executes -- the control-rate numerator

    // vision
    int  num_keypoints = 32;    // SpatialSoftmax keypoints; feature dim is 2*this
    bool separate_encoder_per_camera = true;

    // UNet1d
    std::vector<int> down_dims = {512, 1024, 2048};
    int kernel_size = 5;
    int n_groups    = 8;
    int step_embed_dim = 128;
    bool film_scale = true;     // FiLM modulates scale as well as bias

    // diffusion
    DPScheduler scheduler = DPScheduler::DDPM;
    int   num_train_timesteps = 100;
    int   num_inference_steps = 100;   // DDPM default is the train count; DDIM is short
    float beta_start = 1e-4f, beta_end = 0.02f;
    std::string beta_schedule = "squaredcos_cap_v2";
    std::string prediction_type = "epsilon";
    bool  clip_sample = true;
    float clip_sample_range = 1.0f;

    float gn_eps = 1e-5f;       // torch GroupNorm default

    // Feature dim contributed by one camera at one observation step.
    int cam_feature_dim() const { return num_keypoints*2; }

    // What the UNet is conditioned on: (state + all cameras) at every obs step.
    int global_cond_dim() const {
        return (state_dim + n_cams*cam_feature_dim())*n_obs_steps;
    }
    int cond_dim() const { return step_embed_dim + global_cond_dim(); }
};

} // namespace tcpu
