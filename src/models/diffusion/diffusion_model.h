/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "models/diffusion/config.h"
#include "models/diffusion/rgb_encoder.h"
#include "models/diffusion/scheduler.h"
#include "models/diffusion/unet1d.h"
#include <string>
#include <vector>

// Diffusion Policy end to end: n_obs_steps frames and states in, an executable
// action chunk out. Weights + stats from tools/convert_diffusion.py.
//
// Two things separate this from every other policy in the engine and both are
// load-bearing:
//   * it consumes a HISTORY. n_obs_steps frames per camera and n_obs_steps
//     states, not one of each, so the vision tower runs n_cams*n_obs_steps times
//     per query and that cost is real.
//   * the UNet runs once per denoising step. The step count is a deployment
//     knob, not a weight, and DDPM-at-train-count vs DDIM-at-10 differ by an
//     order of magnitude in latency for identical weights.
//
// State and action use MIN_MAX normalization to [-1, 1] (which is what makes the
// scheduler's clip_sample_range of 1.0 meaningful); images use MEAN_STD.

namespace tcpu {

struct DiffusionModel {
    DPConfig cfg;
    std::vector<DPRgbEncoder> encoders;   // one per camera, or one shared
    DPUNet1d unet;
    DPNoiseScheduler sched;

    std::vector<std::string> cam_names;
    std::vector<float> state_min, state_max, action_min, action_max;
    std::vector<float> img_mean, img_std;   // [n_cams, 3]

    bool load(const std::string& dir);

    int chunk()      const { return cfg.n_action_steps; }
    int horizon()    const { return cfg.horizon; }
    int action_dim() const { return cfg.action_dim; }
    int state_dim()  const { return cfg.state_dim; }
    int n_obs()      const { return cfg.n_obs_steps; }

    // frames: n_obs_steps * n_cams uint8 HWC images at img_h x img_w, ordered
    //   obs-major then camera (frame[s][c]), oldest observation first.
    // state:  n_obs_steps * state_dim in raw robot units, oldest first.
    // noise:  optional injected noise. When given it is
    //   [1 + num_inference_steps, horizon, action_dim]: the prior first, then
    //   one buffer per DDPM step. Null draws nothing and starts from zeros,
    //   which is only useful for shape checks -- a real query must supply it.
    // actions: [n_action_steps, action_dim].
    void predict(const uint8_t* const* frames, const float* state,
                 const float* noise, bool unnormalize, float* actions);

  private:
    std::vector<float> imgbuf;      // preprocessed image, reused across frames
    BackboneScratch scratch;
    std::vector<float> gcond;       // global conditioning vector
    std::vector<float> sample, eps; // [horizon, action_dim]

    void preprocess(const uint8_t* src, int cam, float* dst) const;
};

} // namespace tcpu
