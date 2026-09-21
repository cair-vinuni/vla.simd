/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "models/diffusion/config.h"
#include <vector>

// The reverse process, reimplemented from `diffusers` DDPMScheduler /
// DDIMScheduler as lerobot configures them (prediction_type "epsilon",
// variance_type "fixed_small", eta 0, set_alpha_to_one). Both are needed
// because the two are reported as separate rows: DDPM at the trained step
// count and DDIM at a short one are different deployment propositions.
//
// DDPM is stochastic -- it adds noise at every step but the last -- so a parity
// test can only be exact if the reference's noise is injected rather than drawn.
// step() therefore takes the noise for that step; the caller owns where it comes
// from, exactly as the campaign already does for Octo's DDPM head.

namespace tcpu {

struct DPNoiseScheduler {
    DPConfig cfg;
    std::vector<double> alphas_cumprod;   // [num_train_timesteps]
    std::vector<int>    timesteps;        // descending, length num_inference_steps

    // Builds the beta schedule and the inference timestep ladder. Returns false
    // on a schedule name we have not implemented, rather than silently running
    // the wrong one.
    bool init(const DPConfig& c);

    // One reverse step, in place on `sample` [horizon, action_dim].
    //   model_out  the UNet's epsilon prediction, same shape
    //   noise      [horizon, action_dim], used only by DDPM and only when t > 0;
    //              may be null when the caller knows it is unused.
    void step(float* sample, const float* model_out, const float* noise,
              int t, int step_index) const;

  private:
    double prev_alpha_cumprod(int t) const;
};

} // namespace tcpu
