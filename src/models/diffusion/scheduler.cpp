/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "models/diffusion/scheduler.h"
#include <algorithm>
#include <cmath>

namespace tcpu {

// Betas are computed and carried in double. diffusers builds them in float64
// (python math + numpy) before the tensor ever becomes fp32, and alphas_cumprod
// is a running product over up to 100 terms -- accumulating that in fp32 drifts
// far enough to move the last few steps of a DDPM trajectory.
static double alpha_bar_cosine(double t) {
    const double c = std::cos((t + 0.008)/1.008 * M_PI/2.0);
    return c*c;
}

bool DPNoiseScheduler::init(const DPConfig& c) {
    cfg = c;
    const int N = cfg.num_train_timesteps;
    if (N <= 0 || cfg.num_inference_steps <= 0) return false;
    if (cfg.prediction_type != "epsilon") return false;   // the only one lerobot trains

    std::vector<double> betas((size_t)N);
    if (cfg.beta_schedule == "squaredcos_cap_v2") {
        // Glide cosine schedule, capped at 0.999 to keep alpha_bar away from 0.
        for (int i=0; i<N; i++) {
            const double t1 = (double)i/(double)N;
            const double t2 = (double)(i+1)/(double)N;
            betas[i] = std::min(1.0 - alpha_bar_cosine(t2)/alpha_bar_cosine(t1), 0.999);
        }
    } else if (cfg.beta_schedule == "linear") {
        for (int i=0; i<N; i++)
            betas[i] = (double)cfg.beta_start +
                       ((double)cfg.beta_end - (double)cfg.beta_start)*(double)i/(double)(N-1);
    } else if (cfg.beta_schedule == "scaled_linear") {
        const double a = std::sqrt((double)cfg.beta_start), b = std::sqrt((double)cfg.beta_end);
        for (int i=0; i<N; i++) {
            const double v = a + (b - a)*(double)i/(double)(N-1);
            betas[i] = v*v;
        }
    } else {
        return false;
    }

    alphas_cumprod.resize((size_t)N);
    double run = 1.0;
    for (int i=0; i<N; i++) { run *= (1.0 - betas[i]); alphas_cumprod[i] = run; }

    // "leading" spacing, the diffusers default for both schedulers.
    const int ratio = N/cfg.num_inference_steps;
    timesteps.clear();
    for (int i=cfg.num_inference_steps-1; i>=0; i--) timesteps.push_back(i*ratio);
    return true;
}

double DPNoiseScheduler::prev_alpha_cumprod(int t) const {
    return t >= 0 ? alphas_cumprod[(size_t)t] : 1.0;
}

void DPNoiseScheduler::step(float* sample, const float* model_out, const float* noise,
                            int t, int step_index) const {
    const size_t n = (size_t)cfg.horizon*cfg.action_dim;
    const double lo = -(double)cfg.clip_sample_range, hi = (double)cfg.clip_sample_range;

    // The two schedulers disagree about what "the previous timestep" is, and it
    // matters as soon as the inference ladder is shorter than the training one.
    // DDPM walks its own ladder (prev = the next entry, -1 at the end); DDIM
    // subtracts the fixed stride. Deriving one from the other would be right for
    // DDPM-100 and wrong for DDIM-10, which is exactly the row we care about.
    int prev_t;
    if (cfg.scheduler == DPScheduler::DDPM)
        prev_t = (step_index + 1 < (int)timesteps.size()) ? timesteps[(size_t)step_index+1] : -1;
    else
        prev_t = t - cfg.num_train_timesteps/cfg.num_inference_steps;

    const double ap   = alphas_cumprod[(size_t)t];
    const double app  = prev_alpha_cumprod(prev_t);
    const double bp   = 1.0 - ap;
    const double sqrt_ap = std::sqrt(ap), sqrt_bp = std::sqrt(bp);

    if (cfg.scheduler == DPScheduler::DDPM) {
        const double bpp   = 1.0 - app;
        const double cur_a = ap/app;
        const double cur_b = 1.0 - cur_a;
        const double c0    = (std::sqrt(app)*cur_b)/bp;      // pred_original coeff
        const double c1    = std::sqrt(cur_a)*bpp/bp;        // current sample coeff

        double var = 0.0;
        if (t > 0) var = std::sqrt(std::max((1.0 - app)/(1.0 - ap)*cur_b, 1e-20));

        for (size_t i=0; i<n; i++) {
            double x0 = ((double)sample[i] - sqrt_bp*(double)model_out[i])/sqrt_ap;
            if (cfg.clip_sample) x0 = std::min(hi, std::max(lo, x0));
            double v = c0*x0 + c1*(double)sample[i];
            if (t > 0 && noise) v += var*(double)noise[i];
            sample[i] = (float)v;
        }
    } else {
        // DDIM with eta = 0: deterministic, so no noise term at all.
        const double dir = std::sqrt(std::max(0.0, 1.0 - app));
        const double sqrt_app = std::sqrt(app);
        for (size_t i=0; i<n; i++) {
            double x0 = ((double)sample[i] - sqrt_bp*(double)model_out[i])/sqrt_ap;
            if (cfg.clip_sample) x0 = std::min(hi, std::max(lo, x0));
            sample[i] = (float)(sqrt_app*x0 + dir*(double)model_out[i]);
        }
    }
}

} // namespace tcpu
