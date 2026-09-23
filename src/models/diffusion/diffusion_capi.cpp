/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// C ABI for Diffusion Policy. The one departure from the other models' shape is
// the observation history: `frames` is n_obs_steps * n_cams images and `state`
// is n_obs_steps vectors, oldest first. There is deliberately no single-frame
// overload -- DP conditions on the history, and a convenience entry point that
// silently repeated one frame would produce a plausible chunk from an input the
// policy was never trained on.

#include <memory>
#include <vector>
#include "vla_simd.h"
#include "models/diffusion/diffusion_model.h"

using tcpu::DiffusionModel;

extern "C" {

void* vla_diffusion_load(const char* model_dir) try {
    if (!model_dir) return nullptr;
    auto m = std::make_unique<DiffusionModel>();
    if (!m->load(model_dir)) return nullptr;
    return m.release();
} catch (...) {
    return nullptr;
}

void vla_diffusion_free(void* h) {
    delete static_cast<DiffusionModel*>(h);
}

#define DP_GET(name, expr)                                        \
    int32_t vla_diffusion_##name(void* h) {                       \
        if (!h) return VLA_ERR_ARG;                               \
        auto* m = static_cast<DiffusionModel*>(h);                \
        (void)m;                                                  \
        return (int32_t)(expr);                                   \
    }

DP_GET(chunk,        m->chunk())
DP_GET(horizon,      m->horizon())
DP_GET(action_dim,   m->action_dim())
DP_GET(state_dim,    m->state_dim())
DP_GET(n_cams,       m->cfg.n_cams)
DP_GET(n_obs_steps,  m->n_obs())
DP_GET(img_h,        m->cfg.img_h)
DP_GET(img_w,        m->cfg.img_w)
DP_GET(num_steps,    m->sched.timesteps.size())
DP_GET(is_ddim,      m->cfg.scheduler == tcpu::DPScheduler::DDIM ? 1 : 0)

#undef DP_GET

int32_t vla_diffusion_predict(void* h, const uint8_t* frames, const float* state,
                              int32_t unnormalize, float* actions,
                              const float* noise) try {
    if (!h || !frames || !state || !actions) return VLA_ERR_ARG;

    auto* m = static_cast<DiffusionModel*>(h);
    const size_t stride = (size_t)m->cfg.img_h*m->cfg.img_w*3;
    const int n = m->n_obs()*m->cfg.n_cams;

    std::vector<const uint8_t*> ip((size_t)n);
    for (int i=0; i<n; i++) ip[(size_t)i] = frames + (size_t)i*stride;

    m->predict(ip.data(), state, noise, unnormalize != 0, actions);
    return VLA_OK;
} catch (...) {
    return VLA_ERR_EXCEPTION;
}

} // extern "C"
