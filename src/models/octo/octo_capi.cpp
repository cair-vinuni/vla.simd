/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// C ABI for OctoModel, for ctypes/FFI callers (e.g. a policy server that does
// its own image preprocessing and feeds pre-resized uint8 frames).
// Built as the shared lib vla_simd_octo. Contract: include/vla_simd.h.

#include <memory>
#include "vla_simd.h"
#include "models/octo/octo_model.h"

using tcpu::OctoModel;

extern "C" {

void* vla_octo_load(const char* model_dir, const char* tok_dir) try {
    if (!model_dir || !tok_dir) return nullptr;
    auto m = std::make_unique<OctoModel>();
    if (!m->load(model_dir, tok_dir)) return nullptr;
    return m.release();
} catch (...) {
    return nullptr;
}

void vla_octo_free(void* h) { delete static_cast<OctoModel*>(h); }

int32_t vla_octo_horizon(void* h)    { return h ? static_cast<OctoModel*>(h)->head.cfg.horizon : 0; }
int32_t vla_octo_action_dim(void* h) { return h ? static_cast<OctoModel*>(h)->head.cfg.action_dim : 0; }
int32_t vla_octo_max_window(void* h) { return h ? static_cast<OctoModel*>(h)->tf.cfg.max_horizon : 0; }

int32_t vla_octo_predict(void* h, const uint8_t* primary, const uint8_t* wrist,
                         int32_t wnd, const uint8_t* timestep_mask, const char* instruction,
                         uint64_t seed, int32_t unnormalize, float* actions) {
    return vla_octo_predict_ex(h, primary, wrist, wnd, timestep_mask, instruction,
                               nullptr, nullptr, seed, unnormalize, actions);
}

int32_t vla_octo_predict_ex(void* h, const uint8_t* primary, const uint8_t* wrist,
                            int32_t wnd, const uint8_t* timestep_mask,
                            const char* instruction, const float* noise, const float* z,
                            uint64_t seed, int32_t unnormalize, float* actions) try {
    if (!h || !primary || !timestep_mask || !instruction || !actions)
        return VLA_ERR_ARG;
    auto* m = static_cast<OctoModel*>(h);
    // The position tables are sized [max_horizon, tokens, d]; a larger window
    // would index straight past the weight arena.
    if (wnd < 1 || wnd > m->tf.cfg.max_horizon) return VLA_ERR_SHAPE;
    // noise and z must be supplied together: half-injected sampling would compare
    // the engine against a reference trajectory it never actually followed.
    if ((noise == nullptr) != (z == nullptr)) return VLA_ERR_ARG;
    m->predict(primary, wrist, wnd, timestep_mask, instruction, noise, z,
               seed, unnormalize != 0, actions);
    return VLA_OK;
} catch (...) {
    return VLA_ERR_EXCEPTION;
}

} // extern "C"
