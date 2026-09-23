/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// C ABI for ActModel, for ctypes/FFI callers (a policy server or the parity
// script, feeding raw uint8 camera frames at the checkpoint's resolution).
// Built as the shared lib vla_simd_act. Contract: include/vla_simd.h.

#include <memory>
#include "vla_simd.h"
#include "models/act/act_model.h"
#include <vector>

using tcpu::ActModel;

extern "C" {

void* vla_act_load(const char* model_dir) try {
    if (!model_dir) return nullptr;
    auto m = std::make_unique<ActModel>();
    if (!m->load(model_dir)) return nullptr;
    return m.release();
} catch (...) {
    // A malformed .meta reaches std::stoi / a size-driven resize, and letting
    // either unwind through the caller's C frames terminates the process.
    return nullptr;
}

void vla_act_free(void* h) { delete static_cast<ActModel*>(h); }

int32_t vla_act_chunk(void* h)      { return h ? static_cast<ActModel*>(h)->chunk() : 0; }
int32_t vla_act_action_dim(void* h) { return h ? static_cast<ActModel*>(h)->action_dim() : 0; }
int32_t vla_act_state_dim(void* h)  { return h ? static_cast<ActModel*>(h)->state_dim() : 0; }
int32_t vla_act_n_cams(void* h)     { return h ? static_cast<ActModel*>(h)->n_cams : 0; }
int32_t vla_act_img_h(void* h)      { return h ? static_cast<ActModel*>(h)->img_h : 0; }
int32_t vla_act_img_w(void* h)      { return h ? static_cast<ActModel*>(h)->img_w : 0; }

int32_t vla_act_predict(void* h, const uint8_t* images, const float* state,
                        int32_t unnormalize, float* actions) try {
    if (!h || !images || !state || !actions) return VLA_ERR_ARG;

    auto* m = static_cast<ActModel*>(h);
    const size_t stride = (size_t)m->img_h*m->img_w*3;
    std::vector<const uint8_t*> ip(m->n_cams);
    for (int c=0; c<m->n_cams; c++)
        ip[c] = images+(size_t)c*stride;

    m->predict(ip.data(), state, unnormalize != 0, actions);
    return VLA_OK;
} catch (...) {
    return VLA_ERR_EXCEPTION;
}

} // extern "C"
