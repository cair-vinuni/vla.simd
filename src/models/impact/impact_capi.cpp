/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// C ABI for ImpactModel, for ctypes/FFI callers. Built as the shared lib
// vla_simd_impact.
// Contract: include/vla_simd.h.
//
// The instruction is passed on every predict() to match the other
// language-conditioned models' ABI, but ImpactModel::set_instruction() skips the
// work when the string has not changed - so a control loop repeating the same
// sentence pays for the text tower once per episode, not once per query.

#include <memory>
#include "vla_simd.h"
#include "models/impact/impact_model.h"
#include <cstring>
#include <vector>

using tcpu::ImpactModel;

extern "C" {

void* vla_impact_load(const char* model_dir) try {
    if (!model_dir) return nullptr;
    auto m = std::make_unique<ImpactModel>();
    if (!m->load(model_dir)) return nullptr;
    return m.release();
} catch (...) {
    // A malformed .meta reaches std::stoi / a size-driven resize, and letting
    // either unwind through the caller's C frames terminates the process.
    return nullptr;
}

void vla_impact_free(void* h) { delete static_cast<ImpactModel*>(h); }

int32_t vla_impact_chunk(void* h)      { return h ? static_cast<ImpactModel*>(h)->chunk() : 0; }
int32_t vla_impact_action_dim(void* h) { return h ? static_cast<ImpactModel*>(h)->action_dim() : 0; }
int32_t vla_impact_state_dim(void* h)  { return h ? static_cast<ImpactModel*>(h)->state_dim() : 0; }
int32_t vla_impact_n_cams(void* h)     { return h ? static_cast<ImpactModel*>(h)->n_cams : 0; }
int32_t vla_impact_img_h(void* h)      { return h ? static_cast<ImpactModel*>(h)->img_h : 0; }
int32_t vla_impact_img_w(void* h)      { return h ? static_cast<ImpactModel*>(h)->img_w : 0; }
int32_t vla_impact_n_text(void* h)     { return h ? static_cast<ImpactModel*>(h)->n_text() : 0; }

int32_t vla_impact_set_instruction(void* h, const char* instruction) try {
    if (!h || !instruction) return VLA_ERR_ARG;
    return static_cast<ImpactModel*>(h)->set_instruction(instruction) ? VLA_OK : VLA_ERR_ARG;
} catch (...) {
    return VLA_ERR_EXCEPTION;
}

int32_t vla_impact_predict(void* h, const uint8_t* frames, const float* state,
                           const char* instruction, int32_t unnormalize, float* actions) try {
    if (!h || !frames || !state || !instruction || !actions) return VLA_ERR_ARG;

    auto* m = static_cast<ImpactModel*>(h);
    if (!m->set_instruction(instruction)) return VLA_ERR_ARG;

    const size_t stride = (size_t)m->img_h*m->img_w*3;
    std::vector<const uint8_t*> ip(m->n_cams);
    for (int c=0; c<m->n_cams; c++)
        ip[c] = frames+(size_t)c*stride;

    m->predict(ip.data(), state, unnormalize != 0, actions);
    return VLA_OK;
} catch (...) {
    return VLA_ERR_EXCEPTION;
}

/* Language-side intermediates of the CURRENT instruction, for the parity harness.
 * Valid after set_instruction() or the first predict(). */
int32_t vla_impact_tokens(void* h, int32_t* ids, int32_t* compact, int32_t* mask,
                          int32_t max_elems) try {
    if (!h) return VLA_ERR_ARG;
    auto* m = static_cast<ImpactModel*>(h);
    const int n = m->n_text();
    if (max_elems < n || m->token_ids().size() != (size_t)n) return VLA_ERR_ARG;
    for (int i = 0; i < n; i++) {
        if (ids)     ids[i]     = m->token_ids()[(size_t)i];
        if (compact) compact[i] = m->compact_ids()[(size_t)i];
        if (mask)    mask[i]    = m->token_mask()[(size_t)i];
    }
    return n;
} catch (...) {
    return VLA_ERR_EXCEPTION;
}

int32_t vla_impact_film(void* h, float* gamma, float* beta, int32_t max_elems) try {
    if (!h) return VLA_ERR_ARG;
    auto* m = static_cast<ImpactModel*>(h);
    const int n = (int)m->film_gamma().size();
    if (n == 0) return m->token_ids().empty() ? VLA_ERR_ARG : 0;
    if (max_elems < n) return VLA_ERR_ARG;
    if (gamma) std::memcpy(gamma, m->film_gamma().data(), sizeof(float)*(size_t)n);
    if (beta)  std::memcpy(beta,  m->film_beta().data(),  sizeof(float)*(size_t)n);
    return n;
} catch (...) {
    return VLA_ERR_EXCEPTION;
}

} // extern "C"
