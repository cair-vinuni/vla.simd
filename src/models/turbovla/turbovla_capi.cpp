/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// C ABI for TurboVlaModel, for ctypes/FFI callers (a LIBERO policy server or the
// parity script). Built as the shared lib vla_simd_turbovla.
// Contract: include/vla_simd.h.

#include "vla_simd.h"
#include "models/turbovla/turbovla_model.h"
#include <cstring>
#include <memory>

using tcpu::TurboVlaModel;

namespace {

const std::vector<float>* pick(const TurboVlaModel* m, int32_t which) {
    const tcpu::TurboVlaTrace& t = m->trace();
    switch (which) {
        case VLA_TURBOVLA_PIXEL_VALUES:  return &t.pixel_values;
        case VLA_TURBOVLA_DINO_TOKENS:   return &t.dino_tokens;
        case VLA_TURBOVLA_VISION_PROJ:   return &t.vision_proj;
        case VLA_TURBOVLA_VISUAL_TOKENS: return &t.visual_tokens;
        case VLA_TURBOVLA_INPUT_IDS:     return &t.input_ids;
        case VLA_TURBOVLA_POSITION_IDS:  return &t.position_ids;
        case VLA_TURBOVLA_TEXT_PAD_MASK: return &t.text_pad_mask;
        case VLA_TURBOVLA_SELF_ATTN:     return &t.self_attn;
        case VLA_TURBOVLA_BERT_HIDDEN:   return &t.bert_hidden;
        case VLA_TURBOVLA_TEXT_TOKENS:   return &t.text_tokens;
        case VLA_TURBOVLA_FUSED_VISUAL:  return &t.fused_visual;
        case VLA_TURBOVLA_FUSED_TEXT:    return &t.fused_text;
        case VLA_TURBOVLA_CONDITION:     return &t.condition;
        case VLA_TURBOVLA_STATE_NORM:    return &t.state_norm;
        case VLA_TURBOVLA_STATE_TOKENS:  return &t.state_tokens;
        case VLA_TURBOVLA_ACTIONS_NORM:  return &t.actions_norm;
        default:                         return nullptr;
    }
}

} // namespace

extern "C" {

void* vla_turbovla_load(const char* model_dir) try {
    if (!model_dir) return nullptr;
    auto m = std::make_unique<TurboVlaModel>();
    if (!m->load(model_dir)) return nullptr;
    return m.release();
} catch (...) {
    // A malformed .meta reaches a size-driven resize; letting bad_alloc unwind
    // through the caller's C frames terminates the process.
    return nullptr;
}

void vla_turbovla_free(void* h) { delete static_cast<TurboVlaModel*>(h); }

#define TVLA_GET(name, expr) \
    int32_t vla_turbovla_##name(void* h) { \
        const auto* m = static_cast<const TurboVlaModel*>(h); \
        return m ? (int32_t)(expr) : 0; \
    }

TVLA_GET(chunk,        m->chunk())
TVLA_GET(action_dim,   m->action_dim())
TVLA_GET(state_dim,    m->state_dim())
TVLA_GET(n_views,      m->n_views())
TVLA_GET(img_size,     m->img_size())
TVLA_GET(n_patches,    m->n_patches())
TVLA_GET(text_pad,     m->text_pad())
TVLA_GET(hidden,       m->hidden())
TVLA_GET(vis_dim,      m->vision.cfg.hidden)
TVLA_GET(text_hidden,  m->text.cfg.hidden)
TVLA_GET(state_tokens, m->head.cfg.state_tokens)

#undef TVLA_GET

int32_t vla_turbovla_pad_length(void* h, const char* instruction) try {
    if (!h || !instruction) return VLA_ERR_ARG;
    return (int32_t)static_cast<const TurboVlaModel*>(h)->pad_length(instruction);
} catch (...) {
    return VLA_ERR_EXCEPTION;
}

int32_t vla_turbovla_predict(void* h, const uint8_t* frames, const float* state,
                             const char* instruction, int32_t unnormalize,
                             float* actions) try {
    if (!h || !frames || !state || !instruction || !actions) return VLA_ERR_ARG;
    static_cast<const TurboVlaModel*>(h)->predict(frames, state, instruction,
                                                  unnormalize != 0, actions);
    return VLA_OK;
} catch (...) {
    return VLA_ERR_EXCEPTION;
}

int32_t vla_turbovla_tensor(void* h, int32_t which, float* out, int32_t max_elems) try {
    if (!h) return VLA_ERR_ARG;
    const auto* m = static_cast<const TurboVlaModel*>(h);
    const std::vector<float>* t = pick(m, which);
    if (!t) return VLA_ERR_ARG;
    const int32_t n = (int32_t)t->size();
    if (!out) return n;
    if (max_elems < n) return VLA_ERR_SHAPE;
    std::memcpy(out, t->data(), (size_t)n*sizeof(float));
    return n;
} catch (...) {
    return VLA_ERR_EXCEPTION;
}

} // extern "C"
