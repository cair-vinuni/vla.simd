/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "config.h"
#include "siglip_vision.h"
#include "smollm2_lm.h"
#include "action_expert.h"
#include <cstdint>
#include <string>
#include <vector>

// End-to-end SmolVLA: image(s) + instruction tokens + proprio state -> action chunk.
// Orchestration mirrors lerobot VLAFlowMatching.sample_actions + embed_prefix:
//   vision -> (x sqrt(hidden)) ++ lang embeds (x sqrt(hidden)) ++ state_proj  = prefix
//   VLM prefix (prefix-LM mask) -> K/V cache -> action expert flow-matching denoise.
// The prefix-LM 2D mask and positions are built here (make_att_2d_masks equivalent).

namespace tcpu {

struct SmolvlaModel {
    SiglipVision vit;
    SmollmVlm    vlm;
    ActionExpert aex;

    std::vector<uint16_t> emb;              // token_embd [vocab, hidden] (bf16)
    int vocab = 0, hidden = 0, max_state_dim = 32, n_views = 2;
    std::vector<float> state_w, state_b;    // state_proj [hidden, max_state_dim], [hidden]

    std::vector<float> state_mean, state_std, action_mean, action_std;
    int real_state_dim = 0, real_action_dim = 0;
    float norm_eps = 1e-8f;

    bool load(const std::string& dir);

    // Quantize the matmul groups SMOLVLA_INT8 selects to symmetric W8A8; called
    // by load(). Off unless the variable is set - the int8 path is lossy and
    // opt-in (see the bitmask table in smolvla_model.cpp).
    void apply_int8();

    // Preprocessed inputs (as the model sees them). pixels_all:[n_views,3,512,512] in [-1,1].
    // lang_tokens/lang_mask:[n_lang]. state:[max_state_dim] normalized. noise:[chunk,max_action_dim].
    // Returns normalized actions [chunk, max_action_dim].
    std::vector<float> predict_normalized(const float* pixels_all, int n_views,
                                          const int32_t* lang_tokens, const int32_t* lang_mask, int n_lang,
                                          const float* state, const float* noise,
                                          const float* prev, const float* weights, float max_guidance) const;

    // Demo path: raw state -> normalize -> predict -> unnormalize.
    // Returns real actions [chunk, max_action_dim] (first real_action_dim entries un-normalized).
    std::vector<float> predict(const float* pixels_all, int n_views,
                               const int32_t* lang_tokens, const int32_t* lang_mask, int n_lang,
                               const float* raw_state, const float* noise,
                               const float* prev, int n_prev, const float* weights, float max_guidance) const;
};

} // namespace tcpu
