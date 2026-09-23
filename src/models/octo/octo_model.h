/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "nn/t5_encoder.h"
#include "small_stem.h"
#include "octo_transformer.h"
#include "diffusion_head.h"
#include "tokenizer/t5_tokenizer.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Octo-Small orchestration: instruction + image window -> 4x7 action chunk.
// Weights from tools/convert_octo.py, tokenizer from convert_t5_tokenizer.py.

namespace tcpu {

struct OctoModel {
    T5Tokenizer tok;
    nn::T5Encoder t5;
    SmallStem stem_primary, stem_wrist;
    OctoTransformer tf;
    DiffusionHead head;
    std::vector<float> act_mean, act_std, act_mask;   // [action_dim] dataset stats

    bool load(const std::string& dir, const std::string& tok_dir);

    // Quantize the matmul groups OCTO_INT8 selects to symmetric W8A8; called by
    // load(). Lossy and opt-in, exactly like ACT's and SmolVLA's - see the mask
    // comment in octo_model.cpp for the groups and what is deliberately left out.
    void apply_int8();

    // primary [wnd,256,256,3] uint8 HWC, wrist [wnd,128,128,3]; timestep_mask [wnd]
    // (0 = history padding). noise [flat], z [steps*flat]: pass nullptr to draw from
    // seed. actions [horizon, action_dim]; unnormalize applies dataset stats.
    // The T5 encoding is cached per instruction string (static factor, RAM-for-time).
    void predict(const uint8_t* primary, const uint8_t* wrist, int wnd,
                 const uint8_t* timestep_mask, const std::string& instruction,
                 const float* noise, const float* z, uint64_t seed,
                 bool unnormalize, float* actions) const;

private:
    const float* lang_encode(const std::string& instruction) const;

    mutable std::unordered_map<std::string, std::vector<float>> lang_cache;
    mutable std::vector<uint8_t> win_p, win_w;
    mutable std::vector<float> win_sp, win_sw;
};

} // namespace tcpu
