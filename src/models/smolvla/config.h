/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// SmolVLA model dimensions (HuggingFaceVLA/smolvla_libero). See docs/04-smolvla-design.md.

namespace tcpu {

struct SmolvlaConfig {
    // VLM text tower (SmolLM2)
    int hidden      = 960;
    int n_q         = 15;
    int n_kv        = 5;
    int head_dim    = 64;
    int ffn         = 2560;
    int n_layers    = 32;
    float rms_eps   = 1e-5f;
    float rope_base = 10000.0f;   // apply_rope hardcoded default (NOT config rope_theta)

    // Action expert
    int expert_h    = 480;
    int expert_ffn  = 1280;
    int self_attn_every_n = 2;

    // Flow matching
    int chunk          = 50;
    int num_steps      = 10;
    int max_action_dim = 32;
    double min_period  = 4e-3;
    double max_period  = 4.0;

    int q_full()  const { return n_q  * head_dim; }
    int kv_full() const { return n_kv * head_dim; }
};

} // namespace tcpu
