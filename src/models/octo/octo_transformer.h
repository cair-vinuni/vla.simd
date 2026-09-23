/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "nn/encoder.h"
#include "nn/linear.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Octo ViT-S block transformer. Token sequence (window w, repeat_task_tokens=true):
//   [task_language(16) | per t: obs_primary(256) obs_wrist(64) obs_language(16) readout(1)]
// Learned positional embeddings per group (obs groups indexed by timestep, truncated
// from max_horizon). Block-causal attention (use_correct_attention=true semantics).
// Weights: octo.meta/octo.bin from tools/convert_octo.py.
// Built from nn::EncoderLayer / nn::Linear - all kernel and layout choices live
// in the HAL; this file only assembles tokens and drives the blocks.

namespace tcpu {

struct OctoConfig {
    int d = 384, n_layers = 12, heads = 6, head_dim = 64, mlp = 1536;
    int max_horizon = 10, n_task = 16, tok_primary = 256, tok_wrist = 64, n_readout = 1;
    int t5_dim = 768, stem_dim = 512;
    float ln_eps = 1e-6f;
    // false = gelu-tanh (flax nn.gelu, i.e. the JAX Octo checkpoints), true =
    // exact erf (torch F.gelu, i.e. anything converted through lerobot). Set
    // from `gelu_erf` in octo.meta; see nn::EncoderLayer::Gelu.
    bool gelu_erf = false;
};

struct OctoTransformer {
    OctoConfig cfg;
    std::vector<float> data;
    nn::Linear proj_task, proj_prim, proj_wrist;   // token projections
    const float *pos_task = nullptr;     // [n_task, d]
    const float *pos_prim = nullptr;     // [max_horizon, tok_primary, d]
    const float *pos_wrist = nullptr;    // [max_horizon, tok_wrist, d]
    const float *pos_readout = nullptr;  // [max_horizon, n_readout, d]
    std::vector<nn::EncoderLayer> layers;
    const float *final_s = nullptr, *final_b = nullptr;
    // additive mask per (wnd, timestep_mask) config - static, built once (RAM-for-time)
    mutable std::unordered_map<int, std::vector<float>> mask_cache;
    mutable nn::Scratch scratch;   // persistent activation buffers (incl. the K^T panel)

    bool load(const std::string& dir);

    int tokens_per_step() const { return cfg.tok_primary+cfg.tok_wrist+cfg.n_task+cfg.n_readout; }
    int total_tokens(int wnd) const { return cfg.n_task+wnd*tokens_per_step(); }
    // wnd is unnamed: the readout of timestep t sits at the end of that step's
    // block, so the window length does not enter the index. Kept in the
    // signature because every caller has it and dropping it reads as a bug.
    int readout_index(int /*wnd*/, int t) const { return cfg.n_task+(t+1)*tokens_per_step()-1; }

    // Boolean keep mask (1 = attend, 0 = block): block-causal rules AND key-side pad.
    // timestep_mask[t] = 1 for a real timestep, 0 for history padding.
    void build_mask(int wnd, const uint8_t* timestep_mask, bool wrist, uint8_t* keep) const;

    // t5_out [n_task, t5_dim]; stem_p [wnd, tok_primary, stem_dim]; stem_w [wnd,
    // tok_wrist, stem_dim] -> out [total_tokens(wnd), d] (post final LayerNorm).
    // last_token_only: inference fast path - the final layer computes attention/MLP
    // only for the last readout token (all K/V still computed); only that row of
    // `out` is written. Exact for that row.
    void forward(const float* t5_out, const float* stem_p, const float* stem_w,
                 int wnd, const uint8_t* timestep_mask, float* out,
                 bool last_token_only = false) const;
};

} // namespace tcpu
