/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "config.h"
#include "smollm2_lm.h"   // VlmKV
#include "nn/linear.h"
#include <string>
#include <vector>

// SmolVLA action expert + flow-matching denoise. Reimplements vla.cpp
// build_expert_self_attn_layer / build_expert_cross_attn_layer + the denoise loop
// (third_party/vla.cpp/src/models/smolvla.cpp:680-749, 1253-1279) over the HAL.
//
// The expert runs one stream (the action suffix, `chunk` tokens). Even layers are
// self-attention (attend the VLM prefix K/V cache concatenated with the suffix's own
// K/V); odd layers are cross-attention (reproject the cached VLM K/V through the
// expert's k/v proj, attend prefix only, rebased positions). Flow matching: Euler
// integrate x <- x + dt*v over num_steps. Weights are fp32 in the checkpoint.

namespace tcpu {

struct ExpertLayerW {
    bool is_self_attn;
    const float* ln_in;      // [expert_h]
    nn::Linear q;            // [q_full, expert_h]
    nn::Linear k, v;         // self: [kv_full, expert_h]; cross: [kv_full, kv_full]
    nn::Linear o;            // [expert_h, q_full]
    const float* ln_post;    // [expert_h]
    nn::Linear gate, up, down;
};

struct FlowW {
    const float *ain_w, *ain_b;    // action_in_proj      [expert_h, max_action_dim], [expert_h]
    const float *at1_w, *at1_b;    // action_time_mlp_in  [expert_h, 2*expert_h],     [expert_h]
    const float *at2_w, *at2_b;    // action_time_mlp_out [expert_h, expert_h],       [expert_h]
    const float *aout_w, *aout_b;  // action_out_proj     [max_action_dim, expert_h], [max_action_dim]
};

// Buffers that live across the denoise loop instead of being rebuilt per step.
//
// Mainly the self-attention K/V concat: only the last `chunk` rows change
// between steps, but the whole [n_prefix+chunk, kv_full] pair was recopied every
// layer of every step, ~72 MB per query at n_prefix 177. The rest are the
// per-step activations, ~1 MB of malloc/free per step.
struct DenoiseScratch {
    std::vector<float> h, hn, q, attn, o, hn2, g, u, gu, dn, ks, vs;
    std::vector<float> aemb, te, at, mlp1, hf;   // embed_suffix + out norm, per step
    std::vector<std::vector<float>> Kf, Vf;   // per self-attn layer, prefix half written once
};

struct ActionExpert {
    SmolvlaConfig cfg;
    std::vector<float> blob;
    std::vector<ExpertLayerW> layers;
    const float* out_norm = nullptr;  // [expert_h]
    FlowW flow{};
    mutable DenoiseScratch ds;        // one denoise at a time, like the rest of the engine

    // Reads <dir>/aex.meta and <dir>/aex.bin (tools/convert_hf_safetensors.py).
    bool load(const std::string& dir);

    // action_in_proj + sinusoidal time emb + action_time_mlp. x_t:[chunk,max_action_dim]
    // -> suffix:[chunk,expert_h].
    void embed_suffix(const float* x_t, float time, float* suffix) const;

    // One denoise step. kv = per-layer VLM K/V cache ([n_prefix, kv_full] each).
    // mask_full:[chunk, n_prefix+chunk] additive; mask_prefix:[chunk, n_prefix] additive.
    // pos_full:[chunk] self-attn positions; pos_rebased:[chunk] cross-attn positions.
    // cK/cV:[n_layers][n_prefix*kv_full] precomputed cross-attn K/V (empty entry => compute
    // locally). x_t:[chunk,max_action_dim] -> v_t:[chunk,max_action_dim].
    // Buffers come from `ds`, which denoise() sizes and whose K/V prefix half it
    // fills once; a direct caller must call prepare_denoise first.
    void denoise_step(const std::vector<VlmKV>& kv, int n_prefix, const float* x_t, float time,
                      const float* mask_full, const float* mask_prefix,
                      const int* pos_full, const int* pos_rebased, float* v_t,
                      const std::vector<std::vector<float>>& cK,
                      const std::vector<std::vector<float>>& cV) const;

    // Size `ds` and write the step-invariant half of the self-attention K/V concat.
    void prepare_denoise(const std::vector<VlmKV>& kv, int n_prefix) const;

    // Full flow-matching denoise. noise:[chunk,max_action_dim]. mask_full/pos_full as dumped
    // by the reference (pos_rebased and mask_prefix are derived here). Writes actions to out.
    void denoise(const std::vector<VlmKV>& kv, int n_prefix, const float* noise,
                 const float* mask_full, const int* pos_full, float* out) const;
};

} // namespace tcpu
