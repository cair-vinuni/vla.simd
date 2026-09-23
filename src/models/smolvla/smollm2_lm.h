/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "config.h"
#include "nn/linear.h"
#include <string>
#include <vector>

// SmolLM2 VLM prefix tower (SmolVLA). Reimplements vla.cpp build_vlm_layer
// over the HAL. Standard
// Llama block: input_layernorm -> GQA (RoPE-NeoX, masked) -> post_attention_layernorm
// -> SwiGLU MLP. Matmul weights are bf16 in the checkpoint; nn::Linear picks the
// per-backend representation (see linear.h).

namespace tcpu {

struct VlmLayerW {
    const float* ln_in;       // [hidden]           (fp32)
    const float* ln_post;     // [hidden]           (fp32)
    nn::Linear q, k, v, o;    // attention projections (bf16 checkpoint)
    nn::Linear gate, up, down;
};

// Per-layer K/V cache entry: post-RoPE K and raw V, both [seq, n_kv, head_dim].
struct VlmKV {
    std::vector<float> k, v;
};

struct SmollmVlm {
    SmolvlaConfig cfg;
    std::vector<float> fnorms;        // fp32 region: per-layer norms + output norm
    std::vector<uint16_t> wbf;        // bf16 region (freed after init unless the backend keeps raw)
    std::vector<VlmLayerW> layers;

    // Reads <dir>/vlm.meta and <dir>/vlm.bin (produced by tools/convert_hf_safetensors.py).
    bool load(const std::string& dir);

    // Prefix forward. embs:[seq,hidden] row-major. mask:[seq,seq] additive (0 keep, -inf block).
    // pos[seq] token positions. Fills the per-layer K/V cache. embs is not modified.
    void prefix_forward(const float* embs, const float* mask, const int* pos, int seq,
                        std::vector<VlmKV>& kv_out) const;
};

} // namespace tcpu
