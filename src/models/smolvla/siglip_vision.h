/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "nn/encoder.h"
#include <cstdint>
#include <string>
#include <vector>

// SmolVLM2-500M vision tower (SigLIP) + pixel-shuffle connector. Standard SigLIP
// encoder: Conv patch embed + learned position embed, 12 layers (LayerNorm, MHA with
// bias, GELU-tanh MLP), post_layernorm. Connector: pixel shuffle (scale_factor 4)
// then modality projection (Linear, no bias) -> LM hidden. 1024 patches -> 64 tokens.
// Mirrors transformers SmolVLMVisionTransformer + SmolVLMConnector. Each layer is a
// nn::EncoderLayer (the block structure matches exactly); full attention is a zero
// additive mask.

namespace tcpu {

struct VitConfig {
    int hidden = 768;
    int n_heads = 12;
    int head_dim = 64;
    int inter = 3072;
    int n_layers = 12;
    int patch = 16;
    int img = 512;
    int n_patches = 1024;
    float ln_eps = 1e-6f;
    int scale_factor = 4;
    int mm_out = 960;
    int n_img_tok = 64;
    int patch_dim() const { return 3*patch*patch; }
    int shuffled_dim() const { return hidden*scale_factor*scale_factor; }
};

struct SiglipVision {
    VitConfig cfg;
    std::vector<float> fnorms;        // fp32 region: biases, norms, position embed
    std::vector<uint16_t> wbf;        // bf16 region (freed after init unless the backend keeps raw)
    nn::Linear patch_lin;             // patch embed as a GEMM over extracted patches
    const float* pos_emb = nullptr;
    std::vector<nn::EncoderLayer> layers;
    const float *post_ln_w = nullptr, *post_ln_b = nullptr;
    nn::Linear mm_proj;               // [mm_out, shuffled_dim], no bias
    mutable nn::Scratch scratch;      // reused across layers and views

    // Reads <dir>/vit.meta and <dir>/vit.bin (tools/convert_hf_safetensors.py).
    bool load(const std::string& dir);

    // pixels: [3, img, img] CHW, already normalized to [-1,1].
    // out: [n_img_tok, mm_out] image embeddings (before the sqrt(hidden) prefix scaling).
    void encode(const float* pixels, float* out) const;

    // Same, against a caller-owned scratch. The member `scratch` is shared, so
    // views encoded concurrently (TCPU_VIEW_THREADS) must each bring their own.
    void encode(const float* pixels, float* out, nn::Scratch& s) const;
};

} // namespace tcpu
