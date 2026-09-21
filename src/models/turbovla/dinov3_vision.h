/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "config.h"
#include "nn/linear.h"
#include <string>
#include <vector>

// DINOv3 ViT-B/16, the TurboVLA vision tower (fine-tuned, so its weights ship
// inside the TurboVLA checkpoint rather than coming from the gated HF repo).
//
// One block: LN -> MHA with 2D RoPE -> residual -> LN -> gelu(erf) MLP ->
// residual. Two things separate it from nn::EncoderLayer, which is why the loop
// lives here: the rotary embedding is applied to q/k between the projections and
// the attention, and the MLP activation is exact GELU, not the tanh
// approximation nn::EncoderLayer fuses into its GEMM epilogue.
//
// LayerScale is folded into o_proj / down_proj by the converter (lambda1 is a
// per-channel scale on exactly the channels those two write), so a block here is
// plain pre-norm shape. The backbone's final LayerNorm is NOT loaded: TurboVLA
// reads hidden_states[-1], which transformers records before it.
//
// The token sequence is [cls | 4 registers | 256 patches]; only the patch rows
// are returned, and only they carry RoPE.

namespace tcpu {

struct Dinov3Layer {
    const float *ln1_w = nullptr, *ln1_b = nullptr;
    const float *ln2_w = nullptr, *ln2_b = nullptr;
    nn::Linear wq, wk, wv, wo;   // wk has no bias (key_bias=false)
    nn::Linear up, down;
};

struct Dinov3Vision {
    TurboVisionConfig cfg;
    std::vector<float> data;                 // weight arena, kept mapped
    const float* cls_token = nullptr;        // [hidden]
    const float* reg_tokens = nullptr;       // [prefix-1, hidden]
    nn::Linear patch;                        // conv 16x16 stride 16 == linear over patches
    std::vector<Dinov3Layer> layers;

    // RoPE tables for the fixed grid, built once at load: [n_patches, head_dim].
    std::vector<float> rope_cos, rope_sin;
    int grid = 0;                            // patches per side

    bool load(const std::string& dir, int img_size);

    int n_patches() const { return grid*grid; }
    int n_tokens() const { return cfg.prefix + n_patches(); }

    // pixels [n_views, 3, img, img] normalized CHW -> out [n_views, n_patches,
    // hidden] (the patch rows of hidden_states[-1], prefix dropped).
    //
    // The views go through as ONE batch: they share every weight, and only the
    // attention is per-view (view 0's tokens must not see view 1's). At 261
    // tokens a view the GEMMs are weight-streaming bound, so one pass over the
    // 342 MB arena for both views beats two - the encoder is a per-token
    // function everywhere else, so this changes no value.
    void encode(const float* pixels, int n_views, float* out) const;

  private:
    mutable std::vector<float> tok, patches, pemb, h, q, k, v, att, ff;
};

} // namespace tcpu
