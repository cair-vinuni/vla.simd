/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "config.h"
#include "nn/attention.h"
#include "nn/linear.h"
#include <cstdint>
#include <string>
#include <vector>

// Vision projection + the vision-language interaction stack (GroundingDINO's
// feature enhancer, as TurboVLA configures it).
//
// VisionProjection:  output_norm(skip(t) + mlp(input_norm(t))), mlp = 768 ->
// 1024 -gelu-> 256. Then a per-view embedding is added and the views are
// flattened into one [n_views*n_patches, 256] visual stream.
//
// Each of the 6 interaction steps is a BiAttentionBlock followed by a text
// encoder layer:
//
//   v, l = LN_v(v), LN_l(l)                 <- residual_style "normalized": the
//   dv, dl = bi_attention(v, l)                residual is the NORMALIZED tensor,
//   v, l = v + gamma_v*dv, l + gamma_l*dl      not the block input
//   l = text_layer(l)                       <- post-norm, ReLU, sub-sentence mask
//
// bi_attention is one score matrix used both ways: vision queries text with the
// text padding masked out, text queries vision unmasked (its transpose). Both
// sides project through a 1024-wide inner space with 4 heads, so head_dim is 256.

namespace tcpu {

struct FusionLayer {
    const float *ln_v_w = nullptr, *ln_v_b = nullptr;
    const float *ln_l_w = nullptr, *ln_l_b = nullptr;
    nn::Linear qv, ql, vv, vl;     // v_proj, l_proj, values_v_proj, values_l_proj
    nn::Linear out_v, out_l;
    const float *gamma_v = nullptr, *gamma_l = nullptr;
    nn::MhaMasked text_attn;       // q,k,v,out of the text encoder layer
    const float *n1_w = nullptr, *n1_b = nullptr;
    nn::Linear ff1, ff2;
    const float *n2_w = nullptr, *n2_b = nullptr;
};

struct TurboFusion {
    TurboFusionConfig cfg;
    std::vector<float> data;
    const float *in_norm_w = nullptr, *in_norm_b = nullptr;
    nn::Linear vp1, vp2, skip;
    const float *out_norm_w = nullptr, *out_norm_b = nullptr;
    const float* view_emb = nullptr;          // [n_views, hidden]
    std::vector<FusionLayer> layers;

    bool load(const std::string& dir);

    // dino [n_views, n_patches, vis_dim] -> visual [n_views*n_patches, hidden],
    // view embedding added. proj_out receives the projection output before the
    // view embedding, which is what the golden dump records.
    void project_vision(const float* dino, int n_views, int n_patches,
                        float* visual, float* proj_out) const;

    // visual [n_vis, hidden] and text [n_text, hidden] in place through the
    // interaction stack. text_pad [n_text]: 1 where the text token is padding
    // (masked out of the vision->text attention). text_mask [n_text*n_text]:
    // additive sub-sentence mask for the text layers.
    void forward(float* visual, int n_vis, float* text, int n_text,
                 const uint8_t* text_pad, const float* text_mask) const;

  private:
    mutable nn::Scratch sc;
    mutable std::vector<float> nv, nl, qvb, qlb, vvb, vlb, av, al, dv, dl, kmask, ff;
};

} // namespace tcpu
