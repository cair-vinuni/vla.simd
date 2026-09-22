/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// IMPACT (Instruction-Modulated Perception + ACTion chunking) dims, one struct
// per .meta file written by tools/convert_impact.py. Defaults are the
// reference configuration (ResNet-18 + FiLM, a frozen T5-small text tower, a
// 6-layer encoder over 634 tokens and a 4-layer decoder over a 50-step chunk);
// load() overwrites every field from the meta and then checks the shapes close,
// so nothing here is load-bearing.
// See docs/14-impact-design.md.

namespace tcpu {

// impact.meta - the DETR-style transformer. Identical to ACT's apart from
// n_enc / n_dec / chunk and the text tail (n_text), which is the only new field
// in the sequence layout: [latent(1) | state(1) | cams(n*fh*fw) | text(n_text)].
struct ImpactConfig {
    int dim = 512, heads = 8, head_dim = 64, ff = 3200;
    int n_enc = 6, n_dec = 4;
    int chunk = 50, state_dim = 6, action_dim = 6;
    int n_1d = 2;             // learned 1D pos embeddings: latent + state token
    int n_text = 32;          // text tokens appended to the encoder sequence
    float ln_eps = 1e-5f;
};

// text.meta - the shared nn::T5Encoder (here T5-small) plus, appended past it in
// the same arena, the d_model -> dim text projection, the learned text position
// table and the FiLM head. `film_ch` records the per-stage channel counts the
// head emits gamma/beta for, so the backbone and the head cannot disagree
// silently about how the flat gamma/beta buffer is cut up.
struct ImpactTextConfig {
    int proj_dim = 512;       // == ImpactConfig::dim
    int n_text = 32;
    int n_film = 4;           // FiLM points (ResNet-18 stages)
    int film_total = 960;     // sum of film_ch == 64+128+256+512
    long encoder_floats = 0;  // what nn::T5Encoder itself consumed
};

} // namespace tcpu
