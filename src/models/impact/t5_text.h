/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "nn/linear.h"
#include "nn/t5_encoder.h"
#include <string>
#include <vector>

// IMPACT's text tower: the shared nn::T5Encoder (here T5-small, frozen) and,
// appended past it in the same arena, three heads that turn its output into the
// two ways language enters the policy:
//
//   proj      [d_model -> dim]        the 32 text tokens the encoder sequence carries
//   text_pos  [n_text, dim]           their learned position, for the transformer's
//                                     pos array - added to queries and keys only,
//                                     never to the token, as ACT does everywhere
//   film      [d_model -> 2*film_total]  gamma and beta for the ResNet stages
//
// The FiLM head reads a mask-aware mean-pool of the T5 output, so padded
// positions do not drag the pooled vector, and it is zero-initialized: with
// (1 + gamma) modulation in the backbone that makes an untrained head exactly
// the identity.
//
// A fixed instruction corpus tokenizes to a few dozen sentencepiece pieces, so
// training can re-index the 32,128-row embedding table down to those and carry
// the mapping. A word outside it is a word the language pathway cannot see, so
// the degradation warns rather than passing silently.
//
// Everything here is an EPISODE constant. The instruction does not change within
// an episode, so ImpactModel calls encode() once from set_instruction() and every
// predict() in that episode reuses the result.

namespace tcpu {

struct ImpactTextConfig {
    int proj_dim = 512;
    int n_text = 32;
    int film_total = 960;
    long encoder_floats = 0;  // what nn::T5Encoder itself consumed
};

struct ImpactText {
    ImpactTextConfig cfg;
    nn::T5Encoder t5;
    nn::Linear proj;                 // [d_model -> proj_dim]
    nn::Linear film;                 // [d_model -> 2*film_total]
    const float* text_pos = nullptr;  // [n_text, proj_dim]
    std::vector<float> tail;          // arena past what nn::T5Encoder consumed
    std::vector<int32_t> vocab_map;   // [vocab_full], -1 outside the pruned table

    bool load(const std::string& dir, int vocab_full, int unk_id);

    int proj_dim() const { return cfg.proj_dim; }
    int n_text() const { return cfg.n_text; }
    int film_total() const { return cfg.film_total; }

    // ids [seq] in the FULL sentencepiece vocabulary -> compact [seq]. Returns
    // the number substituted with <unk>; the first call to hit any warns.
    int remap(const int* ids, int seq, int* compact) const;

    // compact ids + attn_mask [seq] (1 = real token, 0 = pad) ->
    //   tokens [seq, proj_dim]   text tokens for the encoder sequence
    //   gamma / beta [film_total]  the backbone's per-stage modulation
    void encode(const int* compact, const int* attn_mask, int seq,
                float* tokens, float* gamma, float* beta) const;

  private:
    int unk_compact = 0;
    mutable bool warned_oov = false;
    mutable std::vector<float> h, pooled, gb;
};

} // namespace tcpu
