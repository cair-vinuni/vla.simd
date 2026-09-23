/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "config.h"
#include "nn/attention.h"
#include "nn/linear.h"
#include <string>
#include <vector>

// BERT-base text encoder plus the 768 -> 256 text projection, driven the way
// TurboVLA drives it (inherited from GroundingDINO):
//
//   * the attention mask BERT sees is NOT the padding mask. It is the
//     sub-sentence mask: tokens between two special tokens ([CLS] [SEP] . ?)
//     attend only each other, and every other row is the identity. Padding rows
//     therefore attend only themselves and still produce a real hidden state.
//   * position ids restart at 0 inside each sub-sentence, so they are supplied,
//     not implied.
//
// Both come from build_masks(), which is a transcription of
// generate_masks_with_special_tokens including its edge cases - a special token
// in the last column sets only its own diagonal, which leaves the whole
// preceding span on the identity. That is load-bearing behaviour, not a bug to
// fix here.
//
// A block is post-norm: x = LN(x + attn(x)); x = LN(x + mlp(x)), gelu(erf).

namespace tcpu {

struct BertLayer {
    nn::MhaMasked attn;              // q,k,v,out_dense in one
    const float *ln1_w = nullptr, *ln1_b = nullptr;   // attention.output.LayerNorm
    nn::Linear up, down;
    const float *ln2_w = nullptr, *ln2_b = nullptr;   // output.LayerNorm
};

struct BertText {
    TurboTextConfig cfg;
    int out_dim = 256;                       // the fusion width the projection targets
    std::vector<float> data;
    const float* word_emb = nullptr;         // [vocab, hidden]
    const float* pos_emb = nullptr;          // [max_pos, hidden]
    const float* type_emb = nullptr;         // [type_vocab, hidden]
    const float *emb_ln_w = nullptr, *emb_ln_b = nullptr;
    std::vector<BertLayer> layers;
    nn::Linear text_proj;                    // [out_dim, hidden]

    bool load(const std::string& dir, int fusion_hidden);

    // Sub-sentence mask + position ids for one tokenized row. ids [n];
    // position_ids [n]; mask [n*n] additive (0 keep, -inf block).
    void build_masks(const int* ids, int n, const TurboVlaConfig& tc,
                     int* position_ids, float* mask) const;

    // ids/position_ids [n], mask [n*n] additive -> hidden [n, cfg.hidden].
    void encode(const int* ids, const int* position_ids, const float* mask, int n,
                float* hidden) const;

    // hidden [n, cfg.hidden] -> tokens [n, out_dim].
    void project(const float* hidden, int n, float* tokens) const;

  private:
    mutable nn::Scratch sc;
    mutable std::vector<float> x, h, ff;
};

} // namespace tcpu
