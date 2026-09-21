/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// TurboVLA dims, one struct per .meta file written by
// tools/turbovla/convert_turbovla.py. Defaults are the released LIBERO
// checkpoints (DINOv3 ViT-B/16 @256, BERT-base, 6 interaction layers, a 3-layer
// ACT decoder over a 12-step chunk); load() overwrites every field from the meta
// and then checks the shapes close, so nothing here is load-bearing.
// See docs/12-turbovla-design.md.

namespace tcpu {

// vision.meta - DINOv3 ViT-B/16. prefix = cls + register tokens, dropped from
// the output. LayerScale is folded into o_proj / down_proj by the converter, so
// there is no lambda tensor at runtime.
struct TurboVisionConfig {
    int hidden = 768, n_heads = 12, head_dim = 64, inter = 3072;
    int n_layers = 12, patch = 16, prefix = 5;
    float rope_theta = 100.0f, ln_eps = 1e-5f;

    int patch_dim() const { return 3*patch*patch; }
};

// text.meta - BERT-base encoder. The 768 -> fusion-dim text projection sits at
// the end of the same arena.
struct TurboTextConfig {
    int hidden = 768, n_heads = 12, head_dim = 64, inter = 3072;
    int n_layers = 12, vocab = 30522, max_pos = 512, type_vocab = 2;
    float ln_eps = 1e-12f;
};

// fusion.meta - vision projection + view embedding + N x (BiAttentionBlock,
// text encoder layer). embed is the bi-attention inner width, which is 4x the
// model width here, so its head_dim is embed/fusion_heads = 256.
struct TurboFusionConfig {
    int hidden = 256, embed = 1024, n_layers = 6;
    int fusion_heads = 4, text_heads = 4, text_ff = 1024;
    int vis_dim = 768, vis_mlp = 1024, n_views = 2;
    float ln_eps = 1e-5f;

    int fusion_head_dim() const { return embed/fusion_heads; }
    int text_head_dim() const { return hidden/text_heads; }
};

// head.meta - state projection + pre-norm ACT decoder + action MLP.
struct TurboHeadConfig {
    int hidden = 256, n_layers = 3, n_heads = 8, ff = 2048;
    int chunk = 12, action_dim = 7, state_dim = 8;
    int state_tokens = 2, state_hidden = 256, mlp_hidden = 512, mlp_layers = 3;
    float ln_eps = 1e-5f;

    int head_dim() const { return hidden/n_heads; }
};

// config.meta - everything that is not a weight: shapes, image normalization,
// and the WordPiece ids the sub-sentence mask keys off.
struct TurboVlaConfig {
    int img = 256, n_views = 2;
    int text_pad = 21;        // the padded text length every instruction reaches
    int max_text_len = 256;   // the tokenizer's cap, before per-instruction padding
    int sub_sentence = 1;     // 1 = BERT sees the sub-sentence mask + custom positions
    int chunk = 12, action_dim = 7, state_dim = 8;
    float img_mean[3] = {0.485f, 0.456f, 0.406f};
    float img_std[3]  = {0.229f, 0.224f, 0.225f};
    int cls_id = 101, sep_id = 102, dot_id = 1012, question_id = 1029;
    int pad_id = 0, unk_id = 100, max_wordpiece = 18;
    float gripper_deadband = 0.0f;
};

} // namespace tcpu
