/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "action_head.h"
#include "bert_text.h"
#include "config.h"
#include "dinov3_vision.h"
#include "fusion.h"
#include "tokenizer/bert_tokenizer.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// TurboVLA orchestration: two camera frames + an instruction + the robot state
// -> a chunk of actions. Everything the reference policy does around the network
// lives here - image normalization, the per-instruction text padding, state
// normalization, and the LIBERO min/max action mapping.
//
//   frames      -> DINOv3 (per view) -> vision projection + view embedding
//   instruction -> WordPiece -> BERT  -> text projection
//        both -> 6x fusion -> condition = [fused visual | fused text]
//        condition + 2 state tokens -> ACT decoder -> MLP -> tanh
//
// Text padding is part of the model's output, not an optimization. The decoder
// cross-attends the padded text rows with no mask, and the checkpoint pins a
// padded length per training instruction (11/14/21 for the LIBERO suites), so
// the same instruction must reach the same length here. Rows past that length
// are literal zeros out of BERT, which the text projection turns into its own
// bias. text_pad.txt carries the table; an instruction that is not in it falls
// back to config.meta's text_pad, exactly as the reference's dict lookup does.
//
// Weights from tools/turbovla/convert_turbovla.py; see docs/12-turbovla-design.md.

namespace tcpu {

// Every intermediate of the last predict(), kept so a parity checker can compare
// module by module instead of only at the action. A few MB, written once per
// inference; see vla_turbovla_tensor in include/vla_simd.h.
struct TurboVlaTrace {
    std::vector<float> pixel_values;    // [n_views, 3, img, img]
    std::vector<float> dino_tokens;     // [n_views, n_patches, vis_dim]
    std::vector<float> vision_proj;     // [n_views, n_patches, hidden]
    std::vector<float> visual_tokens;   // [n_views*n_patches, hidden]
    std::vector<float> input_ids;       // [text_pad]  (ids are exact in fp32)
    std::vector<float> position_ids;    // [text_pad]
    std::vector<float> text_pad_mask;   // [text_pad]  1 = padding
    std::vector<float> self_attn;       // [text_pad, text_pad] 1 = attend
    std::vector<float> bert_hidden;     // [text_pad, text hidden]
    std::vector<float> text_tokens;     // [text_pad, hidden]
    std::vector<float> fused_visual;    // [n_views*n_patches, hidden]
    std::vector<float> fused_text;      // [text_pad, hidden]
    std::vector<float> condition;       // [n_views*n_patches + text_pad, hidden]
    std::vector<float> state_norm;      // [state_dim]
    std::vector<float> state_tokens;    // [state_tokens, hidden]
    std::vector<float> actions_norm;    // [chunk, action_dim]
};

struct TurboVlaModel {
    TurboVlaConfig cfg;
    Dinov3Vision vision;
    BertText text;
    TurboFusion fusion;
    TurboActionHead head;
    BertTokenizer tok;

    // LIBERO eval protocol (turbovla/evaluation/policy.py), from stats.bin.
    std::vector<float> proprio_mean, proprio_std, action_min, action_max;
    std::map<std::string, int> pad_by_instruction;

    bool load(const std::string& dir);

    int chunk() const { return head.cfg.chunk; }
    int action_dim() const { return head.cfg.action_dim; }
    int state_dim() const { return head.cfg.state_dim; }
    int n_views() const { return cfg.n_views; }
    int img_size() const { return cfg.img; }
    int n_patches() const { return vision.n_patches(); }
    int n_visual() const { return cfg.n_views*vision.n_patches(); }
    int text_pad() const { return cfg.text_pad; }
    int hidden() const { return fusion.cfg.hidden; }

    // The padded text length this instruction is pinned to.
    int pad_length(const std::string& instruction) const;

    // frames: n_views uint8 RGB HWC images at img_size x img_size, in the
    // checkpoint's view order (agentview, wrist). state [state_dim] in raw robot
    // units. actions [chunk*action_dim]; unnormalize maps them to env units
    // (arm from the LIBERO min/max, gripper to a hard +-1).
    void predict(const uint8_t* frames, const float* state, const std::string& instruction,
                 bool unnormalize, float* actions) const;

    const TurboVlaTrace& trace() const { return tr; }

  private:
    void encode_text(const std::string& instruction) const;
    mutable TurboVlaTrace tr;
    mutable std::vector<int> ids_i, pos_i;
    mutable std::vector<float> gmask, ghidden, memory;
};

} // namespace tcpu
