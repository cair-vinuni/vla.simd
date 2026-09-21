/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "impact_transformer.h"
#include "resnet_film.h"
#include "t5_text.h"
#include "tokenizer/t5_tokenizer.h"
#include <cstdint>
#include <string>
#include <vector>

// IMPACT orchestration: camera frames + joint state + an instruction ->
// chunk x action_dim. Everything the lerobot processor pipeline does around the
// network lives here: image rescale + per-channel normalization, state
// normalization, tokenization, and the un-normalization of the predicted actions
// (MEAN_STD, eps 1e-8). Weights + stats from tools/impact/convert_impact.py.
//
// The instruction is an EPISODE constant, and this class is built around that.
// set_instruction() runs the whole language pathway once - tokenize, remap,
// T5-small, the text projection and the FiLM head - and caches its three outputs
// (text tokens, gamma, beta). predict() then costs nothing for language beyond
// the 32 extra columns in the encoder. Calling predict() without a prior
// set_instruction() is an error rather than a silent unconditioned rollout.

namespace tcpu {

struct ImpactModel {
    int img_h = 480, img_w = 640, n_cams = 2;
    float norm_eps = 1e-8f;
    std::vector<std::string> cam_names;

    ResNetFilm backbone;      // shared by every camera
    ImpactText text;
    T5Tokenizer tok;
    ImpactTransformer tf;
    std::vector<float> state_mean, state_std, action_mean, action_std;
    std::vector<float> img_mean, img_std;   // [n_cams, 3]

    bool load(const std::string& dir);

    // Quantize the matmul groups IMPACT_INT8 selects to symmetric W8A8; called
    // by load(). Lossy and opt-in, exactly as ACT's and Octo's - see the mask
    // comment in impact_model.cpp for the groups and what is left out.
    void apply_int8();

    int chunk() const { return tf.cfg.chunk; }
    int action_dim() const { return tf.cfg.action_dim; }
    int state_dim() const { return tf.cfg.state_dim; }
    int n_text() const { return tf.cfg.n_text; }

    // Run the language pathway and cache it for the episode. Re-encoding is
    // skipped when the string is unchanged, so a caller may pass the instruction
    // on every predict() and still pay for it once per episode. Returns false
    // only if the model is not loaded; an out-of-vocabulary word warns (once) and
    // is substituted rather than failing, exactly as the reference does.
    bool set_instruction(const std::string& instruction);

    // The tokenization set_instruction() produced, for the parity harness:
    // ids/compact/mask are [n_text] and `real` is the unpadded length.
    const std::vector<int>& token_ids() const { return ids; }
    const std::vector<int>& compact_ids() const { return compact; }
    const std::vector<int>& token_mask() const { return mask; }
    int n_real_text() const { return n_real; }
    const std::vector<float>& film_gamma() const { return gamma; }
    const std::vector<float>& film_beta() const { return beta; }

    // images: n_cams frames, uint8 HWC at img_h x img_w, in the checkpoint's
    // camera order (cam_names). state [state_dim] in raw robot units.
    // actions [chunk, action_dim]; unnormalize applies the dataset stats.
    void predict(const uint8_t* const* images, const float* state,
                 bool unnormalize, float* actions) const;

    // Backbone only, for one camera: FiLM-modulated feature map tokens
    // [fh*fw, backbone.out_channels()]. predict() calls this per camera.
    void encode_view(const uint8_t* image, int cam, float* feat) const;

    void feat_size(int* fh, int* fw) const { backbone.feat_size(img_h, img_w, fh, fw); }

  private:
    // The episode's language state. `have_text` is what makes a missing
    // set_instruction() loud instead of a zero-conditioned rollout.
    bool have_text = false;
    std::string cached_instruction;
    std::vector<int> ids, compact, mask;
    int n_real = 0;
    std::vector<float> text_tok, gamma, beta, text_hidden;

    mutable std::vector<ImpactBackboneScratch> bscratch;
    mutable std::vector<std::vector<float>> norm, feats;
};

} // namespace tcpu
