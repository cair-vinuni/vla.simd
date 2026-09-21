/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "act_transformer.h"
#include "resnet_backbone.h"
#include <cstdint>
#include <string>
#include <vector>

// ACT orchestration: camera frames + joint state -> chunk x action_dim.
// Everything the lerobot processor pipeline does around the network lives here:
// image rescale + per-channel normalization, state normalization, and the
// un-normalization of the predicted actions (MEAN_STD, eps 1e-8).
// Weights + stats from tools/act/convert_act.py.

namespace tcpu {

struct ActModel {
    int img_h = 480, img_w = 640, n_cams = 2;
    float norm_eps = 1e-8f;
    std::vector<std::string> cam_names;

    ResNetBackbone backbone;   // shared by every camera
    ActTransformer tf;
    std::vector<float> state_mean, state_std, action_mean, action_std;
    std::vector<float> img_mean, img_std;   // [n_cams, 3]

    bool load(const std::string& dir);

    int chunk() const { return tf.cfg.chunk; }
    int action_dim() const { return tf.cfg.action_dim; }
    int state_dim() const { return tf.cfg.state_dim; }

    // images: n_cams frames, uint8 HWC at img_h x img_w, in the checkpoint's camera
    // order (cam_names). state [state_dim] in raw robot units.
    // actions [chunk, action_dim]; unnormalize applies the dataset stats.
    void predict(const uint8_t* const* images, const float* state,
                 bool unnormalize, float* actions) const;

    // Backbone only, for one camera: normalized feature map tokens
    // [fh*fw, backbone.out_channels()]. predict() calls this per camera.
    void encode_view(const uint8_t* image, int cam, float* feat) const;

    void feat_size(int* fh, int* fw) const { backbone.feat_size(img_h, img_w, fh, fw); }

  private:
    mutable std::vector<BackboneScratch> bscratch;
    mutable std::vector<std::vector<float>> norm, feats;
};

} // namespace tcpu
