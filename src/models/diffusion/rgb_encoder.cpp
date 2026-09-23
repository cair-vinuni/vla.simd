/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "models/diffusion/rgb_encoder.h"
#include "models/arena.h"
#include "ops/conv_ops.h"
#include "ops/lm_ops.h"
#include <vector>

namespace tcpu {

// numpy's linspace, not torch's. The reference builds the coordinate grid with
// np.linspace and notes that torch.linspace "behaves slightly differently and
// causes a small degradation in pc_success of pre-trained models" -- so the
// difference is real enough that somebody measured it. numpy accumulates
// start + i*step in float64 and then pins the endpoint exactly; reproducing that
// and casting once at the end is what keeps our keypoint coordinates on the
// reference's values rather than a ulp off across the whole grid.
static void linspace_np(std::vector<float>& v, double start, double stop, int n) {
    v.resize(n);
    if (n == 1) { v[0] = (float)start; return; }
    const double step = (stop - start)/(double)(n - 1);
    for (int i=0; i<n; i++) v[i] = (float)(start + step*(double)i);
    v[n-1] = (float)stop;
}

bool DPRgbEncoder::load(const std::string& dir, const std::string& name,
                        const DPConfig& c) {
    cfg = c;
    if (!backbone.load(dir, name + "_backbone")) return false;

    // Feature-map size after the backbone, from the size the encoder is actually
    // fed (post crop), because the grid is built for that map.
    const int in_h = cfg.crop_h > 0 ? cfg.crop_h : cfg.img_h;
    const int in_w = cfg.crop_w > 0 ? cfg.crop_w : cfg.img_w;
    backbone.feat_size(in_h, in_w, &fh, &fw);
    if (fh <= 0 || fw <= 0) return false;

    if (!read_arena(dir + "/" + name + ".bin", data)) return false;

    const int C  = backbone.out_channels();
    const int K  = cfg.num_keypoints;
    const int F  = K*2;

    ArenaCursor<float> take{data};

    // 1x1 conv to keypoint channels, stored as a plain [K, C] linear.
    const float* kw = take((size_t)K*C);
    const float* kb = take(K);
    // Output projection over the flattened keypoint coordinates.
    const float* ow = take((size_t)F*F);
    const float* ob = take(F);
    if (!take.done()) return false;

    to_keypoints.init(kw, kb, K, C, nn::Linear::Role::Generic);
    out.init(ow, ob, F, F, nn::Linear::Role::Generic);

    // Normalized coordinate grid, x varying fastest (row-major over the map).
    std::vector<float> px, py;
    linspace_np(px, -1.0, 1.0, fw);
    linspace_np(py, -1.0, 1.0, fh);
    grid.resize((size_t)fh*fw*2);
    for (int y=0; y<fh; y++)
        for (int x=0; x<fw; x++) {
            grid[((size_t)y*fw + x)*2 + 0] = px[x];
            grid[((size_t)y*fw + x)*2 + 1] = py[y];
        }
    return true;
}

void DPRgbEncoder::forward(const float* x, BackboneScratch& s, float* feat) const {
    const int C   = backbone.out_channels();
    const int K   = cfg.num_keypoints;
    const int npx = fh*fw;
    const int in_h = cfg.crop_h > 0 ? cfg.crop_h : cfg.img_h;
    const int in_w = cfg.crop_w > 0 ? cfg.crop_w : cfg.img_w;

    std::vector<float> fmap((size_t)npx*C);
    backbone.forward(x, in_h, in_w, s, fmap.data());

    std::vector<float> kp((size_t)npx*K);
    to_keypoints.forward(kp.data(), fmap.data(), npx);

    spatial_softmax(feat, kp.data(), grid.data(), npx, K);

    std::vector<float> tmp(K*2);
    out.forward(tmp.data(), feat, 1);
    relu(tmp.data(), K*2);
    for (int i=0; i<K*2; i++) feat[i] = tmp[i];
}

} // namespace tcpu
