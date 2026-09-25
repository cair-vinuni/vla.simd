/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "act_model.h"
#include "hal/common/env.h"
#include "hal/common/threads.h"
#include "nn/encoder.h"
#include "io/files.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
using std::size_t;

namespace tcpu {

bool ActModel::load(const std::string& dir) {
    const io::Mount mount(dir);
    if (!mount.ok()) return false;
    io::InFile cfgf(dir + "/config.txt");
    if (!cfgf) return false;
    std::string line;
    cam_names.clear();
    while (std::getline(cfgf, line)) {
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if      (key == "img_h"   ) ss >> img_h;
        else if (key == "img_w"   ) ss >> img_w;
        else if (key == "n_cams"  ) ss >> n_cams;
        else if (key == "norm_eps") ss >> norm_eps;
        else if (key.rfind("cam", 0) == 0) {
            std::string name;
            ss >> name;
            cam_names.push_back(name);
        }
    }

    backbone.prof = tf.prof = hal::env::int_env("ACT_PROFILE", 0);
    if (!backbone.load(dir)) return false;

    // ACT_INT8 bit 32 routes every backbone conv through the W8A8 kernel. The stem
    // is included: it is the one conv reading real camera pixels, and those are
    // already a quantized 8-bit signal to begin with.
    // ACT_I8_CONV_FROM=N keeps the first N conv stages (0 = the stem, 1.. = the
    // basic blocks) in fp32 and quantizes the rest. A per-tensor activation scale
    // costs more precision the sparser and more outlier-heavy a feature map is, so
    // which end of the network to protect is a measurement, not a guess.
    // ACT_I8_CONV_TO=N stops quantizing at stage N (exclusive); -1 = to the end.
    const int i8 = hal::env::int_env("ACT_INT8", 0);
    if (i8 & 32)
        backbone.quantize_convs(hal::env::int_env("ACT_I8_CONV_FROM", 0),
                                hal::env::int_env("ACT_I8_CONV_TO", -1));
    if (!tf.load(dir, backbone.out_channels(), i8)) return false;

    io::InFile st(dir + "/stats.bin", std::ios::binary);
    if (!st) return false;
    const int sd = tf.cfg.state_dim;
    const int ad = tf.cfg.action_dim;
    std::vector<float> raw((size_t)2*sd + 2*ad + (size_t)n_cams*6);
    st.read((char*)raw.data(), raw.size()*sizeof(float));
    if (!st) return false;

    const float* p = raw.data();
    state_mean.assign(p, p+sd);
    p += sd;
    state_std.assign(p, p+sd);
    p += sd;
    action_mean.assign(p, p+ad);
    p += ad;
    action_std.assign(p, p+ad);
    p += ad;
    img_mean.resize((size_t)n_cams*3);
    img_std.resize((size_t)n_cams*3);
    for (int c=0; c<n_cams; c++) {
        for (int i=0; i<3; i++) img_mean[(size_t)c*3+i] = *p++;
        for (int i=0; i<3; i++) img_std [(size_t)c*3+i] = *p++;
    }

    bscratch.resize(n_cams);
    norm.resize(n_cams);
    feats.resize(n_cams);
    return true;
}

void ActModel::encode_view(const uint8_t* image, int cam, float* feat) const {
    const size_t npx = (size_t)img_h*img_w;
    std::vector<float>& nx = norm[cam];
    if (nx.size() < npx*3) nx.resize(npx*3);

    // pixels -> [0,1] -> (x - mean)/(std + eps), same op order as the lerobot
    // NormalizerProcessorStep so the backbone sees bit-identical inputs
    const float* mean = img_mean.data()+(size_t)cam*3;
    const float* std_ = img_std.data()+(size_t)cam*3;
    float* dst = nx.data();
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (size_t p=0; p<npx; p++)
        for (int c=0; c<3; c++)
            dst[p*3+c] = (image[p*3+c]/255.0f - mean[c])/(std_[c]+norm_eps);

    backbone.forward(nx.data(), img_h, img_w, bscratch[cam], feat);
}

void ActModel::predict(const uint8_t* const* images, const float* state,
                       bool unnormalize, float* actions) const {
    // ACT_PROFILE=1: per-stage latency on stderr
    static const bool prof = std::getenv("ACT_PROFILE") != nullptr;
    const double t0 = nn::now_ms();

    int fh = 0, fw = 0;
    backbone.feat_size(img_h, img_w, &fh, &fw);
    const size_t nfeat = (size_t)fh*fw*backbone.out_channels();

    std::vector<const float*> fp(n_cams);
    for (int c=0; c<n_cams; c++) {
        if (feats[c].size() < nfeat) feats[c].resize(nfeat);
        fp[c] = feats[c].data();
    }

    // The cameras are independent. TCPU_VIEW_THREADS=N runs them concurrently on
    // N OMP threads each (SmolVLA's trick) instead of one after another on the
    // whole team; each view already owns its own scratch, so the math is
    // untouched either way. Default off - measured on i5-12400F, the tiled convs
    // already fill the machine and splitting the team only costs efficiency.
    hal::for_each_view(n_cams, hal::env::view_threads(),
                       [&](int c) { encode_view(images[c], c, feats[c].data()); });
    const double t1 = nn::now_ms();

    std::vector<float> sn(tf.cfg.state_dim);
    for (int i=0; i<tf.cfg.state_dim; i++)
        sn[i] = (state[i]-state_mean[i])/(state_std[i]+norm_eps);

    tf.forward(fp.data(), n_cams, fh, fw, sn.data(), actions);
    const double t2 = nn::now_ms();

    if (unnormalize) {
        for (int h=0; h<tf.cfg.chunk; h++)
            for (int a=0; a<tf.cfg.action_dim; a++) {
                float* v = actions+(size_t)h*tf.cfg.action_dim+a;
                *v = *v*action_std[a] + action_mean[a];
            }
    }

    if (prof)
        std::fprintf(stderr, "[act] backbone %.1f ms (%d cams)  transformer %.1f ms  total %.1f ms\n",
                     t1-t0, n_cams, t2-t1, t2-t0);
}

} // namespace tcpu
