/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "impact_model.h"
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

bool ImpactModel::load(const std::string& dir) {
    const io::Mount mount(dir);
    if (!mount.ok()) return false;
    io::InFile cfgf(dir + "/config.txt");
    if (!cfgf) return false;
    int vocab_full = 32128, unk_id = 2;
    std::string line;
    cam_names.clear();
    while (std::getline(cfgf, line)) {
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if      (key == "img_h"     ) ss >> img_h;
        else if (key == "img_w"     ) ss >> img_w;
        else if (key == "n_cams"    ) ss >> n_cams;
        else if (key == "norm_eps"  ) ss >> norm_eps;
        else if (key == "vocab_full") ss >> vocab_full;
        else if (key == "unk_id"    ) ss >> unk_id;
        else if (key.rfind("cam", 0) == 0) {
            std::string name;
            ss >> name;
            cam_names.push_back(name);
        }
    }

    backbone.tag = tf.tag = "impact";
    backbone.prof = tf.prof = hal::env::int_env("IMPACT_PROFILE", 0);
    if (!backbone.load(dir, "vision")) return false;

    // IMPACT_INT8 -> quantize the transformer GEMMs to W8A8. A bitmask, so a layer
    // group can be A/B'd against fp32 on its own: 1 encoder attention projections,
    // 2 encoder w1, 4 encoder w2, 8 the camera-token projection, 16 the whole
    // decoder, 32 the backbone convolutions. 63 = all.
    // IMPACT_INT8 bit 32 routes every backbone conv through the W8A8 kernel. This is
    // the group ACT measured as both the largest speed win on the Pi 5 and the one
    // whose error the chunk averages out.
    // IMPACT_I8_CONV_FROM / _TO bracket which conv stages quantize (0 = stem,
    // 1.. = the basic blocks; _TO is exclusive, -1 = to the end).
    const int i8 = hal::env::int_env("IMPACT_INT8", 0);
    if (i8 & 32)
        backbone.quantize_convs(hal::env::int_env("IMPACT_I8_CONV_FROM", 0),
                                hal::env::int_env("IMPACT_I8_CONV_TO", -1));
    if (!tf.load(dir, backbone.out_channels(), i8, true)) return false;
    if (!tok.load(dir)) {
        std::fprintf(stderr, "impact: cannot load %s/vocab.txt\n", dir.c_str());
        return false;
    }
    if (!text.load(dir, vocab_full, unk_id)) return false;

    // The FiLM head and the backbone have to agree on how one flat gamma/beta
    // buffer is cut up. They are written by the same converter, so a mismatch
    // means the two arenas came from different runs - checked here rather than
    // discovered as a silently mis-modulated feature map.
    if (text.film_total() != backbone.film_total()) {
        std::fprintf(stderr, "impact: FiLM head emits %d channels but the backbone's "
                     "%zu points want %d - text.bin and vision.bin disagree\n",
                     text.film_total(), backbone.film_after.size(), backbone.film_total());
        return false;
    }
    if (text.n_text() != tf.cfg.n_text) {
        std::fprintf(stderr, "impact: text tower has n_text %d, transformer expects %d\n",
                     text.n_text(), tf.cfg.n_text);
        return false;
    }
    if (text.proj_dim() != tf.cfg.dim) {
        std::fprintf(stderr, "impact: text projection is %d wide, transformer is %d\n",
                     text.proj_dim(), tf.cfg.dim);
        return false;
    }

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

bool ImpactModel::set_instruction(const std::string& instruction) {
    if (tf.cfg.dim < 1) return false;
    if (have_text && instruction == cached_instruction) return true;
    const int L = tf.cfg.n_text;

    ids = tok.encode(instruction, L);
    ids.resize(L, tok.pad_id);

    // 1 for a real token, 0 for padding. The padded rows are still encoded - T5
    // produces a genuine hidden state for them - and are masked as keys later;
    // what this mask controls is where the tower may attend, which is the part
    // no downstream mask can repair.
    mask.assign(L, 0);
    n_real = 0;
    for (int i = 0; i < L; i++) {
        if (ids[(size_t)i] == tok.pad_id) break;
        mask[(size_t)i] = 1;
        n_real = i+1;
    }

    compact.assign(L, 0);
    text.remap(ids.data(), L, compact.data());

    text_tok.assign((size_t)L*tf.cfg.dim, 0.0f);
    gamma.assign((size_t)text.film_total(), 0.0f);
    beta.assign((size_t)text.film_total(), 0.0f);
    text.encode(compact.data(), mask.data(), L, text_tok.data(), gamma.data(), beta.data());

    cached_instruction = instruction;
    have_text = true;
    return true;
}

void ImpactModel::encode_view(const uint8_t* image, int cam, float* feat) const {
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

    backbone.forward(nx.data(), img_h, img_w, bscratch[cam], feat,
                     gamma.empty() ? nullptr : gamma.data(),
                     beta.empty()  ? nullptr : beta.data());
}

void ImpactModel::predict(const uint8_t* const* images, const float* state,
                          bool unnormalize, float* actions) const {
    if (!have_text) {
        std::fprintf(stderr, "impact: predict() before set_instruction(). IMPACT is "
                     "instruction-conditioned; running it unconditioned would silently "
                     "produce a different policy.\n");
        return;
    }

    // IMPACT_PROFILE=1: per-stage latency on stderr
    static const bool prof = std::getenv("IMPACT_PROFILE") != nullptr;
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
    // N OMP threads each instead of one after another on the whole team; each
    // view already owns its own scratch, so the math is untouched either way.
    hal::for_each_view(n_cams, hal::env::view_threads(),
                       [&](int c) { encode_view(images[c], c, feats[c].data()); });
    const double t1 = nn::now_ms();

    std::vector<float> sn(tf.cfg.state_dim);
    for (int i=0; i<tf.cfg.state_dim; i++)
        sn[i] = (state[i]-state_mean[i])/(state_std[i]+norm_eps);

    tf.forward(fp.data(), n_cams, fh, fw, sn.data(), actions, text_tok.data(), text.text_pos,
               n_real);
    const double t2 = nn::now_ms();

    if (unnormalize) {
        for (int h=0; h<tf.cfg.chunk; h++)
            for (int a=0; a<tf.cfg.action_dim; a++) {
                float* v = actions+(size_t)h*tf.cfg.action_dim+a;
                *v = *v*action_std[a] + action_mean[a];
            }
    }

    if (prof)
        std::fprintf(stderr, "[impact] backbone %.1f ms (%d cams)  transformer %.1f ms  "
                             "total %.1f ms  (text cached: 0.0 ms)\n",
                     t1-t0, n_cams, t2-t1, t2-t0);
}

} // namespace tcpu
