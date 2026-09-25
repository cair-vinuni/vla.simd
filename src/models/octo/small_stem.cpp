/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../arena.h"
#include "small_stem.h"
#include "ops/conv_ops.h"
#include "io/files.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace tcpu {

bool SmallStem::load(const std::string& dir, const std::string& name) {
    io::InFile meta(dir + "/" + name + ".meta");
    if (!meta) return false;
    std::string line;
    while (std::getline(meta, line)) {
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if      (key == "in_ch"    ) ss >> cfg.in_ch;
        else if (key == "n_layers" ) ss >> cfg.n_layers;
        else if (key == "k"        ) ss >> cfg.k;
        else if (key == "stride"   ) ss >> cfg.stride;
        else if (key == "pad"      ) ss >> cfg.pad;
        // cfg.features is a fixed int[MAX_STEM_LAYERS]; n_layers comes from this
        // same file, so an unclamped loop writes through the struct.
        else if (key == "features" ) for (int i=0; i<cfg.n_layers && i<SmallStemConfig::MAX_LAYERS; i++) ss >> cfg.features[i];
        else if (key == "embed_dim") ss >> cfg.embed_dim;
        else if (key == "gn_groups") ss >> cfg.gn_groups;
        else if (key == "gn_eps"   ) ss >> cfg.gn_eps;
    }

    if (!read_arena(dir + "/" + name + ".bin", data)) return false;

    // in_ch is exactly 6: encode() writes 3 obs + 3 goal channels per pixel.
    // stride/gn_groups are divisors below, so zero is a fault, not a default.
    if (cfg.n_layers < 1 || cfg.n_layers > SmallStemConfig::MAX_LAYERS ||
        cfg.in_ch != 6 || cfg.k < 1 || cfg.embed_dim < 1 ||
        cfg.stride < 1 || cfg.pad < 0 || cfg.gn_groups < 1) return false;
    for (int i=0; i<cfg.n_layers; i++)
        if (cfg.features[i] < 1 || cfg.features[i] % cfg.gn_groups != 0) return false;

    // The walk below is driven by .meta shapes over a buffer sized by the actual
    // .bin. A stale meta beside a shorter bin used to hand out pointers past the
    // allocation, and Conv2d::init copies immediately - so the size check has to
    // happen before the init, not after the loop.
    ArenaCursor<float> take{data};
    layers.resize(cfg.n_layers);

    int cin = cfg.in_ch;
    for (int i=0; i<cfg.n_layers; i++) {
        const int cout = cfg.features[i];
        const float* w = take((size_t)cout*cfg.k*cfg.k*cin);
        const float* b = take(cout);
        if (!take.ok) return false;
        layers[i].conv.init(w, b, cout, cfg.k, cin);   // packs (all Cout here are x16)
        layers[i].gn_scale = take(cout);
        layers[i].gn_bias  = take(cout);
        if (!take.ok) return false;
        cin = cout;
    }

    const float* w = take((size_t)cfg.embed_dim*cin);
    const float* b = take(cfg.embed_dim);
    if (!take.ok) return false;
    embed.init(w, b, cfg.embed_dim, cin, nn::Linear::Role::StemGemm);
    return take.done();
}

void SmallStem::encode(const uint8_t* obs, int H, int W, float* out) const {
    // OCTO_PROFILE_STEM=1: per-op attribution inside the stem (stderr)
    static const bool prof = std::getenv("OCTO_PROFILE_STEM") != nullptr;
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b-a).count();
    };
    auto t0 = clk::now();

    std::vector<float> a ((size_t)H*W*cfg.in_ch);
    std::vector<float> b;
    for (int p = 0; p < H*W; p++) {
        float* dst = a.data()+(size_t)p*cfg.in_ch;

        for (int c = 0; c < 3; c++)
            dst[c] = obs[(size_t)p*3+c]/127.5f - 1.0f;

        for (int c = 0; c < 3; c++)
            dst[3+c] = -1.0f;
    }
    auto t1 = clk::now();

    double conv_ms[8] = {0};
    double gn_ms = 0;

    int h   = H;
    int w   = W;
    for (int i=0; i<cfg.n_layers; i++) {
        const int cout = cfg.features[i];
        const int ho = (h+2*cfg.pad-cfg.k)/cfg.stride+1;
        const int wo = (w+2*cfg.pad-cfg.k)/cfg.stride+1;
        b.resize((size_t)ho*wo*cout);

        auto c0 = clk::now();
        layers[i].conv.forward(b.data(), a.data(), h, w, cfg.stride, cfg.pad);
        auto c1 = clk::now();

        a.resize(b.size());
        // GN+ReLU as one op; whether it fuses or runs as two passes is the
        // backend's call (hal/conv)
        groupnorm(a.data(), b.data(), layers[i].gn_scale, layers[i].gn_bias,
                  ho*wo, cout, cfg.gn_groups, cfg.gn_eps, /*fuse_relu=*/true);
        auto c3 = clk::now();

        conv_ms[i] = ms(c0, c1);
        gn_ms += ms(c1, c3);

        h   = ho;
        w   = wo;
    }
    auto t2 = clk::now();

    embed.forward(out, a.data(), h*w);
    if (prof) {
        std::fprintf(stderr,
            "stem %dx%d: prep %.1f | conv %.1f/%.1f/%.1f/%.1f | gn+relu %.1f | embed %.1f | total %.1f ms\n",
            H, W, ms(t0, t1), conv_ms[0], conv_ms[1], conv_ms[2], conv_ms[3],
            gn_ms, ms(t2, clk::now()), ms(t0, clk::now()));
    }
}

} // namespace tcpu
