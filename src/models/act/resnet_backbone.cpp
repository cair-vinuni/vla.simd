/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../arena.h"
#include "resnet_backbone.h"
#include "ops/conv_ops.h"
#include "ops/lm_ops.h"
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
using std::size_t;

namespace tcpu {

static inline int conv_out(int in, int k, int stride, int pad) {
    return (in+2*pad-k)/stride + 1;
}

// ACT_PROFILE=2 breaks the backbone down per conv. The clock reads are behind a
// constant so the default path keeps the tight loop it had.
static int prof_level() {
    static const int v = [] {
        const char* e = std::getenv("ACT_PROFILE");
        return e ? std::atoi(e) : 0;
    }();
    return v;
}

namespace {
struct StageTimer {
    bool on;
    std::chrono::steady_clock::time_point t;
    explicit StageTimer(bool enabled) : on(enabled) {
        if (on) t = std::chrono::steady_clock::now();
    }
    // GFLOP/s alongside the wall time: a conv that is merely big looks the same as
    // one that is running badly until the rate is next to it.
    void lap(const char* what, double gflop) {
        if (!on) return;
        auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now-t).count();
        std::fprintf(stderr, "[act]   %-22s %6.2f ms  %6.1f GFLOP/s\n",
                     what, ms, ms > 0 ? gflop/(ms*1e-3) : 0.0);
        t = now;
    }
};
} // namespace

// ACT_INT8 bit 32 routes every backbone conv through the W8A8 kernel. The stem
// is included: it is the one conv reading real camera pixels, and those are
// already a quantized 8-bit signal to begin with.
static bool conv_int8_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("ACT_INT8");
        return e && (std::atoi(e) & 32) != 0;
    }();
    return v;
}

// ACT_I8_CONV_FROM=N keeps the first N conv stages (0 = the stem, 1.. = the
// basic blocks) in fp32 and quantizes the rest. A per-tensor activation scale
// costs more precision the sparser and more outlier-heavy a feature map is, so
// which end of the network to protect is a measurement, not a guess.
static int conv_int8_from() {
    static const int v = [] {
        const char* e = std::getenv("ACT_I8_CONV_FROM");
        return e ? std::atoi(e) : 0;
    }();
    return v;
}

// ACT_I8_CONV_TO=N stops quantizing at stage N (exclusive); -1 = to the end.
static int conv_int8_to() {
    static const int v = [] {
        const char* e = std::getenv("ACT_I8_CONV_TO");
        return e ? std::atoi(e) : -1;
    }();
    return v;
}

bool ResNetBackbone::load(const std::string& dir, const std::string& name) {
    std::ifstream meta(dir + "/" + name + ".meta");
    if (!meta) return false;

    struct BlockMeta { int cin, cout, stride, has_down; };
    std::vector<BlockMeta> bm;
    std::string line;
    while (std::getline(meta, line)) {
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if      (key == "in_ch"      ) ss >> cfg.in_ch;
        else if (key == "stem_out"   ) ss >> cfg.stem_out;
        else if (key == "stem_k"     ) ss >> cfg.stem_k;
        else if (key == "stem_stride") ss >> cfg.stem_stride;
        else if (key == "stem_pad"   ) ss >> cfg.stem_pad;
        else if (key == "pool_k"     ) ss >> cfg.pool_k;
        else if (key == "pool_stride") ss >> cfg.pool_stride;
        else if (key == "pool_pad"   ) ss >> cfg.pool_pad;
        else if (key == "block") {
            BlockMeta b{};
            ss >> b.cin >> b.cout >> b.stride >> b.has_down;
            bm.push_back(b);
        }
    }

    if (!read_arena(dir + "/" + name + ".bin", data)) return false;

    // The walk is driven by .meta shapes over a buffer sized by the actual .bin,
    // and Conv2d::init COPIES its weights immediately - so a stale meta beside a
    // shorter bin has to be caught here, not by the off == data.size() check at
    // the end, which only runs once every layer has already read past the array.
    size_t off = 0;
    bool ok = true;
    auto take = [&](size_t n) -> const float* {
        if (n > data.size() - off) { ok = false; return nullptr; }
        const float* p = data.data()+off;
        off += n;
        return p;
    };

    if (cfg.in_ch != 3) return false;
    const float* w = take((size_t)cfg.stem_out*cfg.stem_k*cfg.stem_k*cfg.in_ch);
    const float* b = take(cfg.stem_out);
    if (!ok) return false;
    stem.init(w, b, cfg.stem_out, cfg.stem_k, cfg.in_ch);

    const int k = cfg.block_k;
    blocks.resize(bm.size());
    for (size_t i=0; i<bm.size(); i++) {
        BasicBlock& blk = blocks[i];
        blk.cin      = bm[i].cin;
        blk.cout     = bm[i].cout;
        blk.stride   = bm[i].stride;
        blk.has_down = bm[i].has_down != 0;

        // Without a downsample the residual reads the block input in place.
        if (blk.cin != (i ? blocks[i-1].cout : cfg.stem_out) || blk.cout < 1 || blk.stride < 1) return false;
        if (!blk.has_down && (blk.stride != 1 || blk.cin != blk.cout)) {
            std::fprintf(stderr, "act backbone: block %zu has no downsample but "
                         "changes shape (cin %d cout %d stride %d)\n",
                         i, blk.cin, blk.cout, blk.stride);
            return false;
        }

        w = take((size_t)blk.cout*k*k*blk.cin);
        b = take(blk.cout);
        if (!ok) return false;
        blk.conv1.init(w, b, blk.cout, k, blk.cin);

        w = take((size_t)blk.cout*k*k*blk.cout);
        b = take(blk.cout);
        if (!ok) return false;
        blk.conv2.init(w, b, blk.cout, k, blk.cout);

        if (blk.has_down) {
            w = take((size_t)blk.cout*blk.cin);
            b = take(blk.cout);
            if (!ok) return false;
            blk.down.init(w, b, blk.cout, 1, blk.cin);
        }
    }
    if (off != data.size()) return false;

    if (conv_int8_enabled()) {
        const int from = conv_int8_from();
        const int to   = conv_int8_to() < 0 ? (int)blocks.size()+1 : conv_int8_to();
        auto want = [&](int stage) { return stage >= from && stage < to; };

        int n = 0;
        if (want(0)) n += stem.init_int8();
        for (size_t i=0; i<blocks.size(); i++) {
            if (!want((int)i+1)) continue;
            BasicBlock& blk = blocks[i];
            n += blk.conv1.init_int8();
            n += blk.conv2.init_int8();
            if (blk.has_down) n += blk.down.init_int8();
        }
        if (prof_level() >= 2 || n == 0)
            std::fprintf(stderr, "[act] int8 backbone convs: %d%s\n", n,
                         n ? "" : " (no int8 kernel on this CPU - staying fp32)");
    }
    return true;
}

void ResNetBackbone::feat_size(int H, int W, int* fh, int* fw) const {
    int h = conv_out(H, cfg.stem_k, cfg.stem_stride, cfg.stem_pad);
    int w = conv_out(W, cfg.stem_k, cfg.stem_stride, cfg.stem_pad);
    h = conv_out(h, cfg.pool_k, cfg.pool_stride, cfg.pool_pad);
    w = conv_out(w, cfg.pool_k, cfg.pool_stride, cfg.pool_pad);

    for (const BasicBlock& blk : blocks) {
        h = conv_out(h, cfg.block_k, blk.stride, 1);
        w = conv_out(w, cfg.block_k, blk.stride, 1);
    }
    *fh = h;
    *fw = w;
}

void ResNetBackbone::forward(const float* x, int H, int W, BackboneScratch& s, float* out) const {
    StageTimer tm(prof_level() >= 2);
    auto gflop = [](long long npix, int cout, int K) {
        return 2.0*(double)npix*cout*K/1e9;
    };

    int h = conv_out(H, cfg.stem_k, cfg.stem_stride, cfg.stem_pad);
    int w = conv_out(W, cfg.stem_k, cfg.stem_stride, cfg.stem_pad);
    s.a.resize((size_t)h*w*cfg.stem_out);
    stem.forward(s.a.data(), x, H, W, cfg.stem_stride, cfg.stem_pad);
    tm.lap("stem 7x7/s2", gflop((long long)h*w, cfg.stem_out, cfg.stem_k*cfg.stem_k*cfg.in_ch));

    const int ph = conv_out(h, cfg.pool_k, cfg.pool_stride, cfg.pool_pad);
    const int pw = conv_out(w, cfg.pool_k, cfg.pool_stride, cfg.pool_pad);
    s.b.resize((size_t)ph*pw*cfg.stem_out);
    maxpool2d_relu(s.b.data(), s.a.data(), h, w, cfg.stem_out, cfg.pool_k, cfg.pool_stride, cfg.pool_pad);
    h = ph;
    w = pw;
    tm.lap("maxpool 3x3/s2", 0.0);

    // s.b carries the block input; conv2 writes s.c and the two swap, so the
    // residual identity is read in place (only a strided block copies, via down).
    const int k = cfg.block_k;
    int bi = 0;
    for (const BasicBlock& blk : blocks) {
        const int oh = conv_out(h, k, blk.stride, 1);
        const int ow = conv_out(w, k, blk.stride, 1);
        const size_t n = (size_t)oh*ow*blk.cout;

        const float* idn = s.b.data();
        if (blk.has_down) {
            s.res.resize(n);
            blk.down.forward(s.res.data(), s.b.data(), h, w, blk.stride, 0);
            idn = s.res.data();
        }

        s.a.resize(n);
        blk.conv1.forward(s.a.data(), s.b.data(), h, w, blk.stride, 1);
        relu(s.a.data(), (int)n);

        s.c.resize(n);
        blk.conv2.forward(s.c.data(), s.a.data(), oh, ow, 1, 1);

        float* o = s.c.data();
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (size_t i=0; i<n; i++) {
            const float v = o[i]+idn[i];
            o[i] = v > 0.0f ? v : 0.0f;
        }

        s.b.swap(s.c);
        h = oh;
        w = ow;

        if (tm.on) {
            char name[32];
            std::snprintf(name, sizeof name, "block%d %dch/s%d", bi, blk.cout, blk.stride);
            const double g = gflop((long long)oh*ow, blk.cout, k*k*blk.cin)
                           + gflop((long long)oh*ow, blk.cout, k*k*blk.cout)
                           + (blk.has_down ? gflop((long long)oh*ow, blk.cout, blk.cin) : 0.0);
            tm.lap(name, g);
        }
        bi++;
    }

    std::memcpy(out, s.b.data(), (size_t)h*w*out_channels()*sizeof(float));
}

} // namespace tcpu
