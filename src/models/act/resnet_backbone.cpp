/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../arena.h"
#include "resnet_backbone.h"
#include "ops/conv_ops.h"
#include "ops/lm_ops.h"
#include "nn/encoder.h"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
using std::size_t;

namespace tcpu {

static inline int conv_out(int in, int k, int stride, int pad) {
    return (in+2*pad-k)/stride + 1;
}

namespace {
struct StageTimer {
    const char* tag;
    bool on;
    double t = 0;
    StageTimer(const char* tag_, bool enabled) : tag(tag_), on(enabled) {
        if (on) t = nn::now_ms();
    }
    // GFLOP/s alongside the wall time: a conv that is merely big looks the same as
    // one that is running badly until the rate is next to it.
    void lap(const char* what, double gflop) {
        if (!on) return;
        const double now = nn::now_ms();
        const double ms = now-t;
        std::fprintf(stderr, "[%s]   %-22s %6.2f ms  %6.1f GFLOP/s\n",
                     tag, what, ms, ms > 0 ? gflop/(ms*1e-3) : 0.0);
        t = now;
    }
};
} // namespace

bool ResNetBackbone::load(const std::string& dir, const std::string& name) {
    std::ifstream meta(dir + "/" + name + ".meta");
    if (!meta) return false;

    struct BlockMeta { int cin, cout, stride, has_down; };
    std::vector<BlockMeta> bm;
    film_after.clear();
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
        else if (key == "gn_group_size") ss >> cfg.gn_group_size;
        else if (key == "block") {
            BlockMeta b{};
            ss >> b.cin >> b.cout >> b.stride >> b.has_down;
            bm.push_back(b);
        }
        else if (key == "film_after") {
            int idx = -1;
            ss >> idx;
            film_after.push_back(idx);
        }
    }

    if (!read_arena(dir + "/" + name + ".bin", data)) return false;

    // The walk is driven by .meta shapes over a buffer sized by the actual .bin,
    // and Conv2d::init COPIES its weights immediately - so a stale meta beside a
    // shorter bin has to be caught here, not by the off == data.size() check at
    // the end, which only runs once every layer has already read past the array.
    ArenaCursor<float> take{data};

    const int gs = cfg.gn_group_size;
    auto norm = [&](const float** gn, int c) -> const float* {
        if (!gs) return take(c);
        if (c % gs) take.ok = false;
        gn[0] = take(c);
        gn[1] = take(c);
        return nullptr;
    };

    if (cfg.in_ch != 3 || gs < 0) return false;
    const float* w = take((size_t)cfg.stem_out*cfg.stem_k*cfg.stem_k*cfg.in_ch);
    const float* b = norm(stem_gn, cfg.stem_out);
    if (!take.ok) return false;
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
            std::fprintf(stderr, "%s backbone: block %zu has no downsample but "
                         "changes shape (cin %d cout %d stride %d)\n",
                         tag, i, blk.cin, blk.cout, blk.stride);
            return false;
        }

        w = take((size_t)blk.cout*k*k*blk.cin);
        b = norm(blk.gn1, blk.cout);
        if (!take.ok) return false;
        blk.conv1.init(w, b, blk.cout, k, blk.cin);

        w = take((size_t)blk.cout*k*k*blk.cout);
        b = norm(blk.gn2, blk.cout);
        if (!take.ok) return false;
        blk.conv2.init(w, b, blk.cout, k, blk.cout);

        if (blk.has_down) {
            w = take((size_t)blk.cout*blk.cin);
            b = norm(blk.gnd, blk.cout);
            if (!take.ok) return false;
            blk.down.init(w, b, blk.cout, 1, blk.cin);
        }
    }
    if (!take.done()) return false;

    // A FiLM point that names a block this backbone does not have would silently
    // modulate nothing (or read past the gamma buffer), so reject it at load.
    for (int idx : film_after) {
        if (idx < 0 || idx >= (int)blocks.size()) {
            std::fprintf(stderr, "%s backbone: film_after %d out of range "
                         "(%zu blocks)\n", tag, idx, blocks.size());
            return false;
        }
    }
    if (std::adjacent_find(film_after.begin(), film_after.end(),
                           [](int x, int y) { return x >= y; }) != film_after.end()) {
        std::fprintf(stderr, "%s backbone: film_after must be strictly ascending - the "
                     "gamma/beta buffer is cut up in that order\n", tag);
        return false;
    }
    return true;
}

void ResNetBackbone::quantize_convs(int from, int to) {
    if (to < 0) to = (int)blocks.size()+1;
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
    if (prof >= 2 || n == 0)
        std::fprintf(stderr, "[%s] int8 backbone convs: %d%s\n", tag, n,
                     n ? "" : " (no int8 kernel on this CPU - staying fp32)");
}

int ResNetBackbone::film_total() const {
    int n = 0;
    for (int idx : film_after)
        n += blocks[(size_t)idx].cout;
    return n;
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

void ResNetBackbone::forward(const float* x, int H, int W, BackboneScratch& s, float* out,
                             const float* gamma, const float* beta) const {
    StageTimer tm(tag, prof >= 2);
    auto gflop = [](long long npix, int cout, int K) {
        return 2.0*(double)npix*cout*K/1e9;
    };
    const int gs = cfg.gn_group_size;
    auto norm = [gs](float* v, const float* const* gn, int npx, int C, bool relu_out) {
        groupnorm(v, v, gn[0], gn[1], npx, C, C/gs, 1e-5f, relu_out);
    };

    int h = conv_out(H, cfg.stem_k, cfg.stem_stride, cfg.stem_pad);
    int w = conv_out(W, cfg.stem_k, cfg.stem_stride, cfg.stem_pad);
    s.a.resize((size_t)h*w*cfg.stem_out);
    stem.forward(s.a.data(), x, H, W, cfg.stem_stride, cfg.stem_pad);
    if (gs) norm(s.a.data(), stem_gn, h*w, cfg.stem_out, false);
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
    size_t film_off = 0, film_next = 0;
    int bi = 0;
    for (const BasicBlock& blk : blocks) {
        const int oh = conv_out(h, k, blk.stride, 1);
        const int ow = conv_out(w, k, blk.stride, 1);
        const size_t n = (size_t)oh*ow*blk.cout;

        const float* idn = s.b.data();
        if (blk.has_down) {
            s.res.resize(n);
            blk.down.forward(s.res.data(), s.b.data(), h, w, blk.stride, 0);
            if (gs) norm(s.res.data(), blk.gnd, oh*ow, blk.cout, false);
            idn = s.res.data();
        }

        s.a.resize(n);
        blk.conv1.forward(s.a.data(), s.b.data(), h, w, blk.stride, 1);
        if (gs) norm(s.a.data(), blk.gn1, oh*ow, blk.cout, true);
        else relu(s.a.data(), (int)n);

        s.c.resize(n);
        blk.conv2.forward(s.c.data(), s.a.data(), oh, ow, 1, 1);
        if (gs) norm(s.c.data(), blk.gn2, oh*ow, blk.cout, false);

        const float *fg = nullptr, *fb = nullptr;
        if (film_next < film_after.size() && film_after[film_next] == bi) {
            if (gamma && beta) {
                fg = gamma+film_off;
                fb = beta +film_off;
            }
            film_off += (size_t)blk.cout;
            film_next++;
        }

        float* o = s.c.data();
        if (fg) {
            const int C = blk.cout;
            const long long npx = (long long)oh*ow;
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static)
#endif
            for (long long p=0; p<npx; p++) {
                float* row = o+(size_t)p*C;
                const float* id = idn+(size_t)p*C;
                for (int c=0; c<C; c++) {
                    const float v = row[c]+id[c];
                    row[c] = (1.0f+fg[c])*(v > 0.0f ? v : 0.0f) + fb[c];
                }
            }
        } else {
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static)
#endif
            for (size_t i=0; i<n; i++) {
                const float v = o[i]+idn[i];
                o[i] = v > 0.0f ? v : 0.0f;
            }
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
