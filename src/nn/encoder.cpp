/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "encoder.h"
#include "../ops/lm_ops.h"
#include <chrono>
#include <cstdio>
#include <cstddef>
using std::size_t;

namespace tcpu {
namespace nn {

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void Prof::tic() { if (on) t0 = now_ms(); }
void Prof::toc(double& acc) { if (on) acc += now_ms()-t0; }
void Prof::report() const {
    if (!on) return;
    double sum = assemble+mask+ln+qkv+attn+proj+mlp+res;
    std::printf("  [octo tf] assemble %6.1f  mask %6.1f  ln %6.1f  qkv %6.1f  attn %6.1f"
                "  proj %6.1f  mlp %6.1f  res %6.1f | other %6.1f | wall %6.1f ms\n",
                assemble, mask, ln, qkv, attn, proj, mlp, res, wall-sum, wall);
}

void EncoderLayer::forward(float* x, int total, const float* mask, Scratch& s, Prof* prof) const {
    if (s.h.size()  < (size_t)total*d  ) s.h.resize ((size_t)total*d);
    if (s.ff.size() < (size_t)total*mlp) s.ff.resize((size_t)total*mlp);

    // Per layer, not per backend: an int8 wo or w2 has no fused epilogue, and
    // taking it anyway costs an allocation and a second pass per call.
    const bool fuse_attn = attn.wo.add_ok();
    const bool fuse_mlp  = w2.add_ok();

    if (prof) prof->tic();
    layernorm(s.h.data(), x, ln1_s, ln1_b, total, d, ln_eps);
    if (prof) prof->toc(prof->ln);

    if (fuse_attn) {
        // residual fused into the wo epilogue: x += proj (bit-identical add)
        attn.forward(x, s.h.data(), total, mask, -1, s, prof, true);
    } else {
        // proj lands back in h: x_norm (h) is consumed before the projection writes
        attn.forward(s.h.data(), s.h.data(), total, mask, -1, s, prof);

        if (prof) prof->tic();
        for (size_t i=0; i<(size_t)total*d; i++)
            x[i] += s.h[i];
        if (prof) prof->toc(prof->res);
    }

    if (prof) prof->tic();
    layernorm(s.h.data(), x, ln2_s, ln2_b, total, d, ln_eps);
    if (prof) prof->toc(prof->ln);

    if (prof) prof->tic();
    if (gelu == Gelu::Tanh) {
        w1.forward_gelu(s.ff.data(), s.h.data(), total);
    } else {
        // no fused erf epilogue: plain GEMM then the activation in place
        w1.forward(s.ff.data(), s.h.data(), total);
        gelu_erf(s.ff.data(), total*mlp);
    }
    if (fuse_mlp) {
        w2.forward_add(x, s.ff.data(), total);
    } else {
        w2.forward(s.h.data(), s.ff.data(), total);
    }
    if (prof) prof->toc(prof->mlp);

    if (!fuse_mlp) {
        if (prof) prof->tic();
        for (size_t i=0; i<(size_t)total*d; i++)
            x[i] += s.h[i];
        if (prof) prof->toc(prof->res);
    }
}

void EncoderLayer::forward_last_row(float* x, int ro, int total, const float* mask, Scratch& s) const {
    if (s.h.size()  < (size_t)total*d) s.h.resize ((size_t)total*d);
    if (s.ff.size() < (size_t)mlp    ) s.ff.resize((size_t)mlp);

    const bool fuse_attn = attn.wo.add_ok();
    const bool fuse_mlp  = w2.add_ok();
    float* xr = x+(size_t)ro*d;

    layernorm(s.h.data(), x, ln1_s, ln1_b, total, d, ln_eps);
    if (fuse_attn) {
        attn.forward(xr, s.h.data(), total, mask, ro, s, nullptr, true);
    } else {
        attn.forward(s.h.data(), s.h.data(), total, mask, ro, s, nullptr);

        for (int i=0; i<d; i++)
            xr[i] += s.h[i];
    }

    layernorm(s.h.data(), xr, ln2_s, ln2_b, 1, d, ln_eps);
    if (gelu == Gelu::Tanh) {
        w1.forward_gelu(s.ff.data(), s.h.data(), 1);
    } else {
        w1.forward(s.ff.data(), s.h.data(), 1);
        gelu_erf(s.ff.data(), mlp);
    }
    if (fuse_mlp) {
        w2.forward_add(xr, s.ff.data(), 1);
    } else {
        w2.forward(s.h.data(), s.ff.data(), 1);

        for (int i=0; i<d; i++)
            xr[i] += s.h[i];
    }
}

} // namespace nn
} // namespace tcpu
