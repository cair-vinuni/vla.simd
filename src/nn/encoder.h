/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "attention.h"
#include "linear.h"

namespace tcpu {
namespace nn {

struct Prof {
    bool on = false;
    double assemble = 0, mask = 0, ln = 0, qkv = 0, attn = 0, proj = 0, mlp = 0, res = 0, wall = 0;
    void tic();
    void toc(double& acc);
    void report() const;
  private:
    double t0 = 0;
};

// Pre-LN transformer encoder layer: LN -> masked MHA -> residual -> LN ->
// gelu MLP -> residual. Weights are raw pointers into the model's arena;
// Linears are HAL-prepared (see linear.h roles).
//
// The gelu form is per-model, not cosmetic. JAX/flax `nn.gelu` defaults to the
// tanh approximation, so models converted from flax want Gelu::Tanh (and get the
// fused GEMM epilogue). PyTorch `F.gelu` defaults to the exact erf form, so a
// checkpoint trained through torch wants Gelu::Erf or every MLP carries a small
// systematic offset -- large enough to move an action chunk, too small to look
// like a bug.
struct EncoderLayer {
    enum class Gelu { Tanh, Erf };
    Gelu gelu = Gelu::Tanh;
    const float *ln1_s = nullptr, *ln1_b = nullptr;
    const float *ln2_s = nullptr, *ln2_b = nullptr;
    MhaMasked attn;
    Linear w1, w2;   // init with Role::Mlp
    int d = 0, mlp = 0;
    float ln_eps = 1e-6f;

    // Full block over x [total, d], in place.
    void forward(float* x, int total, const float* mask, Scratch& s, Prof* prof) const;

    // Readout-only final layer: K/V still see every token, but only row ro of
    // x is updated (the only row the action head consumes).
    void forward_last_row(float* x, int ro, int total, const float* mask, Scratch& s) const;
};

} // namespace nn
} // namespace tcpu
