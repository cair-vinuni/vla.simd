/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "linear.h"
#include <vector>

namespace tcpu {
namespace nn {

// Conditional-MLP denoiser (the score network of a DDPM head): in_proj ->
// N x (LN -> wide -> silu -> proj -> residual) -> silu -> out_proj. Single
// token; all linears are Role::Generic (every branch ran these plain).
struct DiffusionMlp {
    struct Block {
        const float *ln_s = nullptr, *ln_b = nullptr;
        Linear d0, d1;
    };
    Linear in_proj, out_proj;
    std::vector<Block> blocks;
    int hidden = 0;
    float ln_eps = 1e-6f;

    // in [in_proj.K] -> out [out_proj.N]
    void forward(float* out, const float* in) const;
};

} // namespace nn
} // namespace tcpu
