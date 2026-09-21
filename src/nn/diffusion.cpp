/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "diffusion.h"
#include "../ops/lm_ops.h"
#include <vector>
#include <cstddef>
using std::size_t;

namespace tcpu {
namespace nn {

void DiffusionMlp::forward(float* out, const float* in) const {
    const int H = hidden;
    std::vector<float> h    (H);
    std::vector<float> n    (H);
    std::vector<float> wide (4*H);
    in_proj.forward(h.data(), in, 1);

    for (const auto& B : blocks) {
        layernorm(n.data(), h.data(), B.ln_s, B.ln_b, 1, H, ln_eps);
        B.d0.forward(wide.data(), n.data(), 1);
        silu(wide.data(), 4*H);
        B.d1.forward(n.data(), wide.data(), 1);

        for (int i=0; i<H; i++)
            h[i] += n[i];
    }

    silu(h.data(), H);
    out_proj.forward(out, h.data(), 1);
}

} // namespace nn
} // namespace tcpu
