/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdint>
#include <vector>

namespace tcpu {
namespace nn {

// 2D conv layer with HAL-prepared weights (square kernel, NHWC). Weights are
// packed once for the packed-panel GEMM; on Apple with Accelerate enabled the
// conv GEMM runs from the raw weights on the AMX units instead.
struct Conv2d {
    // W [Cout, k, k, Cin] row-major; Cout must be a multiple of 16.
    void init(const float* W, const float* bias, int Cout, int k, int Cin);

    // x [H, W, Cin] -> out [Hout, Wout, Cout]
    void forward(float* out, const float* x, int H, int Wd, int stride, int pad) const;

    // Opt in to the W8A8 conv (ops/quant_ops.h + hal/common/conv.cpp). Same
    // contract as nn::Linear::init_int8: lossy, explicit, and a no-op returning
    // false when the CPU has no int8 kernel or Cout % 16 != 0.
    bool init_int8();

    int Cout = 0, k = 0, Cin = 0;

  private:
    const float* W = nullptr;
    const float* bias = nullptr;
    std::vector<float> packed;
    const int8_t* Wq = nullptr;
    std::vector<int32_t> packed_i8;   // int32-typed: see packed_i8_words
    std::vector<float> wscale;
};

} // namespace nn
} // namespace tcpu
