/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "conv.h"
#include "../hal/common/blas.h"
#include "../ops/conv_ops.h"
#include "../ops/quant_ops.h"
#include "../ops/lm_ops.h"
#include <cstddef>
using std::size_t;

namespace tcpu {
namespace nn {

void Conv2d::init(const float* W_, const float* bias_, int Cout_, int k_, int Cin_) {
    W    = W_;
    bias = bias_;
    Cout = Cout_;
    k    = k_;
    Cin  = Cin_;

    // conv2d_packed walks Cout/16 whole panels, so a partial trailing block
    // leaves those output channels unwritten and the caller reads uninitialized
    // memory. Linear::init and Conv2d::init_int8 both check this. Every shipped
    // backbone is a multiple of 16, so the unpacked fallback costs nothing.
    if (Cout%16 != 0 || hal::accel_on()) {
        packed.clear();
        return;
    }
    packed.resize((size_t)Cout*k*k*Cin);
    pack_weights16(W, packed.data(), Cout, k*k*Cin);
}

bool Conv2d::init_int8() {
    const int K = k*k*Cin;
    if (!int8_gemm_available() || !W || Cout%16 != 0) return false;

    packed_i8.resize(packed_i8_words(Cout, K));
    wscale.resize(Cout);
    pack_weights_i8(W, (int8_t*)packed_i8.data(), wscale.data(), Cout, K);
    Wq = (const int8_t*)packed_i8.data();

    packed.clear();          // the fp32 panels are dead once int8 owns forward()
    packed.shrink_to_fit();
    return true;
}

void Conv2d::forward(float* out, const float* x, int H, int Wd, int stride, int pad) const {
    if (Wq) {
        // thread_local: ActModel encodes cameras on several threads through one
        // Conv2d, and per-object scratch raced.
        static thread_local std::vector<int8_t> xq;
        conv2d_i8(out, x, Wq, wscale.data(), bias, H, Wd, Cin, Cout, k, stride, pad, xq);
        return;
    }
    if (hal::accel_on())
        conv2d_blas(out, x, W, bias, H, Wd, Cin, Cout, k, stride, pad);
    else if (!packed.empty())
        conv2d_packed(out, x, packed.data(), bias, H, Wd, Cin, Cout, k, stride, pad);
    else
        conv2d(out, x, W, bias, H, Wd, Cin, Cout, k, stride, pad);   // Cout % 16 != 0
}

} // namespace nn
} // namespace tcpu
