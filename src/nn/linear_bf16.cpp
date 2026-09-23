/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// Linear::init_bf16 lives in its own TU, linked after the hot nn/hal objects:
// adding it to linear.cpp shifted attention.o/encoder.o code placement and cost
// a consistent ~6% end-to-end on the Cortex-A72 (layout-sensitive in-order
// core). Load-time-only code - keep it out of the hot link region.

#include "linear.h"
#include "../hal/arch.h"
#include "../hal/common/blas.h"
#include "../hal/common/env.h"
#include "../ops/lm_ops.h"
#include <cstddef>
using std::size_t;

namespace tcpu {
namespace nn {

void Linear::init_bf16(const uint16_t* Wb16, const float* bias_, int N_, int K_, Role role_) {
#if TCPU_HAL_NEON
    // bf16 packed panels: identical values to the raw bf16 kernel, half the
    // bytes streamed vs fp32 (the Pi GEMMs are bandwidth-bound). TCPU_BF16_DEQ=1
    // opts into fp32 panels for A/B (2x weight RAM - the full model then
    // exceeds a 4GB Pi; module tests only).
    if (role_ != Role::Generic && hal::env::packed() && N_%16 == 0 && !hal::env::bf16_deq()) {
        W    = nullptr;
        bias = bias_;
        N    = N_;
        K    = K_;
        role = role_;
        Wp   = nullptr;
        Wr16 = nullptr;
        packed.clear();
        deq.clear();
        packed_bf16.resize((size_t)N_*K_);
        for (int b=0; b<N_/16; b++)
            for (int k=0; k<K_; k++)
                for (int j=0; j<16; j++)
                    packed_bf16[((size_t)b*K_+k)*16+j] = Wb16[(size_t)(b*16+j)*K_+k];
        Wb = packed_bf16.data();
        return;
    }
#endif

    if (bf16_keeps_raw()) {
        W    = nullptr;
        bias = bias_;
        N    = N_;
        K    = K_;
        role = role_;
        Wp   = nullptr;
        Wb   = nullptr;
        Wr16 = Wb16;
        packed.clear();
        packed_bf16.clear();
        deq.clear();
        return;
    }

    deq.resize((size_t)N_*K_);
    for (size_t i=0; i<deq.size(); i++)
        deq[i] = bf16_f32(Wb16[i]);

    init(deq.data(), bias_, N_, K_, role_);
    if (Wp) {
        W = nullptr;
        deq.clear();
        deq.shrink_to_fit();
    }
}

bool Linear::bf16_keeps_raw() {
#if TCPU_HAL_X86
    return !hal::env::bf16_deq();
#elif TCPU_HAL_APPLE
    return false;
#elif TCPU_HAL_NEON
    return !hal::env::packed();
#else
    return true;
#endif
}

} // namespace nn
} // namespace tcpu
