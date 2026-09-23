/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "linear.h"
#include "../hal/arch.h"
#include "../hal/common/blas.h"
#include "../hal/common/env.h"
#include "../ops/lm_ops.h"
#include "../ops/quant_ops.h"
#include <cstring>
#include <cstddef>
using std::size_t;

namespace tcpu {
namespace nn {

using hal::accel_on;

void Linear::init(const float* W_, const float* bias_, int N_, int K_, Role role_) {
    W    = W_;
    bias = bias_;
    N    = N_;
    K    = K_;
    role = role_;
    Wp   = nullptr;
    Wb   = nullptr;
    Wr16 = nullptr;
    packed.clear();
    packed_bf16.clear();

    if (role == Role::Generic) return;

    // Pack once for the packed-panel kernel (TCPU_PACKED=0 keeps the raw path;
    // the packer produces N/16 full blocks, so N must be a multiple of 16).
    if ((hal::env::packed() || role == Role::StemGemm) && N%16 == 0) {
        packed.resize((size_t)N*K);
        pack_weights16(W, packed.data(), N, K);
        Wp = packed.data();
#if TCPU_HAL_NEON
        // bf16 mirror of the packed weights (same [b][k][16] layout), used for
        // the memory-bound MLP GEMMs when TCPU_BF16_MLP=1. Round-to-nearest-even
        // to bf16 (top 16 bits). Pi backend only - the one place it was measured.
        if (role == Role::Mlp && hal::env::bf16_mlp()) {
            packed_bf16.resize(packed.size());
            for (size_t i=0; i<packed.size(); i++) {
                uint32_t u;
                std::memcpy(&u, &packed[i], 4);
                packed_bf16[i] = (u & 0x7fffffffu) > 0x7f800000u ? (uint16_t)((u >> 16) | 0x40u)
                                                                 : (uint16_t)((u+0x7fff+((u>>16) & 1)) >> 16);
            }
            Wb = packed_bf16.data();
        }
#endif
    }
}

bool Linear::init_int8() {
    if (!int8_gemm_available() || N%16 != 0) return false;
    if (!W && !Wb && !Wr16) return false;

    // The packer wants a plain [N,K] fp32 matrix, but a layer holds whichever
    // representation its init picked: raw fp32 (init), packed bf16 panels or raw
    // bf16 (init_bf16's per-backend choices - SmolVLA's towers are bf16, ACT's
    // are fp32). Rebuild [N,K] transiently here rather than teach the packer four
    // layouts; this runs once, at load, and the temporary dies with the scope.
    std::vector<float> tmp;
    const float* src = W;
    if (!src) {
        tmp.resize((size_t)N*K);
        if (Wr16) {
            for (size_t i=0; i<tmp.size(); i++) {
                const uint32_t u = (uint32_t)Wr16[i] << 16;
                std::memcpy(&tmp[i], &u, 4);
            }
        } else {
            // packed panels [N/16][K][16]: element [b][k][j] is W[b*16+j][k]
            for (int b=0; b<N/16; b++)
                for (int k=0; k<K; k++)
                    for (int j=0; j<16; j++) {
                        const uint32_t u = (uint32_t)Wb[((size_t)b*K+k)*16+j] << 16;
                        std::memcpy(&tmp[(size_t)(b*16+j)*K+k], &u, 4);
                    }
        }
        src = tmp.data();
    }

    packed_i8.resize(packed_i8_words(N, K));
    wscale.resize(N);
    pack_weights_i8(src, (int8_t*)packed_i8.data(), wscale.data(), N, K);
    Wq = (const int8_t*)packed_i8.data();

    // The fp32/bf16 panels are dead weight once the int8 path owns forward():
    // drop them (4 or 2 bytes/weight against 1) rather than keep a fallback
    // nothing selects. W / Wr16 point into the model's own arena, so they are
    // left alone - only what this object allocated is freed.
    packed.clear();
    packed.shrink_to_fit();
    packed_bf16.clear();
    packed_bf16.shrink_to_fit();
    Wp = nullptr;
    Wb = nullptr;
    return true;
}

// Linear::init_bf16 / bf16_keeps_raw live in linear_bf16.cpp (linked last):
// load-time-only code here shifted the hot nn objects and measurably slowed
// the Cortex-A72 attention path.

void Linear::forward(float* out, const float* x, int seq) const {
    if (Wq) {
        // thread_local, not a member: the view fan-out runs several threads
        // through one Linear, and per-object scratch raced. One buffer per thread
        // also costs less RAM than one per layer.
        static thread_local std::vector<int8_t> xq;
        static thread_local std::vector<float> ascale;
        const size_t need = (size_t)seq*i8_kpad(K);
        if (xq.size() < need) xq.resize(need);
        if (ascale.size() < (size_t)seq) ascale.resize(seq);
        dense_linear_i8(out, x, Wq, wscale.data(), bias, seq, N, K,
                        xq.data(), ascale.data());
        return;
    }

    if (Wr16) {
        dense_linear_bf16(out, x, Wr16, bias, seq, N, K);
        return;
    }

    if (role != Role::Generic) {
        if (accel_on()) {
            dense_linear_blas(out, x, W, bias, seq, N, K);
            return;
        }
        if (Wb) {
            dense_linear_packed_bf16(out, x, Wb, bias, seq, N, K);
            return;
        }
        if (Wp) {
            dense_linear_packed(out, x, Wp, bias, seq, N, K);
            return;
        }
    }

    dense_linear(out, x, W, bias, seq, N, K);
}

void Linear::forward_gelu(float* out, const float* x, int seq) const {
#if TCPU_ISA_X86
    if (Wp && !Wr16 && hal::env::fuse_gelu()) {
        dense_linear_packed_gelu(out, x, Wp, bias, seq, N, K);
        return;
    }
#endif
    forward(out, x, seq);
    gelu_tanh(out, seq*N);
}

void Linear::forward_add(float* out, const float* x, int seq) const {
#if TCPU_ISA_X86
    if (Wp && hal::env::fuse_res()) {
        dense_linear_packed_add(out, x, Wp, bias, seq, N, K);
        return;
    }
#endif
    std::vector<float> tmp((size_t)seq*N);
    forward(tmp.data(), x, seq);
    for (size_t i=0; i<(size_t)seq*N; i++)
        out[i] += tmp[i];
}

bool Linear::add_native() {
#if TCPU_ISA_X86
    return hal::env::packed() && hal::env::fuse_res();
#else
    return false;
#endif
}

void Linear::forward_kt(float* out_t, const float* x, int seq, int ldo) const {
    if (accel_on()) {
        dense_linear_blas_kt(out_t, x, W, bias, seq, N, K, ldo);
        return;
    }

    if (!Wp) {   // no fp32 panels (raw-bf16 path): compute plain and transpose
        std::vector<float> tmp((size_t)seq*N);
        forward(tmp.data(), x, seq);
        for (int t=0; t<seq; t++)
            for (int n=0; n<N; n++)
                out_t[(size_t)n*ldo+t] = tmp[(size_t)t*N+n];
        return;
    }

    dense_linear_packed_kt(out_t, x, Wp, bias, seq, N, K, ldo);
}

bool Linear::kt_native() {
#if TCPU_ISA_X86
    return hal::env::packed();
#else
    return accel_on();
#endif
}

} // namespace nn
} // namespace tcpu
