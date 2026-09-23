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

// A dense layer with HAL-prepared weights. The model hands over raw fp32
// W [N,K] (nn.Linear layout) + optional bias and a role; the backend decides
// the runtime representation (packed 16-wide panels, vendor BLAS straight from
// the raw weights, opt-in bf16 mirror) and the kernel. Models never pick.
//
// Roles mirror how the hardware branches routed their GEMMs:
//   Generic   plain dense_linear everywhere (T5, diffusion head)
//   Gemm      qkv / projections: packed panels (TCPU_PACKED), or Accelerate on Apple
//   Mlp       like Gemm, plus the Pi's opt-in bf16 packed weights (TCPU_BF16_MLP)
//   StemGemm  like Gemm but always packed (the stems never had a TCPU_PACKED gate)
struct Linear {
    enum class Role { Generic, Gemm, Mlp, StemGemm };

    // W/bias must outlive this object (models keep the weight arena mapped).
    void init(const float* W, const float* bias, int N, int K, Role role);

    // Same layer from bf16 weights (top 16 bits of the fp32; lossless for bf16
    // checkpoints). Per-backend policy: x86 dequantizes once into packed fp32
    // panels (TCPU_BF16_DEQ=0 keeps the raw bf16 + native bf16 row GEMM); Apple
    // dequantizes once and routes to Accelerate; Pi NEON packs bf16 panels
    // (half the bytes streamed - its GEMMs are bandwidth-bound); scalar keeps
    // the raw pointer. Unless bf16_keeps_raw(), every path owns a copy and the
    // caller may free the bf16 blob after all init_bf16 calls.
    void init_bf16(const uint16_t* Wb16, const float* bias, int N, int K, Role role);
    static bool bf16_keeps_raw();

    // Opt in to the symmetric W8A8 kernel for this layer (see ops/quant_ops.h).
    // Call after init() or init_bf16() - it quantizes from whichever
    // representation that left behind, and frees the one it replaces. Routes
    // every later forward() through the integer kernel, which is lossy: a model
    // enables it per layer and owns the accuracy argument. Returns false and
    // leaves the layer on its fp32 path when the backend or CPU has no int8
    // kernel, or when the shape does not fit it (N % 16 != 0), so a caller can
    // enable unconditionally and still get a working model.
    bool init_int8();
    bool is_int8() const { return Wq != nullptr; }

    // out [seq, N] = x [seq, K] * W^T + bias
    void forward(float* out, const float* x, int seq) const;

    // forward + gelu_tanh on the output. x86 packed path fuses the gelu into
    // the GEMM store epilogue (bit-identical values, one less pass over out);
    // every other path runs forward then the eltwise gelu.
    void forward_gelu(float* out, const float* x, int seq) const;

    // out += x*W^T + bias: residual add fused into the GEMM store epilogue
    // (bit-identical to forward + separate add). Call only when add_native().
    void forward_add(float* out, const float* x, int seq) const;

    // True when forward_add is available (x86 packed path, TCPU_FUSE_RES).
    static bool add_native();

    // Whether this layer has it. The fused epilogue lives in the packed fp32
    // kernel, which init_int8() frees, so an int8 layer takes forward_add's
    // fallback: a seq*N allocation and a second pass, per call.
    bool add_ok() const { return add_native() && Wp; }

    // out_t[n*ldo + t]: attention's K^T layout straight from the GEMM.
    // Call only when kt_native(); columns [seq, ldo) are left untouched
    // (callers keep the buffer zeroed).
    void forward_kt(float* out_t, const float* x, int seq, int ldo) const;

    // True when this backend produces K^T at no extra cost (x86 packed
    // in-register transpose epilogue; Apple transposed-output sgemm).
    static bool kt_native();

    // Whether this layer can. init_int8() frees the fp32 panels and there is no
    // transposed int8 store, so an int8 layer falls back to an allocation plus a
    // scalar scatter transpose - slower than the GEMM it accelerates, and on
    // Apple it computes K in fp32 from the weights init_int8 left behind.
    bool kt_ok() const { return kt_native() && !Wq; }

    int N = 0, K = 0;

  private:
    Role role = Role::Generic;
    const float* W = nullptr;
    const float* bias = nullptr;
    const float* Wp = nullptr;          // packed panels (pack_weights16)
    const uint16_t* Wb = nullptr;       // bf16 packed panels (Pi Mlp opt-in, or init_bf16)
    const uint16_t* Wr16 = nullptr;     // raw bf16 [N,K] (init_bf16 keeps-raw paths)
    const int8_t* Wq = nullptr;         // int8 panels (init_int8)
    std::vector<float> packed;
    std::vector<uint16_t> packed_bf16;
    std::vector<float> deq;             // owned fp32 dequant of a bf16 checkpoint
    std::vector<int32_t> packed_i8;   // int32-typed: see packed_i8_words
    std::vector<float> wscale;          // per-output-row weight scale
};

} // namespace nn
} // namespace tcpu
