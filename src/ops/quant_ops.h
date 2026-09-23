/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstddef>
#include <cstdint>

// Symmetric W8A8 GEMM (int8 weights x int8 activations, int32 accumulate).
//
// Model-agnostic, like the rest of ops/: a caller hands over fp32 weights once,
// gets back int8 panels + per-output-row scales, and from then on calls
// dense_linear_i8 with fp32 activations that the kernel quantizes per token.
//
// Numerics. Both sides are symmetric absmax with no zero point, so the inner
// loop is a plain signed dot with no cross-terms:
//   w_q[n,k] = round(W[n,k] / s_w[n]),  s_w[n] = max_k|W[n,k]| / 127
//   x_q[t,k] = round(x[t,k] / s_a[t]),  s_a[t] = max_k|x[t,k]| / 127
//   out[t,n]  = bias[n] + s_a[t]*s_w[n] * SUM_k w_q[n,k]*x_q[t,k]
// The K-loop stays integer and the two scales are applied once per output, not
// per element ("delayed scaling"). Scales are per row / per token rather than
// per tensor because that is what keeps a ResNet's channel spread and a
// transformer's token outliers from collapsing onto one exponent.
//
// This is NOT in the fp32-reorder class the rest of the engine keeps: it is a
// lossy representation and needs its own accuracy argument per model. Callers
// opt in explicitly; nothing routes here by default.
//
// Overflow: |w_q| <= 127 and |x_q| <= 127, so one product <= 16129 and an int32
// accumulator is safe to K = 2^31/16129 = 133k - far beyond any K here (the
// widest is SmolVLA's 12288 modality projection), so the kernel never flushes.
// The x86 kernel biases activations to unsigned, which widens one factor to 255
// and the corrected accumulator to ~6e8 - still inside int32 at every K here.

namespace tcpu {

// True when this build has an int8 kernel AND the CPU supports it (checked
// once, at first call). False -> callers must keep their fp32 path.
bool int8_gemm_available();

// K rounded up to the kernel's k-group (4 for both NEON sdot and x86 vpdpbusd);
// weights and activations are both zero-padded to it so the hot loop has no tail.
int i8_kpad(int K);

// Packed int8 weights for dense_linear_i8. Layout mirrors pack_weights16's
// [block][k][16] but with the 4 k-values of a dot-product group made
// contiguous per row: [N/16][Kp/4][16][4]. One 64-byte group feeds either four
// vdotq_s32 (NEON, 4 lanes = 4 output rows) or two vpdpbusd (x86, 8 dwords =
// 8 output rows). Requires N % 16 == 0.
//
// The buffer carries an int32 row-sum table after the panels - see i8_rowsums.
size_t packed_i8_floats(int N, int K);   // int8 elements in Wq, panels + row sums
// Same buffer counted in int32 words. Owners allocate std::vector<int32_t> and
// hand out (int8_t*): the row-sum table is then read as the int32 objects it is,
// instead of punned out of int8 storage.
size_t packed_i8_words(int N, int K);
void pack_weights_i8(const float* W, int8_t* Wq, float* wscale, int N, int K);

// sum_k w_q[n,k] for each output row, stored by pack_weights_i8 after the
// panels. x86 vpdpbusd is unsigned x signed, so the kernel feeds activations as
// x_u = x_q + 128 and subtracts 128 * rowsum[n] to recover the signed dot. NEON
// sdot is signed x signed and ignores this table.
const int32_t* i8_rowsums(const int8_t* Wq, int N, int K);

// out [seq,N] = x [seq,K] * W^T + bias, with x quantized per token internally.
// xq_scratch must hold seq*i8_kpad(K) int8 and ascale_scratch seq floats; both
// are caller-owned so a control loop allocates nothing per call.
void dense_linear_i8(float* out, const float* x, const int8_t* Wq, const float* wscale,
                     const float* bias, int seq, int N, int K,
                     int8_t* xq_scratch, float* ascale_scratch);

// Quantize x [seq,K] -> xq [seq,i8_kpad(K)] + per-token scale. Exposed because
// a caller that feeds the same activations to several layers (attention's q/k/v)
// can quantize once and call the kernel directly.
void quantize_act_i8(const float* x, int8_t* xq, float* ascale, int seq, int K);

// q[i] = clamp(rne(x[i]*inv), -127, 127). One shared scale, so conv (per-tensor)
// and the row quantizers round identically.
void quantize_span_i8(const float* x, int8_t* q, size_t n, float inv);

// As dense_linear_i8 but the activations are already quantized.
void dense_linear_i8_pre(float* out, const int8_t* xq, const float* ascale,
                         const int8_t* Wq, const float* wscale, const float* bias,
                         int seq, int N, int K);

// Scalar reference for the same call, compiled on every backend. Integer
// accumulation has no reorder freedom, so a vector kernel must match it exactly;
// the tests assert that.
void dense_linear_i8_ref(float* out, const int8_t* xq, const float* ascale,
                         const int8_t* Wq, const float* wscale, const float* bias,
                         int seq, int N, int K);

} // namespace tcpu
