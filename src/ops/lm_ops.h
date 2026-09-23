/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdint>
#include <cstring>

// Portable LM-layer op API. Layout convention:
//   activations: row-major [seq, hidden], element (t,h) at x[t*hidden + h]
//   heads:       row-major [seq, n_heads, head_dim]
// Implementations live in src/hal/ (one backend per target: x86 AVX2,
// Apple-Silicon NEON + Accelerate, generic ARM NEON, scalar). Callers see the
// same math a given platform's hardware branch shipped.

namespace tcpu {

// out[t,:] = x[t,:] / sqrt(mean(x[t,:]^2) + eps) * w   (per token)
void rmsnorm(float* out, const float* x, const float* w, int seq, int hidden, float eps);

// In-place NeoX RoPE on x [seq, n_heads, head_dim]. pos[t] is the position of token t.
// Half-split pairs (i, i+head_dim/2); theta_i = pos * base^(-2i/head_dim). Matches ggml NEOX
// with freq_scale=1, ext_factor=0, attn_factor=1. n_dims == head_dim (full rotation).
void rope_neox(float* x, const int* pos, int seq, int n_heads, int head_dim, float base);

// GQA attention with an explicit additive mask and distinct query/key lengths (cross-attn).
// Q [seq_q,n_q,hd], K/V [seq_k,n_kv,hd] -> out [seq_q,n_q,hd]. mask[seq_q,seq_k] is added to
// scores before softmax (0 = keep, -inf = block). Matches SmolVLA eager_attention_forward.
// K_pre: optional pre-transposed keys [n_kv, head_dim, hal::kt_stride(seq_k)] (e.g. from
// dense_linear_packed_kt); skips the internal transpose. AVX2 path only - pass K too.
void gqa_attention_masked(float* out, const float* Q, const float* K, const float* V,
                          int seq_q, int seq_k, int n_q, int n_kv, int head_dim,
                          float scale, const float* mask, const float* K_pre = nullptr);

// Dense GQA attention: gqa_attention_masked's semantics with no mask at all
// (every query attends every key), for callers that would otherwise pass an
// all-zero [seq_q, seq_k] array. Same arguments minus the mask. Backends may
// block it differently from the masked path - the NEON one does, because the
// masked path's per-tile panel re-streaming is what bounds a 1024-token ViT -
// but the values are those of the masked op with a zero mask.
void gqa_attention_dense(float* out, const float* Q, const float* K, const float* V,
                         int seq_q, int seq_k, int n_q, int n_kv, int head_dim,
                         float scale, const float* K_pre = nullptr);

// MHA with a per-head additive bias (T5 relative position bias). Q [seq_q,n,hd],
// K/V [seq_k,n,hd] -> out [seq_q,n,hd]. bias[n,seq_q,seq_k] added to scores before
// softmax (also carries any -inf pad mask). T5 uses scale=1.0 (folded into init).
void mha_attention_bias(float* out, const float* Q, const float* K, const float* V,
                        int seq_q, int seq_k, int n_heads, int head_dim,
                        float scale, const float* bias);

// out[i] = silu(g[i]) * u[i] = g[i]/(1+exp(-g[i])) * u[i]   (SwiGLU gate)
void silu_gate(float* out, const float* g, const float* u, int n);

// silu/swish in place: x[i] = x[i]/(1+exp(-x[i]))
void silu(float* x, int n);

// LayerNorm with weight+bias (per token over hidden): out = (x-mean)/sqrt(var+eps)*w + b.
void layernorm(float* out, const float* x, const float* w, const float* b, int seq, int hidden, float eps);

// GELU (gelu_pytorch_tanh), in place over n elements.
void gelu_tanh(float* x, int n);  // gelu_pytorch_tanh (SigLIP MLP)

// Exact GELU, in place over n elements: x * 0.5 * (1 + erf(x/sqrt(2))). This is
// torch's nn.GELU() / HF "gelu", NOT the tanh approximation above - the two
// differ by ~1e-3 around |x| ~ 2, which is a hundred times the fp32 noise floor
// a golden test allows. DINOv3, BERT and the TurboVLA projections need this one.
void gelu_erf(float* x, int n);

// ReLU in place over n elements.
void relu(float* x, int n);

// Mish in place: x[i] = x[i] * tanh(softplus(x[i])). Softplus is evaluated the
// stable way torch does it -- log1p(exp(x)) overflows in fp32 well before the
// activations a UNet actually reaches, and the threshold branch is what keeps
// this bit-comparable with the reference at large |x|.
void mish(float* x, int n);

// Dense fp32 linear: out[t,n] = sum_k x[t,k]*W[n,k] + bias[n]. W is [N,K] row-major
// (nn.Linear layout). bias optional. x:[seq,K] out:[seq,N]. Threaded over rows.
void dense_linear(float* out, const float* x, const float* W, const float* bias,
                  int seq, int N, int K);

// Same, but weights are bf16 (uint16 = top 16 bits of fp32; lossless for bf16 checkpoints).
// Each output-row block is dequantized to an fp32 scratch once, then reused across all
// activation rows, so the conversion cost is amortized (no per-use overhead) while the
// resident weights stay half-size. Activations + accumulation are fp32.
void dense_linear_bf16(float* out, const float* x, const uint16_t* W, const float* bias,
                       int seq, int N, int K);

inline float bf16_f32(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Packed-panel GEMM (BLIS-style 6x16 micro-kernel). The C tile lives in registers, so
// the inner loop is FMA-bound instead of load-port-bound like the row-blocked kernel
// (~25% of peak). Requires N % 16 == 0 and weights repacked ONCE via pack_weights16
// (per 16-row block b: Wp[b*K*16 + k*16 + j] = W[(b*16+j)*K + k]). Each output element
// accumulates k in order (plain dot order; fp32 reorder class, same as dense_linear).
void pack_weights16(const float* W, float* Wp, int N, int K);
void dense_linear_packed(float* out, const float* x, const float* Wp, const float* bias,
                         int seq, int N, int K);

// As dense_linear_packed with gelu_tanh applied in the store epilogue (same vector
// formula on the same values -> bit-identical to dense_linear_packed + gelu_tanh),
// skipping one full read+write pass over the output. AVX2 only.
void dense_linear_packed_gelu(float* out, const float* x, const float* Wp, const float* bias,
                              int seq, int N, int K);

// As dense_linear_packed but ACCUMULATES into out (out += x*Wp^T + bias): the
// residual add fused into the store epilogue, same x + (c+bias) add as the
// separate pass -> bit-identical. AVX2 only.
void dense_linear_packed_add(float* out, const float* x, const float* Wp, const float* bias,
                             int seq, int N, int K);

// As dense_linear_packed, but the packed weights are bf16 (top 16 bits of the fp32,
// same [b][k][16] layout) - halves the weight bytes streamed, for memory-bound GEMMs.
// Not bit-exact (weights lose 16 mantissa bits). NEON kernel; scalar fallback.
void dense_linear_packed_bf16(float* out, const float* x, const uint16_t* Wp, const float* bias,
                              int seq, int N, int K);

// Same GEMM, output stored TRANSPOSED: out_t[n*ldo + t] (in-register 8x8 transpose
// epilogue, masked stores). Produces attention's K^T layout straight from the K
// projection, skipping the separate transpose pass. Same accumulation order as
// dense_linear_packed -> identical values. AVX2 only (falls back to packed + copy).
void dense_linear_packed_kt(float* out_t, const float* x, const float* Wp, const float* bias,
                            int seq, int N, int K, int ldo);

} // namespace tcpu
