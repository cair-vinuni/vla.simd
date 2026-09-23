/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdint>
#include <vector>

// Convolution-stack ops (fp32, NHWC channel-last, matching flax). Used by CNN image
// encoders (e.g. Octo SmallStem). Model-agnostic.

namespace tcpu {

// 2D convolution, square kernel/stride, symmetric zero padding. Single image.
//   x [H, W, Cin]  ->  out [Hout, Wout, Cout],  Hout = (H + 2*pad - k)/stride + 1.
// W is [Cout, k, k, Cin] row-major (patch layout matches im2col row). bias optional.
// Implemented as im2col + dense_linear so it reuses the threaded GEMM kernel.
void conv2d(float* out, const float* x, const float* W, const float* bias,
            int H, int Wd, int Cin, int Cout, int k, int stride, int pad);

// Same, with weights pre-packed by pack_weights16 (N = Cout, K = k*k*Cin);
// requires Cout % 16 == 0. Uses the packed-panel GEMM.
void conv2d_packed(float* out, const float* x, const float* Wp, const float* bias,
                   int H, int Wd, int Cin, int Cout, int k, int stride, int pad);

// Same conv, GEMM routed to a vendor BLAS (Accelerate/AMX on Apple); raw W layout.
// W8A8 convolution (see hal/common/conv_i8.cpp). Weights are pre-packed by
// pack_weights_i8 over K = k*k*Cin; the activation scale is per input tensor.
// xq_scratch is caller-owned and grown once, so a control loop allocates
// nothing per frame.
void conv2d_i8(float* out, const float* x, const int8_t* Wq, const float* wscale,
               const float* bias, int H, int Wd, int Cin, int Cout, int k,
               int stride, int pad, std::vector<int8_t>& xq_scratch);

void conv2d_blas(float* out, const float* x, const float* W, const float* bias,
                 int H, int Wd, int Cin, int Cout, int k, int stride, int pad);

// Max pooling, square kernel/stride, symmetric padding. Single image, channel-last:
//   x [H, W, C]  ->  out [Hout, Wout, C],  Hout = (H + 2*pad - k)/stride + 1.
// Padded cells count as -inf (torch MaxPool2d, ceil_mode=false).
void maxpool2d(float* out, const float* x, int H, int Wd, int C, int k, int stride, int pad);

// maxpool2d with the preceding ReLU folded into its epilogue, for the ResNet stem
// where the pool is the relu'd map's only consumer. ReLU is monotonic, so
// max_i relu(v_i) == max(0, max_i v_i) == relu(max_i v_i): the two orders are the
// same float, and running the pool first means relu touches the pooled map (4x
// smaller) instead of the conv output. Saves a full read and write of the stem's
// 19.7 MB at 480x640. The one input that would differ -- a window with every tap
// out of bounds, which the separate pass would leave at -inf -- cannot occur while
// pad < k, and is 0 here rather than -inf.
void maxpool2d_relu(float* out, const float* x, int H, int Wd, int C, int k, int stride, int pad);

// --- 1D convolution stack (NLC channel-last), for temporal UNets -------------
// Same im2col + GEMM shape as conv2d, one spatial axis instead of two. The action
// sequence is the "length" axis and channels stay last, so groupnorm() below
// applies unchanged with n_pixels = T.

// 1D convolution, symmetric zero padding.
//   x [T, Cin] -> out [Tout, Cout],  Tout = (T + 2*pad - k)/stride + 1.
// W is [Cout, k, Cin] row-major (torch Conv1d weight is [Cout, Cin, k]; the
// converter transposes). bias optional.
void conv1d(float* out, const float* x, const float* W, const float* bias,
            int T, int Cin, int Cout, int k, int stride, int pad);

// The im2col expansion conv1d runs on, exposed because the UNet's blocks hold
// their weights as nn::Linear (packed panels) and drive the GEMM themselves:
//   col [Tout, k*Cin], row ot = the k taps at ot*stride - pad, zero outside.
void im2col1d(std::vector<float>& col, const float* x, int T, int Cin,
              int k, int stride, int pad, int Tout);

// 1D transposed convolution ("upsample"), symmetric padding, no output padding.
//   x [T, Cin] -> out [Tout, Cout],  Tout = (T-1)*stride - 2*pad + k.
// W is [Cin, k, Cout] row-major (torch ConvTranspose1d weight is [Cin, Cout, k];
// the converter transposes). Scatter-accumulate rather than im2col: the kernel
// sizes here are tiny and the gather form would need a fractionally-strided
// index map for no gain.
void conv_transpose1d(float* out, const float* x, const float* W, const float* bias,
                      int T, int Cin, int Cout, int k, int stride, int pad);

// Spatial soft-argmax (Finn et al.; the robomimic/Diffusion-Policy pooling).
// Per channel, softmax over the n_pixels spatial positions, then the expected
// position under that distribution:
//   x [n_pixels, C] -> out [C, 2], out[c] = sum_p softmax_p(x[.,c]) * grid[p].
// grid is [n_pixels, 2] of normalized (x, y) coordinates, caller-supplied so the
// op stays free of any convention about how the grid is laid out.
void spatial_softmax(float* out, const float* x, const float* grid,
                     int n_pixels, int C);

// GroupNorm over one image: x [n_pixels, C] (flattened H*W, channel-last).
// Per group g (C/groups consecutive channels), stats over all pixels and the group's
// channels; out = (x-mean)/sqrt(var+eps)*scale + bias (scale/bias per channel).
// fuse_relu additionally applies max(0, .) to the output (GN+ReLU is a standard
// conv-stack pair; how the two ops combine is a backend decision - see hal/conv).
void groupnorm(float* out, const float* x, const float* scale, const float* bias,
               int n_pixels, int C, int groups, float eps, bool fuse_relu = false);

} // namespace tcpu
