/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// W8A8 convolution: the int8 twin of conv2d_packed, same im2col-panel structure
// and the same output values up to quantization.
//
// The activation scale is PER TENSOR here, not per row as in dense_linear_i8,
// and that is a deliberate structural choice rather than a shortcut. An im2col
// row is a patch: k*k pixels concatenated. Per-row scales would have to be
// derived from the expanded matrix, which means materializing it in fp32 first
// and paying a full quantizing pass over k*k times the input bytes - for the
// first ResNet block that is a 44 MB matrix to scan, and it measured as most of
// the win. One scale for the whole input tensor instead lets the quantization
// happen ONCE over the input (1/(k*k) of the work), after which im2col moves
// int8 bytes - so the expansion itself also gets 4x cheaper. Zero padding stays
// exact because the quantization is symmetric: 0 maps to 0.
//
// What it costs is precision on feature maps with a wide spatial spread, where
// dim pixels share an exponent with bright ones. The per-output-channel weight
// scales are kept, since that is the axis a folded BatchNorm spreads most.

#include "../arch.h"
#include "../../ops/conv_ops.h"
#include "../../ops/quant_ops.h"
#include "env.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif
using std::size_t;

namespace tcpu {

// One (oy, ox, ky) row of an int8 patch - the byte twin of im2col_row: in NHWC
// the k taps along kx are contiguous, so the in-bounds span is one memcpy and
// only the part hanging off the edge is zeroed.
static inline void im2col_row_i8(int8_t* dst, const int8_t* x, int H, int Wd, int Cin,
                                 int k, int iy, int ix0) {
    if (iy < 0 || iy >= H) {
        std::memset(dst, 0, (size_t)k*Cin);
        return;
    }
    int kx0 = ix0 < 0 ? -ix0 : 0;
    int kx1 = ix0+k > Wd ? Wd-ix0 : k;
    if (kx0 > k) kx0 = k;
    if (kx1 < kx0) kx1 = kx0;

    if (kx0 > 0)
        std::memset(dst, 0, (size_t)kx0*Cin);
    if (kx1 > kx0)
        std::memcpy(dst+(size_t)kx0*Cin, x+((size_t)iy*Wd+ix0+kx0)*Cin,
                    (size_t)(kx1-kx0)*Cin);
    if (kx1 < k)
        std::memset(dst+(size_t)kx1*Cin, 0, (size_t)(k-kx1)*Cin);
}

static void im2col_panel_i8(int8_t* col, const int8_t* x, int H, int Wd, int Cin,
                            int k, int stride, int pad, int Wout, int p0, int rows, int Kp) {
    const int K = k*k*Cin;
    for (int r=0; r<rows; r++) {
        const int oy = (p0+r)/Wout;
        const int ox = (p0+r)%Wout;
        int8_t* row = col+(size_t)r*Kp;

        for (int ky=0; ky<k; ky++)
            im2col_row_i8(row+(size_t)ky*k*Cin, x, H, Wd, Cin, k,
                          oy*stride-pad+ky, ox*stride-pad);
        if (Kp > K) std::memset(row+K, 0, (size_t)(Kp-K));   // k-group padding
    }
}

// Per-tensor absmax of the input, then convert. Both passes are threaded; the
// tensor is read twice, which is cheap next to the k*k expansion that follows.
static float quantize_tensor_i8(const float* x, int8_t* xq, size_t n) {
    float amax = 0.0f;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) reduction(max:amax) if(n > (size_t)hal::env::omp_min())
#endif
    for (size_t i=0; i<n; i++) {
        const float a = std::fabs(x[i]);
        if (a > amax) amax = a;
    }
    const float s = amax > 0.0f ? amax*hal::env::i8_clip()/127.0f : 1.0f;
    const float inv = 1.0f/s;

    // std::lrint does not vectorize - GCC emits a libm call per element.
    const size_t chunk = 4096;   // multiple of the 32-wide body: one tail, at the end
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(n > (size_t)hal::env::omp_min())
#endif
    for (size_t i=0; i<n; i += chunk) {
        const size_t m = n-i < chunk ? n-i : chunk;
        quantize_span_i8(x+i, xq+i, m, inv);
    }
    return s;
}

static int conv_panel_i8(int npix, int K, int nth) {
    const int budget = 4*hal::env::conv_budget(); // bytes: int8 rows are 4x denser
    const int p_cache  = K     >= budget ? 1 : budget/K;
    const int p_thread = 4*nth >= npix   ? 1 : npix/(4*nth);
    return p_cache < p_thread ? p_cache : p_thread;
}

void conv2d_i8(float* out, const float* x, const int8_t* Wq, const float* wscale,
               const float* bias, int H, int Wd, int Cin, int Cout, int k,
               int stride, int pad, std::vector<int8_t>& xq_scratch) {
    const int Hout = (H +2*pad-k)/stride + 1;
    const int Wout = (Wd+2*pad-k)/stride + 1;
    const int K    = k*k*Cin;
    const int Kp   = i8_kpad(K);
    const int npix = Hout*Wout;

    const size_t nin = (size_t)H*Wd*Cin;
    if (xq_scratch.size() < nin) xq_scratch.resize(nin);
    const float sa = quantize_tensor_i8(x, xq_scratch.data(), nin);
    const int8_t* xi = xq_scratch.data();

    int nth = 1;
#if defined(_OPENMP)
    nth = omp_get_max_threads();
#endif
    const int P = conv_panel_i8(npix, Kp, nth);
    if (!hal::env::conv_tile() || (size_t)npix*Kp <= 1024*1024 || P >= npix
        || P < hal::env::conv_min_panel()) {
        static thread_local std::vector<int8_t> col;
        static thread_local std::vector<float>  as;
        if (col.size() < (size_t)npix*Kp) col.resize((size_t)npix*Kp);
        as.assign((size_t)npix, sa);
        int8_t* c = col.data();
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) if((size_t)npix*Kp > (size_t)hal::env::omp_min())
#endif
        for (int r0=0; r0<npix; r0+=64)
            im2col_panel_i8(c+(size_t)r0*Kp, xi, H, Wd, Cin, k, stride, pad, Wout,
                            r0, npix-r0 < 64 ? npix-r0 : 64, Kp);
        dense_linear_i8_pre(out, c, as.data(), Wq, wscale, bias, npix, Cout, K);
        return;
    }

    const int npanels = (npix+P-1)/P;
#if defined(_OPENMP)
    #pragma omp parallel
#endif
  {
    static thread_local std::vector<int8_t> col;
    static thread_local std::vector<float> as;
    if (col.size() < (size_t)P*Kp) col.resize((size_t)P*Kp);
    if (as.size()  < (size_t)P   ) as.assign((size_t)P, sa);
    else for (int i=0; i<P; i++) as[i] = sa;   // one scale for every row of this panel
    int8_t* c = col.data();
#if defined(_OPENMP)
    #pragma omp for schedule(dynamic)
#endif
    for (int p=0; p<npanels; p++) {
        const int p0   = p*P;
        const int rows = npix-p0 < P ? npix-p0 : P;
        im2col_panel_i8(c, xi, H, Wd, Cin, k, stride, pad, Wout, p0, rows, Kp);
        dense_linear_i8_pre(out+(size_t)p0*Cout, c, as.data(), Wq, wscale, bias,
                            rows, Cout, K);
    }
  }
}

} // namespace tcpu
