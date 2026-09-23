/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// Conv (im2col + GEMM) and GroupNorm. The GEMM inside conv routes like any
// other linear (packed panels, or Accelerate via conv2d_blas).

#include "blas.h"
#include "env.h"
#include "../../ops/conv_ops.h"
#include "../../ops/lm_ops.h"
#include "../../ops/quant_ops.h"
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <cstddef>
using std::size_t;

namespace tcpu {

static const float* im2col(const float* x, int H, int Wd, int Cin,
                           int k, int stride, int pad, int Hout, int Wout);
template <class T>
static void im2col_panel(T* col, const T* x, int H, int Wd, int Cin,
                         int k, int stride, int pad, int Wout, int p0, int rows, int Kp);

void conv2d(float* out, const float* x, const float* W, const float* bias,
            int H, int Wd, int Cin, int Cout, int k, int stride, int pad) {
    const int Hout = (H +2*pad-k)/stride + 1;
    const int Wout = (Wd+2*pad-k)/stride + 1;
    const int K = k*k*Cin;
    dense_linear(out, im2col(x, H, Wd, Cin, k, stride, pad, Hout, Wout), W, bias,
                 Hout*Wout, Cout, K);
}

// Panel size for the tiled conv: as many output pixels as fit the cache budget,
// but never so many that a thread would get less than ~4 panels of work.
static int conv_panel(int npix, int K, int nth, int budget) {
    const int p_cache  = K       >= budget ? 1 : budget/K;
    const int p_thread = 4*nth   >= npix   ? 1 : npix/(4*nth);
    return p_cache < p_thread ? p_cache : p_thread;
}

void conv2d_packed(float* out, const float* x, const float* Wp, const float* bias,
                   int H, int Wd, int Cin, int Cout, int k, int stride, int pad) {
    const int Hout = (H +2*pad-k)/stride + 1;
    const int Wout = (Wd+2*pad-k)/stride + 1;
    const int K    = k*k*Cin;
    const int npix = Hout*Wout;

    // Whole-image im2col writes k*k times the input bytes to DRAM and reads them
    // back for the GEMM: at 480x640 that is ~700 MB per ResNet-18 pass, and the
    // convs stop scaling with cores (measured 1.2x for 8x the threads). Tiling by
    // output-pixel panels keeps each panel's expansion in L2 - the GEMM sees the
    // same rows in the same order, so every output element is bit-identical.
    int nth = 1;
#if defined(_OPENMP)
    nth = omp_get_max_threads();
#endif
    // A panel this thin cannot amortize the weight matrix it streams, so the
    // untiled path - which threads inside the GEMM and keeps each worker on its
    // own weight panels - wins even though its im2col goes through DRAM.
    const int P = conv_panel(npix, K, nth, hal::env::conv_budget());   // floats of im2col per thread
    if (!hal::env::conv_tile() || (size_t)npix*K <= 1024*1024 || P >= npix
        || P < hal::env::conv_min_panel()) {
        dense_linear_packed(out, im2col(x, H, Wd, Cin, k, stride, pad, Hout, Wout), Wp, bias,
                            npix, Cout, K);
        return;
    }

    const int npanels = (npix+P-1)/P;
#if defined(_OPENMP)
    #pragma omp parallel
#endif
  {
    // grow-only per worker; the GEMM below runs serially inside this region
    // (nested parallelism is off by default), so each panel is one thread's work
    static thread_local std::vector<float> col;
    if (col.size() < (size_t)P*K) col.resize((size_t)P*K);
    float* c = col.data();
#if defined(_OPENMP)
    #pragma omp for schedule(dynamic)
#endif
    for (int p=0; p<npanels; p++) {
        const int p0   = p*P;
        const int rows = npix-p0 < P ? npix-p0 : P;
        im2col_panel(c, x, H, Wd, Cin, k, stride, pad, Wout, p0, rows, K);
        dense_linear_packed(out+(size_t)p0*Cout, c, Wp, bias, rows, Cout, K);
    }
  }
}

void conv2d_blas(float* out, const float* x, const float* W, const float* bias,
                 int H, int Wd, int Cin, int Cout, int k, int stride, int pad) {
    const int Hout = (H +2*pad-k)/stride + 1;
    const int Wout = (Wd+2*pad-k)/stride + 1;
    dense_linear_blas(out, im2col(x, H, Wd, Cin, k, stride, pad, Hout, Wout), W, bias,
                      Hout*Wout, Cout, k*k*Cin);
}

// One (oy, ox, ky) row of the im2col patch: the k taps along kx.
//
// In NHWC the taps of a row sit at x[((iy*Wd) + ix0+kx)*Cin], i.e. back to back in
// memory, so the whole row is one contiguous run - only the part that hangs off
// the left or right edge has to be zeroed. Copying it tap by tap instead costs a
// memcpy call per tap, and for the 7x7/s2 stem a tap is Cin=3 floats: 49 twelve-byte
// calls per output pixel, 3.8M of them per frame, which is why the stem measured
// 224 GFLOP/s against 600 for every other conv. Same bytes, same order, one call.
template <class T>
static inline void im2col_row(T* dst, const T* x, int H, int Wd, int Cin,
                              int k, int iy, int ix0) {
    if (iy < 0 || iy >= H) {
        std::memset(dst, 0, sizeof(T)*k*Cin);
        return;
    }

    // taps [kx0, kx1) are the ones inside the image
    int kx0 = ix0 < 0 ? -ix0 : 0;
    int kx1 = ix0+k > Wd ? Wd-ix0 : k;
    if (kx0 > k) kx0 = k;
    if (kx1 < kx0) kx1 = kx0;

    if (kx0 > 0)
        std::memset(dst, 0, sizeof(T)*(size_t)kx0*Cin);
    if (kx1 > kx0)
        std::memcpy(dst+(size_t)kx0*Cin, x+((size_t)iy*Wd+ix0+kx0)*Cin,
                    sizeof(T)*(size_t)(kx1-kx0)*Cin);
    if (kx1 < k)
        std::memset(dst+(size_t)kx1*Cin, 0, sizeof(T)*(size_t)(k-kx1)*Cin);
}

// im2col for one panel: output pixels [p0, p0+rows) in raster order, one row of
// col per pixel (same layout the whole-image form produces).
template <class T>
static void im2col_panel(T* col, const T* x, int H, int Wd, int Cin,
                         int k, int stride, int pad, int Wout, int p0, int rows, int Kp) {
    const int K = k*k*Cin;

    for (int r=0; r<rows; r++) {
        const int oy = (p0+r)/Wout;
        const int ox = (p0+r)%Wout;
        T* row = col+(size_t)r*Kp;

        for (int ky=0; ky<k; ky++)
            im2col_row(row+(size_t)ky*k*Cin, x, H, Wd, Cin, k,
                       oy*stride-pad+ky, ox*stride-pad);
        if (Kp > K) std::memset(row+K, 0, sizeof(T)*(Kp-K));   // k-group padding
    }
}

static const float* im2col(const float* x, int H, int Wd, int Cin,
                           int k, int stride, int pad, int Hout, int Wout) {
    const int K = k*k*Cin;
    static thread_local std::vector<float> col;
    if (col.size() < (size_t)Hout*Wout*K) col.resize((size_t)Hout*Wout*K);
    float* c = col.data();

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int oy=0; oy<Hout; oy++) {
        for (int ox=0; ox<Wout; ox++) {
            float* row = c+((size_t)oy*Wout+ox)*K;
            for (int ky=0; ky<k; ky++)
                im2col_row(row+(size_t)ky*k*Cin, x, H, Wd, Cin, k,
                           oy*stride-pad+ky, ox*stride-pad);
        }
    }
    return c;
}

void extract_patches(const float* pixels, int img, int patch, float* out) {
    const int grid = img/patch, pd = 3*patch*patch;
    for (int ph = 0; ph < grid; ph++)
        for (int pw = 0; pw < grid; pw++) {
            float* dst = out + ((size_t)(ph*grid+pw))*pd;
            for (int ic = 0; ic < 3; ic++)
                for (int kh = 0; kh < patch; kh++)
                    for (int kw = 0; kw < patch; kw++)
                        dst[ic*patch*patch + kh*patch + kw] =
                            pixels[((size_t)ic*img + (ph*patch+kh))*img + (pw*patch+kw)];
        }
}

void maxpool2d_relu(float* out, const float* x, int H, int Wd, int C, int k, int stride, int pad) {
    const int Hout = (H +2*pad-k)/stride + 1;
    const int Wout = (Wd+2*pad-k)/stride + 1;
    const float NINF = -std::numeric_limits<float>::infinity();

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int oy=0; oy<Hout; oy++) {
        for (int ox=0; ox<Wout; ox++) {
            float* o = out+((size_t)oy*Wout+ox)*C;
            for (int c=0; c<C; c++)
                o[c] = NINF;

            for (int ky=0; ky<k; ky++) {
                const int iy = oy*stride-pad+ky;
                if (iy < 0 || iy >= H) continue;

                for (int kx=0; kx<k; kx++) {
                    const int ix = ox*stride-pad+kx;
                    if (ix < 0 || ix >= Wd) continue;

                    const float* xp = x+((size_t)iy*Wd+ix)*C;
                    for (int c=0; c<C; c++)
                        o[c] = xp[c] > o[c] ? xp[c] : o[c];
                }
            }
            // the folded ReLU; see conv_ops.h for why this is the same float as
            // relu'ing the whole conv output first
            for (int c=0; c<C; c++)
                if (!(o[c] > 0.0f)) o[c] = 0.0f;
        }
    }
}

void groupnorm(float* out, const float* x, const float* scale, const float* bias,
               int n_pixels, int C, int groups, float eps, bool fuse_relu) {
    const int gc = C/groups;
    if (n_pixels < 1 || gc < 1) return;

    // Two sequential sweeps: stats, then normalize. Parallelizing over groups
    // instead re-scanned the whole tensor per group (3 x groups strided sweeps),
    // all of it from DRAM past L2.
    //
    // Shifted-data variance, shift = each channel's first pixel. One pass, and
    // stable where E[x^2]-mean^2 cancels. Exact:
    //   sum_p (x-m)^2 = S2 + 2(K-m)S1 + n(K-m)^2,  S1=sum(x-K), S2=sum((x-K)^2)
    std::vector<double> S1(C, 0.0), S2(C, 0.0), K(C);
    for (int c=0; c<C; c++) K[c] = x[c];

    const bool par = (size_t)n_pixels*C > (size_t)hal::env::omp_min();
    int nth = 1;
#if defined(_OPENMP)
    if (par) nth = omp_get_max_threads();
#endif
    if (C >= 16*nth) {
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) if(nth > 1)
#endif
        for (int cb=0; cb<C; cb+=16) {
            const int ce = std::min(cb+16, C);
            for (int p=0; p<n_pixels; p++) {
                const float* xp = x+(size_t)p*C;
                for (int c=cb; c<ce; c++) {
                    const double d = (double)xp[c] - K[c];
                    S1[c] += d;
                    S2[c] += d*d;
                }
            }
        }
    } else {
        std::vector<double> part((size_t)nth*2*C, 0.0);
#if defined(_OPENMP)
        #pragma omp parallel num_threads(nth)
#endif
      {
        int tid = 0;
#if defined(_OPENMP)
        tid = omp_get_thread_num();
#endif
        double* l1 = part.data() + (size_t)tid*2*C;
        double* l2 = l1 + C;
#if defined(_OPENMP)
        #pragma omp for schedule(static)
#endif
        for (int p=0; p<n_pixels; p++) {
            const float* xp = x+(size_t)p*C;
            for (int c=0; c<C; c++) {
                const double d = (double)xp[c] - K[c];
                l1[c] += d;
                l2[c] += d*d;
            }
        }
      }
        for (int t=0; t<nth; t++) {
            const double* l1 = part.data() + (size_t)t*2*C;
            for (int c=0; c<C; c++) { S1[c] += l1[c]; S2[c] += l1[C+c]; }
        }
    }

    // out = x*A + B with A = inv*scale, B = bias - mean*inv*scale
    std::vector<float> A(C), B(C);
    const double n = (double)n_pixels*gc;
    for (int g=0; g<groups; g++) {
        const int c0 = g*gc, c1 = c0+gc;

        double tot = 0.0;
        for (int c=c0; c<c1; c++) tot += S1[c] + (double)n_pixels*K[c];
        const double mean = tot/n;

        double ss = 0.0;
        for (int c=c0; c<c1; c++) {
            const double dk = K[c]-mean;
            ss += S2[c] + 2.0*dk*S1[c] + (double)n_pixels*dk*dk;
        }
        const double inv = 1.0/std::sqrt(ss/n + eps);

        for (int c=c0; c<c1; c++) {
            A[c] = (float)(inv*scale[c]);
            B[c] = (float)(bias[c] - mean*inv*scale[c]);
        }
    }

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(par)
#endif
    for (int p=0; p<n_pixels; p++) {
        const float* xp = x+(size_t)p*C;
        float* op = out+(size_t)p*C;
        for (int c=0; c<C; c++) {
            const float v = xp[c]*A[c] + B[c];
            op[c] = (fuse_relu && v < 0.0f) ? 0.0f : v;
        }
    }
}


// --- 1D convolution stack ----------------------------------------------------
// The UNet1D's sequence is short (horizon 64 down to 16) while its channel counts
// are large (512..2048), so the GEMM is wide and shallow: whole-sequence im2col
// costs k*T*Cin floats, a few hundred KB at the widest, and never justifies the
// panel tiling conv2d needs.

void im2col1d(std::vector<float>& col, const float* x, int T, int Cin,
              int k, int stride, int pad, int Tout) {
    const int K = k*Cin;
    if (col.size() < (size_t)Tout*K) col.resize((size_t)Tout*K);
    for (int ot=0; ot<Tout; ot++) {
        float* row = col.data()+(size_t)ot*K;
        for (int kt=0; kt<k; kt++) {
            const int t = ot*stride - pad + kt;
            float* dst = row+(size_t)kt*Cin;
            if (t < 0 || t >= T) std::memset(dst, 0, sizeof(float)*(size_t)Cin);
            else                 std::memcpy(dst, x+(size_t)t*Cin, sizeof(float)*(size_t)Cin);
        }
    }
}

void conv_transpose1d(float* out, const float* x, const float* W, const float* bias,
                      int T, int Cin, int Cout, int k, int stride, int pad) {
    const int Tout = (T-1)*stride - 2*pad + k;

    // Seed with the bias, then scatter-accumulate each input position's kernel
    // footprint. Accumulation order is fixed by the loop nest, so the float sum
    // is deterministic across runs and threads.
    const int CB = 64;
#if defined(_OPENMP)
    #pragma omp parallel if((size_t)T*Cin*Cout > (size_t)hal::env::omp_min())
#endif
  {
    static thread_local std::vector<float> buf;
    const size_t nx = (size_t)T*Cin;
    if (buf.size() < nx + (size_t)Tout*CB) buf.resize(nx + (size_t)Tout*CB);
    float* xT  = buf.data();
    float* acc = xT + nx;
    for (int t=0; t<T; t++)
        for (int ci=0; ci<Cin; ci++) xT[(size_t)ci*T+t] = x[(size_t)t*Cin+ci];
#if defined(_OPENMP)
    #pragma omp for schedule(static)
#endif
    for (int c0=0; c0<Cout; c0+=CB) {
        const int nc = std::min(CB, Cout-c0);
        std::memset(acc, 0, sizeof(float)*(size_t)Tout*CB);
        if (bias) for (int ot=0; ot<Tout; ot++)
                      std::memcpy(acc+(size_t)ot*CB, bias+c0, sizeof(float)*(size_t)nc);
        float w[CB] = {};
        for (int kt=k-1; kt>=0; kt--) {
            for (int ci=0; ci<Cin; ci++) {
                std::memcpy(w, W+((size_t)ci*k + kt)*Cout+c0, sizeof(float)*(size_t)nc);
                const float* xc = xT+(size_t)ci*T;
                for (int t=0; t<T; t++) {
                    const int ot = t*stride - pad + kt;
                    if (ot < 0 || ot >= Tout) continue;
                    const float v = xc[t];
                    if (v == 0.0f) continue;
                    float* a = acc+(size_t)ot*CB;
                    for (int co=0; co<CB; co++) a[co] += v*w[co];
                }
            }
        }
        for (int ot=0; ot<Tout; ot++)
            std::memcpy(out+(size_t)ot*Cout+c0, acc+(size_t)ot*CB, sizeof(float)*(size_t)nc);
    }
  }
}

void spatial_softmax(float* out, const float* x, const float* grid,
                     int n_pixels, int C) {
    // x is channel-last, so a channel's n_pixels activations are strided by C.
    // Softmax per channel with the standard max shift, accumulating the expected
    // coordinate in the same pass as the normalization.
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int c=0; c<C; c++) {
        float m = -std::numeric_limits<float>::infinity();
        for (int p=0; p<n_pixels; p++) {
            const float v = x[(size_t)p*C + c];
            if (v > m) m = v;
        }
        float sum = 0.f, ex = 0.f, ey = 0.f;
        for (int p=0; p<n_pixels; p++) {
            const float e = std::exp(x[(size_t)p*C + c] - m);
            sum += e;
            ex  += e*grid[(size_t)p*2 + 0];
            ey  += e*grid[(size_t)p*2 + 1];
        }
        const float inv = 1.0f/sum;
        out[(size_t)c*2 + 0] = ex*inv;
        out[(size_t)c*2 + 1] = ey*inv;
    }
}

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
    const int P = conv_panel(npix, Kp, nth, 4*hal::env::conv_budget());   // bytes: int8 rows are 4x denser
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
            im2col_panel(c+(size_t)r0*Kp, xi, H, Wd, Cin, k, stride, pad, Wout,
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
        im2col_panel(c, xi, H, Wd, Cin, k, stride, pad, Wout, p0, rows, Kp);
        dense_linear_i8_pre(out+(size_t)p0*Cout, c, as.data(), Wq, wscale, bias,
                            rows, Cout, K);
    }
  }
}

} // namespace tcpu
