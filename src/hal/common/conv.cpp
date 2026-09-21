/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// Conv (im2col + GEMM) and GroupNorm. The GEMM inside conv routes like any
// other linear (packed panels, or Accelerate via conv2d_blas). GroupNorm is
// per-backend: the Pi NEON backend ships the single-stats-pass + fused
// normalize+ReLU rewrite (its stems are DRAM-bound); the other backends keep
// the classic per-group op with relu applied as its own pass after (exactly
// what those branches ran).

#include "../arch.h"
#include "blas.h"
#include "env.h"
#include "../../ops/conv_ops.h"
#include "../../ops/lm_ops.h"
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#if TCPU_HAL_NEON || TCPU_HAL_APPLE
#include <arm_neon.h>
#include <cstddef>
using std::size_t;
#endif

namespace tcpu {

static void im2col(std::vector<float>& col, const float* x, int H, int Wd, int Cin,
                   int k, int stride, int pad, int Hout, int Wout);
static void im2col_panel(float* col, const float* x, int H, int Wd, int Cin,
                         int k, int stride, int pad, int Wout, int p0, int rows);

void conv2d(float* out, const float* x, const float* W, const float* bias,
            int H, int Wd, int Cin, int Cout, int k, int stride, int pad) {
    const int Hout = (H +2*pad-k)/stride + 1;
    const int Wout = (Wd+2*pad-k)/stride + 1;
    const int K = k*k*Cin;
    std::vector<float> col;
    im2col(col, x, H, Wd, Cin, k, stride, pad, Hout, Wout);
    dense_linear(out, col.data(), W, bias, Hout*Wout, Cout, K);
}

// Panel size for the tiled conv: as many output pixels as fit the cache budget,
// but never so many that a thread would get less than ~4 panels of work.
static int conv_panel(int npix, int K, int nth) {
    const int budget = hal::env::conv_budget();   // floats of im2col per thread
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
    const int P = conv_panel(npix, K, nth);
    if (!hal::env::conv_tile() || (size_t)npix*K <= 1024*1024 || P >= npix
        || P < hal::env::conv_min_panel()) {
        std::vector<float> col;
        im2col(col, x, H, Wd, Cin, k, stride, pad, Hout, Wout);
        dense_linear_packed(out, col.data(), Wp, bias, npix, Cout, K);
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
        im2col_panel(c, x, H, Wd, Cin, k, stride, pad, Wout, p0, rows);
        dense_linear_packed(out+(size_t)p0*Cout, c, Wp, bias, rows, Cout, K);
    }
  }
}

void conv2d_blas(float* out, const float* x, const float* W, const float* bias,
                 int H, int Wd, int Cin, int Cout, int k, int stride, int pad) {
    const int Hout = (H +2*pad-k)/stride + 1;
    const int Wout = (Wd+2*pad-k)/stride + 1;
    std::vector<float> col;
    im2col(col, x, H, Wd, Cin, k, stride, pad, Hout, Wout);
    dense_linear_blas(out, col.data(), W, bias, Hout*Wout, Cout, k*k*Cin);
}

// One (oy, ox, ky) row of the im2col patch: the k taps along kx.
//
// In NHWC the taps of a row sit at x[((iy*Wd) + ix0+kx)*Cin], i.e. back to back in
// memory, so the whole row is one contiguous run - only the part that hangs off
// the left or right edge has to be zeroed. Copying it tap by tap instead costs a
// memcpy call per tap, and for the 7x7/s2 stem a tap is Cin=3 floats: 49 twelve-byte
// calls per output pixel, 3.8M of them per frame, which is why the stem measured
// 224 GFLOP/s against 600 for every other conv. Same bytes, same order, one call.
static inline void im2col_row(float* dst, const float* x, int H, int Wd, int Cin,
                              int k, int iy, int ix0) {
    if (iy < 0 || iy >= H) {
        std::memset(dst, 0, sizeof(float)*k*Cin);
        return;
    }

    // taps [kx0, kx1) are the ones inside the image
    int kx0 = ix0 < 0 ? -ix0 : 0;
    int kx1 = ix0+k > Wd ? Wd-ix0 : k;
    if (kx0 > k) kx0 = k;
    if (kx1 < kx0) kx1 = kx0;

    if (kx0 > 0)
        std::memset(dst, 0, sizeof(float)*(size_t)kx0*Cin);
    if (kx1 > kx0)
        std::memcpy(dst+(size_t)kx0*Cin, x+((size_t)iy*Wd+ix0+kx0)*Cin,
                    sizeof(float)*(size_t)(kx1-kx0)*Cin);
    if (kx1 < k)
        std::memset(dst+(size_t)kx1*Cin, 0, sizeof(float)*(size_t)(k-kx1)*Cin);
}

// im2col for one panel: output pixels [p0, p0+rows) in raster order, one row of
// col per pixel (same layout the whole-image form produces).
static void im2col_panel(float* col, const float* x, int H, int Wd, int Cin,
                         int k, int stride, int pad, int Wout, int p0, int rows) {
    const int K = k*k*Cin;

    for (int r=0; r<rows; r++) {
        const int oy = (p0+r)/Wout;
        const int ox = (p0+r)%Wout;
        float* row = col+(size_t)r*K;

        for (int ky=0; ky<k; ky++)
            im2col_row(row+(size_t)ky*k*Cin, x, H, Wd, Cin, k,
                       oy*stride-pad+ky, ox*stride-pad);
    }
}

static void im2col(std::vector<float>& col, const float* x, int H, int Wd, int Cin,
                   int k, int stride, int pad, int Hout, int Wout) {
    const int K = k*k*Cin;
    col.resize((size_t)Hout*Wout*K);

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int oy=0; oy<Hout; oy++) {
        for (int ox=0; ox<Wout; ox++) {
            float* row = col.data()+((size_t)oy*Wout+ox)*K;
            for (int ky=0; ky<k; ky++)
                im2col_row(row+(size_t)ky*k*Cin, x, H, Wd, Cin, k,
                           oy*stride-pad+ky, ox*stride-pad);
        }
    }
}

void maxpool2d(float* out, const float* x, int H, int Wd, int C, int k, int stride, int pad) {
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
        }
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

#if defined(_OPENMP)
    #pragma omp parallel
#endif
  {
    std::vector<double> l1(C, 0.0), l2(C, 0.0);
#if defined(_OPENMP)
    #pragma omp for schedule(static) nowait
#endif
    for (int p=0; p<n_pixels; p++) {
        const float* xp = x+(size_t)p*C;
        for (int c=0; c<C; c++) {
            const double d = (double)xp[c] - K[c];
            l1[c] += d;
            l2[c] += d*d;
        }
    }
#if defined(_OPENMP)
    #pragma omp critical
#endif
    for (int c=0; c<C; c++) { S1[c] += l1[c]; S2[c] += l2[c]; }
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
    #pragma omp parallel for schedule(static)
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
// panel tiling conv2d needs. Same dense_linear underneath, so it routes to the
// same backend kernel.

void im2col1d(std::vector<float>& col, const float* x, int T, int Cin,
              int k, int stride, int pad, int Tout) {
    const int K = k*Cin;
    col.resize((size_t)Tout*K);
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

void conv1d(float* out, const float* x, const float* W, const float* bias,
            int T, int Cin, int Cout, int k, int stride, int pad) {
    const int Tout = (T + 2*pad - k)/stride + 1;
    std::vector<float> col;
    im2col1d(col, x, T, Cin, k, stride, pad, Tout);
    dense_linear(out, col.data(), W, bias, Tout, Cout, k*Cin);
}

void conv_transpose1d(float* out, const float* x, const float* W, const float* bias,
                      int T, int Cin, int Cout, int k, int stride, int pad) {
    const int Tout = (T-1)*stride - 2*pad + k;

    // Seed with the bias, then scatter-accumulate each input position's kernel
    // footprint. Accumulation order is fixed by the loop nest (input-major, then
    // kernel tap), so the float sum is deterministic across runs and threads.
    if (bias) for (int ot=0; ot<Tout; ot++)
                  std::memcpy(out+(size_t)ot*Cout, bias, sizeof(float)*(size_t)Cout);
    else      std::memset(out, 0, sizeof(float)*(size_t)Tout*Cout);

    for (int t=0; t<T; t++) {
        const float* xt = x+(size_t)t*Cin;
        for (int kt=0; kt<k; kt++) {
            const int ot = t*stride - pad + kt;
            if (ot < 0 || ot >= Tout) continue;
            float* dst = out+(size_t)ot*Cout;
            for (int ci=0; ci<Cin; ci++) {
                const float v = xt[ci];
                if (v == 0.0f) continue;
                const float* w = W+((size_t)ci*k + kt)*Cout;
                for (int co=0; co<Cout; co++) dst[co] += v*w[co];
            }
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

} // namespace tcpu
