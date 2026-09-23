/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "image.h"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace tcpu {

void resize_with_pad(const uint8_t* src, int h, int w, int S, float* dst) {
    if (h <= 0 || w <= 0 || S <= 0) {                            // all-pad, never 0/0
        for (size_t i = 0; i < (size_t)3*S*S; i++) dst[i] = -1.0f;
        return;
    }

    const double ratio = std::max((double)w/S, (double)h/S);
    int rw = (int)(w/ratio), rh = (int)(h/ratio);
    if (rw < 1) rw = 1;
    if (rh < 1) rh = 1;
    const int pad_w = S-rw, pad_h = S-rh;
    const double sx = (double)w/rw, sy = (double)h/rh;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < S; y++) {
        const int oy = y - pad_h;
        for (int c = 0; c < 3; c++)                              // pad 0 -> *2-1 = -1
            std::fill_n(dst + (size_t)c*S*S + (size_t)y*S, oy < 0 ? S : pad_w, -1.0f);
        if (oy < 0) continue;
        const double fy = (oy+0.5)*sy - 0.5;
        int y0 = (int)std::floor(fy); const double dy = fy - y0;
        const int y1 = std::min(y0+1, h-1); y0 = std::max(y0, 0);
        for (int ox = 0; ox < rw; ox++) {
            const double fx = (ox+0.5)*sx - 0.5;
            int x0 = (int)std::floor(fx); const double dx = fx - x0;
            const int x1 = std::min(x0+1, w-1); x0 = std::max(x0, 0);
            for (int c = 0; c < 3; c++) {
                const double p00 = src[((size_t)y0*w+x0)*3+c], p01 = src[((size_t)y0*w+x1)*3+c];
                const double p10 = src[((size_t)y1*w+x0)*3+c], p11 = src[((size_t)y1*w+x1)*3+c];
                const double v = (p00*(1-dx) + p01*dx)*(1-dy) + (p10*(1-dx) + p11*dx)*dy;
                dst[(size_t)c*S*S + (size_t)(pad_h+oy)*S + (pad_w+ox)] = (float)(v/255.0*2.0 - 1.0);
            }
        }
    }
}

} // namespace tcpu
