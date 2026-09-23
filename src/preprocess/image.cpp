/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "image.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include "stb_image.h"

namespace tcpu {

// bilinear resize (w x h x 3, HWC uint8) -> (ow x oh x 3, HWC float 0..255)
static void resize_bilinear(const uint8_t* src, int w, int h, int ow, int oh, float* dst) {
    const float sx_scale = (float)w/ow;
    const float sy_scale = (float)h/oh;

    for (int oy=0; oy<oh; oy++) {
        float sy = (oy+0.5f)*sy_scale - 0.5f;
        int y0   = (int)std::floor(sy);
        float dy = sy-y0;
        int y0c  = y0 < 0 ? 0 : (y0 > h-1 ? h-1 : y0);
        int y1c  = y0+1 < 0 ? 0 : (y0+1 > h-1 ? h-1 : y0+1);

        for (int ox=0; ox<ow; ox++) {
            float sx = (ox+0.5f)*sx_scale - 0.5f;
            int x0   = (int)std::floor(sx);
            float dx = sx-x0;
            int x0c  = x0 < 0 ? 0 : (x0 > w-1 ? w-1 : x0);
            int x1c  = x0+1 < 0 ? 0 : (x0+1 > w-1 ? w-1 : x0+1);

            for (int c=0; c<3; c++) {
                float v00 = src[3*(y0c*w+x0c)+c];
                float v01 = src[3*(y0c*w+x1c)+c];
                float v10 = src[3*(y1c*w+x0c)+c];
                float v11 = src[3*(y1c*w+x1c)+c];
                float v0  = v00*(1-dx) + v01*dx;
                float v1  = v10*(1-dx) + v11*dx;
                dst[3*(oy*ow+ox)+c] = v0*(1-dy) + v1*dy;
            }
        }
    }
}

void resize_rgb_chw01(const uint8_t* src, int h, int w, int S, float* dst) {
    std::vector<float> hwc((size_t)3*S*S);
    resize_bilinear(src, w, h, S, S, hwc.data());
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < S*S; i++)
            dst[(size_t)c*S*S + i] = hwc[(size_t)3*i + c]*(1.0f/255.0f);
}

void resize_with_pad(const uint8_t* src, int h, int w, int S, float* dst) {
    for (size_t i = 0; i < (size_t)3*S*S; i++) dst[i] = -1.0f;   // pad 0 -> *2-1 = -1
    if (h <= 0 || w <= 0 || S <= 0) return;                      // all-pad, never 0/0

    const double ratio = std::max((double)w/S, (double)h/S);
    int rw = (int)(w/ratio), rh = (int)(h/ratio);
    if (rw < 1) rw = 1;
    if (rh < 1) rh = 1;
    const int pad_w = S-rw, pad_h = S-rh;
    const double sx = (double)w/rw, sy = (double)h/rh;

    for (int oy = 0; oy < rh; oy++) {
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

bool load_resize_with_pad(const char* path, int S, float* out) {
    int w, h, nc;
    uint8_t* data = stbi_load(path, &w, &h, &nc, 3);   // force RGB
    if (!data) return false;
    resize_with_pad(data, h, w, S, out);
    stbi_image_free(data);
    return true;
}

bool load_rgb_resized(const char* path, int target, uint8_t* out) {
    return load_rgb_resized_wh(path, target, target, out);
}

bool load_rgb_resized_wh(const char* path, int ow, int oh, uint8_t* out) {
    int w, h, nc;
    uint8_t* data = stbi_load(path, &w, &h, &nc, 3); // force RGB
    if (!data) return false;

    std::vector<float> resized((size_t)ow*oh*3);
    resize_bilinear(data, w, h, ow, oh, resized.data());
    stbi_image_free(data);

    for (size_t i=0; i<resized.size(); i++) {
        float v = resized[i]+0.5f;
        out[i] = (uint8_t)(v < 0.0f ? 0 : v > 255.0f ? 255 : v);
    }
    return true;
}

} // namespace tcpu
