/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// Image loading (stb_image, vendored from TinyChatEngine) and the two resize
// conventions the models need.
//
// Both resizes are bilinear, align_corners=False, no antialias, matching torch's
// F.interpolate(mode="bilinear") and so lerobot's pipelines. HF's image
// processors use PIL bicubic instead, so a checkpoint validated through PIL
// differs slightly (within model noise).

#pragma once
#include <cstdint>

namespace tcpu {

// Load an image file (jpg/png/...) and bilinear-resize to target x target, uint8
// HWC (for models that consume raw uint8 pixels, e.g. Octo). False on failure.
bool load_rgb_resized(const char* path, int target, uint8_t* out);

// Same, to a non-square ow x oh (ACT keeps the camera's native 640x480 frame).
bool load_rgb_resized_wh(const char* path, int ow, int oh, uint8_t* out);

// Bilinear-resize an HWC uint8 RGB frame to [3,S,S] float in [0,1]. This is
// torch's uint8 -> float()/255 -> F.interpolate(bilinear, align_corners=False)
// with the divide moved past the interpolation, which is the same linear
// combination up to fp32 rounding and, unlike resizing in uint8 space, does not
// requantize the intermediate to 1/255. MicroVLA needs it: LIBERO renders
// 256x256 and RADIO runs at 224.
void resize_rgb_chw01(const uint8_t* src, int h, int w, int S, float* dst);

// lerobot resize_with_pad: scale the longer side to S keeping the aspect ratio,
// pad the remainder top-left with 0, then map [0,255] -> [-1,1] (mean = std =
// 0.5). src is HWC uint8 RGB, dst is [3,S,S] float. Padding is written as -1.0f
// because the pad happens before the range map, as in lerobot.
//
// Computed in double: once per frame, and compared element-wise against the
// NumPy reference.
void resize_with_pad(const uint8_t* src, int h, int w, int S, float* dst);

// Load an image file and resize_with_pad it into out [3,S,S]. False on failure.
bool load_resize_with_pad(const char* path, int S, float* out);

} // namespace tcpu
