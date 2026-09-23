/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// The resize is bilinear, align_corners=False, no antialias, matching torch's
// F.interpolate(mode="bilinear") and so lerobot's pipelines. HF's image
// processors use PIL bicubic instead, so a checkpoint validated through PIL
// differs slightly (within model noise).

#pragma once
#include <cstdint>

namespace tcpu {

// lerobot resize_with_pad: scale the longer side to S keeping the aspect ratio,
// pad the remainder top-left with 0, then map [0,255] -> [-1,1] (mean = std =
// 0.5). src is HWC uint8 RGB, dst is [3,S,S] float. Padding is written as -1.0f
// because the pad happens before the range map, as in lerobot.
//
// Computed in double: once per frame, and compared element-wise against the
// NumPy reference.
void resize_with_pad(const uint8_t* src, int h, int w, int S, float* dst);

} // namespace tcpu
