/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// AMD Zen SIMD primitives.
//
// The scalar primitives (simd_dot / simd_axpy / hsum8 / hmax8 / exp256_ps) are
// shared verbatim with the Intel AVX2 backend rather than copied: Zen 3 and
// Raptor Lake have the same 2x256-bit FMA issue width and the same ~4-cycle FMA
// latency, so the measured tuning (4 accumulator chains in simd_dot) is optimal
// on both, and sharing keeps the two x86 backends in ONE numerics class - the
// Cephes exp polynomial and every accumulation order stay bit-identical, which
// is what lets the AMD backend reuse the Intel goldens unchanged.
//
// What Zen 3 does NOT share is the memory-side behaviour: 32 KB / 8-way L1 (vs
// 48 KB / 12-way) and 2 load ports (vs 3). Everything that diverges for that
// reason lives in amd/attn.cpp and hal/common/layout.h, not here.
#include "../avx2/simd.h"
