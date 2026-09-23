/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "arch.h"

// Pulls in the selected backend's SIMD primitives (simd_dot / simd_axpy / exp /
// horizontal reductions). HAVE_SIMD guards the shared op bodies that only need
// a dot/axpy pair and fall back to plain loops on the scalar backend.
#if TCPU_HAL_X86
#include "avx2/simd.h"
#define HAVE_SIMD 1
#elif TCPU_ISA_ARM
#include "neon/simd.h"
#define HAVE_SIMD 1
#endif
