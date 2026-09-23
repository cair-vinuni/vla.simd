/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "common/env.h"

// HAL backend selection. Exactly one TCPU_HAL_* is 1; everything below hal/ keys
// off these, nothing above hal/ may. The two NEON tunings are distinct backends
// on purpose: the M4 (wide OoO core + AMX via Accelerate) and the Cortex-A72
// (narrow, load-limited) want different kernels, and their kernels are not
// numerically interchangeable (accumulator-chain counts differ). Selection is
// compile-time and matches where each tuning was measured.
#if defined(__AVX2__)
  #define TCPU_HAL_X86 1
  #define TCPU_HAL_NAME "x86-avx2"
#elif defined(__ARM_NEON) && defined(__aarch64__) && defined(__APPLE__)
  #define TCPU_HAL_APPLE 1
  #define TCPU_HAL_NAME "apple"
#elif defined(__ARM_NEON) && defined(__aarch64__)
  #define TCPU_HAL_NEON 1
  #define TCPU_HAL_NAME "neon"
#else
  #define TCPU_HAL_SCALAR 1
  #define TCPU_HAL_NAME "scalar"
#endif

#ifndef TCPU_HAL_X86
  #define TCPU_HAL_X86 0
#endif
#ifndef TCPU_HAL_APPLE
  #define TCPU_HAL_APPLE 0
#endif
#ifndef TCPU_HAL_NEON
  #define TCPU_HAL_NEON 0
#endif
#ifndef TCPU_HAL_SCALAR
  #define TCPU_HAL_SCALAR 0
#endif

// `apple` and `neon` are different tunings of one instruction set, so code that
// needs only "this is ARM NEON" - the int8 sdot kernels, whose instructions both
// parts have - keys on this rather than on TCPU_HAL_NEON, which would exclude
// Apple silicon from a kernel it can run.
#define TCPU_ISA_ARM (TCPU_HAL_NEON || TCPU_HAL_APPLE)

namespace tcpu {
namespace hal {
inline const char* backend_name() { return TCPU_HAL_X86 && env::zen() ? "amd-zen" : TCPU_HAL_NAME; }
} // namespace hal
} // namespace tcpu
