/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// HAL backend selection. Exactly one TCPU_HAL_* is 1; everything below hal/ keys
// off these, nothing above hal/ may. The two NEON tunings are distinct backends
// on purpose: the M4 (wide OoO core + AMX via Accelerate) and the Cortex-A72
// (narrow, load-limited) want different kernels, and their kernels are not
// numerically interchangeable (accumulator-chain counts differ). Selection is
// compile-time and matches where each tuning was measured.
//
// The same split now exists inside x86: AMD Zen and Intel Raptor Lake run the
// same AVX2 ISA but want different attention kernels (Zen 3's 32 KB / 8-way L1
// and 2 load ports punish the layouts that are free on a P-core). TCPU_HAL_AMD
// is selected by CMake when the build host is AuthenticAMD (-DVLA_HAL=amd|avx2
// overrides); TCPU_ISA_X86 is the family macro for code that only needs "this
// is x86 with AVX2" - the GEMM kernels, the fused-epilogue capabilities, the
// SIMD primitives - all of which both x86 backends share verbatim.
#if defined(__AVX2__)
  #if defined(TCPU_HAL_PREFER_AMD)
    #define TCPU_HAL_AMD 1
    #define TCPU_HAL_NAME "amd-zen"
  #else
    #define TCPU_HAL_X86 1
    #define TCPU_HAL_NAME "x86-avx2"
  #endif
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
#ifndef TCPU_HAL_AMD
  #define TCPU_HAL_AMD 0
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

// x86-AVX2 family: 1 for both the Intel and the AMD backend, 0 everywhere else
// (so on ARM/scalar this is exactly the constant TCPU_HAL_X86 used to be).
#define TCPU_ISA_X86 (TCPU_HAL_X86 || TCPU_HAL_AMD)
// ...and the same for AArch64: `apple` and `neon` are different tunings of one
// instruction set, so code that needs only "this is ARM NEON" - the int8 sdot
// kernels, whose instructions both parts have - keys on this rather than on
// TCPU_HAL_NEON, which would exclude Apple silicon from a kernel it can run.
#define TCPU_ISA_ARM (TCPU_HAL_NEON || TCPU_HAL_APPLE)

namespace tcpu {
namespace hal {
inline const char* backend_name() { return TCPU_HAL_NAME; }
} // namespace hal
} // namespace tcpu
