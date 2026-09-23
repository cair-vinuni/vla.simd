/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "env.h"
#include "../arch.h"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstddef>
using std::size_t;

namespace tcpu {
namespace hal {
namespace env {

int int_env(const char* name, int dflt) {
    const char* e = std::getenv(name);
    if (!e || !*e) return dflt;
    char* end;
    const long long v = std::strtoll(e, &end, 10);
    return *end || v < INT_MIN || v > INT_MAX ? dflt : (int)v;
}
static bool flag_on(const char* name, bool dflt) {
    const char* e = std::getenv(name);
    if (!e || !e[0]) return dflt;
    return e[0] != '0';
}

bool packed() {
    static const bool v = flag_on("TCPU_PACKED", true);
    return v;
}

bool accel() {
#if defined(TCPU_ACCELERATE)
    static const bool v = flag_on("TCPU_ACCEL", true);
    return v;
#else
    return false;
#endif
}

bool attn_blas() {
    static const bool v = flag_on("TCPU_ATTN_BLAS", true);
    return v;
}

int attn_dynamic() {
    static const int v = int_env("TCPU_ATTN_DYNAMIC", 8);   // measured best on M4; 0 = static
    return v;
}

bool attn_dense() {
    static const bool v = flag_on("TCPU_ATTN_DENSE", true);
    return v;
}

int attn_qblock() {
    static const int v = [] {
        const int n = int_env("TCPU_ATTN_QB", 16);   // measured best on Cortex-A76
        // Rounded UP to a multiple of 4, not just clamped: the dense QK kernel
        // works in 4-query micro-tiles and writes all four rows of the last tile,
        // so an odd block (TCPU_ATTN_QB=5) wrote past the score buffer.
        return n < 4 ? 4 : n > 1024 ? 1024 : (n + 3) & ~3;
    }();
    return v;
}

int attn_force() {
    static const int v = [] {
        const char* e = std::getenv("TCPU_ATTN");
        return !e ? 0 : e[0] == 's' ? 1 : e[0] == 't' ? 2 : 0;
    }();
    return v;
}

bool attn_qtile() {
    static const bool v = flag_on("TCPU_ATTN_QTILE", true);
    return v;
}

bool conv_tile() {
    static const bool v = flag_on("TCPU_CONV_TILE", true);
    return v;
}

int conv_budget() {
    // 32 K floats (128 KB) on the Pi, 128 K (512 KB) elsewhere. The old global 512 KB
    // was measured on an i5-12400F and is exactly a Cortex-A76's per-core L2, so four
    // Pi threads each holding one needed 2.1 MB of a 2 MB shared L3 and the im2col
    // expansion round-tripped through DRAM. Measured on ACT, paired ABBA, 5/5 pairs
    // each: 128 KB -12.9%, 256 KB -10.9%, 1 MB +9.9%, and below 128 KB it turns back
    // up (64 KB +5.5%, 32 KB +21.4%) as the panel stops amortizing its weight matrix.
    // Bit-identical at every value -- same SHA-256 over the action bytes.
#if TCPU_HAL_NEON
    static const int v = [] {
        const int n = int_env("TCPU_CONV_BUDGET", 32*1024);
        return n < 4096 ? 4096 : n;
    }();
#else
    static const int v = [] {
        const int n = int_env("TCPU_CONV_BUDGET", 128*1024);
        return n < 4096 ? 4096 : n;
    }();
#endif
    return v;
}

int conv_min_panel() {
    static const int v = int_env("TCPU_CONV_MINP", 16);
    return v;
}

bool fuse_gelu() {
    static const bool v = flag_on("TCPU_FUSE_GELU", true);
    return v;
}

bool fuse_res() {
    static const bool v = flag_on("TCPU_FUSE_RES", true);
    return v;
}

int head_threads() {
#if TCPU_HAL_APPLE
    static const int v = int_env("TCPU_HEAD_THREADS", 4);
#else
    static const int v = int_env("TCPU_HEAD_THREADS", 0);
#endif
    return v;
}

int expert_threads() {
    static const int v = int_env("TCPU_EXPERT_THREADS", 0);
    return v;
}

int view_threads() {
    static const int v = int_env("TCPU_VIEW_THREADS", 0);
    return v;
}

bool simd_silu() {
    static const bool v = flag_on("TCPU_SIMD_SILU", true);
    return v;
}

bool simd_erf() {
    static const bool v = flag_on("TCPU_SIMD_ERF", true);
    return v;
}

bool simd_mish() {
    static const bool v = flag_on("TCPU_SIMD_MISH", true);
    return v;
}

bool bf16_mlp() {
    static const bool v = [] {
        const char* e = std::getenv("TCPU_BF16_MLP");
        return e && e[0] == '1';
    }();
    return v;
}

bool bf16_deq() {
#if TCPU_HAL_NEON
    static const bool v = flag_on("TCPU_BF16_DEQ", false);   // bf16 panels are the native Pi format
#else
    static const bool v = flag_on("TCPU_BF16_DEQ", true);
#endif
    return v;
}

bool gemm_force_static() {
    static const bool v = [] {
        const char* e = std::getenv("TCPU_GEMM_SCHED");
        return e && e[0] == 's';
    }();
    return v;
}

int gemm_threads() {
    static const int v = int_env("TCPU_GEMM_THREADS", 0);
    return v;
}

int gemm_chunk() {
    static const int v = int_env("TCPU_GEMM_CHUNK", 0);   // 0 = unset
    return v;
}

int omp_min() {
    static const int v = std::max(0, int_env("TCPU_OMP_MIN", 8192));
    return v;
}

int i8_mblock() {
    static const int v = int_env("TCPU_I8_MBLOCK", 128);   // token rows held resident
    return v;
}

int gemm_mblock() {
    // Off by default: measured neutral (752 vs 754 ms, min of 5, SmolVLA on a
    // Core Ultra 9 285K), where a 36 MB L3 already holds the activation matrix
    // and the re-streaming it removes was hitting cache anyway. Kept as a hook
    // for small-cache targets, where the int8 kernel's identical blocking is
    // worth up to 1.8x; it needs a measurement on one of those before it moves.
    static const int v = int_env("TCPU_GEMM_MBLOCK", 0);
    return v;
}

int i8_mr() {
    static const int v = [] {
        const int m = int_env("TCPU_I8_MR", 4);
        return m < 1 ? 1 : m > 6 ? 6 : m;   // the kernel is templated for 1..6
    }();
    return v;
}

float i8_clip() {
    static const float v = [] {
        const char* e = std::getenv("TCPU_I8_CLIP");
        const float f = e ? (float)std::atof(e) : 1.0f;
        return f > 0.0f && f <= 1.0f ? f : 1.0f;
    }();
    return v;
}

} // namespace env
} // namespace hal
} // namespace tcpu
