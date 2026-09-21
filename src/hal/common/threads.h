/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace tcpu {
namespace hal {

// Scoped OMP team clamp for tiny-op regions (seq=1 GEMM loops etc.): past ~4
// threads the fork/join overhead dominates and grows with team size (measured
// 2.9 ms @4T vs 9 ms @10T for the 20-step diffusion head on M4). max_threads
// <= 0 or >= the current team is a no-op. Threading only - never changes math.
struct ScopedTeamClamp {
#if defined(_OPENMP)
    int saved = 0;
    explicit ScopedTeamClamp(int max_threads) {
        const int cur = omp_get_max_threads();
        if (max_threads > 0 && max_threads < cur) {
            saved = cur;
            omp_set_num_threads(max_threads);
        }
    }
    ~ScopedTeamClamp() {
        if (saved > 0) omp_set_num_threads(saved);
    }
#else
    explicit ScopedTeamClamp(int) {}
#endif
    ScopedTeamClamp(const ScopedTeamClamp&) = delete;
    ScopedTeamClamp& operator=(const ScopedTeamClamp&) = delete;
};

} // namespace hal
} // namespace tcpu
