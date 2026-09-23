/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <exception>
#include <thread>
#include <vector>

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

template<class F> void for_each_view(int n, int threads, F&& fn) {
    if (threads <= 0 || n < 2) {
        for (int i = 0; i < n; i++) fn(i);
        return;
    }
    std::vector<std::exception_ptr> err(n);
    std::vector<std::thread> workers;
    workers.reserve(n);
    try {
        for (int i = 0; i < n; i++) {
            workers.emplace_back([&, i] {
#if defined(_OPENMP)
                omp_set_num_threads(threads);
#endif
                try { fn(i); } catch (...) { err[i] = std::current_exception(); }
            });
        }
    } catch (...) {
        for (std::thread& w : workers) w.join();
        throw;
    }
    for (std::thread& w : workers) w.join();
    for (std::exception_ptr& e : err) if (e) std::rethrow_exception(e);
}

} // namespace hal
} // namespace tcpu
