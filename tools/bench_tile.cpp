/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// GFLOP/s of the packed fp32 GEMM (dense_linear_packed) on the policies' layer
// shapes, at one thread and at the full OMP team: the measurement step of the
// register-tile selection. Build once per NEON candidate and compare:
//
//     cmake -B build-mr5 -DVLA_NEON_MR=5 && cmake --build build-mr5 --target bench_tile
//     OMP_NUM_THREADS=8 build-mr5/bench_tile

#if defined(_WIN32)
#define _WIN32_WINNT 0x0A00   // SetProcessInformation
#include <windows.h>
#endif
#include "hal/arch.h"
#include "ops/lm_ops.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

struct Shape { const char* name; int m, n, k; };

// tokens x outputs x inputs
const Shape kShapes[] = {
    {"act_ffn",     602,  3200, 512},   // ACT / IMPACT encoder FFN, 600 visual tokens
    {"siglip_mlp",  1024, 3072, 768},   // SmolVLA SigLIP MLP, 512 px / 16
    {"smollm2_mlp", 241,  2560, 960},   // SmolVLA SmolLM2 MLP, image + text + state tokens
    {"expert_mlp",  50,   2048, 720},   // SmolVLA flow-matching expert, chunk 50
    {"octo_mlp",    340,  1536, 384},   // Octo-Small transformer MLP
    {"unet_conv",   32,   1024, 5120},  // Diffusion Policy UNet conv1d as im2col, kernel 5
};

double gflops(const Shape& s, int threads) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float> w((size_t)s.n * s.k), wp(w.size()), x((size_t)s.m * s.k), b(s.n),
        y((size_t)s.m * s.n);
    for (float& v : w) v = u(rng);
    for (float& v : x) v = u(rng);
    for (float& v : b) v = u(rng);
    tcpu::pack_weights16(w.data(), wp.data(), s.n, s.k);
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#else
    (void)threads;
#endif
    auto run = [&] { tcpu::dense_linear_packed(y.data(), x.data(), wp.data(), b.data(), s.m, s.n, s.k); };
    for (int i = 0; i < 3; i++) run();
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    run();
    const double once = std::chrono::duration<double>(clock::now() - t0).count();
    const int reps = std::max(1, (int)(0.05 / std::max(once, 1e-6)));   // ~50 ms blocks
    double best = 1e30;
    for (int block = 0; block < 7; block++) {
        t0 = clock::now();
        for (int r = 0; r < reps; r++) run();
        best = std::min(best, std::chrono::duration<double>(clock::now() - t0).count() / reps);
    }
    return 2.0 * s.m * s.n * s.k / best * 1e-9;
}

} // namespace

int main() {
#if defined(_WIN32)
    // Started from a shell without a foreground window (SSH), Windows would
    // throttle it as background work; see policy_server._disable_power_throttling.
    PROCESS_POWER_THROTTLING_STATE state{PROCESS_POWER_THROTTLING_CURRENT_VERSION,
                                         PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0};
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state, sizeof state);
#endif
#if defined(_OPENMP)
    const int team = omp_get_max_threads();
#else
    const int team = 1;
#endif
#if TCPU_HAL_NEON
    const int mr = TCPU_NEON_MR;
#else
    const int mr = 0;   // fixed tile on this backend
#endif
    std::printf("backend,mr,shape,m,n,k,threads,gflops\n");
    for (const Shape& s : kShapes)
        for (int t : {1, team})
            std::printf("%s,%d,%s,%d,%d,%d,%d,%.1f\n", tcpu::hal::backend_name(), mr, s.name, s.m, s.n,
                        s.k, t, gflops(s, t));
    return 0;
}
