/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// All TCPU_* runtime knobs, parsed once. Defaults are per-backend and reproduce
// each hardware branch (see docs/07-hal-design.md for the matrix). Every knob is
// an A/B hook: flipping it must never change which model runs, only how.

namespace tcpu {
namespace hal {
namespace env {

// TCPU_PACKED=0 falls back to the unpacked dense_linear (all backends).
bool packed();

// TCPU_ACCEL=0 disables Accelerate/AMX GEMM routing (Apple; false elsewhere).
bool accel();

// TCPU_ATTN_BLAS=0 disables the per-head sgemm attention path (Apple).
bool attn_blas();

// TCPU_ATTN_DYNAMIC=N: dynamic chunk for attention query rows, 0 = static
// (Apple NEON path; default 8, measured best on M4 4P+6E).
int attn_dynamic();

// TCPU_ATTN_DENSE=0 sends the unmasked (ViT) attention back through the masked
// kernel with a zero mask, i.e. exactly what it did before the dense path
// existed. Values are identical either way - the dense path was written to keep
// the FMA order - so this is purely the A/B hook for the traffic win.
bool attn_dense();

// TCPU_ATTN_QB=N: query rows per block in the dense (unmasked) NEON attention.
// A block makes one pass over the head's K^T and V panels, so this is the panel
// re-streaming factor: seq_q/N passes instead of seq_q/4. Default 16.
int attn_qblock();

// TCPU_ATTN=skip|tiled forces one masked-attention path on the Pi NEON backend
// (0 = auto mask-density dispatch, 1 = skip, 2 = tiled).
int attn_force();

// TCPU_ATTN_QTILE=0 reverts the x86 masked attention to per-query QK
// (no 6-row K-panel sharing).
bool attn_qtile();

// TCPU_CONV_TILE=0 reverts conv2d_packed to the whole-image im2col (one big
// buffer through DRAM) instead of cache-resident output-pixel panels. Values are
// identical either way - only the buffer's residency changes.
bool conv_tile();

// TCPU_CONV_BUDGET=N: floats of im2col a conv panel may hold, which sets the panel
// thickness (P = N/K). The tiled conv runs its GEMM serially inside one thread, so
// this is that thread's working set: too large and the expansion round-trips through
// DRAM instead of staying resident, too small and the panel stops amortizing the
// weight matrix it streams. Default 32768 (128 KB) on the Pi NEON backend and 131072
// (512 KB) elsewhere: the old global 512 KB is exactly a Cortex-A76's per-core L2,
// the wrong side of the boundary once four threads share a 2 MB L3. Values are
// identical either way and only residency changes -- verified as the same SHA-256
// over the action bytes at every budget tried. See env.cpp for the measurements.
int conv_budget();

// TCPU_CONV_MINP=N: panels thinner than N rows fall back to the whole-image path.
// Tiling parallelizes over output-pixel panels and runs the GEMM serially inside
// each, so every panel re-streams the whole weight matrix; the untiled path
// instead threads inside the GEMM, where each worker owns a fixed set of weight
// panels. A late ResNet stage (few pixels, K in the thousands) lands on thin
// panels and pays that re-streaming many times over, so it wants the untiled
// path; early stages, where im2col dwarfs the weights, want tiles. Values are
// identical either way. Default 16 (measured on i5-12400F, see docs/10-act-design.md).
int conv_min_panel();

// TCPU_FUSE_GELU=0 splits the MLP back into GEMM + separate gelu pass
// (x86 packed path only; other backends always run split).
bool fuse_gelu();

// TCPU_FUSE_RES=0 restores the separate residual-add pass after the wo/w2
// GEMMs (x86 packed path only; other backends always run the separate pass).
bool fuse_res();

// TCPU_HEAD_THREADS: OMP team clamp around the diffusion-head denoise loop.
// Default 4 on Apple (tiny-op fork/join tax grows with team size), 0 = off
// elsewhere (branch behavior).
int head_threads();

// TCPU_EXPERT_THREADS: OMP team clamp around SmolVLA's flow-matching denoise
// loop. Same tiny-op fork/join tax as head_threads(), one model up: the loop is
// 10 steps x 32 layers of seq-50 ops, so the region count dominates past ~8
// threads. 0 = off (inherit the global team).
int expert_threads();

// TCPU_VIEW_THREADS=N: encode the camera views concurrently, N OMP threads per
// view, instead of one view after another on the whole team. The views are
// independent, and a single SigLIP pass scales badly (measured 1.77x for 8x the
// threads), so two efficient small teams beat one inefficient big one. Threading
// only - each view's math is untouched. 0 = off (sequential).
int view_threads();

// TCPU_SIMD_SILU=0 reverts silu_gate to the scalar libm loop. Default on: the
// SIMD path uses the same Cephes exp as gelu_tanh (poly-exp tolerance class,
// measured max rel 2.7e-07), one exp per vector instead of one per element.
bool simd_silu();

// TCPU_SIMD_ERF=0 reverts gelu_erf to the scalar std::erf loop. Default on: the
// SIMD path evaluates erf as Abramowitz-Stegun 7.1.26 over the backend's Cephes
// exp, 1.5e-07 absolute on erf, so under 1e-06 absolute on the gelu output.
bool simd_erf();

// TCPU_BF16_MLP=1: run MLP-role GEMMs from bf16 packed weights (Pi NEON backend;
// memory-bound MLP, 5th-decimal accuracy class). Default off.
bool bf16_mlp();

// TCPU_BF16_DEQ=0: x86 keeps bf16-checkpoint weights resident and runs the
// native bf16 row GEMM instead of dequantizing to packed fp32 panels at load
// (Linear::init_bf16). Default on: the packed panels measured faster on i9,
// at 2x the weight RAM. Other backends have a fixed policy (see linear.cpp).
bool bf16_deq();

// TCPU_GEMM_SCHED=static reverts the Pi packed GEMMs to static scheduling
// (dynamic whole-panel is the measured ARM default).
bool gemm_force_static();

// TCPU_GEMM_THREADS / TCPU_GEMM_CHUNK: hybrid-core recruitment experiment hooks
// for the packed GEMMs (see lm_ops history). 0 / unset = off.
int gemm_threads();
int gemm_chunk();   // 0 = unset (x86 hook uses 4; Pi uses one whole panel)

// Token rows per tile in the int8 dotprod GEMM (TCPU_I8_MR). 4 measured best on
// a Cortex-A76; the kernel is templated for 1..6.
int i8_mr();

// Token rows cache-blocked per pass over the weight panels in the int8 GEMM
// (TCPU_I8_MBLOCK); 0 disables the blocking.
int i8_mblock();

// TCPU_GEMM_MBLOCK: token rows the fp32 packed GEMM holds resident while every
// weight panel sweeps them. Same trade as i8_mblock: without it the (panel, tile)
// loop re-streams the whole activation matrix once per 16-column panel, ~600 MB
// per GEMM for a 1024x3072 ViT MLP. Blocking the token axis leaves the K loop
// alone, so the result is bitwise identical. Default 0, see env.cpp.
int gemm_mblock();

// TCPU_OMP_MIN: elements below which the elementwise ops and the token-wise
// norms stay on the calling thread. Forking a full team costs more than the work
// itself for the flow-matching expert (chunk 50) and the Octo diffusion head
// (n = 64).
int omp_min();

// TCPU_I8_CLIP: fraction of absmax the int8 conv quantizer scales to. A ReLU map
// is long-tailed, so plain absmax spends the range on a few outliers; clipping
// saturates those and gives the bulk more levels. Clamped to (0, 1].
float i8_clip();

} // namespace env
} // namespace hal
} // namespace tcpu
