# Snapdragon X benchmark

With the fastest configuration per model, FP32 median latency ranges from 97.9 ms (Octo-Small) to
1,435.2 ms (SmolVLA). INT8 (W8A8) is faster for 5 of 5 models, up to 2.92× for Diffusion Policy.

## Test setup

| Item | Value |
| --- | --- |
| Device | ASUS Vivobook 14 (X1407QA), 16 GB |
| CPU | Qualcomm Snapdragon X X1-26-100, 8 Oryon cores, 8 threads, 2.96 GHz |
| SIMD used | NEON; dotprod (`sdot`) for INT8 |
| Memory | 16 GB LPDDR5X-8448 |
| OS | Windows 11 Home 25H2 (build 26200) |
| Compiler | clang 23.1.2 (llvm-mingw 20260922, UCRT); `-march=armv8-a+simd`, dotprod per function behind `IsProcessorFeaturePresent` |
| Python, NumPy | 3.12.14, 2.5.3 |
| Power | AC power, Balanced scheme with the Best performance overlay |
| Commit | `7636baabd17191ad705260f46c1aace5f816b249` (`fix-inference-and-ci`) |
| Date | 2026-09-30 11:04 to 12:00 UTC+7 |

## Method

- Harness: `vla-simd-serve --model <model> --model-dir <gguf> --bench <N> --json`, one fresh process
  per configuration, driven by `tools/bench_sweep.py`.
- Warmup: 5 untimed predictions after loading, then 50 timed predictions for the reported
  configuration (20 when a query takes more than 3 s). Search runs use 20 timed predictions.
- Inputs: new random `uint8` frames for every query, fixed state and instruction, seed 0.
- Latency: wall time of one complete `predict` call, covering image preprocessing, every encoder,
  all solver steps, and action unnormalization. Network and gRPC time are excluded.
- Memory: peak resident set size (RSS) of the benchmark process. The model's share is peak RSS minus
  the RSS of the same interpreter after importing the server but before loading a checkpoint (30 MiB
  on this device).
- VRAM: not applicable. vla.simd runs on the CPU only.
- Tuning: for every model and precision, the thread count is swept over 8, 6, 4, 2 and 1, highest
  first so that no run inherits turbo power budget from a lighter one before it. At the fastest
  count, each runtime setting that leaves results unchanged is tried in turn and kept only when it
  is at least 2% faster twice, the second time against a fresh baseline. INT8 rows also try dropping
  one layer group at a time. The results table reports the final measurement of the fastest
  configuration.

## Model configurations

| Model | Checkpoint | Weights | Cameras and frame size | Network input | Action chunk | State | Language | Action decoding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | `act-so101-multi-task.gguf` | 236 MB, FP32 (59.0M values) | 2 × 480×640 | 480×640 | 50 × 6 | 6 | none | single pass |
| IMPACT | `impact-so101-multi-task.gguf` | 384 MB, FP32 (95.6M values) | 2 × 480×640 | 480×640 | 50 × 6 | 6 | T5-small, cached per instruction | single pass |
| SmolVLA | `smolvla-so101-multi-task.gguf` | 1008 MB, 75% BF16 / 25% FP32 (402.7M values) | 2 × any size | 512×512, padded | 50 × 6 | 6 | SmolLM2, every query | flow matching, 10 steps |
| Octo-Small | `octo-small-so101-multi-task.gguf` | 548 MB, FP32 (136.7M values) | primary 256×256, wrist 128×128, window 2 | 256×256 / 128×128 | 4 × 6 | none | T5-base, cached per instruction | diffusion head, 20 steps |
| TurboVLA | `turbovla-libero-f32.gguf` | 862 MB, FP32 (215.5M values) | 2 × 256×256 | 256×256 | 12 × 7 | 8 | BERT-base, every query | action head, single pass |
| Diffusion Policy | `diffusion-so101-tape.gguf` | 1111 MB, FP32 (277.8M values) | 2 × 480×640 × 2 obs steps | 480×640 | 32 × 6 (horizon 64) | 6 | none | DDIM, 10 steps |

## Engine configuration

| Item | Value |
| --- | --- |
| Backend | `neon` |
| GEMM register tile (mr × nr) | 4×16 (NEON packed micro-kernel, `VLA_NEON_MR=4`, selected below) |
| INT8 (W8A8) path | dotprod (`sdot`) |
| Build | llvm-mingw, CMake and Ninja: `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++` |

Fastest settings per model:

| Model | Precision | Threads | INT8 mask | Other settings |
| --- | --- | --- | --- | --- |
| ACT | FP32 | 8 | none | defaults |
| ACT | INT8 | 8 | 63 (all groups) | `TCPU_I8_MBLOCK=0` |
| IMPACT | FP32 | 8 | none | defaults |
| IMPACT | INT8 | 8 | 63 (all groups) | `TCPU_CONV_BUDGET=131072` |
| SmolVLA | FP32 | 8 | none | defaults |
| SmolVLA | INT8 | 8 | 63 (all groups) | defaults |
| Octo-Small | FP32 | 6 | none | defaults |
| Octo-Small | INT8 | 8 | 63 (all groups) | `TCPU_GEMM_SCHED=static`, `TCPU_I8_MBLOCK=0` |
| TurboVLA | FP32 | 8 | none | `OMP_PLACES=cores`, `OMP_PROC_BIND=close` |
| Diffusion Policy | FP32 | 8 | none | defaults |
| Diffusion Policy | INT8 | 8 | 3 (all groups) | `TCPU_I8_MBLOCK=0` |

## Results

| Model | Precision | Median (ms) | p10 (ms) | p90 (ms) | p95 (ms) | Actions/s | Peak RSS (MiB) | Model RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | FP32 | 170.6 | 168.4 | 177.6 | 184.7 | 293.1 | 594 | 565 |
| ACT | INT8 | 74.7 | 74.2 | 75.9 | 79.2 | 669.3 | 483 | 453 |
| IMPACT | FP32 | 191.2 | 188.7 | 195.6 | 198.7 | 261.6 | 765 | 735 |
| IMPACT | INT8 | 77.8 | 77.0 | 79.1 | 81.3 | 642.6 | 765 | 735 |
| SmolVLA | FP32 | 1,435 | 1,427 | 1,462 | 1,487 | 34.8 | 1,957 | 1,927 |
| SmolVLA | INT8 | 660.4 | 653.1 | 679.9 | 684.5 | 75.7 | 1,957 | 1,927 |
| Octo-Small | FP32 | 97.9 | 97.3 | 98.5 | 98.8 | 40.9 | 1,077 | 1,048 |
| Octo-Small | INT8 | 44.4 | 43.5 | 45.9 | 46.6 | 90.2 | 1,078 | 1,048 |
| TurboVLA | FP32 | 217.0 | 214.4 | 234.1 | 238.0 | 55.3 | 1,770 | 1,740 |
| Diffusion Policy | FP32 | 495.3 | 493.2 | 501.3 | 506.5 | 64.6 | 2,159 | 2,129 |
| Diffusion Policy | INT8 | 169.6 | 168.6 | 172.7 | 176.9 | 188.7 | 2,152 | 2,122 |

TurboVLA has no INT8 path.

## Thread sweep

Median latency in ms with default settings, from the search runs. The fastest entry in each row is
in bold.

| Model | Precision | 1 | 2 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- |
| ACT | FP32 | 1,054 | 535.1 | 277.7 | 200.3 | **154.9** |
| ACT | INT8 | 487.3 | 279.2 | 148.5 | 105.3 | **84.6** |
| IMPACT | FP32 | 1,185 | 606.4 | 313.3 | 224.0 | **170.5** |
| IMPACT | INT8 | 496.9 | 285.9 | 152.3 | 108.2 | **86.9** |
| SmolVLA | FP32 | 8,275 | 4,227 | 2,443 | 1,839 | **1,432** |
| SmolVLA | INT8 | 3,510 | 2,049 | 1,085 | 827.8 | **656.8** |
| Octo-Small | FP32 | 537.0 | 271.4 | 143.8 | **97.7** | 111.0 |
| Octo-Small | INT8 | 284.9 | 146.1 | 80.0 | 58.2 | **47.4** |
| TurboVLA | FP32 | 1,331 | 759.5 | 410.8 | 270.6 | **223.5** |
| Diffusion Policy | FP32 | 2,274 | 1,413 | 844.1 | 614.7 | **506.4** |
| Diffusion Policy | INT8 | 1,008 | 530.8 | 323.4 | 251.6 | **204.1** |

## Register-tile selection

Snapdragon X is the one CPU here without a tuned backend, so the NEON tile was selected with
Algorithm 1 of the paper. Inputs: L = 4 fp32 lanes per register, Rmax = 32 vector registers, Pfma =
4 and τ = 4 cycles (Oryon's four 128-bit FMA pipes), at least two vector loads per cycle, and g =
16, the width of the engine's packed weight panels. Constraint (3) needs mr·nv ≥ 16 independent
accumulators.

| nr | nv | Largest mr within (2) and (3) | Registers | Score φ | Spill check |
| --- | --- | --- | --- | --- | --- |
| 4 | 1 | none: (3) needs mr ≥ 16, (2) allows 15 | n/a | n/a | n/a |
| 8 | 2 | 10 | 32 of 32 | 1.67 | no 8-wide kernel |
| 16 | 4 | 5 | 29 of 32 | 2.22 | none |
| 32 | 8 | 2 | 26 of 32 | 1.60 | no 32-wide kernel |

The packed layout is 16 wide, so only nr = 16 candidates have a kernel to compile; nr = 8 and nr =
32 also score lower than 5×16. Algorithm 1 therefore proposes 5×16. The spill check compiles
`src/hal/neon/gemm.cpp` with each tile and counts vector reloads from the stack:

| Tile | Registers | Stack reloads | Stack stores |
| --- | --- | --- | --- |
| 3×16 | 19 | 0 | 0 |
| 4×16 | 24 | 0 | 0 |
| 5×16 | 29 | 0 | 0 |
| 6×16 | 34 | 11 | 12 |

`tools/bench_tile` then measured each candidate on the six layer shapes, best of seven timed blocks,
at 1 and 8 threads (GFLOP/s, higher is better, fastest in bold):

| 1 thread: layer (M × N × K) | 3×16 | 4×16 | 5×16 | 6×16 |
| --- | --- | --- | --- | --- |
| ACT / IMPACT encoder FFN (602 × 3200 × 512) | 69.2 | **92.5** | 81.7 | 45.1 |
| SmolVLA SigLIP MLP (1024 × 3072 × 768) | 69.3 | **93.1** | 82.4 | 44.6 |
| SmolVLA SmolLM2 MLP (241 × 2560 × 960) | 69.1 | **92.3** | 81.3 | 45.0 |
| SmolVLA expert MLP (50 × 2048 × 720) | 68.0 | **89.4** | 82.2 | 45.0 |
| Octo-Small MLP (340 × 1536 × 384) | 69.5 | **93.2** | 82.6 | 45.4 |
| Diffusion UNet conv (im2col) (32 × 1024 × 5120) | 67.0 | **92.2** | 77.6 | 44.8 |

| 8 threads: layer (M × N × K) | 3×16 | 4×16 | 5×16 | 6×16 |
| --- | --- | --- | --- | --- |
| ACT / IMPACT encoder FFN (602 × 3200 × 512) | 546.0 | **732.3** | 646.6 | 356.3 |
| SmolVLA SigLIP MLP (1024 × 3072 × 768) | 546.1 | **732.1** | 649.3 | 354.5 |
| SmolVLA SmolLM2 MLP (241 × 2560 × 960) | 544.5 | **729.2** | 644.0 | 354.7 |
| SmolVLA expert MLP (50 × 2048 × 720) | 531.8 | **695.5** | 639.9 | 354.0 |
| Octo-Small MLP (340 × 1536 × 384) | 547.2 | **735.3** | 648.3 | 358.3 |
| Diffusion UNet conv (im2col) (32 × 1024 × 5120) | 512.0 | **684.7** | 589.5 | 350.3 |

End to end, both builds served every model at 8 threads with default settings, in alternating order
(median of the run medians, ms):

| Model | 4×16 | 5×16 | 5×16 / 4×16 |
| --- | --- | --- | --- |
| ACT | 163.0 | 163.6 | 1.004 |
| IMPACT | 164.8 | 165.8 | 1.006 |
| SmolVLA | 1,372.7 | 1,427.2 | 1.040 |
| Octo-Small | 89.8 | 89.0 | 0.992 |
| TurboVLA | 245.9 | 268.2 | 1.091 |
| Diffusion Policy | 496.1 | 507.6 | 1.023 |

4×16 is the fastest candidate on all six shapes at both thread counts, and end to end 5×16 ranges
from 0.8% faster to 9.1% slower. Algorithm 1's proposal fails the paper's acceptance test (the
selected tile must be the fastest measured candidate), so the Snapdragon X build keeps 4×16
(`VLA_NEON_MR=4`, the default). The Cortex-A76 went the same way: the score favored other tiles and
measurement chose 4×16. `-DVLA_NEON_MR` keeps the tile a build option for other Arm cores.

## Notes

- The server opts out of Windows power throttling (EcoQoS). Started over SSH, the process otherwise
  counts as background work and was held to half the cores: ACT took 275 ms instead of 147 ms at 8
  threads.
- Python is the native ARM64 CPython 3.12; an x64 Python under emulation cannot load the ARM64
  engine.
- Sustained load settles about 11% slower than the first minute on this laptop (ACT at 8 threads:
  148 ms cold, 164 ms after several minutes). The sweep runs back to back, so the reported numbers
  are the sustained ones.
- Latency for SmolVLA and TurboVLA includes encoding the instruction on every query. IMPACT and
  Octo-Small cache the encoded instruction, and the benchmark repeats one instruction, so their
  timed queries skip the text encoder.
- The INT8 row of IMPACT uses `impact-int8-so101-multi-task.gguf`, the checkpoint trained for W8A8;
  the other INT8 rows quantize the FP32 checkpoint at load. INT8 changes the numerics; no accuracy
  is measured here.
- The Diffusion Policy checkpoint was converted from a local SO-101 training run with
  `tools/convert_diffusion.py --scheduler DDIM --steps 10`; it is not published on the Hub.
