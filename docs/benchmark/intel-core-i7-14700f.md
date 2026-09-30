# Intel Core i7-14700F benchmark

With the fastest configuration per model, FP32 median latency ranges from 52.7 ms (Octo-Small) to
861.1 ms (SmolVLA). INT8 (W8A8) is faster for 5 of 5 models, up to 2.83× for Diffusion Policy.

## Test setup

| Item | Value |
| --- | --- |
| Device | Desktop, Intel Core i7-14700F, 64 GB |
| CPU | Intel Core i7-14700F, Raptor Lake Refresh, 8 P-cores + 12 E-cores, 28 threads, up to 5.4 GHz |
| SIMD used | AVX2 + FMA; AVX-VNNI for INT8 |
| Memory | 64 GB |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic |
| Compiler | c++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0; `-mavx2 -mfma -mf16c`, AVX-VNNI per function behind a CPUID check |
| Python, NumPy | 3.12.13, 2.5.3 |
| Power | `intel_pstate`, `powersave` governor, EPP `performance` |
| Commit | `7636baabd17191ad705260f46c1aace5f816b249` (`fix-inference-and-ci`) |
| Date | 2026-09-30 11:05 to 11:44 UTC+7 |

## Method

- Harness: `vla-simd-serve --model <model> --model-dir <gguf> --bench <N> --json`, one fresh process
  per configuration, driven by `tools/bench_sweep.py`.
- Warmup: 5 untimed predictions after loading, then 50 timed predictions for the reported
  configuration (20 when a query takes more than 3 s). Search runs use 20 timed predictions.
- Inputs: new random `uint8` frames for every query, fixed state and instruction, seed 0.
- Latency: wall time of one complete `predict` call, covering image preprocessing, every encoder,
  all solver steps, and action unnormalization. Network and gRPC time are excluded.
- Memory: peak resident set size (RSS) of the benchmark process. The model's share is peak RSS minus
  the RSS of the same interpreter after importing the server but before loading a checkpoint (29 MiB
  on this device).
- VRAM: not applicable. vla.simd runs on the CPU only. The machine's two NVIDIA RTX 3090 GPUs were
  idle.
- Tuning: for every model and precision, the thread count is swept over 28, 20, 16, 12, 8, 4 and 1,
  highest first so that no run inherits turbo power budget from a lighter one before it. At the
  fastest count, each runtime setting that leaves results unchanged is tried in turn and kept only
  when it is at least 2% faster twice, the second time against a fresh baseline. INT8 rows also try
  dropping one layer group at a time. The results table reports the final measurement of the fastest
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
| Backend | `x86-avx2` |
| GEMM register tile (mr × nr) | 6×16 (AVX2 packed micro-kernel, fixed for the x86 backends) |
| INT8 (W8A8) path | AVX-VNNI (`vpdpbusd`) |
| Build | `cmake --preset release && cmake --build build` (Release, `-O3 -funroll-loops -ffp-contract=fast`) |

Fastest settings per model:

| Model | Precision | Threads | INT8 mask | Other settings |
| --- | --- | --- | --- | --- |
| ACT | FP32 | 8 | none | defaults |
| ACT | INT8 | 8 | 63 (all groups) | defaults |
| IMPACT | FP32 | 28 | none | `TCPU_CONV_MINP=32` |
| IMPACT | INT8 | 8 | 63 (all groups) | `TCPU_I8_MR=6` |
| SmolVLA | FP32 | 8 | none | defaults |
| SmolVLA | INT8 | 8 | 63 (all groups) | `TCPU_I8_MR=6`, `TCPU_OMP_MIN=2048` |
| Octo-Small | FP32 | 8 | none | defaults |
| Octo-Small | INT8 | 8 | 63 (all groups) | one thread per P-core (CPUs 0,2,4,6,8,10,12,14) |
| TurboVLA | FP32 | 28 | none | defaults |
| Diffusion Policy | FP32 | 8 | none | defaults |
| Diffusion Policy | INT8 | 8 | 3 (all groups) | one thread per P-core (CPUs 0,2,4,6,8,10,12,14) |

## Results

| Model | Precision | Median (ms) | p10 (ms) | p90 (ms) | p95 (ms) | Actions/s | Peak RSS (MiB) | Model RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | FP32 | 119.4 | 119.2 | 119.6 | 119.9 | 418.8 | 588 | 559 |
| ACT | INT8 | 53.3 | 52.9 | 53.5 | 53.7 | 938.8 | 482 | 453 |
| IMPACT | FP32 | 113.4 | 113.0 | 113.6 | 1,145 | 441.0 | 766 | 736 |
| IMPACT | INT8 | 52.6 | 51.7 | 52.9 | 53.0 | 951.0 | 764 | 735 |
| SmolVLA | FP32 | 861.1 | 859.2 | 986.5 | 1,104 | 58.1 | 2,249 | 2,220 |
| SmolVLA | INT8 | 402.6 | 400.3 | 476.0 | 589.8 | 124.2 | 2,249 | 2,220 |
| Octo-Small | FP32 | 52.7 | 52.4 | 53.4 | 53.7 | 75.9 | 1,077 | 1,048 |
| Octo-Small | INT8 | 29.8 | 29.6 | 31.3 | 31.3 | 134.1 | 1,078 | 1,048 |
| TurboVLA | FP32 | 132.7 | 132.0 | 133.3 | 134.0 | 90.5 | 1,771 | 1,741 |
| Diffusion Policy | FP32 | 469.4 | 468.7 | 470.0 | 470.1 | 68.2 | 2,166 | 2,137 |
| Diffusion Policy | INT8 | 165.7 | 165.5 | 165.9 | 166.0 | 193.1 | 2,152 | 2,123 |

TurboVLA has no INT8 path.

## Thread sweep

Median latency in ms with default settings, from the search runs. The fastest entry in each row is
in bold.

| Model | Precision | 1 | 4 | 8 | 12 | 16 | 20 | 28 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | FP32 | 572.0 | 181.3 | **119.6** | 157.1 | 138.2 | 137.4 | 1,461 |
| ACT | INT8 | 237.5 | 79.7 | **53.3** | 83.3 | 69.7 | 67.5 | 57.3 |
| IMPACT | FP32 | 572.0 | 180.2 | 119.8 | 155.3 | 137.9 | 138.7 | **112.4** |
| IMPACT | INT8 | 238.7 | 79.7 | **53.9** | 83.9 | 71.7 | 67.7 | 4,609 |
| SmolVLA | FP32 | 4,497 | 1,300 | **837.5** | 1,345 | 1,152 | 1,061 | 860.7 |
| SmolVLA | INT8 | 2,118 | 668.7 | **421.5** | 788.1 | 667.9 | 591.9 | 8,308 |
| Octo-Small | FP32 | 256.1 | 83.0 | **53.2** | 86.4 | 72.7 | 67.9 | 53.5 |
| Octo-Small | INT8 | 130.8 | 43.8 | **30.0** | 55.5 | 47.7 | 43.2 | 35.9 |
| TurboVLA | FP32 | 670.0 | 208.0 | 136.9 | 213.2 | 185.9 | 165.3 | **131.4** |
| Diffusion Policy | FP32 | 1,497 | 582.1 | **469.2** | 562.4 | 526.8 | 559.9 | 534.7 |
| Diffusion Policy | INT8 | 568.5 | 212.6 | **169.5** | 233.1 | 209.4 | 213.2 | 198.2 |

## Register-tile selection

Uses the existing x86 AVX2 6×16 tile (Algorithm 1 selects it for any AVX2 core: 15 of 16 registers,
matching BLIS on Haswell and Zen); no search needed.

## Notes

- With all 28 logical CPUs busy, some runs collapsed by one to two orders of magnitude (ACT FP32
  1,461 ms, IMPACT INT8 4,609 ms, SmolVLA INT8 8,308 ms at 28 threads): a single preempted OpenMP
  thread stalls every barrier. IMPACT and TurboVLA FP32 were still fastest at 28 threads in their
  final runs, but only 7% and 4% ahead of 8 threads; for serving, 8 threads avoids the collapse.
- Latency for SmolVLA and TurboVLA includes encoding the instruction on every query. IMPACT and
  Octo-Small cache the encoded instruction, and the benchmark repeats one instruction, so their
  timed queries skip the text encoder.
- The INT8 row of IMPACT uses `impact-int8-so101-multi-task.gguf`, the checkpoint trained for W8A8;
  the other INT8 rows quantize the FP32 checkpoint at load. INT8 changes the numerics; no accuracy
  is measured here.
- The Diffusion Policy checkpoint was converted from a local SO-101 training run with
  `tools/convert_diffusion.py --scheduler DDIM --steps 10`; it is not published on the Hub.
