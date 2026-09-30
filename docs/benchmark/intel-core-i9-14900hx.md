# Intel Core i9-14900HX benchmark

With the fastest configuration per model, FP32 median latency ranges from 49.9 ms (Octo-Small) to
780.4 ms (SmolVLA). INT8 (W8A8) is faster for 5 of 5 models, up to 2.67× for Diffusion Policy.

## Test setup

| Item | Value |
| --- | --- |
| Device | Laptop, Intel Core i9-14900HX, 32 GB |
| CPU | Intel Core i9-14900HX, Raptor Lake Refresh, 8 P-cores + 16 E-cores, 32 threads, up to 5.8 GHz |
| SIMD used | AVX2 + FMA; AVX-VNNI for INT8 |
| Memory | 32 GB |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-106-generic |
| Compiler | c++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0; `-mavx2 -mfma -mf16c`, AVX-VNNI per function behind a CPUID check |
| Python, NumPy | 3.12.13, 2.5.3 |
| Power | AC adapter, `intel_pstate`, `powersave` governor, EPP `performance` |
| Commit | `7636baabd17191ad705260f46c1aace5f816b249` (`fix-inference-and-ci`) |
| Date | 2026-09-30 11:04 to 11:34 UTC+7 |

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
- VRAM: not applicable. vla.simd runs on the CPU only. The laptop's NVIDIA RTX 5070 Laptop GPU was
  idle.
- Tuning: for every model and precision, the thread count is swept over 32, 24, 16, 12, 8, 4 and 1,
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
| ACT | FP32 | 32 | none | defaults |
| ACT | INT8 | 8 | 63 (all groups) | one thread per P-core (CPUs 0,2,4,6,8,10,12,14), `TCPU_VIEW_THREADS=4` |
| IMPACT | FP32 | 32 | none | `TCPU_CONV_MINP=8` |
| IMPACT | INT8 | 8 | 63 (all groups) | one thread per P-core (CPUs 0,2,4,6,8,10,12,14), `TCPU_I8_MBLOCK=0`, `TCPU_VIEW_THREADS=4` |
| SmolVLA | FP32 | 32 | none | `TCPU_VIEW_THREADS=16` |
| SmolVLA | INT8 | 8 | 63 (all groups) | one thread per P-core (CPUs 0,2,4,6,8,10,12,14) |
| Octo-Small | FP32 | 8 | none | one thread per P-core (CPUs 0,2,4,6,8,10,12,14) |
| Octo-Small | INT8 | 4 | 63 (all groups) | `TCPU_I8_MBLOCK=0`, `TCPU_I8_MR=6` |
| TurboVLA | FP32 | 32 | none | defaults |
| Diffusion Policy | FP32 | 8 | none | one thread per P-core (CPUs 0,2,4,6,8,10,12,14) |
| Diffusion Policy | INT8 | 8 | 3 (all groups) | one thread per P-core (CPUs 0,2,4,6,8,10,12,14), `TCPU_I8_MBLOCK=0`, `TCPU_I8_MR=6` |

## Results

| Model | Precision | Median (ms) | p10 (ms) | p90 (ms) | p95 (ms) | Actions/s | Peak RSS (MiB) | Model RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | FP32 | 99.6 | 98.3 | 103.0 | 103.5 | 502.0 | 597 | 567 |
| ACT | INT8 | 49.1 | 48.7 | 49.5 | 49.6 | 1,017.8 | 556 | 527 |
| IMPACT | FP32 | 97.3 | 96.5 | 101.1 | 102.1 | 513.9 | 765 | 736 |
| IMPACT | INT8 | 48.2 | 47.9 | 48.6 | 48.8 | 1,036.4 | 765 | 735 |
| SmolVLA | FP32 | 780.4 | 775.8 | 785.3 | 786.7 | 64.1 | 2,249 | 2,220 |
| SmolVLA | INT8 | 401.7 | 400.2 | 403.0 | 403.8 | 124.5 | 2,249 | 2,219 |
| Octo-Small | FP32 | 49.9 | 49.3 | 50.3 | 50.3 | 80.2 | 1,078 | 1,048 |
| Octo-Small | INT8 | 44.8 | 44.3 | 50.7 | 52.9 | 89.3 | 1,078 | 1,048 |
| TurboVLA | FP32 | 116.0 | 114.2 | 117.8 | 118.5 | 103.4 | 1,771 | 1,741 |
| Diffusion Policy | FP32 | 341.1 | 339.4 | 343.4 | 344.1 | 93.8 | 2,166 | 2,136 |
| Diffusion Policy | INT8 | 127.6 | 127.3 | 128.6 | 129.3 | 250.8 | 2,152 | 2,122 |

TurboVLA has no INT8 path.

## Thread sweep

Median latency in ms with default settings, from the search runs. The fastest entry in each row is
in bold.

| Model | Precision | 1 | 4 | 8 | 12 | 16 | 24 | 32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | FP32 | 588.7 | 178.6 | 116.8 | 170.5 | 147.7 | 123.2 | **98.2** |
| ACT | INT8 | 254.1 | 80.7 | **55.6** | 83.7 | 71.7 | 73.6 | 70.7 |
| IMPACT | FP32 | 618.2 | 192.9 | 120.7 | 168.2 | 149.3 | 127.4 | **99.9** |
| IMPACT | INT8 | 263.0 | 84.5 | **56.6** | 88.5 | 76.4 | 74.9 | 71.0 |
| SmolVLA | FP32 | 4,376 | 1,374 | 903.0 | 1,255 | 1,097 | 899.3 | **789.2** |
| SmolVLA | INT8 | 2,120 | 664.8 | **429.5** | 730.5 | 642.1 | 588.6 | 526.6 |
| Octo-Small | FP32 | 281.0 | 86.1 | 60.4 | 96.9 | 80.1 | 67.6 | **54.5** |
| Octo-Small | INT8 | 143.5 | **49.2** | 49.5 | 54.3 | 59.6 | 60.2 | 64.5 |
| TurboVLA | FP32 | 737.8 | 222.3 | 127.8 | 228.8 | 187.4 | 152.8 | **116.0** |
| Diffusion Policy | FP32 | 1,562 | 530.7 | 377.1 | 473.7 | 459.1 | 430.2 | **359.8** |
| Diffusion Policy | INT8 | 604.4 | 208.2 | **163.1** | 211.1 | 191.2 | 206.0 | 202.6 |

## Register-tile selection

Uses the existing x86 AVX2 6×16 tile (Algorithm 1 selects it for any AVX2 core: 15 of 16 registers,
matching BLIS on Haswell and Zen); no search needed.

## Notes

- The package ran at a median 92 °C (peak 97 °C) through the sweep, with the average core clock at a
  median 2.1 GHz (sampled every 30 s): under sustained all-core load this laptop is power- and
  temperature-limited, well below its 5.8 GHz single-core peak.
- Latency for SmolVLA and TurboVLA includes encoding the instruction on every query. IMPACT and
  Octo-Small cache the encoded instruction, and the benchmark repeats one instruction, so their
  timed queries skip the text encoder.
- The INT8 row of IMPACT uses `impact-int8-so101-multi-task.gguf`, the checkpoint trained for W8A8;
  the other INT8 rows quantize the FP32 checkpoint at load. INT8 changes the numerics; no accuracy
  is measured here.
- The Diffusion Policy checkpoint was converted from a local SO-101 training run with
  `tools/convert_diffusion.py --scheduler DDIM --steps 10`; it is not published on the Hub.
