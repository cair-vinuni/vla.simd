# Intel Core i5-12400F benchmark

With the fastest configuration per model, FP32 median latency ranges from 61.9 ms (Octo-Small) to
972.2 ms (SmolVLA). INT8 (W8A8) is faster for 5 of 5 models, up to 2.81× for Diffusion Policy.

## Test setup

| Item | Value |
| --- | --- |
| Device | Desktop, Intel Core i5-12400F, 16 GB |
| CPU | Intel Core i5-12400F, Alder Lake, 6 P-cores, 12 threads, up to 4.4 GHz |
| SIMD used | AVX2 + FMA; AVX-VNNI for INT8 |
| Memory | 16 GB |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic |
| Compiler | c++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0; `-mavx2 -mfma -mf16c`, AVX-VNNI per function behind a CPUID check |
| Python, NumPy | 3.12.13, 2.5.3 |
| Power | `intel_pstate`, `powersave` governor, EPP `performance` |
| Commit | `7636baabd17191ad705260f46c1aace5f816b249` (`fix-inference-and-ci`) |
| Date | 2026-09-30 11:04 to 11:38 UTC+7 |

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
- VRAM: not applicable. vla.simd runs on the CPU only. The machine's NVIDIA RTX 3060 was idle.
- Tuning: for every model and precision, the thread count is swept over 12, 8, 6, 4, 2 and 1,
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
| ACT | FP32 | 12 | none | defaults |
| ACT | INT8 | 12 | 63 (all groups) | defaults |
| IMPACT | FP32 | 12 | none | defaults |
| IMPACT | INT8 | 12 | 63 (all groups) | defaults |
| SmolVLA | FP32 | 12 | none | defaults |
| SmolVLA | INT8 | 12 | 63 (all groups) | `TCPU_OMP_MIN=2048` |
| Octo-Small | FP32 | 12 | none | `OMP_PLACES=cores`, `OMP_PROC_BIND=close` |
| Octo-Small | INT8 | 6 | 63 (all groups) | defaults |
| TurboVLA | FP32 | 12 | none | defaults |
| Diffusion Policy | FP32 | 6 | none | defaults |
| Diffusion Policy | INT8 | 12 | 3 (all groups) | defaults |

## Results

| Model | Precision | Median (ms) | p10 (ms) | p90 (ms) | p95 (ms) | Actions/s | Peak RSS (MiB) | Model RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | FP32 | 139.4 | 137.9 | 144.0 | 146.0 | 358.7 | 590 | 561 |
| ACT | INT8 | 59.2 | 58.9 | 61.0 | 61.3 | 844.0 | 483 | 453 |
| IMPACT | FP32 | 140.3 | 139.6 | 144.7 | 146.4 | 356.3 | 766 | 736 |
| IMPACT | INT8 | 60.5 | 60.2 | 62.6 | 63.3 | 826.9 | 765 | 735 |
| SmolVLA | FP32 | 972.2 | 965.8 | 984.2 | 986.2 | 51.4 | 2,250 | 2,220 |
| SmolVLA | INT8 | 462.6 | 455.9 | 487.6 | 525.5 | 108.1 | 2,250 | 2,220 |
| Octo-Small | FP32 | 61.9 | 61.5 | 64.6 | 69.5 | 64.6 | 1,078 | 1,048 |
| Octo-Small | INT8 | 35.0 | 34.9 | 35.4 | 35.6 | 114.3 | 1,078 | 1,048 |
| TurboVLA | FP32 | 155.4 | 155.0 | 162.3 | 165.2 | 77.2 | 1,771 | 1,741 |
| Diffusion Policy | FP32 | 499.6 | 498.5 | 502.6 | 503.7 | 64.0 | 2,166 | 2,136 |
| Diffusion Policy | INT8 | 178.1 | 177.2 | 185.0 | 186.5 | 179.7 | 2,152 | 2,122 |

TurboVLA has no INT8 path.

## Thread sweep

Median latency in ms with default settings, from the search runs. The fastest entry in each row is
in bold.

| Model | Precision | 1 | 2 | 4 | 6 | 8 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | FP32 | 726.0 | 383.4 | 205.8 | 142.0 | 174.9 | **139.3** |
| ACT | INT8 | 306.3 | 167.4 | 92.2 | 66.3 | 74.5 | **59.2** |
| IMPACT | FP32 | 724.5 | 387.1 | 206.6 | 144.1 | 176.2 | **139.6** |
| IMPACT | INT8 | 309.4 | 169.6 | 94.5 | 66.1 | 74.9 | **59.9** |
| SmolVLA | FP32 | 5,252 | 2,660 | 1,413 | 1,005 | 1,392 | **974.6** |
| SmolVLA | INT8 | 2,434 | 1,269 | 683.5 | 505.5 | 653.0 | **470.9** |
| Octo-Small | FP32 | 322.4 | 168.7 | 91.0 | 63.8 | 86.8 | **63.1** |
| Octo-Small | INT8 | 167.2 | 88.8 | 49.7 | **34.8** | 43.2 | 35.1 |
| TurboVLA | FP32 | 849.0 | 432.2 | 225.4 | 161.6 | 222.0 | **156.3** |
| Diffusion Policy | FP32 | 1,828 | 1,011 | 629.2 | **499.8** | 570.8 | 550.7 |
| Diffusion Policy | INT8 | 689.1 | 380.0 | 232.9 | 185.8 | 196.5 | **178.5** |

## Register-tile selection

Uses the existing x86 AVX2 6×16 tile (Algorithm 1 selects it for any AVX2 core: 15 of 16 registers,
matching BLIS on Haswell and Zen); no search needed.

## Notes

- This machine also coordinated the other devices over SSH during the run, a light load.
- Latency for SmolVLA and TurboVLA includes encoding the instruction on every query. IMPACT and
  Octo-Small cache the encoded instruction, and the benchmark repeats one instruction, so their
  timed queries skip the text encoder.
- The INT8 row of IMPACT uses `impact-int8-so101-multi-task.gguf`, the checkpoint trained for W8A8;
  the other INT8 rows quantize the FP32 checkpoint at load. INT8 changes the numerics; no accuracy
  is measured here.
- The Diffusion Policy checkpoint was converted from a local SO-101 training run with
  `tools/convert_diffusion.py --scheduler DDIM --steps 10`; it is not published on the Hub.
