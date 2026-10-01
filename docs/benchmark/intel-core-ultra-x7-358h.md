# Intel Core Ultra X7 358H benchmark

With the fastest configuration per model, FP32 median latency ranges from 68.3 ms (Octo-Small) to
976.9 ms (SmolVLA). INT8 (W8A8) is faster for 5 of 5 models, up to 2.11× for Diffusion Policy.

## Test setup

| Item | Value |
| --- | --- |
| Device | AAEON CEXD-INTRBL, Intel Core Ultra X7 358H, 64 GB |
| CPU | Intel Core Ultra X7 358H, Panther Lake, 4 P-cores + 8 E-cores + 4 low-power E-cores, 16 threads, up to 4.8 GHz |
| SIMD used | AVX2 + FMA; AVX-VNNI for INT8 |
| Memory | 64 GB |
| OS | Ubuntu 24.04.4 LTS, kernel 7.0.0-31-generic |
| Compiler | c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0; `-mavx2 -mfma -mf16c`, AVX-VNNI per function behind a CPUID check |
| Python, NumPy | 3.12.3, 2.5.3 |
| Power | `performance` power profile and ACPI platform profile, `intel_pstate`, `powersave` governor, EPP `performance` |
| Commit | `c60b286d53a1fe31752d5c98237e0a314d655729` (`main`) |
| Date | 2026-10-01 09:40 to 12:35 UTC+7 |

## Method

- Harness: `vla-simd-serve --model <model> --model-dir <gguf> --bench <N> --json`, one fresh process
  per configuration, driven by `tools/bench_sweep.py`.
- Warmup: 5 untimed predictions after loading, then 50 timed predictions for the reported
  configuration (20 when a query takes more than 3 s). Search runs use 20 timed predictions.
- Inputs: new random `uint8` frames for every query, fixed state and instruction, seed 0.
- Latency: wall time of one complete `predict` call, covering image preprocessing, every encoder,
  all solver steps, and action unnormalization. Network and gRPC time are excluded.
- Memory: peak resident set size (RSS) of the benchmark process. The model's share is peak RSS minus
  the RSS of the same interpreter after importing the server but before loading a checkpoint (32 MiB
  on this device).
- VRAM: not applicable. vla.simd runs on the CPU only. The integrated GPU and NPU were not used.
- Tuning: for every model and precision, the thread count is swept over 16, 12, 8, 4, 2 and 1,
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
| ACT | FP32 | 16 | none | defaults |
| ACT | INT8 | 16 | 63 (all groups) | `TCPU_I8_MBLOCK=0`, `TCPU_VIEW_THREADS=8` |
| IMPACT | FP32 | 16 | none | defaults |
| IMPACT | INT8 | 16 | 63 (all groups) | defaults |
| SmolVLA | FP32 | 16 | none | defaults |
| SmolVLA | INT8 | 16 | 63 (all groups) | `TCPU_VIEW_THREADS=8` |
| Octo-Small | FP32 | 16 | none | defaults |
| Octo-Small | INT8 | 16 | 63 (all groups) | `TCPU_CONV_MINP=32`, `TCPU_I8_MBLOCK=0` |
| TurboVLA | FP32 | 16 | none | defaults |
| Diffusion Policy | FP32 | 16 | none | defaults |
| Diffusion Policy | INT8 | 16 | 3 (all groups) | defaults |

## Results

| Model | Precision | Median (ms) | p10 (ms) | p90 (ms) | p95 (ms) | Actions/s | Peak RSS (MiB) | Model RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | FP32 | 135.4 | 132.8 | 139.1 | 139.6 | 369.3 | 595 | 563 |
| ACT | INT8 | 65.3 | 60.8 | 67.1 | 68.4 | 765.1 | 607 | 575 |
| IMPACT | FP32 | 136.2 | 131.8 | 140.1 | 142.0 | 367.2 | 769 | 737 |
| IMPACT | INT8 | 72.5 | 63.8 | 75.1 | 76.5 | 689.9 | 768 | 737 |
| SmolVLA | FP32 | 976.9 | 961.2 | 991.7 | 1,005 | 51.2 | 2,253 | 2,221 |
| SmolVLA | INT8 | 566.4 | 556.0 | 578.5 | 579.4 | 88.3 | 2,253 | 2,221 |
| Octo-Small | FP32 | 68.3 | 63.4 | 70.4 | 70.8 | 58.6 | 1,082 | 1,050 |
| Octo-Small | INT8 | 42.4 | 37.9 | 45.1 | 45.7 | 94.4 | 1,081 | 1,050 |
| TurboVLA | FP32 | 162.2 | 155.0 | 166.6 | 167.7 | 74.0 | 1,775 | 1,743 |
| Diffusion Policy | FP32 | 378.9 | 373.3 | 381.8 | 387.8 | 84.4 | 2,169 | 2,138 |
| Diffusion Policy | INT8 | 179.9 | 153.7 | 184.3 | 188.7 | 177.8 | 2,156 | 2,124 |

TurboVLA has no INT8 path.

## Thread sweep

Median latency in ms with default settings, from the search runs. The fastest entry in each row is
in bold.

| Model | Precision | 1 | 2 | 4 | 8 | 12 | 16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | FP32 | 669.5 | 349.9 | 191.1 | 153.5 | 117.0 | **101.5** |
| ACT | INT8 | 268.8 | 143.5 | 86.9 | 81.9 | 74.3 | **69.2** |
| IMPACT | FP32 | 670.7 | 351.3 | 210.2 | 183.1 | 155.1 | **133.8** |
| IMPACT | INT8 | 271.9 | 145.2 | 86.7 | 81.7 | 74.2 | **69.7** |
| SmolVLA | FP32 | 4,889 | 2,539 | 1,496 | 1,370 | 1,092 | **979.3** |
| SmolVLA | INT8 | 2,227 | 1,192 | 707.1 | 731.4 | 621.7 | **584.7** |
| Octo-Small | FP32 | 299.2 | 153.9 | 84.8 | 79.5 | 59.5 | **54.1** |
| Octo-Small | INT8 | 146.6 | 83.5 | 49.9 | 50.1 | 52.9 | **49.6** |
| TurboVLA | FP32 | 790.8 | 415.4 | 240.3 | 227.6 | 181.6 | **117.9** |
| Diffusion Policy | FP32 | 1,634 | 889.6 | 549.3 | 478.1 | 413.5 | **377.8** |
| Diffusion Policy | INT8 | 620.6 | 348.2 | 210.4 | 190.7 | 169.0 | **155.8** |

## Register-tile selection

Panther Lake is a new microarchitecture for this engine, so Algorithm 1 of the paper was applied to
it. Both core types run the existing AVX2 backend: Rmax = 16 vector registers, L = 8 fp32 lanes, and
Pfma·τ = 8 for two 256-bit FMA issues per cycle at four cycles. AVX2 broadcasts the activation, so q
= 1 and Λ = nv + mr. Over the engine's 16-wide packing granularity (g = 16 and its divisors and
multiples up to 32):

| nr | nv | Largest mr within (2) and (3) | Registers | Score φ |
| --- | --- | --- | --- | --- |
| 8 | 1 | 14 | 16 of 16 | 0.93 |
| 16 | 2 | 6 | 15 of 16 | 1.50 |
| 32 | 4 | 2 | 13 of 16 | 1.33 |

Algorithm 1 selects 6×16, the tile the x86 backend already uses (the same choice as for Haswell, Zen
and Raptor Lake), so no new kernel or build option is needed. `tools/bench_tile` measured that tile
on each core type, pinned with `taskset` (GFLOP/s, best of seven timed blocks):

| Layer (M × N × K) | P-core ×1 | P-core ×4 | E-core ×1 | E-core ×8 | LP E-core ×1 | all cores ×16 |
| --- | --- | --- | --- | --- | --- | --- |
| ACT / IMPACT encoder FFN (602 × 3200 × 512) | 136.7 | 521.1 | 73.6 | 579.7 | 68.2 | 1005.5 |
| SmolVLA SigLIP MLP (1024 × 3072 × 768) | 135.1 | 513.1 | 75.0 | 580.9 | 66.9 | 1006.7 |
| SmolVLA SmolLM2 MLP (241 × 2560 × 960) | 135.6 | 523.1 | 74.8 | 589.6 | 65.8 | 1043.9 |
| SmolVLA expert MLP (50 × 2048 × 720) | 134.5 | 501.0 | 73.3 | 564.8 | 66.2 | 887.4 |
| Octo-Small MLP (340 × 1536 × 384) | 138.9 | 523.2 | 73.3 | 572.6 | 64.8 | 1004.1 |
| Diffusion UNet conv (im2col) (32 × 1024 × 5120) | 119.8 | 480.7 | 71.5 | 541.3 | 63.0 | 974.8 |

A P-core's peak at 4.8 GHz with two 256-bit FMA issues per cycle is 153.6 GFLOP/s.

## Notes

- The checkpoints reached this device over a slow network link, so the sweep ran in two parts with
  the same build and settings: ACT, IMPACT and TurboVLA at 09:40 to 09:48, and Octo-Small, SmolVLA
  and Diffusion Policy at 12:00 to 12:26, once their checkpoints had arrived and matched their Hub
  checksums. No download ran during either part.
- Before the run, Firefox in another account's desktop session was stopped and the power profile was
  switched from balanced to performance; the remaining desktop session left the CPU 99.3% idle.
- The package ran at a median 77 °C (peak 100 °C), with the average core clock at a median 2.2 GHz
  across all 16 cores (sampled every 30 s).
- This mobile processor spends a short-term turbo power budget: after idle or lighter runs, the
  first 20 to 30 s of all-core load ran up to 30% faster (ACT FP32 at 16 threads: 99 to 101 ms, then
  125 to 140 ms). Thread-sweep points that follow lighter runs fall partly in that window, and
  inside it the sweep accepted settings that do not hold. Every setting the sweep picked was
  therefore checked again at sustained power, interleaved after a 60 s heat soak. `TCPU_CONV_BUDGET`
  for ACT and IMPACT (defaults 1 to 2% faster) and `TCPU_HEAD_THREADS=8` for Octo-Small FP32 (under
  2%, 67.0 against 68.2 ms) did not hold, so those four rows list default settings and were measured
  again with them, 5 warmup and 50 timed queries after a heat soak. The INT8 settings of ACT (65.2
  against 70.5 ms), Octo-Small (39.4 against 50.8 ms) and SmolVLA (564.8 against 592.5 ms) held. 16
  threads also beat 12 for ACT FP32 (134 against 154 ms) and TurboVLA (163 against 182 ms), and 8
  and 4 for Octo-Small INT8 (48.2 and 46.3 ms). Every row in the results table was measured at
  sustained power.
- Latency for SmolVLA and TurboVLA includes encoding the instruction on every query. IMPACT and
  Octo-Small cache the encoded instruction, and the benchmark repeats one instruction, so their
  timed queries skip the text encoder.
- The INT8 row of IMPACT uses `impact-int8-so101-multi-task.gguf`, the checkpoint trained for W8A8;
  the other INT8 rows quantize the FP32 checkpoint at load. INT8 changes the numerics; no accuracy
  is measured here.
- The Diffusion Policy checkpoint was converted from a local SO-101 training run with
  `tools/convert_diffusion.py --scheduler DDIM --steps 10` and is published as
  [khanhnd61/diffusion-so101-tape-gguf](https://huggingface.co/khanhnd61/diffusion-so101-tape-gguf)
  (revision `a1fa4bb`).
