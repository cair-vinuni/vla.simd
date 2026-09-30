# Raspberry Pi 5 benchmark

With the fastest configuration per model, FP32 median latency ranges from 669.8 ms (Octo-Small) to
10,424.1 ms (SmolVLA). INT8 (W8A8) is faster for 5 of 5 models, up to 3.96× for Diffusion Policy.

## Test setup

| Item | Value |
| --- | --- |
| Device | Raspberry Pi 5 Model B Rev 1.1, 16 GB |
| CPU | Broadcom BCM2712, 4 × Arm Cortex-A76, 4 threads, 2.4 GHz |
| SIMD used | NEON; dotprod (`sdot`) for INT8 |
| Memory | 16 GB LPDDR4X |
| OS | Debian GNU/Linux 13 (trixie), kernel 6.18.34+rpt-rpi-2712 |
| Compiler | c++ (Debian 14.2.0-19) 14.2.0; `-march=armv8-a+simd`, dotprod per function behind a HWCAP check |
| Python, NumPy | 3.12.14, 2.5.3 |
| Power | Mains power, `ondemand` governor, active cooler with fan |
| Commit | `7636baabd17191ad705260f46c1aace5f816b249` (`fix-inference-and-ci`) |
| Date | 2026-09-30 11:05 to 13:29 UTC+7 |

## Method

- Harness: `vla-simd-serve --model <model> --model-dir <gguf> --bench <N> --json`, one fresh process
  per configuration, driven by `tools/bench_sweep.py`.
- Warmup: 5 untimed predictions after loading, then 50 or 20 timed predictions for the reported
  configuration (20 when a query takes more than 3 s). Search runs use 10 timed predictions.
- Inputs: new random `uint8` frames for every query, fixed state and instruction, seed 0.
- Latency: wall time of one complete `predict` call, covering image preprocessing, every encoder,
  all solver steps, and action unnormalization. Network and gRPC time are excluded.
- Memory: peak resident set size (RSS) of the benchmark process. The model's share is peak RSS minus
  the RSS of the same interpreter after importing the server but before loading a checkpoint (31 MiB
  on this device).
- VRAM: not applicable. vla.simd runs on the CPU only.
- Tuning: for every model and precision, the thread count is swept over 4, 3, 2 and 1, highest first
  so that no run inherits turbo power budget from a lighter one before it. At the fastest count,
  each runtime setting that leaves results unchanged is tried in turn and kept only when it is at
  least 2% faster twice, the second time against a fresh baseline. INT8 rows also try dropping one
  layer group at a time. The results table reports the final measurement of the fastest
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
| GEMM register tile (mr × nr) | 4×16 (NEON packed micro-kernel, `VLA_NEON_MR=4`) |
| INT8 (W8A8) path | dotprod (`sdot`) |
| Build | `cmake --preset release && cmake --build build` (Release, `-O3 -funroll-loops -ffp-contract=fast`) |

Fastest settings per model:

| Model | Precision | Threads | INT8 mask | Other settings |
| --- | --- | --- | --- | --- |
| ACT | FP32 | 4 | none | defaults |
| ACT | INT8 | 4 | 63 (all groups) | defaults |
| IMPACT | FP32 | 4 | none | defaults |
| IMPACT | INT8 | 4 | 63 (all groups) | defaults |
| SmolVLA | FP32 | 4 | none | defaults |
| SmolVLA | INT8 | 4 | 63 (all groups) | defaults |
| Octo-Small | FP32 | 4 | none | defaults |
| Octo-Small | INT8 | 4 | 63 (all groups) | `TCPU_ATTN_QB=8` |
| TurboVLA | FP32 | 4 | none | `OMP_PLACES=cores`, `OMP_PROC_BIND=close` |
| Diffusion Policy | FP32 | 3 | none | defaults |
| Diffusion Policy | INT8 | 4 | 3 (all groups) | defaults |

## Results

| Model | Precision | Median (ms) | p10 (ms) | p90 (ms) | p95 (ms) | Actions/s | Peak RSS (MiB) | Model RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | FP32 | 1,251 | 1,230 | 1,272 | 1,278 | 40.0 | 594 | 563 |
| ACT | INT8 | 487.9 | 468.6 | 498.2 | 503.2 | 102.5 | 485 | 454 |
| IMPACT | FP32 | 1,316 | 1,281 | 1,350 | 1,356 | 38.0 | 768 | 736 |
| IMPACT | INT8 | 497.4 | 477.4 | 503.5 | 508.8 | 100.5 | 767 | 736 |
| SmolVLA | FP32 | 10,424 | 10,384 | 10,458 | 10,466 | 4.8 | 1,959 | 1,928 |
| SmolVLA | INT8 | 3,961 | 3,933 | 3,995 | 3,999 | 12.6 | 1,959 | 1,928 |
| Octo-Small | FP32 | 669.8 | 660.1 | 696.5 | 697.8 | 6.0 | 1,080 | 1,049 |
| Octo-Small | INT8 | 354.3 | 339.1 | 361.7 | 362.4 | 11.3 | 1,080 | 1,049 |
| TurboVLA | FP32 | 1,590 | 1,548 | 1,626 | 1,650 | 7.5 | 1,773 | 1,742 |
| Diffusion Policy | FP32 | 4,265 | 4,213 | 4,304 | 4,309 | 7.5 | 2,168 | 2,137 |
| Diffusion Policy | INT8 | 1,078 | 1,062 | 1,091 | 1,106 | 29.7 | 2,154 | 2,123 |

TurboVLA has no INT8 path.

## Thread sweep

Median latency in ms with default settings, from the search runs. The fastest entry in each row is
in bold.

| Model | Precision | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| ACT | FP32 | 3,241 | 1,681 | 1,239 | **985.0** |
| ACT | INT8 | 1,177 | 683.8 | 516.0 | **482.8** |
| IMPACT | FP32 | 3,270 | 1,944 | 1,551 | **1,285** |
| IMPACT | INT8 | 1,213 | 702.1 | 576.5 | **498.2** |
| SmolVLA | FP32 | 27,725 | 15,049 | 12,209 | **10,294** |
| SmolVLA | INT8 | 9,826 | 5,775 | 4,559 | **3,946** |
| Octo-Small | FP32 | 1,621 | 917.6 | 728.5 | **655.9** |
| Octo-Small | INT8 | 813.8 | 463.0 | 377.8 | **350.4** |
| TurboVLA | FP32 | 3,880 | 2,355 | 1,870 | **1,584** |
| Diffusion Policy | FP32 | 7,118 | 4,832 | **4,293** | 4,411 |
| Diffusion Policy | INT8 | 2,285 | 1,373 | 1,163 | **1,082** |

## Register-tile selection

Uses the NEON 4×16 tile that Algorithm 1 and the paper's measurements selected for the Cortex-A76;
no search needed.

## Notes

- The board ran at its 85 °C soft temperature limit for most of the sweep, with the active cooler's
  fan running: median 85.1 °C (peak 86.2 °C), the firmware capped the clock in 49% of the 30 s
  samples, and the median sampled clock was 2.15 GHz (1.5 to 2.4 GHz). Better cooling would lower
  these latencies.
- Search runs used 10 timed queries instead of 20 to bound the total time on this board.
- Latency for SmolVLA and TurboVLA includes encoding the instruction on every query. IMPACT and
  Octo-Small cache the encoded instruction, and the benchmark repeats one instruction, so their
  timed queries skip the text encoder.
- The INT8 row of IMPACT uses `impact-int8-so101-multi-task.gguf`, the checkpoint trained for W8A8;
  the other INT8 rows quantize the FP32 checkpoint at load. INT8 changes the numerics; no accuracy
  is measured here.
- The Diffusion Policy checkpoint was converted from a local SO-101 training run with
  `tools/convert_diffusion.py --scheduler DDIM --steps 10`; it is not published on the Hub.
