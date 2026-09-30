# Apple M4 benchmark

With the fastest configuration per model, FP32 median latency ranges from 47.9 ms (Octo-Small) to
602.4 ms (SmolVLA). INT8 (W8A8) is faster for 4 of 5 models, up to 2.40× for Diffusion Policy.

## Test setup

| Item | Value |
| --- | --- |
| Device | Mac mini (Mac16,10), 24 GB |
| CPU | Apple M4, 4 performance + 6 efficiency cores, 10 threads |
| SIMD used | NEON, dotprod (`sdot`) for INT8; AMX through Accelerate |
| Memory | 24 GB unified LPDDR5X |
| OS | macOS 26.5.1 (25F80) |
| Compiler | Apple clang 21.0.0 with Homebrew libomp 22.1.8; `-mcpu=native` |
| Python, NumPy | 3.12.13, 2.5.3 |
| Power | AC power, Low Power Mode off |
| Commit | `7636baabd17191ad705260f46c1aace5f816b249` (`fix-inference-and-ci`) |
| Date | 2026-09-30 11:04 to 11:36 UTC+7 |

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
- Tuning: for every model and precision, the thread count is swept over 10, 8, 6, 4, 2 and 1,
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
| Backend | `apple` |
| GEMM register tile (mr × nr) | Accelerate `sgemm` (AMX) for GEMM and MLP layers; 6×16 NEON packed micro-kernel for the rest |
| INT8 (W8A8) path | dotprod (`sdot`) |
| Build | `cmake --preset release && cmake --build build` (Release, `-O3 -funroll-loops -ffp-contract=fast`), Homebrew `libomp` |

Fastest settings per model:

| Model | Precision | Threads | INT8 mask | Other settings |
| --- | --- | --- | --- | --- |
| ACT | FP32 | 4 | none | `TCPU_VIEW_THREADS=2` |
| ACT | INT8 | 10 | 62 (all but encoder attention) | `TCPU_I8_MBLOCK=0`, `TCPU_VIEW_THREADS=5` |
| IMPACT | FP32 | 4 | none | `TCPU_VIEW_THREADS=2` |
| IMPACT | INT8 | 10 | 62 (all but encoder attention) | `TCPU_I8_MBLOCK=0`, `TCPU_VIEW_THREADS=5` |
| SmolVLA | FP32 | 4 | none | `TCPU_VIEW_THREADS=2` |
| SmolVLA | INT8 | 10 | 62 (all but ViT attention) | `TCPU_EXPERT_THREADS=4`, `TCPU_VIEW_THREADS=5` |
| Octo-Small | FP32 | 4 | none | defaults |
| Octo-Small | INT8 | 10 | 62 (all but transformer attention) | `TCPU_I8_MBLOCK=0` |
| TurboVLA | FP32 | 10 | none | defaults |
| Diffusion Policy | FP32 | 4 | none | defaults |
| Diffusion Policy | INT8 | 10 | 3 (all groups) | `TCPU_I8_MBLOCK=0`, `TCPU_OMP_MIN=32768` |

## Results

| Model | Precision | Median (ms) | p10 (ms) | p90 (ms) | p95 (ms) | Actions/s | Peak RSS (MiB) | Model RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | FP32 | 89.4 | 87.5 | 92.7 | 93.4 | 559.2 | 680 | 650 |
| ACT | INT8 | 69.1 | 68.5 | 69.8 | 69.9 | 723.6 | 661 | 632 |
| IMPACT | FP32 | 92.2 | 90.1 | 95.3 | 95.8 | 542.4 | 823 | 793 |
| IMPACT | INT8 | 70.5 | 70.0 | 71.0 | 71.2 | 709.2 | 801 | 771 |
| SmolVLA | FP32 | 602.4 | 598.5 | 608.9 | 612.4 | 83.0 | 3,085 | 3,055 |
| SmolVLA | INT8 | 494.4 | 493.1 | 495.5 | 496.1 | 101.1 | 3,027 | 2,997 |
| Octo-Small | FP32 | 47.9 | 47.8 | 48.0 | 48.0 | 83.6 | 1,103 | 1,073 |
| Octo-Small | INT8 | 50.5 | 50.3 | 50.8 | 50.9 | 79.2 | 1,119 | 1,089 |
| TurboVLA | FP32 | 114.5 | 114.2 | 114.7 | 114.8 | 104.8 | 2,606 | 2,576 |
| Diffusion Policy | FP32 | 433.1 | 432.4 | 434.2 | 434.4 | 73.9 | 2,169 | 2,140 |
| Diffusion Policy | INT8 | 180.2 | 179.8 | 180.7 | 180.8 | 177.6 | 2,380 | 2,350 |

TurboVLA has no INT8 path.

## Thread sweep

Median latency in ms with default settings, from the search runs. The fastest entry in each row is
in bold.

| Model | Precision | 1 | 2 | 4 | 6 | 8 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | FP32 | 122.8 | 102.1 | **94.5** | 98.0 | 98.5 | 100.6 |
| ACT | INT8 | 378.8 | 201.9 | 114.2 | 105.0 | 96.7 | **93.6** |
| IMPACT | FP32 | 128.1 | 105.8 | **96.6** | 99.5 | 99.8 | 101.4 |
| IMPACT | INT8 | 384.5 | 205.1 | 116.5 | 106.9 | 97.8 | **94.7** |
| SmolVLA | FP32 | 977.3 | 774.0 | **664.1** | 681.3 | 675.7 | 678.5 |
| SmolVLA | INT8 | 4,130 | 2,160 | 1,226 | 1,065 | 954.7 | **914.6** |
| Octo-Small | FP32 | 68.0 | 54.9 | **47.9** | 53.0 | 52.2 | 54.5 |
| Octo-Small | INT8 | 263.6 | 142.8 | 84.5 | 82.1 | 78.3 | **78.2** |
| TurboVLA | FP32 | 251.1 | 164.4 | 120.3 | 118.9 | 114.9 | **114.7** |
| Diffusion Policy | FP32 | 550.5 | 472.1 | **433.4** | 443.0 | 441.7 | 450.7 |
| Diffusion Policy | INT8 | 666.3 | 365.2 | 222.4 | 214.6 | 199.6 | **196.6** |

## Register-tile selection

Uses the existing Apple backend: Accelerate routes the large layers to AMX and the 6×16 NEON kernel
covers the rest; no search needed.

## Notes

- The desktop's animated aerial wallpaper kept a video decoder at about 10% of one core during the
  run.
- Latency for SmolVLA and TurboVLA includes encoding the instruction on every query. IMPACT and
  Octo-Small cache the encoded instruction, and the benchmark repeats one instruction, so their
  timed queries skip the text encoder.
- The INT8 row of IMPACT uses `impact-int8-so101-multi-task.gguf`, the checkpoint trained for W8A8;
  the other INT8 rows quantize the FP32 checkpoint at load. INT8 changes the numerics; no accuracy
  is measured here.
- The Diffusion Policy checkpoint was converted from a local SO-101 training run with
  `tools/convert_diffusion.py --scheduler DDIM --steps 10`; it is not published on the Hub.
