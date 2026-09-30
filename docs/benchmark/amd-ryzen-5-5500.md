# AMD Ryzen 5 5500 benchmark

With the fastest configuration per model, FP32 median latency ranges from 73.0 ms (Octo-Small) to
1,161.9 ms (SmolVLA).

## Test setup

| Item | Value |
| --- | --- |
| Device | Desktop, AMD Ryzen 5 5500, 16 GB |
| CPU | AMD Ryzen 5 5500, Zen 3, 6 cores, 12 threads, up to 4.27 GHz |
| SIMD used | AVX2 + FMA; no AVX-VNNI |
| Memory | 16 GB DDR4 |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic |
| Compiler | c++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0; `-mavx2 -mfma -mf16c`, AVX-VNNI per function behind a CPUID check |
| Python, NumPy | 3.12.14, 2.5.3 |
| Power | `amd-pstate-epp`, `powersave` governor, EPP `performance` |
| Commit | `7636baabd17191ad705260f46c1aace5f816b249` (`fix-inference-and-ci`) |
| Date | 2026-09-30 11:04 to 12:04 UTC+7 |

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
| Backend | `amd-zen` |
| GEMM register tile (mr × nr) | 6×16 (AVX2 packed micro-kernel, fixed for the x86 backends) |
| INT8 (W8A8) path | not available (no AVX-VNNI) |
| Build | `cmake --preset release && cmake --build build` (Release, `-O3 -funroll-loops -ffp-contract=fast`) |

Fastest settings per model:

| Model | Precision | Threads | INT8 mask | Other settings |
| --- | --- | --- | --- | --- |
| ACT | FP32 | 6 | none | defaults |
| IMPACT | FP32 | 6 | none | `TCPU_CONV_BUDGET=32768` |
| SmolVLA | FP32 | 6 | none | defaults |
| Octo-Small | FP32 | 6 | none | defaults |
| TurboVLA | FP32 | 6 | none | defaults |
| Diffusion Policy | FP32 | 6 | none | defaults |

## Results

| Model | Precision | Median (ms) | p10 (ms) | p90 (ms) | p95 (ms) | Actions/s | Peak RSS (MiB) | Model RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | FP32 | 170.3 | 169.0 | 173.2 | 175.3 | 293.6 | 590 | 560 |
| IMPACT | FP32 | 170.8 | 169.9 | 171.9 | 172.1 | 292.8 | 766 | 736 |
| SmolVLA | FP32 | 1,162 | 1,153 | 1,168 | 1,169 | 43.0 | 2,250 | 2,220 |
| Octo-Small | FP32 | 73.0 | 72.2 | 73.6 | 73.8 | 54.8 | 1,078 | 1,048 |
| TurboVLA | FP32 | 187.6 | 186.6 | 188.7 | 189.5 | 64.0 | 1,771 | 1,741 |
| Diffusion Policy | FP32 | 738.0 | 735.5 | 741.8 | 743.1 | 43.4 | 2,166 | 2,137 |

This CPU has no AVX-VNNI, so the engine has no INT8 path here and INT8 rows are omitted.

## Thread sweep

Median latency in ms with default settings, from the search runs. The fastest entry in each row is
in bold.

| Model | Precision | 1 | 2 | 4 | 6 | 8 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | FP32 | 819.2 | 423.0 | 231.4 | **174.1** | 208.7 | 186.7 |
| IMPACT | FP32 | 819.4 | 427.6 | 230.7 | **177.1** | 214.7 | 189.9 |
| SmolVLA | FP32 | 5,712 | 2,880 | 1,532 | **1,168** | 1,602 | 1,302 |
| Octo-Small | FP32 | 360.8 | 184.4 | 100.1 | **73.5** | 96.0 | 77.9 |
| TurboVLA | FP32 | 953.6 | 479.9 | 249.8 | **188.8** | 254.4 | 191.0 |
| Diffusion Policy | FP32 | 1,992 | 1,165 | 874.4 | **741.3** | 858.7 | 939.6 |

## Register-tile selection

Uses the existing x86 AVX2 6×16 tile (Algorithm 1 selects it for any AVX2 core: 15 of 16 registers,
matching BLIS on Haswell and Zen); no search needed.

## Notes

- The sweep also ran the INT8 flags here. Without AVX-VNNI the engine keeps every layer in FP32, so
  those runs duplicate the FP32 rows and are not reported.
- Latency for SmolVLA and TurboVLA includes encoding the instruction on every query. IMPACT and
  Octo-Small cache the encoded instruction, and the benchmark repeats one instruction, so their
  timed queries skip the text encoder.
- The INT8 row of IMPACT uses `impact-int8-so101-multi-task.gguf`, the checkpoint trained for W8A8;
  the other INT8 rows quantize the FP32 checkpoint at load. INT8 changes the numerics; no accuracy
  is measured here.
- The Diffusion Policy checkpoint was converted from a local SO-101 training run with
  `tools/convert_diffusion.py --scheduler DDIM --steps 10`; it is not published on the Hub.
