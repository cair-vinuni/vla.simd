# {Device} benchmark

{One or two sentences: the fastest configuration per model family and the headline
latency range on this device.}

## Test setup

| Item | Value |
| --- | --- |
| Device | {product name, RAM size} |
| CPU | {model, microarchitecture, cores and threads (P+E split if hybrid), max clock} |
| SIMD used | {AVX2 + FMA, AVX-VNNI / NEON, dotprod, bf16 / ...} |
| Memory | {total RAM, type and speed if known} |
| OS | {distribution or OS version, kernel} |
| Compiler | {compiler and version, target flags such as `-march=native` or `-mcpu=...`} |
| Python, NumPy | {3.12.x, 2.x.x} |
| Power | {AC or battery, CPU governor or power profile} |
| Commit | `{full SHA}` (`fix-inference-and-ci`) |
| Date | {YYYY-MM-DD HH:MM UTC+7} |

## Method

- Harness: `vla-simd-serve --model {model} --model-dir {gguf} --bench {N} --json`,
  one fresh process per configuration.
- Warmup: {W} untimed predictions after loading, then {N} timed predictions.
- Inputs: new random `uint8` frames for every query, fixed state and instruction,
  seed 0.
- Latency: wall time of one complete `predict` call, covering image preprocessing,
  every encoder, all solver steps, and action unnormalization. Network and gRPC
  time are excluded.
- Memory: peak resident set size (RSS) of the benchmark process. The model's
  share is peak RSS minus the RSS of the same interpreter after importing the
  server but before loading a checkpoint ({baseline} MiB on this device).
- VRAM: not applicable. vla.simd runs on the CPU only.
- Tuning: for every model and precision, the thread count is swept over
  {list}, together with the runtime settings listed under Engine configuration.
  The results table reports the fastest median.

## Model configurations

| Model | Checkpoint | Weights | Cameras and frame size | Network input | Action chunk | State | Language | Action decoding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | `{file}.gguf` | {MB, dtype} | {n × H×W} | {H×W} | {steps × dims} | {dims} | none | single pass |
| IMPACT | `{file}.gguf` | {MB, dtype} | {n × H×W} | {H×W} | {steps × dims} | {dims} | T5-small, cached | single pass |
| SmolVLA | `{file}.gguf` | {MB, dtype} | {n × H×W} | {px} | {steps × dims} | {dims} | SmolLM2 | flow matching, {k} steps |
| Octo-Small | `{file}.gguf` | {MB, dtype} | {primary, wrist} | 256 / 128 | {steps × dims} | none | T5-base | diffusion head, {k} steps |
| TurboVLA | `{file}.gguf` | {MB, dtype} | {n × H×W} | {px} | {steps × dims} | {dims} | BERT | {head} |
| Diffusion Policy | `{file}.gguf` | {MB, dtype} | {n × H×W × obs steps} | {H×W} | {steps × dims} (horizon {h}) | {dims} | none | {DDIM}, {k} steps |

## Engine configuration

| Item | Value |
| --- | --- |
| Backend | {`x86-avx2`, `amd-zen`, `neon`, or `apple`} |
| GEMM register tile (mr × nr) | {6×16 / 4×16 / ...} |
| INT8 (W8A8) path | {AVX-VNNI / dotprod / not available} |
| Build | {install command or CMake preset and flags} |

Fastest settings per model:

| Model | Precision | Threads | INT8 mask | Other settings |
| --- | --- | --- | --- | --- |
| ACT | FP32 | {t} | none | {env or flags, or defaults} |
| ACT | INT8 | {t} | {mask} | {...} |
| ... | | | | |

## Results

| Model | Precision | Median (ms) | p10 (ms) | p90 (ms) | p95 (ms) | Actions/s | Peak RSS (MiB) | Model RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | FP32 | | | | | | | |
| ACT | INT8 | | | | | | | |
| IMPACT | FP32 | | | | | | | |
| IMPACT | INT8 | | | | | | | |
| SmolVLA | FP32 | | | | | | | |
| SmolVLA | INT8 | | | | | | | |
| Octo-Small | FP32 | | | | | | | |
| Octo-Small | INT8 | | | | | | | |
| TurboVLA | FP32 | | | | | | | |
| Diffusion Policy | FP32 | | | | | | | |
| Diffusion Policy | INT8 | | | | | | | |

TurboVLA has no INT8 path. INT8 rows are omitted on CPUs without AVX-VNNI or dotprod.

## Thread sweep

Median latency in ms. The fastest entry in each row is in bold.

| Model | Precision | {t1} | {t2} | {t3} | {...} |
| --- | --- | --- | --- | --- | --- |
| ACT | FP32 | | | | |

## Register-tile selection

{Only for a CPU without an existing tuned backend. Inputs to Algorithm 1 of the
paper (L, Rmax, Pfma, Pld, τ, g), the feasible candidates with their scores and
spill check, and the measured GFLOP/s of each candidate on the six benchmarked
layer shapes. Otherwise: "Uses the existing {backend} tile; no search needed."}

## Notes

{Anything that affects the numbers: thermal throttling, background load, settings
that failed or were skipped and why.}
