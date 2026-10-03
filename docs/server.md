# Server options

Options of `vla-simd-serve` beyond the quick start in the [README](../README.md).

`--bench N` (or `--soak SEC`; `--json` for JSON output) times N queries after
`--warmup` untimed ones (default 2) and exits. JSON includes median/p95 latency,
process peak RSS and the RSS before loading, in bytes, checkpoint path,
configuration, backend, and runtime settings.
`--int8 MASK` runs the W8A8 path on CPUs with AVX-VNNI or dotprod
(`impact-int8-so101-multi-task.gguf` was trained for it: serve it with `--int8 63`):

| Setting | Effect |
| --- | --- |
| `--int8 MASK` | `ACT_INT8` / `IMPACT_INT8`: 1 encoder attention, 2 encoder w1, 4 encoder w2, 8 token projections, 16 decoder, 32 ResNet convolutions. `SMOLVLA_INT8`: 1/2/4 ViT attention/w1/w2, 8 ViT patch embed and connector, 16 language model, 32 action expert. `OCTO_INT8`: 1/2/4 transformer attention/w1/w2, 8 token projections, 16 stem convolutions, 32 diffusion head. `DIFFUSION_INT8`: 1 UNet convolutions, 2 ResNet convolutions |
| `DP_SCHEDULER`, `DP_STEPS` | Diffusion Policy sampler (`DDPM` or `DDIM`) and step count |
| `SMOLVLA_NUM_STEPS` | SmolVLA flow-matching steps |
| `--rtc-horizon H` | SmolVLA real-time chunking (lerobot's RTC): guide each chunk toward the previous one's unexecuted actions over the next H (0, the default, is off). `--rtc-max-guidance` caps the guidance weight (10), `--rtc-delay D` fixes the frozen prefix instead of measuring it from latency and `--fps`. Run the client with `--aggregate_fn_name=latest_only`, and raise `--rtc-delay` when the network adds latency |
| `TCPU_VIEW_THREADS`, `TCPU_EXPERT_THREADS`, `TCPU_OMP_MIN` | threading of the camera views, the SmolVLA expert loop, and the size below which small ops stay single-threaded |
| `TCPU_ZEN=0`, `TCPU_ZEN=1` | force the Intel or the AMD Zen attention layout on x86; the default follows the CPU vendor |
| `TCPU_BF16_MLP=1`, `TCPU_BF16_DEQ=0` | bf16 MLP weights on the Pi; keep bf16 checkpoint weights resident on x86 |

## Concurrency

Inference runs one request at a time, outside the session lock, so the robot
client's `SendObservations` returns as soon as the observation is validated and
queued even while a prediction is in progress. The observation queue holds one
entry and keeps the freshest, so the model always works on the newest
observation the client has sent. `Ready` or new policy instructions arriving
mid-prediction discard that prediction's result instead of handing it to the new
session.

## Windows on Arm

On Windows on Arm (Snapdragon X), build with [llvm-mingw](https://github.com/mstorsjo/llvm-mingw),
CMake and Ninja on `PATH`, and a native ARM64 Python: an x64 Python cannot load
the ARM64 engine. This installs the engine and `--bench`; the `serve` extra
(lerobot, torch) is untested there.

```bat
set CMAKE_GENERATOR=Ninja
set CC=clang
set CXX=clang++
uv venv .serve --python cpython-3.12-windows-aarch64-none
uv pip install --python .serve\Scripts\python.exe --only-binary numpy .
```
