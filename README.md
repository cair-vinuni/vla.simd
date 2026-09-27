<div align="center">
  <img src="assets/logo.png" alt="vla.simd" width="380">

<h1 style="border: none;">vla.simd</h1>

<p><b>Efficient CPU Inference for Language-Conditioned Manipulation</b></p>

<p>
    <a href="https://arxiv.org/abs/2609.24274">
      <img src="https://img.shields.io/badge/arXiv-2609.24274-b31b1b.svg" alt="Paper">
    </a>
    <a href="https://vla-simd.github.io/">
      <img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project page">
    </a>
    <a href="https://huggingface.co/collections/khanhnd61/vlasimd-model-bundle-6ab649fa9d1f2e8b66512a31">
      <img
        src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-vla.simd%20bundle-yellow.svg"
        alt="Hugging Face: vla.simd model bundle">
    </a>
</p>

</div>

A C++ inference engine for Vision-Language-Action policies on CPUs,
with its own tensors, operators, and SIMD kernels. The model runtime has no
GPU, CUDA, or ggml dependency.

One codebase supports x86-64, Apple Silicon, and Raspberry Pi. CMake configures
the target's instruction-set flags, while the hardware abstraction layer
selects AVX2 kernels tuned for Intel or AMD Zen, NEON kernels for ARM with
Accelerate support on Apple Silicon, or a portable scalar fallback.

## Rollout

A rollout has two parts: the `vla.simd` server loads a GGUF checkpoint and serves
actions on the CPU, and lerobot's client drives the robot against it. They run
on the same machine or on two; only the client talks to the robot.

### 1. Server

One environment serves every policy.
On Apple Silicon, install Homebrew's OpenMP first: `brew install cmake libomp`.

```sh
uv venv .serve --prompt serve --python 3.12
uv pip install --python .serve '.[serve]' --torch-backend cpu --no-sources
```

`vla-simd-serve` loads the GGUF and listens for the client.
`--model-dir` accepts a local `.gguf` file or
`hf://<user>/<repo>[@<revision>]/<file>.gguf`.

```sh
export CORES=6    # 8 on the M4, 16 on the i9, 12 on the Ryzen, 4 on a Pi 5

OMP_NUM_THREADS=$CORES .serve/bin/vla-simd-serve --model impact --port 8080 \
    --model-dir hf://khanhnd61/impact-so101-multi-task-gguf/impact-so101-multi-task.gguf
```

Supported values of `--model`:

| `--model` | Notes |
| --- | --- |
| `impact`, `act`, `smolvla` | nothing extra |
| `turbovla` | add `--task "<instruction>"` unless the GGUF records one; frames consumed as given, at the checkpoint's resolution |
| `octo` | add `--cams front,wrist`, the robot's camera names, primary first; the GGUF records none. `--cams front` serves a robot without a wrist camera |
| `diffusion` | prefix `DP_SCHEDULER=DDIM DP_STEPS=10`; the 2-frame history is assembled from the stream |

<details>
<summary><b>Server options</b></summary>

`--bench N` (or `--soak SEC`; `--json` for JSON output) times N queries after
warmup and exits. JSON includes median/p95 latency, process peak RSS in bytes,
checkpoint path, configuration, backend, and runtime settings.
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

</details>

<details>
<summary><b>Docker</b></summary>

The server also runs in a container, in place of the `.serve` install above. The
image builds the package for the platform it is built on, x86-64 with AVX2 or
aarch64 (a Raspberry Pi 5), and fetches the GGUF itself; the `vla-simd-cache`
volume keeps the download between runs:

```sh
docker build -t vla-simd .
docker run --rm -p 127.0.0.1:8080:8080 -v vla-simd-cache:/root/.cache vla-simd \
    --model impact --host 0.0.0.0 \
    --model-dir hf://khanhnd61/impact-so101-multi-task-gguf/impact-so101-multi-task.gguf
```

`--host 0.0.0.0` listens inside the container; `-p 127.0.0.1:8080:8080` decides
who can reach it from outside. `docker build --platform linux/arm64 -t vla-simd .`
builds the Pi image on an x86-64 host under QEMU.

</details>

### 2. Client

The server is a drop-in replacement for `lerobot.async_inference.policy_server`,
so the robot side is lerobot's own async client, `lerobot-vla-simd`, from the
[lerobot fork](https://github.com/khanhnd61-vr/lerobot). The `serve` install
above already includes it. On a robot machine that does not run the server,
install the client with:

```sh
uv venv .client --prompt client --python 3.12
uv pip install --python .client --group client --torch-backend cpu --no-sources
```

With the server running, start the client with `--policy_type` matching the
server's `--model`. Use `.serve/bin/lerobot-vla-simd` if you installed the client
in the serving environment instead:

```sh
.client/bin/lerobot-vla-simd --server_address=127.0.0.1:8080 \
    --policy_type=impact \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }" \
    --actions_per_chunk=50 \
    --task="put the tape into the box"
```

Octo and Diffusion Policy see consecutive frames only if the client sends every frame,
so run the client with `--chunk_size_threshold=1.0` for them,
and `--actions_per_chunk=4` for Octo.

## Build and test

The C++ engine needs CMake 3.21+, a C++17 compiler, and OpenMP for parallel
inference. Python and Torch are needed for conversion and serving, not for the
shared libraries.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_COMPILE_WARNING_AS_ERROR=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

Use `-DVLA_SCALAR=ON` to test the portable backend on a SIMD-capable host.
For memory and undefined-behavior checks, configure a separate Debug build with
`-DVLA_SANITIZE=address,undefined,float-cast-overflow`. On x86,
`TCPU_ZEN=1 ctest --test-dir build --output-on-failure` exercises the Zen dispatch
path. This does not replace testing on AMD hardware.

The Python regressions use the standard library's test runner. The serving
environment runs the core and protocol tests:

```sh
OMP_NUM_THREADS=4 VLA_TEST_BUILD=build \
  .serve/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Reference tests skip when Diffusers or a CMake build is absent. To run every
Python test in one environment, install the serving dependencies with the
Diffusion extra:

```sh
uv venv .venv-test --python 3.12
uv pip install --python .venv-test -r pyproject.toml --extra serve \
  'lerobot[diffusion]' --torch-backend cpu --no-sources
OMP_NUM_THREADS=4 VLA_TEST_BUILD=build HF_HUB_OFFLINE=1 \
  .venv-test/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

The reference tests use random weights and matched noise, so they need no model
downloads. Trained ACT, IMPACT, SmolVLA, and TurboVLA comparisons, measured
packing and inference results, and remaining validation gaps are in the
[benchmark report](docs/benchmark.md).

### Continuous integration

The [workflow](.github/workflows/build.yml) always checks every tracked Markdown
file with PyMarkdown and validates its own YAML and shell commands with actionlint.
Changes limited to Markdown, README images, or Markdown lint settings skip the
build jobs. Use the manual workflow trigger to run the full matrix anyway.

Code changes run these checks:

| Check | Coverage |
| --- | --- |
| C++ matrix | GCC and Clang on x86; Release, Debug, and scalar; ARM NEON; Apple Accelerate; Zen dispatch on x86 |
| Linux Release | CMake installation and C consumer; all Python, protocol, and Diffusers reference tests, with skips treated as failures |
| Sanitizers | Address, undefined behavior, and float-to-integer overflow |
| Wheel | Core tests in isolated Python mode against the installed package; ABI checks for all six libraries |
| Docker | Native x86 and ARM builds; protocol tests and ABI checks against the installed image |

The Release build also supplies the installation and reference checks. uv caches
Python dependencies. Docker installs serving dependencies before copying source
files, then caches build layers separately for x86 and ARM. C++ tests have a
two-minute timeout; Python subprocess checks have a ten-second timeout.

Actions are pinned to commit SHAs: checkout 7.0.1, setup-uv 10.2.0,
setup-buildx-action 4.4.1, and build-push-action 7.4.0. Tool versions are uv
0.12.19, PyMarkdown 0.9.40, and actionlint 1.7.12. The actionlint download is
verified by SHA-256. The standalone `install` job is now part of Linux Release;
branch protection that required `install` must use the Release matrix check.

Run the same Markdown check locally:

```sh
git ls-files -z '*.md' | xargs -0 uvx --from pymarkdownlnt==0.9.40 \
  pymarkdown --config .pymarkdown.json scan
```

The lint settings allow the README's HTML header and collapsible sections,
enable GitHub tables, and limit prose lines to 100 characters. Code blocks and
tables can be wider. Live external-link checks stay outside CI to avoid network
flakiness.

## Converter environments

Create one environment per group. Install groups from this repository's root;
`uv pip install --group` installs the conversion dependencies without building
the engine.

| Purpose | Group or extra | Python | Compatibility |
| --- | --- | --- | --- |
| Serving all models | `.[serve]` | 3.12+ | LeRobot fork wire protocol; CPU Torch |
| Robot client only | `client` | 3.12+ | Same fork as the server |
| Torch-free safetensors conversion | `numpy` | 3.10+ | NumPy 2.x |
| ACT, Diffusion, Torch SmolVLA conversion | `lerobot` | 3.12+ | LeRobot 0.6.x dependency limits |
| IMPACT conversion | `impact` | 3.12+ | Fork with the IMPACT implementation |
| TurboVLA conversion | `turbovla` | 3.10+ | Transformers 4.57.1 preserves DINOv3 hidden-state semantics |
| Original JAX Octo conversion | `octo` | 3.10–3.11 | Legacy JAX/Flax/NumPy pins |

```sh
uv venv .lerobot --python 3.12
uv pip install --python .lerobot --group lerobot --torch-backend cpu --no-sources
uv venv .turbovla --python 3.12
uv pip install --python .turbovla --group turbovla --torch-backend cpu --no-sources
uv venv .octo --python 3.11
uv pip install --python .octo --group octo
```

TurboVLA also requires the upstream code checkout described in
`tools/convert_turbovla.py`. Newer Transformers versions change that model's
outputs or remove APIs it uses. The [dependency audit](docs/benchmark.md#dependency-decisions)
records the tested versions and retained caps. Keep Octo in its own environment;
its NumPy 1.x requirement conflicts with modern LeRobot.

## Citation

```bibtex
@article{nguyen2026vlasimd,
  title   = {{vla.simd}: Efficient {CPU} Inference for Language-Conditioned Manipulation},
  author  = {Nguyen, Khanh D. and Truong, Hoang M. and Le, An T.},
  journal = {arXiv preprint arXiv:2609.24274},
  year    = {2026}
}
```

## License

`vla.simd` is released under the [Apache 2.0 license](LICENSE).

## Acknowledgements

- [ACT](https://huggingface.co/papers/2304.13705) and
  [LeRobot](https://github.com/huggingface/lerobot) - reference implementations
  and the async-inference protocol
- [Octo](https://github.com/octo-models/octo) - the authoritative JAX model
- [TurboVLA](https://github.com/H-EmbodVis/TurboVLA) - the LIBERO checkpoints
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) - the U-Net action denoiser (Chi et al., 2023)
- [TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine) - CPU ops for LLMs, not based on ggml
- [VAMP](https://github.com/KavrakiLab/vamp) - SIMD accelerator in the same robotics domain
