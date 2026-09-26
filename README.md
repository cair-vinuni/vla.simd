<div align="center">
  <img src="assets/logo.png" alt="vla.simd" width="380">

<h1 style="border: none;">vla.simd</h1>

<p><b>Efficient CPU Inference for Language-Conditioned Manipulation</b></p>

<p>
    <a href="https://arxiv.org/abs/2609.24274"><img src="https://img.shields.io/badge/arXiv-2609.24274-b31b1b.svg" alt="Paper"></a>
    <a href="https://vla-simd.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
    <a href="https://huggingface.co/collections/khanhnd61/vlasimd-model-bundle-6ab649fa9d1f2e8b66512a31"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-vla.simd%20bundle-yellow.svg" alt="Hugging Face: vla.simd model bundle"></a>
</p>

</div>

A pure C++ inference engine for Vision-Language-Action policies on **CPUs**,
with no GPU, CUDA, or ggml dependency. Built with its own tensors, operators,
and SIMD kernels, the engine keeps kernels easy to inspect, tune, and replace.

One codebase supports x86-64, Apple Silicon, and Raspberry Pi. CMake configures
the target's instruction-set flags, while the hardware abstraction layer
selects AVX2 kernels tuned for Intel or AMD Zen, NEON kernels for ARM with
Accelerate support on Apple Silicon, or a portable scalar fallback.

## Build

```sh
cmake -S . -B build && cmake --build build -j"$(getconf _NPROCESSORS_ONLN)"
ctest --test-dir build --output-on-failure
```

Apple Silicon needs Homebrew's OpenMP first: `brew install cmake libomp`.
Presets (`cmake --preset <name>`): `release`, `debug`, `ci` (release plus
`-Werror`). Configure with `-DVLA_SANITIZE=address,undefined` for a sanitizer
build.

`pip install .` builds the same libraries through scikit-build-core into the
`vla-simd` Python package, next to the policy server; [Serve](#serve) installs it
that way. From a checkout, `python vla_simd/policy_server.py` runs the server
against `build/` instead.

## Rollout

A rollout has two parts: the `vla.simd` server loads a GGUF checkpoint and serves
actions on the CPU, and lerobot's client drives the robot against it. They run
on the same machine or on two; only the client talks to the robot.

### 1. Start the server

One environment serves every policy. Installing the package builds the engine:

```sh
uv venv .serve --prompt serve --python 3.12
uv pip install --python .serve '.[serve]' --torch-backend cpu
```

`vla-simd-serve` loads the GGUF and listens for the client.
`--model-dir` is either a path to `.gguf` file or `hf://<user>/<repo>[@<revision>]/<file>.gguf`.
`$CORES` is the OpenMP thread count:

```sh
export CORES=6    # 8 on the M4, 16 on the i9, 12 on the Ryzen, 4 on a Pi 5

OMP_NUM_THREADS=$CORES .serve/bin/vla-simd-serve --model impact --port 8080 \
    --model-dir hf://khanhnd61/impact-so101-multi-task-gguf/impact-so101-multi-task.gguf
```

Refer the table below for valid values of `--model`:

| `--model` | notes |
| --- | --- |
| `impact`, `act`, `smolvla` | nothing extra |
| `turbovla` | add `--task "<instruction>"` unless the GGUF records one; frames consumed as given, at the checkpoint's resolution |
| `octo` | add `--cams front,wrist`, the robot's camera names, primary first; the GGUF records none. `--cams front` serves a robot without a wrist camera |
| `diffusion` | prefix `DP_SCHEDULER=DDIM DP_STEPS=10`; the 2-frame history is assembled from the stream |

<details>
<summary><b>Server options</b></summary>

`--bench N` (or `--soak SEC`; `--json` for JSON output) times N queries after
warmup and exits, reporting the backend it ran on.
`--int8 MASK` runs the W8A8 path on CPUs with AVX-VNNI or dotprod
(`impact-int8-so101-multi-task.gguf` was trained for it: serve it with `--int8 63`):

| knob | effect |
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

### 2. Run the rollout client

The server is a drop-in replacement for `lerobot.async_inference.policy_server`,
so the robot side is lerobot's own async client, `lerobot-vla-simd`, from the
[lerobot fork](https://github.com/khanhnd61-vr/lerobot). Install it into
`.serve`, or into any Python 3.12 venv on the robot's machine:

```sh
uv pip install --python .serve \
    'lerobot[async,feetech] @ git+https://github.com/khanhnd61-vr/lerobot@4b33b84296c0880ebce778d69f16a38d33825575'
```

Then, with the server running, drive the robot. `--policy_type` matches the
server's `--model`, `--server_address` is where the server listens, and
`--task` is the instruction, sent with every observation:

```sh
.serve/bin/lerobot-vla-simd --server_address=127.0.0.1:8080 --policy_type=impact \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 --robot.id=my_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }" \
    --actions_per_chunk=50 \
    --task="put the tape into the box"
```

Octo and Diffusion Policy see consecutive frames only if the client sends every
frame, so run the client with `--chunk_size_threshold=1.0` for them, and
`--actions_per_chunk=4` for Octo, whose chunk is 4 actions.
`--replay.repo_id=<user>/<dataset>` in place of the `--robot.*` flags checks a
server with no robot attached: it sends recorded frames and prints the returned
actions next to the recorded ones.

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

## Acknowledgement

- [ACT](https://huggingface.co/papers/2304.13705) and [lerobot](https://github.com/huggingface/lerobot) - the reference implementations and the async-inference protocol
- [Octo](https://github.com/octo-models/octo) - the authoritative JAX model
- [TurboVLA](https://github.com/H-EmbodVis/TurboVLA) - the LIBERO checkpoints
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) - the U-Net action denoiser (Chi et al., 2023)
- [TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine) - CPU ops for LLMs, not based on ggml
- [VAMP](https://github.com/KavrakiLab/vamp) - SIMD accelerator in the same robotics domain
