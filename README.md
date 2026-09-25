<div align="center">
  <img src="assets/logo.png" alt="vla.simd" width="380">

<h1 style="border: none;">vla.simd</h1>

<p><b>Efficient CPU Inference for Language-Conditioned Manipulation</b></p>

<p>
    <a href="https://arxiv.org/abs/2609.24274">📑 Paper</a> |
    <a href="https://vla-simd.github.io/">🌐 Project Page</a>
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

## Checkpoints

The policies evaluated in the paper are published as GGUF in the
[vla.simd model bundle](https://huggingface.co/collections/khanhnd61/vlasimd-model-bundle-6ab649fa9d1f2e8b66512a31)
on the Hugging Face Hub:

| policy | checkpoints |
| --- | --- |
| ACT | [act-so101-multi-task-gguf](https://huggingface.co/khanhnd61/act-so101-multi-task-gguf) |
| IMPACT | [impact-so101-multi-task-gguf](https://huggingface.co/khanhnd61/impact-so101-multi-task-gguf) (its `impact-int8-*` file was trained for W8A8: serve it with `--int8 63`), [impact-so101-long-gguf](https://huggingface.co/khanhnd61/impact-so101-long-gguf), [impact-libero-gguf](https://huggingface.co/khanhnd61/impact-libero-gguf) |
| SmolVLA | [smolvla-so101-multi-task-gguf](https://huggingface.co/khanhnd61/smolvla-so101-multi-task-gguf), [smolvla-so101-long-gguf](https://huggingface.co/khanhnd61/smolvla-so101-long-gguf) |
| Octo | [octo-small-so101-multi-task-gguf](https://huggingface.co/khanhnd61/octo-small-so101-multi-task-gguf), [octo-small-so101-long-gguf](https://huggingface.co/khanhnd61/octo-small-so101-long-gguf) |
| TurboVLA | [vrfai/turbovla-libero-gguf](https://huggingface.co/vrfai/turbovla-libero-gguf) (vla.cpp's, see below) |

## Serve

Serving is one environment for every policy, and the only one the robot needs.
Installing the package builds the engine:

```sh
uv venv .serve --prompt serve --python 3.12
uv pip install --python .serve '.[serve]' --torch-backend cpu
```

One `vla-simd-serve` serves every policy (ACT, IMPACT, SmolVLA, Octo, TurboVLA
and Diffusion Policy) from its GGUF; `--model` picks which, and `$CORES` is the
OpenMP thread count. `--model-dir` is a `.gguf` file, a
directory holding exactly one, or `hf://<user>/<repo>[@<revision>]`, with
`/<file>.gguf` appended when the repo holds several:

```sh
export CORES=6    # 8 on the M4, 16 on the i9, 12 on the Ryzen, 4 on a Pi 5

OMP_NUM_THREADS=$CORES .serve/bin/vla-simd-serve --model impact --port 8080 \
    --model-dir hf://khanhnd61/impact-so101-multi-task-gguf/impact-so101-multi-task.gguf \
    --task "put the tape into the box"
```

The GGUF carries the weights, tokenizer, normalization statistics and camera
order, so nothing else is needed.

| `--model` | notes |
| --- | --- |
| `impact`, `act`, `smolvla` | nothing extra |
| `turbovla` | add `--task "<instruction>"` unless the GGUF records one; frames consumed as given, at the checkpoint's resolution |
| `octo` | add `--cams front,wrist`, the robot's camera names, primary first; the GGUF records none. `--cams front` serves a robot without a wrist camera |
| `diffusion` | prefix `DP_SCHEDULER=DDIM DP_STEPS=10`; the 2-frame history is assembled from the stream |

Octo and Diffusion Policy see consecutive frames only if the client sends every
frame, so run the client with `--chunk_size_threshold=1.0` for them.

`--bench N` (or `--soak SEC`; `--json` for JSON output) times N queries after
warmup and exits, reporting the backend it ran on.
`--int8 MASK` runs the W8A8 path on CPUs with AVX-VNNI or dotprod:

| knob | effect |
| --- | --- |
| `--int8 MASK` | `ACT_INT8` / `IMPACT_INT8`: 1 encoder attention, 2 encoder w1, 4 encoder w2, 8 token projections, 16 decoder, 32 ResNet convolutions. `SMOLVLA_INT8`: 1/2/4 ViT attention/w1/w2, 8 ViT patch embed and connector, 16 language model, 32 action expert. `OCTO_INT8`: 1/2/4 transformer attention/w1/w2, 8 token projections, 16 stem convolutions, 32 diffusion head. `DIFFUSION_INT8`: 1 UNet convolutions, 2 ResNet convolutions |
| `DP_SCHEDULER`, `DP_STEPS` | Diffusion Policy sampler (`DDPM` or `DDIM`) and step count |
| `SMOLVLA_NUM_STEPS` | SmolVLA flow-matching steps |
| `--rtc-horizon H` | SmolVLA real-time chunking (lerobot's RTC): guide each chunk toward the previous one's unexecuted actions over the next H (0, the default, is off). `--rtc-max-guidance` caps the guidance weight (10), `--rtc-delay D` fixes the frozen prefix instead of measuring it from latency and `--fps`. Run the client with `--aggregate_fn_name=latest_only`, and raise `--rtc-delay` when the network adds latency |
| `TCPU_VIEW_THREADS`, `TCPU_EXPERT_THREADS`, `TCPU_OMP_MIN` | threading of the camera views, the SmolVLA expert loop, and the size below which small ops stay single-threaded |
| `TCPU_ZEN=0`, `TCPU_ZEN=1` | force the Intel or the AMD Zen attention layout on x86; the default follows the CPU vendor |
| `TCPU_BF16_MLP=1`, `TCPU_BF16_DEQ=0` | bf16 MLP weights on the Pi; keep bf16 checkpoint weights resident on x86 |

The server is a drop-in replacement for `lerobot.async_inference.policy_server`,
so the robot side runs lerobot unchanged except for its client,
`lerobot-vla-simd`, which ships in the
[lerobot fork](https://github.com/khanhnd61-vr/lerobot). It installs into
`.serve`, or into any Python 3.12 venv on the robot when the server runs in
Docker or on another machine:

```sh
uv pip install --python .serve \
    'lerobot[async,feetech] @ git+https://github.com/khanhnd61-vr/lerobot@4b33b84296c0880ebce778d69f16a38d33825575'

.serve/bin/lerobot-vla-simd --server_address=127.0.0.1:8080 --policy_type=<name> \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 --robot.id=my_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }" \
    --actions_per_chunk=50 \
    --task="pick up the tape"
```

### vla.cpp GGUF

[vla.cpp](https://github.com/VinRobotics/vla.cpp) publishes its own GGUFs for
SmolVLA, TurboVLA and Octo (it has no ACT, IMPACT or Diffusion Policy), and
those load the same way:

```sh
OMP_NUM_THREADS=$CORES .serve/bin/vla-simd-serve --model smolvla \
    --model-dir hf://vrfai/smolvla-libero-gguf --task "put the bowl on the plate"
```

A vla.cpp GGUF does not carry everything the engine reads. The server fetches
the rest once into `~/.cache/vla_simd/gguf` (under `$VLA_SIMD_CACHE` when set),
and a file placed beside the `.gguf` takes precedence:

| `--model` | from the Hub | notes |
| --- | --- | --- |
| `smolvla` | `tok/` (SmolVLM2-500M-Instruct) | `pos_ids shifted` in a `config.txt` beside the GGUF for a checkpoint trained with transformers 4.55-4.57 |
| `turbovla` | `vocab.txt` (bert-base-uncased), `stats.bin` (TurboVLA's `libero_all4_stats.json`) | vla.cpp's GGUF lacks DINOv3's final norm, so the engine runs without it as vla.cpp does, and warns |
| `octo` | nothing | set `VLA_OCTO_UNNORM_DATASET` when the GGUF has several datasets' statistics, as vla.cpp requires. `vrfai/octo-small-libero-gguf` was finetuned on the primary camera alone (its wrist tower is untrained), so serve it with `--cams primary` |

### Docker

The image builds the package for the platform it is built on, x86-64 with AVX2
or aarch64 (a Raspberry Pi 5):

```sh
docker build -t vla-simd .
.serve/bin/hf download khanhnd61/act-so101-multi-task-gguf act-so101-multi-task.gguf --local-dir act
docker run --rm -p 127.0.0.1:8080:8080 -v "$PWD/act:/m:ro" vla-simd \
    --model act --model-dir /m/act-so101-multi-task.gguf --host 0.0.0.0
```

`docker build --platform linux/arm64 -t vla-simd .` builds the Pi image on an
x86-64 host under QEMU.

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
