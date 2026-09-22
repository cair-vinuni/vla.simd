<div align="center">
  <img src="assets/logo.png" alt="vla.simd" width="380">

<h1 style="border: none;">vla.simd</h1>

<p><b>Efficient CPU Inference for Language-Conditioned Manipulation</b></p>

<p align="center">
    <a href="#">📑 Paper</a> |
    <a href="https://vla-simd.github.io/">🌐 Project Page</a>
</p>

</div>

A pure C++ inference engine for Vision-Language-Action policies on **CPUs**, with no GPU, CUDA, or ggml dependency. Built with its own tensors, operators, and SIMD kernels, the engine keeps kernels easy to inspect, tune, and replace.

One codebase supports x86-64, Apple Silicon, and Raspberry Pi. CMake configures the target’s instruction-set flags, while the hardware abstraction layer selects AVX2 kernels tuned for Intel or AMD Zen, NEON kernels for ARM with Accelerate support on Apple Silicon, or a portable scalar fallback.

## Build

```sh
cmake -S . -B build && cmake --build build -j
```

Apple Silicon needs Homebrew's OpenMP first: `brew install cmake libomp`.
Presets: `release`, `debug`, `ci` (release plus `-Werror`).

| Option | Default | What it does |
|---|---|---|
| `VLA_HAL` | `auto` | x86 backend: `avx2` (Intel tuning) or `amd` (Zen). `auto` reads the build host's vendor, and does not when cross-compiling |
| `VLA_LTO` | OFF | link-time optimization; measured neutral here, so off |
| `VLA_SANITIZE` | "" | e.g. `address,undefined`, or `thread` |
| `VLA_ARM_NATIVE` | OFF | build aarch64 with `-mcpu=native`. Measured 8% *slower* than the baseline on a Cortex-A76, hence off |
| `VLA_ARM_FLAGS` | "" | replace the aarch64 `-march`/`-mcpu` flags outright |

## Convert and Serve

The engine loads flat `.meta`/`.bin` arenas, not framework checkpoints, so every
policy is converted once, offline. **Each converter must run in an environment
matching the one its checkpoint was trained in**, not a single shared venv: see
the tiers in [tools/requirements-convert.txt](tools/requirements-convert.txt).
Nothing a converter needs is required at runtime, so the robot never carries
torch.

One `serve/policy_server.py` serves every policy; `--model` picks which. It is a
drop-in replacement for `lerobot.async_inference.policy_server`, so the robot
side is lerobot unchanged apart from its client: `lerobot-vla-simd` ships in the
forked [lerobot](https://github.com/khanhnd61-vr/lerobot), which
adds the vla.simd and vla.cpp CLIs. It can replay a recorded dataset at the
server with no hardware attached:

```sh
lerobot-vla-simd --server_address=127.0.0.1:8080 --replay.repo_id=<dataset>
```

> That protocol is pickle over an unauthenticated socket, which trusts the peer.
> Keep it on a lab LAN or an SSH tunnel, never on an open network.

### Environments

Serving is one environment for every policy. Converting is not: a torch
converter has to run in the environment its checkpoint was **trained** in, so
each box below says which venv it belongs to.

| venv | Needed for |
|---|---|
| `.venv-serve` | serving, every model |
| `.venv-convert` | torch-free converts (SmolVLA; runs on a Pi) |
| `.venv-train` | converting ACT, IMPACT, Diffusion |
| `.venv-turbovla` | converting TurboVLA |
| `.venv-octo` | converting Octo, frozen JAX on Python 3.10 |

```sh
# serving, every model
uv venv .venv-serve
uv pip install --python .venv-serve -r serve/requirements.txt

# torch-free converts
uv venv .venv-convert
uv pip install --python .venv-convert -r tools/requirements-convert.txt

# ACT, IMPACT, Diffusion — or reuse the training venv
uv venv .venv-train
uv pip install --python .venv-train lerobot

# TurboVLA
uv venv .venv-turbovla
uv pip install --python .venv-turbovla torch 'transformers>=4.57,<5'

# Octo
uv venv --python 3.10 .venv-octo
git clone https://github.com/octo-models/octo third_party/octo
uv pip install --python .venv-octo -e ./tools/octo
uv pip install --python .venv-octo -e third_party/octo --no-deps
```

`$CORES` is the OpenMP thread count used by every serve command below:

```sh
export CORES=6    # 8 on the M4, 16 on the i9, 12 on the Ryzen, 4 on a Pi 5
```

### Models

We provide a converter and a server recipe for six policies, expand the one you
need.

<details open>
<summary><b>IMPACT</b></summary>

Convert, in `.venv-train`:

```sh
.venv-train/bin/python tools/impact/convert_impact.py --checkpoint <dir> --out build/impact
```

Serve, in `.venv-serve`:

```sh
OMP_NUM_THREADS=$CORES \
.venv-serve/bin/python serve/policy_server.py --model impact --model-dir build/impact --port 8080
```

</details>

<details>
<summary><b>ACT</b></summary>

Convert, in `.venv-train`:

```sh
.venv-train/bin/python tools/act/convert_act.py <hub-id-or-dir> build/act
```

Serve, in `.venv-serve`:

```sh
OMP_NUM_THREADS=$CORES \
.venv-serve/bin/python serve/policy_server.py --model act --model-dir build/act --port 8080
```

</details>

<details>
<summary><b>Octo-Small</b></summary>

Convert, base model and tokenizer in `.venv-octo`;
a lerobot finetune reuses the base T5 tower and runs in `.venv-train`:

```sh
.venv-octo/bin/python    tools/octo/convert_octo.py build/octo
.venv-octo/bin/python    tools/octo/convert_t5_tokenizer.py build/octo_tok
.venv-train/bin/python   tools/octo/convert_lerobot_octo.py --checkpoint <id> \
    --out build/octo_ft --t5-from build/octo
```

Serve, in `.venv-serve`; `--cams` primary first (the converter records no names):

```sh
OMP_NUM_THREADS=$CORES \
.venv-serve/bin/python serve/policy_server.py --model octo --model-dir build/octo \
    --cams primary,wrist --port 8080
```

</details>

<details>
<summary><b>TurboVLA</b></summary>

Convert, in `.venv-turbovla`:

```sh
.venv-turbovla/bin/python tools/turbovla/convert_turbovla.py --ckpt <file.pth> --out build/turbovla
```

Serve, in `.venv-serve`; frames consumed as given at the checkpoint's resolution:

```sh
OMP_NUM_THREADS=$CORES \
.venv-serve/bin/python serve/policy_server.py --model turbovla --model-dir build/turbovla --port 8080
```

</details>

<details>
<summary><b>SmolVLA</b></summary>

Convert, in `.venv-convert` (no torch, so this one runs on a Pi):

```sh
.venv-convert/bin/python tools/smolvla/convert_hf_safetensors.py <hub-id> build/smolvla \
    --task "Put the tape into the box"
.venv-convert/bin/python tools/smolvla/convert_tokenizer.py
```

Serve, in `.venv-serve`:

```sh
OMP_NUM_THREADS=$CORES \
.venv-serve/bin/python serve/policy_server.py --model smolvla --model-dir build/smolvla --port 8080
```

</details>

<details>
<summary><b>Diffusion Policy</b></summary>

Convert, in `.venv-train`:

```sh
.venv-train/bin/python tools/diffusion/convert_diffusion.py --checkpoint <dir> --out build/diffusion
```

Serve, in `.venv-serve`; the 2-frame history is assembled from the stream:

```sh
DP_SCHEDULER=DDIM DP_STEPS=10 OMP_NUM_THREADS=$CORES \
.venv-serve/bin/python serve/policy_server.py --model diffusion --model-dir build/diffusion --port 8080
```

</details>

## License

`vla.simd` is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

- [ACT](https://huggingface.co/papers/2304.13705) and [lerobot](https://github.com/huggingface/lerobot) - the reference implementations and the async-inference protocol
- [Octo](https://github.com/octo-models/octo) - the authoritative JAX model
- [TurboVLA](https://github.com/H-EmbodVis/TurboVLA) - the LIBERO checkpoints
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) (Chi et al. 2023)
- [TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine) - CPU ops for LLMs, non-ggml based
- [VAMP](https://github.com/KavrakiLab/vamp) - SIMD accelerator in the same robotics domain
- [stb_image](https://github.com/nothings/stb) - image loading
