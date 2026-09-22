<div align="center">
  <img src="assets/logo.png" alt="vla.simd" width="380">

<h1 style="border: none;">vla.simd</h1>

<p><b>Efficient CPU Inference for Language-Conditioned Manipulation</b></p>

<p align="center">
    <a href="https://arxiv.org/abs/2609.24274">📑 Paper</a> |
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

## Convert

The engine loads flat `.meta`/`.bin` arenas, not framework checkpoints, so every
policy is converted once, offline. Nothing a converter needs is required at
runtime, so the robot never carries torch.

**A torch converter must run in the environment its checkpoint was trained in**,
not one shared venv. Four environments cover the six policies, all of them
extras in [pyproject.toml](pyproject.toml):

| venv | install | Python | converts |
|---|---|---|---|
| `.lerobot` | `-e '.[lerobot]'` | >=3.12 | ACT, IMPACT, Diffusion, SmolVLA |
| `.turbovla` | `-e '.[turbovla]'` | >=3.10 | TurboVLA |
| `.octo` | `-e '.[octo]'` | 3.10 or 3.11 | Octo |
| `.numpy` | `-e '.[numpy]'` | >=3.10 | Torch-free conversion (optional) |

Create only the one you need:

```sh
uv venv .numpy --prompt numpy                   && uv pip install --python .numpy    -e '.[numpy]'
uv venv .lerobot --prompt lerobot --python 3.12 && uv pip install --python .lerobot  -e '.[lerobot]'
uv venv .turbovla --prompt turbovla             && uv pip install --python .turbovla -e '.[turbovla]'

# Octo also needs the upstream checkout, installed without its own pins
uv venv .octo --prompt octo --python 3.10 && uv pip install --python .octo -e '.[octo]'
git clone https://github.com/octo-models/octo third_party/octo
uv pip install --python .octo -e third_party/octo --no-deps
```

Convert once per checkpoint:

```sh
# IMPACT
(lerobot) $ python tools/convert_impact.py --ckpt <dir> --out build/impact

# ACT
(lerobot) $ python tools/convert_act.py --ckpt <hub-id-or-dir> --out build/act

# Diffusion Policy
(lerobot) $ python tools/convert_diffusion.py --ckpt <dir> --out build/diffusion

# SmolVLA
(lerobot) $ python tools/convert_lerobot_ckpt.py --ckpt <dir> --out build/smolvla

# TurboVLA
(turbovla) $ python tools/convert_turbovla.py --ckpt <file.pth> --out build/turbovla

# Octo base model and tokenizer
(octo) $ python tools/convert_octo.py build/octo
(octo) $ python tools/convert_t5_tokenizer.py build/octo_tok
```

## Serve

Serving is one environment for every policy, and the only one the robot needs:

```sh
uv venv .serve --prompt serve --python 3.12
uv pip install --python .serve -e '.[serve]'
```

One `serve/policy_server.py` serves every policy; `--model` picks which, and
`$CORES` is the OpenMP thread count:

```sh
export CORES=6    # 8 on the M4, 16 on the i9, 12 on the Ryzen, 4 on a Pi 5

(serve) $ OMP_NUM_THREADS=$CORES python serve/policy_server.py \
              --model <name> --model-dir build/<name> --port 8080
```

| `--model` | notes |
|---|---|
| `impact`, `act`, `smolvla` | nothing extra |
| `turbovla` | frames consumed as given, at the checkpoint's resolution |
| `octo` | add `--cams primary,wrist`, primary first; the converter records no names |
| `diffusion` | prefix `DP_SCHEDULER=DDIM DP_STEPS=10`; the 2-frame history is assembled from the stream |

The server is a drop-in replacement for `lerobot.async_inference.policy_server`,
so the robot side is lerobot unchanged apart from its client: `lerobot-vla-simd`
ships in the forked [lerobot](https://github.com/khanhnd61-vr/lerobot), which
adds the `vla.simd` CLI:

```sh
lerobot-vla-simd --server_address=127.0.0.1:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 --robot.id=my_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" \
    --actions_per_chunk=50 \
    --task="pick up the tape"
```

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
