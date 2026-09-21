# `vla.simd`

A pure C++ inference engine for Vision-Language-Action policies on the **CPU**.
No GPU, no CUDA, no ggml: own tensors, own ops, own SIMD kernels, so the kernels
stay first-class and swappable.

One tree builds on x86-64, Apple Silicon and Raspberry Pi. CMake picks the ISA
flags per target and the HAL (`src/hal/arch.h`) selects the matching kernel
backend: `avx2/` (Intel tuning), `amd/` (Zen), `apple/` (NEON + Accelerate),
`neon/` (baseline ARMv8), or `scalar/` as the portable fallback.

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

## Performance

Measured warm on an Apple M4, an i9-14900HX, a Ryzen 5 5500 and a Raspberry Pi 5,
at the thread count in each column header. Bold is the fastest device in the row.

| Query latency, ms | Chunk | M4 8t | i9 16t | Ryzen 12t | Pi 5 4t |
|---|---|---:|---:|---:|---:|
| ACT | 100 x 6 | **86.9** | 112.1 | 159.5 | 915.9 |
| Octo-Small | 4 x 6 | **52.9** | 84.0 | 83.8 | 513.7 |
| IMPACT | 50 x 6 | **110.3** | 149.6 | 190.8 | 1,212 |
| TurboVLA | 12 x 7 | **137.7** | 207.1 | 212.4 | 1,285 |
| SmolVLA (450M) | 50 x 6 | **685.5** | 1,183 | 1,320 | 8,187 |
| SmolVLA-`prune6` | 50 x 6 | **516.5** | n/a | n/a | 7,982 |
| Diffusion Policy, `DDIM-10` | 32 x 6 | **551.7** | 584.3 | 1,091 | 5,078 |
| Diffusion Policy, `DDPM-100` | 32 x 6 | 4,450 | **4,446** | 9,008 | 40,055 |

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

```sh
uv venv .venv-serve                                          # serves every model
uv pip install --python .venv-serve -r serve/requirements.txt

uv venv .venv-convert                                        # torch-free converts
uv pip install --python .venv-convert -r tools/requirements-convert.txt

uv venv .venv-train                                          # ACT, IMPACT, Diffusion
uv pip install --python .venv-train lerobot                  # or reuse the training venv

uv venv .venv-turbovla                                       # TurboVLA only
uv pip install --python .venv-turbovla torch 'transformers>=4.57,<5'

uv venv --python 3.10 .venv-octo                             # Octo only, frozen JAX
git clone https://github.com/octo-models/octo third_party/octo
uv pip install --python .venv-octo -e ./tools/octo
uv pip install --python .venv-octo -e third_party/octo --no-deps
```

`$CORES` is the OpenMP thread count. The column headers in the table above give
the value each measurement used:

```sh
export CORES=6    # 8 on the M4, 16 on the i9, 12 on the Ryzen, 4 on a Pi 5
```

### ACT

Convert, in `.venv-train`:

```sh
.venv-train/bin/python tools/act/convert_act.py <hub-id-or-dir> build/act
```

Serve, in `.venv-serve`:

```sh
OMP_NUM_THREADS=$CORES \
.venv-serve/bin/python serve/policy_server.py --model act --model-dir build/act --port 8080
```

### Octo-Small

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

### IMPACT

Convert, in `.venv-train`:

```sh
.venv-train/bin/python tools/impact/convert_impact.py --checkpoint <dir> --out build/impact
```

Serve, in `.venv-serve`:

```sh
OMP_NUM_THREADS=$CORES \
.venv-serve/bin/python serve/policy_server.py --model impact --model-dir build/impact --port 8080
```

### TurboVLA

Convert, in `.venv-turbovla`:

```sh
.venv-turbovla/bin/python tools/turbovla/convert_turbovla.py --ckpt <file.pth> --out build/turbovla
```

Serve, in `.venv-serve`; frames consumed as given at the checkpoint's resolution:

```sh
OMP_NUM_THREADS=$CORES \
.venv-serve/bin/python serve/policy_server.py --model turbovla --model-dir build/turbovla --port 8080
```

### SmolVLA

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

### Diffusion Policy

Convert, in `.venv-train`:

```sh
.venv-train/bin/python tools/diffusion/convert_diffusion.py --checkpoint <dir> --out build/diffusion
```

Serve, in `.venv-serve`; the 2-frame history is assembled from the stream:

```sh
DP_SCHEDULER=DDIM DP_STEPS=10 OMP_NUM_THREADS=$CORES \
.venv-serve/bin/python serve/policy_server.py --model diffusion --model-dir build/diffusion --port 8080
```

## Quantization

Shared, opt-in and off by default: one symmetric int8 `sdot` kernel, enabled per
layer group by a `<MODEL>_INT8` bitmask. It needs a hardware dot product,
`asimddp` on ARM or AVX-VNNI on x86, checked at runtime; it falls back to fp32
silently otherwise. It is **lossy**, so nothing routes there by default and each
model owns its own accuracy argument.

| Bit | `SMOLVLA_INT8` | `ACT_INT8` / `IMPACT_INT8` | `OCTO_INT8` |
|---:|---|---|---|
| 1 | vision attention projections (q/k/v/o) | encoder attention projections | transformer attention projections |
| 2 | vision MLP fc1 | encoder w1 | transformer MLP w1 |
| 4 | vision MLP fc2 | encoder w2 | transformer MLP w2 |
| 8 | patch embed + connector projection | token projections (image 1x1, state) | token projections + both stem embeds |
| 16 | VLM text tower | the whole decoder | stem convolutions |
| 32 | action expert (denoise loop) | ResNet-18 convs (stem included) | diffusion score net |

`63` is every group. Octo has no text-tower bit: its T5 output is a pure function
of the instruction and is cached per episode, so quantizing it would trade
accuracy for a saving that amortizes to nothing.

The mask is read at load, so it prefixes the same commands as above:

```sh
# every group but the decoder: ~2.2x on a Pi 5, at 2.8% chunk error
ACT_INT8=47 OMP_NUM_THREADS=$CORES \
.venv-serve/bin/python serve/policy_server.py --model act --model-dir build/act --port 8080

# all but vision fc2 (bit 4), which carries most of the error: ~1.6x at 0.7%
SMOLVLA_INT8=59 OMP_NUM_THREADS=$CORES \
.venv-serve/bin/python serve/policy_server.py --model smolvla --model-dir build/smolvla --port 8080

# every group
IMPACT_INT8=63 OMP_NUM_THREADS=$CORES \
.venv-serve/bin/python serve/policy_server.py --model impact --model-dir build/impact --port 8080
```

## License

Apache 2.0, see [LICENSE](LICENSE).

## Acknowledgement

- [ACT](https://huggingface.co/papers/2304.13705) and [lerobot](https://github.com/huggingface/lerobot) - the reference implementations and the async-inference protocol
- [Octo](https://github.com/octo-models/octo) - the authoritative JAX model
- [TurboVLA](https://github.com/H-EmbodVis/TurboVLA) - the LIBERO checkpoints
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) (Chi et al. 2023)
- [TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine) - CPU ops for LLMs, non-ggml based
- [VAMP](https://github.com/KavrakiLab/vamp) - SIMD accelerator in the same robotics domain
- [stb_image](https://github.com/nothings/stb) - image loading
