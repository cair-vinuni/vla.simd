# Converter environments

Environments for converting checkpoints to GGUF. The published GGUFs need none of this.

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
outputs or remove APIs it uses. The [dependency audit](benchmark/intel-core-ultra9-285k.md#dependency-decisions)
records the tested versions and retained caps. Keep Octo in its own environment;
its NumPy 1.x requirement conflicts with modern LeRobot.
