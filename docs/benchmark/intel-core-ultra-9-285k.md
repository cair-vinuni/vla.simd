# Inference benchmarks and validation, September 2026

The audit covered checkpoint parsing and conversion, model preprocessing, numerical
operators and schedulers, server sessions, dependencies, and packaging. Tests ran
locally on an Intel Core Ultra 9 285K with GCC 13.3 and Python 3.12.3. Downloads,
reference checkouts, and build environments were kept outside the repository.

## Changes and regression coverage

| Area | Failure addressed | Check |
| --- | --- | --- |
| GGUF reader | Overflowing shapes and byte counts, truncated arrays and padding, duplicate metadata, invalid offsets, stale state on reopen | `test_gguf` |
| Embedded files | Path traversal, colliding file names, misordered tensor parts | `test_gguf`, Python round trip |
| Extraction | Destination symlinks could redirect writes; nested directories were incomplete | Python extraction test against `vla-simd-gguf` |
| Loading | Concurrent loads could consume the same mounted binary arena | Concurrent mount regression |
| JSON | Invalid number syntax, control characters, malformed Unicode escapes, reused values retaining data | `test_gguf` |
| BF16 conversion | Some NaN payloads rounded into infinity or finite values | C++ and Python bit-pattern checks |
| Safetensors | Invalid headers, shapes, or offsets could select incorrect data | Python malformed-input tests |
| Diffusion | One-training-step linear schedules divided by zero; invalid settings were accepted | `test_math`, Diffusers reference trajectories |
| Image preprocessing | Negative output dimensions became huge unsigned write counts | `test_math` |
| Tokenizers | Reloading kept vocabulary and merge entries from the old model | `test_math` |
| Server | Reset raced with pending requests; failed inference suppressed retries; cache mutation raced with token lookup | Protocol tests using LeRobot wire types and a fake engine |
| Input/output validation | Nonfinite observations or actions could pass through inference | Python boundary checks; numerical tests reject NaN and infinity |
| INT8 | Tiny finite inputs underflowed row scales; oversized reductions could overflow VNNI accumulators | Quantization edge cases and conservative reduction bounds |
| BF16 reload | Reinitializing an INT8 layer retained stale quantized weights | INT8-to-BF16 regression, including raw BF16 |
| Metadata | Invalid numbers, dimensions, normalization statistics, and convolution geometry reached model setup | Loader regressions and public-checkpoint loading |
| Benchmarking | Missing tail latency, memory, and configuration made results harder to compare | Benchmark JSON regression |
| Staging | Interrupted or concurrent population could publish partial sidecars; replaced checkpoints reused stale metadata | Concurrent, interrupted, and replacement tests |

Mounts are scoped to the loading thread. Nested loaders on that thread share one
parse; simultaneous loads on other threads own their arenas. A `Mount` and its
`InFile` reads must stay on the same thread. The public model loaders already do.

`vla-simd-gguf extract` creates files exclusively and refuses symlinks in the
output directory or its descendants. Use a fresh directory for another extraction.
An I/O failure can leave partial output; the tool reports failure and does not
overwrite existing files on retry.

The numerical comparison helper now fails on nonfinite values and mismatched
shapes. Previously, NaN comparisons could leave the recorded worst error at zero.
Existing operator tests cover matrix products, attention, normalization,
activations, convolution, and quantized paths. Passing tests establish the tested
contracts, not correctness for every possible shape or floating-point input.

## Loader contracts and numerical edge cases

The numerical pass reproduced two additional failures:

- INT8 row scales could underflow for tiny finite values. A zero scale or an
  overflowing reciprocal sent invalid values into integer conversion. The scale
  now has a minimum of `FLT_MIN`, so its reciprocal stays finite. Inputs below
  that quantization resolution can round to zero. Ordinary activation scales
  and checkpoint weights keep their existing arithmetic.
- Reinitializing an INT8 `Linear` from BF16 weights left its old quantized
  weights active on the raw-BF16 and NEON paths. BF16 initialization now clears
  that state before selecting a representation.

The new regressions fail against the pre-change library. They cover subnormal
inputs, SIMD tails, quantization padding, the symmetric integer range, and
INT8-to-BF16 reinitialization. INT8 selection also limits the reduction width so
AVX-VNNI's unsigned intermediate dot product fits in a signed 32-bit accumulator.
Larger layers retain their floating-point path.

Model loaders now reject fractional or out-of-range integer metadata before
conversion, check tensor products against their signed integer indexing limit,
and validate attention dimensions, normalization parameters, and convolution
geometry before packing. SmolVLA checks mixed FP32/BF16 file lengths before
allocating its weight arrays. Diffusion rejects unknown scheduler names,
malformed boolean fields, invalid group sizes, and incompatible UNet horizons.
Normalization statistics must be finite, with nonnegative standard deviations
and ordered min/max bounds. Reloading clears model-dependent language and
position caches.

These checks cover the supported model layouts. They are not a general-purpose
checkpoint sandbox or a claim that every malformed file has been tested.

## Reference mathematics

| Reference | Inputs | Result |
| --- | --- | --- |
| ACT, public trained checkpoint | Three seeded image/state observations; normalized and physical actions | Maximum normalized absolute error `7.75e-7` |
| IMPACT, public trained checkpoint | Three seeded image/state observations and T5 language conditioning; normalized and physical actions | Maximum normalized absolute error `1.68e-6` |
| TurboVLA, public LIBERO checkpoint | Three seeded observations, alternating instructions, physical action mapping | Maximum normalized absolute error `7.45e-7` with Transformers 4.57.1 |
| SmolVLA, public trained checkpoint | Three seeded observations, alternating instructions, tokenizer, square and 480×640 inputs, matched initial noise | Maximum normalized absolute error `1.22e-6` |
| Diffusion Policy | Random weights, two cameras, two observation steps, 64×64 images, horizon 16, UNet widths 32/64/128, seven sampling steps | Complete DDPM and DDIM outputs match within `3e-4` absolute/relative tolerance |
| DDPM/DDIM schedulers | Linear, scaled-linear, cosine; 1/1, 100/7, 100/10, 100/100 train/inference steps; clipping on/off | All 48 trajectories pass against Diffusers 0.39.0 and 0.40.0 |

The stochastic comparisons supply identical prior and per-step noise to both
implementations. Matching a seed across different random-number generators would
not establish parity. The Diffusion Policy test uses untrained weights and does
not establish task success or trained-checkpoint quality.

Reproduce the small reference tests in the LeRobot converter environment:

```sh
OMP_NUM_THREADS=4 VLA_TEST_BUILD=build .lerobot/bin/python -m unittest discover -s tests -p test_reference.py -v
```

Full trained-checkpoint comparisons use local reference checkpoints and matching
GGUF files:

```sh
OMP_NUM_THREADS=4 .lerobot/bin/python tools/check_model_parity.py /path/to/act /path/to/act.gguf --build build
OMP_NUM_THREADS=4 .lerobot/bin/python tools/check_model_parity.py /path/to/smolvla /path/to/smolvla.gguf --model smolvla --build build
OMP_NUM_THREADS=4 .impact/bin/python tools/check_model_parity.py /path/to/impact /path/to/impact.gguf --model impact --build build
OMP_NUM_THREADS=4 .turbovla/bin/python tools/check_model_parity.py /path/to/turbovla_libero.pth /path/to/turbovla.gguf --model turbovla --reference-code /path/to/TurboVLA --task 'pick up the tape' --build build
```

The IMPACT check downloads its frozen T5-small reference and tokenizer. TurboVLA
needs BERT tokenizer/config files; the released checkpoint supplies model weights.
ACT conversion was also repeated with the updated converter environment.

Reference revisions used:

| Artifact | Repository | Revision |
| --- | --- | --- |
| ACT reference | `khanhnd61/act-matched_so101-multi-task-clean` | `104e72d4b8dd8e69de2b25f31aacadb2716d68b7` |
| ACT GGUF | `khanhnd61/act-so101-multi-task-gguf` | `3a44af95d6b150ea1d8a7be1ada5a94d37cdf464` |
| IMPACT reference | `khanhnd61/impact_so101-multi-task-clean` | `0f7e1bdbb4e922d3c55f26d9af8f94d4c3d23905` |
| IMPACT GGUF | `khanhnd61/impact-so101-multi-task-gguf` | `dc9943947c2341b9f789ebb0bb39771357c5c912` |
| SmolVLA reference | `khanhnd61/smolvla_so101-multi-task-clean` | `543faa4b2a621f611b672caf3a49130efb58bec0` |
| SmolVLA GGUF | `khanhnd61/smolvla-so101-multi-task-gguf` | `61798ce16b61347c9eaffe4209af3d435c763a3f` |
| Octo reference artifact | `khanhnd61/octo-small_so101-multi-task-clean` | `55872b36da31f14e36cfea0c7fd0399d9e8e1a96` |
| Octo GGUF | `khanhnd61/octo-small-so101-multi-task-gguf` | `be198aa809f41b942bf0b032d970612a60bda0f9` |
| TurboVLA weights | `H-EmbodVis/TurboVLA` | `cb5300544693013164c4bb251a13036002a55c81` |
| TurboVLA code | `H-EmbodVis/TurboVLA` on GitHub | `b29ab1420baa5c663ec935df513f2012430beb67` |

The reference tool now accepts `--model smolvla`. It compares the tokenizer,
matched-noise action trajectories, and normalization using both square inputs
and 480×640 inputs that require resizing and padding. The reference configuration
contains an empty `prune_vlm_layers` field absent from current LeRobot; the tool
accepts that empty field but refuses nonempty pruning when the reference lacks
support. The reference runs the checkpoint's BF16 tower values in FP32, matching
the runtime's numerical target.

The public PyTorch Octo checkpoint records `n_inference_samples=8`; the current
runtime exposes a single matched-noise trajectory. A single-trajectory parity
check must explicitly use one reference sample. It must also use the matching
PyTorch implementation: the original JAX Octo differs in weight standardization,
normalization epsilon, and GELU. The installed serving fork and its public tree
lack the Octo policy, so full trained-model Octo parity remains unverified.

No robot rollout or new quantization-accuracy claim is made here.

## Packing performance

`tools/_gguf.py` streams tensor data in 1 MiB chunks, builds a mutable header, and
publishes through a unique temporary file followed by atomic replacement. It no
longer imports NumPy. Layout validation finishes before the destination changes.

Seven subprocess runs packed four 64 MiB zero-filled files plus a small text
config on the local `/tmp` filesystem:

| Metric, median | Before | After |
| --- | --- | --- |
| Packing wall time | 170.48 ms | 61.30 ms |
| Process peak RSS | 294,088 KiB | 19,424 KiB |

This is a 2.78× packing speedup and 93.4% lower process peak memory. Every output
had SHA-256 `31efe827e9f1d803d016dcf88bff3a3c68acb4a9b7a90e5fdf201d0a9da6d120`.
These are warm local-filesystem measurements with sparse zero-filled inputs, no
`fsync`, and no durable-storage throughput claim. They measure checkpoint writing,
not model inference latency.

Reproduce each sample in a fresh process (NumPy is needed by the old module):

```sh
git show 7354b6d:tools/_gguf.py > /tmp/vla-gguf-before.py
for i in 1 2 3 4 5 6 7; do
  .lerobot/bin/python tools/bench_pack.py --module /tmp/vla-gguf-before.py
  .lerobot/bin/python tools/bench_pack.py
done
```

`peak_rss` uses `resource.getrusage`: KiB on Linux, bytes on macOS. The existing ISA
dispatch, tiling controls, and backend defaults remain available.

## Shared INT8 attention inputs

Attention now quantizes shared Q/K/V inputs once per call, using caller-owned
scratch buffers and the existing prequantized GEMM. Independent inputs keep
separate quantization. Regression tests cover Q/K sharing, K/V sharing, all three
sharing, and unequal query/key sequence lengths; results must be bit-identical.
The active residual paths already use persistent scratch, so no residual-buffer
refactor was needed.

Two paired runs compared the parent revision `dce1b93` with this change on the
local Intel host. Each process loaded both libraries, warmed each four times,
checked exact action equality, then alternated five blocks of five queries per
library. Inputs and noise were fixed; `OCTO_INT8=1`, `OMP_NUM_THREADS=4`,
`OMP_PROC_BIND=true`, and CPU affinity `0-3` were used. Timing covers a complete
`engine.predict` call with a warm language cache.

| Octo run | Before median | After median | Before block range | After block range |
| --- | --- | --- | --- | --- |
| Initial | 59.05 ms | 57.82 ms | 58.53–59.13 ms | 56.97–58.13 ms |
| Final build | 59.85 ms | 58.61 ms | 58.31–71.24 ms | 57.40–60.15 ms |

The reported median is the median of five block medians. Both runs show about
2.1% lower latency. The final run has overlapping ranges and a baseline outlier;
this is a modest local result, not a cross-hardware guarantee. ACT and SmolVLA
also produced identical actions, but their timing variance was too large to
support a speedup claim. Native Pi and Apple measurements remain outstanding.

For a deployment baseline, run the existing benchmark against each build with
identical model files, environment, CPU affinity, and query counts:

```sh
OMP_PROC_BIND=true taskset -c 0-3 .serve/bin/python -m vla_simd.policy_server \
  --model octo --model-dir /path/to/octo.gguf --cams front,wrist \
  --lib /path/to/build/libvla_simd_octo.so --threads 4 --int8 1 --bench 50 --json
```

This command measures one library per process; it does not reproduce the paired
ordering above. JSON includes p95 latency, process peak RSS in bytes, the supplied
checkpoint path, model configuration, platform, NumPy version, and runtime
settings. RSS includes model loading and Python. The checkpoint field is a path,
not a content hash; record the artifact revision alongside the result.

## Dependency decisions

PyPI and GitHub were rechecked on September 27. Every dependency group resolved
separately, and the existing PR already selects the newest compatible stable
versions. No additional dependency edits were needed. Converter
environments remain separate; combining the legacy Octo and modern LeRobot
requirements cannot resolve.

| Environment | Versions exercised |
| --- | --- |
| Core / TurboVLA | NumPy 2.5.3, PyTorch 2.14.0 CPU, torchvision 0.29.0 CPU, safetensors 0.8.0, Transformers 4.57.1, Hub 0.36.2 |
| Serve / ACT / IMPACT / Diffusion | LeRobot fork 0.6.2 at `fa12de51f15836733c7ac55e973595f671d53e4f`, NumPy 2.2.6, PyTorch 2.11.0 CPU, torchvision 0.26.0 CPU, Transformers 5.5.4, Diffusers 0.39.0, Hub 1.33.0 |
| Build | scikit-build-core 1.1.0; Docker uv 0.12.19; checkout action v7.0.1 |
| Octo | Existing JAX 0.4.20 / Flax 0.7.5 / NumPy 1.24.3 stack; dependency resolution on Python 3.11 |

Available releases include Transformers 5.17.0, Hub 2.0.0, Diffusers 0.40.0,
PyTorch 2.14.0, and NumPy 2.5.3.

LeRobot's constraints prevent installing the newest PyTorch, NumPy, and Hub in
its environment. The NumPy 2.2.6 floor also preserves the package's Python 3.10
support; newer Python versions can resolve newer NumPy in the core environment.
The fork update changes documentation only relative to the previous pin.

TurboVLA requires `transformers>=4.57.1,<4.57.2`. Version 4.57.6 loads and exports
successfully but changes the DINOv3 hidden-state normalization consumed by the
upstream encoder: the tested action error reaches 0.436. Version 5.17.0 fails
construction because the upstream BERT wrapper calls the removed
`get_extended_attention_mask`. Keeping the cap preserves the trained model's
computation. A future migration needs an explicit encoder change and parity tests.
The legacy Octo stack follows the constraints of its
[upstream requirements](https://github.com/octo-models/octo/blob/main/requirements.txt).

## Research cross-checks and validation limits

The [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
informed alignment and metadata validation. The
[Diffusers schedulers](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers)
are the executable reference for timestep spacing, clipping, and posterior math.

[FlashAttention](https://arxiv.org/abs/2205.14135) motivates reducing attention
memory traffic; its GPU results do not predict gains for these CPU kernels.
[T-MAC](https://github.com/microsoft/T-MAC) explores lookup-table computation for
low-bit weights. Adopting that approach here would require a new weight format,
representative VLA accuracy measurements, and backend benchmarks. Neither paper
justifies changing the current numerical path without those checks.

The final local checks pass all three C++ suites in Release, scalar, and
address/undefined/float-cast-overflow sanitizer builds, with warnings as errors.
Raw BF16, forced Zen dispatch, and one/eight-thread operator runs also pass.
All 20 Python tests pass without skips in the reference environment. ACT,
IMPACT, SmolVLA, and TurboVLA trained-checkpoint comparisons pass at the errors
reported above. Diffusion uses the synthetic reference model described above.

The installed CMake consumer links and runs. Wheel and Docker checks load all
six shared libraries and verify the ABI. The installed wheel passes 13 core tests
and skips the CMake-only extraction test; the full reference environment runs it.
Docker passes all four protocol tests. Markdown lint and local links
pass. The earlier CI audit also passed actionlint, ShellCheck, and 16 change-filter
cases. CI still covers ARM NEON and Apple Accelerate; those native runs, AMD
hardware, robot rollouts, and a remote CI run were unavailable locally.

The parser tests target reproduced failures; they are not an exhaustive fuzzing
campaign or a security certification for arbitrary checkpoints. The model-shape
checks extend the GGUF reader's validation, but do not exhaust
all malformed metadata or weights.
Staging uses filesystem fingerprints rather than content hashes, and assumes the
checkpoint and sidecars are stable while loading. Its cache directory should be
owned by the serving user.

The server serializes session changes, observation adaptation, and inference on
one session lock. A new observation waits while prediction is running. This
keeps resets and history consistent, but its effect on camera throughput needs a
robot-client measurement before further concurrency changes.

## Next improvements by expected value

The fastest route to broader adoption is a reproducible model contract: pinned
checkpoint, preprocessing, matched-noise action error, latency percentiles,
memory, and a working client invocation. The SmolVLA parity check and richer
benchmark JSON cover part of that work.
[llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)
is a useful reporting reference. Robot measurements also need observation-to-action
age, executed chunk length, and task success; synthetic throughput cannot answer
those questions.

| Priority | Work | Evidence needed |
| --- | --- | --- |
| 1 | Recover the matching PyTorch Octo implementation and add single-trajectory parity | Matched preprocessing/noise and both normalized and physical action error |
| 2 | Publish comparable policy benchmarks on x86, Pi, and Apple | Fixed checkpoint revisions, affinity, thread count, tail latency, RSS, and repeated runs |
| 3 | Measure observation age with the existing async/RTC paths | Robot-client traces and task success before changing concurrency or chunk schedules |
| 4 | Extend fused INT8 epilogues or combine projection calls | A measured memory or dispatch bottleneck and lower whole-policy latency |
| 5 | Evaluate additional ARM or grouped low-bit kernels | Native measurements, explicit packing/ISA contracts, and action-quality validation |

[LeRobot async inference](https://huggingface.co/docs/lerobot/async) and
[real-time chunking](https://huggingface.co/docs/lerobot/rtc) provide deployment
references for item 3. [KleidiAI](https://github.com/ARM-software/kleidiai) is an ARM
microkernel comparison target before writing another backend. T-MAC's LLM results
do not establish VLA accuracy or latency. Streaming-attention changes likewise
need evidence of a remaining score-buffer or bandwidth bottleneck beyond the
existing tiled paths. No new kernel dependency or weight format is warranted by
the measurements collected here.
