# Inference audit, September 2026

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

## Reference mathematics

| Reference | Inputs | Result |
| --- | --- | --- |
| ACT, public trained checkpoint | Three seeded image/state observations; normalized and physical actions | Maximum normalized absolute error `7.75e-7` |
| IMPACT, public trained checkpoint | Three seeded image/state observations and T5 language conditioning; normalized and physical actions | Maximum normalized absolute error `1.69e-6` |
| TurboVLA, public LIBERO checkpoint | Three seeded observations, alternating instructions, physical action mapping | Maximum normalized absolute error `4.92e-7` with Transformers 4.57.1 |
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
| SmolVLA GGUF | `khanhnd61/smolvla-so101-multi-task-gguf` | `61798ce16b61347c9eaffe4209af3d435c763a3f` |
| Octo GGUF | `khanhnd61/octo-small-so101-multi-task-gguf` | `be198aa809f41b942bf0b032d970612a60bda0f9` |
| TurboVLA weights | `H-EmbodVis/TurboVLA` | `cb5300544693013164c4bb251a13036002a55c81` |
| TurboVLA code | `H-EmbodVis/TurboVLA` on GitHub | `b29ab1420baa5c663ec935df513f2012430beb67` |

SmolVLA and Octo public bundles loaded and produced finite inference outputs. Their
full trained-model reference parity was not repeated in this audit. The installed
LeRobot fork does not include an Octo policy implementation. No robot rollout or
new quantization-accuracy claim is made here.

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

`peak_rss` uses `resource.getrusage`: KiB on Linux, bytes on macOS. There is no
retained kernel tuning change. The existing ISA dispatch, tiling controls, and
backend defaults remain available.

## Dependency decisions

Ranges select current stable versions within each model's constraints. Converter
environments remain separate; combining the legacy Octo and modern LeRobot
requirements cannot resolve.

| Environment | Versions exercised |
| --- | --- |
| Core / TurboVLA | NumPy 2.5.3, PyTorch 2.14.0 CPU, torchvision 0.29.0 CPU, safetensors 0.8.0, Transformers 4.57.1, Hub 0.36.2 |
| Serve / ACT / IMPACT / Diffusion | LeRobot fork 0.6.2 at `fa12de51f15836733c7ac55e973595f671d53e4f`, NumPy 2.2.6, PyTorch 2.11.0 CPU, torchvision 0.26.0 CPU, Transformers 5.5.4, Diffusers 0.39.0, Hub 1.33.0 |
| Build | scikit-build-core 1.1.0; Docker uv 0.12.19; checkout action v7.0.1 |
| Octo | Existing JAX 0.4.20 / Flax 0.7.5 / NumPy 1.24.3 stack; dependency resolution on Python 3.11 |

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

During the initial audit, all three C++ suites passed in Release, scalar, and
sanitizer builds with warnings as errors. Forced Zen dispatch and one/eight-thread operator runs also
passed. All 16 Python tests passed in the reference environment; the minimal
wheel environment passed with six optional tests skipped. Each of the six shared
libraries loaded from the wheel and Docker image and exported the ABI symbol.
The installed CMake consumer linked and ran, all three Docker protocol tests
passed, and every dependency group resolved separately. CI retains ARM NEON and
Apple Accelerate coverage and adds a scalar row and Python/protocol regressions.

The [CI follow-up](../README.md#continuous-integration) passed all 19 Python tests
without skips in a fresh environment using the workflow's install command. It
also passed Release, scalar, and sanitizer tests, the installed CMake consumer,
and wheel and Docker package checks. The new regressions cover atomic checkpoint
publication, integer metadata bounds, tensor-part collisions, invalid engine
outputs, and observation timing. CI now requires the complete reference suite
and tests installed packages in isolated Python mode.

All three Markdown files passed lint and local-link checks; actionlint and
ShellCheck passed for the workflow. The change filter passed 16 cases covering
documentation, code, deletions, renames, and missing base revisions. A local Docker
rebuild reused the serving-dependency layer. GitHub CI runtime and remote cache
reuse have not been measured. Native ARM, Apple, and AMD hardware runs were
unavailable locally. No remote CI run was triggered.

The parser tests target reproduced failures; they are not an exhaustive fuzzing
campaign or a security certification for arbitrary checkpoints. Model-specific
metadata loaders still have less defensive shape validation than the GGUF reader.
Staging uses filesystem fingerprints rather than content hashes, and assumes the
checkpoint and sidecars are stable while loading. Its cache directory should be
owned by the serving user.

The server serializes session changes, observation adaptation, and inference on
one session lock. A new observation waits while prediction is running. This
keeps resets and history consistent, but its effect on camera throughput needs a
robot-client measurement before further concurrency changes.
