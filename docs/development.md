# Development

Building the engine, running the tests, and what CI checks.

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
[benchmark report](benchmark/intel-core-ultra-9-285k.md).

## Continuous integration

The [workflow](../.github/workflows/build.yml) always checks every tracked Markdown
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
