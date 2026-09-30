# Docker

Serving from a container instead of a local environment.

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
