# Device benchmarks

Seven devices ran commit `7636baabd17191ad705260f46c1aace5f816b249` on 2026-09-30, and the Core
Ultra X7 358H ran `c60b286d53a1fe31752d5c98237e0a314d655729`, the merge of that work into `main`, on
2026-10-01; the two commits differ only in documentation. All used the protocol in
[TEMPLATE.md](TEMPLATE.md): the six policies below, each at its fastest measured thread count and
runtime settings, 5 warmup and 50 timed queries (20 when a query takes more than 3 s). Values are
median latency in ms for one complete action chunk; lower is better. INT8 is the W8A8 path; no
accuracy is measured in these reports.

| Model | Precision | [Pi 5](raspberry-pi-5.md) | [Snapdragon X](snapdragon-x.md) | [M4](apple-m4.md) | [Ryzen 5 5500](amd-ryzen-5-5500.md) | [i5-12400F](intel-core-i5-12400f.md) | [i7-14700F](intel-core-i7-14700f.md) | [i9-14900HX](intel-core-i9-14900hx.md) | [Ultra X7 358H](intel-core-ultra-x7-358h.md) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | FP32 | 1,251 | 170.6 | 89.4 | 170.3 | 139.4 | 119.4 | 99.6 | 135.4 |
| ACT | INT8 | 487.9 | 74.7 | 69.1 | n/a | 59.2 | 53.3 | 49.1 | 65.3 |
| IMPACT | FP32 | 1,316 | 191.2 | 92.2 | 170.8 | 140.3 | 113.4 | 97.3 | 136.2 |
| IMPACT | INT8 | 497.4 | 77.8 | 70.5 | n/a | 60.5 | 52.6 | 48.2 | 72.5 |
| SmolVLA | FP32 | 10,424 | 1,435 | 602.4 | 1,162 | 972.2 | 861.1 | 780.4 | 976.9 |
| SmolVLA | INT8 | 3,961 | 660.4 | 494.4 | n/a | 462.6 | 402.6 | 401.7 | 566.4 |
| Octo-Small | FP32 | 669.8 | 97.9 | 47.9 | 73.0 | 61.9 | 52.7 | 49.9 | 68.3 |
| Octo-Small | INT8 | 354.3 | 44.4 | 50.5 | n/a | 35.0 | 29.8 | 44.8 | 42.4 |
| TurboVLA | FP32 | 1,590 | 217.0 | 114.5 | 187.6 | 155.4 | 132.7 | 116.0 | 162.2 |
| Diffusion Policy | FP32 | 4,265 | 495.3 | 433.1 | 738.0 | 499.6 | 469.4 | 341.1 | 378.9 |
| Diffusion Policy | INT8 | 1,078 | 169.6 | 180.2 | n/a | 178.1 | 165.7 | 127.6 | 179.9 |

TurboVLA has no INT8 path, so it has no INT8 row; n/a marks the Ryzen 5 5500, which has no AVX-VNNI.
The Raspberry Pi 5 ran at its soft temperature limit; see its report. Each report lists the device,
build, model configurations, engine settings, memory, and the full thread sweep.

Reproduce a device with `tools/bench_sweep.py` from a Release build of that commit; the Snapdragon X
report also covers the Windows on Arm build and its register-tile selection.

The audit and parity report is [intel-core-ultra-9-285k.md](intel-core-ultra-9-285k.md).
