# Results status

`2026-09-07-planner.json` is the unchanged JSON emitted by a read-only call to the installed `resource_plan.build_plan` for the pinned Qwen3.6 checkpoint, two GPUs, context4096 and one KV slot. It records a projection, not actual allocation, residency or performance. The plan's family-independent claims are analyzed in `../colibri_execution.md`.

`2026-09-07-device-attributes.json` is a read-only CUDA driver-attribute snapshot. It created no CUDA context and launched no workload. It confirms installed-device limits useful for the first primitive suite; it is not a throughput or thermal result.

No microbenchmark, inference timing, training result or best configuration has been generated in this project yet. Hardware observations and historical user measurements are separately labeled in `../hardware.md` and `../experiments/LOG.md`.
