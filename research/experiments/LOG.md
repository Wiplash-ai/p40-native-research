# Experiment log

## D000 — installed-source and idle-hardware discovery

- Date: 2026-09-07, around 22:59–23:05 UTC.
- Hypothesis: the supplied generic CUDA knobs describe Qwen3.6's installed execution path.
- Prediction: CUDA_DENSE/ATTN/PIPE alter major Qwen stages; 16.2 GB is a capacity reservation problem.
- Smallest test: read source and safetensors headers; invoke read-only resource planner; inspect nvidia-smi, NUMA and BMC telemetry without loading weights.
- Observation: generic knobs do not reach Qwen stages; expert payload exactly 16.232 decimal GB. Planner hints overstate family capability. Qwen serves serially with one KV slot. Actual expert CUDA path uses packed W4A32, no DP4A. Both GPUs NUMA0; SATA model disk. Repeated critical FAN2/FAN3 sensor events.
- Decision: reject knob sweep as originally proposed; implement family-aware capability audit and guarded measurement before kernels. Investigate dense/head and recurrence GPU execution. Resolve cooling before workload runs.
- Source commit: `12a5c464b5c1f8292d578c62458706bc32d6ac95`.
- qwen36 binary SHA256: `4df8e8dfc9b1295f3ede38c05adc08d2dde00ed50bda8daa064e53ff7bb84521`.
- CUDA object SHA256: `9e61f8a300c47ea8832b86151637af31508c28a5e394526708cc5cf8691b5573`.
- Model config SHA256: `7606939241b64e82308369b308af8abe85d24e70846704c41ed4e0b59687a674`.
- Evidence: [planner](../results/2026-09-07-planner.json), [execution analysis](../colibri_execution.md), [hardware](../hardware.md).
- GPU/kernel/inference benchmark outcome: **not run**. No measured best configuration or measured post-optimization bottleneck exists.

## Historical user observations — not controlled comparisons

| Configuration described by user | Output/time | Approximate rate | Limitation |
|---|---|---|---|
| CPU + NUMA | 219 tokens /702 s | 0.312 tok/s | Prompt/content differed; unclear timing boundaries |
| CPU without NUMA | 11 /66 s | 0.167 tok/s | Much shorter run; unclear policy/threads |
| Two P40 CUDA | 214 /143 s | 1.50 tok/s | No per-stage trace; low sampled utilization; heat-soak concern |

These motivate tests but do not establish a controlled 4.8× speedup or the best configuration.

## Required record for every future experiment

ID, status, hypothesis, prediction, exact change, source/binary/model/prompt identities, command and safe environment allowlist, CPU affinity/NUMA, GPU UUIDs, power cap, clocks, payload shapes and byte accounting, warm/cold cache state, temperature before/during/after, numerical checks, token/step counts, metrics, repetitions and dispersion, profiler applicability, quality comparison, confounders, decision, next experiment. Link raw stdout/stderr/telemetry/results rather than replacing them with a narrative.
