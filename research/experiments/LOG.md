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

## T01 — guarded benchmark harness (local mocked acceptance)

- Date: 2026-09-08.
- Status: pass (CPU-only harness acceptance; no workload or CUDA initialization).
- Hypothesis: a standard-library guard can fail closed on injected unsafe telemetry while owning only its process group and recording durable cleanup.
- Exact change: added safe allowlisted snapshot hashing, durable result schema/validator, JSONL telemetry parser, host lock, guarded process-group cleanup, restoration record, dry-run matrix, and public fixed prompt fixture.
- Command: `python3 -m unittest discover -s tests -v`; result: 21 tests passed. Also ran `python3 scripts/run_matrix.py --dry-run`, `python3 scripts/guard.py --dry-run -- /bin/echo inert`, and `python3 -m py_compile scripts/snapshot.py scripts/guard.py scripts/run_matrix.py scripts/summarize.py`.
- Evidence: injected rising temperature, stale polling, fan-critical, device-error, subprocess failure, watchdog timeout, lock contention, failed restore, disconnect cleanup, and unrelated-process protection are covered by tests; dry-run reports `cuda_initialized: false`.
- Confounders: telemetry/power controls are fake providers in this task; no host telemetry, BMC, GPU, CUDA, model, SSH, production source, or production service was touched.
- Decision: T01 acceptance evidence is [recorded](../results/T01-acceptance.json). Next eligible task is T03 source-only audit/instrumentation; workload-bearing T02 remains dependent on T00 thermal evidence.

## T03 — Qwen family capability manifest and instrumentation contract

- Date: 2026-09-08.
- Status: blocked (runtime acceptance intentionally not executed).
- Hypothesis: a family-specific static manifest and strict direct-output parser prevent generic Colibri CUDA flags from being benchmarked as Qwen capabilities and make future timing/counter evidence unambiguous.
- Exact change: added a 43-setting Qwen3.6 capability manifest with one of `engine-read`, `launcher-only`, `backend-reached`, or `unsupported` for every examined setting; added a standard-library parser/counter contract and source-isolated instrumentation boundary proposal.
- Command: `python3 -m unittest discover -s tests -v`; result: 33 tests passed. Also ran `python3 -m py_compile scripts/colibri_t03.py` and `git diff --check`.
- Evidence: regression checks pin generic no-op controls (`CUDA_DENSE`, CUDA attention/sharding/pipeline, DRAFT, and CUDA MTP) to `unsupported`; simulated direct-execution records validate non-negative nanosecond durations, count/byte units, and avoid double-counting overlap intervals.
- What was measured: local parser and manifest behavior only. No Colibri source, CUDA context, GPU event, model, benchmark, SSH target, or production checkout was touched.
- Blocker: T03's runtime acceptance requires optional CUDA-event timing on a tiny guarded fixture after T00 passes. The user prohibited CUDA initialization/model execution in this assignment, and T00 has not unlocked that fixture. Instrumentation-off output equivalence likewise requires a separate experimental source worktree and CPU fixture; this assignment authorized only proposed patch boundaries.
- Decision: record `blocked`, not `pass`; preserve the acceptance gap rather than inferring runtime behavior from static tests. Next eligible action is T00 completion; after its gate and explicit authorization, finish T03's isolated experimental-source and guarded-fixture acceptance before T04.
