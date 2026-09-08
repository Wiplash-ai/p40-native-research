# Delegation brief

This is a staged research programme, not a single implementation ticket. Delegate only one task per agent/session and require the listed acceptance evidence before unlocking the next task. The experimental repository is `/home/jordanculver/Laboratory/p40-native-research`; production Colibri stays untouched at `/home/jordanculver/colibri_engine`.

## Current gate

The former quiet-mode fan cap was corrected to persistent 100% BMC-zone PWM on 2026-09-08; all chassis fans then reported roughly 2,000 RPM. This is a configuration correction, not a thermal acceptance result. GPU, model, CUDA-kernel, and training workloads remain blocked until T00 records five clean idle minutes, a stable sensor explanation, and any needed physical airflow follow-up. Do not silence sensor thresholds, clear the SEL, or issue raw BMC fan commands.

## Ready now: three bounded assignments

| Assignment | Suggested model | Scope | Must not do | Done when |
|---|---|---|---|---|
| T01 guarded harness | Luna | Implement the mock-tested result schema, snapshot, watchdog, telemetry parsing, cooldown cleanup and dry-run benchmark matrix in the research repo | SSH workload, CUDA init, production source edits | Tests cover critical fan, stale telemetry, rising heat, timeout, cleanup failure and unrelated-process protection; `--dry-run` creates no CUDA context |
| T00 thermal investigation | Luna, serially | Collect read-only BMC/GPU telemetry, correlate fan events and identify the physical header/module from official server documentation | Fan-control writes, threshold changes, SEL clearing, any load | `T00-thermal-investigation.md` says pass or `blocked_thermal`, with timestamps and the specific physical follow-up if blocked |
| T03 capability/timing design | Terra | Convert the Colibri source audit into a machine-readable capability manifest; add parser/counter tests and propose isolated instrumentation patch boundaries | Benchmarking, flag sweeps, editing production checkout | Every listed Qwen flag is classified engine-read/launcher-only/backend-reached/unsupported and mock timing/counter tests pass |

T00 and T01 can run in parallel; T03 may proceed in parallel only if it changes isolated manifest/test files. Do not have multiple agents modify the same scripts or run any simultaneous server workloads.

## Unlock sequence

1. **T00 + T01 pass** → Terra performs **T02 P01–P04** only: device metadata, vector memory controls, FP32 baseline GEMM/matvec, and a DP4A correctness/canary test. The guarded runner owns all GPU processes.
2. **T02 + T03 pass** → **T04**, fixed Qwen 512-token baseline. Compare only source-reached controls: CPU/NUMA placement, GPU count, packed W4 and dual projection. Do not test `CUDA_DENSE`, CUDA attention, CUDA pipe, MTP, or oversized expert-cache settings: the audited Qwen engine does not consume them.
3. **T04 identifies the dominant stage** → **T05** moves precisely one CPU stage to GPU, with CPU fallback and equivalence tests. **T06** separately tests W4A8 DP4A versus lookup-table arithmetic.
4. Only a surviving primitive/architecture result unlocks **T07–T10**: device-resident recurrence, tiny learning experiments, real multi-sequence scheduling, then multi-P40 scale decisions.

## Required handoff format

Every assignment ends with:

```text
Task ID:
Commit(s):
Changed paths:
Exact commands and environment:
Tests and their output:
Raw telemetry/result paths:
Acceptance decision: pass / fail / blocked
What was measured (not inferred):
Unresolved risks:
Next eligible task:
```

An unmeasured theoretical speedup, a successful compile, or a clean `nvidia-smi` snapshot is not an acceptance result. Every benchmark requires its configuration, code hash, seed, telemetry and cooldown record. Full task specifications and copy-paste prompts are in [EXECUTION_PLAN.md](EXECUTION_PLAN.md).
