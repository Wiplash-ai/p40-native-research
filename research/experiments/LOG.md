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

## T02 / P01 — first guarded device-copy canary

- Date: 2026-09-08.
- Status: pass (one-second GPU0 canary only; 5- and 10-second gates remain).
- Hypothesis: the corrected full-speed fan policy and server-side process/power guard can execute a bounded P40 memory primitive without heat soak, retained VRAM, fresh BMC fault, or lost cleanup state.
- Prediction: a 64 MiB FP32 device copy at a 125 W cap stays well below the 65°C abort threshold and returns both devices to <=40°C through a five-minute cooldown.
- Exact change: added an isolated `sm_61` P01/P03 benchmark binary, root-owned BMC serialization helper, forced-command canary executor, temporary 125 W cap/restore path, and durable server result persistence.
- Result: P01 GPU0 copied two 64 MiB arrays for 2,176 iterations in 1,004.7 ms, reporting 290.691 GB/s algorithmic read+write bandwidth and exact copy verification. GPU0 was 35°C at start and 34–36°C during the five-minute cooldown; both GPUs finished at 0 MiB and 250 W, all eight fans read 2000–2100 RPM, and no new critical SEL event appeared.
- Evidence: [P01 record](../results/T02-P01-gpu0-64m-1s.md) and [raw durable JSON](../results/raw/T02-P01-gpu0-64m-1s-5c7dac13.json).
- Decision: do not extrapolate this to Qwen or peak bandwidth. Advance only to 5- then 10-second P01 canaries under the same 125 W policy; P03 follows only if both cool-down gates pass.

## T02 / P01 — five-second guarded device-copy canary

- Date: 2026-09-08.
- Status: pass (five-second GPU0 gate; ten-second gate remains).
- Hypothesis: five seconds of sustained memory traffic at the same conservative cap remains thermally stable after a full heat-soak hold.
- Prediction: the 64 MiB P01 copy remains below the 65°C abort threshold, leaves no GPU allocation, introduces no fresh fan-critical SEL event, and restores GPU0 to its original 250 W cap.
- Exact command: `p40_bench --primitive P01-copy --gpu 0 --bytes 67108864 --duration 5 --memory-cap-mib 256 --seed 1`, invoked only by the forced-command server guard.
- Result: 10,848 iterations in 5,009.64 ms; 290.638 GB/s algorithmic read+write bandwidth; exact copy check passed. GPU0 started at 35°C, reached a sampled 39°C during the work, then remained at 36°C through the five-minute cooldown. Both GPUs ended empty and at 250 W; all eight fans were 2000–2100 RPM; no new critical SEL record occurred.
- Evidence: [P01 five-second record](../results/T02-P01-gpu0-64m-5s.md) and [raw durable JSON](../results/raw/T02-P01-gpu0-64m-5s-5d5c8c2e.json).
- Decision: bandwidth is within 0.02% of the one-second result. Advance only to the identical 10-second P01 gate under the same 125 W policy; do not infer model-inference performance.

## T02 / P01 — ten-second guarded device-copy canary

- Date: 2026-09-08.
- Status: pass (P01 duration-gate sequence complete).
- Hypothesis: a final ten-second, 125 W saturated memory-traffic workload remains thermally safe through the full server-controlled cooldown.
- Prediction: P01 stays below its 65°C abort threshold, produces no fresh fan-critical event, leaves no allocation, and restores the original GPU power limit.
- Exact command: `p40_bench --primitive P01-copy --gpu 0 --bytes 67108864 --duration 10 --memory-cap-mib 256 --seed 1`, invoked only by the forced-command server guard.
- Result: 21,664 iterations in 10,005.1 ms; 290.620 GB/s algorithmic read+write bandwidth; exact copy check passed. GPU0 began at 35°C, reached a sampled 41°C peak, and was 36°C after cooldown. Both GPUs ended empty and at 250 W; all eight fans were 2000–2100 RPM; no new critical SEL event occurred.
- Evidence: [P01 ten-second record](../results/T02-P01-gpu0-64m-10s.md) and [raw durable JSON](../results/raw/T02-P01-gpu0-64m-10s-37f975ae.json).
- Tooling correction: the local client timed out at 380 seconds while the server completed its valid final BMC checks; increased its display timeout to 480 seconds. This does not alter server guard limits.
- Decision: P01 is stable only for this bounded, 125 W primitive. Begin P03 DP4A GEMV at one second under the same guard; do not load Colibri or extrapolate to inference yet.

## T02 / P03 — first guarded Pascal DP4A GEMV canary

- Date: 2026-09-08.
- Status: pass (correctness and one-second thermal gate only).
- Hypothesis: an explicit `sm_61` packed INT8 DP4A GEMV can run correctly and safely on GPU0 under the same bounded 125 W guard.
- Prediction: GPU result exactly matches a host `int64` reference; temperature stays below 65°C, no fresh BMC critical event occurs, memory is released, and the original power cap is restored.
- Exact command: `p40_bench --primitive P03-dp4a-gemv --gpu 0 --bytes 67108864 --duration 1 --memory-cap-mib 256 --seed 1`, invoked only by the forced-command server guard.
- Result: 512×2048 GEMV, 84,128 iterations in 982.141 ms, 179.637 INT8 GOP/s under multiply-plus-accumulate accounting, exact reference match. GPU0 was 36°C at start and a sampled 37°C peak; it returned to 35–36°C through cooldown. Both GPUs ended empty and at 250 W; fans stayed 2000–2100 RPM; no new critical SEL occurred.
- Evidence: [P03 record](../results/T02-P03-gpu0-1s.md) and [raw durable JSON](../results/raw/T02-P03-gpu0-1s-c5cf3200.json). Source SASS inspection directly identified `IDP.4A.S8.S8`.
- Interpretation: this fixed small-GEMV kernel is deliberately a thermal/correctness baseline. Two DP4A loop iterations per thread, per-output reduction overhead, and poor input reuse explain why 179.637 GOP/s cannot be compared with large-GEMM theoretical peaks.
- Decision: preserve P03 as control. Design a bounded, tiled comparison with exact reference and byte accounting before expanding the canary allowlist. Do not infer Qwen or Kimi performance from this result.

## T03 — asynchronous Qwen expert-event instrumentation build

- Date: 2026-09-08.
- Status: build pass; runtime measurement pending.
- Hypothesis: Qwen's actual async expert issue/take path needs its own CUDA-event accounting; the prior synchronous-wrapper instrumentation cannot reveal expert H2D, kernel, and D2H time in Qwen decode.
- Exact change: experimental-only patch adds four persistent events per device to bracket issue upload, kernels, and download, then records their elapsed time only after the existing take-side stream synchronization. `COLI_CUDA_PROFILE=1` remains opt-in; profile failures disable only metrics, never expert execution or fallback.
- Validation: generated unified diff validated against the pinned remote source; patch applied only to `/home/jordanculver/p40-native-research/colibri-engine`; `make -C c qwen36 CUDA=1 CUDA_ARCH=sm_61 NVCC=/usr/bin/nvcc` passed. The experimental binary exits with expected launcher guidance when no model is supplied. No model, CUDA context, or inference workload was run.
- Evidence: [build record](../results/T03-async-profile-build.md) and [patch](../../patches/0001-qwen36-async-expert-profile.patch).
- Decision: prepare a separately allowlisted, short guarded model canary. Do not classify event timing as working until a model run produces nonzero timings and completes all thermal/cleanup checks.

## T04A — fixed Qwen model-canary guard (local mocked acceptance)

- Date: 2026-09-08.
- Status: pass for source-only guard acceptance; no CUDA, model load, SSH, or server change occurred.
- Hypothesis: a separate forced-command executor can make the first Qwen measurement narrow enough to preserve the thermal and recovery guarantees used by P01/P03.
- Exact change: added `remote/p40-qwen-canary-guard.py`. It admits only a `dry_run` boolean; pins the experimental Qwen binary, Kreuzzelg snapshot, public prompt, GPUs 0 and 1, 16 output tokens, an 1,800-second watchdog, a 125 W cap per GPU, five-minute cooldown, and a clean `env -i` profile. It captures engine stdout/stderr to root-owned files instead of PIPEs, records hashes plus a bounded tail, and always attempts both cap restorations.
- Validation: `python3 -m unittest discover -s tests -v` passed 51 tests and `git diff --check` passed. Tests cover no-CUDA dry run, no arbitrary command surface, exact pinned argv, both-GPU cap/restore record, hot-start rejection before process creation, extensionless installed-base-guard loading, and new fan-critical-event process-group termination plus restoration.
- Evidence: [source-only acceptance](../results/T04A-model-guard-acceptance.json).
- Confounders: the model binary, model snapshot, GPU state, BMC state, process, and cooldown were mocked. This is not a Qwen benchmark and establishes no runtime timing or performance result.
- Decision: deploy the guard under a new restricted SSH key, first run its dry-run path, inspect the fixed `env -i` binary loader path without a model, then execute exactly one 16-output dual-P40 125 W canary only if live preflight is cool and idle.

## T04 — first guarded Qwen decode canary

- Date: 2026-09-08.
- Status: pass for the fixed 16-output canary; not yet a 512-output control or a best-setting result.
- Hypothesis: the real Qwen async expert-event counters work on Pascal and a short, fully resident two-P40 decode can run through a mandatory cooldown without thermal or fan faults.
- Exact profile: experimental binary SHA-256 `d7e8a284a4f864faddd4df6c16c1fbd3c8176222375fb23725af53c453f69051`; pinned Kreuzzelg snapshot; fixed 15-token public prompt; 16 outputs; GPUs 0,1; `CUDA_EXPERT_GB=auto`; `COLI_DENSE_I8=1`; `COLI_TIMERS=1`; `COLI_CUDA_PROFILE=1`; 125 W per GPU.
- Result: TTFT 2.60 s. Fifteen decode steps took 10.113 s (1.48 tok/s); engine-reported end-to-end output rate was 1.26 tok/s (12.7 s for 16). Real group counters were H2D 62 ms, kernels 934 ms, D2H 25 ms across 2,397 groups/9,600 experts. Decode phase wall time was led by DeltaNet at 354.79 ms/token and LM head at 156.84 ms/token; MoE total was 71.71 ms/token.
- Safety: GPU peaks were 42°C/43°C and 7,895 MiB/card; every fan remained 2,000–2,100 RPM; no fresh fan-critical SEL appeared; GPUs were empty after execution and the guard restored both 250 W limits after its five-minute cooldown.
- Decision: the evidence shifts the next optimization target to a measured CPU-owned phase (DeltaNet first, LM head second), not higher expert residency or generic CUDA flags. First add a separate fixed 64-output control guard under the same 125 W safety policy, then advance deliberately toward the 512-output repetitions.
- Evidence: [T04 record](../results/T04-Qwen-16-dual-p40-125w.md).

## T04 — 64-output control cooldown gate

- Date: 2026-09-08.
- Status: failed cooldown gate; do not promote its performance data to a baseline.
- Result: the fixed 64-output control generated successfully at 1.50 engine-reported output tok/s (1.58 decode tok/s over 63 steps). DeltaNet remained the largest measured phase at 333.38 ms/token, ahead of LM head at 147.48 and MoE at 66.62. Peak samples were 47°C/49°C and no fan or device fault occurred.
- Failure: the old executor checked for <=40°C exactly at its five-minute deadline. GPU1 was still 42°C, triggering `cooldown_timeout` even though it cooled to 40°C about three minutes later. Cleanup nevertheless released both GPUs and restored both 250 W limits.
- Exact correction: change cooldown semantics from "five minutes then <=40°C" to "at least five minutes and <=40°C", with a conservative 15-minute cap. Unit tests cover continuing after the five-minute mark and failing only at the extended maximum.
- Decision: deploy the corrected base guard and repeat this identical 64-output control. Do not run 512 outputs or vary settings first.
- Evidence: [control record](../results/T04-control-64-dual-p40-125w.md).

## T04 — 64-output corrected-policy repeat

- Date: 2026-09-08.
- Status: pass.
- Exact profile: unchanged fixed Qwen 64-output, two-P40, 125 W profile; only the corrected post-run condition (at least five minutes and <=40°C, max 15 minutes) differed from the prior attempt.
- Result: 1.52 engine-reported output tok/s (1.58 decode tok/s over 63 steps), TTFT 2.28 s, and 630.92 ms/token total decode. Phase values were stable: DeltaNet 330.63, LM head 146.83, attention 85.89, MoE 67.57 ms/token. Expert events were H2D 186, kernels 2,458, D2H 64 ms over 6,226 groups/24,960 experts.
- Safety: peak samples 47°C/48°C and 7,895 MiB/card; fans 2,000–2,100 RPM; no fresh critical event; 8.8-minute cooldown to <=40°C; both GPUs empty and original 250 W limits restored.
- Decision: establish this as the first sustained model control. The next workload-bearing gate is a separate 256-output fixed plateau, not a performance knob sweep or 512-output baseline. The warm page cache/load time is a remaining confounder for startup metrics.
- Evidence: [passing control record](../results/T04-control-64-dual-p40-125w.md).

## T04 — 256-output fixed thermal plateau

- Date: 2026-09-08.
- Status: pass as a single fixed-profile control; do not extend the
  output-length ladder under this thermal profile.
- Hypothesis: the accepted 64-output Qwen timing split and full expert
  residency remain stable over a longer decode without crossing the guarded
  thermal/fan/cleanup requirements.
- Exact profile: unchanged experimental binary, pinned Kreuzzelg snapshot,
  fixed 15-token public prompt, GPUs 0 and 1, `CUDA_EXPERT_GB=auto`,
  `COLI_DENSE_I8=1`, timers and async profile enabled, 125 W/card; 256 outputs.
- Result: 1.54 engine output tok/s (1.56 decode tok/s over 255 steps), TTFT
  2.59 s. DeltaNet remained 337.99 ms/token, including 234.7 ms/token in its
  projections; LM head was 146.91, attention 88.27, and MoE 67.15. All
  10,240 experts stayed resident with zero CPU misses and no swaps.
- Safety: sampled peaks were 55 C / 57 C and 7,895 MiB/card. Fans stayed
  2,000–2,100 RPM, no fresh fault appeared, allocations were released, and
  250 W limits were restored. The <=40 C recovery point arrived at the end of
  the 15-minute maximum cooldown.
- Decision: a 512-output control cannot answer a new performance question and
  adds unacceptable heat-soak time. Stop the length ladder. Advance to a
  source-isolated T05 Q8-weight/FP32-activation DeltaNet projection control;
  prove one operator's parity and complete transfer-inclusive latency before
  connecting it to Qwen or testing DP4A activation quantization.
- Evidence: [256-output record](../results/T04-control-256-dual-p40-125w.md).

## T05A — DeltaNet Q8-weight GPU offload control

- Date: 2026-09-08.
- Status: superseded by the later executed records below.
- Hypothesis: the existing CUDA format-1 operator can accelerate Qwen's
  measured DeltaNet projection bottleneck without changing its Q8-weight,
  FP32-activation representation.
- Source finding: Qwen's x86 `matmul_q()` is FP32 activation times a
  per-output scaled int8 weight row. It does **not** quantize activations in
  the normal AVX2 decode path. `coli_cuda_matmul()` already caches the same
  format-1 weights/scales and transfers host FP32 input/output around a GPU
  reduction kernel. Qwen does not currently call that general API.
- Smallest test: a synthetic, deterministic, GPU0-only 2048x8192 control,
  with the two other DeltaNet shapes (2048x4096 and 4096x2048), CPU-equivalent
  reference, numerical gate, cached steady-state timing, separate upload
  timing, and the existing 125 W server guard.
- Decision: build/test the fixture and its dry-run guard before any CUDA
  workload. Do not touch the dirty remote backend worktree, load Qwen, or
  promote a projection microbenchmark to an end-to-end speed claim.
- Design: [T05 experiment](003-qwen-deltanet-q8-offload.md).

## T05A — cached `dn_qkv` Q8 GPU control

- Date: 2026-09-08.
- Status: numerical and GPU-latency pass; CPU/GPU speed ratio not promotable.
- Exact profile: synthetic deterministic Q8 weights / per-row scales and FP32
  activation; Qwen-equivalent AVX2/FMA CPU control; one cached GPU0 generic
  format-1 tensor; 5 samples of 8 calls; 125 W GPU0 guard.
- Result: CPU median 4.645 ms/call; transfer-inclusive cached GPU median
  0.335 ms/call; observed 13.86x CPU/GPU ratio. First call with CUDA init/weight
  upload was 21.228 ms. Max absolute error 1.526e-5 and max relative error
  5.965e-6; the declared per-element FP32 gate passed.
- Safety: GPU0 sampled peak 36 C, GPU1 37 C; all fans 2,000–2,100 RPM; no new
  fault; minimum five-minute cooldown; GPU0 250 W cap restored. The work
  completed between telemetry polls, so fixture cached-tensor byte accounting
  is the residency evidence.
- Correction: the initial fixture inherited OpenMP policy and reused one
  16 MiB weight tensor, so the CPU control can be cache-hot and does not match
  Qwen's 30-layer working set. The GPU number and numerical gate remain valid,
  but the ratio does not clear the 15% threshold.
- Decision: pin a Qwen-like 24-thread OpenMP environment in the guard. T05B
  demonstrated the cache effect and is performance-inconclusive; proceed to
  T05C, a 30-layer (about 960 MB Q8) cache-aware triplet sweep before touching
  model execution.
- Evidence: [T05A record](../results/T05A-dn-qkv-2048x8192.md).

## T05C — cache-aware 30-layer Q8 DeltaNet sweep

- Date: 2026-09-08.
- Status: rejected on the numerical gate; performance result not promotable.
- Hypothesis: the apparent single-projection CUDA benefit survives Qwen's
  full 30-DeltaNet-layer Q8 working set, which is about 960 MiB and cannot
  remain in the host's 60 MiB per-socket L3 cache.
- Exact profile: GPU0 only at 125 W; 30 distinct qkv (2048x8192), z
  (2048x4096), and out (4096x2048) Q8 projections; fixed 24-thread OpenMP
  environment; three CPU and GPU sweeps; generic cached format-1 API.
- Result: CPU median 38.785 ms/sweep; GPU complete median 17.910 ms/sweep;
  observed 2.166x ratio. First sweep including initialization and about 960
  MiB upload was 162.836 ms. All 90 tensors were cached (1,008,353,280 bytes).
  The output failed the per-element FP32 gate: maximum absolute error
  9.155e-4, relative error 4.286e-4.
- Safety: preflight 33 C / 35 C; GPU0 sampled peak 34 C, all fans
  2,000–2,100 RPM; no fresh fault; minimum five-minute cooldown; zero retained
  GPU allocation and 250 W cap restored.
- Decision: reject the numerical control. T05D changes diagnostics only:
  separately record qkv, z, out reduction, and host-boundary propagation
  errors. Do not touch Qwen integration or infer end-to-end speed. The source
  inspection also shows DeltaNet includes two FP32 b/a projections outside
  this Q8 sweep, so even a future pass is a partial-phase result.
- Evidence: [T05C record](../results/T05C-dn-sweep-30x-triplet.md).

## T05D — error attribution for the rejected Q8 sweep

- Date: 2026-09-08.
- Status: generic format-1 CUDA dense matvec rejected for strict Qwen
  DeltaNet integration.
- Exact change: no GPU computation change from T05C. Added CPU-only
  attribution of qkv, z, out-with-identical-input, and boundary-propagation
  error after the first sweep.
- Result: qkv (2.289e-5 absolute / 1.296e-5 relative) and z
  (1.907e-5 / 1.580e-5) passed. The direct out kernel with the same
  GPU-derived input failed (9.155e-4 / 3.022e-4); CPU computation with a
  GPU-derived qkv/z boundary also failed (7.324e-4 / 5.817e-4).
- Interpretation: the generic GPU 256-thread reduction order itself is out of
  tolerance, and independently small earlier differences propagate through the
  host boundary. Do not loosen the numerical threshold or integrate this path.
- Safety: 125 W GPU0 guard, peak 34 C, all fans 2,000–2,100 RPM, no new fault,
  five-minute cooldown, zero retained VRAM, and cap restored to 250 W.
- Decision: T05E is a standalone custom Q8 out-projection kernel that mirrors
  Qwen's 32-lane AVX accumulator/reduction order. It needs a parity result
  before a 30-layer test, production-source patch, or end-to-end claim.
- Evidence: [T05D record](../results/T05D-dn-error-attribution.md).

## T05E — exact-order standalone Q8 out projection

- Date: 2026-09-08.
- Status: numerical pass, performance hold.
- Hypothesis: keeping Qwen's 32 FMA streams and final reduction tree on Pascal
  eliminates the format-1 generic CUDA rounding failure.
- Result: exact bit identity against the AVX2/FMA reference. CPU median
  0.1400 ms/call; GPU complete median 0.1247 ms/call; 1.123x observed ratio.
  First GPU call with initialization and 8 MiB upload took 1.694 ms.
- Safety: GPU0 peak 34 C; all fans 2,000–2,100 RPM; no fault; five-minute
  cooldown; no retained allocation; 250 W cap restored.
- Decision: mathematical hypothesis passes, but 12.3% misses the 15% keep
  threshold. T05F changes only final reduction transport from shared memory
  plus barrier to a warp-shuffle tree with the exact same arithmetic order.
  It must retain bit identity and meet the speed threshold before chain work.
- Evidence: [T05E record](../results/T05E-dn-out-cpuorder.md).

## T05F — exact-order warp-shuffle out projection

- Date: 2026-09-08.
- Status: rejected for the isolated projection.
- Exact change: replaced T05E's shared-memory final reduction and barrier
  with the equivalent warp-shuffle reduction tree; FMA streams and input,
  weights, scales, guard, and CPU control were unchanged.
- Result: bit-identical output. CPU median 0.1161 ms/call; GPU complete median
  0.1245 ms/call; 0.932x observed ratio.
- Decision: stop tuning the isolated 8 MiB out projection. Its CPU control is
  cache-sensitive and no longer answers the product question. T05G applies the
  exact-order kernel across the 30-layer, 90-projection, 960 MiB Q8 sweep. It
  must retain exact parity and exceed the 15% keep threshold; otherwise reject
  direct Q8 projection offload.
- Safety: GPU0 peak 34 C; all fans 2,000–2,100 RPM; no fault; five-minute
  cooldown; no retained VRAM; cap restored.
- Evidence: [T05F record](../results/T05F-dn-out-shuffle.md).

## T05G — exact-order full Q8 DeltaNet sweep

- Date: 2026-09-08.
- Status: pass for synthetic Q8 projection offload; no Qwen integration claim.
- Hypothesis: the exact arithmetic kernel can retain bit identity and clear
  the 15% performance gate when measured across Qwen's cache-resistant,
  30-DeltaNet-layer Q8 working set.
- Exact profile: GPU0 only at 125 W; 30 distinct `qkv` (2048x8192), `z`
  (2048x4096), and `out` (4096x2048) Q8 projections; 24-thread CPU control;
  three CPU and GPU sweeps; transfer-inclusive cached GPU timing; exact
  CPU-order CUDA kernel.
- Result: bit-identical output, 36.6309 ms CPU median versus 13.1462 ms GPU
  median (2.786x). The first initialized/uploaded sweep took 148.572 ms; 90
  tensors accounted for 1,008,353,280 cached bytes.
- Safety: sampled peak 34 C on GPU0, 35 C on GPU1; all fans 2,000–2,100 RPM;
  no cleanup action; independent post-run check confirmed empty cards and
  restored 250 W caps.
- Decision: advance to one opt-in, experimental-source integration canary.
  Preserve fallback and prove byte-identical fixed-output behavior before any
  64-output performance comparison. Do not combine this with FP32 DeltaNet,
  expert, attention, or multi-GPU changes.
- Evidence: [T05G record](../results/T05G-dn-full-sweep-cpuorder.md).
