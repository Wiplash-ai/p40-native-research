# E009 / T14 — Qwen shared-expert exact-Q8 control

Status: pass for the standalone exact-Q8 control; full-model integration has
not run.

## Hypothesis

With every routed expert resident in VRAM, the dominant remaining T13 decode
phase is not an expert cache miss. Qwen's CPU shared expert—two `2048 -> 512`
projections, SiLU product, and a `512 -> 2048` projection in each of 40
layers—takes about 70.74 ms/token while routed GPU groups execute. The exact
Q8 CPU-order helper may reduce that phase without changing sampled output.

## Smallest test

`qwen_shared_expert_cpuorder_control` creates 40 synthetic shared MLPs using
the real Qwen dense-Q8 shapes. It preserves the actual per-layer order:
gate projection, up projection, CPU SiLU product, then down projection. It
uses the same 32-lane CPU-order CUDA helper as T05/T11, transfers activations
at the real host boundary, and requires bit-identical gate, up, activation,
and down vectors against the AVX2/FMA Qwen control.

The fixture has 120 Q8 matrices and 125,829,120 bytes of Q8 weights. It is
bounded to GPU 0, 1--3 repetitions, one sweep per sample, and a 144--176 MiB
host allocation envelope. A dry run must report `cuda_initialized:false`.

The fixture compiled on the server explicitly for `sm_61` with
`CUDA_VISIBLE_DEVICES=""`; its dry run emitted the expected schema and
`cuda_initialized:false`. Compilation did not load a model or initialize CUDA.

## Guarded result

The fixed GPU0-only run (`run_id` `63e129e0-a7e6-455a-9218-29c7120c2324`)
passed its bit-identity, 125 W, fan, allocation-release, cooldown, and power
restoration gates. It accounted for all 120 Q8 matrices (125,829,120 bytes)
and 128,368,640 host bytes. All output comparisons were bit-identical:
zero mismatches and zero numerical error.

CPU samples were 10.7883, 10.7545, and 10.9191 ms per 40-layer sweep; GPU
transfer-inclusive samples were 5.59371, 5.57591, and 5.52076 ms. The medians
are 10.7883 and 5.57591 ms, respectively: **1.9348x** CPU/GPU. First
initialization, upload, and sweep took 45.628 ms. The work completed between
telemetry polls, so 120-matrix byte accounting is the residency evidence.

Telemetry never exceeded 37 C. Every fan remained 2,000--2,100 RPM; the
guard released allocations, completed cooldown, and restored the GPU0 cap to
250 W. GPU1 stayed idle.

## Integration decision

The result clears the 15% speed rule, so add a **separate opt-in registry flag
in a fourth isolated source copy**. The integration must change only shared
expert Q8 matvec dispatch, require exactly 120 shared matrices alongside the
previous 90+1+40 cache, and preserve the 16-output oracle. Do not combine it
with MoE W4A8/DP4A, expert-tier, or attention changes.

Raw evidence: `research/results/raw/T14-shared-expert-cpuorder-63e129e0.*`.
