# E009 / T14 — Qwen shared-expert exact-Q8 control

Status: source-only benchmark fixture; CPU-only compile and dry run pass; no
GPU workload has run.

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

## Acceptance and decision rule

Run only under a new restricted 125 W GPU0 guard, after T13's recovery plus a
fresh two-sample cool/idle preflight. Require bit identity, successful BMC
fan/thermal/allocation/power restoration, and transfer-inclusive GPU median at
least 15% faster than CPU. If it passes, add a **separate opt-in registry flag
in a fourth isolated source copy**; do not combine its integration canary with
MoE W4A8/DP4A, expert-tier, or attention changes.

If it fails the speed gate, keep the exact CPU shared path and shift to the
reviewed Pascal W4A8/DP4A standalone expert primitive. That approximate path
requires its own quality gate and cannot claim byte identity with Qwen.
