# T23 — exact-Qwen DeltaNet pair issue/join canary

Date: 2026-09-09

## Question

Can Qwen3.6's exact cached Q8 DeltaNet QKV and Z projections share one input
upload and overlap their unchanged GPU kernels with the independent CPU B/A
projections, without changing generated output?

## Change under test

An isolated copy of the accepted T16 engine added the opt-in
`COLI_CUDA_DN_PAIR=1` path. For a single-token DeltaNet layer it:

1. uploads `x` once;
2. issues the existing CPU-order Q8 QKV and Z kernels in one dedicated stream;
3. computes B and A on the CPU while that pair is pending;
4. downloads and synchronizes QKV/Z immediately before the existing recurrence.

Only one pair may be pending. A failed issue or join shuts the cache down and
recomputes both projections with the established serial exact-Q8 path.

## Exact-Qwen 16-token acceptance

- Fixed model, prompt, two P40s, existing exact-Q8 caches, and 125 W/card
  guard were retained.
- The exact output SHA-256 matched the accepted 16-token oracle:
  `43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a`.
- Both required runtime markers appeared: the 90 DeltaNet/LM-head/attention/
  shared-MLP cache and `QKV/Z pair issue/join active`.
- Decode phase timers: DeltaNet 25.25 ms/token, attention 5.28, MoE 31.14,
  LM head 3.34, total 65.01 ms/token over 15 decoded tokens. Reported overall
  speed was 8.53 tok/s for this short response and includes prefill, so it is
  not comparable to the fixed 64-token baseline.

## Safety

The guard recorded peak 43 C / 44 C, approximately 7.5 GiB/card during
warmstart, fans at 2,000–2,100 RPM, no safety action, a clean cooldown, VRAM
release, and restoration of both original 250 W power limits.

## Decision

The implementation clears exactness and safety gates only. It has not yet
cleared the performance gate: run the separately pinned 64-token Qwen
comparison and compare its fixed decode phase timers against T16/T22 before
promoting it. No production Colibri source changed.

## Evidence

`research/results/raw/T23-qwen-deltanet-pair-7b453043-a9c2-4132-9d4a-95e4597570af.{json,stderr.txt,stdout.txt}`.
