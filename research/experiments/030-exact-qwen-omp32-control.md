# E030 / T38 — exact Qwen OMP32 control

## Hypothesis

The DeltaNet recurrence assigns independent value heads to OpenMP workers. On a
24-physical-core host with SMT, 32 workers might improve whole-token latency by
giving the recurrent loop more schedulable workers without altering model
arithmetic.

## One-variable change

Reuse the retained T24 exact-Q8 binary, prompt, model snapshot, two-GPU
configuration, 64-output oracle, 125 W/card guard, and OpenMP affinity
settings. Change only `OMP_NUM_THREADS=24` to `OMP_NUM_THREADS=32`.

## Acceptance gates

- Canonical stdout SHA-256 must remain
  `5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.
- The guarded run must pass the existing locked power, release, cooldown, and
  restoration checks after two idle preflights separated by more than 60
  seconds.
- Retain the setting only if the full fixed-64 exact-Qwen rate improves beyond
  normal measurement noise. Stage timing explains a result; it does not
  substitute for end-to-end rate.

## Result

All correctness and safety gates passed, but T38 is rejected: 32 workers
produced 12.62 tok/s versus T35's 12.88 tok/s (-2.0%). DeltaNet regressed from
24.39 to 26.53 ms/token despite a smaller MoE total. Retain the 24-worker
control. See [T38](../results/T38-exact-qwen-omp32.md).
