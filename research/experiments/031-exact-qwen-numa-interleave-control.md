# E031 / T39 — exact Qwen blanket NUMA-interleave control

## Hypothesis

The CPU-side DeltaNet state and host-side expert work might benefit from an
OS-level memory policy that distributes allocations across both sockets, while
the process still uses the retained fixed 24-worker OpenMP configuration.

## One-variable change

Prefix the unchanged fixed-64 T24 process with
`/usr/bin/numactl --interleave=all`. The exact Qwen binary, model, prompt,
GPU selection, expert residency, OpenMP environment, output oracle, and guard
all remain unchanged. No `COLI_NUMA` environment variable is set: the direct
Qwen implementation does not consume it.

## Acceptance gates

- Canonical stdout SHA-256 must remain
  `5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.
- The run must pass locked 125 W/card monitoring, release, cooldown, and power
  restoration after two idle preflights separated by more than 60 seconds.
- Retain blanket interleaving only when the fixed-64 end-to-end rate improves
  beyond measurement noise. A stage-level change alone does not retain it.

## Result

Correctness and safety passed, but blanket interleaving is rejected. T39
produced 9.84 tok/s and 86.60 ms/token versus the retained T35 control's 12.88
tok/s and 64.20 ms/token: a 23.6% rate regression. DeltaNet grew from 24.39
to 44.94 ms/token, especially its `l2n+rec` subgroup (6.8 to 22.5 ms/token).
This strongly indicates that striped page placement is hostile to the
recurrent state locality pattern; it does not evaluate or invalidate a
carefully first-touched per-head state allocation. See
[T39](../results/T39-exact-qwen-numa-interleave.md).
