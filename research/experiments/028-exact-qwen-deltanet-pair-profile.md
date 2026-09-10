# E028 / T36 — exact Qwen DeltaNet-pair dependency profile

## Hypothesis

The accepted exact-Q8 QKV/Z pair may still contain enough CPU launch or join
overhead for CUDA Graphs or more asynchronous submission to improve full Qwen
decode.

## One-variable change

With `COLI_DN_PAIR_PROFILE=1` only, preserve the accepted QKV/Z pair's data
flow and numerical kernels. Add host wall-clock counters for issue, independent
CPU B/A, and join. Add persistent CUDA events on the existing pair stream:
H2D input, QKV kernel, Z kernel, and both D2H results. Emit aggregate counters
only after the existing take-side stream synchronization.

## Acceptance gates

- Canonical fixed-64 stdout SHA-256 must remain
  `5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.
- Every one of 30 layers × 63 decode steps must produce exactly one finite host
  record and one finite GPU record: 1,890 each.
- GPU aggregate total must equal its component sum within 0.02 ms.
- The existing locked 125 W/card workload guard must pass two idle preflights,
  thermal monitoring, cooldown, allocation release, and power restoration.

## Decision rule

This is not a speed benchmark because timing events add work. If host issue plus
join is materially smaller than the existing pair's stream interval, CUDA
Graphs cannot remove enough time to justify a graph implementation. Then
profile the remaining DeltaNet subphases before changing execution.

## Result

Passed all exactness, counter, and safety gates. The result is recorded in
[T36](../results/T36-exact-qwen-deltanet-pair-profile.md). CUDA Graph work is
not selected: the directly observable host issue plus join is only 1.89
ms/token, and most of it is already outside the pair stream's critical work.
