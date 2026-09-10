# E034 / T42 — exact Qwen asynchronous expert-tier event profile

## Hypothesis

Qwen routes its eight selected MoE experts by a fixed expert-home rule across
the two P40s. qt_issue() launches each device's resident-expert group before
the CPU runs the shared MLP, but qt_take() later synchronizes devices in
order. The important unknown is whether the two cards' kernels overlap in the
real model, or whether their event times add on the single-token critical path.

## One-variable change

An isolated T24 source copy adds default-off COLI_QTIER_PROFILE=1. It reuses
four CUDA events per device around the existing asynchronous expert path:
start to post-H2D, post-H2D to post-kernel, and post-kernel to post-D2H.
Events are read only after the existing cudaStreamSynchronize() in take; they
do not change expert routing, kernels, weights, arithmetic, device assignment,
or normal (profile-off) execution.

## Acceptance gates

- Canonical 64-output stdout SHA-256 remains
  5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f.
- Both device profile lines must report nonzero groups, experts, rows, and
  kernel event time.
- Use the standard two idle preflights, 125 W/card cap, cooldown, release,
  and restoration protocol.
- Treat this as attribution, not a performance candidate: CUDA events add
  profiling work and its decode rate is not a selection metric.

## Result

T42 passed all gates. Across 63 timed decode tokens, device 0 recorded 1,191.78
ms kernel time (18.92 ms/token) and device 1 recorded 1,236.75 ms (19.63
ms/token). Their 2,428.53 ms aggregate kernel time is larger than the 2,007.8
ms MoE wall stage, proving useful inter-device overlap; the GPU critical-path
lower bound is the slower device's 19.63 ms/token, not the 38.55 ms/token sum.

The profile also recorded per-token H2D/D2H event times of 0.79/0.56 ms on
device 0 and 0.68/0.50 ms on device 1. Existing end-to-end timers show 2.79
ms/token issue, 16.76 ms shared-MLP overlap work, and 3.72 ms qt_take.
Therefore an immediate parallel-take rewrite is not justified: most expert GPU
work is already hidden beneath the shared MLP. The next measurement should
split qt_take host synchronization from weighted host accumulation before
attempting an optimization.

See [T42](../results/T42-exact-qwen-qtier-async-profile.md).
