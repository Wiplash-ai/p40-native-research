# E017 / T22 - exact Qwen asynchronous-expert profile

Status: pass; exact profile completed.

## Why this test

T16's exact 64-token timing puts DeltaNet at 31.36 ms/token and routed MoE at
30.63 ms/token. Its qtier H2D/kernel/D2H counters were all zero, but source
inspection shows that their CUDA events live only in the synchronous
expert_group_impl() path. Qwen calls asynchronous issue() and take().

## Hypothesis and prediction

Persistent events placed around the async issue upload, kernels, and download
will provide nonzero expert transfer/kernel totals without changing generated
text. The exact 64-token stdout oracle must remain unchanged.

## Smallest safe experiment

Copy the accepted T16 source to colibri-t22-async-profile; apply only the
existing opt-in 0001-qwen36-async-expert-profile.patch; build CPU-hidden for
sm_61; then run one new forced-command 64-token canary with the same pinned
model, prompt, two P40s, 125 W/card cap, caches, and cooldown guard.

The canary must require:

1. T16's exact stdout SHA-256;
2. the existing exact-Q8 cache marker;
3. a positive async H2D, kernel, and D2H total; and
4. normal thermal/fan/cleanup completion.

## Decision rule

This is instrumentation only. It may classify MoE time as transfer-, kernel-,
or synchronization-dominated, but cannot justify a performance claim or a
source integration. The next optimization choice must use the resulting
shares alongside T16's DeltaNet and attention timers.

## Result

The exact 64-token stdout hash passed at 11.87 tok/s (70.46 ms/token decode).
The async counters reported 6,226 group calls and 24,960 experts: 90 ms H2D,
2,433 ms kernel, and 66 ms D2H. These values are summed device timelines and
can overlap across the two P40s, so they do not equal token wall time; their
94.0% kernel share nevertheless rejects PCIe transfer as the routed-expert
bottleneck.

The full decode profile has two comparable dominant phases: MoE 31.65
ms/token (44.9%) and DeltaNet 29.99 ms/token (42.6%). MoE is mixed: its
16.80 ms/token shared-MLP portion uses the exact CPU-order Q8 projection path,
while the routed expert device section is kernel-dominated. DeltaNet and the
smaller 5.47 ms/token attention phase each invoke the same CPU-order path,
which synchronizes after every projection. The next isolated experiment is
therefore launch/synchronization reduction for exact Q8 projections, not a
PCIe or W4A8 change.
