# E038 / T46 — exact Qwen shared/expert timeline ineligible-window control

T45's fixed trace established that a few layer/token pairs have no GPU-0
routed expert. T46 changes only the profile accounting: those pairs are
reported as `no-gpu0`, while event creation, recording, elapsed-query, and
unexpected collection failures remain fatal. The guarded invariant is now
`valid windows + no-gpu0 == 63 × 40`, with zero real failures. Model math,
event placement, weights, routing, and the 64-token exact-output oracle are
unchanged.

## Result

T46 completed under the guarded exact-Qwen protocol. The fixed 64-token
stdout matched the canonical SHA-256
`5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.
The new accounting reconciled all 2,520 decode-layer opportunities: 2,518
valid GPU-0 windows, 2 legitimate `no-gpu0` cases, and zero event failures.

Across valid windows, the shared exact-Q8 CUDA stream occupied 1,060.48 ms
(0.421 ms/window). The signed `dense-end-to-expert-done` total was
-168.91 ms (-0.067 ms/window). By the instrumentation contract, negative
means the GPU-0 expert D2H completion preceded the shared stream's end: the
shared stream extended the local GPU-0 tail by about 67 microseconds per valid
window. This is a device-0-local relationship, not an end-to-end additive
latency: GPU 1 is independent and was the slower expert stream in T42/T43.

The profile is attribution only. Its instrumented 12.84 tok/s / 64.07
ms-token result must not be compared with uninstrumented speed controls.
T26 already rejected a shared gate/up scheduling change. Therefore T46 closes
shared-MLP dispatch as the next optimization lead; the next profile should
quantify the device-1 expert tail distribution before any residency or
load-balancing intervention.

Safety passed at 125 W/card: both P40s peaked at 44 C, peak allocation was
7,893 MiB/card, all eight fans remained at 2,000–2,100 RPM, allocations were
released, cooldown completed, and both power limits were restored to 250 W.

Raw evidence:
`research/results/raw/T46-qwen-shared-expert-timeline-35f20d59-ba65-4d7f-933f-0ae6667b31a5.{json,stderr.txt,stdout.txt}`.
