# E035 / T43 — exact Qwen expert-tier host-take profile

## Hypothesis

T42 proved that the two P40 expert streams overlap, but the exact-Qwen timer
still assigns several milliseconds per decode token to qt_take. That time could
be either host-side weighted accumulation or a residual wait for the slower
device stream. Only the latter is a useful latency target.

## One-variable change

An isolated T24 source copy adds default-off COLI_QTIER_TAKE_PROFILE=1.
For each existing qt_take device group, it measures the elapsed host time
inside coli_cuda_expert_group_take separately from the unchanged ordered
val[k] times row accumulation loop. It uses host monotonic clocks only; no CUDA
events, routing, GPU kernels, arithmetic, or normal profile-off execution are
changed.

## Acceptance gates

- Canonical 64-output stdout SHA-256 remains
  5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f.
- Both device host-take lines must have nonzero group counts.
- Standard dual preflight, 125 W/card, cooldown, release, and restoration
  protocol must pass.

## Result

T43 passed exactly. Across all 6,226 device groups, device 0 spent 2.61 ms
waiting and 4.11 ms accumulating; device 1 spent 250.57 ms waiting and 3.98
ms accumulating. Normalized by group count, device 1 consumes 80.6 microseconds
of residual host wait per group while either device's CPU accumulation costs
only about 1.3 microseconds per group.

There are roughly 40 device-1 groups per model token, so the result is
consistent with the existing 3.33 ms/token qt_take timer: the tail is device
1 stream completion, not CPU accumulation. Do not parallelize or rewrite the
host accumulation loop. The next small falsification is to launch device 1's
slower expert group first, retaining its output accumulation order, to measure
whether launch skew contributes enough to matter.

See [T43](../results/T43-exact-qwen-qtier-take-host-profile.md).
