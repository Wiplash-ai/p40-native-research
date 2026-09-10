# E029 / T37 — exact Qwen DeltaNet remainder profile

## Hypothesis

After the accepted QKV/Z pair, DeltaNet's remaining time may be dominated by
CPU-side recurrence or by the synchronous exact-Q8 output-projection boundary.

## One-variable change

With `COLI_DN_FINE_PROFILE=1` only, retain the accepted pair path and add
host clocks around Q/K repetition+normalization, recurrent delta-rule state,
gated RMS normalization, and the final output-projection call. The final
projection calls the same exact-Q8 CUDA kernel and copies as the normal path;
persistent CUDA events bracket only its H2D, kernel, and D2H intervals.

## Acceptance gates

- The fixed-64 canonical stdout SHA-256 must remain
  `5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.
- Each of 30 layers × 63 decode steps must report exactly one finite host and
  final-output GPU record: 1,890 each.
- GPU total must equal H2D + kernel + D2H within 0.02 ms.
- The existing locked 125 W/card thermal, cooldown, release, and restoration
  guard must pass after two idle preflights more than 60 seconds apart.

## Decision rule

This is instrumentation rather than a speed comparison. If the output
projection is kernel-dominated and its host-call time is near the event total,
launch/graph work is not justified. Select the largest remaining exact CPU
component for the next real-Qwen execution experiment.

## Result

Passed all gates. The output projection is 2.784 ms/token of exact-Q8 GPU
kernel time within a 3.627 ms/token stream interval and a 3.990 ms/token host
call. The largest CPU component is recurrent state update at 5.730 ms/token.
See [T37](../results/T37-exact-qwen-deltanet-remainder-profile.md).
