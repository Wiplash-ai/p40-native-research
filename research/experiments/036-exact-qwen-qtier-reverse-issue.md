# E036 / T44 — exact Qwen expert-tier reverse issue order

## Hypothesis

T43 attributes the qt_take tail to the device-1 expert stream rather than to
host accumulation. The existing issue loop enqueues device 0 before device 1;
starting device 1 first might reduce enough launch skew to shorten the
autoregressive critical path.

## One-variable change

An isolated source copy adds default-off `COLI_QTIER_REVERSE_ISSUE=1`. It
reverses only the order in which existing per-device expert groups are issued:
device 1 then device 0. `qt_take` remains device 0 then device 1, as do expert
home placement, routing, weights, kernels, result accumulation, and all model
math.

## Acceptance gates

- Canonical 64-output stdout SHA-256 remains
  `5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.
- The stderr marker confirms reverse launch order is active.
- Standard dual preflight, 125 W/card, cooldown, release, and restoration
  protocol must pass.

## Result

T44 passed exactly and emitted its active marker, but it did not improve
performance. It recorded 12.83 tok/s and 64.17 ms/token, compared with the
adjacent unchanged T24 control at 12.90 tok/s and 63.93 ms/token. The small
difference is in the wrong direction and is below a claimable gain even before
accounting for run-to-run noise.

The test falsifies issue-order skew as a useful target. Do not alter ordered
result collection or pursue a host-side join rewrite; T43 already showed that
CPU accumulation is insignificant. The next profiling work should audit the
exact shared MLP path and its overlap with the resident-expert critical path.

See [T44](../results/T44-exact-qwen-qtier-reverse-issue.md).
