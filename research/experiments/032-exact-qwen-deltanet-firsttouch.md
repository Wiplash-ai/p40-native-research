# E032 / T40 — exact Qwen DeltaNet per-head first-touch control

## Hypothesis

The default Qwen path serially resets all recurrent-state pages before the
static OpenMP value-head loop. On the two-socket host this can place all
`DN_rec` pages near the launcher thread, forcing remote memory traffic for
workers on the other socket. First-touching each independent 64 KiB head state
from its eventual static worker should improve recurrence locality without
changing model arithmetic.

## One-variable change

Use an isolated copy of the retained exact T24 source with only the opt-in
`COLI_DN_FIRSTTOUCH=1` reset path. It zeros each `DN_rec[layer][head]` slice in
an OpenMP `schedule(static)` head-parallel loop before generation; `DN_conv`
remains serial because its normal execution is serial. The patch queries and
reports the Linux node for each first-touched head page. All model/GPU/OpenMP
controls and the fixed 64-output oracle remain unchanged.

## Acceptance gates

- Canonical stdout SHA-256 must remain
  `5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.
- The source must report 960 first-touched recurrent heads (30 DeltaNet layers
  times 32 value heads), with both NUMA nodes represented and no unknown page
  node.
- The run must pass the normal dual-preflight 125 W/card thermal guard,
  release, cooldown, and restoration protocol.
- Retain it only if fixed-64 end-to-end Qwen throughput improves beyond noise;
  otherwise record the placement evidence and reject it.

## Result

All correctness, placement, and thermal gates passed: the exact stdout hash
matched and the source reported 599 node-0 + 361 node-1 head slices with no
unknown pages. Yet the immediate unmodified T24 control measured 12.90 tok/s
where T40 measured 8.21 tok/s (-36.4%). Reject first-touch placement. Its
DeltaNet time more than doubled and every other timed stage also regressed, so
do not pursue further isolated NUMA placement tweaks. See
[T40](../results/T40-exact-qwen-deltanet-firsttouch.md).
