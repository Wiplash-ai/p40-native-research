# E025 / T32 — bounded real-Qwen gate/up capture for correction replay

Status: complete — exact-output T32 collection passed; production unchanged.

## Question

T31 shows that the W4A8 tail is entirely attributable to quantizing the input
feeding gate/up.  It does not say whether the coordinates with the largest
activation residual are the coordinates that matter after the two projections,
SiLU, and down matrix.

Collect the smallest self-contained sample that can falsify that hypothesis:
three fixed layers, four fixed decode positions, and all eight routed experts.
The first two positions form calibration; the next two are untouched holdout.

## Exactness boundary

The normal Qwen W4/FP32 expert path remains the returned value.  T32 only
copies the input, exact intermediates, original packed W4 weights/scales, and
the groupwise-Q8 input representation into a bounded sidecar.  It must not
modify routing, sampling, output text, or model state.

## Gates

1. Reuse T31's fixed 16-token exact stdout SHA-256.
2. Require a valid 96-record `WGCAP v1` sidecar under 512 MiB.
3. Require 48 fixed calibration records and 48 fixed holdout records.
4. Capture source must report no failed copies, invalid metadata, or overflow.
5. Treat runtime duration as collection overhead, never as a performance result.

## Follow-up

T33 will replay each record on CPU and compare residual-magnitude selection to
stored column sensitivity, a local Jacobian score, and an offline oracle at
K=0/4/8/16/32.  It must establish reconstruction parity before drawing any
quality conclusion from correction rankings.

## Result

The fixed 16-token output SHA-256 remained
`43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a`.
The guard accepted the 154,540,864-byte sidecar with SHA-256
`e6e670f1884c33f43c17b7e8acdab102bf9eeec6d182944f2b6f220db7ff051e`:
96 records, 48 calibration / 48 holdout, all fixed route tuples exactly once,
and zero capture-source errors. It is retained locally and on the server at
the guarded result path; the binary is intentionally excluded from ordinary
Git because it exceeds common hosted-repository object limits.

The collection is not a speed result. The runtime includes intentionally extra
raw gate/up execution and ~155 MiB of sidecar transfer/write work. Safety
passed: 40 C / 41 C peak temperatures, 7,895 MiB/card peak VRAM, 2,000–2,100
RPM fans, five-minute cooldown, release of allocations, and restoration to
250 W/card.
