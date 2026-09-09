# E014 / T19 — Pascal W4A8 DP4A 16-warp CTA tile

Status: pass as a measurement; reject Tile16 as the preferred variant.

## Hypothesis

T17 established the unrolled, eight-warp-per-CTA W4A8 control. A 16-warp tile
could reduce block-launch scheduling overhead for the 512 output rows, but may
lower occupancy or register availability on Pascal. This is the largest
plausible tile with one warp per output row inside P40's 1,024-thread CTA
limit; it is the relevant analogue of the external llama.cpp tile idea.

## Test and gate

Compile only `P40_W4A8_TILE=16`, retaining unrolling. Integer parity, error,
SASS, host transfers, weight layout, seed, and GPU0 guard remain fixed. Keep
tile 16 only if it is at least 5% faster than T17's 0.0609619 ms; otherwise
retain tile 8. This remains a synthetic approximate primitive, not a Qwen
integration authorization.

## Static acceptance

The `sm_61` CPU-hidden build produced SHA-256
`4222a603420ed94a856d249cdd0e6b2decafb9240c9f94ff7808b5e3cd5de43e`.
Its dry-run returned before CUDA initialization, and SASS contains 16
`IDP.4A.S8.S8` instructions. Runtime is still gated separately by a distinct
fixed-command key, two live idle/cool samples at least 60 seconds apart, a
125 W GPU0 cap, and cooldown/restore checks.

## Guarded result

T19 passed integer parity, its 0.384% relative-L2 error, the `IDP.4A` SASS
check, and every thermal/cleanup gate. Its transfer-inclusive W4A8 median was
0.0808166 ms, however, versus T17 Tile8's 0.0609619 ms: 1.32569x slower. The
Tile16 binary's own W4/FP32 control was 0.115078 ms (1.42394x internal
speedup), but that separately compiled baseline is not used for cross-binary
selection. GPU0 remained at or below 36 C; GPU1 at or below 36 C; all fans
were 2,000--2,100 RPM; allocation release, cooldown, and the 250 W restore
passed.

Decision: retain the unrolled eight-warp CTA. The small external llama.cpp
tile insight does not transfer to this warp-per-output Qwen-shaped control.
This remains a synthetic approximate primitive, not a Qwen integration result.
