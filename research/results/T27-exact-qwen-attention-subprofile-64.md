# T27 — exact-Qwen attention subprofile, 64 tokens

Date: 2026-09-09

## Scope

T27 inherits the accepted experimental DeltaNet QKV/Z pair from T24 and adds
only default-off timing reads behind `COLI_ATTN_PROFILE=1`. It leaves every
attention operation, Q8 cache, projection call, and output order unchanged.
It is an instrumentation result, not a speed comparison: timing reads add
overhead, so its 69.37 ms/token total must not be compared to T24's 64.04.

## Exactness and timing

The fixed 64-token stdout SHA-256 exactly matched the established oracle:
`5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.

| Exact decode interval | ms/token |
| --- | ---: |
| Q projection | 1.66 |
| K projection | 0.36 |
| V projection | 0.35 |
| CPU attention middle | 2.25 |
| O projection | 1.15 |
| Attention total | 5.80 |

Q and K must complete before normalization, RoPE, scoring, and softmax; O
depends on the completed gated context. V alone can potentially overlap CPU
Q/K preparation, but its 0.35 ms interval (or 0.71 ms if K and V could both be
hidden) is an upper bound below 1.1% of T24 total decode. The profile therefore
does not justify an async-attention implementation.

## Decision

Reject attention synchronization work as the next implementation target. Keep
the exact attention cache as-is. Resume approximate research only with a new
actual-Qwen expert shadow control that improves on T21's rejected plain
per-row W4A8 using groupwise scales and explicit outlier handling. Do not
modify production Colibri.

## Safety

The guard passed. Peak sampled temperatures were 45 C on each P40; fans were
2,000–2,100 RPM. Allocations were released, cooldown completed, and both caps
were restored to 250 W.

## Evidence

`research/results/raw/T27-qwen-attention-profile-38a5e84f-f852-48f5-b62d-46491062c830.{json,stderr.txt,stdout.txt}`.
