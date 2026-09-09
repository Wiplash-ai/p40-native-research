# T24 — exact-Qwen DeltaNet pair 64-token comparison

Date: 2026-09-09

## Result

The opt-in exact-Q8 QKV/Z pair issue/join path passed the fixed 64-token Qwen
comparison. It produced the accepted stdout SHA-256
`5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`,
activated the required pair marker, and completed all thermal, fan, allocation
release, cooldown, and power-restoration checks.

## Matched comparison

Both T16 and T24 used the same model, prompt, 64 generated tokens, two P40s,
125 W/card guard, all four exact-Q8 caches, and the same output oracle. T24
changes only the isolated `COLI_CUDA_DN_PAIR=1` execution path.

| Decode metric | T16 serial exact-Q8 | T24 pair issue/join | Change |
| --- | ---: | ---: | ---: |
| DeltaNet | 31.36 ms/token | 24.38 ms/token | -22.3% |
| Total | 70.81 ms/token | 64.04 ms/token | -9.6% |
| Step total | 71.3 ms/token | 64.6 ms/token | -9.4% |
| Reported speed | 11.84 tok/s | 12.92 tok/s | +9.1% |
| Attention | 5.46 ms/token | 5.52 ms/token | +1.1% |
| MoE total | 30.63 ms/token | 30.79 ms/token | +0.5% |

The DeltaNet-only reduction is the expected signature: QKV/Z now share one
upload and their GPU work overlaps the independent CPU B/A projections. The
small changes in the other phases are ordinary run-to-run variation, not a
claim of improvement.

## Safety

The guard observed 45 C / 46 C peaks, fans at 2,000–2,100 RPM, no corrective
action, zero VRAM during cooldown, no cooldown failure, and restoration of the
original 250 W limits. A post-run read-only snapshot confirmed 38 C / 40 C,
P8, zero VRAM, and 250 W limits.

## Decision

Promote this exact DeltaNet execution-path optimization in the experimental
Colibrì line only. Production remains untouched. Re-profiled T24 now places
shared MLP (17.12 ms/token) and routed MoE (30.79 ms/token) ahead of attention
(5.52 ms/token), so the next candidate is the analogous exact shared gate/up
issue/join path, followed by an attention dependency audit.

## Evidence

`research/results/raw/T24-qwen-deltanet-pair-67da18c4-4922-4bf3-92f9-b11140c7a19b.{json,stderr.txt,stdout.txt}`.
