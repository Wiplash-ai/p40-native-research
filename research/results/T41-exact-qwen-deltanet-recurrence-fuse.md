# T41 — exact Qwen DeltaNet recurrence-pass fusion

Status: rejected. The fused CPU recurrence emitted its required marker and
preserved the canonical generated output, but it did not outperform the
adjacent unchanged exact-Q8 control.

| Measure | Adjacent T24 control | T41 fused recurrence | Difference |
| --- | ---: | ---: | ---: |
| Exact stdout SHA-256 | matched | matched | none |
| Engine rate | 12.90 tok/s | 12.72 tok/s | -1.4% |
| Total decode | 63.93 ms/token | 64.84 ms/token | +1.4% |
| DeltaNet | 24.41 ms/token | 25.14 ms/token | +3.0% |
| DeltaNet `l2n+rec` | 6.0 ms/token | 6.3 ms/token | +5.0% |
| Attention | 5.56 ms/token | 5.59 ms/token | +0.5% |
| MoE total | 30.61 ms/token | 30.77 ms/token | +0.5% |
| LM head | 3.35 ms/token | 3.34 ms/token | -0.3% |

T41's fixed 64-token stdout SHA-256 is
`5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f` and
stderr confirms `[dn-rec-fuse] active`. It fused state scale with KV
accumulation and state update with output accumulation, retaining the same
scalar expressions and `kk`/`vv` order. The end-to-end result is nevertheless
slower than the immediately preceding unchanged T24 control, so do not retain
or add further numerical validation for this non-winning path.

Safety passed: two idle preflights more than 60 seconds apart preceded the
locked 125 W/card run. It peaked at 45 C / 45 C and 7,893 MiB / 7,893 MiB,
then released allocations, completed cooldown, and restored both cards to 250
W with 2,000–2,100 RPM fans.

Raw evidence: `research/results/raw/T41-qwen-dn-rec-fuse-31c3a614-7b9f-4e03-886a-bb8f3e511936.{json,stderr.txt,stdout.txt}`.
