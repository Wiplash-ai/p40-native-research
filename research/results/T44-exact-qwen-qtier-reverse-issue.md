# T44 — exact Qwen expert-tier reverse issue order

Status: complete and rejected. Production Colibrì remains unchanged.

| Metric | T24 unchanged control | T44 reverse issue order |
| --- | ---: | ---: |
| Decode rate | 12.90 tok/s | 12.83 tok/s |
| Total decode time | 63.93 ms/token | 64.17 ms/token |
| MoE stage | 30.61 ms/token | 30.95 ms/token |

`COLI_QTIER_REVERSE_ISSUE=1` made device 1 issue before device 0 while keeping
the established device-0 then device-1 result collection and all numerical
work unchanged. The active marker appeared and canonical stdout SHA-256
remained `5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.

This single control does not establish a meaningful speed difference, but it
does falsify the hypothesized direction: reversing the small host enqueue
skew did not produce a gain. Retain the normal issue order and do not add a
parallel host join based on this result.

Safety passed under 125 W/card. Maximum GPU temperatures were 44 C / 44 C,
peak allocation was 7,893 MiB/card, fans remained at 2,000–2,100 RPM, weights
were released, cooldown completed, and limits returned to 250 W/card.

Raw evidence: `research/results/raw/T44-qwen-qtier-reverse-issue-e9435a11-13ae-4502-8d96-4f410d7cabdd.{json,stderr.txt,stdout.txt}`.
