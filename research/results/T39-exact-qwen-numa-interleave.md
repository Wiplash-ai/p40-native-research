# T39 — exact Qwen blanket NUMA-interleave control

Status: rejected; exact output and thermal controls passed, but OS-level
blanket interleaving substantially regressed fixed-output Qwen decoding.

| Measure | T35 / default policy | T39 / `--interleave=all` | Difference |
| --- | ---: | ---: | ---: |
| Exact stdout SHA-256 | matched | matched | none |
| Engine rate | 12.88 tok/s | 9.84 tok/s | -23.6% |
| Total decode | 64.20 ms/token | 86.60 ms/token | +34.9% |
| DeltaNet | 24.39 ms/token | 44.94 ms/token | +84.3% |
| DeltaNet `l2n+rec` | 6.8 ms/token | 22.5 ms/token | +230.9% |
| Attention | 5.44 ms/token | 6.06 ms/token | +11.4% |
| MoE total | 31.02 ms/token | 32.23 ms/token | +3.9% |
| LM head | 3.35 ms/token | 3.37 ms/token | +0.6% |

T39 preserved the canonical stdout SHA-256
`5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.
Its only process-level difference from T35 was the fixed
`/usr/bin/numactl --interleave=all` prefix. The 20.55 ms/token DeltaNet
increase dominates the 22.40 ms/token whole-token regression, so retain the
default OS memory policy and do not use generic interleaving for this hybrid
CPU/GPU path.

Safety passed: two idle preflights more than 60 seconds apart showed empty
GPUs and 2,000–2,100 RPM fans. The 125 W/card guard peaked at 45 C / 45 C and
9,725 MiB / 7,895 MiB, released allocations, completed cooldown, and restored
both cards to 250 W.

Raw evidence: `research/results/raw/T39-qwen-numa-interleave-644dfa59-e1e1-49f7-9cc1-2b2f9abcf8ed.{json,stderr.txt,stdout.txt}`.
