# T40 — exact Qwen DeltaNet per-head first-touch control

Status: rejected. The experimental source achieved verified two-node placement
and byte-identical generated output, but it substantially regressed the
adjacent unchanged T24 control.

| Measure | Adjacent T24 control | T40 first-touch | Difference |
| --- | ---: | ---: | ---: |
| Exact stdout SHA-256 | matched | matched | none |
| Engine rate | 12.90 tok/s | 8.21 tok/s | -36.4% |
| Total decode | 63.93 ms/token | 107.56 ms/token | +68.2% |
| DeltaNet | 24.41 ms/token | 55.64 ms/token | +127.9% |
| Attention | 5.56 ms/token | 7.66 ms/token | +37.8% |
| MoE total | 30.61 ms/token | 40.53 ms/token | +32.4% |
| LM head | 3.35 ms/token | 3.73 ms/token | +11.3% |

The source's Linux page query reported all 960 recurrent head slices with
known placement: 599 on NUMA node 0 and 361 on node 1. Thus the locality
mechanism actually engaged; it did not merely set an ignored environment
variable. Nevertheless its immediate unchanged-T24 A/B control was 12.90
tok/s, ruling out general host drift as an explanation for T40's 8.21 tok/s.
The T40 patch remains experimental and default-off; do not integrate it.

The broad stage regressions mean this experiment does not establish a simple
one-component causal story beyond the retention decision. With both P40s and
the dense/GPU-facing path on node 0, moving only recurrent state to workers on
both sockets creates an unfavorable overall placement. Further NUMA changes
need a complete allocation/affinity design and are not the next optimization
priority.

Safety passed. T40's guarded run peaked at 48 C / 45 C and 9,725 MiB / 7,895
MiB; its adjacent T24 control peaked at 47 C / 46 C and 7,893 MiB / 7,893 MiB.
Both ran under 125 W/card caps, released allocations, completed cooldown, and
restored 250 W/card limits with fans at 2,000–2,100 RPM.

Raw evidence:

- `research/results/raw/T40-qwen-dn-firsttouch-349b3acd-2db6-4086-9b71-4105baf063e8.{json,stderr.txt,stdout.txt}`
- `research/results/raw/T40-adjacent-t24-control-92b1a957-e7c5-408f-93b8-f7f5c22c0ec1.{json,stderr.txt,stdout.txt}`
