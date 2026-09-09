# E024 / T31 — real-Qwen top-32 groupwise stage-attribution shadow

Status: complete — input-side gate/up quantization owns the remaining tail;
production unchanged.

## Question

T28–T30 establish that groupwise Q8 activation scaling helps but neither
top-8 nor top-32 sparse correction clears the high-error tail. This control
does not choose another K. It separates the two quantized stages using the
same real routed Qwen activations and top-32 formulation.

1. **Input side:** approximate groupwise/top-32 gate/up, then exact W4/FP32
   down.
2. **Hidden side:** exact W4/FP32 gate/up, then approximate groupwise/top-32
   down.

Both results compare against the exact Qwen expert output. The existing exact
W4/FP32 path still returns all model values.

## Gate

The run must keep the fixed exact stdout oracle, report exactly two finite
metric records per routed expert group, and complete the normal thermal guard.
It is attribution, not a quality pass: select the next formulation only after
the worse stage is empirically identified.

## Result

The fixed 16-token stdout SHA-256 exactly matched
`43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a`.
Both streams produced 2,397 finite real routed-expert group records while the
unchanged exact W4/FP32 path supplied the model values.

| Shadow stage | Median relative L2 | p95 | p99 | Maximum | Above 5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Approximate groupwise/top-32 gate/up, exact down | 1.62% | 2.50% | 7.31% | 12.42% | 45 |
| Exact gate/up, approximate groupwise/top-32 down | 1.37% | 2.05% | 2.42% | 2.87% | 0 |

The high-error groups are exclusively in the input-side stream. The next
experiment must therefore test a sensitivity-aware gate/up correction or an
input-side representation change. Do not add more down correction, expand the
generic top-K capacity, measure approximate-output speed, or change production.

Safety passed: 46 C / 47 C sampled peaks, 7,895 MiB/card peak VRAM, healthy
2,000–2,100 RPM fans, allocation release, cooldown, and restoration from 125 W
to 250 W on both cards.
