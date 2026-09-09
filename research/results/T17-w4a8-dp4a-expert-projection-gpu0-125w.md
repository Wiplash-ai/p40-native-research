# T17 — Pascal W4A8 DP4A MoE-projection control

Status: pass as a synthetic arithmetic/kernel control; no Qwen integration.

The test modeled four per-device routed Qwen expert projections with packed
signed W4 weights, Qwen's `2048 -> 512` shape, one FP32 input per expert, and
the same host input/output boundary in both variants. The new path quantized
inputs to Q8 on GPU and used Pascal `IDP.4A.S8.S8`; the control retained W4
weights and FP32 activations.

| Measure | Result |
| --- | ---: |
| Integer W4/Q8 parity | Exact |
| Relative L2 vs W4/FP32 | 0.00384378 |
| W4A8 DP4A median | 0.0609619 ms |
| W4/FP32 median | 0.098694 ms |
| Transfer-inclusive speedup | 1.61895x |
| Tile / experts | 8 warp rows / 4 |

The relative-L2 error is activation-quantization error, not an output-text
measure. It cleared the predeclared 2% control gate and the 1.15x speed gate.
SASS directly contains `IDP.4A.S8.S8` instructions.

GPU0 peaked at 37 C under its temporary 125 W cap; GPU1 was idle at 37 C.
Every fan was 2,000--2,100 RPM. The forced runner released allocations,
completed cooldown, and restored GPU0 to 250 W.

Decision: test only loop unrolling next. Do not alter Colibri, apply this to
Qwen, or make a model-quality claim yet.

Raw evidence: `research/results/raw/T17-w4a8-dp4a-cfb76b1a.*`.
