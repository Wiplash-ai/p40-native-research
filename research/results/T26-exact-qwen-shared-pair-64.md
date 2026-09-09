# T26 — exact-Qwen shared-MLP pair 64-token comparison

Date: 2026-09-09

## Matched comparison

T26 differs from accepted T24 only by `COLI_CUDA_SHARED_PAIR=1`. Both use the
same Qwen checkpoint, prompt, 64 outputs, two P40s, 125 W/card guard,
exact-Q8 caches, accepted DeltaNet QKV/Z pair, and output oracle.

| Decode metric | T24 DeltaNet pair | T26 + shared gate/up pair | Change |
| --- | ---: | ---: | ---: |
| Shared MLP | 17.12 ms/token | 17.47 ms/token | +2.0% |
| MoE total | 30.79 ms/token | 31.73 ms/token | +3.1% |
| Total | 64.04 ms/token | 65.26 ms/token | +1.9% |
| Reported rate | 12.92 tok/s | 12.68 tok/s | -1.9% |

The 64-token stdout SHA-256 exactly matched the fixed oracle:
`5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.

## Decision

Reject the simple same-stream shared gate/up issue/join path. It preserves
exact output but increases end-to-end latency. The evidence is consistent with
GPU-0 contention against the concurrently issued routed-MoE work; that is an
inference, not a separately measured causal proof. Do not integrate this path
or spend further work micro-optimizing it. The accepted experimental DeltaNet
pair remains enabled in its isolated engine; production Colibri remains
untouched.

## Safety

The guard passed without action. Peak sampled temperatures were 45 C and 46 C;
all fans remained 2,000–2,100 RPM. VRAM was released, cooldown completed, and
both caps were restored to 250 W.

## Evidence

`research/results/raw/T26-qwen-shared-pair-898a2a0a-807c-42b6-a06b-347a2acdcd8d.{json,stderr.txt,stdout.txt}`.
