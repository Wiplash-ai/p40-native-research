# E020 / T27 — exact-Q8 attention dependency profile

Status: complete; reject async-attention implementation as low-headroom.

## Question

Can the existing exact-Q8 attention path benefit from reduced launch or
synchronization overhead without changing model mathematics?

T26 measured attention at 5.48 ms/token, only 8.4% of total decode. Therefore
even a perfect attention elimination is capped below the accepted DeltaNet
gain. No optimization is justified until the serial Q, K, V, and O projections
are separated from the CPU attention middle on the real checkpoint.

## Source dependency audit

At `qwen36.c:1813` the decode path invokes, in order, `q = x Wq^T`,
`k = x Wk^T`, and `v = x Wv^T`, each through `matmul_d`. They share the exact
same input but are separate calls. Q and K are required before Q/K RMSNorm,
RoPE, scoring, and softmax. V is not used until after softmax, when it is read
from the KV cache. The O projection depends on the gated context. This means a
future execution candidate may overlap V with CPU Q/K preparation, but cannot
arbitrarily reorder Q/K or O.

Q is 2048-to-8192; K and V are each 2048-to-512; O is 4096-to-2048. The
initial profile must measure transfer-inclusive exact-Q8 call intervals, not
infer their cost from dimensions.

## T27 control

- Exact output math and execution order are unchanged.
- Default-off `COLI_ATTN_PROFILE=1` adds decode-only wall-clock buckets for Q,
  K, V, CPU attention middle, and O.
- Inherit only the accepted DeltaNet QKV/Z pair from T24. The rejected shared
  pair is absent.
- Run the fixed 64-token real-Qwen oracle under the normal two-preflight,
  125 W/card, fan, cooldown, raw-evidence protocol.
- Accept only instrumentation/evidence. This run must not claim a speedup.

## Decision gate

Build an async attention control only if the exact-Qwen profile identifies a
meaningful synchronizing projection interval that can overlap a proven
independent CPU region. Otherwise record attention as a low-headroom target and
return to a higher-impact full-path bottleneck.

## T27 result

The guarded fixed-64 actual-Qwen run matched the output oracle exactly. Its
exact-Q8 transfer-inclusive subprofile was Q 1.66, K 0.36, V 0.35, CPU middle
2.25, and O 1.15 ms/token (attention total 5.80 ms/token). Timing-only source
changes made the total unsuitable for a speed comparison.

Q/K/O have no independent CPU work that can hide their projection intervals.
V can overlap only Q/K CPU preparation. Even treating K+V as fully hideable
gives a 0.71 ms/token upper bound, below 1.1% of accepted T24 total decode and
before implementation overhead. Reject an async-attention control. Resume the
user-ranked groupwise/outlier W4A8 real-expert shadow investigation instead.
