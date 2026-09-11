# T47 — Qwen speculative-verifier feasibility gate

Status: complete decision gate. No GPU workload was run. Production Colibrì
remains unchanged by this conclusion.

The sole remaining design that could plausibly clear a 20% end-to-end gain is
external speculative decoding: a small draft proposes several tokens and the
Qwen target scores them together. The actual engine does not presently expose
the needed verifier.

- `generate()` is serial: choose one token, then call `step(..., 1, ...)`.
- `step(..., S, ...)` returns only the final row's logits.
- `tf_nll()` scores a known continuation through repeated single-token
  `step(..., 1, ...)` calls.
- There is no Qwen MTP/draft/acceptance/state-snapshot protocol in the engine
  or its OpenAI-serving path.
- The multi-row forward still performs routed MoE per row; no candidate-batch
  expert verifier exists.

A real implementation would require target changes well beyond an optimization
control: expose per-position logits, add deterministic accept/replay, snapshot
and restore attention KV plus DeltaNet recurrent state, implement batch-aware
routed MoE, and measure a tokenizer-compatible draft's acceptance rate. The
15-token T46 prefill timing is not evidence that this complete system wins.

Decision: no evidence supports spending further P40/Qwen optimization time on
a greater-than-20-percent outcome. Stop this line of experiments after T46.
The accepted exact-Q8 T24 configuration is the deployment candidate; all
rejected or instrumentation-only paths remain out of production.
