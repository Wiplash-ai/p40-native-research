# E039 / T47 — Qwen speculative-verifier feasibility gate

## Question

Could external draft-model speculative decoding plausibly improve the pinned
Qwen3.6 exact path by more than 20% end to end on these two P40s?

This is the sole remaining architectural candidate after the exact-path
profiles. It would need a cheap draft model, a target verifier that scores a
candidate suffix in one forward pass, rollback of both attention KV and
DeltaNet recurrent state on rejection, and measured acceptance high enough to
overcome draft and verifier work.

## Evidence

The exact engine has a batched `step(Model *, ids, S, pos_base)` function, but
returns logits only for the last row. Its ordinary `generate()` loop selects
one argmax token then calls `step(..., 1, ...)` for every subsequent token.
The only existing teacher-forced routine, `tf_nll()`, likewise advances with
one `step(..., 1, ...)` per scored token. There is no Qwen draft, MTP,
acceptance, candidate-logit, or state-snapshot interface in this engine or
the OpenAI server path.

The existing multi-row forward is not a verifier hidden behind another name:
the Qwen MoE path iterates each row and uses the small decode-scale expert
issue/take interface. A usable verifier would therefore require new
per-position logits, candidate state snapshots/restores, deterministic
accept/replay behavior, and a new target-batch MoE execution path. It is not a
flag, a kernel choice, or a small integration of an Ollama draft model.

T46's real-Qwen prefill timing is not acceptance evidence. It shows a
15-token prompt forward of 445 ms, while the retained exact decode control is
about 64 ms/token; it does not measure candidate acceptance, rollback, or a
batch verifier's end-to-end cost. Without those measurements and a compatible
draft model, a greater-than-20-percent net claim would be speculation.

## Decision

Stop the P40 Qwen kernel/execution-path program here. Do not build a new
speculative-verifier subsystem for this model under the current optimization
scope. Preserve the accepted exact-Q8 T24 path for production serving and
move experimental effort to the separate small-model swarm/search system,
where heterogeneous executors can be evaluated with an explicit evidence
budget.
