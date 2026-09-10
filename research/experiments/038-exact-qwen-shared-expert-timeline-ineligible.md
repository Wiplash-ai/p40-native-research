# E038 / T46 — exact Qwen shared/expert timeline ineligible-window control

T45's fixed trace established that a few layer/token pairs have no GPU-0
routed expert. T46 changes only the profile accounting: those pairs are
reported as `no-gpu0`, while event creation, recording, elapsed-query, and
unexpected collection failures remain fatal. The guarded invariant is now
`valid windows + no-gpu0 == 63 × 40`, with zero real failures. Model math,
event placement, weights, routing, and the 64-token exact-output oracle are
unchanged.
