# E027 / T34 — real-Qwen down-norm proxy16 correction shadow

Status: complete and rejected; production Colibri is untouched.

T33's per-expert replay selected the down-norm proxy as the cheapest
influence-aware candidate. T34 is the required full execution-path test, not a
promotion. It fixes `K=16` and evaluates, for each groupwise-Q8 gate/up input
coordinate `c`, the score:

```
abs(e_c) * sqrt(sum_i ||W_down[:, i]||^2 *
  (u_i * SiLU'(g_i) * W_gate[i, c] + SiLU(g_i) * W_up[i, c])^2)
```

The shadow computes raw groupwise-Q8 gate/up, ranks all 2,048 input columns,
adds only the selected FP32 residual contributions, and applies the exact W4
down projection. It then queues the unchanged W4/FP32 group path, which is the
only value returned to Qwen.

The guard pins a 16-token prompt/model/engine tuple at 125 W per P40. It
requires the old stdout SHA, exactly 2,397 finite shadow records, and exactly
2,397 GPU-cost records. Cost uses CUDA events around the shadow kernels and a
same-stream exact gate/up+down reference; it intentionally excludes D2H copies.

Accept only if exact output remains unchanged, no numerical/thermal guard
fails, and the new fidelity point justifies its measured cost. A slower shadow
is a rejection, even if its approximation error is low.

## Result

The guard passed exact output, 2,397 finite shadow records, 2,397 cost
records, 125 W/card safety, and cooldown. But the proxy did not reduce the
real-path tail: median relative L2 was 1.664%, p99 7.739%, maximum 11.075%,
and 45 records remained above the 5% gate. Its median CUDA-kernel cost was
3.331 ms per group versus 0.352 ms for the same-stream exact gate/up+down
reference: 9.72x slower (p95 17.07x; maximum 30.24x). Those event values
exclude the shadow D2H copy, so they are a favorable cost comparison.

Decision: reject the all-column down-norm proxy as an execution-path
optimization. The T33 per-expert replay result did not predict routed-group
tail behavior, and the selector is inherently too expensive at decode. Keep
the exact Q8 overlap path as the active performance direction; no proxy16
code is eligible for production or wider language-quality testing.
