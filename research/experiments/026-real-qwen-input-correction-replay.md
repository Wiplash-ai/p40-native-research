# E026 / T33 — CPU-only replay of real-Qwen input correction

Status: complete — CPU-only replay accepted as a ranking screen, not a runtime
optimization.

T33 reads the guarded sidecar only: it opens no checkpoint, initializes no
CUDA, and cannot modify Colibri. It reconstructs captured raw gate/up, hidden,
and exact-down output before evaluating groupwise-Q8 gate/up input correction.

| Ranking | Scope |
| --- | --- |
| `residual_magnitude` | Existing largest-residual baseline. |
| `downnorm_proxy` | Down-aware diagonal Jacobian proxy across all 2,048 columns. |
| `effect_top64` | Nonlinear output-effect rerank inside the fixed top-64 residual set. |

All rankings use the exact W4 down projection at K = 0, 4, 8, 16, and 32.
Steps 0–1 are calibration; steps 2–3 are untouched holdout. `effect_top64` is
not a global oracle: its candidate set is explicitly bounded before scoring.

No replay result changes inference. A GPU shadow candidate needs a better
holdout result than magnitude at matching K, no holdout record above 5%, and a
priced implementation cost.

## Result

CPU reconstruction against captured GPU values was sound: on holdout, raw
gate/up reconstruction was `6.03e-7`, hidden was `8.47e-7`, and exact-down
output was `3.66e-7` aggregate relative L2. That validates using the sidecar
for a real-Qwen correction screen.

| Holdout method | K | Aggregate relative L2 | p99 | Maximum | Above 5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Residual magnitude | 16 | 1.700% | 3.200% | 3.200% | 0 / 48 |
| Down-norm proxy | 16 | 1.650% | 2.952% | 2.952% | 0 / 48 |
| Bounded nonlinear effect | 16 | 1.648% | 2.994% | 2.994% | 0 / 48 |
| Residual magnitude | 32 | 1.552% | 2.809% | 2.809% | 0 / 48 |
| Down-norm proxy | 32 | 1.512% | 2.711% | 2.711% | 0 / 48 |
| Bounded nonlinear effect | 32 | 1.504% | 2.719% | 2.719% | 0 / 48 |

Both influence-aware rankings improve aggregate holdout error only modestly;
they do not establish a practical speed/quality point. The next meaningful
control is a single exact-output GPU shadow using one fixed candidate (choose
the cheaper down-norm proxy at K=16), followed by an explicit cost profile.
Do not integrate it or infer end-to-end language quality from replay alone.
