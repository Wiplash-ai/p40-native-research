# T05D — 30-layer Q8 error attribution

Date: 2026-09-08. Status: **generic format-1 CUDA path rejected for strict
Qwen DeltaNet integration**.

T05D repeated the fixed T05C computation without changing GPU work. It only
added CPU-side error attribution after the first GPU sweep. This distinguishes
the direct qkv/z projections from the out projection and the explicit
CPU-hosted boundary between them.

| Comparison | Correct under fixed gate | Max absolute error | Max relative error |
| --- | --- | ---: | ---: |
| qkv GPU versus CPU | yes | 2.289e-5 | 1.296e-5 |
| z GPU versus CPU | yes | 1.907e-5 | 1.580e-5 |
| out GPU versus CPU | **no** | 9.155e-4 | 4.286e-4 |
| out GPU versus CPU with identical GPU-derived input | **no** | 9.155e-4 | 3.022e-4 |
| CPU out with CPU versus GPU-derived boundary | **no** | 7.324e-4 | 5.817e-4 |

The direct out comparison shows that the generic CUDA kernel's 256-thread
reduction order is independently outside the gate. The boundary comparison
shows that small qkv/z differences then propagate through the nonlinear
DeltaNet path in a real integration. Retuning the acceptance threshold would
hide an end-to-end correctness risk, so it is not allowed.

The observed timing (36.339 ms CPU median / 17.503 ms GPU median) is recorded
only as diagnostic context, not an optimization result.

## Decision

Do not wire the existing generic format-1 API into Qwen DeltaNet. The next
smallest test is T05E: an experimental standalone Q8 kernel for the 4096x2048
out projection that deliberately follows Qwen's 32 AVX lane accumulator and
reduction order. It will test parity before a 30-layer sweep or a source
integration patch. If it cannot meet the fixed gate, reject direct per-layer
Q8 offload rather than loosening accuracy.

## Safety

The guard completed with no thermal or fan abort, GPU0 sampled at 34 C or
below, every fan remained 2,000–2,100 RPM, no allocation remained, GPU0's
250 W cap was restored, and the five-minute cooldown passed.

Raw evidence:

- durable guard record: results/t05d-diagnostic-controls/b313f749-7a55-40a6-817b-69508b39a530.json
- fixture output: results/t05d-diagnostic-controls/b313f749-7a55-40a6-817b-69508b39a530.stdout.txt
