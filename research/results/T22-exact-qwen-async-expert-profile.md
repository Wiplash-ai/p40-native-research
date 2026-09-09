# T22 - exact Qwen asynchronous-expert profile

Status: pass for instrumentation; no production source changed.

An isolated copy of the accepted T16 exact-Q8 source received only persistent
CUDA events around Qwen's asynchronous routed-expert issue/take path. The
fixed 64-token output exactly matched the established oracle, so instrumentation
did not change model text.

| Measure | Result |
| --- | ---: |
| Output SHA-256 | 5909...f35f exact |
| Engine rate | 11.87 tok/s |
| Decode total | 70.46 ms/token |
| MoE | 31.65 ms/token (44.9%) |
| DeltaNet | 29.99 ms/token (42.6%) |
| Attention | 5.47 ms/token (7.8%) |
| LM head | 3.35 ms/token (4.8%) |
| Async group calls / experts | 6,226 / 24,960 |
| Async H2D / kernel / D2H | 90 / 2,433 / 66 ms |
| Async device-time share | 3.5% / 94.0% / 2.5% |

The event totals sum device timelines across both P40s and include prefill, so
they are not directly comparable with decode wall time. They do establish
that the routed expert CUDA section is kernel-dominated rather than
transfer-bound.

## Evidence-backed priority

1. Exact Q8 projection synchronization. DeltaNet's 90 cached projections use
   the CPU-order helper, which uploads input, launches one matvec, downloads
   output, then synchronizes for every projection. DeltaNet is 29.99
   ms/token. Shared MLP uses the same helper and accounts for 16.80 ms/token
   within MoE. The smallest next test should reduce launches and host-device
   round trips while preserving exact Q8 arithmetic.
2. Routed-expert kernel work. The routed portion is not PCIe-limited: 94.0%
   of its recorded device timeline is kernels. Plain per-row W4A8 is rejected
   by T21 quality data, so any new kernel work needs a different numerical
   formulation and an isolated quality control.
3. Attention projection synchronization. Attention is smaller at 5.47
   ms/token but calls the same synchronous CPU-order helper. It is a later
   beneficiary of a validated exact projection batch/graph path.

Safety passed: 44 C peak on each GPU, 125 W temporary caps, stable
2,000--2,100 RPM fans, released allocations, cooldown to <=40 C, and restored
250 W limits.

Raw evidence: research/results/raw/T22-qwen-async-profile-846a85bb-38ab-4f45-a54d-80b24cdf0780.*.
