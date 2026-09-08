# T05E — CPU-reduction-order Q8 DeltaNet out kernel

Date: 2026-09-08. Status: **numerical pass; performance below keep threshold**.

T05E is a standalone sm_61 CUDA prototype for only the Qwen-shaped Q8
4096x2048 dn_out projection. It keeps Qwen's 32 independent FMA streams and
the same final reduction tree. It does not alter Colibri or the generic CUDA
backend.

| Metric | Result |
| --- | ---: |
| Q8 weights | 8 MiB |
| First GPU call, initialization + upload | 1.694 ms |
| CPU median | 0.1400 ms/call |
| GPU complete median | 0.1247 ms/call |
| Observed speed ratio | 1.123x |
| Maximum absolute error | 0 |
| Maximum relative error | 0 |
| Numerical gate | pass, bit-identical |

This proves the P40 can reproduce the CPU's Q8 arithmetic when the operation
order is controlled. However, 12.3% is below the project's 15% keep threshold,
and it covers only the out projection. It is therefore not a candidate for
Qwen integration yet.

## Next test

T05F changes only the final reduction implementation: use warp shuffles rather
than shared memory plus a barrier, while preserving the same 32 input FMA
streams and addition order. It must remain bit-identical and exceed the keep
threshold before the complete DeltaNet chain is considered.

## Safety

GPU0 stayed at or below 34 C, every fan remained 2,000–2,100 RPM, no critical
fan event appeared, allocations were released, the 250 W power limit was
restored, and the enforced cooldown passed.

Raw evidence:

- [durable guard record](raw/T05E-cpuorder-b75d4c64.json)
- [fixture output](raw/T05E-cpuorder-b75d4c64.stdout.txt)
