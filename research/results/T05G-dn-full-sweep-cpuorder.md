# T05G — exact-order full Q8 DeltaNet sweep

Date: 2026-09-08. Status: **pass for the synthetic Q8-projection phase**.

T05G applied T05E's CPU-order-preserving Q8 CUDA kernel to the complete
synthetic Qwen working set: 30 distinct DeltaNet triplets (90 projections,
960 MiB of Q8 weights). This is the cache-resistant performance control that
the isolated 8 MiB tests could not provide.

| Metric | Result |
| --- | ---: |
| CPU median | 36.6309 ms/sweep |
| GPU complete median | 13.1462 ms/sweep |
| CPU/GPU ratio | 2.786x |
| First sweep, including initialization/upload | 148.572 ms |
| Cached tensors | 90 / 1,008,353,280 bytes |
| Maximum absolute / relative error | 0 / 0 |
| Numerical gate | bit-identical pass |

The three individual CPU samples were 36.6342, 36.6309, and 36.4618 ms. The
GPU complete samples, including the activation/result transfers, were 13.3091,
13.1187, and 13.1462 ms. The result exceeds the predeclared 15% keep threshold
without relaxing Qwen's arithmetic contract.

## Scope and decision

This promotes the exact Q8 projection kernel only. It does **not** establish
an end-to-end Qwen speedup: DeltaNet also has CPU FP32 `dn_b`/`dn_a` work,
recurrent/state work, and host control flow. It does establish that the 90 Q8
projection calls are a valid candidate for an opt-in experimental Qwen path.

T06 is therefore a source-isolated, one-setting integration canary in the
clean experimental Colibri worktree. It must preserve CPU fallback, execute
only Q8 `dn_qkv`, `dn_z`, and `dn_out` through this exact kernel, retain the
existing CPU FP32/recurrent work, prove byte-identical generated output against
the fixed 16-token control, and then measure the same 64-output profile. No
other knobs, formats, GPU sharding, or mathematical changes belong in that
comparison.

## Safety

The GPU0-only 125 W guard passed. Its sampled peak was 34 C (GPU1 35 C), every
fan remained at 2,000–2,100 RPM, no cleanup action was required, and the guard
reported restoration. A post-run independent check found both cards at zero
MiB, 33 C / 35 C, and restored 250 W caps.

Raw evidence:

- [durable guard record](raw/T05G-full-sweep-92726f48.json)
- [fixture output](raw/T05G-full-sweep-92726f48.stdout.txt)
