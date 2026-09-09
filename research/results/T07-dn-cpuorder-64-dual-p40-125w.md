# T07 — Exact-Q8 DeltaNet Qwen 64-output comparison

T07 is the accepted T06C experimental binary with only `N_NEW=64` changed.
It retained the fixed model, prompt, two-P40 layout, 125 W/card guard, Q8
expert cache, CPU placement, and 64-output oracle used by T04.

| Metric | T04 control | T07 exact-Q8 DeltaNet | Change |
| --- | ---: | ---: | ---: |
| Engine output rate | 1.52 tok/s | 2.23 tok/s | 1.47x |
| Decode rate | 1.58 tok/s | 2.29 tok/s | 1.45x |
| DeltaNet | 330.63 ms/token | 60.36 ms/token | 5.48x lower |
| Total decode step | — | 436.3 ms/token | — |

The 419-byte output SHA-256 exactly matched the accepted control:
`5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.
The log confirms all 90 exact-Q8 DeltaNet matrices were resident on GPU0 and
the two-GPU expert tier held all 10,240 experts with no CPU miss or swap.

The remaining measured decode costs are LM head 177.08 ms/token, attention
113.58, and MoE 83.12. This result validates only the DeltaNet change; it does
not establish a benefit from CUDA attention, dense CUDA flags, cache resizing,
or another precision change.

Safety: sampled GPU0/GPU1 peaks were 46 C / 45 C and peak VRAM was 8,855 /
7,895 MiB. All fans remained 2,000–2,100 RPM, no fresh system event occurred,
the cards reached <=40 C during guard cooldown, allocations were released,
and original 250 W caps were restored.

Raw evidence:

- [guard record](raw/T07-qwen-dn-cpuorder-91f547eb.json)
- [engine output](raw/T07-qwen-dn-cpuorder-91f547eb.stdout.txt)
- [engine timers](raw/T07-qwen-dn-cpuorder-91f547eb.stderr.txt)
