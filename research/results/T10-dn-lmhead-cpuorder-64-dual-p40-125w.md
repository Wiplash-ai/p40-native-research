# T10 — Exact-Q8 DeltaNet plus LM-head sustained Qwen comparison

T10 is the only 64-output comparison of the accepted T09 implementation. It
changes only `N_NEW=64` and preserves the pinned model, prompt, two-P40
layout, 125 W/card cap, expert cache, CPU placement, and numerical oracle.

| Metric | T04 control | T07 DeltaNet | T10 DeltaNet + LM head |
| --- | ---: | ---: | ---: |
| Engine output rate | 1.52 tok/s | 2.23 tok/s | 3.59 tok/s |
| Decode rate | 1.58 tok/s | 2.29 tok/s | 3.76 tok/s |
| DeltaNet | 330.63 ms/token | 60.36 ms/token | 47.91 ms/token |
| LM head | 146.83 ms/token | 177.08 ms/token | 4.04 ms/token |
| Attention | 85.89 ms/token | 113.58 ms/token | 123.78 ms/token |
| MoE total | 67.57 ms/token | 83.12 ms/token | 90.27 ms/token |
| Total decode step | 630.92 ms/token | 436.3 ms/token | 267.2 ms/token |

The 419-byte generated response exactly matched the fixed SHA-256 oracle:
`5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.
The engine confirmed the exact-Q8 cache for 90 DeltaNet matrices plus the LM
head on GPU0. All 10,240 experts were resident across the two cards, with no
CPU expert miss or swap. No CUDA diagnostic or CPU fallback occurred.

T10 is 1.61x faster than T07 and 2.36x faster than the accepted T04 64-output
control. Attention and MoE are now the measured performance ceiling; the LM
head is not.

Safety: peak sampled GPU0/GPU1 temperatures were 47 C / 44 C and peak VRAM
was 9,343 / 7,895 MiB. Fans held 2,000–2,100 RPM. The guard completed its
cooldown to <=40 C, released allocations, and restored both 250 W caps.

Raw evidence:

- [guard record](raw/T10-qwen-lmhead-cpuorder-61c09de3.json)
- [engine output](raw/T10-qwen-lmhead-cpuorder-61c09de3.stdout.txt)
- [engine timers](raw/T10-qwen-lmhead-cpuorder-61c09de3.stderr.txt)
