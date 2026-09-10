# T36 — exact Qwen DeltaNet-pair dependency profile

Status: passed as instrumentation evidence; this is not a performance
comparison and production remains unchanged.

The existing accepted exact-Q8 QKV/Z pair path was executed with one opt-in
counter layer. It retains the exact kernels, ordering, pair stream, CPU B/A
overlap, canonical prompt, and fixed 64-token output oracle. The counters add
CUDA-event and host-clock work, so its 12.70 tok/s engine rate must not be
compared with T24/T35.

| Interval | Aggregate | Per pair | Per decode token |
| --- | ---: | ---: | ---: |
| H2D input | 14.714 ms | 0.00779 ms | 0.234 ms |
| QKV exact-Q8 kernel | 264.968 ms | 0.14019 ms | 4.206 ms |
| Z exact-Q8 kernel | 165.791 ms | 0.08772 ms | 2.632 ms |
| D2H outputs | 117.216 ms | 0.06202 ms | 1.861 ms |
| GPU pair stream total | 562.689 ms | 0.29772 ms | 8.932 ms |
| Host pair issue | — | 0.019 ms | 0.570 ms |
| Independent CPU B/A | — | 0.246 ms | 7.380 ms |
| Host pair join | — | 0.044 ms | 1.320 ms |

All 1,890 expected layer/decode pairs produced finite host and CUDA-event
records. The GPU total equals the reported component sum. The canonical stdout
SHA-256 remained
`5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.

The critical interpretation is that CPU B/A is intentionally concurrent with
the GPU pair: the pair's inclusive DeltaNet projection subphase is 9.3
ms/token, close to the 8.93 ms/token GPU-stream interval, rather than the
serial sum of GPU and CPU B/A work. Direct CPU submission and wait cost only
1.89 ms/token. CUDA Graphs could at most attack a fraction of that host work
and cannot remove the 6.84 ms/token of QKV/Z arithmetic or the 2.09 ms/token
of copies; no graph implementation is justified by this profile.

The remaining measured DeltaNet time is the next target: `conv` 4.0,
`l2n+rec` 6.3, and `norm+out` 5.6 ms/token. Before changing one, add equally
strict exact-path attribution for its CPU/GPU boundaries.

Safety controls passed: the two manual preflights were 35 C / 37 C and then
35 C / 37 C, with all fans at 2,000–2,100 RPM. The guarded run peaked at 44 C
/ 44 C and 9,725 MiB / 7,895 MiB. It released both GPUs, completed cooldown,
and restored both power caps to 250 W/card.

Raw evidence: `research/results/raw/T36-qwen-deltanet-pair-profile-ad91ae78-6890-41a3-8c37-05e8f0d0b6d4.{json,stderr.txt,stdout.txt}`.
