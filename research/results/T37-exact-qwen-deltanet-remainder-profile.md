# T37 — exact Qwen DeltaNet remainder profile

Status: passed as instrumentation evidence; production remains unchanged.

T37 retained the accepted exact-Q8 QKV/Z pair and added opt-in timing around
the rest of DeltaNet. The final output projection uses the same exact-Q8 CUDA
kernel, buffers, copies, and return path as the retained configuration. Timing
adds work, so its 12.75 tok/s engine rate is not a performance comparison.

| Interval | Per layer | Per decode token |
| --- | ---: | ---: |
| Q/K repeat + L2 normalization, host | 0.036 ms | 1.080 ms |
| Recurrent delta-rule state update, host | 0.191 ms | 5.730 ms |
| Gated RMS normalization, host | 0.072 ms | 2.160 ms |
| Final output-projection call, host | 0.133 ms | 3.990 ms |
| Final-output H2D | 0.01427 ms | 0.428 ms |
| Final-output exact-Q8 kernel | 0.09278 ms | 2.784 ms |
| Final-output D2H | 0.01384 ms | 0.415 ms |
| Final-output GPU total | 0.12089 ms | 3.627 ms |

All 1,890 expected host and GPU records were finite. The GPU total equals its
component sum, and the canonical stdout SHA-256 remained
`5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.

The final output-projection's host call is only 0.36 ms/token above the GPU
event total, while the kernel itself is 76.8% of that stream. CUDA Graphs or
launch-only work therefore has little headroom; it cannot remove the 2.784
ms/token exact-Q8 arithmetic. The next exact-path target is the 5.730
ms/token CPU recurrence, not the output boundary.

The total DeltaNet measurement was 25.17 ms/token: pair projection 8.0,
convolution 4.1, L2-normalize plus recurrence 6.8, and gated norm plus output
6.2. The detailed counters explain the latter two groups without claiming
their instrumented total is comparable with T35.

Safety controls passed: manual preflights were 35 C / 36 C with all fans at
2,000–2,100 RPM. The 125 W/card guard peaked at 43 C / 43 C and 7,893 MiB /
7,893 MiB, then released allocations, completed cooldown, and restored both
power caps to 250 W/card.

Raw evidence: `research/results/raw/T37-qwen-deltanet-remainder-profile-785746c4-53b1-41ef-a710-f540c4530006.{json,stderr.txt,stdout.txt}`.
