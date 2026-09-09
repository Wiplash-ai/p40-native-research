# E011 / T16 — Qwen exact-Q8 shared-MLP 64-output comparison

Status: pass.

## Hypothesis

T15's exact 16-output canary moved the intended shared MLP phase from 69.86
to 17.14 ms/token. If that result is not merely cold-process/short-run noise,
the same isolated binary should preserve the previously accepted 64-output
text while improving T13's 151.14 ms/token decode total.

## One change

T16 imports the T15 guard/binary and replaces only the pinned `N_NEW=16` with
`N_NEW=64`. It retains the source SHA, model snapshot, prompt, two-GPU
assignment, exact-Q8 cache flags, 125 W/card policy, watchdog, fan checks,
cooldown, and allocation/power restoration. It rejects any stdout other than
the established 64-output SHA-256
`5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.

## Static acceptance

The dedicated forced-command identity accepts no model/runtime arguments; its
local guard/client acceptance suite passes. The runtime test needs two
independent cool, zero-VRAM, zero-utilization samples at least 60 seconds
apart before launch.

## Guarded result

The fixed run (`run_id` `6047e0b6-0324-4b8b-bcec-f5a13495af5b`) passed all
gates. Its stdout SHA-256 exactly matched the established 64-output oracle:
`5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`. The
required marker listed every exact-Q8 cache component, including all 120
shared MLP matrices; there was no CUDA diagnostic or fallback.

Engine rate was 11.84 tok/s (5.4 seconds for 64 outputs; TTFT 0.89 s), versus
T13's 6.07 tok/s under its otherwise matching 64-output profile. Across 63
decode steps, total time was 70.81 ms/token, versus 151.14 in T13. The stable
phase breakdown was DeltaNet 31.36 ms, attention 5.46 ms, MoE 30.63 ms
(shared 17.17; router 7.54), and LM head 3.35 ms/token.

Every expert remained resident, with zero actual CPU misses or swaps. Sampled
peaks were 45 C / 45 C and 8,801 / 7,895 MiB. Fans held 2,000--2,100 RPM;
the runner released allocations, completed cooldown to 38 C / 40 C, and
restored both 250 W limits.

## Decision gate

T16 met every gate: 1.95x engine-rate improvement and 2.13x lower decode time
than T13 while preserving exact text. Retain the exact shared path. The next
work is a standalone Pascal W4A8/DP4A expert-GEMV control for the remaining
30.63 ms/token MoE path—not a direct production MoE rewrite.
