# E011 / T16 — Qwen exact-Q8 shared-MLP 64-output comparison

Status: static acceptance pass; guarded GPU test pending a fresh cool/idle
preflight.

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

## Decision gate

T16 is a pass only on exact output, required cache marker, zero fallback,
thermal/fan/cleanup pass, and total decode improvement over T13. If it passes,
the remaining principal bottleneck is the 30.70 ms/token MoE path, so the next
work is a standalone Pascal W4A8/DP4A expert-GEMV control—not a direct
production MoE rewrite. If it fails parity/safety, retain T13 and investigate
the smallest failing path.
