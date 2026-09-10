# E037 / T45 — exact Qwen shared/expert timeline profile

## Hypothesis

The exact-Q8 shared MLP is reported as a large inclusive subphase, but it is
issued after resident expert groups and uses a separate GPU-0 stream. It may
already finish before GPU-0's routed-expert D2H completion. If so, speeding the
shared MLP cannot shorten the MoE critical path; it may instead increase the
contention that rejected T26's shared gate/up pair.

## One-variable change

An isolated T44 source copy adds default-off
`COLI_SHARED_EXPERT_TIMELINE=1`. It records reusable CUDA events around the
unchanged three synchronous shared-Q8 matvec calls and after the existing
GPU-0 resident-expert D2H. After the unchanged `qt_take` synchronization, it
reports the dense-stream interval and the signed interval from shared-stream
end to expert-group completion. Positive means the routed expert still ran
after the shared MLP ended. The profile adds no dependency, ordering, kernel,
weight, routing, or arithmetic change.

## Acceptance gates

- The canonical 64-output stdout SHA-256 remains
  `5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.
- Exactly 2,520 timeline windows must be reported: 63 decoded tokens times 40
  MoE layers. Missing or failed cross-stream event reads fail the guarded run.
- Standard dual preflight, 125 W/card, cooldown, release, and restoration
  protocol must pass.

## Prediction and decision gate

The expected result is a positive mean end-to-expert interval, showing that the
shared stream is normally hidden by GPU-0 routed experts. If it is consistently
negative, shared work extends the critical path and deserves a fresh,
contention-aware redesign. This is attribution only; its instrumented speed is
not a configuration-selection result.
