# Research execution contract

Read README.md, EXECUTION_PLAN.md, research/hardware.md, and the specific assigned task before acting. Execute one task ID at a time. The requested work is authorized; do not repeatedly ask for approval for scoped implementation or routine read-only discovery. A failed thermal gate is a hardware prerequisite, not a permission prompt.

- Production source: `/home/jordanculver/colibri_engine` on excalibur. Preserve that checkout and its binaries. Use a separate source worktree or clone pinned to the recorded commit. Never run `make clean`, pull, rebuild, or change service settings in production while doing an experiment.
- Current phase: discovery complete; benchmark implementation and workload execution pending. Do not launch inference, training, stress tools, clock changes, or fan-control commands merely by reading this repository.
- Sustained workload requires the thermal gates in research/hardware.md. Never hide fan alerts by changing thresholds. Record a failed gate and continue CPU-light implementation/documentation work.
- One benchmark process group, one server-wide lock, one configuration change per comparison. Save original device settings before an authorized temporary power change; restore them on every exit, including watchdog termination. Do not stop unrelated processes.
- Pin source commit, binary hash, dataset/tokenizer/model identity, prompt tokens, token budget, and relevant environment for every run. Never save arbitrary environment dumps containing secrets.
- Every experimental dispatch must log its actual selected kernel and fallbacks. A setting existing in a README is not evidence the selected engine uses it.
- Keep exact W4A32 and Q8-weight/FP32-activation controls. W4A8/DP4A changes activation precision and requires explicit numerical and language-quality results.
- Keep stock routing, expert count, sampling, tokenizer, and checkpoint fixed for runtime optimizations. Architecture experiments may change math but must train/evaluate their own small models.
- No fabricated measurements, extrapolated speedups labeled measured, or claims of novelty. Use `not_measured`, `unsupported`, `thermal_abort`, and `inconclusive` explicitly.
- No giant checkpoint downloads, cloud purchases, public repository creation, upstream PRs, or model uploads as part of this discovery handoff.
- Before ending a task: update the experiment log, list changed paths, state tests actually run, and identify the next task ID.
