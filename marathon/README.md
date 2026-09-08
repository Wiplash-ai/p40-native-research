# P40 research marathon controller

This controller is the safe bridge between Wiphand scheduling and a native Codex thread. It does **not** use Wiphand's `codex_article` executor: that executor creates a fresh unsandboxed article job and cannot resume a native thread.

The controller uses the local Codex app-server protocol to read the ChatGPT-backed Codex rate limit, refuse low headroom, start or resume exactly one native thread, attach a thread goal, and dispatch exactly one gated task per run. It never redeems rate-limit reset credits. State and JSONL protocol logs live under this directory and are intentionally ignored by Git.

## Task policy

| Task | Allowed from marathon | Completion proof required before next task |
|---|---|---|
| T01 | Local, workspace-write implementation only | `research/results/T01-acceptance.json` with `status: "pass"` |
| T00 | Serial read-only server telemetry only | `research/results/T00-thermal-investigation.md` with `Acceptance decision: pass` |
| T03 | Source/CPU fixture work only | `research/results/T03-acceptance.json` with `status: "pass"` |
| T02+ | Blocked by this controller | A human-reviewed supervisor extension after T00/T01/T03 evidence exists |

The controller intentionally stops before the first GPU workload. This keeps the thermal and benchmark safeguards meaningful; it is not a loop to bypass either safety or product limits.

The initial Wiphand executor mounts only the research repository and Codex home. It receives neither SSH material nor `/dev/nvidia*` devices. A later, separately reviewed T00 executor will receive a dedicated forced-command read-only telemetry key, never the user's general SSH directory. T02+ must run guarded GPU commands on the server itself after the documented gates pass.

## Local commands

```sh
scripts/codex_marathon.py status
scripts/codex_marathon.py run --task auto
scripts/codex_marathon.py run --task T01
```

`run` may consume Codex usage. It should be launched by the Wiphand Docker request only after its dry run proves the mounted Codex home and repository paths are correct.
