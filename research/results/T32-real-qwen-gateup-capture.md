# T32 — bounded real-Qwen gate/up capture

T32 ran the isolated `sm_61` binary only. It left production Colibri untouched
and returned the existing exact W4/FP32 expert outputs.

| Check | Result |
| --- | --- |
| Exact 16-token stdout SHA-256 | matched T31 oracle |
| WGCAP records | 96 / 96 |
| Calibration / holdout | 48 / 48 |
| Artifact size | 154,540,864 bytes |
| Artifact SHA-256 | `e6e670f1884c33f43c17b7e8acdab102bf9eeec6d182944f2b6f220db7ff051e` |
| Capture errors | 0 |
| GPU peak temperature | 40 C / 41 C |
| GPU peak VRAM | 7,895 MiB / card |
| Post-run caps | restored to 250 W / card |

The 148 MiB binary sidecar is retained at
`research/results/raw/T32-qwen-gateup.wgcap` locally and at the server guarded
result path. It is excluded from ordinary Git due to hosted-object limits;
small guard, stdout/stderr, and replay evidence are versioned alongside this
record.
