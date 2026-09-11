"""Read-only Stage-0 capacity discovery for the server-local model stack."""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class CapacityProbeError(ValueError):
    """The fixed read-only capacity probe did not return its expected schema."""


@dataclass(frozen=True)
class CapacitySnapshot:
    collected_at: str
    qwen_service: str
    gpus: list[dict[str, Any]]
    ollama_models: list[dict[str, Any]]
    ai_ssd: str

    def json(self) -> dict[str, Any]:
        return asdict(self)


REMOTE_SCRIPT = r'''set -eu
python3 - <<'PY'
import csv
import json
import subprocess

def text(argv):
    return subprocess.run(argv, capture_output=True, text=True, check=False).stdout.strip()

service = text(["sudo", "systemctl", "is-active", "colibri-qwen36.service"])
gpu_rows = list(csv.reader(text([
    "nvidia-smi", "--query-gpu=index,name,temperature.gpu,power.draw,memory.used,memory.total",
    "--format=csv,noheader,nounits",
]).splitlines()))
gpus = [dict(zip(("index", "name", "temperature_c", "power_w", "memory_used_mib", "memory_total_mib"), row)) for row in gpu_rows]
ollama = json.loads(text(["curl", "--fail", "--silent", "--show-error", "http://172.17.0.1:11434/api/tags"]))
ai_ssd = text(["sh", "-c", "df -h /mnt/ai-ssd | tail -1"])
print(json.dumps({"qwen_service": service, "gpus": gpus, "ollama": ollama, "ai_ssd": ai_ssd}, sort_keys=True))
PY
'''


def ssh_argv(host: str, timeout: int) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}", host, REMOTE_SCRIPT]


def parse_snapshot(text: str) -> CapacitySnapshot:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise CapacityProbeError("probe returned invalid JSON") from error
    required = {"qwen_service", "gpus", "ollama", "ai_ssd"}
    if not isinstance(data, dict) or set(data) != required:
        raise CapacityProbeError("probe returned an unexpected schema")
    if data["qwen_service"] not in {"active", "inactive", "failed", "activating", "deactivating"}:
        raise CapacityProbeError("invalid Qwen service state")
    if not isinstance(data["gpus"], list) or len(data["gpus"]) != 2:
        raise CapacityProbeError("expected exactly two GPU records")
    models = data["ollama"].get("models") if isinstance(data["ollama"], dict) else None
    if not isinstance(models, list):
        raise CapacityProbeError("Ollama model inventory missing")
    return CapacitySnapshot(
        collected_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        qwen_service=data["qwen_service"],
        gpus=data["gpus"],
        ollama_models=models,
        ai_ssd=data["ai_ssd"],
    )


def collect(host: str, timeout: int = 15) -> CapacitySnapshot:
    result = subprocess.run(
        ssh_argv(host, timeout), capture_output=True, text=True, timeout=timeout + 10, check=False
    )
    if result.returncode:
        message = result.stderr.strip() or "no stderr"
        raise CapacityProbeError(f"read-only capacity SSH failed with exit {result.returncode}: {message}")
    return parse_snapshot(result.stdout)


def write_snapshot(snapshot: CapacitySnapshot, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot.json(), indent=2, sort_keys=True) + "\n")
