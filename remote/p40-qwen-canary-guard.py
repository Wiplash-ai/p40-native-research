#!/usr/bin/env python3
"""Forced-command guard for one fixed, short experimental Qwen canary.

No request controls model, prompt, GPUs, environment, output count, power, or
timeout. This intentionally remains separate from the P01/P03 executor.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-canary-guard.py")
SPEC = importlib.util.spec_from_file_location("p40_primitive_guard", SOURCE)
assert SPEC and SPEC.loader
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

EXPECTED_ORIGINAL_COMMAND = "p40-qwen-canary"
RESULTS_ORIGINAL_COMMAND = "p40-qwen-canary-results"
ENGINE = Path("/home/jordanculver/p40-native-research/colibri-engine/c/qwen36")
MODEL = Path("/mnt/ai-ssd/huggingface/hub/models--Kreuzzelg--qwen36-35b-a3b-colibri-i4/snapshots/9ccfbfa09bc55410a9c9102558ee63077d8a477d")
PROMPT = Path("/home/jordanculver/p40-native-research/fixtures/prompt.txt")
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/qwen-canaries")
GPUS = (0, 1)
OUTPUT_TOKENS = 16
SAFE_POWER_W = 125
WATCHDOG_SECONDS = 1800
OUTPUT_TAIL_BYTES = 64 * 1024


class UnsafeRequest(ValueError):
    pass


def parse_request(text: str) -> dict:
    try:
        request = json.loads(text)
    except json.JSONDecodeError as error:
        raise UnsafeRequest("request must be one JSON object") from error
    if not isinstance(request, dict) or set(request) != {"dry_run"} or not isinstance(request["dry_run"], bool):
        raise UnsafeRequest("only dry_run is accepted")
    return request


def model_argv() -> list[str]:
    environment = {
        "COLI_CUDA": "1", "COLI_GPUS": "0,1", "CUDA_EXPERT_GB": "auto",
        "COLI_DENSE_I8": "1", "COLI_TIMERS": "1", "COLI_CUDA_PROFILE": "1",
        "OMP_NUM_THREADS": "24", "OMP_DYNAMIC": "FALSE", "OMP_WAIT_POLICY": "PASSIVE",
        "OMP_PROC_BIND": "close", "OMP_PLACES": "cores", "Q36_MAXT": "8192",
        "PILOT": "0", "HOT": "0", "COLIBRI_RESIDENT": "0", "SNAP": str(MODEL),
        "N_NEW": str(OUTPUT_TOKENS), "NOSTREAM": "1",
    }
    return ["/usr/bin/env", "-i", "PATH=/usr/bin:/bin", *[f"{k}={v}" for k, v in environment.items()], str(ENGINE), "256", "4", str(PROMPT)]


def persist(result: dict) -> Path:
    RESULT_DIRECTORY.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = RESULT_DIRECTORY / f"{result['run_id']}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
    return target


def recent_results() -> list[dict]:
    if not RESULT_DIRECTORY.is_dir():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(RESULT_DIRECTORY.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]]


def output_summary(path: Path) -> dict:
    """Return bounded, durable evidence without risking a PIPE deadlock."""
    payload = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "tail": payload[-OUTPUT_TAIL_BYTES:].decode("utf-8", errors="replace"),
    }


def run(request: dict) -> dict:
    result = {"schema_version": "p40-qwen-canary-v1", "run_id": str(uuid.uuid4()), "request": request,
              "command": model_argv(), "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "fail",
              "failure_reason": None, "telemetry": [], "cleanup": {"actions": [], "power_restored": False}}
    if request["dry_run"]:
        result.update({"status": "dry_run", "cuda_initialized": False})
        return result
    if not ENGINE.is_file() or not os.access(ENGINE, os.X_OK) or not MODEL.is_dir() or not PROMPT.is_file():
        result["failure_reason"] = "pinned_artifact_missing"; return result
    process = None; original = {}; stdout_handle = None; stderr_handle = None
    try:
        gpus, fans = base.gpu_state(), base.fan_state(); baseline = base.critical_fan_events()
        result["telemetry"].append({"at": result["started_at"], "gpus": gpus, "fans": fans})
        if reason := base.unsafe_reason(gpus, fans, require_idle=True): result["failure_reason"] = reason; return result
        original = {gpu["index"]: gpu["power_limit_w"] for gpu in gpus if gpu["index"] in GPUS}
        if set(original) != set(GPUS): result["failure_reason"] = "missing_target_gpu"; return result
        for gpu in GPUS: base.set_power_limit(gpu, SAFE_POWER_W)
        RESULT_DIRECTORY.mkdir(mode=0o700, parents=True, exist_ok=True)
        stdout_path = RESULT_DIRECTORY / f"{result['run_id']}.stdout.txt"
        stderr_path = RESULT_DIRECTORY / f"{result['run_id']}.stderr.txt"
        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_handle = stderr_path.open("w", encoding="utf-8")
        process = subprocess.Popen(result["command"], start_new_session=True, text=True, stdout=stdout_handle, stderr=stderr_handle)
        deadline, next_bmc = time.monotonic() + WATCHDOG_SECONDS, time.monotonic()
        while process.poll() is None:
            gpus, fans = base.gpu_state(), base.fan_state()
            result["telemetry"].append({"at": dt.datetime.now(dt.timezone.utc).isoformat(), "gpus": gpus, "fans": fans})
            if time.monotonic() >= deadline: result["failure_reason"] = "watchdog_timeout"; break
            if reason := base.unsafe_reason(gpus, fans, require_idle=False): result["failure_reason"] = reason; break
            if time.monotonic() >= next_bmc:
                if base.new_critical_events(baseline, base.critical_fan_events()): result["failure_reason"] = "new_fan_critical_event"; break
                next_bmc = time.monotonic() + 5
            time.sleep(1)
        if result["failure_reason"]: result["cleanup"]["actions"].extend(base.terminate(process))
        else:
            process.wait(timeout=10)
            if process.returncode: result["failure_reason"] = "engine_failed"
        if stdout_handle:
            stdout_handle.close(); stdout_handle = None
            result["stdout"] = output_summary(stdout_path)
        if stderr_handle:
            stderr_handle.close(); stderr_handle = None
            result["stderr"] = output_summary(stderr_path)
        result["cooldown_failure"] = base.cool_down(result["telemetry"], baseline)
        if result["cooldown_failure"] and not result["failure_reason"]: result["failure_reason"] = result["cooldown_failure"]
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        result["failure_reason"] = f"guard_error:{type(error).__name__}"
        if process: result["cleanup"]["actions"].extend(base.terminate(process))
    finally:
        if stdout_handle: stdout_handle.close()
        if stderr_handle: stderr_handle.close()
        for gpu, watts in original.items():
            try: base.set_power_limit(gpu, watts)
            except (OSError, RuntimeError, subprocess.SubprocessError): result["cleanup"]["power_restore_error"] = True
        result["cleanup"]["power_restored"] = bool(original) and "power_restore_error" not in result["cleanup"]
        result["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    if result["failure_reason"] is None and result["cleanup"]["power_restored"]: result["status"] = "pass"
    return result


def main() -> int:
    original = os.environ.get("SSH_ORIGINAL_COMMAND")
    if original == RESULTS_ORIGINAL_COMMAND: print(json.dumps(recent_results(), sort_keys=True)); return 0
    if original != EXPECTED_ORIGINAL_COMMAND: print("p40 qwen key only accepts: p40-qwen-canary or p40-qwen-canary-results", file=sys.stderr); return 126
    try: result = run(parse_request(sys.stdin.read()))
    except UnsafeRequest as error: print(json.dumps({"status": "rejected", "error": str(error)})); return 2
    result["result_path"] = str(persist(result)); print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {"pass", "dry_run"} else 1


if __name__ == "__main__": raise SystemExit(main())
