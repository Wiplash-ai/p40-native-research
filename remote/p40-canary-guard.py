#!/usr/bin/env python3
"""Forced SSH command for the first bounded P40 primitive canaries.

The command accepts a small JSON profile, never a shell command.  It is for
P01/P03 only; Qwen, Colibri, arbitrary executables, and long workloads remain
outside this gate.  It runs on excalibur so cleanup and power restoration do
not depend on the SSH client remaining connected.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


EXPECTED_ORIGINAL_COMMAND = "p40-canary"
RESULTS_ORIGINAL_COMMAND = "p40-canary-results"
BENCHMARK = Path("/home/jordanculver/p40-native-research/colibri-engine/benchmarks/p40_bench")
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/canaries")
ALLOWED_PRIMITIVES = {"P01-copy", "P03-dp4a-gemv"}
ALLOWED_DURATIONS = {1, 5, 10}
MIN_BYTES = 64 * 1024 * 1024
MAX_BYTES = 256 * 1024 * 1024
MAX_MEMORY_CAP_MIB = 1024
SAFE_POWER_W = 125
TEMP_LIMIT_C = 65.0
COOLDOWN_LIMIT_C = 40.0
MIN_COOLDOWN_SECONDS = 300
MAX_COOLDOWN_SECONDS = 900


class UnsafeRequest(ValueError):
    pass


def parse_request(text: str) -> dict:
    try:
        request = json.loads(text)
    except json.JSONDecodeError as error:
        raise UnsafeRequest("request must be one JSON object") from error
    if not isinstance(request, dict) or set(request) != {
        "primitive", "gpu", "duration_seconds", "bytes", "memory_cap_mib", "seed", "dry_run"
    }:
        raise UnsafeRequest("request fields are fixed")
    if request["primitive"] not in ALLOWED_PRIMITIVES:
        raise UnsafeRequest("primitive is not allowlisted")
    if request["gpu"] not in {0, 1}:
        raise UnsafeRequest("GPU must be 0 or 1")
    if request["duration_seconds"] not in ALLOWED_DURATIONS:
        raise UnsafeRequest("duration must be 1, 5, or 10 seconds")
    if not isinstance(request["bytes"], int) or not MIN_BYTES <= request["bytes"] <= MAX_BYTES:
        raise UnsafeRequest("bytes outside canary allocation range")
    if not isinstance(request["memory_cap_mib"], int) or not 1 <= request["memory_cap_mib"] <= MAX_MEMORY_CAP_MIB:
        raise UnsafeRequest("memory cap outside safe range")
    if not isinstance(request["seed"], int) or not isinstance(request["dry_run"], bool):
        raise UnsafeRequest("invalid seed or dry-run flag")
    return request


def benchmark_argv(request: dict) -> list[str]:
    return [
        str(BENCHMARK),
        "--primitive", request["primitive"],
        "--gpu", str(request["gpu"]),
        "--bytes", str(request["bytes"]),
        "--duration", str(request["duration_seconds"]),
        "--memory-cap-mib", str(request["memory_cap_mib"]),
        "--seed", str(request["seed"]),
    ]


def command(argv: list[str], timeout: int = 15) -> str:
    completed = subprocess.run(argv, check=True, text=True, capture_output=True, timeout=timeout)
    return completed.stdout


def gpu_state() -> list[dict]:
    output = command(
        [
            "/usr/bin/nvidia-smi",
            "--query-gpu=index,temperature.gpu,power.limit,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    result = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) != 5:
            continue
        result.append(
            {
                "index": int(row[0]),
                "temperature_c": float(row[1]),
                "power_limit_w": float(row[2]),
                "memory_used_mib": float(row[3]),
                "utilization_percent": float(row[4]),
            }
        )
    if len(result) < 2:
        raise RuntimeError("missing P40 GPU telemetry")
    return result


def fan_state() -> list[dict]:
    output = command(["/usr/bin/sudo", "-n", "/usr/local/sbin/p40-bmc-read", "sensor"])
    fans = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) >= 3 and fields[0].upper().startswith("FAN"):
            try:
                rpm = float(fields[1])
            except ValueError:
                rpm = None
            fans.append({"name": fields[0], "rpm": rpm})
    if len(fans) < 8:
        raise RuntimeError("missing fan telemetry")
    return fans


def critical_fan_events() -> set[str]:
    """Return SEL IDs only; historical events are a baseline, not a fresh abort."""
    output = command(["/usr/bin/sudo", "-n", "/usr/local/sbin/p40-bmc-read", "sel"])
    events = set()
    for line in output.splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) >= 6 and "Fan FAN" in line and "Lower Critical" in line and "Asserted" in line:
            events.add(fields[0])
    return events


def new_critical_events(baseline: set[str], current: set[str]) -> set[str]:
    return current - baseline


def unsafe_reason(gpus: list[dict], fans: list[dict], *, require_idle: bool) -> str | None:
    if any(gpu["temperature_c"] >= TEMP_LIMIT_C for gpu in gpus):
        return "temperature_limit"
    if any(fan["rpm"] is None or fan["rpm"] < 500 for fan in fans):
        return "fan_critical"
    if require_idle and any(gpu["temperature_c"] > COOLDOWN_LIMIT_C or gpu["memory_used_mib"] != 0 for gpu in gpus):
        return "not_cool_or_idle"
    return None


def set_power_limit(gpu: int, watts: float) -> None:
    command(["/usr/bin/sudo", "-n", "/usr/bin/nvidia-smi", "-i", str(gpu), "-pl", str(int(round(watts)))])


def terminate(process: subprocess.Popen[str]) -> list[str]:
    actions: list[str] = []
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        actions.append("term_process_group")
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
            actions.append("kill_process_group")
    return actions


def cool_down(samples: list[dict], baseline_events: set[str]) -> str | None:
    started = time.monotonic()
    while True:
        gpus, fans = gpu_state(), fan_state()
        reason = unsafe_reason(gpus, fans, require_idle=False)
        samples.append({"at": dt.datetime.now(dt.timezone.utc).isoformat(), "gpus": gpus, "fans": fans})
        if reason:
            return reason
        if new_critical_events(baseline_events, critical_fan_events()):
            return "new_fan_critical_event"
        elapsed = time.monotonic() - started
        if elapsed >= MIN_COOLDOWN_SECONDS and all(gpu["temperature_c"] <= COOLDOWN_LIMIT_C for gpu in gpus):
            return None
        if elapsed >= MAX_COOLDOWN_SECONDS:
            return "cooldown_timeout"
        time.sleep(5)


def run(request: dict) -> dict:
    result = {
        "schema_version": "p40-canary-v1",
        "run_id": str(uuid.uuid4()),
        "request": request,
        "command": benchmark_argv(request),
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "fail",
        "failure_reason": None,
        "telemetry": [],
        "cleanup": {"actions": [], "power_restored": False},
    }
    if request["dry_run"]:
        result.update({"status": "dry_run", "cuda_initialized": False})
        return result
    if not BENCHMARK.is_file() or not os.access(BENCHMARK, os.X_OK):
        result["failure_reason"] = "benchmark_binary_missing"
        return result
    original_limit: float | None = None
    process: subprocess.Popen[str] | None = None
    try:
        initial_gpus, initial_fans = gpu_state(), fan_state()
        baseline_events = critical_fan_events()
        result["telemetry"].append({"at": result["started_at"], "gpus": initial_gpus, "fans": initial_fans})
        reason = unsafe_reason(initial_gpus, initial_fans, require_idle=True)
        if reason:
            result["failure_reason"] = reason
            return result
        original_limit = next(gpu["power_limit_w"] for gpu in initial_gpus if gpu["index"] == request["gpu"])
        set_power_limit(request["gpu"], SAFE_POWER_W)
        process = subprocess.Popen(
            result["command"], start_new_session=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        deadline = time.monotonic() + request["duration_seconds"] + 8
        next_poll = 0.0
        next_bmc_poll = time.monotonic() + 5
        while process.poll() is None:
            if time.monotonic() >= deadline:
                result["failure_reason"] = "watchdog_timeout"
                break
            if time.monotonic() >= next_poll:
                gpus, fans = gpu_state(), fan_state()
                result["telemetry"].append({"at": dt.datetime.now(dt.timezone.utc).isoformat(), "gpus": gpus, "fans": fans})
                reason = unsafe_reason(gpus, fans, require_idle=False)
                if reason:
                    result["failure_reason"] = reason
                    break
                next_poll = time.monotonic() + 1
            if time.monotonic() >= next_bmc_poll:
                if new_critical_events(baseline_events, critical_fan_events()):
                    result["failure_reason"] = "new_fan_critical_event"
                    break
                next_bmc_poll = time.monotonic() + 5
            time.sleep(0.05)
        if result["failure_reason"]:
            result["cleanup"]["actions"].extend(terminate(process))
        else:
            stdout, stderr = process.communicate(timeout=2)
            result["stdout"] = stdout
            result["stderr"] = stderr
            if process.returncode:
                result["failure_reason"] = "benchmark_failed"
        if process.poll() is None:
            result["cleanup"]["actions"].extend(terminate(process))
        result["cooldown_failure"] = cool_down(result["telemetry"], baseline_events)
        if result["cooldown_failure"] and not result["failure_reason"]:
            result["failure_reason"] = result["cooldown_failure"]
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        result["failure_reason"] = f"guard_error:{type(error).__name__}"
        if process:
            result["cleanup"]["actions"].extend(terminate(process))
    finally:
        if original_limit is not None:
            try:
                set_power_limit(request["gpu"], original_limit)
                result["cleanup"]["power_restored"] = True
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                result["cleanup"]["power_restore_error"] = type(error).__name__
        result["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    if result["failure_reason"] is None and result["cleanup"]["power_restored"]:
        result["status"] = "pass"
    return result


def persist(result: dict) -> Path:
    """Persist before writing SSH stdout, so client loss cannot lose evidence."""
    RESULT_DIRECTORY.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = RESULT_DIRECTORY / f"{result['run_id']}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
    return target


def recent_results(limit: int = 10) -> list[dict]:
    if not RESULT_DIRECTORY.is_dir():
        return []
    records = []
    for path in sorted(RESULT_DIRECTORY.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            records.append({"schema_version": "p40-canary-v1", "status": "corrupt_result", "file": path.name})
    return records


def main() -> int:
    original_command = os.environ.get("SSH_ORIGINAL_COMMAND")
    if original_command == RESULTS_ORIGINAL_COMMAND:
        print(json.dumps(recent_results(), sort_keys=True))
        return 0
    if original_command != EXPECTED_ORIGINAL_COMMAND:
        print("p40 canary key only accepts: p40-canary or p40-canary-results", file=sys.stderr)
        return 126
    try:
        request = parse_request(sys.stdin.read())
    except UnsafeRequest as error:
        print(json.dumps({"status": "rejected", "error": str(error)}, sort_keys=True))
        return 2
    result = run(request)
    path = persist(result)
    result["result_path"] = str(path)
    try:
        print(json.dumps(result, sort_keys=True))
    except BrokenPipeError:
        pass
    return 0 if result["status"] in {"pass", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
