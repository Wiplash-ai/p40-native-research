#!/usr/bin/env python3
"""Host-locking guarded process runner; telemetry is injected by a provider."""
from __future__ import annotations
import argparse, fcntl, json, os, signal, subprocess, time, uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable
from snapshot import collect

FAILURES = {"stale_telemetry", "fan_critical", "device_error", "temperature_limit", "temperature_slope", "watchdog_timeout", "subprocess_failure"}
_LOCAL_LOCKS: set[str] = set()

@dataclass
class Telemetry:
    timestamp: float
    temperatures: dict[str, float]
    fan_critical: bool = False
    device_error: bool = False

def parse_telemetry(line: str) -> Telemetry:
    """Parse an isolated collector's JSONL record; reject incomplete records."""
    raw = json.loads(line)
    if not isinstance(raw.get("temperatures"), dict) or "timestamp" not in raw:
        raise ValueError("telemetry requires timestamp and temperatures")
    return Telemetry(float(raw["timestamp"]), {str(k): float(v) for k, v in raw["temperatures"].items()},
                     bool(raw.get("fan_critical", False)), bool(raw.get("device_error", False)))

def parse_telemetry_jsonl(path: str) -> list[Telemetry]:
    """Parse a collector file completely, failing closed on a bad record."""
    with open(path, encoding="utf-8") as source:
        return [parse_telemetry(line) for line in source if line.strip()]

class HostLock:
    def __init__(self, path: str): self.path, self.file = path, None
    def __enter__(self):
        self.path = os.path.abspath(self.path)
        if self.path in _LOCAL_LOCKS:
            raise RuntimeError("benchmark_lock_held")
        self.file = open(self.path, "a+")
        try: fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close(); raise RuntimeError("benchmark_lock_held")
        _LOCAL_LOCKS.add(self.path)
        self.file.write(str(os.getpid()) + "\n"); self.file.flush(); return self
    def __exit__(self, *_):
        if self.file:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN); self.file.close(); _LOCAL_LOCKS.discard(self.path)

class Watchdog:
    def __init__(self, now: Callable[[], float] = time.monotonic, stale_seconds=3, max_temp=65, slope_temp=55, slope_limit=.5):
        self.now, self.stale_seconds, self.max_temp, self.slope_temp, self.slope_limit = now, stale_seconds, max_temp, slope_temp, slope_limit
        self.history: list[Telemetry] = []
    def check(self, sample: Telemetry | None) -> str | None:
        if sample is None or self.now() - sample.timestamp > self.stale_seconds: return "stale_telemetry"
        if sample.fan_critical: return "fan_critical"
        if sample.device_error: return "device_error"
        if any(t >= self.max_temp for t in sample.temperatures.values()): return "temperature_limit"
        self.history.append(sample); self.history = self.history[-11:]
        if len(self.history) >= 2:
            first, last = self.history[0], self.history[-1]; elapsed = last.timestamp - first.timestamp
            if elapsed >= 10 and any(last.temperatures.get(k, -999) >= self.slope_temp and (last.temperatures.get(k, 0)-first.temperatures.get(k, 0))/elapsed > self.slope_limit for k in last.temperatures): return "temperature_slope"
        return None

def cleanup_process_group(process: subprocess.Popen, grace_seconds: float = 1) -> dict:
    actions = []
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM); actions.append("term_process_group")
        try: process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait(); actions.append("kill_process_group")
    for pipe in (process.stdout, process.stderr):
        if pipe: pipe.close()
    return {"status": "complete", "actions": actions, "returncode": process.returncode}

def run(command: list[str], telemetry: Callable[[], Telemetry | None], *, timeout=30, poll_seconds=.05, lock_path="/tmp/p40-native-benchmark.lock", restore_power: Callable[[], None] | None = None, now=time.monotonic) -> dict:
    run_id, started = str(uuid.uuid4()), now(); watchdog, samples = Watchdog(now=now), []
    result = {"schema_version":"p40-result-v1", "task":"T01", "run_id":run_id, "command":command, "snapshot":collect(), "telemetry":samples, "failure_reason":None}
    cleanup = {"status":"not_started", "actions":[]}
    try:
        with HostLock(lock_path):
            process = subprocess.Popen(command, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            while process.poll() is None:
                sample = telemetry()
                if sample: samples.append(asdict(sample))
                reason = watchdog.check(sample)
                if reason is None and now() - started > timeout: reason = "watchdog_timeout"
                if reason:
                    result["failure_reason"] = reason; cleanup = cleanup_process_group(process); break
                time.sleep(poll_seconds)
            if process.poll() is not None and result["failure_reason"] is None and process.returncode != 0: result["failure_reason"] = "subprocess_failure"
            if cleanup["status"] == "not_started": cleanup = cleanup_process_group(process)
    except RuntimeError as error: result["failure_reason"] = str(error); cleanup = {"status":"not_started", "actions":[]}
    finally:
        if restore_power:
            try: restore_power(); cleanup["actions"].append("power_restored")
            except Exception as error: cleanup["status"] = "restore_failed"; cleanup["restore_error"] = str(error)
    result["cleanup"] = cleanup
    result["status"] = "pass" if result["failure_reason"] is None and cleanup["status"] == "complete" else "fail"
    return result

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--timeout", type=float, default=30); parser.add_argument("--lock", default="/tmp/p40-native-benchmark.lock"); parser.add_argument("command", nargs=argparse.REMAINDER); args=parser.parse_args()
    if args.command[:1] == ["--"]: args.command = args.command[1:]
    if not args.command: parser.error("command required after --")
    if not args.dry_run:
        parser.error("T01 supports --dry-run only; a later reviewed task must explicitly enable execution")
    print(json.dumps({"command":args.command,"cuda_initialized":False,"lock":args.lock}, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
