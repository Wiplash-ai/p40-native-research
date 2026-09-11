"""One bounded, telemetry-recorded Ollama executor benchmark.

Run this on the model host. It never pulls a model, edits service settings, or
starts Qwen. The caller must explicitly select an already-installed model.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROMPT = "Reply with exactly sixty-four lowercase English words, separated by single spaces. No punctuation."


class BenchmarkError(RuntimeError):
    """A bounded executor run lacks a safety or GPU-use prerequisite."""


def nvidia_snapshot() -> list[dict[str, float | int]]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,temperature.gpu,power.draw,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True,
    )
    records = []
    for line in output.splitlines():
        index, temperature, power, memory, utilization = [part.strip() for part in line.split(",")]
        records.append({
            "index": int(index), "temperature_c": int(temperature), "power_w": float(power),
            "memory_mib": int(memory), "utilization_pct": int(utilization),
        })
    if len(records) != 2:
        raise BenchmarkError("expected exactly two P40 telemetry records")
    return records


def require_safe_start(limit_c: int) -> list[dict[str, float | int]]:
    samples = nvidia_snapshot()
    if max(sample["temperature_c"] for sample in samples) >= limit_c:
        raise BenchmarkError(f"GPU temperature is already at or above {limit_c} C")
    return samples


def post_json(url: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 -- fixed caller endpoint
        return json.loads(response.read())


@dataclass(frozen=True)
class BenchmarkResult:
    schema_version: str
    status: str
    model: str
    url: str
    context: int
    completion_limit: int
    started_at: str
    elapsed_s: float
    eval_count: int
    eval_duration_s: float
    eval_tokens_per_s: float
    prompt_eval_count: int
    prompt_eval_duration_s: float
    gpu_start: list[dict[str, float | int]]
    gpu_peak: list[dict[str, float | int]]
    gpu_end: list[dict[str, float | int]]
    gpu_observed: bool
    thermal_limit_c: int
    response_text: str
    response_sha256: str


def benchmark(*, url: str, model: str, context: int, completion_limit: int, timeout: int, limit_c: int) -> BenchmarkResult:
    import hashlib

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gpu_start = require_safe_start(limit_c)
    samples = list(gpu_start)
    stop = threading.Event()

    def monitor() -> None:
        while not stop.is_set():
            try:
                samples.extend(nvidia_snapshot())
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            stop.wait(0.25)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        response = post_json(url, {
            "model": model,
            "prompt": PROMPT,
            "stream": False,
            "keep_alive": "0s",
            "think": False,
            "options": {"num_ctx": context, "num_predict": completion_limit, "seed": 42, "temperature": 0},
        }, timeout)
    finally:
        stop.set()
        thread.join(timeout=2)
    elapsed_s = time.monotonic() - started
    gpu_end = nvidia_snapshot()
    samples.extend(gpu_end)
    peak_by_gpu = []
    for index in (0, 1):
        records = [sample for sample in samples if sample["index"] == index]
        peak_by_gpu.append({
            "index": index,
            "temperature_c": max(int(sample["temperature_c"]) for sample in records),
            "power_w": max(float(sample["power_w"]) for sample in records),
            "memory_mib": max(int(sample["memory_mib"]) for sample in records),
            "utilization_pct": max(int(sample["utilization_pct"]) for sample in records),
        })
    if max(sample["temperature_c"] for sample in peak_by_gpu) >= limit_c:
        raise BenchmarkError(f"thermal ceiling reached: {peak_by_gpu}")
    eval_count = int(response.get("eval_count", 0))
    eval_duration_s = int(response.get("eval_duration", 0)) / 1_000_000_000
    if eval_count < 1 or eval_duration_s <= 0:
        raise BenchmarkError(f"Ollama returned no decode timing: {response}")
    gpu_observed = any(int(sample["memory_mib"]) > 0 for sample in peak_by_gpu)
    if not gpu_observed:
        raise BenchmarkError("Ollama completed without observable GPU memory residency")
    prompt_duration = int(response.get("prompt_eval_duration", 0)) / 1_000_000_000
    response_text = str(response.get("response", ""))
    if not response_text:
        raise BenchmarkError("Ollama returned no completion text")
    return BenchmarkResult(
        schema_version="swarm-ollama-benchmark-v1", status="pass", model=model, url=url,
        context=context, completion_limit=completion_limit, started_at=started_at,
        elapsed_s=round(elapsed_s, 4), eval_count=eval_count, eval_duration_s=eval_duration_s,
        eval_tokens_per_s=eval_count / eval_duration_s,
        prompt_eval_count=int(response.get("prompt_eval_count", 0)),
        prompt_eval_duration_s=prompt_duration, gpu_start=gpu_start, gpu_peak=peak_by_gpu,
        gpu_end=gpu_end, gpu_observed=gpu_observed,
        thermal_limit_c=limit_c,
        response_text=response_text,
        response_sha256=hashlib.sha256(response_text.encode()).hexdigest(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://172.17.0.1:11434/api/generate")
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--completion-limit", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--thermal-limit-c", type=int, default=70)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = benchmark(
            url=args.url, model=args.model, context=args.context, completion_limit=args.completion_limit,
            timeout=args.timeout, limit_c=args.thermal_limit_c,
        )
    except (BenchmarkError, OSError, subprocess.SubprocessError, ValueError, urllib.error.URLError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, sort_keys=True))
        return 1
    output = json.dumps(asdict(result), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
