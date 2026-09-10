#!/usr/bin/env python3
"""Forced-command identity for an exact-Q8 DeltaNet-pair timing audit."""
from __future__ import annotations

import importlib.util
import math
import re
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t24-qwen-deltanet-pair-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t24-qwen-deltanet-pair-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t36_base", SourceFileLoader("p40_t36_base", str(SOURCE)))
assert SPEC and SPEC.loader
t24 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t24
SPEC.loader.exec_module(t24)

OUTPUT_TOKENS = 64
EXPECTED_ORIGINAL_COMMAND = "p40-t36-qwen-deltanet-pair-profile"
RESULTS_ORIGINAL_COMMAND = "p40-t36-qwen-deltanet-pair-profile-results"
ENGINE = Path("/home/jordanculver/p40-native-research/colibri-t36-dn-pair-profile/c/qwen36")
ENGINE_SHA256 = "009dd7daea34d65aba67aa442b79e7ff0d914642008ebd3b37b26a9f36a2ca64"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t36-qwen-deltanet-pair-profile")
PROFILE_ID = "t36-exact-qwen-deltanet-pair-profile-64"
SCHEMA_VERSION = "p40-t36-qwen-deltanet-pair-profile-v1"
EXPECTED_STDOUT_SHA256 = "5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f"
EXPECTED_PAIR_CALLS = 30 * (OUTPUT_TOKENS - 1)
HOST_PATTERN = re.compile(
    rb"\[timers\]   dn-pair-host: issue ([0-9.]+) \| b\+a ([0-9.]+) \| join ([0-9.]+) ms/pair \(calls ([0-9]+)\)"
)
GPU_PATTERN = re.compile(
    rb"\[dn-pair-gpu\] samples=([0-9]+) h2d_total_ms=([0-9.]+) qkv_total_ms=([0-9.]+) z_total_ms=([0-9.]+) d2h_total_ms=([0-9.]+) total_ms=([0-9.]+)"
)


_base_model_argv = t24.model_argv
_base_run = t24.run


def model_argv() -> list[str]:
    result: list[str] = []
    for argument in _base_model_argv():
        result.append(argument)
        if argument == "COLI_CUDA_DN_PAIR=1":
            result.append("COLI_DN_PAIR_PROFILE=1")
    return [f"N_NEW={OUTPUT_TOKENS}" if argument.startswith("N_NEW=") else argument
            for argument in result]


def parse_profile(stderr: bytes) -> dict | None:
    host = HOST_PATTERN.search(stderr)
    gpu = GPU_PATTERN.search(stderr)
    if not host or not gpu:
        return None
    host_values = [float(value) for value in host.groups()[:3]]
    gpu_values = [float(value) for value in gpu.groups()[1:]]
    host_calls, gpu_samples = int(host.group(4)), int(gpu.group(1))
    if host_calls != EXPECTED_PAIR_CALLS or gpu_samples != EXPECTED_PAIR_CALLS:
        return None
    if not all(math.isfinite(value) and value >= 0.0 for value in host_values + gpu_values):
        return None
    if abs(gpu_values[-1] - sum(gpu_values[:-1])) > 0.02:
        return None
    return {
        "pair_calls": host_calls,
        "host_ms_per_pair": {"issue": host_values[0], "cpu_ba": host_values[1], "join": host_values[2]},
        "gpu_total_ms": {"h2d": gpu_values[0], "qkv": gpu_values[1], "z": gpu_values[2],
                         "d2h": gpu_values[3], "total": gpu_values[4]},
    }


def run(request: dict) -> dict:
    result = _base_run(request)
    if not request["dry_run"] and result.get("status") == "pass":
        profile = parse_profile(Path(result.get("stderr", {}).get("path", "")).read_bytes())
        if profile is None:
            result["failure_reason"] = "deltanet_pair_profile_missing_or_invalid"
            result["status"] = "fail"
        else:
            result["deltanet_pair_profile"] = profile
    return result


def configure(module) -> None:
    module.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
    module.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
    module.ENGINE = ENGINE
    module.ENGINE_SHA256 = ENGINE_SHA256
    module.RESULT_DIRECTORY = RESULT_DIRECTORY
    module.PROFILE_ID = PROFILE_ID
    module.SCHEMA_VERSION = SCHEMA_VERSION
    module.EXPECTED_STDOUT_SHA256 = EXPECTED_STDOUT_SHA256
    module.model_argv = model_argv


configure(t24)
configure(t24.t23)
configure(t24.t23.t15)
configure(t24.t23.t15.t12)
configure(t24.t23.t15.t12.t09)
configure(t24.t23.t15.t12.t09.t06)
configure(t24.t23.t15.t12.t09.t06.qwen)
t24.t23.t15.t12.t09.t06.qwen.OUTPUT_TOKENS = OUTPUT_TOKENS
t24.t23.t15.t12.t09.t06.qwen.run = run


def main() -> int:
    return t24.t23.t15.main()


if __name__ == "__main__":
    raise SystemExit(main())
