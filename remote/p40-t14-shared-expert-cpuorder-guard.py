#!/usr/bin/env python3
"""Forced-command guard for the bounded T14 exact-Q8 shared-expert control."""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t05a-q8-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t05a-q8-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t14_base", SourceFileLoader("p40_t14_base", str(SOURCE)))
assert SPEC and SPEC.loader
t05 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t05
SPEC.loader.exec_module(t05)

t05.EXPECTED_ORIGINAL_COMMAND = "p40-t14-shared-expert-cpuorder"
t05.RESULTS_ORIGINAL_COMMAND = "p40-t14-shared-expert-cpuorder-results"
t05.BENCHMARK = Path("/home/jordanculver/p40-native-research/t14-shared-expert-bench/qwen_shared_expert_cpuorder_control")
t05.BENCHMARK_SHA256 = "3e6b1ef1c65ec5cb2e937e91089763047bf9755655a2870209683773ddb86228"
t05.RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t14-shared-expert-cpuorder-controls")
t05.PROFILE_ID = "t14-shared-expert-cpuorder-40x"
t05.SCHEMA_VERSION = "p40-t14-shared-expert-cpuorder-q8-v1"
t05.WATCHDOG_SECONDS = 180
t05.FIXED_ARGUMENTS = (
    "--profile", "shared-expert-cpuorder-40x-2048-512-2048", "--gpu", "0",
    "--repetitions", "3", "--calls-per-sample", "1", "--memory-cap-mib", "160", "--seed", "1",
)


def main() -> int:
    return t05.main()


if __name__ == "__main__":
    raise SystemExit(main())
