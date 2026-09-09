#!/usr/bin/env python3
"""Forced-command guard for the bounded T19 16-warp DP4A control."""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t17-w4a8-dp4a-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t17-w4a8-dp4a-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t19_base", SourceFileLoader("p40_t19_base", str(SOURCE)))
assert SPEC and SPEC.loader
t17 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t17
SPEC.loader.exec_module(t17)

t17.t05.EXPECTED_ORIGINAL_COMMAND = "p40-t19-w4a8-dp4a-tile16"
t17.t05.RESULTS_ORIGINAL_COMMAND = "p40-t19-w4a8-dp4a-tile16-results"
t17.t05.BENCHMARK = Path("/home/jordanculver/p40-native-research/qwen_w4a8_dp4a_tile16_control")
t17.t05.BENCHMARK_SHA256 = "4222a603420ed94a856d249cdd0e6b2decafb9240c9f94ff7808b5e3cd5de43e"
t17.t05.RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t19-w4a8-dp4a-tile16-controls")
t17.t05.PROFILE_ID = "t19-w4a8-dp4a-expert-projection-tile16"
t17.t05.SCHEMA_VERSION = "t19-w4a8-dp4a-expert-proj-v1"
t17.t05.WATCHDOG_SECONDS = 180
t17.t05.FIXED_ARGUMENTS = (
    "--profile", "w4a8-dp4a-expert-proj-4x-2048-512", "--gpu", "0",
    "--repetitions", "3", "--calls-per-sample", "64", "--memory-cap-mib", "32", "--seed", "1",
)


def main() -> int:
    return t17.main()


if __name__ == "__main__":
    raise SystemExit(main())
