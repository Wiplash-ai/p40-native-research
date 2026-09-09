#!/usr/bin/env python3
"""Forced-command identity for the bounded Qwen LM-head exact-Q8 control."""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t05a-q8-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t05a-q8-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t08_base", SourceFileLoader("p40_t08_base", str(SOURCE)))
assert SPEC and SPEC.loader
t05 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t05
SPEC.loader.exec_module(t05)

t05.EXPECTED_ORIGINAL_COMMAND = "p40-t08-lmhead-cpuorder"
t05.RESULTS_ORIGINAL_COMMAND = "p40-t08-lmhead-cpuorder-results"
t05.BENCHMARK = Path("/home/jordanculver/p40-native-research/t08-lmhead-bench/qwen_lmhead_cpuorder_control")
t05.BENCHMARK_SHA256 = "25d36744dca6761d93082500f8c49b70779835c5ca1001fbd45391fc2d70d777"
t05.RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t08-lmhead-cpuorder-controls")
t05.PROFILE_ID = "t08-lmhead-cpuorder-2048x248044"
t05.SCHEMA_VERSION = "p40-t08-lmhead-cpuorder-v1"
t05.WATCHDOG_SECONDS = 180
t05.FIXED_ARGUMENTS = (
    "--profile", "lmhead-cpuorder-2048x248044", "--gpu", "0",
    "--repetitions", "3", "--calls-per-sample", "1", "--memory-cap-mib", "512", "--seed", "1",
)


def main() -> int:
    return t05.main()


if __name__ == "__main__":
    raise SystemExit(main())
