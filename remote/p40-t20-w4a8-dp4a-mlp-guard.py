#!/usr/bin/env python3
"""Forced-command guard for the bounded T20 full W4A8 DP4A MLP control."""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t17-w4a8-dp4a-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t17-w4a8-dp4a-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t20_base", SourceFileLoader("p40_t20_base", str(SOURCE)))
assert SPEC and SPEC.loader
t17 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t17
SPEC.loader.exec_module(t17)

t17.t05.EXPECTED_ORIGINAL_COMMAND = "p40-t20-w4a8-dp4a-mlp"
t17.t05.RESULTS_ORIGINAL_COMMAND = "p40-t20-w4a8-dp4a-mlp-results"
t17.t05.BENCHMARK = Path("/home/jordanculver/p40-native-research/qwen_w4a8_dp4a_mlp_control")
t17.t05.BENCHMARK_SHA256 = "cabeebe044008c931df0f1d08e12c8201bd571c91015cf8bd89ceea3cdc0b42e"
t17.t05.RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t20-w4a8-dp4a-mlp-controls")
t17.t05.PROFILE_ID = "t20-w4a8-dp4a-expert-mlp-4x"
t17.t05.SCHEMA_VERSION = "t20-w4a8-dp4a-expert-mlp-v1"
t17.t05.WATCHDOG_SECONDS = 180
t17.t05.FIXED_ARGUMENTS = (
    "--profile", "w4a8-dp4a-expert-mlp-4x-2048-512-2048", "--gpu", "0",
    "--repetitions", "3", "--calls-per-sample", "32", "--memory-cap-mib", "32", "--seed", "1",
)


def main() -> int:
    return t17.main()


if __name__ == "__main__":
    raise SystemExit(main())
