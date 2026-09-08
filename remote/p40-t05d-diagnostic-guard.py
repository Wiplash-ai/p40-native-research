#!/usr/bin/env python3
"""Distinct forced-command identity for the T05D error-attribution sweep."""
from __future__ import annotations

from importlib.machinery import SourceFileLoader
import importlib.util
import sys
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t05a-q8-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t05a-q8-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t05d_base", SourceFileLoader("p40_t05d_base", str(SOURCE)))
assert SPEC and SPEC.loader
t05a = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t05a
SPEC.loader.exec_module(t05a)

t05a.EXPECTED_ORIGINAL_COMMAND = "p40-t05d-diagnostic"
t05a.RESULTS_ORIGINAL_COMMAND = "p40-t05d-diagnostic-results"
t05a.BENCHMARK = Path("/home/jordanculver/p40-native-research/colibri-t05a/p40_benchmarks/qwen_dn_sweep_control")
t05a.BENCHMARK_SHA256 = "3d8043ee2846e2d5942a823bdbeaa43fffea331e9175da92d5d8c790233115a7"
t05a.RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t05d-diagnostic-controls")
t05a.PROFILE_ID = "t05d-dn-error-attribution-30x-triplet"
t05a.SCHEMA_VERSION = "p40-t05d-diagnostic-v1"
t05a.WATCHDOG_SECONDS = 240
t05a.FIXED_ARGUMENTS = (
    "--profile", "dn-sweep-30x-triplet-2048-8192-4096", "--gpu", "0",
    "--repetitions", "3", "--sweeps-per-sample", "1", "--memory-cap-mib", "1088",
    "--seed", "1",
)


def main() -> int:
    return t05a.main()


if __name__ == "__main__":
    raise SystemExit(main())
