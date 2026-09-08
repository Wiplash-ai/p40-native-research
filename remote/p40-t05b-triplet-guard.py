#!/usr/bin/env python3
"""Distinct forced-command identity for the bounded T05B DeltaNet triplet."""
from __future__ import annotations

from importlib.machinery import SourceFileLoader
import importlib.util
import sys
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t05a-q8-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t05a-q8-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t05b_base", SourceFileLoader("p40_t05b_base", str(SOURCE)))
assert SPEC and SPEC.loader
t05a = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t05a
SPEC.loader.exec_module(t05a)

t05a.EXPECTED_ORIGINAL_COMMAND = "p40-t05b-triplet"
t05a.RESULTS_ORIGINAL_COMMAND = "p40-t05b-triplet-results"
t05a.BENCHMARK = Path("/home/jordanculver/p40-native-research/colibri-t05a/p40_benchmarks/qwen_dn_triplet_control")
t05a.BENCHMARK_SHA256 = "b60c55bb40d4c57fe8fd0796627089725eb76092236de676bd6ffdde9f7d5d2e"
t05a.RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t05b-triplet-controls")
t05a.PROFILE_ID = "t05b-dn-triplet-2048-8192-4096"
t05a.SCHEMA_VERSION = "p40-t05b-triplet-v1"
t05a.FIXED_ARGUMENTS = (
    "--profile", "dn-triplet-2048-8192-4096", "--gpu", "0", "--repetitions", "5",
    "--calls-per-sample", "4", "--memory-cap-mib", "64", "--seed", "1",
)


def main() -> int:
    return t05a.main()


if __name__ == "__main__":
    raise SystemExit(main())
