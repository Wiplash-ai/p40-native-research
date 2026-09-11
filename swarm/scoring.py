"""Evidence-only branch ranking for the bounded search controller.

This module deliberately ranks measured outcomes, not model confidence or
natural-language rationales.  It recommends a next allocation; it never
dispatches a model, edits a worktree, or promotes a branch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


REQUIRED_KINDS = frozenset({"static", "test"})


@dataclass(frozen=True)
class BranchScore:
    branch_id: str
    decision: str
    score: int
    passed_kinds: list[str]
    failed_kinds: list[str]
    reasons: list[str]

    def json(self) -> dict[str, Any]:
        return asdict(self)


def score_branch(*, branch_id: str, evidence: list[dict[str, Any]]) -> BranchScore:
    """Return a deterministic promotion/expand/prune recommendation.

    Passing static and test evidence are prerequisites. A model response or a
    benchmark can add no score until those gates pass, preventing confident
    prose or a microbenchmark from outranking working software.
    """
    passed = sorted({item["kind"] for item in evidence if item["exit_code"] == 0})
    failed = sorted({item["kind"] for item in evidence if item["exit_code"] != 0})
    failed_required = sorted(REQUIRED_KINDS.intersection(failed))
    passed_required = REQUIRED_KINDS.intersection(passed)
    reasons: list[str] = []
    if failed_required:
        reasons.append("required validation failed: " + ", ".join(failed_required))
        return BranchScore(branch_id, "prune", -100, passed, failed, reasons)
    missing = sorted(REQUIRED_KINDS - passed_required)
    if missing:
        reasons.append("missing required validation: " + ", ".join(missing))
        return BranchScore(branch_id, "expand", 25 * len(passed_required), passed, failed, reasons)
    score = 100
    reasons.append("static and test validation passed")
    benchmark_gain = 0
    for item in evidence:
        if item["kind"] != "benchmark" or item["exit_code"] != 0:
            continue
        value = item.get("metrics", {}).get("improvement_pct")
        if isinstance(value, (int, float)) and value > 0:
            benchmark_gain = max(benchmark_gain, min(20, round(value)))
    if benchmark_gain:
        score += benchmark_gain
        reasons.append(f"validated benchmark gain: +{benchmark_gain}")
    return BranchScore(branch_id, "promote-ready", score, passed, failed, reasons)


def rank_task(store: Any, task_id: str) -> list[dict[str, Any]]:
    """Rank task branches by hard evidence, with stable UUID tie-breaking."""
    task = store.task(task_id)
    scored = [score_branch(branch_id=branch["id"], evidence=store.branch_evidence(branch["id"])) for branch in task["branches"]]
    return [item.json() for item in sorted(scored, key=lambda item: (-item.score, item.branch_id))]
