#!/usr/bin/env python3
"""Task-gated Wiphand-to-native-Codex controller for P40 research.

This program deliberately dispatches one bounded native Codex turn per
invocation.  It is not a quota-bypass loop and it does not run GPUs locally:
server work, when later unlocked, must be performed through SSH and its
server-side guard.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("P40_MARATHON_REPOSITORY", Path(__file__).resolve().parents[1])).resolve()
MARATHON = ROOT / "marathon"
STATE_PATH = MARATHON / "state.json"
QUOTA_PATH = MARATHON / "quota.json"
LOCK_PATH = MARATHON / "lock"
LOG_DIR = MARATHON / "logs"
QUEUE_DIR = MARATHON / "queue"
QUEUE_PATH = QUEUE_DIR / "trigger.json"
MIN_REMAINING = float(os.environ.get("P40_MARATHON_MIN_FIVE_HOUR_REMAINING_PERCENT", "10"))
MODEL = os.environ.get("P40_MARATHON_MODEL", "gpt-5.6-luna")
MAX_TURN_SECONDS = int(os.environ.get("P40_MARATHON_MAX_TURN_SECONDS", "9000"))


@dataclass(frozen=True)
class Task:
    id: str
    sandbox: str
    marker: str
    prompt: str


TASKS: tuple[Task, ...] = (
    Task(
        "T01",
        "workspace-write",
        "research/results/T01-acceptance.json",
        """Implement T01 only in this repository. Read AGENTS.md if present, EXECUTION_PLAN.md,
DELEGATION_BRIEF.md, research/hardware.md, and research/experiments/001-primitives.md first.
Build the mock-tested guarded benchmark harness: safe snapshot, durable result schema, telemetry
parsing, host-wide lock, child process cleanup, cooling/watchdog checks, a dry-run matrix, and a
fixed prompt fixture. Do not SSH, initialize CUDA, load a model, change fans/BMC settings, alter
production Colibri, or begin T00/T02/T03. Run local tests. End by writing
research/results/T01-acceptance.json containing at least {"task":"T01","status":"pass"}
only if the stated acceptance criteria are actually met; otherwise use status "fail" or "blocked"
and explain why. Commit only task-scoped files and record the acceptance evidence in the experiment
log. Your final response must state the exact commands, changed paths, evidence, risks, and next
eligible task.""",
    ),
    Task(
        "T00",
        "danger-full-access",
        "research/results/T00-thermal-investigation.md",
        """Perform T00 only, serially. Read EXECUTION_PLAN.md, DELEGATION_BRIEF.md and
research/hardware.md. You may use the configured read-only SSH credentials to collect telemetry
from jordanculver@192.168.1.194. Do not run model/CUDA/training workloads, do not change BMC fan
settings, thresholds or SEL, do not restart services, and do not edit production Colibri. The fan
policy is currently fixed at 100 percent, but that configuration record does not pass T00.
Capture five serial idle minutes of BMC/GPU/CPU telemetry, investigate fan/header identity using
read-only evidence, and write research/results/T00-thermal-investigation.md with an explicit
`Acceptance decision: pass`, `fail`, or `blocked_thermal`. A pass requires the documented gate,
not a guessed explanation. Commit only T00-scoped research records. Report exact commands,
timestamps, evidence, risks, and the next eligible task.""",
    ),
    Task(
        "T03",
        "workspace-write",
        "research/results/T03-acceptance.json",
        """Implement T03 only. Read EXECUTION_PLAN.md, DELEGATION_BRIEF.md, and
research/colibri_execution.md. Build a machine-readable Qwen/Colibri capability manifest and
parser/counter fixture tests. Classify every examined setting as engine-read, launcher-only,
backend-reached, or unsupported. Propose isolated instrumentation patch boundaries, but do not
modify /home/jordanculver/colibri_engine, SSH, initialize CUDA, benchmark, load a model, or begin
T00/T02. Run local tests. Write research/results/T03-acceptance.json with
{"task":"T03","status":"pass"} only if the acceptance criteria are met; otherwise write
"fail" or "blocked" with evidence. Commit task-scoped files and report commands, paths, evidence,
risks, and next eligible task.""",
    ),
)


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_event(kind: str, **data: Any) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {"at": int(time.time()), "kind": kind, **data}
    with (LOG_DIR / "controller.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(_json_line(record) + "\n")


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"version": 1, "completed": []}
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid marathon state: {error}") from error
    if not isinstance(state, dict) or not isinstance(state.get("completed", []), list):
        raise RuntimeError("invalid marathon state shape")
    return state


def save_state(state: dict[str, Any]) -> None:
    MARATHON.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)


def save_quota_snapshot(summary: dict[str, Any]) -> None:
    """Persist only non-secret quota state for the isolated runner."""
    MARATHON.mkdir(parents=True, exist_ok=True)
    snapshot = {"checked_at": int(time.time()), **summary}
    temporary = QUOTA_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(QUOTA_PATH)


def fresh_quota_snapshot() -> dict[str, Any]:
    try:
        snapshot = json.loads(QUOTA_PATH.read_text(encoding="utf-8"))
        checked_at = int(snapshot["checked_at"])
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("no valid host quota snapshot; run `scripts/codex_marathon.py prepare` first") from error
    age = int(time.time()) - checked_at
    if age < 0 or age > 900:
        raise RuntimeError(f"host quota snapshot is stale ({age}s); no turn was started")
    if snapshot.get("rate_limit_reached") or snapshot.get("spend_control_reached"):
        raise RuntimeError("host quota snapshot reports a reached limit; no turn was started")
    if float(snapshot.get("five_hour_remaining_percent", 0)) < MIN_REMAINING:
        raise RuntimeError("host quota snapshot is below the configured five-hour headroom")
    return snapshot


def marker_status(marker: Path, task_id: str) -> str | None:
    """Return pass/fail/blocked only for an explicit marker belonging to task_id."""
    if not marker.exists():
        return None
    if marker.suffix == ".json":
        try:
            document = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if document.get("task") != task_id:
            return None
        value = str(document.get("status", "")).strip().lower()
    else:
        content = marker.read_text(encoding="utf-8").lower()
        value = next(
            (
                candidate
                for candidate in ("blocked_thermal", "blocked", "fail", "pass")
                if f"acceptance decision: {candidate}" in content
            ),
            "",
        )
    return value if value in {"pass", "fail", "blocked", "blocked_thermal"} else None


def completed_tasks(state: dict[str, Any]) -> set[str]:
    return {str(value) for value in state.get("completed", [])}


def select_task(requested: str, state: dict[str, Any]) -> Task:
    tasks = {task.id: task for task in TASKS}
    if requested != "auto":
        if requested not in tasks:
            raise RuntimeError(f"unknown task {requested}")
        return tasks[requested]
    complete = completed_tasks(state)
    # T01 and T03 are local-only. Complete both before the separately secured
    # telemetry task, so a scheduled runner never receives general SSH access.
    for task_id in ("T01", "T03", "T00"):
        task = tasks[task_id]
        if task.id not in complete:
            return task
    raise RuntimeError("all controller-managed tasks are complete; T02+ requires human-reviewed extension")


def task_passed(task: Task) -> bool:
    return marker_status(ROOT / task.marker, task.id) == "pass"


def rate_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract only quota state; never persist account identity or reset-credit IDs."""
    limits = payload.get("rateLimits", payload)
    # app-server currently returns one selected limit with ``primary``/``secondary``
    # windows. Accept the earlier list form too so a protocol update fails closed
    # only when neither exposes the five-hour window.
    primary = limits.get("primary")
    if not isinstance(primary, dict):
        buckets = limits.get("rateLimits", limits.get("buckets", []))
        if not isinstance(buckets, list):
            buckets = []
        primary = next((item for item in buckets if item.get("id") == "codex"), None)
        if primary is None:
            primary = next((item for item in buckets if item.get("windowDurationMins") == 300), None)
    if not isinstance(primary, dict):
        raise RuntimeError("could not find the five-hour Codex rate-limit bucket")
    used = primary.get("usedPercent")
    if not isinstance(used, (int, float)):
        raise RuntimeError("five-hour rate-limit bucket did not report usedPercent")
    return {
        "plan": limits.get("planType"),
        "five_hour_used_percent": float(used),
        "five_hour_remaining_percent": max(0.0, 100.0 - float(used)),
        "five_hour_resets_at": primary.get("resetsAt"),
        "rate_limit_reached": limits.get("rateLimitReachedType") is not None,
        "spend_control_reached": bool(limits.get("spendControlReached", False)),
    }


class AppServer:
    def __init__(self) -> None:
        self.process = subprocess.Popen(
            ["codex", "app-server", "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.next_id = 1
        self._request("initialize", {"clientInfo": {"name": "p40-marathon", "version": "1"}}, 30)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def _read(self, timeout: int) -> dict[str, Any]:
        assert self.process.stdout is not None
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError("Codex app-server did not respond in time")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Codex app-server closed its output")
        return json.loads(line)

    def _request(self, method: str, params: dict[str, Any], timeout: int) -> dict[str, Any]:
        assert self.process.stdin is not None
        request_id = self.next_id
        self.next_id += 1
        self.process.stdin.write(_json_line({"id": request_id, "method": method, "params": params}) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + timeout
        while True:
            remaining = max(1, int(deadline - time.monotonic()))
            message = self._read(remaining)
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError(f"Codex {method} failed: {message['error']}")
            return message.get("result", {})

    def rate_limits(self) -> dict[str, Any]:
        return self._request("account/rateLimits/read", {}, 30)

    def start_thread(self, task: Task) -> str:
        result = self._request(
            "thread/start",
            {
                "cwd": str(ROOT),
                "approvalPolicy": "never",
                "sandbox": task.sandbox,
                "model": MODEL,
            },
            60,
        )
        thread = result.get("thread", result)
        thread_id = thread.get("id") if isinstance(thread, dict) else None
        if not thread_id:
            raise RuntimeError("Codex did not return a native thread id")
        self._request(
            "thread/goal/set",
            {
                "threadId": thread_id,
                "objective": "Empirically develop and validate a safer, substantially faster P40-native Qwen3.6 and Colibri inference path; then determine a justified Kimi K3 direction. Follow task gates, preserve production Colibri, and record measured reproducible evidence before advancing.",
                "status": "active",
            },
            30,
        )
        return str(thread_id)

    def start_turn(self, thread_id: str, task: Task) -> str:
        result = self._request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": task.prompt}],
                "effort": "medium",
            },
            60,
        )
        turn = result.get("turn", result)
        turn_id = turn.get("id") if isinstance(turn, dict) else None
        if not turn_id:
            raise RuntimeError("Codex did not return a turn id")
        return str(turn_id)

    def wait_for_turn(self, thread_id: str, turn_id: str, timeout: int) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = int(deadline - time.monotonic())
            if remaining <= 0:
                raise TimeoutError(f"Codex turn {turn_id} exceeded {timeout} seconds")
            # Some app-server builds do not emit terminal notifications to a
            # stdio client while code-mode is active. Poll the authoritative
            # turn list after short quiet windows instead of killing valid work.
            try:
                message = self._read(min(10, max(1, remaining)))
            except TimeoutError:
                result = self._request(
                    "thread/turns/list",
                    {"threadId": thread_id, "limit": 10, "itemsView": "summary"},
                    30,
                )
                for turn in result.get("data", []):
                    if turn.get("id") == turn_id and turn.get("status") != "inProgress":
                        return turn
                continue
            if message.get("method") != "turn/completed":
                continue
            params = message.get("params", {})
            turn = params.get("turn", {})
            if params.get("threadId") == thread_id and turn.get("id") == turn_id:
                return turn

    def start_turn(self, thread_id: str, task: Task) -> str:
        result = self._request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": task.prompt}],
                "effort": "medium",
            },
            60,
        )
        turn = result.get("turn", result)
        turn_id = turn.get("id") if isinstance(turn, dict) else None
        if not turn_id:
            raise RuntimeError("Codex did not return a turn id")
        return str(turn_id)

    def wait_for_turn(self, thread_id: str, turn_id: str, timeout: int) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = int(deadline - time.monotonic())
            if remaining <= 0:
                raise TimeoutError(f"Codex turn {turn_id} exceeded {timeout} seconds")
            message = self._read(min(60, max(1, remaining)))
            if message.get("method") != "turn/completed":
                continue
            params = message.get("params", {})
            turn = params.get("turn", {})
            if params.get("threadId") == thread_id and turn.get("id") == turn_id:
                return turn


@contextlib.contextmanager
def controller_lock() -> Any:
    MARATHON.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another marathon controller invocation holds the lock") from error
        handle.write(f"pid={os.getpid()} at={int(time.time())}\n")
        handle.flush()
        yield


def check_quota(server: AppServer) -> dict[str, Any]:
    summary = rate_summary(server.rate_limits())
    if summary["rate_limit_reached"] or summary["spend_control_reached"]:
        raise RuntimeError("Codex reports a reached limit; no turn was started")
    if summary["five_hour_remaining_percent"] < MIN_REMAINING:
        raise RuntimeError(
            f"only {summary['five_hour_remaining_percent']:.1f}% remains in the five-hour window; "
            f"minimum is {MIN_REMAINING:.1f}%"
        )
    return summary


def command_status() -> int:
    server = AppServer()
    try:
        output = rate_summary(server.rate_limits())
        output["minimum_remaining_percent"] = MIN_REMAINING
        save_quota_snapshot(output)
        print(json.dumps(output, indent=2, sort_keys=True))
    finally:
        server.close()
    return 0


def command_prepare() -> int:
    """Host-only quota preflight; native goals are created with their owning turn."""
    with controller_lock():
        server = AppServer()
        try:
            quota = check_quota(server)
            save_quota_snapshot(quota)
            state = load_state()
            task = select_task("auto", state)
            # Stdio app-server thread IDs are process-local until a turn owns them.
            # Never hand a later process a stale prepared ID.
            state.pop("thread_id", None)
            state["prepared_at"] = int(time.time())
            save_state(state)
            print(json.dumps({"task": task.id, "quota": quota}, indent=2))
            return 0
        finally:
            server.close()


def command_enqueue() -> int:
    """Container-safe handoff: write a fixed request; never invoke Codex here."""
    if os.environ.get("P40_MARATHON_CONTAINER") != "1":
        raise RuntimeError("enqueue is reserved for the isolated Wiphand container")
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"requested": "auto", "enqueued_at": int(time.time()), "source": "wiphand"}
    temporary = QUEUE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(QUEUE_PATH)
    print(json.dumps({"queued": True, "task": "auto"}))
    return 0


def command_host_run() -> int:
    """Service-only native turn on the host; exactly one queued task at a time."""
    if os.environ.get("P40_MARATHON_HOST_RUNNER") != "1":
        raise RuntimeError("host-run is reserved for the user-level systemd service")
    with controller_lock():
        try:
            queued = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
        except FileNotFoundError:
            print("no queued marathon task")
            return 0
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid queued task: {error}") from error
        QUEUE_PATH.unlink()
        requested = str(queued.get("requested", "auto"))
        server = AppServer()
        try:
            quota = check_quota(server)
            save_quota_snapshot(quota)
            state = load_state()
            task = select_task(requested, state)
            if task_passed(task):
                state["completed"] = sorted(completed_tasks(state) | {task.id})
                save_state(state)
                print(f"{task.id} already has a passing acceptance marker; no turn started.")
                return 0
            if task.id == "T00" and os.environ.get("P40_MARATHON_T00_READONLY_GATEWAY") != "1":
                raise RuntimeError("T00 requires the separately reviewed read-only SSH gateway; no turn was started")
            # Keep goal creation and the first turn in this same app-server process.
            # A new stdio app-server cannot resume an unstarted thread from another
            # process, so each bounded scheduled task owns its native goal thread.
            thread_id = server.start_thread(task)
            turn_id = server.start_turn(thread_id, task)
            state.update({
                "thread_id": thread_id,
                "last_task": task.id,
                "last_turn_id": turn_id,
                "last_started_at": int(time.time()),
                "last_quota": quota,
            })
            save_state(state)
            _write_event("host_turn_started", task=task.id, thread_id=thread_id, turn_id=turn_id, quota=quota)
            turn = server.wait_for_turn(thread_id, turn_id, MAX_TURN_SECONDS)
            marker = marker_status(ROOT / task.marker, task.id)
            _write_event("host_turn_completed", task=task.id, thread_id=thread_id, turn_id=turn_id, turn_status=turn.get("status"), marker=marker)
            state.update({
                "last_finished_at": int(time.time()),
                "last_turn_status": turn.get("status"),
                "last_marker_status": marker,
            })
            if marker == "pass":
                state["completed"] = sorted(completed_tasks(state) | {task.id})
            save_state(state)
            print(json.dumps({"task": task.id, "turn_status": turn.get("status"), "marker": marker}, indent=2))
            return 0 if marker == "pass" else 2
        finally:
            server.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="read ChatGPT-backed Codex quota without starting a turn")
    subparsers.add_parser("prepare", help="host-only quota preflight and native-goal thread setup")
    subparsers.add_parser("enqueue", help="isolated Wiphand handoff; queues no model work itself")
    subparsers.add_parser("host-run", help="systemd-only host native Codex task runner")
    args = parser.parse_args()
    if args.command == "status":
        return command_status()
    if args.command == "prepare":
        return command_prepare()
    if args.command == "enqueue":
        return command_enqueue()
    if args.command == "host-run":
        return command_host_run()
    raise AssertionError(f"unexpected command {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, TimeoutError) as error:
        print(f"p40 marathon: {error}", file=sys.stderr)
        raise SystemExit(1)
