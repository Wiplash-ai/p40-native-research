"""Loopback-only JSON API for the bounded Stage-0 controller."""
from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from .scoring import rank_task
from .store import SwarmStateError, SwarmStore


def error(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


class ControllerHandler(BaseHTTPRequestHandler):
    store: SwarmStore
    server_version = "WiplashSwarmS0/0.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _json(self, status: HTTPStatus, body: dict[str, Any] | list[dict[str, Any]]) -> None:
        encoded = json.dumps(body, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 64 * 1024:
            raise SwarmStateError("request body must be 1..65536 bytes")
        try:
            body = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise SwarmStateError("request body must be JSON") from exc
        if not isinstance(body, dict):
            raise SwarmStateError("request body must be an object")
        return body

    @staticmethod
    def _parts(path: str) -> list[str]:
        return [part for part in path.split("?")[0].split("/") if part]

    def do_GET(self) -> None:  # noqa: N802
        parts = self._parts(self.path)
        try:
            if parts == ["v1", "health"]:
                self._json(HTTPStatus.OK, {"status": "ok", "stage": "s0", "dispatch": "disabled"})
                return
            if len(parts) == 3 and parts[:2] == ["v1", "tasks"]:
                self._json(HTTPStatus.OK, self.store.task(parts[2]))
                return
            if len(parts) == 4 and parts[:2] == ["v1", "tasks"] and parts[3] == "evidence":
                task = self.store.task(parts[2])
                body = []
                for branch in task["branches"]:
                    body.extend(self.store.branch_evidence(branch["id"]))
                self._json(HTTPStatus.OK, body)
                return
            if len(parts) == 4 and parts[:2] == ["v1", "tasks"] and parts[3] == "ranking":
                self._json(HTTPStatus.OK, rank_task(self.store, parts[2]))
                return
            self._json(HTTPStatus.NOT_FOUND, error("not_found", "route not found"))
        except KeyError:
            self._json(HTTPStatus.NOT_FOUND, error("not_found", "task not found"))

    def do_POST(self) -> None:  # noqa: N802
        parts = self._parts(self.path)
        try:
            body = self._body()
            if parts == ["v1", "tasks"]:
                task = self.store.create_task(
                    objective=str(body.get("objective", "")),
                    repo_path=str(body.get("repo_path", "")),
                    revision=str(body.get("revision", "")),
                    max_branches=int(body.get("max_branches", 4)),
                    acceptance=list(body.get("acceptance", [])),
                )
                self._json(HTTPStatus.CREATED, task)
                return
            if len(parts) == 4 and parts[:2] == ["v1", "tasks"] and parts[3] == "branches":
                branch = self.store.create_branch(
                    task_id=parts[2], hypothesis=str(body.get("hypothesis", "")),
                    role=str(body.get("role", "")), model_profile=str(body.get("model_profile", "")),
                )
                self._json(HTTPStatus.CREATED, branch)
                return
            if len(parts) == 4 and parts[:2] == ["v1", "branches"] and parts[3] == "promote":
                self._json(HTTPStatus.OK, self.store.promote(parts[2]))
                return
            if len(parts) == 4 and parts[:2] == ["v1", "branches"] and parts[3] == "run":
                self._json(HTTPStatus.CONFLICT, error(
                    "dispatch_disabled", "S0 never executes model-proposed commands or Git changes",
                ))
                return
            self._json(HTTPStatus.NOT_FOUND, error("not_found", "route not found"))
        except KeyError:
            self._json(HTTPStatus.NOT_FOUND, error("not_found", "task or branch not found"))
        except (TypeError, ValueError, SwarmStateError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, error("invalid_request", str(exc)))


def server(*, db_path: Path, host: str = "127.0.0.1", port: int = 8091) -> HTTPServer:
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("S0 controller may bind loopback only")
    store = SwarmStore(db_path)
    handler = type("BoundedControllerHandler", (ControllerHandler,), {"store": store})
    # The controller's metadata path is deliberately serial in S0. Executor
    # concurrency will be owned by the later scheduler, not HTTP threads.
    return HTTPServer((host, port), handler)
