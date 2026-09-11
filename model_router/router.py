#!/usr/bin/env python3
"""A deliberately small OpenAI-compatible router for approved local models.

The router does not accept arbitrary backend URLs, model names, shell commands,
or listener addresses.  It is intended to run *on the P40 server* and only
bind to loopback.  SSH forwarding remains the remote-access boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hmac
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any
from urllib.parse import urlsplit


MAX_MODELS = 8
MAX_IDLE_SECONDS = 3600
MIN_IDLE_SECONDS = 15
CONTROL_HELPER = "/usr/local/sbin/wiplash-model-control"
QWEN_ALIAS = "wiplash/qwen35b"
QWEN_BACKEND_MODEL = "qwen3.6-35b-a3b-colibri-i4-p40"
PRIVATE_OR_LOOPBACK_NETWORKS = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)


def _is_private_or_loopback_address(address: ipaddress.IPv4Address) -> bool:
    return any(address in network for network in PRIVATE_OR_LOOPBACK_NETWORKS)


def _is_private_or_loopback_network(network: ipaddress.IPv4Network) -> bool:
    return any(network.subnet_of(allowed) for allowed in PRIVATE_OR_LOOPBACK_NETWORKS)


class RouterConfigError(ValueError):
    """The immutable deployment configuration is malformed or unsafe."""


class RouterError(RuntimeError):
    """A request cannot be serviced without violating a scheduling guard."""


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    backend_base_url: str
    backend_model: str
    kind: str
    idle_seconds: int

    @property
    def backend(self) -> tuple[str, int, str]:
        parsed = urlsplit(self.backend_base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "172.17.0.1"}:
            raise RouterConfigError(f"{self.alias}: backend must remain a local HTTP address")
        if parsed.port is None:
            raise RouterConfigError(f"{self.alias}: backend port is required")
        return parsed.hostname, parsed.port, parsed.path.rstrip("/")


@dataclass(frozen=True)
class RouterConfig:
    host: str
    port: int
    request_max_bytes: int
    qwen_start_max_temp_c: int
    allowed_networks: tuple[ipaddress.IPv4Network, ...]
    models: dict[str, ModelSpec]

    @classmethod
    def load(cls, path: str | Path) -> "RouterConfig":
        raw = json.loads(Path(path).read_text())
        expected = {"host", "port", "request_max_bytes", "qwen_start_max_temp_c", "allowed_cidrs", "models"}
        unknown = set(raw) - expected
        if unknown:
            raise RouterConfigError(f"unknown configuration keys: {sorted(unknown)}")
        try:
            host_address = ipaddress.ip_address(raw.get("host"))
        except ValueError as error:
            raise RouterConfigError("host must be an IPv4 loopback or private-LAN address") from error
        if not isinstance(host_address, ipaddress.IPv4Address) or not _is_private_or_loopback_address(host_address):
            raise RouterConfigError("router must bind only to an IPv4 loopback or private-LAN address")
        port = raw.get("port")
        if not isinstance(port, int) or not 1024 <= port <= 65535:
            raise RouterConfigError("port must be an unprivileged TCP port")
        max_bytes = raw.get("request_max_bytes")
        if not isinstance(max_bytes, int) or not 1 <= max_bytes <= 4 * 1024 * 1024:
            raise RouterConfigError("request_max_bytes must be between 1 and 4 MiB")
        max_temp = raw.get("qwen_start_max_temp_c")
        if not isinstance(max_temp, int) or not 30 <= max_temp < 70:
            raise RouterConfigError("qwen_start_max_temp_c must be from 30 through 69")
        allowed_cidrs = raw.get("allowed_cidrs")
        if not isinstance(allowed_cidrs, list) or not allowed_cidrs or len(allowed_cidrs) > 8:
            raise RouterConfigError("allowed_cidrs must contain between 1 and 8 private networks")
        allowed_networks: list[ipaddress.IPv4Network] = []
        for cidr in allowed_cidrs:
            try:
                network = ipaddress.ip_network(cidr, strict=True)
            except ValueError as error:
                raise RouterConfigError(f"invalid allowed network: {cidr!r}") from error
            if not isinstance(network, ipaddress.IPv4Network) or not _is_private_or_loopback_network(network):
                raise RouterConfigError("allowed networks must be IPv4 loopback or private-LAN ranges")
            allowed_networks.append(network)
        entries = raw.get("models")
        if not isinstance(entries, list) or not entries or len(entries) > MAX_MODELS:
            raise RouterConfigError(f"models must contain between 1 and {MAX_MODELS} entries")

        models: dict[str, ModelSpec] = {}
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"alias", "backend_base_url", "backend_model", "kind", "idle_seconds"}:
                raise RouterConfigError("each model requires exactly alias, backend_base_url, backend_model, kind, idle_seconds")
            spec = ModelSpec(**entry)
            if not spec.alias.startswith("wiplash/") or spec.alias in models:
                raise RouterConfigError("model aliases must be unique wiplash/* identifiers")
            if spec.kind not in {"colibri_qwen", "ollama"}:
                raise RouterConfigError(f"{spec.alias}: unsupported model kind")
            if not isinstance(spec.idle_seconds, int) or not MIN_IDLE_SECONDS <= spec.idle_seconds <= MAX_IDLE_SECONDS:
                raise RouterConfigError(f"{spec.alias}: idle_seconds must be between {MIN_IDLE_SECONDS} and {MAX_IDLE_SECONDS}")
            if spec.kind == "colibri_qwen":
                if spec.alias != QWEN_ALIAS or spec.backend_model != QWEN_BACKEND_MODEL:
                    raise RouterConfigError("the only service-managed Colibri model is the approved Qwen alias")
            else:
                if not spec.backend_model or any(ch.isspace() for ch in spec.backend_model):
                    raise RouterConfigError(f"{spec.alias}: invalid Ollama backend model")
            spec.backend  # Validate backend before accepting the configuration.
            models[spec.alias] = spec
        return cls(str(host_address), port, max_bytes, max_temp, tuple(allowed_networks), models)


class ServiceControl:
    """Fixed root helper: no request-derived subprocess arguments are allowed."""

    def __init__(self, helper: str = CONTROL_HELPER):
        self.helper = helper

    def qwen_is_active(self) -> bool:
        result = subprocess.run(
            ["sudo", "-n", self.helper, "qwen35b", "is-active"],
            check=False, text=True, capture_output=True, timeout=10,
        )
        return result.returncode == 0

    def qwen_start(self) -> None:
        self._run("start")

    def qwen_stop(self) -> None:
        self._run("stop")

    def _run(self, action: str) -> None:
        if action not in {"start", "stop"}:
            raise RouterError("invalid service action")
        result = subprocess.run(
            ["sudo", "-n", self.helper, "qwen35b", action],
            check=False, text=True, capture_output=True, timeout=30,
        )
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip() or "fixed service helper failed"
            raise RouterError(detail)


class LocalHttp:
    """Small standard-library client to keep the production router dependency-free."""

    def request_json(self, host: str, port: int, path: str, body: dict[str, Any], timeout: int = 10) -> tuple[int, bytes]:
        payload = json.dumps(body, separators=(",", ":")).encode()
        conn = HTTPConnection(host, port, timeout=timeout)
        try:
            conn.request("POST", path, payload, {"Content-Type": "application/json", "Content-Length": str(len(payload))})
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    def get_json(self, host: str, port: int, path: str, timeout: int = 5) -> tuple[int, dict[str, Any]]:
        conn = HTTPConnection(host, port, timeout=timeout)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            raw = response.read()
        finally:
            conn.close()
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RouterError(f"local backend returned invalid JSON: {error}") from error
        return response.status, decoded


class LeaseManager:
    """Mutual exclusion and conservative Qwen/Ollama residency scheduling."""

    def __init__(self, config: RouterConfig, control: ServiceControl | None = None, http: LocalHttp | None = None):
        self.config = config
        self.control = control or ServiceControl()
        self.http = http or LocalHttp()
        self._lock = threading.RLock()
        self._active = {alias: 0 for alias in config.models}
        self._last_used = {alias: 0.0 for alias in config.models}

    def begin(self, spec: ModelSpec) -> None:
        with self._lock:
            if spec.kind == "colibri_qwen":
                self._prepare_qwen(spec)
            else:
                self._prepare_ollama()
            self._active[spec.alias] += 1

    def end(self, spec: ModelSpec) -> None:
        with self._lock:
            self._active[spec.alias] = max(0, self._active[spec.alias] - 1)
            self._last_used[spec.alias] = time.monotonic()

    def reap_once(self, now: float | None = None) -> list[str]:
        now = time.monotonic() if now is None else now
        stopped: list[str] = []
        with self._lock:
            for spec in self.config.models.values():
                if self._active[spec.alias] or not self._last_used[spec.alias]:
                    continue
                if now - self._last_used[spec.alias] < spec.idle_seconds:
                    continue
                try:
                    if spec.kind == "colibri_qwen":
                        if self.control.qwen_is_active():
                            self.control.qwen_stop()
                    else:
                        host, port, _ = spec.backend
                        self.http.request_json(host, port, "/api/generate", {"model": spec.backend_model, "keep_alive": 0}, timeout=10)
                except (OSError, RouterError, subprocess.SubprocessError):
                    # An idle cleanup failure must not turn a completed client request into a crash loop.
                    continue
                self._last_used[spec.alias] = 0.0
                stopped.append(spec.alias)
        return stopped

    def _prepare_qwen(self, spec: ModelSpec) -> None:
        if not self.control.qwen_is_active():
            ollama_specs = [candidate for candidate in self.config.models.values() if candidate.kind == "ollama"]
            for ollama_spec in ollama_specs:
                host, port, _ = ollama_spec.backend
                status, payload = self.http.get_json(host, port, "/api/ps")
                if status == 200 and payload.get("models"):
                    raise RouterError("an Ollama model is resident; wait for its idle unload before starting Qwen")
            temperatures, memory_used = self._gpu_state()
            if any(temp >= self.config.qwen_start_max_temp_c for temp in temperatures):
                raise RouterError("P40 temperature is above the cool-start threshold")
            if any(mib > 256 for mib in memory_used):
                raise RouterError("GPU memory is in use; refusing to start Qwen beside another workload")
            self.control.qwen_start()
        self._wait_for_qwen_health(spec)

    def _wait_for_qwen_health(self, spec: ModelSpec) -> None:
        host, port, _ = spec.backend
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                status, _ = self.http.get_json(host, port, "/health", timeout=3)
                if status == 200:
                    return
            except (OSError, RouterError):
                pass
            time.sleep(0.5)
        raise RouterError("Qwen service did not become healthy within 180 seconds")

    def _prepare_ollama(self) -> None:
        if self.control.qwen_is_active():
            raise RouterError("Qwen is resident; wait for its idle stop before loading an Ollama model")

    @staticmethod
    def _gpu_state() -> tuple[list[int], list[int]]:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used", "--format=csv,noheader,nounits"],
            check=False, text=True, capture_output=True, timeout=10,
        )
        if result.returncode:
            raise RouterError("nvidia-smi preflight failed")
        temperatures: list[int] = []
        memory_used: list[int] = []
        for line in result.stdout.splitlines():
            parts = [item.strip() for item in line.split(",")]
            if len(parts) != 2:
                raise RouterError("unparseable nvidia-smi preflight")
            temperatures.append(int(parts[0]))
            memory_used.append(int(parts[1]))
        if len(temperatures) != 2:
            raise RouterError("expected exactly two P40 GPUs")
        return temperatures, memory_used


class ModelRouter(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, config: RouterConfig, lease_manager: LeaseManager | None = None, api_key: str | None = None):
        self.config = config
        self.leases = lease_manager or LeaseManager(config)
        self.api_key = api_key or os.environ.get("WIPLASH_MODEL_ROUTER_API_KEY", "")
        if len(self.api_key) < 24:
            raise RouterConfigError("WIPLASH_MODEL_ROUTER_API_KEY must be at least 24 characters")
        self._stop_reaper = threading.Event()
        super().__init__((config.host, config.port), RouterHandler)
        self._reaper_thread = threading.Thread(target=self._reaper, name="model-router-reaper", daemon=True)
        self._reaper_thread.start()

    def _reaper(self) -> None:
        while not self._stop_reaper.wait(5):
            self.leases.reap_once()

    def server_close(self) -> None:
        self._stop_reaper.set()
        super().server_close()


class RouterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: ModelRouter

    def log_message(self, format: str, *args: Any) -> None:
        # Avoid storing prompts, completion text, or authorization headers in journalctl.
        return

    def do_GET(self) -> None:
        if not self._authorized():
            return
        if self.path == "/healthz":
            self._json(200, {"ok": True, "models": sorted(self.server.config.models)})
            return
        if self.path == "/v1/models":
            models = [
                {"id": spec.alias, "object": "model", "owned_by": "wiplash-local"}
                for spec in self.server.config.models.values()
            ]
            self._json(200, {"object": "list", "data": models})
            return
        self._error(404, "not_found", "route is not exposed by the local model router")

    def do_POST(self) -> None:
        if not self._authorized():
            return
        if self.path not in {"/v1/chat/completions", "/v1/completions"}:
            self._error(404, "not_found", "route is not exposed by the local model router")
            return
        body = self._read_json_body()
        if body is None:
            return
        alias = body.get("model")
        if not isinstance(alias, str) or alias not in self.server.config.models:
            self._error(400, "model_not_allowed", "choose a model from GET /v1/models")
            return
        spec = self.server.config.models[alias]
        body["model"] = spec.backend_model
        body.pop("keep_alive", None)  # Lifetime is router policy, never client-controlled.
        try:
            self.server.leases.begin(spec)
        except (OSError, RouterError, subprocess.SubprocessError) as error:
            self._error(409, "model_unavailable", str(error))
            return
        try:
            self._proxy(spec, body)
        finally:
            self.server.leases.end(spec)

    def _read_json_body(self) -> dict[str, Any] | None:
        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header or "")
        except ValueError:
            self._error(411, "content_length_required", "a valid Content-Length header is required")
            return None
        if not 1 <= length <= self.server.config.request_max_bytes:
            self._error(413, "request_too_large", "request body exceeds the router limit")
            return None
        try:
            body = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(400, "invalid_json", "request must contain a JSON object")
            return None
        if not isinstance(body, dict):
            self._error(400, "invalid_json", "request must contain a JSON object")
            return None
        return body

    def _authorized(self) -> bool:
        try:
            peer = ipaddress.ip_address(self.client_address[0])
        except ValueError:
            self._error(401, "unauthorized", "authentication required")
            return False
        if not any(peer in network for network in self.server.config.allowed_networks):
            self._error(401, "unauthorized", "authentication required")
            return False
        prefix = "Bearer "
        header = self.headers.get("Authorization", "")
        candidate = header[len(prefix):] if header.startswith(prefix) else ""
        if not candidate or not hmac.compare_digest(candidate, self.server.api_key):
            self._error(401, "unauthorized", "authentication required")
            return False
        return True

    def _proxy(self, spec: ModelSpec, body: dict[str, Any]) -> None:
        host, port, base_path = spec.backend
        payload = json.dumps(body, separators=(",", ":")).encode()
        conn = HTTPConnection(host, port, timeout=600)
        upstream_path = self.path if self.path.startswith(f"{base_path}/") else f"{base_path}{self.path}"
        try:
            conn.request("POST", upstream_path, payload, {
                "Content-Type": "application/json", "Content-Length": str(len(payload)),
                "Accept": self.headers.get("Accept", "application/json"),
            })
            response = conn.getresponse()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in {"connection", "content-length", "transfer-encoding", "keep-alive"}:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            while chunk := response.read(64 * 1024):
                self.wfile.write(chunk)
                self.wfile.flush()
            self.close_connection = True
        except (OSError, TimeoutError) as error:
            if not self.wfile.closed:
                self._error(502, "backend_unreachable", str(error))
        finally:
            conn.close()

    def _json(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, status: int, code: str, message: str) -> None:
        self._json(status, {"error": {"code": code, "message": message, "type": "invalid_request_error"}})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="root-owned router JSON configuration")
    args = parser.parse_args()
    config = RouterConfig.load(args.config)
    server = ModelRouter(config)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
