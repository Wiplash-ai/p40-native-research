import importlib.util
import ipaddress
import json
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_router.router import LeaseManager, ModelRouter, ModelSpec, RouterConfig, RouterConfigError, RouterError, ServiceControl


CONFIG = {
    "host": "192.168.1.194",
    "port": 8100,
    "request_max_bytes": 1024,
    "qwen_start_max_temp_c": 50,
    "allowed_cidrs": ["192.168.1.0/24"],
    "models": [
        {
            "alias": "wiplash/qwen35b", "backend_base_url": "http://127.0.0.1:8000/v1",
            "backend_model": "qwen3.6-35b-a3b-colibri-i4", "kind": "colibri_qwen", "idle_seconds": 120,
        },
        {
            "alias": "wiplash/qwen3-8b", "backend_base_url": "http://172.17.0.1:11434/v1",
            "backend_model": "qwen3:8b", "kind": "ollama", "idle_seconds": 180,
        },
    ],
}


class RecordingControl:
    def __init__(self, active=False):
        self.active = active
        self.actions = []

    def qwen_is_active(self):
        return self.active

    def qwen_start(self):
        self.actions.append("start")
        self.active = True

    def qwen_stop(self):
        self.actions.append("stop")
        self.active = False


class RecordingHttp:
    def __init__(self, ps=None):
        self.ps = ps or {"models": []}
        self.requests = []

    def get_json(self, host, port, path, timeout=5):
        self.requests.append(("GET", host, port, path))
        return 200, self.ps

    def request_json(self, host, port, path, body, timeout=10):
        self.requests.append(("POST", host, port, path, body))
        return 200, b'{}'


class CapturingBackend(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, *args):
        return

    def do_POST(self):
        size = int(self.headers["Content-Length"])
        self.__class__.requests.append((self.path, json.loads(self.rfile.read(size))))
        payload = b'{"id":"local-test","object":"chat.completion","choices":[]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class ModelRouterTests(unittest.TestCase):
    def config(self, changes=None):
        raw = json.loads(json.dumps(CONFIG))
        if changes:
            changes(raw)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "router.json"
            path.write_text(json.dumps(raw))
            return RouterConfig.load(path)

    def test_config_rejects_public_listener_and_unapproved_qwen(self):
        with self.assertRaises(RouterConfigError):
            self.config(lambda raw: raw.update(host="0.0.0.0"))
        with self.assertRaises(RouterConfigError):
            self.config(lambda raw: raw.update(allowed_cidrs=["0.0.0.0/0"]))
        with self.assertRaises(RouterConfigError):
            self.config(lambda raw: raw["models"][0].update(backend_model="arbitrary"))
        with self.assertRaises(RouterConfigError):
            self.config(lambda raw: raw["models"][1].update(backend_base_url="http://example.test:11434/v1"))

    def test_qwen_start_requires_no_resident_ollama_and_cool_empty_gpus(self):
        config = self.config()
        control = RecordingControl()
        http = RecordingHttp()
        manager = LeaseManager(config, control, http)
        with patch.object(LeaseManager, "_gpu_state", return_value=([36, 37], [0, 0])):
            manager.begin(config.models["wiplash/qwen35b"])
        self.assertEqual(control.actions, ["start"])
        manager.end(config.models["wiplash/qwen35b"])

        resident = LeaseManager(config, RecordingControl(), RecordingHttp({"models": [{"name": "qwen3:8b"}]}))
        with self.assertRaisesRegex(RouterError, "Ollama model is resident"):
            resident.begin(config.models["wiplash/qwen35b"])

        hot = LeaseManager(config, RecordingControl(), RecordingHttp())
        with patch.object(LeaseManager, "_gpu_state", return_value=([51, 37], [0, 0])):
            with self.assertRaisesRegex(RouterError, "temperature"):
                hot.begin(config.models["wiplash/qwen35b"])

    def test_ollama_is_exclusive_with_qwen(self):
        config = self.config()
        manager = LeaseManager(config, RecordingControl(active=True), RecordingHttp())
        with self.assertRaisesRegex(RouterError, "Qwen is resident"):
            manager.begin(config.models["wiplash/qwen3-8b"])

    def test_idle_reaper_stops_qwen_and_unloads_ollama(self):
        config = self.config()
        control = RecordingControl(active=True)
        http = RecordingHttp()
        manager = LeaseManager(config, control, http)
        manager._last_used["wiplash/qwen35b"] = 1.0
        manager._last_used["wiplash/qwen3-8b"] = 1.0
        stopped = manager.reap_once(now=1000.0)
        self.assertEqual(stopped, ["wiplash/qwen35b", "wiplash/qwen3-8b"])
        self.assertEqual(control.actions, ["stop"])
        self.assertIn(("POST", "172.17.0.1", 11434, "/api/generate", {"model": "qwen3:8b", "keep_alive": 0}), http.requests)

    def test_fixed_control_command_has_no_request_derived_arguments(self):
        control = ServiceControl("/fixed/helper")
        with patch("model_router.router.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = ""
            run.return_value.stderr = ""
            control.qwen_start()
        self.assertEqual(run.call_args.args[0], ["sudo", "-n", "/fixed/helper", "qwen35b", "start"])
        with self.assertRaises(RouterError):
            control._run("restart; arbitrary")

    def test_deployment_files_are_private_lan_authenticated_and_fixed_scope(self):
        unit = (ROOT / "deployment/wiplash-model-router.service").read_text()
        helper = (ROOT / "deployment/wiplash-model-control").read_text()
        sudoers = (ROOT / "deployment/wiplash-model-router.sudoers").read_text()
        config = (ROOT / "deployment/wiplash-model-router.json").read_text()
        self.assertIn("NoNewPrivileges=true", unit)
        self.assertIn("ProtectSystem=strict", unit)
        self.assertIn("EnvironmentFile=/etc/wiplash-model-router.env", unit)
        self.assertIn('"host": "192.168.1.194"', config)
        self.assertIn('"192.168.1.0/24"', config)
        self.assertNotIn("restart", helper)
        self.assertNotIn("*", sudoers)
        self.assertIn("colibri-qwen36.service", helper)

    def test_openai_proxy_rewrites_only_allowlisted_alias_and_strips_client_lifecycle(self):
        backend = ThreadingHTTPServer(("127.0.0.1", 0), CapturingBackend)
        backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
        backend_thread.start()
        try:
            spec = ModelSpec(
                alias="wiplash/qwen3-8b", backend_base_url=f"http://127.0.0.1:{backend.server_port}/v1",
                backend_model="qwen3:8b", kind="ollama", idle_seconds=180,
            )
            config = RouterConfig(
                "127.0.0.1", 0, 1024, 50, (ipaddress.ip_network("127.0.0.0/8"),), {spec.alias: spec},
            )
            router = ModelRouter(config, LeaseManager(config, RecordingControl(), RecordingHttp()), api_key="x" * 32)
            router_thread = threading.Thread(target=router.serve_forever, daemon=True)
            router_thread.start()
            try:
                unauthenticated = HTTPConnection("127.0.0.1", router.server_port, timeout=3)
                unauthenticated.request("GET", "/v1/models")
                self.assertEqual(unauthenticated.getresponse().status, 401)
                unauthenticated.close()
                conn = HTTPConnection("127.0.0.1", router.server_port, timeout=3)
                body = {"model": "wiplash/qwen3-8b", "messages": [], "keep_alive": -1}
                encoded = json.dumps(body).encode()
                conn.request("POST", "/v1/chat/completions", encoded, {
                    "Content-Type": "application/json", "Content-Length": str(len(encoded)),
                    "Authorization": "Bearer " + "x" * 32,
                })
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                response.read()
                conn.close()
            finally:
                router.shutdown()
                router.server_close()
                router_thread.join(timeout=2)
        finally:
            backend.shutdown()
            backend.server_close()
            backend_thread.join(timeout=2)
        path, request = CapturingBackend.requests[-1]
        self.assertEqual(path, "/v1/v1/chat/completions")
        self.assertEqual(request["model"], "qwen3:8b")
        self.assertNotIn("keep_alive", request)


if __name__ == "__main__":
    unittest.main()
