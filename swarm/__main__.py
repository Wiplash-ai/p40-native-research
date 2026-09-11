"""CLI for the bounded controller and read-only capacity probe."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .api import server
from .probe import CapacityProbeError, collect, write_snapshot


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m swarm")
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe", help="collect fixed read-only S0 capacity evidence")
    probe.add_argument("--host", default="jordanculver@192.168.1.194")
    probe.add_argument("--timeout", type=int, default=15)
    probe.add_argument("--output", type=Path)
    serve = commands.add_parser("serve", help="run the loopback-only S0 API")
    serve.add_argument("--db", type=Path, default=ROOT / "swarm" / "state" / "swarm.sqlite3")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    if args.command == "probe":
        try:
            snapshot = collect(args.host, args.timeout)
        except (CapacityProbeError, OSError) as exc:
            parser.error(str(exc))
        output = args.output
        if output is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output = ROOT / "research" / "results" / "raw" / f"S0-capacity-{stamp}.json"
        write_snapshot(snapshot, output)
        print(json.dumps({"status": "pass", "output": str(output), **snapshot.json()}, sort_keys=True))
        return 0
    httpd = server(db_path=args.db, host=args.host, port=args.port)
    print(json.dumps({"host": args.host, "port": args.port, "stage": "s0"}, sort_keys=True))
    try:
        httpd.serve_forever()
    finally:
        httpd.RequestHandlerClass.store.close()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
