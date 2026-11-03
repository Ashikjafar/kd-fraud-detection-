"""RESTful scoring and case-management API (standard library HTTP server).

Endpoints
    GET  /health                      liveness + model version              (no auth)
    POST /score                       score one transaction                 (customer, analyst, admin)
    GET  /cases?status=open           analyst case queue                    (analyst, admin)
    POST /cases/<id>/resolve          confirm fraud / dismiss + feedback    (analyst, admin)
    GET  /metrics                     training/evaluation metrics           (admin)
    GET  /dashboard                   analytics dashboard (HTML)            (analyst, admin)

Access control: set FRAUD_API_KEYS="key1:admin,key2:analyst,key3:customer" and send
the key in the ``X-API-Key`` header. If the variable is not set the API runs in
open development mode (and says so at start-up).

Run:  python -m fraud_detection.api --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

from .cases import STATUSES, CaseStore, OutboxNotifier
from .preprocess import clean_transactions
from .rules import RuleConfig
from .scoring import Scorer, limit_threads_for_serving

ROLE_PERMISSIONS = {
    "customer": {"score"},
    "analyst": {"score", "cases", "resolve", "dashboard"},
    "admin": {"score", "cases", "resolve", "dashboard", "metrics"},
}
MAX_BODY = 64 * 1024


def parse_api_keys(value: Optional[str]) -> Dict[str, str]:
    keys: Dict[str, str] = {}
    for item in (value or "").split(","):
        if ":" in item:
            key, role = item.strip().rsplit(":", 1)
            if key and role in ROLE_PERMISSIONS:
                keys[key] = role
    return keys


class FraudAPIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, scorer: Scorer, cases: CaseStore, notifier: OutboxNotifier,
                 api_keys: Optional[Dict[str, str]] = None, reports_dir: str = "reports"):
        super().__init__(address, Handler)
        self.scorer, self.cases, self.notifier = scorer, cases, notifier
        self.api_keys = api_keys or {}
        self.reports_dir = Path(reports_dir)


class Handler(BaseHTTPRequestHandler):
    server: FraudAPIServer
    protocol_version = "HTTP/1.1"
    # Headers and body are separate small writes; without TCP_NODELAY, Nagle's algorithm
    # plus delayed ACKs adds ~40 ms to every response.
    disable_nagle_algorithm = True

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    # ---- helpers ----
    def _send(self, status: int, body, content_type: str = "application/json") -> None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _allowed(self, permission: str) -> bool:
        keys = self.server.api_keys
        if not keys:
            return True  # open development mode
        role = keys.get(self.headers.get("X-API-Key", ""))
        if role is None:
            self._send(401, {"error": "missing or invalid X-API-Key"})
            return False
        if permission not in ROLE_PERMISSIONS[role]:
            self._send(403, {"error": f"role '{role}' may not use this endpoint"})
            return False
        return True

    def _read_body(self) -> Optional[bytes]:
        """Always consume the request body first, so an early 401/403/422 reply cannot
        leave unread bytes that a keep-alive connection would parse as the next request."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            self.close_connection = True  # cannot safely resynchronise this connection
            self._send(413 if length > MAX_BODY else 400, {"error": "invalid or oversized body"})
            return None
        return self.rfile.read(length)

    def _parse_json(self, raw: bytes) -> Optional[dict]:
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "body is not valid JSON"})
            return None

    # ---- routes ----
    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/health":
            return self._send(200, {"status": "ok", "model_version": self.server.scorer.version,
                                    "auth": "enabled" if self.server.api_keys else "disabled (development mode)"})
        if url.path == "/cases":
            if not self._allowed("cases"):
                return
            status = parse_qs(url.query).get("status", [None])[0]
            if status and status not in STATUSES:
                return self._send(400, {"error": f"status must be one of {list(STATUSES)}"})
            return self._send(200, {"cases": self.server.cases.list(status), "counts": self.server.cases.stats()})
        if url.path == "/metrics":
            if not self._allowed("metrics"):
                return
            path = self.server.reports_dir / "metrics.json"
            return self._send(200, path.read_bytes()) if path.exists() else self._send(404, {"error": "no metrics yet"})
        if url.path in ("/", "/dashboard"):
            if not self._allowed("dashboard"):
                return
            path = self.server.reports_dir / "dashboard.html"
            if path.exists():
                return self._send(200, path.read_bytes(), "text/html; charset=utf-8")
            return self._send(404, {"error": "dashboard not generated; run python -m fraud_detection.report"})
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        url = urlparse(self.path)
        raw = self._read_body()
        if raw is None:
            return
        if url.path == "/score":
            if not self._allowed("score"):
                return
            body = self._parse_json(raw)
            if body is None:
                return
            try:
                result = self.server.scorer.score(body)
            except ValueError as exc:
                return self._send(422, {"error": str(exc)})
            if result["decision"] in ("review", "block"):
                amount = float(body["amount"])
                case_id = self.server.cases.create(result, amount)
                self.server.notifier.notify(case_id, result)
                result["case_id"] = case_id
            return self._send(200, result)

        match = re.fullmatch(r"/cases/(\d+)/resolve", url.path)
        if match:
            if not self._allowed("resolve"):
                return
            body = self._parse_json(raw)
            if body is None:
                return
            try:
                case = self.server.cases.resolve(
                    int(match.group(1)), str(body.get("status", "")),
                    str(body.get("analyst", "")), str(body.get("notes", "")),
                )
            except ValueError as exc:
                return self._send(422, {"error": str(exc)})
            return self._send(200, case) if case else self._send(404, {"error": "case not found"})
        self._send(404, {"error": "not found"})


def build_server(host: str, port: int, scorer: Scorer, db_path: str = "runtime/cases.db",
                 outbox_path: str = "runtime/alerts_outbox.jsonl", api_keys: Optional[Dict[str, str]] = None,
                 reports_dir: str = "reports") -> FraudAPIServer:
    return FraudAPIServer((host, port), scorer, CaseStore(db_path), OutboxNotifier(outbox_path), api_keys, reports_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fraud scoring API")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model-dir", default=None, help="defaults to the latest version in models/registry.json")
    ap.add_argument("--rules", default=None, help="optional JSON file with rule settings")
    ap.add_argument("--warm-start", default="data/raw_transactions.csv",
                    help="historical transactions used to initialise customer profiles (skipped if missing)")
    ap.add_argument("--allow-open", action="store_true",
                    help="allow listening on a non-loopback address with no API keys (not recommended)")
    args = ap.parse_args()

    limit_threads_for_serving()
    scorer = Scorer.load(args.model_dir, rules=RuleConfig.from_json(args.rules))
    if args.warm_start and Path(args.warm_start).exists():
        import pandas as pd
        cleaned, _ = clean_transactions(pd.read_csv(args.warm_start))
        print(f"warm start: replayed {scorer.warm_start(cleaned):,} historical transactions")
    keys = parse_api_keys(os.environ.get("FRAUD_API_KEYS"))
    if not keys:
        if args.host not in ("127.0.0.1", "localhost", "::1") and not args.allow_open:
            raise SystemExit(
                f"refusing to listen on {args.host} without authentication: set FRAUD_API_KEYS "
                "(e.g. 'key1:admin,key2:analyst') or pass --allow-open to override"
            )
        print("WARNING: FRAUD_API_KEYS not set - running in open development mode (no authentication)")
    server = build_server(args.host, args.port, scorer, api_keys=keys)
    print(f"model {scorer.version} listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
