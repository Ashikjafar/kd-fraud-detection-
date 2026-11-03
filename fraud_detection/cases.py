"""Alert and case management: analyst case queue (SQLite) and alert outbox."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

STATUSES = ("open", "confirmed_fraud", "dismissed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    amount REAL NOT NULL,
    fraud_probability REAL NOT NULL,
    risk_level TEXT NOT NULL,
    decision TEXT NOT NULL,
    reasons TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    analyst TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    reviewed_at TEXT
)
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CaseStore:
    """Persistent queue of flagged transactions awaiting analyst review."""

    def __init__(self, path: str) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(_SCHEMA)

    @contextmanager
    def _conn(self):
        """Open a connection, commit on success (roll back on error) and always close it."""
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def create(self, result: dict, amount: float) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO cases (transaction_id, customer_id, amount, fraud_probability, risk_level,"
                " decision, reasons, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (result["transaction_id"], result["customer_id"], float(amount), result["fraud_probability"],
                 result["risk_level"], result["decision"], json.dumps(result["reasons"]), _now()),
            )
            return int(cur.lastrowid)

    def list(self, status: Optional[str] = None, limit: int = 100) -> List[dict]:
        sql, args = "SELECT * FROM cases", []
        if status:
            sql += " WHERE status = ?"
            args.append(status)
        sql += " ORDER BY fraud_probability DESC, id DESC LIMIT ?"
        args.append(int(limit))
        with self._conn() as c:
            rows = [dict(r) for r in c.execute(sql, args)]
        for r in rows:
            r["reasons"] = json.loads(r["reasons"])
        return rows

    def resolve(self, case_id: int, status: str, analyst: str = "", notes: str = "") -> Optional[dict]:
        if status not in STATUSES or status == "open":
            raise ValueError(f"status must be one of {STATUSES[1:]}")
        with self._conn() as c:
            cur = c.execute(
                "UPDATE cases SET status=?, analyst=?, notes=?, reviewed_at=? WHERE id=?",
                (status, analyst, notes, _now(), int(case_id)),
            )
            if cur.rowcount == 0:
                return None
            row = c.execute("SELECT * FROM cases WHERE id=?", (int(case_id),)).fetchone()
        out = dict(row)
        out["reasons"] = json.loads(out["reasons"])
        return out

    def feedback_labels(self) -> List[dict]:
        """Analyst decisions, to be merged into the next retraining run."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT transaction_id, status = 'confirmed_fraud' AS is_fraud FROM cases WHERE status != 'open'"
            ).fetchall()
        return [{"transaction_id": r["transaction_id"], "is_fraud": int(r["is_fraud"])} for r in rows]

    def stats(self) -> dict:
        with self._conn() as c:
            rows = c.execute("SELECT status, COUNT(*) AS n FROM cases GROUP BY status").fetchall()
        counts = {s: 0 for s in STATUSES}
        counts.update({r["status"]: r["n"] for r in rows})
        return counts


class OutboxNotifier:
    """Writes each alert as a JSON line to an outbox file.

    This is the integration point for e-mail/SMS: replace ``notify`` with a call
    to an SMTP server or an SMS gateway. No message is actually sent by default.
    """

    def __init__(self, outbox_path: str) -> None:
        self.path = Path(outbox_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def notify(self, case_id: int, result: dict) -> None:
        line = {
            "time": _now(), "case_id": case_id, "channel": "email+sms (outbox only)",
            "to": "fraud-analysts", "decision": result["decision"],
            "transaction_id": result["transaction_id"], "fraud_probability": result["fraud_probability"],
        }
        with self.path.open("a") as f:
            f.write(json.dumps(line) + "\n")
