"""SQLite response cache.

Cache keys include provider, model, the full message list, sampling parameters and the
run index, so repeated runs (multi-run stability) are *not* collapsed into one call
while re-evaluations of the same run are free.  The cache doubles as the raw-response
archive that the reproducibility statement refers to.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from ..utils import sha256_text, utc_now


class ResponseCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, provider TEXT, model TEXT, "
            "run_index INTEGER, tag TEXT, request TEXT, response TEXT, created_at TEXT)"
        )
        self._conn.commit()

    @staticmethod
    def make_key(provider: str, model: str, messages: list[dict], params: dict, run_index: int) -> str:
        payload = json.dumps(
            {"provider": provider, "model": model, "messages": messages, "params": params, "run_index": run_index},
            sort_keys=True,
            ensure_ascii=False,
        )
        return sha256_text(payload)

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute("SELECT response FROM responses WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def put(self, key: str, provider: str, model: str, run_index: int, tag: str, request: dict, response: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?)",
                (
                    key,
                    provider,
                    model,
                    run_index,
                    tag,
                    json.dumps(request, ensure_ascii=False),
                    json.dumps(response, ensure_ascii=False),
                    utc_now(),
                ),
            )
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
