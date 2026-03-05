from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from elias_memory.types import MemoryRecord

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class Database:
    def __init__(self, db_path: str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    def _init_schema(self) -> None:
        schema = _SCHEMA_PATH.read_text()
        self._conn.executescript(schema)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def insert(self, rec: MemoryRecord) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO memories
                   (id, content, type, importance, created_at, accessed_at,
                    access_count, metadata, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (rec.id, rec.content, rec.type, rec.importance,
                 rec.created_at.isoformat(), rec.accessed_at.isoformat(),
                 rec.access_count, json.dumps(rec.metadata), rec.embedding),
            )
            self._conn.commit()

    def get(self, memory_id: str) -> MemoryRecord | None:
        row = self.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def delete(self, memory_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.commit()

    def update_access(self, memory_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET access_count = access_count + 1, accessed_at = ? WHERE id = ?",
                (now, memory_id),
            )
            self._conn.commit()

    def update_importance(self, memory_id: str, importance: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET importance = ? WHERE id = ?",
                (max(0.01, min(1.0, importance)), memory_id),
            )
            self._conn.commit()

    def list_all(self) -> list[MemoryRecord]:
        rows = self.execute("SELECT * FROM memories").fetchall()
        return [self._row_to_record(r) for r in rows]

    def _row_to_record(self, row: tuple) -> MemoryRecord:
        return MemoryRecord(
            id=row[0], content=row[1], type=row[2], importance=row[3],
            created_at=datetime.fromisoformat(row[4]),
            accessed_at=datetime.fromisoformat(row[5]),
            access_count=row[6],
            metadata=json.loads(row[7]) if row[7] else {},
            embedding=row[8],
        )

    def close(self) -> None:
        self._conn.close()
