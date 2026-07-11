import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Callable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at   REAL NOT NULL,
  ended_at     REAL,
  model_name   TEXT NOT NULL,
  device       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS detection_events (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id   INTEGER NOT NULL REFERENCES sessions(id),
  track_id     INTEGER NOT NULL,
  class_name   TEXT NOT NULL,
  confidence   REAL NOT NULL,
  entered_at   REAL NOT NULL,
  left_at      REAL,
  max_conf     REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_session ON detection_events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_entered ON detection_events(entered_at);
"""


@dataclass
class EventRow:
    track_id: int
    class_name: str
    confidence: float
    max_conf: float
    entered_at: float
    left_at: float | None


class LoggingStore:
    """Sole SQLite writer for the sidecar. One connection guarded by a lock so
    the capture-request thread and the pipeline thread can share it safely."""

    def __init__(self, db_path: str, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._lock = threading.Lock()
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def start_session(self, model_name: str, device: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions (started_at, model_name, device) VALUES (?, ?, ?)",
                (self._clock(), model_name, device),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def end_session(self, session_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ?",
                (self._clock(), session_id),
            )
            self._conn.commit()

    def record_detection(
        self, session_id: int, track_id: int, cls: str, conf: float, ts: float
    ) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, max_conf FROM detection_events "
                "WHERE session_id = ? AND track_id = ?",
                (session_id, track_id),
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO detection_events "
                    "(session_id, track_id, class_name, confidence, entered_at, left_at, max_conf) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session_id, track_id, cls, conf, ts, None, conf),
                )
            elif conf > row["max_conf"]:
                self._conn.execute(
                    "UPDATE detection_events SET max_conf = ? WHERE id = ?",
                    (conf, row["id"]),
                )
            self._conn.commit()

    def resolve_left(self, session_id: int, track_id: int, ts: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE detection_events SET left_at = ? "
                "WHERE session_id = ? AND track_id = ? AND left_at IS NULL",
                (ts, session_id, track_id),
            )
            self._conn.commit()

    def current_session_id(self) -> int | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM sessions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return int(row["id"]) if row is not None else None

    def query_events(self, session_id: int) -> list[EventRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT track_id, class_name, confidence, max_conf, entered_at, left_at "
                "FROM detection_events WHERE session_id = ? ORDER BY entered_at",
                (session_id,),
            ).fetchall()
        return [
            EventRow(
                track_id=r["track_id"],
                class_name=r["class_name"],
                confidence=r["confidence"],
                max_conf=r["max_conf"],
                entered_at=r["entered_at"],
                left_at=r["left_at"],
            )
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
