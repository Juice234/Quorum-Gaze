from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .models import ResultRow
from .utils import competition_type


# scraped pages go into SQLite so the GUI can reload previous runs
SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    competition TEXT NOT NULL DEFAULT 'dreamcar',
    competition_type TEXT NOT NULL DEFAULT 'unknown',
    year INTEGER NOT NULL,
    week INTEGER NOT NULL,
    title TEXT,
    competition_picture_guid TEXT,
    judge_x INTEGER,
    judge_y INTEGER,
    image_width INTEGER,
    image_height INTEGER,
    clean_image_path TEXT,
    judged_image_path TEXT,
    coordinate_invalid INTEGER,
    status TEXT NOT NULL,
    error TEXT,
    fetched_at TEXT NOT NULL,
    UNIQUE(competition, year, week)
);

CREATE INDEX IF NOT EXISTS idx_results_status ON results(status);
"""


class ResultsDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ResultsDatabase":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def get(self, year: int, week: int, competition: str = "dreamcar") -> sqlite3.Row | None:
        cur = self.conn.execute(
            "SELECT * FROM results WHERE competition = ? AND year = ? AND week = ?",
            (competition, year, week),
        )
        return cur.fetchone()

    def upsert(self, row: ResultRow) -> None:
        payload = row.model_dump()
        payload["competition_type"] = row.competition_type or competition_type(row.competition)
        payload["coordinate_invalid"] = (
            None if row.coordinate_invalid is None else int(row.coordinate_invalid)
        )
        payload["fetched_at"] = row.fetched_at.isoformat()
        columns = [
            "url",
            "competition",
            "competition_type",
            "year",
            "week",
            "title",
            "competition_picture_guid",
            "judge_x",
            "judge_y",
            "image_width",
            "image_height",
            "clean_image_path",
            "judged_image_path",
            "coordinate_invalid",
            "status",
            "error",
            "fetched_at",
        ]
        assignments = ", ".join(f"{col} = excluded.{col}" for col in columns)
        placeholders = ", ".join("?" for _ in columns)
        values = [payload[col] for col in columns]
        self.conn.execute(
            f"""
            INSERT INTO results ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(competition, year, week) DO UPDATE SET {assignments}
            """,
            values,
        )
        self.conn.commit()

    def rows(self) -> list[dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM results ORDER BY competition, year, week")
        return [dict(row) for row in cur.fetchall()]
