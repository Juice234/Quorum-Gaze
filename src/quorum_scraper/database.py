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
    debug_dir TEXT,
    coordinate_source TEXT,
    clean_source_url TEXT,
    judged_source_url TEXT,
    resolver_strategy TEXT,
    raw_coordinate_text TEXT,
    cv_circle_x REAL,
    cv_circle_y REAL,
    label_error_px REAL,
    label_quality TEXT,
    status TEXT NOT NULL,
    error TEXT,
    fetched_at TEXT NOT NULL,
    UNIQUE(competition, year, week)
);

CREATE INDEX IF NOT EXISTS idx_results_status ON results(status);

CREATE TABLE IF NOT EXISTS vision_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id INTEGER NOT NULL,
    competition_type TEXT NOT NULL DEFAULT 'unknown',
    model_name TEXT,
    image_path TEXT,
    image_width INTEGER,
    image_height INTEGER,
    detections_json TEXT,
    num_people INTEGER,
    num_pose_people INTEGER,
    status TEXT NOT NULL,
    error TEXT,
    extracted_at TEXT NOT NULL,
    UNIQUE(result_id, model_name),
    FOREIGN KEY(result_id) REFERENCES results(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_vision_features_result_id ON vision_features(result_id);
CREATE INDEX IF NOT EXISTS idx_vision_features_status ON vision_features(status);

CREATE TABLE IF NOT EXISTS vision_detection_overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id INTEGER NOT NULL,
    detection_index INTEGER NOT NULL,
    manual_role TEXT NOT NULL,
    notes TEXT,
    reviewed_at TEXT NOT NULL,
    UNIQUE(result_id, detection_index),
    FOREIGN KEY(result_id) REFERENCES results(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_vision_detection_overrides_result_id
ON vision_detection_overrides(result_id);

CREATE TABLE IF NOT EXISTS vision_manual_review (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id INTEGER NOT NULL UNIQUE,
    people_detection_ok INTEGER,
    pose_ok INTEGER,
    head_points_ok INTEGER,
    body_orientation_ok INTEGER,
    active_filter_ok INTEGER,
    overall_usable_for_feature_model INTEGER,
    notes TEXT,
    reviewed_at TEXT NOT NULL,
    FOREIGN KEY(result_id) REFERENCES results(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_vision_manual_review_result_id
ON vision_manual_review(result_id);
"""


class ResultsDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        existing = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(results)").fetchall()
        }
        indexes = self.conn.execute("PRAGMA index_list(results)").fetchall()
        has_old_unique = any(
            row["unique"]
            and [
                info["name"]
                for info in self.conn.execute(f"PRAGMA index_info({row['name']})").fetchall()
            ] == ["year", "week"]
            for row in indexes
        )
        if "competition" not in existing or has_old_unique:
            self._rebuild_results_table()
            existing = {
                row["name"]
                for row in self.conn.execute("PRAGMA table_info(results)").fetchall()
            }
        additions = {
            "competition_type": "TEXT",
            "debug_dir": "TEXT",
            "coordinate_source": "TEXT",
            "clean_source_url": "TEXT",
            "judged_source_url": "TEXT",
            "resolver_strategy": "TEXT",
            "raw_coordinate_text": "TEXT",
            "cv_circle_x": "REAL",
            "cv_circle_y": "REAL",
            "label_error_px": "REAL",
            "label_quality": "TEXT",
        }
        for column, column_type in additions.items():
            if column not in existing:
                self.conn.execute(f"ALTER TABLE results ADD COLUMN {column} {column_type}")
        self.conn.execute(
            """
            UPDATE results
            SET competition_type = CASE
                WHEN competition IN ('dreamcar', 'dream_car', 'dc') THEN 'dream_car'
                WHEN competition IN ('midweek', 'mw') THEN 'midweek'
                ELSE 'unknown'
            END
            WHERE competition_type IS NULL OR competition_type = ''
            """
        )
        self.conn.executescript(SCHEMA)

    def _rebuild_results_table(self) -> None:
        old_columns = [
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(results)").fetchall()
        ]
        self.conn.execute("ALTER TABLE results RENAME TO results_old")
        self.conn.executescript(SCHEMA)
        new_columns = [
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(results)").fetchall()
            if row["name"] in old_columns or row["name"] in ("competition", "competition_type")
        ]
        select_columns = [
            "'dreamcar' AS competition" if column == "competition" and column not in old_columns else column
            for column in new_columns
        ]
        select_columns = [
            (
                "'dream_car' AS competition_type"
                if column == "competition_type" and column not in old_columns
                else column
            )
            for column in select_columns
        ]
        self.conn.execute(
            f"""
            INSERT OR IGNORE INTO results ({", ".join(new_columns)})
            SELECT {", ".join(select_columns)}
            FROM results_old
            """
        )
        self.conn.execute("DROP TABLE results_old")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_results_status ON results(status)")

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
            "debug_dir",
            "coordinate_source",
            "clean_source_url",
            "judged_source_url",
            "resolver_strategy",
            "raw_coordinate_text",
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

    def update_validation(
        self,
        year: int,
        week: int,
        competition: str,
        width: int | None,
        height: int | None,
        coordinate_invalid: bool | None,
        error: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE results
            SET image_width = ?,
                image_height = ?,
                coordinate_invalid = ?,
                error = ?
            WHERE competition = ? AND year = ? AND week = ?
            """,
            (
                width,
                height,
                None if coordinate_invalid is None else int(coordinate_invalid),
                error,
                competition,
                year,
                week,
            ),
        )
        self.conn.commit()

    def update_label_validation(
        self,
        year: int,
        week: int,
        competition: str,
        cv_circle_x: float | None,
        cv_circle_y: float | None,
        label_error_px: float | None,
        label_quality: str,
    ) -> None:
        self.conn.execute(
            """
            UPDATE results
            SET cv_circle_x = ?,
                cv_circle_y = ?,
                label_error_px = ?,
                label_quality = ?
            WHERE competition = ? AND year = ? AND week = ?
            """,
            (cv_circle_x, cv_circle_y, label_error_px, label_quality, competition, year, week),
        )
        self.conn.commit()

    def rows(self) -> list[dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM results ORDER BY competition, year, week")
        return [dict(row) for row in cur.fetchall()]

    def usable_rows(self) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT *
            FROM results
            WHERE status = 'ok'
              AND coordinate_invalid = 0
              AND judge_x IS NOT NULL
              AND judge_y IS NOT NULL
              AND clean_image_path IS NOT NULL
              AND image_width IS NOT NULL
              AND image_height IS NOT NULL
              AND (label_quality IS NULL OR label_quality = '' OR label_quality = 'good')
            ORDER BY competition_type, year, week
            """
        )
        return [dict(row) for row in cur.fetchall()]

    def upsert_vision_features(
        self,
        *,
        result_id: int,
        competition_type_value: str,
        model_name: str,
        image_path: str,
        image_width: int | None,
        image_height: int | None,
        detections_json: str,
        num_people: int,
        num_pose_people: int,
        status: str,
        error: str | None,
        extracted_at: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO vision_features (
                result_id,
                competition_type,
                model_name,
                image_path,
                image_width,
                image_height,
                detections_json,
                num_people,
                num_pose_people,
                status,
                error,
                extracted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(result_id, model_name) DO UPDATE SET
                competition_type = excluded.competition_type,
                image_path = excluded.image_path,
                image_width = excluded.image_width,
                image_height = excluded.image_height,
                detections_json = excluded.detections_json,
                num_people = excluded.num_people,
                num_pose_people = excluded.num_pose_people,
                status = excluded.status,
                error = excluded.error,
                extracted_at = excluded.extracted_at
            """,
            (
                result_id,
                competition_type_value,
                model_name,
                image_path,
                image_width,
                image_height,
                detections_json,
                num_people,
                num_pose_people,
                status,
                error,
                extracted_at,
            ),
        )
        self.conn.commit()

    def vision_rows(self, model_name: str | None = None) -> list[dict[str, Any]]:
        if model_name:
            cur = self.conn.execute(
                """
                SELECT vf.*, r.year, r.week, r.url, r.judge_x, r.judge_y
                FROM vision_features vf
                JOIN results r ON r.id = vf.result_id
                WHERE vf.model_name = ?
                ORDER BY vf.competition_type, r.year, r.week
                """,
                (model_name,),
            )
        else:
            cur = self.conn.execute(
                """
                SELECT vf.*, r.year, r.week, r.url, r.judge_x, r.judge_y
                FROM vision_features vf
                JOIN results r ON r.id = vf.result_id
                ORDER BY vf.competition_type, r.year, r.week
                """
            )
        return [dict(row) for row in cur.fetchall()]

    def get_detection_overrides(self, result_id: int) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT *
            FROM vision_detection_overrides
            WHERE result_id = ?
            ORDER BY detection_index
            """,
            (result_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def upsert_detection_override(
        self,
        *,
        result_id: int,
        detection_index: int,
        manual_role: str,
        notes: str | None,
        reviewed_at: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO vision_detection_overrides (
                result_id, detection_index, manual_role, notes, reviewed_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(result_id, detection_index) DO UPDATE SET
                manual_role = excluded.manual_role,
                notes = excluded.notes,
                reviewed_at = excluded.reviewed_at
            """,
            (result_id, detection_index, manual_role, notes, reviewed_at),
        )
        self.conn.commit()

    def delete_detection_overrides(self, result_id: int) -> None:
        self.conn.execute(
            "DELETE FROM vision_detection_overrides WHERE result_id = ?",
            (result_id,),
        )
        self.conn.commit()

    def get_manual_vision_review(self, result_id: int) -> dict[str, Any] | None:
        cur = self.conn.execute(
            "SELECT * FROM vision_manual_review WHERE result_id = ?",
            (result_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def upsert_manual_vision_review(
        self,
        *,
        result_id: int,
        people_detection_ok: int | None,
        pose_ok: int | None,
        head_points_ok: int | None,
        body_orientation_ok: int | None,
        active_filter_ok: int | None,
        overall_usable_for_feature_model: int | None,
        notes: str | None,
        reviewed_at: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO vision_manual_review (
                result_id,
                people_detection_ok,
                pose_ok,
                head_points_ok,
                body_orientation_ok,
                active_filter_ok,
                overall_usable_for_feature_model,
                notes,
                reviewed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(result_id) DO UPDATE SET
                people_detection_ok = excluded.people_detection_ok,
                pose_ok = excluded.pose_ok,
                head_points_ok = excluded.head_points_ok,
                body_orientation_ok = excluded.body_orientation_ok,
                active_filter_ok = excluded.active_filter_ok,
                overall_usable_for_feature_model = excluded.overall_usable_for_feature_model,
                notes = excluded.notes,
                reviewed_at = excluded.reviewed_at
            """,
            (
                result_id,
                people_detection_ok,
                pose_ok,
                head_points_ok,
                body_orientation_ok,
                active_filter_ok,
                overall_usable_for_feature_model,
                notes,
                reviewed_at,
            ),
        )
        self.conn.commit()

    def list_manual_vision_reviews(self) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT vmr.*, r.competition_type, r.year, r.week, r.url
            FROM vision_manual_review vmr
            JOIN results r ON r.id = vmr.result_id
            ORDER BY r.competition_type, r.year, r.week
            """
        )
        return [dict(row) for row in cur.fetchall()]
