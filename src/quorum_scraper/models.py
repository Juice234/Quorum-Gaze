from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class ResultRow(BaseModel):
    url: str
    competition: str = "dreamcar"
    competition_type: str | None = None
    year: int
    week: int
    title: str | None = None
    competition_picture_guid: str | None = None
    judge_x: int | None = None
    judge_y: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    clean_image_path: str | None = None
    judged_image_path: str | None = None
    coordinate_invalid: bool | None = None
    debug_dir: str | None = None
    coordinate_source: str | None = None
    clean_source_url: str | None = None
    judged_source_url: str | None = None
    resolver_strategy: str | None = None
    raw_coordinate_text: str | None = None
    status: str
    error: str | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def with_paths(self, clean_path: Path | None, judged_path: Path | None) -> "ResultRow":
        self.clean_image_path = str(clean_path) if clean_path else None
        self.judged_image_path = str(judged_path) if judged_path else None
        return self


class ImageVariantUrls(BaseModel):
    clean_view_url: str
    judged_view_url: str
    clean_image_url: str | None = None
    judged_image_url: str | None = None
