from __future__ import annotations

from datetime import datetime, timezone

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
    status: str
    error: str | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
