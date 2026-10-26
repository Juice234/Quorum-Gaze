from __future__ import annotations

import logging
import re
from pathlib import Path

BOTB_BASE_URL = "https://www.botb.com"
COMPETITION_PREFIXES = {
    "dreamcar": "dc",
    "dream_car": "dc",
    "midweek": "mw",
}
COMPETITION_TYPES = {
    "dreamcar": "dream_car",
    "dream_car": "dream_car",
    "dc": "dream_car",
    "midweek": "midweek",
    "mw": "midweek",
}


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def competition_prefix(competition: str) -> str:
    key = competition.lower().strip()
    if key not in COMPETITION_PREFIXES:
        raise ValueError(f"Unsupported competition: {competition}")
    return COMPETITION_PREFIXES[key]


def competition_type(competition: str | None) -> str:
    if not competition:
        return "unknown"
    return COMPETITION_TYPES.get(competition.lower().strip(), "unknown")


def winner_code(year: int, week: int, competition: str = "dreamcar") -> str:
    return f"{competition_prefix(competition)}{week:02d}{year:02d}"


def winner_url(year: int, week: int, competition: str = "dreamcar") -> str:
    return BOTB_BASE_URL + f"/winners/{winner_code(year, week, competition)}"


def ensure_out_dirs(out: Path) -> None:
    (out / "images").mkdir(parents=True, exist_ok=True)


def image_paths(out: Path, year: int, week: int, competition: str = "dreamcar") -> tuple[Path, Path]:
    folder = out / "images" / competition.lower().strip() / f"{year:02d}" / f"{week:02d}"
    return folder / "clean.jpg", folder / "judged.jpg"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
