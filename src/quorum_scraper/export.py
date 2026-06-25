from __future__ import annotations

import csv
import logging
from pathlib import Path

from .database import ResultsDatabase

logger = logging.getLogger(__name__)


def export_dataset(out: Path) -> tuple[Path, Path | None]:
    db_path = out / "botb.sqlite"
    csv_path = out / "metadata.csv"
    xlsx_path = out / "metadata.xlsx"

    with ResultsDatabase(db_path) as db:
        rows = db.rows()

    normalized_rows = [_normalize_row(row) for row in rows]
    _write_csv(csv_path, normalized_rows)

    try:
        _write_xlsx(xlsx_path, normalized_rows)
    except ModuleNotFoundError as exc:
        if exc.name != "openpyxl":
            raise
        logger.warning("Skipped XLSX export because openpyxl is not installed")
        return csv_path, None

    return csv_path, xlsx_path


def _normalize_row(row: dict[str, object]) -> dict[str, object]:
    output = dict(row)
    if output.get("coordinate_invalid") == 0:
        output["coordinate_invalid"] = False
    elif output.get("coordinate_invalid") == 1:
        output["coordinate_invalid"] = True
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _write_xlsx(path: Path, rows: list[dict[str, object]]) -> None:
    from openpyxl import Workbook

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "metadata"

    if rows:
        headers = list(rows[0].keys())
        sheet.append(headers)
        for row in rows:
            sheet.append([row.get(header) for header in headers])

    workbook.save(path)

