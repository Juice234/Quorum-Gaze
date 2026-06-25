from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from .database import ResultsDatabase
from .debug_capture import DebugCapture
from .downloader import download_with_context, read_image_size, save_image_bytes
from .export import export_dataset
from .image_resolver import (
    build_view_url,
    find_download_or_view_urls,
    find_competition_picture_guid,
    resolve_image_source,
)
from .models import ResultRow
from .utils import ensure_out_dirs, image_paths, normalize_space, winner_code, winner_url

logger = logging.getLogger(__name__)

# coordinate text can show up in a few different formats on BOTB pages
COORD_PATTERNS = [
    re.compile(r"Judges?\s*\([^)]*\)\s*\(\s*X\s*[:=]?\s*(\d+)\s*Y\s*[:=]?\s*(\d+)\s*\)", re.I),
    re.compile(r"JUDGES?\s+POSITION\s+X\s*:\s*(\d+)\s*Y\s*:\s*(\d+)", re.I),
    re.compile(r"JUDGES?\s+POSITION\s+X\s*:\s*(\d+)\s+Y\s*:\s*(\d+)", re.I),
    re.compile(r"JUDGES?'?\s+DECISION.{0,120}?X\s*[:=]?\s*(\d+).{0,60}?Y\s*[:=]?\s*(\d+)", re.I | re.S),
    re.compile(r"\bX\s*[:=]?\s*(\d+)\s+Y\s*[:=]?\s*(\d+)\b", re.I),
    re.compile(r"Judges?.{0,80}?X\s*[:=]?\s*(\d+).{0,40}?Y\s*[:=]?\s*(\d+)", re.I | re.S),
]


@dataclass
class CoordinateResult:
    x: int | None
    y: int | None
    raw_text: str | None = None
    source: str | None = None


def extract_judge_coordinates(text: str) -> tuple[int | None, int | None]:
    result = extract_judge_coordinate_details(text)
    return result.x, result.y


def extract_judge_coordinate_details(*sources: str | None) -> CoordinateResult:
    combined = "\n".join(source for source in sources if source)
    normalized = normalize_space(combined)
    for pattern in COORD_PATTERNS:
        match = pattern.search(normalized)
        if match:
            raw = match.group(0)
            return CoordinateResult(
                x=int(match.group(1)),
                y=int(match.group(2)),
                raw_text=raw,
                source=_coordinate_source(normalized, match.start(), match.end()),
            )
    return CoordinateResult(None, None, None, None)


def _coordinate_source(text: str, start: int, end: int) -> str:
    window = text[max(0, start - 180) : min(len(text), end + 180)].lower()
    scaled_terms = ("scaled", "preview", "display", "clientwidth", "clientheight", "naturalwidth")
    return "scaled_preview" if any(term in window for term in scaled_terms) else "full_resolution"


def extract_title(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return normalize_space(soup.title.string)
    heading = soup.find(["h1", "h2"])
    return normalize_space(heading.get_text(" ")) if heading else None


def is_coordinate_invalid(
    judge_x: int | None,
    judge_y: int | None,
    width: int | None,
    height: int | None,
) -> bool | None:
    if None in (judge_x, judge_y, width, height):
        return None
    return not (0 <= int(judge_x) < int(width) and 0 <= int(judge_y) < int(height))


def should_skip(row: object | None, clean_path: Path, judged_path: Path, overwrite: bool) -> bool:
    if overwrite or row is None:
        return False
    status = row["status"]  # sqlite row
    return status == "ok" and clean_path.exists() and judged_path.exists()


class BotbScraper:
    # main scraper object used by the GUI when the scrape button is clicked
    def __init__(
        self,
        out: Path,
        *,
        delay: float = 1.5,
        timeout_ms: int = 30_000,
        headless: bool = True,
        overwrite: bool = False,
        competition: str = "dreamcar",
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.out = out
        self.delay = delay
        self.timeout_ms = timeout_ms
        self.headless = headless
        self.overwrite = overwrite
        self.competition = competition
        self.progress_callback = progress_callback
        self.last_debug_summary: dict[str, Any] = {}
        ensure_out_dirs(out)

    def _emit_progress(self, event: str, **payload: Any) -> None:
        if self.progress_callback:
            self.progress_callback({"event": event, **payload})

    async def scrape_range(
        self,
        *,
        start_year: int = 20,
        end_year: int = 26,
        start_week: int = 1,
        end_week: int = 52,
    ) -> None:
        total = (end_year - start_year + 1) * (end_week - start_week + 1)
        completed = 0
        self._emit_progress("range-start", total=total)
        with ResultsDatabase(self.out / "botb.sqlite") as db:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=self.headless)
                context = await browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"
                    )
                )
                try:
                    for year in range(start_year, end_year + 1):
                        for week in range(start_week, end_week + 1):
                            await self.scrape_one(year=year, week=week, db=db, context=context)
                            completed += 1
                            self._emit_progress("range-progress", completed=completed, total=total)
                            if self.delay > 0:
                                await asyncio.sleep(self.delay)
                finally:
                    await context.close()
                    await browser.close()

        export_dataset(self.out)
        self._emit_progress("range-finished", completed=completed, total=total)

    async def scrape_single(self, *, year: int, week: int) -> None:
        self._emit_progress("range-start", total=1)
        with ResultsDatabase(self.out / "botb.sqlite") as db:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=self.headless)
                context = await browser.new_context()
                try:
                    await self.scrape_one(year=year, week=week, db=db, context=context)
                finally:
                    await context.close()
                    await browser.close()

        export_dataset(self.out)
        self._emit_progress("range-progress", completed=1, total=1)
        self._emit_progress("range-finished", completed=1, total=1)

    async def scrape_one(self, *, year: int, week: int, db: ResultsDatabase, context: object) -> None:
        competition = self.competition
        code = winner_code(year, week, competition)
        url = winner_url(year, week, competition)
        clean_path, judged_path = image_paths(self.out, year, week, competition)
        debug = DebugCapture(self.out, year, week, competition)
        summary: dict[str, Any] = {
            "winner_url": url,
            "competition": competition,
            "debug_dir": str(debug.debug_dir),
        }
        self.last_debug_summary = summary

        if should_skip(db.get(year, week, competition), clean_path, judged_path, self.overwrite):
            logger.info("Skipping %s; already downloaded", code)
            self._emit_progress("entry-skip", year=year, week=week, competition=competition, code=code, url=url)
            return

        logger.info("Scraping %s: %s", code, url)
        self._emit_progress("entry-start", year=year, week=week, competition=competition, code=code, url=url)
        page = await context.new_page()
        view_page = await context.new_page()
        judged_view_page = await context.new_page()
        debug.attach(page, "winner_page")
        debug.attach(view_page, "view_clean")
        debug.attach(judged_view_page, "view_judged")
        try:
            response = await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            status = response.status if response else None
            if status and status >= 400:
                await debug.save_page(page, "winner_page")
                debug.save_placeholder_page("view_clean", "winner page was missing; clean view not loaded")
                debug.save_placeholder_page("view_judged", "winner page was missing; judged view not loaded")
                debug.save_json()
                db.upsert(
                    ResultRow(
                        url=url,
                        competition=competition,
                        year=year,
                        week=week,
                        status="missing",
                        error=f"HTTP {status}",
                        debug_dir=str(debug.debug_dir),
                    )
                )
                logger.info("Missing %s: HTTP %s", code, status)
                self._emit_progress("entry-missing", year=year, week=week, competition=competition, code=code, url=url, status=status)
                return

            try:
                await page.wait_for_load_state("networkidle", timeout=min(self.timeout_ms, 10_000))
            except Exception:
                pass

            html = await page.content()
            text = await page.locator("body").inner_text(timeout=5_000)
            await debug.save_page(page, "winner_page")
            title = extract_title(html)
            coord = extract_judge_coordinate_details(text, html)
            judge_x, judge_y = coord.x, coord.y
            guid = await find_competition_picture_guid(page)
            download_urls = find_download_or_view_urls(html, url)
            summary.update(
                {
                    "competition_picture_guid": guid,
                    "download_or_view_urls": download_urls,
                    "judge_x": judge_x,
                    "judge_y": judge_y,
                    "raw_coordinate_text": coord.raw_text,
                }
            )
            if not guid:
                raise RuntimeError("could not find competition picture GUID")

            clean_view_url = build_view_url(url, guid, include_judged=False)
            judged_view_url = build_view_url(url, guid, include_judged=True)
            clean_candidates = [candidate for candidate in download_urls if candidate != clean_view_url]
            judged_candidates = [candidate for candidate in download_urls if candidate != judged_view_url]
            clean_result = await resolve_image_source(
                context=context,
                page=view_page,
                view_url=clean_view_url,
                timeout_ms=self.timeout_ms,
                debug=debug,
                debug_prefix="view_clean",
                candidate_urls=clean_candidates,
                include_judged=False,
            )
            judged_result = await resolve_image_source(
                context=context,
                page=judged_view_page,
                view_url=judged_view_url,
                timeout_ms=self.timeout_ms,
                debug=debug,
                debug_prefix="view_judged",
                candidate_urls=judged_candidates,
                include_judged=True,
            )
            summary.update(
                {
                    "clean_view_url": clean_view_url,
                    "judged_view_url": judged_view_url,
                    "clean_source_url": clean_result.source_url,
                    "judged_source_url": judged_result.source_url,
                    "resolver_strategy": f"clean:{clean_result.strategy}; judged:{judged_result.strategy}",
                }
            )

            if clean_result.unresolved or judged_result.unresolved:
                debug.save_json()
                db.upsert(
                    ResultRow(
                        url=url,
                        competition=competition,
                        year=year,
                        week=week,
                        title=title,
                        competition_picture_guid=guid,
                        judge_x=judge_x,
                        judge_y=judge_y,
                        clean_image_path=clean_result.fallback_path,
                        judged_image_path=judged_result.fallback_path,
                        debug_dir=str(debug.debug_dir),
                        coordinate_source=coord.source,
                        clean_source_url=clean_result.source_url,
                        judged_source_url=judged_result.source_url,
                        resolver_strategy=f"clean:{clean_result.strategy}; judged:{judged_result.strategy}",
                        raw_coordinate_text=coord.raw_text,
                        status="image_unresolved",
                        error="could not resolve one or more full-resolution images; see debug folder",
                    )
                )
                logger.warning("Image unresolved for %s; see %s", code, debug.debug_dir)
                return

            await self._save_resolved_image(context, clean_result, clean_path, referer=url)
            await self._save_resolved_image(context, judged_result, judged_path, referer=url)

            width, height = read_image_size(clean_path)
            coordinate_invalid = is_coordinate_invalid(judge_x, judge_y, width, height)
            result_status = "ok" if judge_x is not None and judge_y is not None else "coordinate_unresolved"

            db.upsert(
                ResultRow(
                    url=url,
                    competition=competition,
                    year=year,
                    week=week,
                    title=title,
                    competition_picture_guid=guid,
                    judge_x=judge_x,
                    judge_y=judge_y,
                    image_width=width,
                    image_height=height,
                    clean_image_path=str(clean_path),
                    judged_image_path=str(judged_path),
                    coordinate_invalid=coordinate_invalid,
                    debug_dir=str(debug.debug_dir),
                    coordinate_source=coord.source,
                    clean_source_url=clean_result.source_url,
                    judged_source_url=judged_result.source_url,
                    resolver_strategy=f"clean:{clean_result.strategy}; judged:{judged_result.strategy}",
                    raw_coordinate_text=coord.raw_text,
                    status=result_status,
                    error=None if result_status == "ok" else "could not extract judge coordinates",
                )
            )
            debug.save_json()
            logger.info("Saved %s (%sx%s)", code, width, height)
            self._emit_progress("entry-ok", year=year, week=week, competition=competition, code=code, url=url)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error scraping %s", code)
            self._emit_progress("entry-error", year=year, week=week, competition=competition, code=code, url=url, error=str(exc))
            for capture_page, prefix in (
                (page, "winner_page"),
                (view_page, "view_clean"),
                (judged_view_page, "view_judged"),
            ):
                try:
                    await debug.save_page(capture_page, prefix)
                except Exception as debug_exc:  # noqa: BLE001
                    debug.save_placeholder_page(prefix, f"debug capture failed: {debug_exc}")
            debug.save_json()
            db.upsert(
                ResultRow(
                    url=url,
                    competition=competition,
                    year=year,
                    week=week,
                    debug_dir=str(debug.debug_dir),
                    status="error",
                    error=str(exc),
                )
            )
        finally:
            await page.close()
            await view_page.close()
            await judged_view_page.close()

    async def _save_resolved_image(
        self,
        context: Any,
        resolved: Any,
        path: Path,
        *,
        referer: str,
    ) -> None:
        if resolved.content is not None:
            save_image_bytes(resolved.content, path, resolved.content_type)
            return
        if not resolved.source_url:
            raise RuntimeError("resolved image has no source URL")
        await download_with_context(
            context,
            resolved.source_url,
            path,
            referer=referer,
            timeout_ms=self.timeout_ms,
        )
