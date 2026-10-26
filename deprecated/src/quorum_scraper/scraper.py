from __future__ import annotations

import asyncio
import base64
import logging
import re
from html import unescape
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode, urljoin

from bs4 import BeautifulSoup
from PIL import Image
from playwright.async_api import async_playwright

from .database import ResultsDatabase
from .export import export_dataset
from .models import ResultRow
from .utils import ensure_out_dirs, image_paths, normalize_space, winner_code, winner_url

logger = logging.getLogger(__name__)

# the judges' pick is written on the page as text; it shows up in a few formats
COORD_PATTERNS = [
    re.compile(r"Judges?\s*\([^)]*\)\s*\(\s*X\s*[:=]?\s*(\d+)\s*Y\s*[:=]?\s*(\d+)\s*\)", re.I),
    re.compile(r"JUDGES?\s+POSITION\s+X\s*:\s*(\d+)\s*Y\s*:\s*(\d+)", re.I),
    re.compile(r"JUDGES?'?\s+DECISION.{0,120}?X\s*[:=]?\s*(\d+).{0,60}?Y\s*[:=]?\s*(\d+)", re.I | re.S),
    re.compile(r"\bX\s*[:=]?\s*(\d+)\s+Y\s*[:=]?\s*(\d+)\b", re.I),
]

# the actual image sits behind this BOTB endpoint; we just need the picture GUID from the page
VIEW_PATH = "/umbraco/surface/CompetitionResultSurface/viewimage"
GUID_RE = re.compile(
    r"(?:competitionPictureGuid|pictureGuid)\s*[:=]\s*['\"]?([0-9a-fA-F-]{32,36})",
    re.IGNORECASE,
)


def find_coordinate(text: str) -> tuple[int | None, int | None]:
    # try each known format until one matches
    text = normalize_space(text)
    for pattern in COORD_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None, None


def find_picture_guid(html: str) -> str | None:
    # the GUID is in a viewimage link or in a bit of page script; grab the first one
    match = GUID_RE.search(unescape(html))
    return match.group(1) if match else None


def build_view_url(base_url: str, guid: str, include_judged: bool) -> str:
    # ask the endpoint for either the clean image or the one with the judges' ball drawn on
    query = urlencode(
        {
            "competitionPictureGuid": guid,
            "includeWinner": "false",
            "includeJudged": "true" if include_judged else "false",
            "includeUserEntries": "false",
            "includeUserClosest": "false",
        }
    )
    return urljoin(base_url, f"{VIEW_PATH}?{query}")


async def save_view_image(page: Any, url: str, path: Path, timeout_ms: int) -> None:
    # BOTB draws the full image onto a canvas so save that instead of the HTML page
    response = await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
    if response and response.status >= 400:
        raise RuntimeError(f"image page returned HTTP {response.status}")

    canvas = page.locator("canvas").first
    await canvas.wait_for(state="visible", timeout=timeout_ms)
    await page.wait_for_function(
        """() => {
            const canvas = document.querySelector('canvas')
            if (!canvas || canvas.width < 2 || canvas.height < 2) return false
            const pixel = canvas.getContext('2d').getImageData(
                Math.floor(canvas.width / 2),
                Math.floor(canvas.height / 2),
                1,
                1
            ).data
            return pixel[3] > 0
        }""",
        timeout=timeout_ms,
    )

    data_url = await canvas.evaluate("(item) => item.toDataURL('image/jpeg', 0.95)")
    if not data_url.startswith("data:image/jpeg;base64,"):
        raise RuntimeError("BOTB canvas did not return a JPEG image")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(data_url.split(",", 1)[1]))


def extract_title(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return normalize_space(soup.title.string)
    return None


def is_coordinate_invalid(judge_x, judge_y, width, height) -> bool | None:
    # a coordinate is bad if it falls outside the image
    if None in (judge_x, judge_y, width, height):
        return None
    return not (0 <= judge_x < width and 0 <= judge_y < height)


class BotbScraper:
    # used by the GUI when you press "Start scrape"
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
        ensure_out_dirs(out)

    def _progress(self, event: str, **payload: Any) -> None:
        if self.progress_callback:
            self.progress_callback({"event": event, **payload})

    async def scrape_range(self, *, start_year=20, end_year=26, start_week=1, end_week=52) -> None:
        with ResultsDatabase(self.out / "botb.sqlite") as db:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=self.headless)
                context = await browser.new_context()
                try:
                    for year in range(start_year, end_year + 1):
                        for week in range(start_week, end_week + 1):
                            await self.scrape_one(year=year, week=week, db=db, context=context)
                            if self.delay > 0:
                                await asyncio.sleep(self.delay)
                finally:
                    await context.close()
                    await browser.close()
        export_dataset(self.out)
        self._progress("range-finished")

    async def scrape_single(self, *, year: int, week: int) -> None:
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

    async def scrape_one(self, *, year: int, week: int, db: ResultsDatabase, context: Any) -> None:
        competition = self.competition
        code = winner_code(year, week, competition)
        url = winner_url(year, week, competition)
        clean_path, judged_path = image_paths(self.out, year, week, competition)

        # skip if we already have this one (unless overwrite is on)
        existing = db.get(year, week, competition)
        if (
            not self.overwrite
            and existing is not None
            and existing["status"] == "ok"
            and clean_path.exists()
            and judged_path.exists()
        ):
            self._progress("entry-skip", code=code)
            return

        self._progress("entry-start", code=code)
        page = await context.new_page()
        try:
            response = await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            if response and response.status >= 400:
                db.upsert(ResultRow(url=url, competition=competition, year=year, week=week,
                                    status="missing", error=f"HTTP {response.status}"))
                self._progress("entry-missing", code=code)
                return

            # wait for the page to settle so the image links have loaded in
            try:
                await page.wait_for_load_state("networkidle", timeout=min(self.timeout_ms, 10_000))
            except Exception:
                pass

            html = await page.content()
            text = await page.locator("body").inner_text(timeout=5_000)
            title = extract_title(html)
            judge_x, judge_y = find_coordinate(text + "\n" + html)
            guid = find_picture_guid(html)
            if not guid:
                raise RuntimeError("could not find competition picture GUID")

            # BOTB uses the same canvas page for both versions
            await save_view_image(page, build_view_url(url, guid, False), clean_path, self.timeout_ms)
            await save_view_image(page, build_view_url(url, guid, True), judged_path, self.timeout_ms)

            with Image.open(clean_path) as image:
                width, height = image.size
            status = "ok" if judge_x is not None else "coordinate_unresolved"
            db.upsert(ResultRow(
                url=url, competition=competition, year=year, week=week, title=title,
                competition_picture_guid=guid, judge_x=judge_x, judge_y=judge_y,
                image_width=width, image_height=height,
                clean_image_path=str(clean_path), judged_image_path=str(judged_path),
                coordinate_invalid=is_coordinate_invalid(judge_x, judge_y, width, height),
                status=status,
                error=None if status == "ok" else "could not read judge coordinates",
            ))
            logger.info("Saved %s (%sx%s)", code, width, height)
            self._progress("entry-ok", code=code)
        except Exception as exc:  # keep going on the next competition if one fails
            logger.exception("Error scraping %s", code)
            db.upsert(ResultRow(url=url, competition=competition, year=year, week=week,
                                status="error", error=str(exc)))
            self._progress("entry-error", code=code, error=str(exc))
        finally:
            await page.close()
