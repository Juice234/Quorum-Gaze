from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from PIL import Image
from playwright.async_api import BrowserContext

logger = logging.getLogger(__name__)


def save_image_bytes(body: bytes, path: Path, content_type: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lowered = (content_type or "").lower()
    if "image" not in lowered and not body.startswith((b"\xff\xd8", b"\x89PNG", b"RIFF")):
        raise RuntimeError(f"download response was not an image: {content_type or 'unknown'}")
    path.write_bytes(body)


async def download_with_context(
    context: BrowserContext,
    url: str,
    path: Path,
    *,
    referer: str | None = None,
    retries: int = 3,
    timeout_ms: int = 30_000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Referer": referer} if referer else None
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = await context.request.get(
                url,
                headers=headers,
                timeout=timeout_ms,
                fail_on_status_code=False,
            )
            if not response.ok:
                raise RuntimeError(f"download failed with HTTP {response.status}")
            body = await response.body()
            content_type = response.headers.get("content-type", "").lower()
            if "image" not in content_type and not body.startswith((b"\xff\xd8", b"\x89PNG", b"RIFF")):
                raise RuntimeError(f"download response was not an image: {content_type or 'unknown'}")
            save_image_bytes(body, path, content_type)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("Image download attempt %s/%s failed for %s: %s", attempt, retries, url, exc)
            if attempt < retries:
                await asyncio.sleep(1.5 * attempt)

    raise RuntimeError(f"could not download image after {retries} attempts: {last_error}")


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size
