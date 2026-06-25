from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from playwright.async_api import Page, Response

SCRIPT_TERMS = (
    "competitionPictureGuid",
    "viewimage",
    "download",
    "includeJudged",
    "includeWinner",
    "Judges",
    "Winner",
    "X:",
    "Y:",
)


def looks_like_image_url(url: str) -> bool:
    lowered = url.lower()
    return (
        lowered.endswith((".jpg", ".jpeg", ".png", ".webp"))
        or "cdn.botb.com" in lowered
        or "media" in lowered
        or "image" in lowered
        or "download" in lowered
        or "viewimage" in lowered
    )


class DebugCapture:
    def __init__(self, out: Path, year: int, week: int, competition: str = "dreamcar") -> None:
        self.debug_dir = out / "debug" / competition.lower().strip() / f"{year:02d}" / f"{week:02d}"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.network: list[dict[str, Any]] = []
        self.links: dict[str, Any] = {}
        self.images: dict[str, Any] = {"candidates": []}

    def attach(self, page: Page, label: str) -> None:
        page.on("response", lambda response: self.record_response(response, label))

    def record_response(self, response: Response, label: str) -> None:
        headers = response.headers
        url = response.url
        lowered = url.lower()
        self.network.append(
            {
                "page": label,
                "url": url,
                "status": response.status,
                "content_type": headers.get("content-type"),
                "content_length": headers.get("content-length"),
                "request_method": response.request.method,
                "resource_type": response.request.resource_type,
                "looks_like_image": looks_like_image_url(url),
                "looks_like_download": "download" in lowered,
                "looks_like_viewimage": "viewimage" in lowered,
                "looks_like_cdn": "cdn.botb.com" in lowered,
                "looks_like_media": "media" in lowered,
            }
        )

    def record_api_response(
        self,
        *,
        label: str,
        url: str,
        status: int,
        headers: dict[str, str],
        method: str = "GET",
    ) -> None:
        lowered = url.lower()
        self.network.append(
            {
                "page": label,
                "url": url,
                "status": status,
                "content_type": headers.get("content-type"),
                "content_length": headers.get("content-length"),
                "request_method": method,
                "resource_type": "api_request",
                "looks_like_image": looks_like_image_url(url),
                "looks_like_download": "download" in lowered,
                "looks_like_viewimage": "viewimage" in lowered,
                "looks_like_cdn": "cdn.botb.com" in lowered,
                "looks_like_media": "media" in lowered,
            }
        )

    async def save_page(self, page: Page, prefix: str) -> None:
        await self._write_text_file(prefix, "html", await self._page_content(page))
        await self._write_text_file(prefix, "txt", await self._page_text(page))
        try:
            await page.screenshot(path=self.debug_dir / f"{prefix}.png", full_page=True)
        except Exception as exc:  # noqa: BLE001
            (self.debug_dir / f"{prefix}.png.error.txt").write_text(str(exc), encoding="utf-8")
        await self.extract_sources(page, prefix)

    def save_placeholder_page(self, prefix: str, message: str) -> None:
        (self.debug_dir / f"{prefix}.html").write_text(f"<!-- {message} -->\n", encoding="utf-8")
        (self.debug_dir / f"{prefix}.txt").write_text(message, encoding="utf-8")
        image = Image.new("RGB", (900, 220), color=(245, 245, 245))
        draw = ImageDraw.Draw(image)
        draw.text((20, 30), message[:500], fill=(20, 20, 20))
        image.save(self.debug_dir / f"{prefix}.png")
        self.links.setdefault(prefix, {"links": [], "script_snippets": []})
        self.images.setdefault(prefix, {"images": [], "backgrounds": []})

    async def save_fallback_screenshot(self, page: Page) -> Path:
        path = self.debug_dir / "fallback_screenshot.png"
        await page.screenshot(path=path, full_page=True)
        return path

    def add_image_candidate(self, source: str, candidate: dict[str, Any]) -> None:
        self.images.setdefault("candidates", []).append({"source": source, **candidate})

    async def extract_sources(self, page: Page, prefix: str) -> None:
        data = await page.evaluate(
            """
            (terms) => {
                const abs = (value) => {
                    try { return new URL(value, document.baseURI).href; } catch { return value; }
                };
                const parseBg = (value) => Array.from(value.matchAll(/url\\((['"]?)(.*?)\\1\\)/g))
                    .map((match) => abs(match[2]))
                    .filter(Boolean);
                const links = Array.from(document.querySelectorAll('a[href]')).map((a) => ({
                    href: abs(a.getAttribute('href')),
                    text: (a.innerText || a.textContent || '').trim().slice(0, 300)
                }));
                const images = Array.from(document.images).map((img) => ({
                    src: img.src || '',
                    currentSrc: img.currentSrc || '',
                    attrSrc: img.getAttribute('src') || '',
                    srcset: img.getAttribute('srcset') || '',
                    width: img.naturalWidth || 0,
                    height: img.naturalHeight || 0
                }));
                const backgrounds = [];
                for (const el of Array.from(document.querySelectorAll('*'))) {
                    const inline = el.getAttribute('style') || '';
                    const computed = window.getComputedStyle(el).backgroundImage || '';
                    for (const url of [...parseBg(inline), ...parseBg(computed)]) {
                        backgrounds.push({url, tag: el.tagName.toLowerCase(), className: el.className || ''});
                    }
                }
                const scriptSnippets = Array.from(document.scripts)
                    .map((script) => script.textContent || '')
                    .filter((text) => terms.some((term) => text.includes(term)))
                    .map((text) => text.slice(0, 5000));
                return {links, images, backgrounds, scriptSnippets};
            }
            """,
            list(SCRIPT_TERMS),
        )
        self.links[prefix] = {
            "links": data.get("links", []),
            "script_snippets": data.get("scriptSnippets", []),
        }
        self.images[prefix] = {
            "images": data.get("images", []),
            "backgrounds": data.get("backgrounds", []),
        }

    async def _page_content(self, page: Page) -> str:
        try:
            return await page.content()
        except Exception as exc:  # noqa: BLE001
            return f"<!-- could not read page content: {exc} -->"

    async def _page_text(self, page: Page) -> str:
        try:
            return await page.locator("body").inner_text(timeout=3_000)
        except Exception as exc:  # noqa: BLE001
            return f"could not read page text: {exc}"

    async def _write_text_file(self, prefix: str, suffix: str, text: str) -> None:
        (self.debug_dir / f"{prefix}.{suffix}").write_text(text, encoding="utf-8", errors="replace")

    def save_json(self) -> None:
        (self.debug_dir / "network.json").write_text(
            json.dumps(self.network, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (self.debug_dir / "links.json").write_text(
            json.dumps(self.links, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (self.debug_dir / "images.json").write_text(
            json.dumps(self.images, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

def extract_image_urls_from_html(html: str) -> list[str]:
    url_pattern = re.compile(r"https?:\\?/\\?/[^\"'\\\s<>]+?\\.(?:jpe?g|png|webp)(?:\\?[^\"'\\\s<>]*)?", re.I)
    urls: list[str] = []
    for match in url_pattern.finditer(html):
        url = match.group(0).replace("\\/", "/")
        urls.append(url)
    return urls
