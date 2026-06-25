from __future__ import annotations

import asyncio
import base64
import re
from dataclasses import dataclass
from html import unescape
from io import BytesIO
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup
from playwright.async_api import BrowserContext, Page, Response

from .debug_capture import DebugCapture, extract_image_urls_from_html, looks_like_image_url

VIEW_PATH = "/umbraco/surface/CompetitionResultSurface/viewimage"
GUID_RE = re.compile(
    r"(?:competitionPictureGuid|competitionPictureGUID|pictureGuid|PictureGuid)"
    r"\s*[:=]\s*['\"]?([0-9a-fA-F-]{32,36})",
    re.IGNORECASE,
)
VIEW_URL_RE = re.compile(
    r"https?://[^\"'\s<>]+/umbraco/surface/CompetitionResultSurface/viewimage\?[^\"'\s<>]+",
    re.IGNORECASE,
)
RELATIVE_VIEW_URL_RE = re.compile(
    r"/umbraco/surface/CompetitionResultSurface/viewimage\?[^\"'\s<>]+",
    re.IGNORECASE,
)


@dataclass
class ResolvedImage:
    source_url: str | None
    strategy: str
    content: bytes | None = None
    fallback_path: str | None = None
    unresolved: bool = False
    content_type: str | None = None


def extract_guid_from_view_url(url: str) -> str | None:
    parsed = urlparse(url)
    values = parse_qs(parsed.query).get("competitionPictureGuid")
    return values[0] if values else None


def build_view_url(base_url: str, guid: str, *, include_judged: bool) -> str:
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


def variant_url(url: str, *, include_judged: bool) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if not query:
        return url
    query["includeWinner"] = ["false"]
    query["includeJudged"] = ["true" if include_judged else "false"]
    query["includeUserEntries"] = ["false"]
    query["includeUserClosest"] = ["false"]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


async def find_competition_picture_guid(page: Page) -> str | None:
    html = await page.content()
    soup = BeautifulSoup(html, "html.parser")

    candidate_values: list[str] = []
    for tag in soup.find_all(True):
        for attr_name, attr_value in tag.attrs.items():
            if attr_name.lower() in {"href", "src", "data-src", "data-href"}:
                if isinstance(attr_value, list):
                    candidate_values.extend(str(item) for item in attr_value)
                else:
                    candidate_values.append(str(attr_value))

    candidate_values.extend(unescape(match.group(0)) for match in VIEW_URL_RE.finditer(html))
    candidate_values.extend(unescape(match.group(0)) for match in RELATIVE_VIEW_URL_RE.finditer(html))

    for value in candidate_values:
        if "viewimage" not in value.lower():
            continue
        guid = extract_guid_from_view_url(unescape(value))
        if guid:
            return guid

    match = GUID_RE.search(html)
    return match.group(1) if match else None


def find_download_or_view_urls(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    for anchor in soup.find_all("a", href=True):
        text = anchor.get_text(" ", strip=True).lower()
        href = str(anchor["href"])
        lowered = href.lower()
        if (
            "download" in text
            or "full size" in text
            or "full-size" in text
            or "download" in lowered
            or "viewimage" in lowered
        ):
            urls.append(urljoin(base_url, href))

    for pattern in (VIEW_URL_RE, RELATIVE_VIEW_URL_RE):
        for match in pattern.finditer(html):
            urls.append(urljoin(base_url, unescape(match.group(0))))

    deduped: list[str] = []
    for url in urls:
        if url not in deduped:
            deduped.append(url)
    return deduped


async def resolve_actual_image_url(page: Page, view_url: str, timeout_ms: int) -> str:
    resolved = await resolve_image_source(
        context=page.context,
        page=page,
        view_url=view_url,
        timeout_ms=timeout_ms,
    )
    if not resolved.source_url:
        raise RuntimeError("could not resolve actual image URL")
    return resolved.source_url


async def resolve_image_source(
    *,
    context: BrowserContext,
    page: Page,
    view_url: str,
    timeout_ms: int,
    debug: DebugCapture | None = None,
    debug_prefix: str = "view_clean",
    candidate_urls: list[str] | None = None,
    include_judged: bool = False,
) -> ResolvedImage:
    urls = _ordered_urls(view_url, candidate_urls or [], include_judged=include_judged)

    for url in urls:
        direct = await _try_direct_response(context, url, timeout_ms, debug, debug_prefix)
        if direct:
            if debug:
                debug.add_image_candidate(debug_prefix, {"url": url, "strategy": direct.strategy})
                debug.save_placeholder_page(debug_prefix, f"Direct image response from {url}")
            return direct

    sniffed: list[ResolvedImage] = []
    tasks: list[asyncio.Task[None]] = []

    def on_response(response: Response) -> None:
        task = asyncio.create_task(_sniff_image_response(response, sniffed))
        tasks.append(task)

    page.on("response", on_response)
    try:
        for navigation_url in urls:
            response = await page.goto(navigation_url, wait_until="domcontentloaded", timeout=timeout_ms)
            if response and response.headers.get("content-type", "").lower().startswith("image/"):
                body = await response.body()
                if debug:
                    debug.add_image_candidate(debug_prefix, {"url": page.url, "strategy": "page_direct_response"})
                    await debug.save_page(page, debug_prefix)
                return ResolvedImage(
                    source_url=page.url,
                    strategy="page_direct_response",
                    content=body,
                    content_type=response.headers.get("content-type"),
                )

            try:
                await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 10_000))
            except Exception:
                pass

            try:
                await page.wait_for_function(
                    "() => Array.from(document.images).some((img) => img.naturalWidth > 0 && img.naturalHeight > 0)",
                    timeout=min(timeout_ms, 5_000),
                )
            except Exception:
                pass

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            if debug:
                await debug.save_page(page, debug_prefix)

            dom = await _dom_image_strategy(page, debug, debug_prefix)
            if dom:
                return dom

            css = await _css_background_strategy(page, debug, debug_prefix)
            if css:
                return css

            if include_judged:
                canvas = await _canvas_render_strategy(page, debug, debug_prefix)
                if canvas:
                    return canvas

            network = _network_sniffing_strategy(sniffed, debug, debug_prefix)
            if network:
                return network

            html = await page.content()
            regex = _html_regex_strategy(html, debug, debug_prefix)
            if regex:
                return regex

        fallback_path = None
        if debug:
            fallback_path = str(await debug.save_fallback_screenshot(page))
        return ResolvedImage(
            source_url=None,
            strategy="screenshot_fallback",
            fallback_path=fallback_path,
            unresolved=True,
        )
    finally:
        try:
            page.remove_listener("response", on_response)
        except Exception:
            pass


def _ordered_urls(view_url: str, candidate_urls: list[str], *, include_judged: bool) -> list[str]:
    urls = [view_url]
    for url in candidate_urls:
        if not url:
            continue
        for candidate in (variant_url(url, include_judged=include_judged), url):
            if candidate not in urls:
                urls.append(candidate)
    return urls


async def _try_direct_response(
    context: BrowserContext,
    url: str,
    timeout_ms: int,
    debug: DebugCapture | None,
    debug_prefix: str,
) -> ResolvedImage | None:
    try:
        response = await context.request.get(url, timeout=timeout_ms, fail_on_status_code=False)
        if debug:
            debug.record_api_response(
                label=debug_prefix,
                url=url,
                status=response.status,
                headers=response.headers,
            )
        content_type = response.headers.get("content-type", "").lower()
        if response.ok and content_type.startswith("image/"):
            return ResolvedImage(
                source_url=url,
                strategy="direct_response",
                content=await response.body(),
                content_type=content_type,
            )
    except Exception:
        return None
    return None


async def _dom_image_strategy(
    page: Page,
    debug: DebugCapture | None,
    debug_prefix: str,
) -> ResolvedImage | None:
    candidates = await page.evaluate(
        """
        () => {
            const srcsetUrls = (srcset) => (srcset || '').split(',')
                .map((part) => part.trim().split(/\\s+/)[0])
                .filter(Boolean);
            return Array.from(document.images).flatMap((img) => {
                const base = [{
                    src: img.currentSrc || img.src,
                    width: img.naturalWidth || 0,
                    height: img.naturalHeight || 0,
                    source: 'img'
                }];
                for (const src of srcsetUrls(img.getAttribute('srcset'))) {
                    base.push({src, width: img.naturalWidth || 0, height: img.naturalHeight || 0, source: 'srcset'});
                }
                return base;
            }).filter((img) => img.src);
        }
        """
    )
    if not candidates:
        return None
    for item in candidates:
        if debug:
            debug.add_image_candidate(debug_prefix, {"strategy": "dom_image", **item})
    largest = max(candidates, key=lambda item: int(item["width"]) * int(item["height"]))
    return ResolvedImage(source_url=str(largest["src"]), strategy="dom_image")


async def _css_background_strategy(
    page: Page,
    debug: DebugCapture | None,
    debug_prefix: str,
) -> ResolvedImage | None:
    urls = await page.evaluate(
        """
        () => {
            const out = [];
            const re = /url\\((['"]?)(.*?)\\1\\)/g;
            for (const el of Array.from(document.querySelectorAll('*'))) {
                const values = [el.getAttribute('style') || '', window.getComputedStyle(el).backgroundImage || ''];
                for (const value of values) {
                    for (const match of value.matchAll(re)) {
                        try { out.push(new URL(match[2], document.baseURI).href); } catch { out.push(match[2]); }
                    }
                }
            }
            return Array.from(new Set(out));
        }
        """
    )
    image_like = [url for url in urls if looks_like_image_url(str(url))]
    for url in image_like:
        if debug:
            debug.add_image_candidate(debug_prefix, {"strategy": "css_background", "url": url})
    return ResolvedImage(source_url=str(image_like[0]), strategy="css_background") if image_like else None


async def _sniff_image_response(response: Response, output: list[ResolvedImage]) -> None:
    try:
        url = response.url
        content_type = response.headers.get("content-type", "").lower()
        if not (content_type.startswith("image/") or looks_like_image_url(url)):
            return
        body = await response.body()
        if not body:
            return
        output.append(
            ResolvedImage(
                source_url=url,
                strategy="network_sniffing",
                content=body,
                content_type=content_type,
            )
        )
    except Exception:
        return


def _network_sniffing_strategy(
    sniffed: list[ResolvedImage],
    debug: DebugCapture | None,
    debug_prefix: str,
) -> ResolvedImage | None:
    if not sniffed:
        return None
    for item in sniffed:
        if debug:
            debug.add_image_candidate(
                debug_prefix,
                {
                    "strategy": "network_sniffing",
                    "url": item.source_url,
                    "bytes": len(item.content or b""),
                    "content_type": item.content_type,
                },
            )
    return max(sniffed, key=lambda item: len(item.content or b""))


async def _canvas_render_strategy(
    page: Page,
    debug: DebugCapture | None,
    debug_prefix: str,
) -> ResolvedImage | None:
    try:
        data = await page.evaluate(
            """
            () => {
                const canvases = [];
                for (const canvas of Array.from(document.querySelectorAll('canvas'))) {
                    if (!canvas.width || !canvas.height) continue;
                    try {
                        canvases.push({
                            width: canvas.width || 0,
                            height: canvas.height || 0,
                            id: canvas.id || '',
                            dataUrl: canvas.toDataURL('image/jpeg', 0.95)
                        });
                    } catch {
                        continue;
                    }
                }
                if (!canvases.length) return null;
                canvases.sort((a, b) => (b.width * b.height) - (a.width * a.height));
                return canvases[0];
            }
            """
        )
    except Exception:
        return None
    if not data or not data.get("dataUrl"):
        return None
    if debug:
        debug.add_image_candidate(
            debug_prefix,
            {
                "strategy": "canvas_render",
                "id": data.get("id"),
                "width": data.get("width"),
                "height": data.get("height"),
            },
        )
    encoded = str(data["dataUrl"]).split(",", 1)[1]
    content = _crop_canvas_padding(base64.b64decode(encoded))
    return ResolvedImage(
        source_url=page.url,
        strategy="canvas_render",
        content=content,
        content_type="image/jpeg",
    )


def _crop_canvas_padding(content: bytes) -> bytes:
    """Remove black page/canvas padding while preserving the image coordinate space."""
    try:
        from PIL import Image

        image = Image.open(BytesIO(content)).convert("RGB")
        full_box = (0, 0, image.width, image.height)
        mask = image.convert("L").point(lambda value: 255 if value > 8 else 0)
        bbox = mask.getbbox()
        if not bbox or bbox == full_box:
            return content

        crop_width = bbox[2] - bbox[0]
        crop_height = bbox[3] - bbox[1]
        if crop_width * crop_height < image.width * image.height * 0.25:
            return content

        output = BytesIO()
        image.crop(bbox).save(output, format="JPEG", quality=95)
        return output.getvalue()
    except Exception:
        return content


def _html_regex_strategy(
    html: str,
    debug: DebugCapture | None,
    debug_prefix: str,
) -> ResolvedImage | None:
    urls = extract_image_urls_from_html(html)
    if not urls:
        return None
    preferred_terms = ("cdn.botb.com", "media", "competition", "winner", "result")
    urls = sorted(
        urls,
        key=lambda url: sum(term in url.lower() for term in preferred_terms),
        reverse=True,
    )
    for url in urls:
        if debug:
            debug.add_image_candidate(debug_prefix, {"strategy": "html_regex", "url": url})
    return ResolvedImage(source_url=urls[0], strategy="html_regex")
