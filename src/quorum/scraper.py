from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image


BOTB_BASE_URL = "https://www.botb.com"


def winner_code(year: int, week: int, competition: str) -> str:
    # BOTB winner URLs use short codes like dc0126 or mw0126.
    competition = competition.lower().strip()
    prefix = "mw" if competition in {"midweek", "mw"} else "dc"
    return f"{prefix}{week:02d}{year:02d}"


def winner_url(year: int, week: int, competition: str) -> str:
    return f"{BOTB_BASE_URL}/winners/{winner_code(year, week, competition)}"


def make_scrape_folder(out_dir: Path, year: int, week: int, competition: str) -> Path:
    folder = out_dir / "scrapes" / competition.lower().strip() / f"{year:02d}" / f"{week:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_page(url: str) -> str:
    # A normal browser user-agent avoids some basic bot blocking.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def extract_judge_coordinates(text: str) -> tuple[int | None, int | None]:
    # This is intentionally simple. Some BOTB pages will need a better scraper later.
    patterns = [
        r"Judges?.{0,120}?X\s*[:=]?\s*(\d+).{0,80}?Y\s*[:=]?\s*(\d+)",
        r"JUDGES?\s+POSITION\s+X\s*:\s*(\d+)\s*Y\s*:\s*(\d+)",
        r"\bX\s*[:=]?\s*(\d+)\s+Y\s*[:=]?\s*(\d+)\b",
    ]
    clean_text = re.sub(r"\s+", " ", text)
    for pattern in patterns:
        match = re.search(pattern, clean_text, flags=re.I | re.S)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None, None


def image_urls_from_html(html: str, page_url: str) -> list[str]:
    # Collect image links from common HTML places: src, data-src, href, and srcset.
    soup = BeautifulSoup(html, "html.parser")
    found: list[str] = []

    for tag in soup.find_all(["img", "source", "a"]):
        for attr in ("src", "data-src", "href"):
            value = tag.get(attr)
            if value:
                found.append(value)
        srcset = tag.get("srcset")
        if srcset:
            for part in srcset.split(","):
                found.append(part.strip().split(" ")[0])

    for match in re.findall(r"https?://[^\"'\s<>]+?\.(?:jpg|jpeg|png|webp)", html, flags=re.I):
        found.append(match)

    urls: list[str] = []
    seen: set[str] = set()
    for value in found:
        url = urljoin(page_url, value.strip())
        if not looks_like_useful_image(url):
            continue
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def looks_like_useful_image(url: str) -> bool:
    # First pass filter: avoid obvious logos/icons/social images by URL.
    lower = url.lower()
    if not lower.startswith(("http://", "https://")):
        return False
    if not re.search(r"\.(jpg|jpeg|png|webp)(\?|$)", lower):
        return False
    blocked_words = ["logo", "icon", "avatar", "facebook", "twitter", "instagram", "youtube"]
    return not any(word in lower for word in blocked_words)


def download_image(url: str, folder: Path, index: int) -> Path:
    suffix = Path(urlparse(url).path).suffix.lower() or ".jpg"
    path = folder / f"image_{index:02d}{suffix}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def useful_saved_image(path: Path) -> bool:
    # Second pass filter: after download, remove tiny images such as logos.
    try:
        image = Image.open(path)
        width, height = image.size
    except Exception:
        return False
    return width >= 300 and height >= 200


def scrape_botb_result(year: int, week: int, competition: str, out_dir: Path) -> dict:
    # Main scraper function used by the GUI. It returns a plain dictionary for simplicity.
    page_url = winner_url(year, week, competition)
    folder = make_scrape_folder(out_dir, year, week, competition)

    html = get_page(page_url)
    (folder / "page.html").write_text(html, encoding="utf-8")

    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text(" ")
    judge_x, judge_y = extract_judge_coordinates(page_text + " " + html)
    image_urls = image_urls_from_html(html, page_url)

    downloaded = []
    for index, image_url in enumerate(image_urls, start=1):
        try:
            image_path = download_image(image_url, folder, index)
            if useful_saved_image(image_path):
                downloaded.append({"url": image_url, "path": str(image_path)})
            else:
                image_path.unlink(missing_ok=True)
                downloaded.append({"url": image_url, "skipped": "too small"})
        except Exception as exc:  # keep scraping even when one image fails
            downloaded.append({"url": image_url, "error": str(exc)})

    result = {
        "url": page_url,
        "year": year,
        "week": week,
        "competition": competition,
        "judge_x": judge_x,
        "judge_y": judge_y,
        "image_count": len([item for item in downloaded if "path" in item]),
        "images": downloaded,
        "folder": str(folder),
    }
    (folder / "metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
