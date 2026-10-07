# A new geolocation benchmark from recent Wikimedia Commons photos (the "plain" design): photos taken after the models' training cut-off, with
# device GPS in their EXIF that agrees with the page coordinate.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.commons_bench scan       (network, CPU; metadata only -> candidates.jsonl)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.commons_bench download   (network; 1024 px thumbnails of candidates.jsonl)

"""scan:     walks Commons uploads in time windows spread over [--start, --end) (upload time), skips bot accounts and non-JPEGs, and keeps photos whose
          EXIF has GPS and a capture time (GPS date stamp if present, else DateTimeOriginal) on or after --taken-after, a camera model, a long side
          >= 1024 px, and, when the page has a camera coordinate, EXIF GPS within 50 m of it. Mapillary imports are dropped. Appends to candidates.jsonl
          (resumable: finished windows are recorded in scan_windows.txt).
download: fetches the 1024 px thumbnail of every candidate (or of --input) into images/<page id>.jpg, skipping files already present."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

OUT = Path("/data/pinpoint/commons26")
API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "commons-recent-gps-photos/0.6 (research script; contact: kinghorton42@gmail.com)"
MAX_GPS_GAP_M = 50.0
MIN_LONG_SIDE = 1024
THUMB_WIDTH = 1024
EXCLUDED_SOURCES = ("mapillary", "osmplus")


def _request(params: dict[str, Any], *, post: bool = False, retries: int = 6) -> dict[str, Any]:
    params = {**params, "format": "json", "maxlag": 5}
    data = urllib.parse.urlencode(params).encode()
    for attempt in range(retries):
        try:
            if post:
                request = urllib.request.Request(API, data=data, headers={"User-Agent": USER_AGENT})
            else:
                request = urllib.request.Request(f"{API}?{data.decode()}", headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            if payload.get("error", {}).get("code") == "maxlag":
                time.sleep(5 * (attempt + 1))
                continue
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt == retries - 1:
                raise
            retry_after = getattr(error, "headers", None) and error.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else 2 ** attempt)
    raise RuntimeError("Commons API kept reporting maxlag")


def _uploads(start: datetime, count: int) -> list[dict[str, Any]]:
    """The `count` uploads just before `start`, newest first."""
    out: list[dict[str, Any]] = []
    cont: str | None = None
    while len(out) < count:
        params = {"action": "query", "list": "allimages", "aisort": "timestamp", "aidir": "older", "ailimit": 500,
                  "aistart": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "aiprop": "timestamp|user|mime"}
        if cont:
            params["aicontinue"] = cont
        payload = _request(params)
        out += payload.get("query", {}).get("allimages", [])
        cont = payload.get("continue", {}).get("aicontinue")
        if not cont:
            break
    return out[:count]


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


def _taken(metadata: dict[str, Any]) -> datetime | None:
    """Capture time: the satellite GPS date stamp when present (camera clocks drift), else EXIF DateTimeOriginal."""
    for key, pattern in (("GPSDateStamp", r"(\d{4}):(\d{2}):(\d{2})"), ("DateTimeOriginal", r"(\d{4}):(\d{2}):(\d{2})")):
        match = re.match(pattern, str(metadata.get(key) or ""))
        if match:
            try:
                return datetime(*map(int, match.groups()), tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def _text(extmetadata: dict[str, Any], key: str) -> str:
    value = (extmetadata.get(key) or {}).get("value", "")
    return re.sub(r"<[^>]+>", "", str(value)).strip()


def _record(page: dict[str, Any], taken_after: datetime) -> dict[str, Any] | None:
    info = (page.get("imageinfo") or [None])[0]
    if not info or info.get("mime") != "image/jpeg" or max(info.get("width", 0), info.get("height", 0)) < MIN_LONG_SIDE:
        return None
    metadata = {m["name"]: m["value"] for m in info.get("metadata") or [] if "name" in m}
    try:
        lat, lon = float(metadata["GPSLatitude"]), float(metadata["GPSLongitude"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (abs(lat) < 1e-6 and abs(lon) < 1e-6) or not metadata.get("Model"):
        return None
    taken = _taken(metadata)
    if taken is None or taken < taken_after:
        return None
    camera = [c for c in page.get("coordinates") or [] if c.get("type") == "camera" and c.get("globe", "earth") == "earth"]
    gap = _haversine_m(lat, lon, float(camera[0]["lat"]), float(camera[0]["lon"])) if camera else None
    if gap is not None and gap > MAX_GPS_GAP_M:
        return None
    ext = info.get("extmetadata") or {}
    artist, credit = _text(ext, "Artist"), _text(ext, "Credit")
    if any(s in f"{page['title']} {artist} {credit}".lower() for s in EXCLUDED_SOURCES):
        return None
    return {
        "page_id": page["pageid"], "title": page["title"], "uploader": info.get("user", ""), "artist": artist, "credit": credit,
        "lat": lat, "lon": lon, "page_gps_gap_m": gap, "taken": taken.strftime("%Y-%m-%d"), "uploaded": info.get("timestamp", ""),
        "make": str(metadata.get("Make") or ""), "model": str(metadata["Model"]), "width": info["width"], "height": info["height"],
        "thumb_url": info.get("thumburl", ""), "page_url": info.get("descriptionurl", ""),
        "license": _text(ext, "LicenseShortName"), "license_url": _text(ext, "LicenseUrl"),
    }


def _details(titles: Sequence[str], taken_after: datetime) -> list[dict[str, Any]]:
    payload = _request({"action": "query", "prop": "imageinfo|coordinates", "titles": "|".join(titles), "iiprop": "timestamp|user|url|mime|size|metadata|extmetadata",
                        "iiurlwidth": THUMB_WIDTH, "colimit": "max", "coprop": "type|globe", "coprimary": "all"}, post=True)
    return [r for page in payload.get("query", {}).get("pages", {}).values() if "pageid" in page and (r := _record(page, taken_after))]


def _chunks(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def scan(start: datetime, end: datetime, windows: int, per_window: int, taken_after: datetime, workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    done_path, out_path = OUT / "scan_windows.txt", OUT / "candidates.jsonl"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    seen = {json.loads(line)["page_id"] for line in out_path.open()} if out_path.exists() else set()
    step = (end - start) / windows
    with ThreadPoolExecutor(workers) as pool:
        for w in range(windows):
            window_end = end - w * step
            key = window_end.strftime("%Y-%m-%dT%H:%M:%SZ")
            if key in done:
                continue
            uploads = _uploads(window_end, per_window)
            titles = [u["title"] for u in uploads if u.get("mime") == "image/jpeg" and not re.search(r"bot\b|bot$", u.get("user", ""), re.I)]
            records = [r for batch in pool.map(lambda t: _details(t, taken_after), list(_chunks(titles, 50))) for r in batch]
            records = [r for r in records if r["page_id"] not in seen]
            seen.update(r["page_id"] for r in records)
            with out_path.open("a", encoding="utf-8") as f:
                f.writelines(json.dumps(r) + "\n" for r in records)
            with done_path.open("a") as f:
                f.write(key + "\n")
            print(f"window {w + 1}/{windows} {key}: {len(uploads)} uploads, {len(titles)} non-bot JPEGs, {len(records)} kept (total {len(seen)})", flush=True)


def _fetch(record: dict[str, Any], folder: Path) -> str:
    path = folder / f"{record['page_id']}.jpg"
    if path.exists():
        return "skip"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(record["thumb_url"], headers={"User-Agent": USER_AGENT}), timeout=60) as response:
                data = response.read()
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(path)
            return "ok"
        except (urllib.error.URLError, TimeoutError) as error:
            retry_after = getattr(error, "headers", None) and error.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else 2 ** (attempt + 1))
    return "fail"


def download(input_path: Path, workers: int) -> None:
    folder = OUT / "images"
    folder.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in input_path.open()]
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda r: _fetch(r, folder), records))
    print({k: results.count(k) for k in ("ok", "skip", "fail")})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("node", choices=("scan", "download"))
    parser.add_argument("--start", default="2026-07-01", help="scan: oldest upload time")
    parser.add_argument("--end", default=None, help="scan: newest upload time (default now)")
    parser.add_argument("--taken-after", default="2026-07-01", help="scan: earliest capture date")
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--per-window", type=int, default=1000, help="scan: uploads read per window")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--input", type=Path, default=OUT / "candidates.jsonl", help="download: records to fetch")
    args = parser.parse_args(argv)
    if args.node == "scan":
        utc = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        end = utc(args.end) if args.end else datetime.now(timezone.utc) - timedelta(hours=1)
        scan(utc(args.start), end, args.windows, args.per_window, utc(args.taken_after), args.workers)
    else:
        download(args.input, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
