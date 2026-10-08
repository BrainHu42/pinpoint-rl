# A new geolocation benchmark from recent Wikimedia Commons photos (the "plain" design): photos taken after the models' training cut-off, with
# device GPS in their EXIF that agrees with the page coordinate.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.commons_bench scan       (network, CPU; metadata only -> candidates.jsonl)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.commons_bench search-scan (network; every geotagged upload, day by day -> candidates.jsonl)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.commons_bench select     (CPU; a capped, spread, region-balanced pick -> selected.jsonl)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.commons_bench download   (network; 1024 px thumbnails of selected.jsonl)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.commons_bench tiers --workers 8   (GPU via scripts/serve27b.sh; locatability tiers)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.commons_bench release    (CPU; split by uploader, strip metadata -> release/)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.commons_bench score --predictions P.csv [--split dev|test|all]
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.commons_bench dedup      (CPU; SigLIP2 embeddings, nearest MP16 / OSV-5M photo)

"""scan:     walks Commons uploads in time windows spread over [--start, --end) (upload time), skips bot accounts and non-JPEGs, and keeps photos whose
          EXIF has GPS and a capture time (GPS date stamp if present, else DateTimeOriginal) on or after --taken-after, a camera model, a long side
          >= 1024 px, and, when the page has a camera coordinate, EXIF GPS within 50 m of it. Mapillary imports are dropped. Appends to candidates.jsonl
          (resumable: finished windows are recorded in scan_windows.txt).
search-scan: the same filters over every JPEG with a camera location (P1259) created on each day in [--start, --end), via search
          (resumable: finished days in search_days.txt). Shares candidates.jsonl with scan (deduplicated by page id).
select:   country of each photo (nearest GeoNames place), then a greedy pick taking continents in turn: at most --per-uploader photos per
          uploader, --min-gap-km between any two, no country above --max-country-share of the target.
download: fetches the 1024 px thumbnail of every selected photo (or of --input) into images/<page id>.jpg, skipping files already present.
dedup:    SigLIP2-giant embeddings of the downloaded photos on the CPU and each one's most similar MP16 and OSV-5M gallery photo (cosine, km);
          flags only, in near_duplicates.jsonl.
tiers:    locatability tier (landmark / city / region / none) and flags (place text, GPS overlay, not a photo) from Qwen3.6-27B, which is
          not shown the location -> tiers.jsonl.
release:  drops non-photos and photos with printed coordinates, splits by uploader (~1,000 dev photos, rest test), strips JPEG metadata
          (EXIF / XMP / IPTC) without re-encoding, writes release/benchmark.csv (no titles or captions), attribution.csv and the SigLIP2 cache.
score:    headline over locatable photos (tier != none), by tier and continent, the mean over continents, GeoScore and median km, with
          standard errors clustered by uploader."""

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
GEONAMES = Path("/data/pinpoint/geonames/allCountries.txt")
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
    return {
        "page_id": page["pageid"], "title": page["title"], "uploader": info.get("user", ""),
        "lat": lat, "lon": lon, "page_gps_gap_m": gap, "taken": taken.strftime("%Y-%m-%d"), "uploaded": info.get("timestamp", ""),
        "make": str(metadata.get("Make") or ""), "model": str(metadata["Model"]), "width": info["width"], "height": info["height"],
    }


def _details(titles: Sequence[str], taken_after: datetime) -> list[dict[str, Any]]:
    """EXIF filter first (cheap), then licence, author and thumbnail for the survivors only: extmetadata makes a request ~10x slower."""
    payload = _request({"action": "query", "prop": "imageinfo|coordinates", "titles": "|".join(titles), "iiprop": "timestamp|user|mime|size|metadata",
                        "colimit": "max", "coprop": "type|globe", "coprimary": "all"}, post=True)
    kept = {r["title"]: r for page in payload.get("query", {}).get("pages", {}).values() if "pageid" in page and (r := _record(page, taken_after))}
    if not kept:
        return []
    payload = _request({"action": "query", "prop": "imageinfo", "titles": "|".join(kept), "iiprop": "url|extmetadata", "iiurlwidth": THUMB_WIDTH}, post=True)
    out = []
    for page in payload.get("query", {}).get("pages", {}).values():
        record, info = kept.get(page.get("title", "")), (page.get("imageinfo") or [None])[0]
        if record is None or info is None:
            continue
        ext = info.get("extmetadata") or {}
        artist, credit = _text(ext, "Artist"), _text(ext, "Credit")
        if any(s in f"{record['title']} {artist} {credit}".lower() for s in EXCLUDED_SOURCES):
            continue
        out.append({**record, "artist": artist, "credit": credit, "thumb_url": info.get("thumburl", ""), "page_url": info.get("descriptionurl", ""),
                    "license": _text(ext, "LicenseShortName"), "license_url": _text(ext, "LicenseUrl")})
    return out


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


def _search_titles(day: str) -> list[str]:
    """Every JPEG created on `day` with a camera location (P1259). Search stops at 10,000 hits per query, so the day is read oldest-first and
    newest-first (days have ~10-11k hits)."""
    titles: dict[str, None] = {}
    for sort in ("create_timestamp_asc", "create_timestamp_desc"):
        offset = 0
        while offset < 10_000:
            payload = _request({"action": "query", "list": "search", "srnamespace": 6, "srlimit": 500, "sroffset": offset, "srprop": "", "srsort": sort,
                                "srinfo": "totalhits", "srsearch": f"haswbstatement:P1259 filemime:image/jpeg creationdate:{day}"})
            hits = payload.get("query", {}).get("search", [])
            titles.update((h["title"], None) for h in hits)
            total = payload.get("query", {}).get("searchinfo", {}).get("totalhits", 0)
            offset = payload.get("continue", {}).get("sroffset", 10_000)
            if not hits or len(titles) >= total:
                break
        if len(titles) >= total:
            break
    return list(titles)


def search_scan(start: datetime, end: datetime, taken_after: datetime, workers: int) -> None:
    """All geotagged JPEGs created in [start, end), day by day, through the same filters as scan."""
    OUT.mkdir(parents=True, exist_ok=True)
    done_path, out_path = OUT / "search_days.txt", OUT / "candidates.jsonl"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    seen = {json.loads(line)["page_id"] for line in out_path.open()} if out_path.exists() else set()
    day = start
    with ThreadPoolExecutor(workers) as pool:
        while day < end:
            key = day.strftime("%Y-%m-%d")
            day += timedelta(days=1)
            if key in done:
                continue
            titles = _search_titles(key)
            records = [r for batch in pool.map(lambda t: _details(t, taken_after), list(_chunks(titles, 50))) for r in batch]
            records = [r for r in records if r["page_id"] not in seen and not re.search(r"bot\b|bot$", r["uploader"], re.I)]
            seen.update(r["page_id"] for r in records)
            with out_path.open("a", encoding="utf-8") as f:
                f.writelines(json.dumps(r) + "\n" for r in records)
            with done_path.open("a") as f:
                f.write(key + "\n")
            print(f"day {key}: {len(titles)} geotagged JPEGs, {len(records)} kept (total {len(seen)})", flush=True)


def _places() -> tuple[Any, Any]:
    """GeoNames populated places (>= 1,000 people) as unit vectors and their country codes, cached in OUT/places.tsv."""
    import numpy as np

    path = OUT / "places.tsv"
    if not path.exists():
        with GEONAMES.open(encoding="utf-8") as src, path.open("w", encoding="utf-8") as dst:
            for line in src:
                f = line.split("\t")
                if f[6] == "P" and f[14].isdigit() and int(f[14]) >= 1000:
                    dst.write(f"{f[4]}\t{f[5]}\t{f[8]}\n")
    rows = [line.rstrip("\n").split("\t") for line in path.open(encoding="utf-8")]
    return _unit(np.array([[float(r[0]), float(r[1])] for r in rows])), np.array([r[2] for r in rows])


def _unit(latlon: Any) -> Any:
    import numpy as np

    lat, lon = np.radians(latlon[:, 0]), np.radians(latlon[:, 1])
    return np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]


def select(target: int, per_uploader: int, min_gap_km: float, max_country_share: float, seed: int) -> None:
    """Greedy pick over continents in turn: at most `per_uploader` photos per uploader, `min_gap_km` between any two, no country above the share."""
    import numpy as np
    from scipy.spatial import cKDTree

    records = [json.loads(line) for line in (OUT / "candidates.jsonl").open()]
    latlon = np.array([[r["lat"], r["lon"]] for r in records])
    vectors, codes = _places()
    countries = codes[cKDTree(vectors).query(_unit(latlon))[1]]
    continent = {f[0]: f[8] for line in GEONAMES.with_name("countryInfo.txt").open(encoding="utf-8")
                 if not line.startswith("#") and len(f := line.rstrip("\n").split("\t")) > 8}
    by_region: dict[str, list[int]] = {}
    for i in np.random.default_rng(seed).permutation(len(records)):
        by_region.setdefault(continent.get(countries[i], "??"), []).append(int(i))
    picked: list[int] = []
    uploads: dict[str, int] = {}
    per_country: dict[str, int] = {}
    tree_points: list[Any] = []
    gap = min_gap_km / 6371.0
    while len(picked) < target and any(by_region.values()):
        for region in sorted(by_region):
            queue = by_region[region]
            while queue:
                i = queue.pop()
                r, c = records[i], countries[i]
                if uploads.get(r["uploader"], 0) >= per_uploader or per_country.get(c, 0) >= max(1, max_country_share * target):
                    continue
                v = _unit(latlon[i:i + 1])[0]
                if tree_points and np.min(np.linalg.norm(np.array(tree_points) - v, axis=1)) < gap:
                    continue
                picked.append(i)
                tree_points.append(v)
                uploads[r["uploader"]] = uploads.get(r["uploader"], 0) + 1
                per_country[c] = per_country.get(c, 0) + 1
                break
            if len(picked) >= target:
                break
    with (OUT / "selected.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps({**records[i], "country": str(countries[i]), "continent": continent.get(countries[i], "??")}) + "\n" for i in picked)
    regions = {k: sum(continent.get(countries[i], "??") == k for i in picked) for k in sorted(by_region)}
    top = sorted(per_country.items(), key=lambda kv: -kv[1])[:10]
    print(f"{len(records)} candidates from {len({r['uploader'] for r in records})} uploaders -> {len(picked)} selected from {len(uploads)} uploaders, "
          f"{len(per_country)} countries\n by continent {regions}\n top countries {top}")


def dedup(batch_size: int, chunk: int) -> None:
    """SigLIP2-giant embeddings of the downloaded photos on the CPU (same model and preprocessing as embed_cache), then each photo's most similar
    MP16 and OSV-5M gallery photo (cosine) and its distance in km. Writes embeddings.f16.npy and near_duplicates.jsonl; drops nothing."""
    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor

    from .embed_cache import MODEL, _pixels
    from .strategy_search import MP16_EMBED, OSV_EMBED

    records = [json.loads(line) for line in (OUT / "selected.jsonl").open()]
    emb_path, ids_path = OUT / "embeddings.f16.npy", OUT / "embedding_ids.txt"
    cached: dict[int, Any] = {}
    if emb_path.exists() and ids_path.exists():
        cached = dict(zip(map(int, ids_path.read_text().split()), np.load(emb_path)))
    missing = [r for r in records if r["page_id"] not in cached]
    if missing:
        torch.set_num_threads(16)
        model, processor = AutoModel.from_pretrained(MODEL, dtype=torch.float32).eval(), AutoProcessor.from_pretrained(MODEL)
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            pixels = torch.stack([_pixels(processor, (OUT / "images" / f"{r['page_id']}.jpg").read_bytes()) for r in batch])
            with torch.inference_mode():
                out = model.get_image_features(pixel_values=pixels)
            for r, e in zip(batch, (out.pooler_output if hasattr(out, "pooler_output") else out).numpy()):
                cached[r["page_id"]] = e.astype(np.float16)
            if (start // batch_size) % 50 == 0 or start + batch_size >= len(missing):
                print(f"embedded {start + len(batch)}/{len(missing)}", flush=True)
        np.save(emb_path, np.stack(list(cached.values())))
        ids_path.write_text("\n".join(map(str, cached)) + "\n")
    queries = np.stack([cached[r["page_id"]] for r in records]).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    latlon = np.array([[r["lat"], r["lon"]] for r in records])
    best: dict[str, tuple[Any, Any]] = {}
    for name, root in (("mp16", MP16_EMBED), ("osv5m", OSV_EMBED)):
        manifest = json.loads((root / "manifest.json").read_text())
        gallery = np.memmap(root / manifest["files"]["embeddings"], dtype=np.float16, mode="r").reshape(-1, manifest["embedding_dim"])
        sim, idx = np.full(len(queries), -1.0, dtype=np.float32), np.zeros(len(queries), dtype=np.int64)
        for start in range(0, len(gallery), chunk):
            block = np.asarray(gallery[start:start + chunk], dtype=np.float32)
            block /= np.linalg.norm(block, axis=1, keepdims=True)
            s = queries @ block.T
            j = s.argmax(1)
            better = s[np.arange(len(queries)), j] > sim
            sim[better], idx[better] = s[np.arange(len(queries)), j][better], start + j[better]
        where = np.memmap(root / manifest["files"]["latlon_deg"], dtype=np.float32, mode="r").reshape(-1, 2)[idx]
        km = [_haversine_m(a, b, float(c), float(d)) / 1000 for (a, b), (c, d) in zip(latlon, where)]
        best[name] = (sim, km)
        print(f"{name}: max cosine >= 0.90 {int((sim >= 0.90).sum())}, >= 0.95 {int((sim >= 0.95).sum())} of {len(sim)}; median max cosine {np.median(sim):.3f}", flush=True)
    with (OUT / "near_duplicates.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(records):
            f.write(json.dumps({"page_id": r["page_id"], **{f"{n}_cos": round(float(best[n][0][i]), 4) for n in best},
                                **{f"{n}_km": round(best[n][1][i], 2) for n in best}}) + "\n")


TIER_PROMPT = """Rate how precisely this photo alone lets an expert locate it.
- "landmark": a famous or uniquely identifiable place (named monument, well-known building or view), or visible text naming the exact site or street. Ordinary churches, houses, shops, chain-store signs, or just a city name are NOT landmark.
- "city": details that pin the city or town (its name in text, its transit vehicles, a known skyline), but not the exact site.
- "region": only country- or region-level cues (language, architecture style, vegetation, road markings, terrain).
- "none": little or no geographic information (indoor scene, close-up, food, animal, plant, people, sky, generic nature).
Answer only JSON: {"tier": "landmark|city|region|none", "place_text": true if visible text names the place, "gps_overlay": true if coordinates are printed on the image, "not_photo": true if it is a map, screenshot, scan, document or artwork}"""
# v2 (stricter landmark, broader none). Against one blind reader on 198 photos: exact tier 113/198, none vs locatable 175/198 (v1: 99, 159).


TIER_SCHEMA = {"type": "json_schema", "json_schema": {"name": "tier", "schema": {
    "type": "object", "required": ["tier", "place_text", "gps_overlay", "not_photo"], "additionalProperties": False,
    "properties": {"tier": {"enum": ["landmark", "city", "region", "none"]}, "place_text": {"type": "boolean"},
                   "gps_overlay": {"type": "boolean"}, "not_photo": {"type": "boolean"}}}}}  # the 27B otherwise writes an analysis first


def _tier(record: dict[str, Any], url: str) -> dict[str, Any]:
    import base64

    image = base64.b64encode((OUT / "images" / f"{record['page_id']}.jpg").read_bytes()).decode()
    body = {"model": "vlm", "temperature": 0, "max_tokens": 120, "response_format": TIER_SCHEMA, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}, {"type": "text", "text": TIER_PROMPT}]}]}
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=300) as response:
                text = json.load(response)["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", text, re.S)
            answer = json.loads(match.group(0)) if match else {}
            if answer.get("tier") in ("landmark", "city", "region", "none"):
                return {"page_id": record["page_id"], **{k: answer.get(k) for k in ("tier", "place_text", "gps_overlay", "not_photo")}}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
            time.sleep(2 ** attempt)
    return {"page_id": record["page_id"], "tier": None}


def tiers(url: str, workers: int, limit: int) -> None:
    """Locatability tier and image flags for every selected photo from a VLM that does not see the location (llama.cpp server, scripts/serve27b.sh).
    Resumable: appends to tiers.jsonl and skips photos already labelled."""
    out_path = OUT / "tiers.jsonl"
    done = {json.loads(line)["page_id"] for line in out_path.open()} if out_path.exists() else set()
    records = [r for r in map(json.loads, (OUT / "selected.jsonl").open()) if r["page_id"] not in done][:limit or None]
    start = time.time()
    with ThreadPoolExecutor(workers) as pool, out_path.open("a", encoding="utf-8") as f:
        for n, result in enumerate(pool.map(lambda r: _tier(r, url), records), 1):
            f.write(json.dumps(result) + "\n")
            if n % 250 == 0 or n == len(records):
                f.flush()
                print(f"labelled {n}/{len(records)} ({n / (time.time() - start):.2f} photos/s)", flush=True)


RELEASE = OUT / "release"
DEV_SHARE = 1000 / 5668  # ~1,000 public dev photos; the rest is the held-out test split


def _strip_metadata(data: bytes) -> bytes:
    """Drop JPEG APP1-APP15 (EXIF, XMP, IPTC: GPS, titles, captions) and comment segments without re-encoding; keeps APP0 (JFIF) and the
    ICC profile in APP2."""
    assert data[:2] == b"\xff\xd8", "not a JPEG"
    out, i = bytearray(b"\xff\xd8"), 2
    while i < len(data):
        if data[i] != 0xFF:
            raise ValueError("bad JPEG marker")
        marker = data[i + 1]
        if marker == 0xDA:  # start of scan: the rest is image data
            out += data[i:]
            break
        length = int.from_bytes(data[i + 2:i + 4], "big")
        segment = data[i:i + 2 + length]
        is_icc = marker == 0xE2 and segment[4:16] == b"ICC_PROFILE\x00"
        if not ((0xE1 <= marker <= 0xEF and not is_icc) or marker == 0xFE):
            out += segment
        i += 2 + length
    return bytes(out)


def release() -> None:
    """The benchmark files: photos labelled by the VLM (non-photos and photos with printed coordinates dropped), split by uploader into dev / test,
    metadata stripped, plus attribution and the SigLIP2 embedding cache `data/benchmarks.py` reads (benchmark `commons26`)."""
    import csv
    import hashlib

    import numpy as np

    from ..data.benchmarks import EMBEDDING_KEY

    records = [json.loads(line) for line in (OUT / "selected.jsonl").open()]
    tier = {t["page_id"]: t for t in map(json.loads, (OUT / "tiers.jsonl").open()) if t.get("tier")}
    if (OUT / "tiers_v1.jsonl").exists():  # a photo flagged as not a photo / with printed coordinates by either prompt is dropped
        for t in map(json.loads, (OUT / "tiers_v1.jsonl").open()):
            if t["page_id"] in tier:
                for flag in ("not_photo", "gps_overlay"):
                    tier[t["page_id"]][flag] = bool(tier[t["page_id"]].get(flag)) or bool(t.get(flag))
    near = {n["page_id"]: n for n in map(json.loads, (OUT / "near_duplicates.jsonl").open())}
    cached = dict(zip(map(int, (OUT / "embedding_ids.txt").read_text().split()), np.load(OUT / "embeddings.f16.npy")))
    dropped = {"no tier": 0, "not a photo": 0, "printed coordinates": 0}
    rows, attribution, embeddings = [], [], []
    (RELEASE / "images").mkdir(parents=True, exist_ok=True)
    for r in records:
        t = tier.get(r["page_id"])
        if t is None:
            dropped["no tier"] += 1
            continue
        if t.get("not_photo") or t.get("gps_overlay"):
            dropped["not a photo" if t.get("not_photo") else "printed coordinates"] += 1
            continue
        group = hashlib.sha256(f"commons26:{r['uploader']}".encode()).hexdigest()
        image_id = f"{r['page_id']}.jpg"
        (RELEASE / "images" / image_id).write_bytes(_strip_metadata((OUT / "images" / image_id).read_bytes()))
        n = near[r["page_id"]]
        rows.append({"IMG_ID": image_id, "LAT": r["lat"], "LON": r["lon"], "split": "dev" if int(group[:8], 16) / 16**8 < DEV_SHARE else "test",
                     "tier": t["tier"], "place_text": int(bool(t.get("place_text"))), "group": group[:12], "country": r["country"],
                     "continent": r["continent"], "taken": r["taken"], "mp16_cos": n["mp16_cos"], "osv5m_cos": n["osv5m_cos"]})
        attribution.append({"IMG_ID": image_id, "artist": r["artist"], "credit": r["credit"], "license": r["license"],
                            "license_url": r["license_url"], "source": r["page_url"]})
        embeddings.append(cached[r["page_id"]])
    for name, table in (("benchmark.csv", rows), ("attribution.csv", attribution)):
        with (RELEASE / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    cache = RELEASE / "image_embeddings" / EMBEDDING_KEY
    cache.mkdir(parents=True, exist_ok=True)
    np.asarray(embeddings, dtype=np.float16).tofile(cache / "embeddings.f16.bin")
    (cache / "image_ids.txt").write_text("\n".join(r["IMG_ID"] for r in rows) + "\n")
    (cache / "manifest.json").write_text(json.dumps({"model_name": "google/siglip2-giant-opt-patch16-384", "device": "cpu (fp32, stored fp16)",
                                                     "embedding_dim": len(embeddings[0]), "num_samples": len(rows),
                                                     "files": {"embeddings": "embeddings.f16.bin", "image_ids": "image_ids.txt"}}, indent=2))
    count = lambda key, value: sum(row[key] == value for row in rows)
    print(f"{len(rows)} photos (dropped {dropped}); dev {count('split', 'dev')}, test {count('split', 'test')}; "
          f"tiers { {k: count('tier', k) for k in ('landmark', 'city', 'region', 'none')} }; groups {len({row['group'] for row in rows})}")


def score(predictions: Path, split: str) -> None:
    """Metrics for a predictions CSV (IMG_ID, LAT, LON): headline over locatable photos (tier != none), by tier, by continent and their mean, with
    standard errors clustered by uploader group."""
    import csv

    import numpy as np

    from ..data.benchmarks import DISTANCE_THRESHOLDS_KM, GEOGUESSR_DECAY_KM, geodesic_km

    rows = [r for r in csv.DictReader((RELEASE / "benchmark.csv").open(encoding="utf-8")) if split == "all" or r["split"] == split]
    pred = {r["IMG_ID"]: (float(r["LAT"]), float(r["LON"])) for r in csv.DictReader(predictions.open(encoding="utf-8"))}
    missing = sum(r["IMG_ID"] not in pred for r in rows)
    if missing:
        print(f"warning: {missing} photos have no prediction; scored as (0, 0)")
    km = geodesic_km(np.array([pred.get(r["IMG_ID"], (0.0, 0.0)) for r in rows]), np.array([[float(r["LAT"]), float(r["LON"])] for r in rows]))
    groups = np.array([r["group"] for r in rows])

    def line(mask: Any) -> str:
        d, g = km[mask], groups[mask]
        cells = []
        for t in DISTANCE_THRESHOLDS_KM:
            y = (d < t).astype(float)
            sums = {}
            for yi, gi in zip(y - y.mean(), g):
                sums[gi] = sums.get(gi, 0.0) + yi
            se = np.sqrt(sum(v * v for v in sums.values())) / len(y)
            cells.append(f"{100 * y.mean():5.1f}±{100 * se:3.1f}")
        geo = np.mean(np.round(5000 * np.exp(-d / GEOGUESSR_DECAY_KM)))
        return f"{' '.join(cells)}  {geo:6.0f}  {np.median(d):8.1f}  n={mask.sum()}"

    tier = np.array([r["tier"] for r in rows])
    continent = np.array([r["continent"] for r in rows])
    print(f"{'':16s} {'  '.join(f'<{t} km'.rjust(8) for t in DISTANCE_THRESHOLDS_KM)}  GeoScore  median km")
    print(f"{'headline':16s} {line(tier != 'none')}")
    print(f"{'all photos':16s} {line(np.ones(len(rows), bool))}")
    for t in ("landmark", "city", "region", "none"):
        print(f"{'tier ' + t:16s} {line(tier == t)}")
    locatable = tier != "none"
    for c in sorted(set(continent)):
        print(f"{'continent ' + c:16s} {line(locatable & (continent == c))}")
    means = [np.mean([(km[locatable & (continent == c)] < t).mean() for c in sorted(set(continent))]) for t in DISTANCE_THRESHOLDS_KM]
    print(f"{'continent mean':16s} {' '.join(f'{100 * m:5.1f}    ' for m in means)}")


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
    parser.add_argument("node", choices=("scan", "search-scan", "select", "download", "dedup", "tiers", "release", "score"))
    parser.add_argument("--start", default="2026-07-01", help="scan: oldest upload time")
    parser.add_argument("--end", default=None, help="scan: newest upload time (default now)")
    parser.add_argument("--taken-after", default="2026-07-01", help="scan: earliest capture date")
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--per-window", type=int, default=1000, help="scan: uploads read per window")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--target", type=int, default=600, help="select: photos to pick")
    parser.add_argument("--per-uploader", type=int, default=2)
    parser.add_argument("--min-gap-km", type=float, default=2.0)
    parser.add_argument("--max-country-share", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--url", default="http://127.0.0.1:8766/v1/chat/completions", help="tiers: OpenAI-compatible VLM endpoint")
    parser.add_argument("--limit", type=int, default=0, help="tiers: label at most N new photos (0 = all)")
    parser.add_argument("--predictions", type=Path, help="score: CSV with IMG_ID, LAT, LON")
    parser.add_argument("--split", default="test", choices=("dev", "test", "all"), help="score: which split")
    parser.add_argument("--input", type=Path, default=OUT / "selected.jsonl", help="download: records to fetch")
    args = parser.parse_args(argv)
    if args.node == "scan":
        utc = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        end = utc(args.end) if args.end else datetime.now(timezone.utc) - timedelta(hours=1)
        scan(utc(args.start), end, args.windows, args.per_window, utc(args.taken_after), args.workers)
    elif args.node == "search-scan":
        utc = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        end = utc(args.end) if args.end else datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        search_scan(utc(args.start), end, utc(args.taken_after), args.workers)
    elif args.node == "select":
        select(args.target, args.per_uploader, args.min_gap_km, args.max_country_share, args.seed)
    elif args.node == "release":
        release()
    elif args.node == "score":
        score(args.predictions, args.split)
    elif args.node == "tiers":
        tiers(args.url, args.workers, args.limit)
    elif args.node == "dedup":
        dedup(batch_size=8, chunk=262_144)
    else:
        download(args.input, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
