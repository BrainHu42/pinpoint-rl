# A new geolocation benchmark from recent Wikimedia Commons photos (the "plain" design): photos taken after the models' training cut-off, with
# device GPS in their EXIF that agrees with the page coordinate.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.commons_bench scan       (network, CPU; metadata only -> candidates.jsonl)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.commons_bench select     (CPU; a capped, spread, region-balanced pick -> selected.jsonl)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.commons_bench download   (network; 1024 px thumbnails of selected.jsonl)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.commons_bench dedup      (CPU; SigLIP2 embeddings, nearest MP16 / OSV-5M photo)

"""scan:     walks Commons uploads in time windows spread over [--start, --end) (upload time), skips bot accounts and non-JPEGs, and keeps photos whose
          EXIF has GPS and a capture time (GPS date stamp if present, else DateTimeOriginal) on or after --taken-after, a camera model, a long side
          >= 1024 px, and, when the page has a camera coordinate, EXIF GPS within 50 m of it. Mapillary imports are dropped. Appends to candidates.jsonl
          (resumable: finished windows are recorded in scan_windows.txt).
select:   country of each photo (nearest GeoNames place), then a greedy pick taking continents in turn: at most --per-uploader photos per
          uploader, --min-gap-km between any two, no country above --max-country-share of the target.
download: fetches the 1024 px thumbnail of every selected photo (or of --input) into images/<page id>.jpg, skipping files already present.
dedup:    SigLIP2-giant embeddings of the downloaded photos on the CPU and each one's most similar MP16 and OSV-5M gallery photo (cosine, km);
          flags only, in near_duplicates.jsonl."""

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
    emb_path = OUT / "embeddings.f16.npy"
    if emb_path.exists():
        queries = np.load(emb_path).astype(np.float32)
    else:
        torch.set_num_threads(16)
        model, processor = AutoModel.from_pretrained(MODEL, dtype=torch.float32).eval(), AutoProcessor.from_pretrained(MODEL)
        rows = []
        for start in range(0, len(records), batch_size):
            pixels = torch.stack([_pixels(processor, (OUT / "images" / f"{r['page_id']}.jpg").read_bytes()) for r in records[start:start + batch_size]])
            with torch.inference_mode():
                out = model.get_image_features(pixel_values=pixels)
            rows.append((out.pooler_output if hasattr(out, "pooler_output") else out).numpy())
            print(f"embedded {start + len(pixels)}/{len(records)}", flush=True)
        queries = np.concatenate(rows)
        np.save(emb_path, queries.astype(np.float16))
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
    parser.add_argument("node", choices=("scan", "select", "download", "dedup"))
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
    parser.add_argument("--input", type=Path, default=OUT / "selected.jsonl", help="download: records to fetch")
    args = parser.parse_args(argv)
    if args.node == "scan":
        utc = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        end = utc(args.end) if args.end else datetime.now(timezone.utc) - timedelta(hours=1)
        scan(utc(args.start), end, args.windows, args.per_window, utc(args.taken_after), args.workers)
    elif args.node == "select":
        select(args.target, args.per_uploader, args.min_gap_km, args.max_country_share, args.seed)
    elif args.node == "dedup":
        dedup(batch_size=8, chunk=262_144)
    else:
        download(args.input, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
