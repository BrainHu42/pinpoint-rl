# Live tool: photos near a point, for verifying a candidate place (TOOL_USE_DESIGN.md; lesson S3: the offline gallery lacks references where
# text-proposed places need them). Source: Wikimedia Commons geosearch (keyless; https://www.mediawiki.org/wiki/API:Geosearch), files within
# a radius in order of distance, with uploader / upload time, and their thumbnails. Leak guards for benchmark queries (commons26 photos are
# Commons uploads): only files uploaded before the query was taken (a copy of the query can't exist yet) and none by the query's uploader
# (`exclude_group`: commons26's hashed uploader id). Every API response and thumbnail is cached under artifacts/tool_use/commons/.
# Second source: Mapillary street-level images (free read token MAPILLARY_TOKEN in .env), same interface (MapillaryImagery).
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.tools.imagery near 48.1905 14.8285 [--source mapillary] [--km 2] [--before 2026-08-13]

"""near: the files a geosearch returns around a point (after the guards): distance, upload date, uploader, title; --fetch also downloads
       their thumbnails (THUMB_PX wide) and prints the local paths."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import threading
import time
from typing import Sequence
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "pinpoint-rl research (brainhu42@gmail.com)"  # Wikimedia API etiquette: identify the client with a contact
CACHE = Path("artifacts/tool_use/commons")
THUMB_PX = 500  # a standard Commons thumbnail step; MegaLoc reads 322 px
MIMES = ("image/jpeg", "image/png", "image/webp")
MIN_INTERVAL = 0.25  # seconds between requests (shared across threads): polite serial-ish rate
# Local search (when the API's search backend is too busy): the geo_tags table dump (https://dumps.wikimedia.org/commonswiki/), indexed by
# `index`; geosearch then only looks up file info by page id.
GEO_DUMP = Path("artifacts/tool_use/dumps/geo_tags_20261003/commonswiki-latest-geo_tags.sql.gz")
GEO_INDEX = CACHE / "geo_index_20261003.npz"


def _thumb_url(info: dict) -> str:
    """THUMB_PX-wide thumbnail of a Commons file from its file URL (.../commons/a/ab/Name.jpg -> .../commons/thumb/a/ab/Name.jpg/500px-Name.jpg);
    the file itself when it is no wider (Commons doesn't upscale) or WebP (its thumbnails change extension)."""

    url = info["url"].split("?", 1)[0]
    if info.get("width", THUMB_PX + 1) <= THUMB_PX or info.get("mime") == "image/webp":
        return url
    head, name = url.rsplit("/", 1)
    return f"{head.replace('/wikipedia/commons/', '/wikipedia/commons/thumb/', 1)}/{name}/{THUMB_PX}px-{name}"


@dataclass(frozen=True)
class Photo:
    page_id: int
    title: str
    lat: float
    lon: float
    dist_m: float
    uploader: str
    uploaded: str  # ISO timestamp of the current version
    thumb_url: str

    @property
    def group(self) -> str:
        """commons26's hashed uploader id (commons_bench release)."""

        return hashlib.sha256(f"commons26:{self.uploader}".encode()).hexdigest()[:12]


class CommonsImagery:
    def __init__(self, cache: Path = CACHE) -> None:
        self.cache = cache
        (cache / "geo").mkdir(parents=True, exist_ok=True)
        (cache / "thumbs").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last = 0.0

    def _get(self, url: str, retries: int = 4, paced: bool = True) -> bytes:
        """paced: API calls keep MIN_INTERVAL between them; thumbnail downloads (CDN, not the API) run unpaced in the caller's threads."""

        for attempt in range(retries):
            if paced:
                with self._lock:
                    wait = self._last + MIN_INTERVAL - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                    self._last = time.monotonic()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=30) as r:
                    return r.read()
            except Exception as e:  # noqa: BLE001  (429 / 5xx / timeouts: back off and retry)
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** (attempt + 1) * (5 if "429" in str(e) else 1))
        raise RuntimeError("unreachable")

    def geosearch(self, lat: float, lon: float, km: float = 10.0, limit: int = 100) -> list[Photo]:
        """Image files within km (<= 10, the API's cap) of a point, nearest first, with uploader, upload time and a thumbnail URL; cached."""

        key = f"{lat:.4f}_{lon:.4f}_{km:g}_{limit}"
        path = self.cache / "geo" / f"{key}.json"
        local = self.cache / "local" / f"{key}.json"  # answered by the local index (same format)
        if path.exists() or local.exists():
            data = json.loads((path if path.exists() else local).read_text(encoding="utf-8"))
        elif GEO_INDEX.exists():
            data = self._local(lat, lon, km, limit)
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(json.dumps(data), encoding="utf-8")
        else:
            params = {"action": "query", "format": "json", "generator": "geosearch", "ggscoord": f"{lat}|{lon}",
                      "ggsradius": int(min(km, 10.0) * 1000), "ggsnamespace": 6, "ggslimit": limit, "prop": "imageinfo|coordinates",
                      "iiprop": "user|timestamp|url|mime", "iiurlwidth": THUMB_PX, "colimit": "max", "maxlag": 5}
            if limit > 50:  # the API renders thumbnail URLs for only 50 files per request: build them from the file URL instead (_thumb_url)
                params.pop("iiurlwidth")
                params["iiprop"] += "|size"
            for attempt in range(12):  # API-level errors (search too busy, maxlag) come back as HTTP 200 and are intermittent: short
                data = json.loads(self._get(f"{API}?{urllib.parse.urlencode(params)}"))  # jittered waits, never cache them
                if "error" not in data:
                    break
                time.sleep(min(30.0, 2.0 * (attempt + 1)) * (0.5 + random.random()))
            else:
                raise RuntimeError(f"Commons geosearch failed at ({lat}, {lon}): {data['error'].get('code')}")
            path.write_text(json.dumps(data), encoding="utf-8")
        out = []
        for page in (data.get("query") or {}).get("pages", {}).values():
            info, coords = (page.get("imageinfo") or [{}])[0], (page.get("coordinates") or [{}])[0]
            if "thumburl" not in info and info.get("url"):
                info["thumburl"] = _thumb_url(info)
            if info.get("mime") not in MIMES or "lat" not in coords or not info.get("thumburl"):
                continue
            la, lo = float(coords["lat"]), float(coords["lon"])  # generator mode returns no distance: compute it
            h = math.sin(math.radians(la - lat) / 2) ** 2 + math.cos(math.radians(lat)) * math.cos(math.radians(la)) * math.sin(math.radians(lo - lon) / 2) ** 2
            out.append(Photo(int(page["pageid"]), page["title"], la, lo, 2 * 6371000.0 * math.asin(math.sqrt(min(1.0, h))),
                             info.get("user", ""), info.get("timestamp", ""), info["thumburl"]))
        return sorted(out, key=lambda p: p.dist_m)

    def _local(self, lat: float, lon: float, km: float, limit: int) -> dict:
        """geosearch from the local index (build_index): the `limit` nearest geotagged pages within km, then their file info by page id,
        50 per API call (a database lookup; the search backend, which throttles, is not involved). Non-file pages come back without imageinfo."""

        import numpy as np

        if not hasattr(self, "_index"):
            with self._lock:
                if not hasattr(self, "_index"):
                    z = np.load(GEO_INDEX)
                    self._index = (z["lat"], z["lon"], z["page"])
        la, lo, page = self._index
        dlat = km / 111.0
        s, e = np.searchsorted(la, lat - dlat), np.searchsorted(la, lat + dlat, side="right")  # sorted by latitude
        bl, bo = la[s:e].astype(np.float64), lo[s:e].astype(np.float64)
        h = np.sin(np.radians(bl - lat) / 2) ** 2 + math.cos(math.radians(lat)) * np.cos(np.radians(bl)) * np.sin(np.radians(bo - lon) / 2) ** 2
        dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        inside = np.flatnonzero(dist <= km)
        pick = inside[np.argsort(dist[inside], kind="stable")[:limit]]
        coords = {int(page[s + i]): (float(bl[i]), float(bo[i])) for i in pick}
        pages = {}
        ids = list(coords)
        for b in range(0, len(ids), 50):
            params = {"action": "query", "format": "json", "pageids": "|".join(map(str, ids[b: b + 50])), "prop": "imageinfo",
                      "iiprop": "user|timestamp|url|mime|size", "maxlag": 5}
            for attempt in range(8):
                got = json.loads(self._get(f"{API}?{urllib.parse.urlencode(params)}"))
                if "error" not in got:
                    break
                time.sleep(min(30.0, 2.0 * (attempt + 1)) * (0.5 + random.random()))
            else:
                raise RuntimeError(f"Commons page lookup failed at ({lat}, {lon}): {got['error'].get('code')}")
            for pid, pg in (got.get("query") or {}).get("pages", {}).items():
                if int(pid) in coords:
                    pg["coordinates"] = [{"lat": coords[int(pid)][0], "lon": coords[int(pid)][1]}]
                    pages[pid] = pg
        return {"query": {"pages": pages}, "source": "local geo_tags index"}

    def near(self, lat: float, lon: float, km: float = 10.0, n: int = 20, before: str | None = None, exclude_group: str | None = None,
             exclude_titles: Sequence[str] = ()) -> list[Photo]:
        """The n nearest guarded files: uploaded before `before` (ISO date; the query's capture date), not by the query's uploader."""

        cut = datetime.fromisoformat(before).replace(tzinfo=timezone.utc) if before else None
        keep = []
        for p in self.geosearch(lat, lon, km, 100 if n <= 50 else 500):
            if exclude_group and p.group == exclude_group or p.title in exclude_titles:
                continue
            if cut and (not p.uploaded or datetime.fromisoformat(p.uploaded.replace("Z", "+00:00")) >= cut):
                continue
            keep.append(p)
        return keep[:n]

    def thumb(self, p: Photo) -> Path | None:
        """The cached thumbnail of a file (downloaded once); None if it can't be fetched."""

        path = self.cache / "thumbs" / f"{p.page_id}.jpg"
        if not path.exists():
            try:
                data = self._get(p.thumb_url, paced=False)
            except Exception:  # noqa: BLE001
                return None
            path.write_bytes(data)
        return path


MAPILLARY_API = "https://graph.mapillary.com/images"
MAPILLARY_TILES = "https://tiles.mapillary.com/maps/vtp/mly1_public/2"  # coverage vector tiles (Mapbox Vector Tile format)
MAPILLARY_CACHE = Path("artifacts/tool_use/mapillary")


def _mapillary_token() -> str:
    """MAPILLARY_TOKEN (a read-only client token, 'MLY|...') from the environment or the repo's git-ignored .env; never logged or cached."""

    import os

    if os.environ.get("MAPILLARY_TOKEN"):
        return os.environ["MAPILLARY_TOKEN"]
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("MAPILLARY_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("no MAPILLARY_TOKEN in the environment or .env")


class MapillaryImagery(CommonsImagery):
    """Street-level photos near a point from Mapillary (free read token), same interface as CommonsImagery. Its bbox search returns HTTP 500
    in dense areas, so points come from the public coverage vector tiles (zoom 14; layer 'image': id, captured_at, sequence_id, is_pano),
    each tile decoded once and cached as a compact point list; a thumbnail URL is looked up per image only when it is fetched. Nearest first
    within km, panoramas skipped (MegaLoc reads perspective photos), at most PER_SEQUENCE frames per capture sequence (one drive would
    otherwise fill every slot), date guard on capture time. The uploader guard does not apply (Mapillary users are not Commons users)."""

    PER_SEQUENCE = 2
    ZOOM = 14

    def __init__(self, cache: Path = MAPILLARY_CACHE) -> None:
        super().__init__(cache)
        (cache / "tiles").mkdir(parents=True, exist_ok=True)
        self._token = _mapillary_token()

    def _tile(self, x: int, y: int) -> dict:
        """One coverage tile's image points (lat, lon, id, captured_at ms, sequence, pano); cached."""

        import numpy as np

        path = self.cache / "tiles" / f"{self.ZOOM}_{x}_{y}.npz"
        if not path.exists():
            import mapbox_vector_tile

            url = f"{MAPILLARY_TILES}/{self.ZOOM}/{x}/{y}?access_token={self._token}"
            try:
                raw = self._get(url)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"Mapillary tile {self.ZOOM}/{x}/{y} failed: {str(e).replace(self._token, '<token>')}") from None
            feats = mapbox_vector_tile.decode(raw, default_options={"y_coord_down": True}).get("image", {}).get("features", [])
            n, ext = 2 ** self.ZOOM, 4096.0
            px = np.asarray([f["geometry"]["coordinates"] for f in feats], np.float64).reshape(-1, 2)
            lon = (x + px[:, 0] / ext) / n * 360.0 - 180.0
            lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + px[:, 1] / ext) / n))))
            props = [f["properties"] for f in feats]
            tmp = path.with_suffix(f".{threading.get_ident()}.tmp.npz")  # atomic: threads may fetch the same tile
            np.savez(tmp, lat=lat, lon=lon, id=np.asarray([int(p["id"]) for p in props], np.int64),
                     captured=np.asarray([int(p.get("captured_at") or 0) for p in props], np.int64),
                     seq=np.asarray([str(p.get("sequence_id", "")) for p in props], dtype=str),
                     pano=np.asarray([bool(p.get("is_pano")) for p in props], dtype=bool))
            tmp.replace(path)
        z = np.load(path)
        types = {"lat": np.float64, "lon": np.float64, "id": np.int64, "captured": np.int64, "seq": str, "pano": bool}
        return {k: z[k].astype(types[k]) for k in z.files}  # empty tiles (no coverage) were saved with default dtypes

    def geosearch(self, lat: float, lon: float, km: float = 1.0, limit: int = 0) -> list[Photo]:
        import numpy as np

        n = 2 ** self.ZOOM
        dlat = km / 111.0
        dlon = dlat / max(math.cos(math.radians(lat)), 0.05)
        tx = lambda lo: int((lo + 180.0) / 360.0 * n)
        ty = lambda la: int((1 - math.asinh(math.tan(math.radians(la))) / math.pi) / 2 * n)
        parts = [self._tile(x, y) for x in range(tx(lon - dlon), tx(lon + dlon) + 1) for y in range(ty(lat + dlat), ty(lat - dlat) + 1)]
        pts = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else {}
        if not parts or not len(pts["id"]):
            return []
        la, lo = np.radians(pts["lat"]), np.radians(pts["lon"])
        h = np.sin((la - math.radians(lat)) / 2) ** 2 + math.cos(math.radians(lat)) * np.cos(la) * np.sin((lo - math.radians(lon)) / 2) ** 2
        dist = 2 * 6371000.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        seen: dict[str, int] = {}
        out = []
        for i in np.flatnonzero((dist <= km * 1000) & ~pts["pano"])[np.argsort(dist[(dist <= km * 1000) & ~pts["pano"]], kind="stable")]:
            seq = str(pts["seq"][i])
            if seen.get(seq, 0) >= self.PER_SEQUENCE:
                continue
            seen[seq] = seen.get(seq, 0) + 1
            when = datetime.fromtimestamp(int(pts["captured"][i]) / 1000, tz=timezone.utc).isoformat() if pts["captured"][i] else ""
            out.append(Photo(int(pts["id"][i]), f"mapillary:{int(pts['id'][i])}:{seq}", float(pts["lat"][i]), float(pts["lon"][i]),
                             float(dist[i]), "", when, ""))
            if limit and len(out) >= limit:
                break
        return out

    def near(self, lat: float, lon: float, km: float = 1.0, n: int = 20, before: str | None = None, exclude_group: str | None = None,
             exclude_titles: Sequence[str] = ()) -> list[Photo]:
        """As CommonsImagery.near; thumbnail URLs (signed, short-lived) of the not yet cached files come from one batched API call."""

        photos = super().near(lat, lon, km, n, before, None, exclude_titles)
        todo = [p for p in photos if not (self.cache / "thumbs" / f"{p.page_id}.jpg").exists()]
        urls = {}
        for s in range(0, len(todo), 50):
            try:
                ids = ",".join(str(p.page_id) for p in todo[s: s + 50])
                rows = json.loads(self._get(f"{MAPILLARY_API}?access_token={self._token}&image_ids={ids}&fields=id,thumb_1024_url"))["data"]
                urls.update({int(r["id"]): r.get("thumb_1024_url", "") for r in rows})
            except Exception:  # noqa: BLE001  (thumb() falls back to one lookup per image)
                pass
        return [replace(p, thumb_url=urls.get(p.page_id, "")) for p in photos]

    def thumb(self, p: Photo) -> Path | None:
        path = self.cache / "thumbs" / f"{p.page_id}.jpg"
        if not path.exists():
            try:
                url = p.thumb_url or json.loads(self._get(f"{MAPILLARY_API.rsplit('/', 1)[0]}/{p.page_id}?access_token={self._token}"
                                                          "&fields=thumb_1024_url"))["thumb_1024_url"]
                path.write_bytes(self._get(url, paced=False))
            except Exception:  # noqa: BLE001
                return None
        return path


SOURCES = {"commons": CommonsImagery, "mapillary": MapillaryImagery}


def build_index(dump: Path = GEO_DUMP) -> None:
    """Primary Earth coordinates of every geotagged Commons page from the geo_tags SQL dump (one row per line), sorted by latitude ->
    GEO_INDEX (float32 lat / lon, uint32 page id). The page ids include non-file pages; their lookups return no file info."""

    import array
    import gzip
    import re

    import numpy as np

    row = re.compile(rb"^\((\d+),(\d+),'earth',1,(-?[\d.]+),(-?[\d.]+),")
    page, la, lo = array.array("I"), array.array("f"), array.array("f")
    with gzip.open(dump, "rb") as f:
        for line in f:
            m = row.match(line)
            if m:
                page.append(int(m[2]))
                la.append(float(m[3]))
                lo.append(float(m[4]))
    la_, lo_, page_ = np.frombuffer(la, np.float32), np.frombuffer(lo, np.float32), np.frombuffer(page, np.uint32)
    ok = (np.abs(la_) <= 90) & (np.abs(lo_) <= 180) & ~((la_ == 0) & (lo_ == 0))
    order = np.argsort(la_[ok], kind="stable")
    np.savez(GEO_INDEX, lat=la_[ok][order], lon=lo_[ok][order], page=page_[ok][order])
    print(f"{ok.sum():,} primary coordinates ({len(page_):,} rows) -> {GEO_INDEX}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("near", "index"))
    parser.add_argument("lat", type=float, nargs="?")
    parser.add_argument("lon", type=float, nargs="?")
    parser.add_argument("--source", choices=tuple(SOURCES), default="commons")
    parser.add_argument("--km", type=float, default=None, help="radius (default: 10 for commons, 1 for mapillary)")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--before", default=None)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "index":
        build_index()
        return 0
    tool = SOURCES[args.source]()
    km = args.km if args.km is not None else (10.0 if args.source == "commons" else 1.0)
    for p in tool.near(args.lat, args.lon, km, args.n, args.before):
        line = f"{p.dist_m:7.0f} m  {p.uploaded[:10]}  {p.uploader[:20]:20s}  {p.title}"
        if args.fetch:
            line += f"  -> {tool.thumb(p)}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
