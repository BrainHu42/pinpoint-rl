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
from dataclasses import dataclass
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

    def _get(self, url: str, retries: int = 4) -> bytes:
        for attempt in range(retries):
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
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            params = {"action": "query", "format": "json", "generator": "geosearch", "ggscoord": f"{lat}|{lon}",
                      "ggsradius": int(min(km, 10.0) * 1000), "ggsnamespace": 6, "ggslimit": limit, "prop": "imageinfo|coordinates",
                      "iiprop": "user|timestamp|url|mime", "iiurlwidth": THUMB_PX, "colimit": "max", "maxlag": 5}
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
            if info.get("mime") not in MIMES or "lat" not in coords or not info.get("thumburl"):
                continue
            la, lo = float(coords["lat"]), float(coords["lon"])  # generator mode returns no distance: compute it
            h = math.sin(math.radians(la - lat) / 2) ** 2 + math.cos(math.radians(lat)) * math.cos(math.radians(la)) * math.sin(math.radians(lo - lon) / 2) ** 2
            out.append(Photo(int(page["pageid"]), page["title"], la, lo, 2 * 6371000.0 * math.asin(math.sqrt(min(1.0, h))),
                             info.get("user", ""), info.get("timestamp", ""), info["thumburl"]))
        return sorted(out, key=lambda p: p.dist_m)

    def near(self, lat: float, lon: float, km: float = 10.0, n: int = 20, before: str | None = None, exclude_group: str | None = None,
             exclude_titles: Sequence[str] = ()) -> list[Photo]:
        """The n nearest guarded files: uploaded before `before` (ISO date; the query's capture date), not by the query's uploader."""

        cut = datetime.fromisoformat(before).replace(tzinfo=timezone.utc) if before else None
        keep = []
        for p in self.geosearch(lat, lon, km):
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
                data = self._get(p.thumb_url)
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
            np.savez(path, lat=lat, lon=lon, id=np.asarray([int(p["id"]) for p in props], np.int64),
                     captured=np.asarray([int(p.get("captured_at") or 0) for p in props], np.int64),
                     seq=np.asarray([str(p.get("sequence_id", "")) for p in props]), pano=np.asarray([bool(p.get("is_pano")) for p in props]))
        z = np.load(path)
        return {k: z[k] for k in z.files}

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
        return super().near(lat, lon, km, n, before, None, exclude_titles)

    def thumb(self, p: Photo) -> Path | None:
        path = self.cache / "thumbs" / f"{p.page_id}.jpg"
        if not path.exists():
            try:
                info = json.loads(self._get(f"{MAPILLARY_API.rsplit('/', 1)[0]}/{p.page_id}?access_token={self._token}&fields=thumb_1024_url"))
                path.write_bytes(self._get(info["thumb_1024_url"]))
            except Exception:  # noqa: BLE001
                return None
        return path


SOURCES = {"commons": CommonsImagery, "mapillary": MapillaryImagery}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("near",))
    parser.add_argument("lat", type=float)
    parser.add_argument("lon", type=float)
    parser.add_argument("--source", choices=tuple(SOURCES), default="commons")
    parser.add_argument("--km", type=float, default=None, help="radius (default: 10 for commons, 1 for mapillary)")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--before", default=None)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args(argv)
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
