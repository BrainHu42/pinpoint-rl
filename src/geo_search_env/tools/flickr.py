# Live tool: Flickr photo search (flickr.photos.search), the live counterpart of the YFCC titles + tags source of experiment/text_search.py
# (YFCC stops in 2014; Flickr has every public geotagged photo to date). Rate limit: 3,600 calls per hour per key (FLICKR_KEY in .env), so
# every response goes into a permanent SQLite cache on the EBS volume and RL / evaluation replays the cache (offline=True never calls Flickr).
# Calls are canonical (normalised text, a fixed parameter set, boxes rounded to 0.01 degrees) so the same action always hits the same entry.
# A cross-process limiter (call log in the cache database) keeps MIN_INTERVAL between calls and at most PER_HOUR in any hour, for all our
# processes together. Leak guards are applied at replay time, not fetch time, so one cached response serves every protocol: the query photo
# itself, its photographer (owner nsid), and, when the query's capture date is known (commons26), every photo uploaded on or after it (a live
# query can only find older uploads; this also drops Flickr copies of the query and the photographer's same-trip uploads).
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.tools.flickr query "Villa d'Este" [--near LAT LON KM] [--offline]
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.tools.flickr usage
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.tools.flickr harvest --set dev3k_setft_own [--max-calls N]
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.tools.flickr report --set dev3k_setft_own   (cache only, no calls)

"""query:   one search, printed as the agent would see it (clusters as in text_search).
usage:   calls in the last hour / day / in total, cached responses.
harvest: every 4B query of a photo set (artifacts/agent/queries_<set>_4b.json), searched worldwide once (deduplicated across photos).
report:  live Flickr vs the offline YFCC replica on the same queries and guards: photos with any match / the tool's own pick within 25 km of
         the truth, overall and where the pipeline is wrong -> artifacts/tool_use/flickr/report_<set>.txt."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Sequence
import urllib.parse
import urllib.request
import zlib

API = "https://api.flickr.com/services/rest/"
CACHE = Path("artifacts/tool_use/flickr")
MIN_INTERVAL = 2.0  # seconds between calls, all processes together (1.1 s with 4 workers drew HTTP 429 after ~600 calls)
WORKERS = 2  # concurrent calls while harvesting (each takes ~5 s)
PER_HOUR = 1800  # half of Flickr's nominal 3,600 per key: its throttle fires below the nominal rate
THROTTLE_PAUSE = 300.0  # seconds all processes wait after a 429, doubling on repeats
PER_PAGE = 250  # the most a geo search returns per call (per_page=500 still gives 250)
EXTRAS = "geo,date_upload,date_taken,owner_name,path_alias,tags,url_m"
YFCC_END = 1420070400  # 2015-01-01 UTC: YFCC100M holds uploads to 2014
MIN_ACCURACY = 11  # as text_search: below city level a photo can't place a 25 km cluster
PHOTO_SETS, AGENT = Path("artifacts/photo_sets"), Path("artifacts/agent")
FLICKR_URL = re.compile(r"flickr\.com/photos/([^/]+)/(\d+)")


def _key() -> str:
    """FLICKR_KEY from the environment or the repo's git-ignored .env; never logged or cached."""

    if os.environ.get("FLICKR_KEY"):
        return os.environ["FLICKR_KEY"]
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("FLICKR_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError("no FLICKR_KEY in the environment or .env")


def norm_text(text: str) -> str:
    return " ".join(text.lower().split())


class CacheMiss(LookupError):
    """An offline FlickrSearch was asked for a call that was never made live."""


@dataclass(frozen=True)
class Photo:
    photo_id: int
    owner: str  # nsid, as MP16's AUTHOR
    alias: str  # path alias (flickr.com/photos/<alias>/), '' if none
    lat: float
    lon: float
    accuracy: int  # 1 world .. 16 street
    uploaded: int  # unix time
    taken: str
    title: str
    tags: str
    url: str  # url_m, '' if none


@dataclass
class Guard:
    """What the environment hides from the agent for one query photo."""

    photo_ids: set[int] = field(default_factory=set)
    owners: set[str] = field(default_factory=set)  # nsids or path aliases
    before: str | None = None  # ISO date: only photos uploaded before it

    def keep(self, p: Photo) -> bool:
        if p.photo_id in self.photo_ids or p.owner in self.owners or (p.alias and p.alias in self.owners):
            return False
        if self.before:
            cut = datetime.fromisoformat(self.before).replace(tzinfo=timezone.utc).timestamp()
            if not p.uploaded or p.uploaded >= cut:
                return False
        return True


@dataclass
class Result:
    query: str
    total: int  # Flickr's count of geotagged matches (before guards; the API returns at most PER_PAGE of them per call)
    photos: list[Photo]  # returned and kept by the guard


class FlickrSearch:
    def __init__(self, cache: Path = CACHE, offline: bool = False) -> None:
        cache.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache / "cache.db", timeout=60, isolation_level=None, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, method TEXT, fetched REAL, body BLOB)")
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (t REAL, method TEXT, ok INTEGER)")
        self.db.execute("CREATE TABLE IF NOT EXISTS pause (until REAL)")
        self.offline = offline
        self.live_calls = 0  # made by this instance

    # ---- calls

    def _wait_turn(self) -> None:
        """Block until a call is allowed under MIN_INTERVAL and PER_HOUR (all processes share the call log), then log the call."""

        while True:
            self.db.execute("BEGIN IMMEDIATE")
            now = time.time()
            last = self.db.execute("SELECT MAX(t) FROM calls").fetchone()[0] or 0.0
            n, first = self.db.execute("SELECT COUNT(*), MIN(t) FROM calls WHERE t > ?", (now - 3600,)).fetchone()
            paused = self.db.execute("SELECT MAX(until) FROM pause").fetchone()[0] or 0.0
            wait = max(last + MIN_INTERVAL - now, (first + 3600 - now) if n >= PER_HOUR else 0.0, paused - now)
            if wait <= 0:
                self.db.execute("INSERT INTO calls VALUES (?, '', 1)", (now,))
                self.db.execute("COMMIT")
                return
            self.db.execute("ROLLBACK")
            time.sleep(min(wait, 60.0))

    def call(self, method: str, **params) -> dict:
        """One API call, from the cache if it was ever made (failures are not cached)."""

        key = json.dumps([method, sorted(params.items())], ensure_ascii=False)
        row = self.db.execute("SELECT body FROM responses WHERE key = ?", (key,)).fetchone()
        if row:
            return json.loads(zlib.decompress(row[0]))
        if self.offline:
            raise CacheMiss(key)
        query = urllib.parse.urlencode({"method": method, "api_key": _key(), "format": "json", "nojsoncallback": 1, **params})
        for attempt in range(5):
            self._wait_turn()
            self.live_calls += 1
            try:
                with urllib.request.urlopen(f"{API}?{query}", timeout=60) as r:
                    data = json.load(r)
                if data.get("stat") == "ok":
                    break
                err = f"{data.get('code')}: {data.get('message')}"
                if data.get("code") in (100, 112, 116):  # bad key, unknown method, bad URL: retrying won't help
                    raise RuntimeError(f"Flickr {method}: {err}")
            except (OSError, ValueError) as e:  # timeouts, 5xx, 429, broken JSON
                err = str(e)
                if "429" in err:  # throttled: every process waits (4 overlapping calls at ~50 / min drew 429s after ~600)
                    self.db.execute("INSERT INTO pause VALUES (?)", (time.time() + THROTTLE_PAUSE * 2 ** attempt,))
            time.sleep(10 * 2 ** attempt)
        else:
            raise RuntimeError(f"Flickr {method} failed after retries: {err}")
        self.db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?)",
                        (key, method, time.time(), zlib.compress(json.dumps(data, ensure_ascii=False).encode(), 6)))
        return data

    def cached(self, method: str, **params) -> bool:
        key = json.dumps([method, sorted(params.items())], ensure_ascii=False)
        return self.db.execute("SELECT 1 FROM responses WHERE key = ?", (key,)).fetchone() is not None

    @staticmethod
    def search_params(text: str, near: tuple[float, float, float] | None = None) -> dict:
        """Canonical flickr.photos.search parameters: geotagged photos matching the text (title, description, tags), most relevant first;
        near (lat, lon, km) becomes a box rounded outward to 0.01 degrees."""

        params = {"text": norm_text(text), "has_geo": 1, "media": "photos", "sort": "relevance", "per_page": PER_PAGE, "extras": EXTRAS}
        if near:
            la, lo, km = near
            dlat, dlon = km / 111.0, km / (111.0 * max(math.cos(math.radians(la)), 0.05))
            params["bbox"] = ",".join(f"{v:.2f}" for v in (math.floor((lo - dlon) * 100) / 100, math.floor((la - dlat) * 100) / 100,
                                                           math.ceil((lo + dlon) * 100) / 100, math.ceil((la + dlat) * 100) / 100))
        return params

    def search(self, text: str, near: tuple[float, float, float] | None = None, guard: Guard | None = None) -> Result:
        """Geotagged Flickr photos matching the text (one call, up to PER_PAGE photos), filtered by the guard (set by the environment)."""

        if not norm_text(text):
            return Result(text, 0, [])
        data = self.call("flickr.photos.search", **self.search_params(text, near))
        photos = []
        for p in data["photos"]["photo"]:
            try:
                photo = Photo(int(p["id"]), p.get("owner", ""), p.get("pathalias") or "", float(p["latitude"]), float(p["longitude"]),
                              int(p.get("accuracy") or 0), int(p.get("dateupload") or 0), p.get("datetaken", ""), p.get("title", ""),
                              p.get("tags", ""), p.get("url_m", ""))
            except (KeyError, ValueError):
                continue
            if guard is None or guard.keep(photo):
                photos.append(photo)
        return Result(text, int(data["photos"].get("total") or 0), photos)

    def matches(self, text: str, near: tuple[float, float, float] | None = None, guard: Guard | None = None,
                min_accuracy: int = MIN_ACCURACY) -> list[tuple]:
        """Rows as text_search's YFCC source: (photo id, owner, lat, lon, accuracy), at least city-level accuracy."""

        return [(p.photo_id, p.owner, p.lat, p.lon, p.accuracy) for p in self.search(text, near, guard).photos if p.accuracy >= min_accuracy]

    def usage(self) -> dict[str, int]:
        now = time.time()
        out = {f"calls, last {name}": self.db.execute("SELECT COUNT(*) FROM calls WHERE t > ?", (now - s,)).fetchone()[0]
               for name, s in (("hour", 3600), ("day", 86400))}
        out["calls, total"] = self.db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        out["cached responses"] = self.db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        return out


# ------------------------------------------------------------------------------------------------------------------- photo sets


def _queries(tag: str) -> dict[str, list[str]]:
    return {k: list(dict.fromkeys(q for q in v if norm_text(q))) for k, v in
            json.loads((AGENT / f"queries_{tag}_4b.json").read_text(encoding="utf-8")).items()}


def guards(tag: str, allow_test: bool = False) -> dict[str, Guard]:
    """Each photo's guard: MP16 photos (id in the file name) hide themselves and their photographer; commons26 photos hide uploads from their
    capture date on and, for files imported from Flickr, the Flickr original and its owner."""

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    out = {e["image_id"]: Guard() for e in photos}
    if photos and photos[0].get("benchmark") == "commons26":
        release = Path("data/pinpoint/commons26/release")
        with (release / "benchmark.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["IMG_ID"] in out:
                    out[row["IMG_ID"]].before = row["taken"] or None
        with (release / "attribution.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                m = FLICKR_URL.search(row["credit"] + " " + row["source"])
                if m and row["IMG_ID"] in out:
                    out[row["IMG_ID"]].owners.add(m.group(1))
                    out[row["IMG_ID"]].photo_ids.add(int(m.group(2)))
    elif photos and "benchmark" not in photos[0]:  # MP16: <a>_<b>_<flickr id>.jpg
        from ..experiment.yfcc_names import _mp16_authors

        authors = _mp16_authors(set(out))
        for image_id, g in out.items():
            g.photo_ids.add(int(Path(image_id).stem.rsplit("_", 1)[-1]))
            if authors.get(image_id):
                g.owners.add(authors[image_id])
    elif not allow_test:
        raise ValueError(f"{tag} holds test photos (im2gps3k / yfcc4k): only when the user asks (--test)")
    else:  # im2gps3k: <flickr id>_<secret>_<server>_<owner>.jpg; yfcc4k: <flickr id>.jpg, owner from the YFCC metadata
        from ..experiment.text_search import _db_dir

        lookup = {}
        for image_id, g in out.items():
            parts = Path(image_id).stem.split("_")
            g.photo_ids.add(int(parts[0]))
            if len(parts) == 4:
                g.owners.add(parts[3])
            else:
                lookup[int(parts[0])] = g
        if lookup:  # meta.photo_id has no index: one scan for all of them
            yfcc = sqlite3.connect(f"file:{_db_dir() / 'yfcc.db'}?mode=ro", uri=True)
            for pid, uid in yfcc.execute("SELECT photo_id, uid FROM meta"):
                if pid in lookup:
                    lookup[pid].owners.add(uid)
    return out


def harvest(tag: str, max_calls: int | None = None, workers: int = WORKERS, allow_test: bool = False) -> None:
    """Calls take ~5 s (relevance-sorted geo search), so `workers` threads overlap them; the shared limiter still paces the calls."""

    import threading
    from concurrent.futures import ThreadPoolExecutor

    guards(tag, allow_test)  # refuses test photo sets unless asked
    tool = FlickrSearch()
    todo = list(dict.fromkeys(norm_text(q) for qs in _queries(tag).values() for q in qs))
    missing = [q for q in todo if not tool.cached("flickr.photos.search", **tool.search_params(q))][:max_calls]
    print(f"{tag}: {len(todo)} distinct queries, {len(missing)} to fetch; {tool.usage()}", flush=True)
    start, local, done = time.time(), threading.local(), [0]

    def fetch(q: str) -> None:
        if not hasattr(local, "tool"):
            local.tool = FlickrSearch()  # one connection per thread
        try:
            local.tool.search(q)
        except RuntimeError as e:
            print(f"  skipped {q!r}: {e}", flush=True)
        done[0] += 1
        if done[0] % 100 == 0:
            print(f"  {done[0]}/{len(missing)}  {(time.time() - start) / 60:.0f} min  {tool.usage()}", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(fetch, missing))
    print(f"done: {(time.time() - start) / 60:.0f} min; {tool.usage()}", flush=True)


def _pipeline_km(tag: str) -> "np.ndarray | None":  # noqa: F821
    import numpy as np

    path, key = {"commons26dev_setft_own": (AGENT / "action_agent_apply_km_setft_own_1km0.5_commons26dev.npz", "commons26dev:setft_own|RL policy, lambda 0"),
                 "dev3k_setft_own": (AGENT / "action_agent_apply_km_setft_own_1km0.5.npz", "dev3k|RL policy, lambda 0")}.get(tag, (None, None))
    if path is None or not path.exists():
        return None
    with np.load(path) as z:
        return z[key] if key in z.files else None


def report(tag: str, allow_test: bool = False) -> None:
    """Cache only (offline): photos whose queries have a match / the tool's own pick within 25 km, live Flickr vs the YFCC replica."""

    import numpy as np

    from ..experiment.text_search import Result as TSResult, TextSearch, _pick, haversine_km

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    queries, guard = _queries(tag), guards(tag, allow_test)
    live, replica = FlickrSearch(offline=True), TextSearch()
    pipe = _pipeline_km(tag)
    rows, misses = [], 0
    for e in photos:
        truth = e["truth"]
        g, qs = guard[e["image_id"]], queries.get(e["image_id"], [])
        mp16 = "benchmark" not in e
        exclude = (next(iter(g.photo_ids)), next(iter(g.owners), "")) if mp16 else None  # YFCC is <= 2014: no leak for 2026 photos
        stats, extra = {}, [False, False]  # Flickr: any match < 25 km uploaded after YFCC (2015 on); any before the guard
        for name in ("flickr", "yfcc"):
            results, n_any, best = [], False, None
            for q in qs:
                if name == "flickr":
                    try:
                        kept = [p for p in live.search(q, guard=g).photos if p.accuracy >= MIN_ACCURACY]
                        everything = [p for p in live.search(q).photos if p.accuracy >= MIN_ACCURACY]
                    except CacheMiss:
                        misses += 1
                        continue
                    near = lambda ps: haversine_km(truth[0], truth[1], np.asarray([p.lat for p in ps]), np.asarray([p.lon for p in ps])) < 25  # noqa: E731
                    if kept:
                        extra[0] = extra[0] or bool((near(kept) & (np.asarray([p.uploaded for p in kept]) >= YFCC_END)).any())
                    if everything:
                        extra[1] = extra[1] or bool(near(everything).any())
                    found = [(p.photo_id, p.owner, p.lat, p.lon, p.accuracy) for p in kept]
                else:
                    found = replica.matches(q, exclude=exclude)[2]
                if found:
                    lat = np.asarray([r[2] for r in found])
                    lon = np.asarray([r[3] for r in found])
                    n_any = n_any or bool((haversine_km(truth[0], truth[1], lat, lon) < 25).any())
                    results.append(TSResult(q, "name", {"Flickr photos": len(found)}, replica._cluster([], found, q, None)))
            best = _pick(results)
            pick = best is not None and float(haversine_km(truth[0], truth[1], np.asarray([best.lat]), np.asarray([best.lon]))[0]) < 25
            stats[name] = (n_any, pick, best is not None)
        rows.append((bool(qs), *stats["flickr"], *stats["yfcc"], *extra))
    r = np.asarray(rows, bool)
    lines = [f"Flickr live vs YFCC replica, {tag}: {len(photos)} photos, {int(r[:, 0].sum())} with 4B queries; cache misses {misses}",
             "  % of photos           any match < 25 km   own pick < 25 km   has a pick"]
    subsets = [("all", np.ones(len(r), bool))]
    if pipe is not None:
        subsets += [("pipeline wrong (>= 25 km)", pipe >= 25), ("pipeline right", pipe < 25)]
    for label, m in subsets:
        lines.append(f"  {label} ({int(m.sum())})")
        for name, c in (("Flickr live", 1), ("YFCC replica", 4)):
            lines.append(f"    {name:<20} {100 * r[m, c].mean():17.1f} {100 * r[m, c + 1].mean():18.1f} {100 * r[m, c + 2].mean():12.1f}")
        lines.append(f"    {'either, any match':<20} {100 * (r[m, 1] | r[m, 4]).mean():17.1f}")
        lines.append(f"    {'Flickr, from 2015 on':<20} {100 * r[m, 7].mean():17.1f}  (any match uploaded after YFCC's last year)")
        lines.append(f"    {'Flickr, no guard':<20} {100 * r[m, 8].mean():17.1f}  (what the leak guard removes, for scale)")
        lines.append(f"    {'Flickr only, pick':<20} {100 * (r[m, 2] & ~r[m, 5]).mean():17.1f}  (live pick right, replica pick not)")
    text = "\n".join(lines)
    print(text)
    (CACHE / f"report_{tag}.txt").write_text(text + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("query", "usage", "harvest", "report"))
    parser.add_argument("text", nargs="?")
    parser.add_argument("--near", type=float, nargs=3, metavar=("LAT", "LON", "KM"))
    parser.add_argument("--offline", action="store_true", help="cache only, never call Flickr")
    parser.add_argument("--set", dest="tag", default="dev3k_setft_own")
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--test", action="store_true", help="allow im2gps3k / yfcc4k photo sets (only when the user asks)")
    args = parser.parse_args(argv)
    if args.cmd == "usage":
        print(FlickrSearch().usage())
    elif args.cmd == "harvest":
        harvest(args.tag, args.max_calls, args.workers, args.test)
    elif args.cmd == "report":
        report(args.tag, args.test)
    else:
        from ..experiment.text_search import Result as TSResult, TextSearch

        tool = FlickrSearch(offline=args.offline)
        res = tool.search(args.text, tuple(args.near) if args.near else None)
        rows = [(p.photo_id, p.owner, p.lat, p.lon, p.accuracy) for p in res.photos if p.accuracy >= MIN_ACCURACY]
        ts = TextSearch(yfcc=False)
        near = tuple(args.near) if args.near else None
        print(TSResult(args.text, "name", {"Flickr photos (total)": res.total, "returned, city-level or better": len(rows)},
                       ts._cluster([], rows, args.text, near)).text().replace("text_search(", "flickr_search(", 1))
        print(f"live calls: {tool.live_calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
