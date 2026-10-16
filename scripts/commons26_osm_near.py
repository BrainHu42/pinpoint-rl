"""Fast near-area OSM check on commons26 dev: one global name lookup per query (OSM FTS + places.db FTS, up to 2,000 matches each), then keep
matches within 25 km of each Pinpoint area. Per (photo, area): distinct queries with a match there, and the rarest such query's match count.
Saves per photo as it goes (jsonl), then reports the check as an area verifier."""
import collections
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUTF = Path("artifacts/commons26/osm_near_dev.jsonl")
bench = {r["IMG_ID"]: r for r in csv.DictReader(open("data/pinpoint/commons26/release/benchmark.csv", encoding="utf-8"))}
dev = [i for i, r in bench.items() if r["split"] == "dev"]


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def work(job):
    from geo_search_env.tools.maps import MAX_MATCHES, MapsSearch, _phrase

    global _M
    try:
        _M
    except NameError:
        _M = MapsSearch()
    i, queries, areas = job
    A = np.asarray(areas, float).reshape(-1, 2)
    hits = [[] for _ in range(len(A))]  # per area: (query, kind, n matches)
    for q, kind in queries:
        ph = _phrase(q)
        if ph is None or not len(A):
            continue
        try:
            rows = _M.osm.execute("SELECT p.lat, p.lon FROM names JOIN places p ON p.rowid = names.rowid WHERE names MATCH ? LIMIT ?", (ph, MAX_MATCHES)).fetchall()
            if _M.places is not None:
                rows += _M.places.execute("SELECT p.lat, p.lon FROM names JOIN places p ON p.rowid = names.rowid WHERE names MATCH ? LIMIT ?", (ph, MAX_MATCHES)).fetchall()
        except Exception:  # noqa: BLE001
            continue
        if not rows:
            continue
        P = np.asarray(rows, float)
        d = hav(P[:, None, 0], P[:, None, 1], A[None, :, 0], A[None, :, 1])  # [matches, areas]
        for k in np.flatnonzero((d < 25).any(0)):
            hits[k].append((q, kind, len(rows)))
    return i, hits


if __name__ == "__main__":
    from geo_search_env.experiment.read_constrain import _spans
    from geo_search_env.experiment.region_evidence import _raw_phrases

    phrases = _raw_phrases("commons26", "4b")
    photos = json.load(open("artifacts/photo_sets/commons26_setft_own/photos.json"))
    pidx = {e["image_id"]: k for k, e in enumerate(photos)}
    zm = np.load("artifacts/pinpoint_set/zoom_commons26_setft_own.npz")
    areas = collections.defaultdict(list)
    for j, p in enumerate(zm["photo"]):
        if zm["source"][j] == 0:
            areas[int(p)].append(zm["center"][j].astype(float).tolist())
    done = {json.loads(l)["image_id"] for l in OUTF.read_text().splitlines()} if OUTF.exists() else set()
    jobs = []
    for i in dev:
        if i in done:
            continue
        ph = list(phrases.get(i, {}))
        qs = [(q, "phrase") for q in ph] + [(s, "span") for s in sorted({s for q in ph for s in _spans(q)} - set(ph))]
        jobs.append((i, qs, areas[pidx[i]]))
    t0 = time.time()
    with OUTF.open("a") as f, ProcessPoolExecutor(8) as pool:
        for n, (i, hits) in enumerate(pool.map(work, jobs, chunksize=4)):
            f.write(json.dumps({"image_id": i, "areas": areas[pidx[i]], "hits": hits}) + "\n")
            if n % 100 == 0:
                f.flush()
                print(n, f"{time.time() - t0:.0f}s", flush=True)
