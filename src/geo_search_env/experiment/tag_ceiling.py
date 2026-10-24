# Would words find the place? A ceiling worked back from the truth for Flickr text search as the agent's channel (FLICKR_PLAN.md, rethink of
# 2026-10-24): each dev3k photo's own Flickr title + tags (YFCC100M metadata; measurement only, never an input) searched in the YFCC copy
# with the photo and its photographer excluded. Words are split into place names (GeoNames towns >= 1,000 people, countries) and content
# words: content words are what a policy describing the photo could write; place names need the name to be legible or known.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.tag_ceiling [--limit N] [--workers 12]

"""Per photo and query set, whether the heaviest result cluster (25 km, each Flickr user weighted 1 in total, as text_search) lies within
25 km of the truth, and whether any of the top 8 does. Query sets: content words alone and in pairs (AND), all words alone and in pairs,
and the 4B's own queries (queries_dev3k_setft_own_4b.json, the pipeline's read chain) as the baseline. Also a pick without the truth:
the heaviest top cluster holding >= half its query's matches (text_search._pick). By the pipeline's loss bucket (viz/errors.py tiers;
0 = right) -> artifacts/tag_ceiling/report.txt, per-photo rows in rows.json."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from multiprocessing import Pool
from pathlib import Path
import sqlite3

import numpy as np

OUT = Path("artifacts/tag_ceiling")
PHOTO_SET = Path("artifacts/photo_sets/dev3k_setft_own")
TRACES = Path("artifacts/viz/traces/dev3k")
QUERIES_4B = Path("artifacts/agent/queries_dev3k_setft_own_4b.json")
YFCC_DB = Path("/opt/dlami/nvme/flickr_ceiling/yfcc.db")  # a copy of data/text_search/yfcc.db on the NVMe drive (random reads)
MAX_ROWS = 10_000
SAMPLE = 500
MIN_ACCURACY = 11
RIGHT_KM = 25.0
CAP_SINGLE, CAP_PAIR = 10, 6  # words searched alone / words whose pairs are searched
JUNK = set("""img dsc dscn dscf jpg jpeg photo photos pic picture image canon nikon sony olympus panasonic fuji fujifilm pentax leica iphone
android instagram square squareformat iphoneography app eos lumix camera lens mm the and with for from this that our my your his her its
vacation holiday holidays trip travel travels day days night week weekend year new old geotagged geotag geo lat lon flickr uploaded upload
untitled family friends me him she they all best nice good great beautiful view views 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013
2014 january february march april may june july august september october november december spring summer autumn fall winter morning
evening afternoon sunday monday tuesday wednesday thursday friday saturday hdr bw blackandwhite black white color colour panorama pano
nikkor sigma tamron""".split())

_state: dict = {}


def _worker_init(towns: frozenset, town_pairs: frozenset) -> None:
    from .text_search import TextSearch

    _state["db"] = sqlite3.connect(f"file:{YFCC_DB}?mode=ro", uri=True)
    _state["ts"] = TextSearch(yfcc=False)  # clustering and town labels only
    _state["towns"], _state["town_pairs"] = towns, town_pairs


def _search(words: tuple[str, ...], author: str, own: int, truth) -> tuple[bool, bool, float, float]:
    """(top cluster < 25 km, any top-8 cluster < 25 km, top cluster weight, top cluster share)."""

    from .text_search import haversine_km

    db = _state["db"]
    match = " ".join(f'"{w}"' for w in words)  # AND of whole words
    ids = [r for (r,) in db.execute(f"SELECT rowid FROM titles WHERE titles MATCH ? LIMIT {MAX_ROWS}", [match])]
    rows = []
    for s in range(0, len(ids), 900):
        chunk = ids[s:s + 900]
        rows += db.execute(f"SELECT photo_id, uid, lat, lon, accuracy FROM meta WHERE rowid IN ({','.join('?' * len(chunk))})", chunk).fetchall()
    rows = [r for r in rows if r[4] >= MIN_ACCURACY and r[1] != author and r[0] != own]
    if not rows:
        return False, False, 0.0, 0.0
    if len(rows) > SAMPLE:
        rng = np.random.default_rng(abs(hash(match)) % 2**32)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), SAMPLE, replace=False))]
    clusters = _state["ts"]._cluster([], rows, " ".join(words), None)
    if not clusters:
        return False, False, 0.0, 0.0
    d = haversine_km(truth[0], truth[1], np.asarray([c.lat for c in clusters]), np.asarray([c.lon for c in clusters]))
    return bool(d[0] < RIGHT_KM), bool((d < RIGHT_KM).any()), clusters[0].weight, clusters[0].share


def _photo(job) -> dict:
    image_id, words, author, own, truth, queries_4b = job
    towns, pairs = _state["towns"], _state["town_pairs"]
    place = [w in towns for w in words]
    for k in range(len(words) - 1):
        if f"{words[k]} {words[k + 1]}" in pairs:
            place[k] = place[k + 1] = True
    content = [w for w, p in zip(words, place) if not p]
    sets = {"content single": [(w,) for w in content[:CAP_SINGLE]],
            "content pair": list(combinations(content[:CAP_PAIR], 2)),
            "all single": [(w,) for w in words[:CAP_SINGLE]],
            "all pair": list(combinations(words[:CAP_PAIR], 2))}
    from .text_search import norm

    sets["4B queries"] = [tuple(norm(q).split()) for q in queries_4b if norm(q).split()]
    out = {"image_id": image_id, "words": len(words), "place_words": int(sum(place)), "content_words": len(content)}
    cache: dict = {}
    for name, qs in sets.items():
        top = anyk = False
        pick = None
        for q in qs:
            if q not in cache:
                cache[q] = _search(q, author, own, truth)
            t, a, w, share = cache[q]
            top, anyk = top or t, anyk or a
            if share >= 0.5 and (pick is None or w > pick[0]):
                pick = (w, t)
        out[name] = [top, anyk, bool(pick and pick[1]), pick is not None]
    out["hits"] = [" ".join(q) for q in sets["content single"] + sets["content pair"] if cache[q][0]]  # content queries whose top cluster is right
    out["content"] = content
    return out


def run(limit: int | None, workers: int) -> None:
    from .text_search import PLACES_DB, _countries, norm
    from .yfcc_names import _mp16_authors

    photos = json.loads((PHOTO_SET / "photos.json").read_text(encoding="utf-8"))[:limit]
    tiers = {e["i"]: e["tier"] for e in json.loads((TRACES / "errors.json").read_text(encoding="utf-8"))["errors"]}
    authors = _mp16_authors({e["image_id"] for e in photos})
    q4b = json.loads(QUERIES_4B.read_text(encoding="utf-8"))
    g = np.load(Path("data/yfcc100m/geo.npz"))
    pid = g["photo_id"]
    order = np.argsort(pid)
    blob, offsets = g["blob"], g["offsets"]
    places = sqlite3.connect(f"file:{PLACES_DB}?mode=ro", uri=True)
    names = {norm(n) for (n,) in places.execute("SELECT name FROM towns")} | {norm(n) for n in _countries()[2].values()}
    towns = frozenset(n for n in names if n and " " not in n)
    town_pairs = frozenset(" ".join(n.split()[:2]) for n in names if n and len(n.split()) >= 2)
    jobs = []
    for e in photos:
        own = int(e["image_id"].rsplit("_", 1)[-1][:-4])
        j = order[np.searchsorted(pid, own, sorter=order)]
        text = bytes(blob[offsets[j]:offsets[j + 1] - 1]).decode("utf-8", "replace") if pid[j] == own else ""
        words = list(dict.fromkeys(w for w in text.split() if len(w) >= 3 and w not in JUNK and not w.isdigit()))
        jobs.append((e["image_id"], words, authors.get(e["image_id"], "?"), own, e["truth"], q4b.get(e["image_id"], [])))
    with Pool(workers, initializer=_worker_init, initargs=(towns, town_pairs)) as pool:
        rows = []
        for n, r in enumerate(pool.imap(_photo, jobs, chunksize=4), 1):
            rows.append(r)
            if n % 250 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)
    for i, r in enumerate(rows):
        r["tier"] = tiers.get(i, 0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    report(rows)


def report(rows: list[dict] | None = None) -> None:
    rows = rows or json.loads((OUT / "rows.json").read_text(encoding="utf-8"))
    tier = np.asarray([r["tier"] for r in rows])
    has_words = np.asarray([r["words"] > 0 for r in rows])
    buckets = [("all photos", np.ones(len(rows), bool)), ("pipeline right", tier == 0), ("retrieval miss (6)", tier == 6),
               ("in pool, not an area (5)", tier == 5), ("right area ranked low / close calls (1, 2, 4)", np.isin(tier, (1, 2, 4))),
               ("near miss (3)", tier == 3)]
    names = ("content single", "content pair", "all single", "all pair", "4B queries")
    lines = [f"Tag ceiling, dev3k: {len(rows)} photos; with title / tag words {100 * has_words.mean():.1f}%; median words "
             f"{np.median([r['words'] for r in rows]):.0f}, place words {np.median([r['place_words'] for r in rows]):.0f}, content words "
             f"{np.median([r['content_words'] for r in rows]):.0f}",
             "  % of photos: top cluster < 25 km for some query (oracle over queries) | any top-8 cluster | pick without the truth right"]
    for label, m in buckets:
        lines.append(f"  {label} ({int(m.sum())})")
        for name in names:
            v = np.asarray([r[name] for r in rows])[m]
            lines.append(f"    {name:<15} top {100 * v[:, 0].mean():5.1f}   any-8 {100 * v[:, 1].mean():5.1f}   pick {100 * v[:, 2].mean():5.1f}"
                         f"   (has a pick {100 * v[:, 3].mean():5.1f})")
    text = "\n".join(lines)
    print(text)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)
    if args.report:
        report()
    else:
        run(args.limit, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
