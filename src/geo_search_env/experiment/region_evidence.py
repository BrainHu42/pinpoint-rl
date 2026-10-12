# Which text-search results give a reliable region? (TEXT_SEARCH_PLAN.md step 0 follow-up.) Restricting Pinpoint to the true region would take dev3k
# from 33.4 to 61.3% < 25 km, but the text tool's blind picks point to wrong regions too often (-7.7). This breaks every query's top result down by
# clue type and specificity and asks, where its region differs from Pinpoint's top-1 region: how often is the query right (a possible fix) and how
# often is Pinpoint right (a possible break)? dev sets only.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.region_evidence search --tag dev3k [--clues 4b] [--workers 8]   (CPU)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.region_evidence report --tag dev3k

"""search: every model-produced phrase of each photo, raw (punctuation kept, so domains and phone numbers are recognised), with its source:
        landmark names (27B, 9B), text read in the photo (9B, full image + 4 crops), place names the 9B read, the 9B's search queries.
        Each is searched (photo and photographer excluded); its top cluster's (country, region) is the label of the nearest index photo
        (artifacts/place_labels). -> artifacts/agent/region_evidence_{tag}.json.
report: per group (source, query kind, exact name match, number of matches, cluster share, words, agreement between a photo's queries):
        results, region right %, and on (photo, region) pairs that differ from Pinpoint's top-1 region: query right vs Pinpoint right."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

OUT = Path("artifacts/agent")
PHOTO_SETS = Path("artifacts/photo_sets")
LABELS_NPZ = Path("artifacts/place_labels/mp16_country_region.npz")
SOURCES = ("landmark 27b", "landmark 9b", "text 9b", "place name 9b", "query 9b")  # first source wins for a repeated phrase


def evidence_path(tag: str, clues: str = "9b") -> Path:
    """region_evidence_{tag}.json for the 9B / 27B clue set (the first runs), region_evidence_{tag}_{clues}.json for others."""

    return OUT / (f"region_evidence_{tag}.json" if clues == "9b" else f"region_evidence_{tag}_{clues}.json")


def _raw_phrases(tag: str, clues: str = "9b") -> dict[str, dict[str, str]]:
    from .text_search import norm

    if clues != "9b":  # one model's clues: text and place names it read (observe extract), its search queries (observe queries)
        parsed = {k: (v.get("parsed") or {}) for k, v in json.loads((OUT / f"observe_{tag}_{clues}.json").read_text(encoding="utf-8")).items()}
        tables = {f"text {clues}": {k: [str(x) for x in v.get("text") or []] for k, v in parsed.items()},
                  f"place name {clues}": {k: [str(x) for x in v.get("place_names") or []] for k, v in parsed.items()},
                  f"query {clues}": {k: v or [] for k, v in json.loads((OUT / f"queries_{tag}_{clues}.json").read_text(encoding="utf-8")).items()}}
        out: dict[str, dict[str, str]] = {}
        for src, table in tables.items():
            for image_id, raws in table.items():
                seen = {norm(p) for p in out.get(image_id, {})}
                for raw in raws:
                    key = norm(raw)
                    if len(key) >= 3 and key not in seen:
                        seen.add(key)
                        out.setdefault(image_id, {})[raw.strip()] = src
        return out

    tables: dict[str, dict[str, list[str]]] = {}
    for name in ("27b", "9b"):
        path = OUT / f"landmark_{tag}_{name}.json"
        if path.exists():
            tables[f"landmark {name}"] = {k: [v["landmark"]] for k, v in json.loads(path.read_text(encoding="utf-8")).items() if v and v.get("landmark")}
    clues = OUT / f"observe_{tag}_9bcrop.json"
    if not clues.exists():
        clues = OUT / f"observe_{tag}_9b.json"
    parsed = {k: (v.get("parsed") or {}) for k, v in json.loads(clues.read_text(encoding="utf-8")).items()}
    tables["text 9b"] = {k: [str(x) for x in v.get("text") or []] for k, v in parsed.items()}
    tables["place name 9b"] = {k: [str(x) for x in v.get("place_names") or []] for k, v in parsed.items()}
    tables["query 9b"] = {k: v or [] for k, v in json.loads((OUT / f"queries_{tag}_9b.json").read_text(encoding="utf-8")).items()}
    out: dict[str, dict[str, str]] = {}
    for src in SOURCES:
        for image_id, raws in tables.get(src, {}).items():
            seen = {norm(p) for p in out.get(image_id, {})}
            for raw in raws:
                key = norm(raw)
                if len(key) >= 3 and key not in seen:
                    seen.add(key)
                    out.setdefault(image_id, {})[raw.strip()] = src
    return out


class _Labeller:
    """(country, region) of a coordinate: the label of the nearest Pinpoint index photo."""

    def __init__(self) -> None:
        from scipy.spatial import cKDTree

        from ..paths import PINPOINT_ROOT

        root = PINPOINT_ROOT / "exp/contrastive_retrieval/checkpoints/attention_retrieval_index"
        index_dir = next(p.parent for p in root.rglob("manifest.json") if p.parent.name != "osv_support_cache")
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        count = manifest["shapes"]["raw_image_embeddings"][0]
        latlon = np.array(np.memmap(index_dir / manifest["files"]["latlon_deg"], dtype=np.float32, mode="r", shape=(count, 2)), np.float64)
        cache = np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
        z = np.load(LABELS_NPZ)
        country = np.full(z["row"].max() + 1, "", dtype=object)
        region = np.full(z["row"].max() + 1, "", dtype=object)
        country[z["row"]], region[z["row"]] = z["country"], z["region"]
        self.country, self.region = country[cache], region[cache]
        self.tree = cKDTree(self._xyz(latlon))

    @staticmethod
    def _xyz(latlon: np.ndarray) -> np.ndarray:
        lat, lon = np.radians(latlon[:, 0]), np.radians(latlon[:, 1])
        return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 1)

    def __call__(self, lat: float, lon: float) -> tuple[str, str]:
        j = int(self.tree.query(self._xyz(np.asarray([[lat, lon]])))[1][0])
        return str(self.country[j]), str(self.region[j])


def _search_chunk(args: tuple[list[dict], dict[str, dict[str, str]], dict[str, str]]) -> list[dict[str, Any]]:
    from .text_search import TextSearch, haversine_km

    photos, phrases, owns = args
    tool, rows = TextSearch(), []
    for e in photos:
        own = owns[e["image_id"]]
        for q, src in phrases.get(e["image_id"], {}).items():
            r = tool.search(q, exclude=own)
            row: dict[str, Any] = {"image_id": e["image_id"], "query": q, "source": src, "kind": r.kind, "hint": r.hint, "loose": r.loose,
                                   "matches": int(sum(r.matches.values())), "words": len(q.split())}
            if r.clusters:
                c = r.clusters[0]
                row |= {"lat": c.lat, "lon": c.lon, "weight": c.weight, "share": c.share, "exact": c.exact, "places": c.places,
                        "km": float(haversine_km(*e["truth"], np.asarray([c.lat]), np.asarray([c.lon]))[0])}
            rows.append(row)
    return rows


def _owns(photos: list[dict]) -> dict[str, tuple[int, str]]:
    """(Flickr photo id, user) of each photo, excluded from the YFCC title search (the photo and its photographer). MP16 ids end in the photo
    id (user from the MP16 metadata); im2gps3k ids are "<photo id>_<secret>_<server>_<user>.jpg"; yfcc4k ids are "<photo id>.jpg" (user
    looked up in the YFCC table). Benchmark photos are themselves in YFCC100M, so this exclusion matters."""

    import sqlite3

    from .text_search import NVME, ROOT
    from .yfcc_names import _mp16_authors

    out: dict[str, tuple[int, str]] = {}
    mp16 = [e for e in photos if "benchmark" not in e]
    if mp16:
        authors = _mp16_authors({e["image_id"] for e in mp16})
        out |= {e["image_id"]: (int(e["image_id"].split("_")[-1].split(".")[0]), authors.get(e["image_id"], "")) for e in mp16}
    bench = [e for e in photos if "benchmark" in e]
    lookup = []
    for e in bench:
        parts = e["image_id"].rsplit(".", 1)[0].split("_")
        if len(parts) >= 4:
            out[e["image_id"]] = (int(parts[0]), parts[-1])
        else:
            lookup.append(e["image_id"])
    if lookup:
        db = NVME / "yfcc.db" if (NVME / "yfcc.db").exists() else ROOT / "yfcc.db"
        ids = {int(i.rsplit(".", 1)[0]): i for i in lookup}
        con = sqlite3.connect(db)
        con.execute("CREATE TEMP TABLE want (photo_id INTEGER PRIMARY KEY)")
        con.executemany("INSERT INTO want VALUES (?)", [(i,) for i in ids])
        uid = dict(con.execute("SELECT m.photo_id, m.uid FROM meta m JOIN want w ON m.photo_id = w.photo_id").fetchall())
        out |= {name: (pid, uid.get(pid, "")) for pid, name in ids.items()}
        print(f"  yfcc users found for {sum(p in uid for p in ids)} of {len(ids)} photos", flush=True)
    return out


def search(tag: str, clues: str = "9b", workers: int = 8) -> None:
    import multiprocessing
    import os
    import time

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    phrases = _raw_phrases(tag, clues)
    photos = [e for e in photos if e["image_id"] in phrases]
    owns = _owns(photos)
    chunks = [photos[i:i + 200] for i in range(0, len(photos), 200)]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # one BLAS thread per worker: clustering is 2x faster
        os.environ.setdefault(var, "1")
    start, rows = time.time(), []
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for k, part in enumerate(pool.imap(_search_chunk, [(c, {e["image_id"]: phrases[e["image_id"]] for e in c}, {e["image_id"]: owns[e["image_id"]] for e in c}) for c in chunks])):
            rows += part
            if k % 10 == 0:
                print(f"  {min((k + 1) * 200, len(photos))} of {len(photos)} photos, {len(rows)} searches, {time.time() - start:.0f} s", flush=True)
    label = _Labeller()
    for row in rows:
        if "lat" in row:
            row["region"] = label(row["lat"], row["lon"])
    path = evidence_path(tag, clues)
    path.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    print(f"{len(rows)} searches for {len(photos)} photos -> {path}")


def report(tag: str, clues: str = "9b") -> None:
    from .agent_targets import same_region

    rows = json.loads(evidence_path(tag, clues).read_text(encoding="utf-8"))
    labels = {p["image_id"]: p for p in json.loads((PHOTO_SETS / tag / "labels.json").read_text(encoding="utf-8"))}
    truth = {k: (v["truth"].get("country") or "", v["truth"].get("region") or "") for k, v in labels.items()}
    pin = {k: ((v["pool"][0].get("country") or "", v["pool"][0].get("region") or "") if v["pool"] else ("", "")) for k, v in labels.items()}
    match = lambda a, b: same_region({"country": a[0], "region": a[1]}, {"country": b[0], "region": b[1]})
    found = [r for r in rows if "region" in r and truth[r["image_id"]][1]]
    agree = collections.Counter((r["image_id"], tuple(r["region"])) for r in found)
    for r in found:
        r["agree"] = agree[(r["image_id"], tuple(r["region"]))]
        r["right"] = match(r["region"], truth[r["image_id"]])
        r["country_right"] = bool(r["region"][0]) and r["region"][0] == truth[r["image_id"]][0]
        r["pin_right"] = match(pin[r["image_id"]], truth[r["image_id"]])
        r["differs"] = not match(r["region"], pin[r["image_id"]])
    print(f"{tag}: {len(rows)} searches, {len(found)} with a top place and a labelled truth region; Pinpoint's top-1 region right on "
          f"{np.mean([match(pin[k], truth[k]) for k in truth if truth[k][1]]):.1%} of labelled photos")
    print(f"  {'group':34s} {'results':>7s} {'region':>7s} {'country':>7s} | differ from Pinpoint's region: {'pairs':>5s} {'query right':>11s} {'Pinpoint right':>14s}")

    def show(name: str, sel: list[dict]) -> None:
        if not sel:
            return
        pairs = {(r["image_id"], tuple(r["region"])): r for r in sel if r["differs"]}
        fix = sum(r["right"] for r in pairs.values())
        brk = sum(r["pin_right"] for r in pairs.values())
        print(f"  {name:34s} {len(sel):7d} {np.mean([r['right'] for r in sel]):7.0%} {np.mean([r['country_right'] for r in sel]):7.0%} | "
              f"{'':30s} {len(pairs):5d} {fix:11d} {brk:14d}")

    groups = {
        "all": lambda r: True,
        "source": lambda r: r["source"],
        "kind": lambda r: r["kind"] + (" (hint)" if r["hint"] else ""),
        "exact name match": lambda r: "exact" if r.get("exact") else "partial",
        "matches": lambda r: "1" if r["matches"] == 1 else "2-5" if r["matches"] <= 5 else "6-50" if r["matches"] <= 50 else "51-999" if r["matches"] < 1000 else "1000+",
        "share of matches": lambda r: ">=0.9" if r["share"] >= 0.9 else "0.5-0.9" if r["share"] >= 0.5 else "<0.5",
        "words": lambda r: str(min(r["words"], 4)) + ("+" if r["words"] >= 4 else ""),
        "queries agreeing (same photo)": lambda r: str(min(r["agree"], 3)) + ("+" if r["agree"] >= 3 else ""),
    }
    for title, key in groups.items():
        print(f" {title}")
        buckets: dict[str, list[dict]] = collections.defaultdict(list)
        for r in found:
            buckets[str(key(r))].append(r)
        for b in sorted(buckets):
            show(b, buckets[b])
    print(" combined rules")
    rules = {
        "exact, <= 5 matches": lambda r: r.get("exact") and r["matches"] <= 5,
        "exact, <= 5 matches, share >= 0.9": lambda r: r.get("exact") and r["matches"] <= 5 and r["share"] >= 0.9,
        "exact, 2+ words, <= 5 matches": lambda r: r.get("exact") and r["words"] >= 2 and r["matches"] <= 5,
        "2+ queries agree": lambda r: r["agree"] >= 2,
        "2+ agree, share >= 0.9": lambda r: r["agree"] >= 2 and r["share"] >= 0.9,
        "domain or phone": lambda r: r["kind"] in ("domain", "phone"),
        "text read, exact, <= 5 matches": lambda r: r["source"] in ("text 9b", "place name 9b") and r.get("exact") and r["matches"] <= 5,
        "landmark, exact": lambda r: r["source"].startswith("landmark") and r.get("exact"),
    }
    for name, rule in rules.items():
        show(name, [r for r in found if rule(r)])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("search", "report"))
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--clues", default="9b", help="clue set: 9b (the 9B / 27B phrases) or a model name with observe extract + queries files")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    search(args.tag, args.clues, args.workers) if args.cmd == "search" else report(args.tag, args.clues)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
