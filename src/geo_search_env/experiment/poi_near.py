# Fuzzy business-name search (AGENT_DIRECTION.md): Overture POIs within ~2 km of every retrieval candidate, fuzzy-matched against the text and place names
# a VLM read in the photo (observe.py), so misreadings ("Ketter" for "Tepper") and partial names still count. Chains (Overture brand set) are down-weighted.
# Needs DuckDB + rapidfuzz (env .venvs/overture) and the places parquet. No package imports, so it runs in that env.
# Usage: PYTHONPATH=src .venvs/overture/bin/python -m geo_search_env.experiment.poi_near --tags dev val --clues 9b

"""Writes artifacts/agent/poifuzzy_{tag}.npz: F [photos, C, 4] = best fuzzy score (0-1, length-weighted) of any read string vs a POI name within 0.5 km
and within 2 km of the candidate; the same within 2 km for non-chain POIs only; 1 if a read string equals a non-chain POI name within 2 km.
geo_tools._load picks these up as the 'fuzzy' features."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Sequence

import numpy as np

PLACES = Path("data/pinpoint/overture/places")
PHOTO_SETS = Path("artifacts/photo_sets")
OUT = Path("artifacts/agent")
CELL = 0.02  # degrees; candidates look up their own and the 8 neighbouring cells (>= ~2 km in every direction below 60 degrees latitude)
MIN_LEN = 4
NAMES = ("fuzzy 0.5", "fuzzy 2", "fuzzy nochain 2", "fuzzy exact 2")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", " ", text).strip()


def _cells(lat: float, lon: float) -> list[tuple[int, int]]:
    a, b = math.floor(lat / CELL), math.floor(lon / CELL)
    return [(a + i, b + j) for i in (-1, 0, 1) for j in (-1, 0, 1)]


def main(argv: Sequence[str] | None = None) -> int:
    import duckdb
    from rapidfuzz import fuzz

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=["dev", "val"])
    parser.add_argument("--clues", default="9b")
    parser.add_argument("--memory", default="10GB")
    args = parser.parse_args(argv)
    sets = {t: json.loads((PHOTO_SETS / t / "photos.json").read_text(encoding="utf-8")) for t in args.tags}
    clues = {t: json.loads((OUT / f"observe_{t}_{args.clues}.json").read_text(encoding="utf-8")) for t in args.tags}
    cells = {c for photos in sets.values() for e in photos for lat, lon in e["pool"] for c in _cells(lat, lon)}
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{args.memory}'")
    con.execute("CREATE TABLE cells (a INTEGER, b INTEGER)")
    con.executemany("INSERT INTO cells VALUES (?, ?)", sorted(cells))
    res = con.execute(f"""
        SELECT names."primary" AS name, (bbox.ymin + bbox.ymax) / 2 AS lat, (bbox.xmin + bbox.xmax) / 2 AS lon, brand IS NOT NULL AS chain
        FROM read_parquet('{PLACES}/*.parquet') p
        JOIN cells c ON c.a = floor(((p.bbox.ymin + p.bbox.ymax) / 2) / {CELL}) AND c.b = floor(((p.bbox.xmin + p.bbox.xmax) / 2) / {CELL})
        WHERE names."primary" IS NOT NULL
    """).fetchnumpy()
    names = [norm(n) for n in res["name"]]
    lat, lon, chain = np.asarray(res["lat"], float), np.asarray(res["lon"], float), np.asarray(res["chain"], bool)
    index: dict[tuple[int, int], list[int]] = {}
    for i, (a, b) in enumerate(zip(np.floor(lat / CELL).astype(int), np.floor(lon / CELL).astype(int))):
        index.setdefault((a, b), []).append(i)
    print(f"{len(cells)} cells, {len(names)} POIs around the candidates", flush=True)
    for tag, photos in sets.items():
        C = max(len(e["pool"]) for e in photos)
        F = np.zeros((len(photos), C, len(NAMES)), np.float32)
        for p, e in enumerate(photos):
            o = (clues[tag].get(e["image_id"]) or {}).get("parsed") or {}
            reads = [r for r in dict.fromkeys(norm(x) for x in (o.get("text") or []) + (o.get("place_names") or [])) if len(r) >= MIN_LEN]
            if not reads:
                continue
            for k, (clat, clon) in enumerate(e["pool"]):
                ids = np.asarray([i for c in _cells(clat, clon) for i in index.get(c, [])], np.int64)
                if not len(ids):
                    continue
                la, lo = np.radians(lat[ids]), np.radians(lon[ids])
                h = np.sin((la - math.radians(clat)) / 2) ** 2 + np.cos(la) * math.cos(math.radians(clat)) * np.sin((lo - math.radians(clon)) / 2) ** 2
                km = 2 * 6371.0088 * np.arcsin(np.sqrt(h))
                for j, d in zip(ids[km <= 2.0], km[km <= 2.0]):
                    n = names[j]
                    if len(n) < MIN_LEN:
                        continue
                    best = max(fuzz.ratio(r, n) if abs(len(r) - len(n)) <= 4 else fuzz.partial_ratio(r, n) * (min(len(r), len(n)) / max(len(r), len(n))) ** 0.5
                               for r in reads) / 100.0
                    if best < 0.8:
                        continue
                    s = best * min(len(n), 20) / 20  # long names count more
                    if d <= 0.5:
                        F[p, k, 0] = max(F[p, k, 0], s)
                    F[p, k, 1] = max(F[p, k, 1], s)
                    if not chain[j]:
                        F[p, k, 2] = max(F[p, k, 2], s)
                        F[p, k, 3] = max(F[p, k, 3], float(n in reads))
            if p % 250 == 0:
                print(f"  {tag}: {p} / {len(photos)}", flush=True)
        np.savez(OUT / f"poifuzzy_{tag}.npz", F=F, names=np.asarray(NAMES), image_ids=np.asarray([e["image_id"] for e in photos]))
        nz = (F[..., 1] > 0).any(1).mean()
        print(f"{tag}: fuzzy POI features -> {OUT / f'poifuzzy_{tag}.npz'}; photos with any match {nz:.1%}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
