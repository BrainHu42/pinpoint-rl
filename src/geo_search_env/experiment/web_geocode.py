# Ceiling of keyless online geocoders for the text clues (TEXT_SEARCH_PLAN.md, after lesson 66: the offline clues are coverage-limited):
# OpenStreetMap Nominatim (named shops, streets, stations, buildings) and the English Wikipedia search API (pages with coordinates). Only the
# clues of Pinpoint's dev3k mistakes are queried (the ceiling question), each unique query once, cached and resumable. Nominatim's usage policy:
# at most 1 request / second with an identifying user agent. dev sets only.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.web_geocode fetch --source nominatim --tag dev3k   (~2 h)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.web_geocode fetch --source wikipedia --tag dev3k
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.web_geocode report --tag dev3k

"""fetch:  up to 5 results (lat, lon, name) per unique clue -> artifacts/agent/web_{source}_{tag}.jsonl (one line per query; reruns skip done ones).
report: per source, Pinpoint's mistakes where a clue's first result / any result is < 25 km from the truth; the same with clues shuffled among
        the mistake photos (luck); and how many of the real ones the offline text search (its top result) doesn't already reach."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
PHOTO_SETS = Path("artifacts/photo_sets")
AGENT = "pinpoint-rl geolocation research pilot (contact: brainhu42@gmail.com)"
URLS = {"nominatim": "https://nominatim.openstreetmap.org/search?format=jsonv2&limit=5&q={q}",
        "wikipedia": "https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrlimit=5&prop=coordinates&format=json&gsrsearch={q}"}
DELAY = {"nominatim": 1.1, "wikipedia": 0.25}


def _mistakes(tag: str, clues: str = "4b") -> tuple[list[dict], dict[str, dict[str, str]]]:
    from .region_evidence import _raw_phrases
    from .strategy_search import _haversine_km

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    phrases = _raw_phrases(tag, clues)
    mistakes = [e for e in photos if _haversine_km(*e["truth"], np.asarray([e["pool"][0]]))[0] >= 25 and e["image_id"] in phrases]
    return mistakes, phrases


def _parse(source: str, body: dict | list) -> list[list]:
    if source == "nominatim":
        return [[float(r["lat"]), float(r["lon"]), r.get("display_name", "")[:120]] for r in body]
    pages = sorted((body.get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 99))
    return [[p["coordinates"][0]["lat"], p["coordinates"][0]["lon"], p["title"]] for p in pages if p.get("coordinates")]


def fetch(source: str, tag: str) -> None:
    mistakes, phrases = _mistakes(tag)
    queries = sorted({q for e in mistakes for q in phrases[e["image_id"]]})
    path = OUT / f"web_{source}_{tag}.jsonl"
    done = {json.loads(line)["query"] for line in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
    todo = [q for q in queries if q not in done]
    print(f"{source}: {len(queries)} unique clues of {len(mistakes)} mistake photos, {len(todo)} to fetch", flush=True)
    with path.open("a", encoding="utf-8") as out:
        for n, q in enumerate(todo):
            request = urllib.request.Request(URLS[source].format(q=urllib.parse.quote(q)), headers={"User-Agent": AGENT})
            results, error = [], None
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=30) as response:
                        results = _parse(source, json.loads(response.read()))
                    error = None
                    break
                except Exception as e:  # noqa: BLE001 - retried, then recorded as failed
                    error = str(e)
                    time.sleep(5 * (attempt + 1))
            out.write(json.dumps({"query": q, "results": results, "error": error}) + "\n")
            out.flush()
            time.sleep(DELAY[source])
            if n % 500 == 0:
                print(f"  {n} of {len(todo)}", flush=True)
    print(f"-> {path}")


def report(tag: str) -> None:
    from .strategy_search import _haversine_km

    mistakes, phrases = _mistakes(tag)
    offline = {}
    for r in json.loads((OUT / f"region_evidence_{tag}_4b.json").read_text(encoding="utf-8")):
        if r.get("km", 1e9) < 25:
            offline[r["image_id"]] = True
    ids = [e["image_id"] for e in mistakes]
    perm = np.random.default_rng(0).permutation(len(ids))
    perm = np.where(perm == np.arange(len(ids)), (perm + 1) % len(ids), perm)
    shuffled = {ids[i]: phrases[ids[perm[i]]] for i in range(len(ids))}
    n_photos = len(json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8")))
    print(f"{tag}: {len(mistakes)} of Pinpoint's mistakes have 4B clues; offline text search (top result) reaches {sum(offline.get(i, False) for i in ids)} of them")
    for source in URLS:
        path = OUT / f"web_{source}_{tag}.jsonl"
        if not path.exists():
            continue
        table = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            table[row["query"]] = row["results"]
        failed = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if json.loads(line)["error"])
        print(f" {source}: {len(table)} clues fetched ({failed} failed), results for {sum(bool(v) for v in table.values())}")
        for name, clue_sets in (("real clues", {i: phrases[i] for i in ids}), ("shuffled clues (luck)", shuffled)):
            first = anyhit = new_first = 0
            for e in mistakes:
                res = [table.get(q) or [] for q in clue_sets[e["image_id"]]]
                firsts = [r[0][:2] for r in res if r]
                alls = [x[:2] for r in res for x in r]
                f = bool(firsts) and (_haversine_km(*e["truth"], np.asarray(firsts)) < 25).any()
                a = bool(alls) and (_haversine_km(*e["truth"], np.asarray(alls)) < 25).any()
                first += f
                anyhit += a
                new_first += f and not offline.get(e["image_id"], False)
            print(f"   {name:22s} a first result < 25 km: {first:4d} (+{100 * first / n_photos:.1f} pts); any of 5: {anyhit:4d}; "
                  f"first result and not reached offline: {new_first}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("fetch", "report"))
    parser.add_argument("--source", choices=tuple(URLS), default="nominatim")
    parser.add_argument("--tag", default="dev3k")
    args = parser.parse_args(argv)
    fetch(args.source, args.tag) if args.cmd == "fetch" else report(args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
