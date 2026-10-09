# Text clues beyond each search's top result (TEXT_SEARCH_PLAN.md, after lesson 66): the right place is somewhere in a clue's results for 231
# of Pinpoint's dev3k mistakes but at the top for only 87. Two ways to use the rest, both as constraints rather than rankings:
#   candidates: a clue confirms one of Pinpoint's own candidates when it has a match near it (an ambiguous name becomes decisive next to a
#               candidate retrieval already proposed);
#   cooccur:    two different clues of the same photo with matches within ~1 km of each other point to the same spot.
# Each is measured against Pinpoint's top-1, with a shuffled-clue control (clues from another photo) for lucky matches. dev sets only.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.clue_points search --tag dev3k --clues 4b   (CPU, raw match points per clue)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.clue_points candidates --tag dev3k --clues 4b
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.clue_points cooccur --tag dev3k --clues 4b

"""search:     every clue of region_evidence's clue set, its raw matches (text_search.TextSearch.matches; each source sampled to SAMPLE; the photo
            and its photographer excluded) -> artifacts/agent/clue_points_{tag}_{clues}.npz: per clue image id, query, source, total matches;
            per match lat, lon, Flickr user (-1 for places), offsets.
candidates: per photo and Pinpoint candidate, the clues with a match within R km of it; switch from Pinpoint's top-1 to the best-supported other
            candidate when the top-1 has no support. Ceiling (a candidate < 25 km from the truth has support while Pinpoint is wrong), rules over
            R and a clue-rarity cap (thresholds chosen on one half of the photos, counted on the other), real vs shuffled clues.
cooccur:    per photo, pairs of clues sharing no word whose matches come within D km; the spot backed by the most clues is the proposal. Proposal
            right / wrong vs Pinpoint, real vs shuffled clues (a photo's clue paired with another photo's clues)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

OUT = Path("artifacts/agent")
PHOTO_SETS = Path("artifacts/photo_sets")
EARTH_KM = 6371.0


def _path(tag: str, clues: str) -> Path:
    return OUT / f"clue_points_{tag}_{clues}.npz"


def _chunk(args: tuple[list[dict], dict[str, dict[str, str]], dict[str, str]]) -> list[tuple]:
    import zlib

    from .text_search import TextSearch

    photos, phrases, authors = args
    tool, out = TextSearch(), []
    for e in photos:
        own = (int(e["image_id"].split("_")[-1].split(".")[0]), authors.get(e["image_id"], ""))
        for q, src in phrases.get(e["image_id"], {}).items():
            kind, places, photos_, counts, _, _ = tool.matches(q, exclude=own)
            lat = [p[1] for p in places] + [p[2] for p in photos_]
            lon = [p[2] for p in places] + [p[3] for p in photos_]
            user = [-1] * len(places) + [zlib.crc32(p[1].encode()) & 0x7FFFFFFF for p in photos_]
            out.append((e["image_id"], q, src, int(sum(counts.values())), lat, lon, user))
    return out


def search(tag: str, clues: str = "4b", workers: int = 4) -> None:
    import multiprocessing
    import os

    from .region_evidence import _raw_phrases
    from .yfcc_names import _mp16_authors

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    phrases = _raw_phrases(tag, clues)
    photos = [e for e in photos if e["image_id"] in phrases]
    authors = _mp16_authors({e["image_id"] for e in photos})
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    chunks = [photos[i:i + 200] for i in range(0, len(photos), 200)]
    rows = []
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for part in pool.imap(_chunk, [(c, {e["image_id"]: phrases[e["image_id"]] for e in c}, authors) for c in chunks]):
            rows += part
    offsets = np.cumsum([0] + [len(r[4]) for r in rows])
    np.savez(_path(tag, clues), image_id=np.asarray([r[0] for r in rows]), query=np.asarray([r[1] for r in rows]),
             source=np.asarray([r[2] for r in rows]), total=np.asarray([r[3] for r in rows]), offsets=offsets,
             lat=np.asarray([x for r in rows for x in r[4]], np.float32), lon=np.asarray([x for r in rows for x in r[5]], np.float32),
             user=np.asarray([x for r in rows for x in r[6]], np.int64))
    print(f"{len(rows)} clues, {offsets[-1]} matches -> {_path(tag, clues)}")


def _load(tag: str, clues: str) -> tuple[list[dict], dict[str, list[dict]]]:
    """Photos and, per image id, its clues: {query, source, total, pts [n, 2] radians}."""

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    with np.load(_path(tag, clues)) as z:
        z = dict(z)
    lat, lon = np.radians(z["lat"].astype(np.float64)), np.radians(z["lon"].astype(np.float64))
    per: dict[str, list[dict]] = {}
    for i, image_id in enumerate(z["image_id"]):
        a, b = z["offsets"][i], z["offsets"][i + 1]
        if b > a:
            per.setdefault(str(image_id), []).append({"query": str(z["query"][i]), "source": str(z["source"][i]), "total": int(z["total"][i]),
                                                      "pts": np.stack([lat[a:b], lon[a:b]], 1)})
    return photos, per


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle km between every row of a [n, 2] and b [m, 2] (radians)."""

    dlat = a[:, None, 0] - b[None, :, 0]
    dlon = a[:, None, 1] - b[None, :, 1]
    h = np.sin(dlat / 2) ** 2 + np.cos(a[:, None, 0]) * np.cos(b[None, :, 0]) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _km(p: Sequence[float], q: Sequence[float]) -> float:
    return float(_dist(np.radians(np.asarray([p], np.float64)), np.radians(np.asarray([q], np.float64)))[0, 0])


def _shuffled(photos: list[dict], per: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Each photo gets the clues of another photo (a fixed derangement over photos with clues): the luck control."""

    ids = [e["image_id"] for e in photos if e["image_id"] in per]
    perm = np.random.default_rng(0).permutation(len(ids))
    perm = np.where(perm == np.arange(len(ids)), (perm + 1) % len(ids), perm)
    return {ids[i]: per[ids[perm[i]]] for i in range(len(ids))}


def _report(label: str, photos: list[dict], chosen: dict[str, Sequence[float]], base: dict[str, float]) -> tuple[int, int]:
    from .strategy_search import _bootstrap

    fixes = breaks = 0
    gain = []
    for e in photos:
        k = e["image_id"]
        d = _km(e["truth"], chosen[k]) if k in chosen else base[k]
        fixes += d < 25 <= base[k]
        breaks += base[k] < 25 <= d
        gain.append((d < 25) * 1.0 - (base[k] < 25))
    c = _bootstrap(np.asarray(gain))
    print(f"    {label:58s} switched {len(chosen):4d}: fixes {fixes:4d}, breaks {breaks:4d}, < 25 km {100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]")
    return fixes, breaks


def candidates(tag: str, clues: str = "4b") -> None:
    photos, per = _load(tag, clues)
    base = {e["image_id"]: _km(e["truth"], e["pool"][0]) for e in photos}
    print(f"{tag}: {len(photos)} photos, {len(per)} with clue matches; Pinpoint < 25 km {np.mean([v < 25 for v in base.values()]):.1%}")
    for name, table in (("real clues", per), ("shuffled clues (luck)", _shuffled(photos, per))):
        print(f" {name}")
        support: dict[float, dict[str, np.ndarray]] = {}  # radius -> image id -> [clues, candidates] min clue total with a match near the candidate
        for radius in (1.0, 5.0):
            out = {}
            for e in photos:
                cl = table.get(e["image_id"], [])
                if not cl:
                    continue
                cand = np.radians(np.asarray(e["pool"], np.float64))
                m = np.full((len(cl), len(cand)), np.inf)
                for i, c in enumerate(cl):
                    near = (_dist(c["pts"], cand) <= radius).any(0)
                    m[i, near] = c["total"]
                out[e["image_id"]] = m
            support[radius] = out
        for radius, out in support.items():
            ceiling = sum(1 for e in photos if e["image_id"] in out and base[e["image_id"]] >= 25 and any(
                np.isfinite(out[e["image_id"]][:, j]).any() and _km(e["truth"], e["pool"][j]) < 25 for j in range(1, len(e["pool"]))))
            print(f"  R = {radius:.0f} km: ceiling (Pinpoint wrong, a supported non-top candidate < 25 km) {ceiling} (+{100 * ceiling / len(photos):.1f} pts)")
            for cap in (3, 10, 100, 10**9):
                chosen = {}
                for e in photos:
                    m = out.get(e["image_id"])
                    if m is None:
                        continue
                    votes = (m <= cap).sum(0)  # clues no more common than cap matching near each candidate
                    if votes[0] > 0 or votes[1:].max(initial=0) == 0:
                        continue
                    j = 1 + int(np.argmax(votes[1:]))
                    chosen[e["image_id"]] = e["pool"][j]
                _report(f"switch to the best-supported candidate (clue matches <= {cap if cap < 10**9 else 'any'})", photos, chosen, base)


def cooccur(tag: str, clues: str = "4b", within_km: float = 1.0, max_points: int = 1000) -> None:
    from .text_search import norm

    photos, per = _load(tag, clues)
    base = {e["image_id"]: _km(e["truth"], e["pool"][0]) for e in photos}

    def words(q: str) -> set[str]:
        return {w for w in norm(q).split() if len(w) >= 3}

    def proposals(own: list[dict], other: list[dict] | None) -> tuple[float, float, int, int] | None:
        """The spot backed by the most clues: (lat, lon, clues backing it, smallest total among them). other: pair own clues only with these."""

        best = None
        pool = [(a, b) for i, a in enumerate(own) for b in (own[i + 1:] if other is None else other)]
        for a, b in pool:
            if words(a["query"]) & words(b["query"]) or len(a["pts"]) > max_points or len(b["pts"]) > max_points:
                continue
            d = _dist(a["pts"], b["pts"])
            i, j = np.unravel_index(int(np.argmin(d)), d.shape)
            if d[i, j] > within_km:
                continue
            spot = (a["pts"][i] + b["pts"][j]) / 2
            backing = [c for c in own + (other or []) if (_dist(c["pts"], spot[None]) <= within_km).any()]
            key = (len(backing), -min(c["total"] for c in backing))
            if best is None or key > best[0]:
                best = (key, float(np.degrees(spot[0])), float(np.degrees(spot[1])))
        return None if best is None else (best[1], best[2], best[0][0], -best[0][1])

    ids = [e["image_id"] for e in photos if len(per.get(e["image_id"], [])) >= 2]
    print(f"{tag}: {len(ids)} photos with >= 2 clues with matches; pairs within {within_km} km, clues sharing no word, <= {max_points} matches each")
    shuffled = _shuffled(photos, per)
    for name, other_of in (("real clues", lambda k: None), ("shuffled control (own clue x another photo's clue)", lambda k: shuffled.get(k, []))):
        found = {}
        for e in photos:
            k = e["image_id"]
            own = per.get(k, [])
            if len(own) < (2 if other_of(k) is None else 1):
                continue
            p = proposals(own, other_of(k))
            if p is not None:
                found[k] = p
        right = sum(_km(e["truth"], found[e["image_id"]][:2]) < 25 for e in photos if e["image_id"] in found)
        fixable = sum(_km(e["truth"], found[e["image_id"]][:2]) < 25 <= base[e["image_id"]] for e in photos if e["image_id"] in found)
        print(f" {name}: proposals on {len(found)} photos, right (< 25 km) {right}, on Pinpoint's mistakes {fixable}")
        for min_backing in (2, 3):
            for cap in (10, 100, 10**9):
                chosen = {k: p[:2] for k, p in found.items() if p[2] >= min_backing and p[3] <= cap
                          and _km(p[:2], next(e["pool"][0] for e in photos if e["image_id"] == k)) >= 25}
                _report(f"switch: >= {min_backing} clues, rarest <= {cap if cap < 10**9 else 'any'} matches, >= 25 km from top-1", photos, chosen, base)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("search", "candidates", "cooccur"))
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--clues", default="4b")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.cmd == "search":
        search(args.tag, args.clues, args.workers)
    elif args.cmd == "candidates":
        candidates(args.tag, args.clues)
    else:
        cooccur(args.tag, args.clues)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
