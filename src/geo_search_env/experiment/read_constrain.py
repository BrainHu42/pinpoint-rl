# Ambiguity-aware text search for the read chain (lesson 115: on commons26 dev ~15% of photos have text naming the place but no search result
# < 25 km; common names resolve to the wrong one). region_evidence keeps only each phrase's top cluster; this keeps all clusters (up to 8, with
# country codes) so cluster choice can use the 4B's country guess, Pinpoint's areas and agreement between a photo's phrases.
# Tuned on commons26 dev (user, 2026-10-16); dev3k for regressions.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.read_constrain search --tag commons26     (CPU, ~10 min)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.read_constrain rules --tag commons26 --split dev

"""search: every phrase of region_evidence (4B text, place names, queries), all clusters -> artifacts/agent/clusters_{tag}_4b.json.
rules: per selection rule, the share of photos with a chosen cluster < 25 km (recall) and the precision of chosen clusters."""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

OUT = Path("artifacts/agent")
PHOTO_SETS = Path("artifacts/photo_sets")


def _path(tag: str) -> Path:
    return OUT / f"clusters_{tag}_4b.json"


def _chunk(args) -> list[dict[str, Any]]:
    from .text_search import TextSearch

    photos, phrases, owns = args
    tool, rows = TextSearch(), []
    for e in photos:
        for q, src in phrases.get(e["image_id"], {}).items():
            r = tool.search(q, exclude=owns[e["image_id"]])
            rows.append({"image_id": e["image_id"], "query": q, "source": src, "kind": r.kind, "hint": r.hint, "loose": r.loose,
                         "clusters": [{"lat": c.lat, "lon": c.lon, "cc": c.label.rsplit(", ", 1)[-1] if ", " in c.label else "",
                                       "weight": c.weight, "share": c.share, "exact": c.exact, "places": c.places, "users": c.users} for c in r.clusters]})
    return rows


def search(tag: str, workers: int = 8) -> None:
    import multiprocessing
    import os

    from .region_evidence import _owns, _raw_phrases

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    phrases = _raw_phrases(tag, "4b")
    photos = [e for e in photos if e["image_id"] in phrases]
    owns = _owns(photos)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    chunks = [photos[i:i + 200] for i in range(0, len(photos), 200)]
    rows = []
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for part in pool.imap(_chunk, [(c, {e["image_id"]: phrases[e["image_id"]] for e in c}, {e["image_id"]: owns[e["image_id"]] for e in c}) for c in chunks]):
            rows += part
    _path(tag).write_text(json.dumps(rows) + "\n", encoding="utf-8")
    print(f"{len(rows)} searches for {len(photos)} photos -> {_path(tag)}")


GENERIC = set("""the of and de la le les du des del della di da do dos das der die und von zu am im an in on at by for to a
monument museum station railway church cathedral chapel temple mosque shrine square street road avenue park garden bridge tower castle palace
center centre shopping mall market hotel restaurant cafe bar school university college hospital library hall house building memorial statue
fountain gate lake river mountain beach island port harbour harbor airport bus stop city town village old new north south east west central
national state public main saint st plaza place via rue strasse calle piazza location photo view interior exterior entrance sign front
festival fest event collection exhibition gallery theatre theater cinema office police fire community club""".split())


def _spans(phrase: str, max_len: int = 3) -> list[str]:
    """Contiguous word spans of a phrase (1-3 words) that aren't only generic words: what a person would type to search a long phrase in parts."""

    words = [w.strip(".,;:!?()[]\"'«»“”") for w in phrase.split()]
    words = [w for w in words if w]
    out = []
    for n in range(1, max_len + 1):
        for i in range(len(words) - n + 1):
            span = words[i:i + n]
            if all(w.lower() in GENERIC or len(w) < 3 for w in span):
                continue
            out.append(" ".join(span))
    return out


def _span_chunk(args) -> list[dict[str, Any]]:
    from .text_search import TextSearch

    items, owns = args
    tool, rows = TextSearch(), []
    for image_id, span in items:
        r = tool.search(span, exclude=owns[image_id])
        rows.append({"image_id": image_id, "query": span, "kind": r.kind, "loose": r.loose, "matches": int(sum(r.matches.values())),
                     "clusters": [{"lat": c.lat, "lon": c.lon, "cc": c.label.rsplit(", ", 1)[-1] if ", " in c.label else "", "weight": c.weight,
                                   "share": c.share, "exact": c.exact, "places": c.places, "users": c.users} for c in r.clusters[:8]]})
    return rows


def spans(tag: str, split: str | None = "dev", workers: int = 8) -> None:
    """Every 1-3 word span of every phrase (not only generic words), searched -> artifacts/agent/spans_{tag}_4b{_split}.json."""

    import multiprocessing
    import os

    from .region_evidence import _owns, _raw_phrases

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    photos = [e for e in photos if split is None or e.get("split") == split]
    phrases = _raw_phrases(tag, "4b")
    photos = [e for e in photos if e["image_id"] in phrases]
    owns = _owns(photos)
    items = sorted({(e["image_id"], s) for e in photos for q in phrases[e["image_id"]] for s in _spans(q)})
    print(f"{len(items)} spans for {len(photos)} photos", flush=True)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    chunks = [items[i:i + 500] for i in range(0, len(items), 500)]
    rows = []
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for k, part in enumerate(pool.imap(_span_chunk, [(c, {i: owns[i] for i, _ in c}) for c in chunks])):
            rows += part
            if k % 20 == 0:
                print(f"  {len(rows)} / {len(items)}", flush=True)
    dst = OUT / f"spans_{tag}_4b{'_' + split if split else ''}.json"
    dst.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    print(f"-> {dst}")


def _country_codes() -> dict[str, str]:
    out = {}
    for line in Path("data/pinpoint/geonames/countryInfo.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            f = line.split("\t")
            out[f[4].lower()] = f[0]
    out |= {"united states of america": "US", "usa": "US", "uk": "GB", "england": "GB", "scotland": "GB", "wales": "GB", "northern ireland": "GB",
            "russia": "RU", "south korea": "KR", "korea": "KR", "czechia": "CZ", "czech republic": "CZ", "vietnam": "VN", "türkiye": "TR",
            "turkey": "TR", "taiwan": "TW", "the netherlands": "NL", "holland": "NL", "iran": "IR", "syria": "SY", "laos": "LA"}
    return out


def rules(tag: str, split: str | None = "dev", ckpt: str = "setft_own") -> None:
    from .names_feature import _names  # noqa: F401  (cities5k; area country codes below)
    from .refine_headroom import _hav

    photos = json.loads((PHOTO_SETS / f"{tag}_{ckpt}" / "photos.json").read_text(encoding="utf-8"))
    keep = {e["image_id"] for e in photos if split is None or e.get("split") == split}
    pidx = {e["image_id"]: k for k, e in enumerate(photos)}
    truth = {e["image_id"]: e["truth"] for e in photos}
    cc_of = _country_codes()
    guess = {k: cc_of.get(str((v.get("parsed") or {}).get("country") or "").strip().lower())
             for k, v in json.loads((OUT / f"observe_{tag}_4b.json").read_text(encoding="utf-8")).items()}
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{ckpt}.npz")
    areas = collections.defaultdict(list)
    for j, p in enumerate(zm["photo"]):
        if zm["source"][j] == 0:
            areas[int(p)].append(zm["center"][j].astype(np.float64))
    area_cc = _area_countries(zm)
    rows = [r for r in json.loads(_path(tag).read_text(encoding="utf-8")) if r["image_id"] in keep]
    by_photo = collections.defaultdict(list)
    for r in rows:
        by_photo[r["image_id"]].append(r)

    def km(i, c):
        return float(_hav(truth[i][0], truth[i][1], c["lat"], c["lon"]))

    def near_area(i, c, radius):
        a = areas.get(pidx[i], [])
        return bool(a) and min(float(_hav(c["lat"], c["lon"], x[0], x[1])) for x in a) < radius

    def pick(rule, i, r):
        cs = r["clusters"]
        if not cs:
            return None
        if rule == "top":
            return cs[0]
        if rule == "top, share>=0.5":
            return cs[0] if cs[0]["share"] >= 0.5 else None
        if rule == "4B country":
            g = guess.get(i)
            return next((c for c in cs if c["cc"] == g), cs[0]) if g else cs[0]
        if rule == "4B country only":
            g = guess.get(i)
            return next((c for c in cs if c["cc"] == g), None) if g else None
        if rule == "area countries":
            ccs = area_cc.get(pidx[i], set())
            return next((c for c in cs if c["cc"] in ccs), cs[0])
        if rule == "near an area (100 km)":
            return next((c for c in cs if near_area(i, c, 100)), cs[0])
        if rule == "4B country, then areas":
            g, ccs = guess.get(i), area_cc.get(pidx[i], set())
            return next((c for c in cs if g and c["cc"] == g), next((c for c in cs if c["cc"] in ccs), cs[0]))
        raise ValueError(rule)

    def agree_pick(i):
        """Clusters supported by more than one phrase of the photo (within 25 km of each other), best summed share."""
        pts = [(c, k) for k, r in enumerate(by_photo[i]) for c in r["clusters"]]
        best, score = None, 0.0
        for c, k in pts:
            s = sum(d["share"] for d, m in pts if m != k and float(_hav(c["lat"], c["lon"], d["lat"], d["lon"])) < 25)
            if s > score:
                best, score = c, s
        return best

    n = len(keep)
    oracle = sum(any(km(i, c) < 25 for r in by_photo[i] for c in r["clusters"]) for i in keep)
    print(f"{tag} {split}: {n} photos, {len(rows)} searches; some cluster of some phrase < 25 km: {oracle / n:.1%}; 4B country right "
          f"{np.mean([guess.get(i) == _true_cc(i) for i in keep]) if _TRUE_CC else float('nan'):.1%}")
    for rule in ("top", "top, share>=0.5", "4B country", "4B country only", "area countries", "near an area (100 km)", "4B country, then areas"):
        hit, chosen, right = 0, 0, 0
        for i in keep:
            ch = [c for c in (pick(rule, i, r) for r in by_photo[i]) if c is not None]
            chosen += len(ch)
            good = [km(i, c) < 25 for c in ch]
            right += sum(good)
            hit += any(good)
        print(f"  {rule:26s} photos with a chosen cluster < 25 km {hit / n:5.1%}   precision {right / max(chosen, 1):5.1%} of {chosen}")
    ag = [agree_pick(i) for i in keep]
    print(f"  {'cross-phrase agreement':26s} photos whose best agreed cluster < 25 km {np.mean([a is not None and km(i, a) < 25 for i, a in zip(keep, ag)]):5.1%}"
          f"   (photos with one: {np.mean([a is not None for a in ag]):5.1%})")


_TRUE_CC: dict[str, str] = {}


def _true_cc(i: str) -> str:
    return _TRUE_CC.get(i, "")


def _area_countries(zm) -> dict[int, set[str]]:
    """Country codes of each photo's Pinpoint areas (nearest GeoNames town of >= 5,000 people)."""

    from scipy.spatial import cKDTree

    lat, lon, cc = [], [], []
    for line in (OUT / "cities5k.tsv").read_text(encoding="utf-8").splitlines():
        _, a, b, c = line.split("\t")
        lat.append(float(a))
        lon.append(float(b))
        cc.append(c)

    def xyz(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)

    sel = zm["source"] == 0
    _, idx = cKDTree(xyz(np.asarray(lat), np.asarray(lon))).query(xyz(zm["center"][sel, 0].astype(np.float64), zm["center"][sel, 1].astype(np.float64)))
    out: dict[int, set[str]] = collections.defaultdict(set)
    for p, k in zip(zm["photo"][sel], idx):
        out[int(p)].add(cc[k])
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("search", "rules", "spans"))
    parser.add_argument("--tag", default="commons26")
    parser.add_argument("--split", default="dev", help="rules: commons26 split (dev / test); 'all' for every photo")
    args = parser.parse_args(argv)
    if args.cmd == "search":
        search(args.tag)
    elif args.cmd == "spans":
        spans(args.tag, None if args.split == "all" else args.split)
    else:
        if args.tag == "commons26":
            _TRUE_CC.update({r["IMG_ID"]: r["country"] for r in csv.DictReader(open("data/pinpoint/commons26/release/benchmark.csv", encoding="utf-8"))})
        rules(args.tag, None if args.split == "all" else args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
