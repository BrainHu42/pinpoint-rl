"""Reverse geocoding as a check (VALUE_PLAN.md; user, 2026-10-21): for each candidate area, collect the names around it (OSM, tools.maps
osm.db, objects within RADIUS km of the area centre) and match the photo's own text (the 4B's verbatim text + place names) against that closed
vocabulary with fuzzy matching, so misreads and partial strings still match. Lessons 118 / 140 did the forward direction (each read
phrase searched globally, exact phrase, matches kept near areas); this is the closed-vocabulary, misread-tolerant direction.

Per area: best full-string ratio, and best word ratio (photo words of >= 6 letters vs name words of >= 6 letters), rapidfuzz 0-100.
Reported: AUC right (centre < 25 km of the truth) vs wrong areas on photos with both; on photos the pipeline gets wrong while some area is
right, how often the right area outscores the pipeline's answer; an in-sample switch sweep (upper bound: thresholds chosen on these photos);
and luck (the same score at another photo's truth).

  PYTHONPATH=src:<dir with rapidfuzz> .venv/bin/python scripts/reverse_text_check.py [--tags commons26dev_setft_own dev3k_setft_own]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

from geo_search_env.experiment.text_search import haversine_km, norm
from geo_search_env.tools.maps import MapsSearch

RADIUS = 10.0
KM = {"dev3k_setft_own": "dev3k|RL policy, lambda 0", "commons26dev_setft_own": "commons26dev:setft_own|RL policy, lambda 0"}
OBS = {"dev3k_setft_own": "observe_dev3k_setft_own_4b.json", "commons26dev_setft_own": "observe_commons26dev_setft_own_4b.json"}
OUT = Path("artifacts/value")


class Names:
    def __init__(self, ms: MapsSearch):
        self.ms, self.cache = ms, {}

    def __call__(self, lat: float, lon: float) -> tuple[list[str], list[str]]:
        lat, lon = float(lat), float(lon)  # sqlite3 doesn't bind numpy float32 (the answer centres): those queries silently matched nothing
        key = (round(lat, 2), round(lon, 2))
        if key not in self.cache:
            dlat, dlon = RADIUS / 111.0, RADIUS / (111.0 * max(math.cos(math.radians(lat)), 0.05))
            rows = self.ms.osm.execute("SELECT p.variants FROM boxes b JOIN places p ON p.rowid = b.id "
                                       "WHERE b.lat_max >= ? AND b.lat_min <= ? AND b.lon_max >= ? AND b.lon_min <= ?",
                                       (lat - dlat, lat + dlat, lon - dlon, lon + dlon)).fetchall()
            names = sorted({norm(v) for (vs,) in rows if vs for v in vs.split(" | ") if len(norm(v)) >= 4})
            words = sorted({w for n in names for w in n.split() if len(w) >= 6})
            if len(self.cache) > 4000:
                self.cache.clear()
            self.cache[key] = (names, words)
        return self.cache[key]


def strings(parsed: dict) -> tuple[list[str], list[str]]:
    raw = [s for s in (parsed.get("text") or []) + (parsed.get("place_names") or []) if isinstance(s, str)]
    full = sorted({norm(s) for s in raw if len(norm(s)) >= 4 and not norm(s).replace(" ", "").isdigit()})
    words = sorted({w for s in full for w in s.split() if len(w) >= 6 and not w.isdigit()})
    return full, words


def score(full, words, vocab) -> tuple[float, float]:
    from rapidfuzz import fuzz, process

    names, nwords = vocab
    f = float(process.cdist(full, names, scorer=fuzz.ratio, workers=8, dtype=np.uint8).max()) if full and names else 0.0
    w = float(process.cdist(words, nwords, scorer=fuzz.ratio, workers=8, dtype=np.uint8).max()) if words and nwords else 0.0
    return f, w


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def run(tag: str, names: Names) -> None:
    z = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz")
    km, chosen = z[KM[tag]], z[KM[tag] + "|center"]
    photos = json.loads(Path(f"artifacts/photo_sets/{tag}/photos.json").read_text(encoding="utf-8"))
    obs = json.loads(Path("artifacts/agent", OBS[tag]).read_text(encoding="utf-8"))
    h = np.load(f"artifacts/agent/hypotheses_{tag}.npz")
    by_photo: dict[int, list[int]] = {}
    for j, p in enumerate(h["photo"]):
        by_photo.setdefault(int(p), []).append(j)
    truths = [tuple(json.loads(p["truth"]) if isinstance(p["truth"], str) else p["truth"]) for p in photos]
    rng = random.Random(0)
    rows = []
    for i, p in enumerate(photos):
        full, words = strings((obs.get(p["image_id"]) or {}).get("parsed") or {})
        if not full:
            continue
        t = truths[i]
        while True:
            o = truths[rng.randrange(len(truths))]
            if abs(o[0] - t[0]) + abs(o[1] - t[1]) > 1:
                break
        areas = []
        for j in by_photo.get(i, []):
            c = h["center"][j]
            d = float(haversine_km(t[0], t[1], np.asarray([c[0]]), np.asarray([c[1]]))[0])
            areas.append({"center": [float(c[0]), float(c[1])], "km": d, "score": score(full, words, names(*c))})
        rows.append({"i": i, "km": float(km[i]), "areas": areas, "chosen": score(full, words, names(*chosen[i])),
                     "truth": score(full, words, names(*t)), "luck": score(full, words, names(*o))})
        if len(rows) % 100 == 0:
            print(f"  {tag}: {len(rows)} photos", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"reverse_text_{tag}.json").write_text(json.dumps(rows), encoding="utf-8")
    report(tag, rows, len(photos))


def report(tag: str, rows: list[dict], n_set: int) -> None:
    print(f"\n{tag}: {n_set} photos, {len(rows)} with read text; areas per photo {np.mean([len(r['areas']) for r in rows]):.1f}")
    for k, name in ((0, "full-string ratio"), (1, "word ratio (>= 6 letters)")):
        pos = [a["score"][k] for r in rows for a in r["areas"] if a["km"] < 25 and any(b["km"] >= 25 for b in r["areas"])]
        neg = [a["score"][k] for r in rows for a in r["areas"] if a["km"] >= 25 and any(b["km"] < 25 for b in r["areas"])]
        print(f"  {name}: AUC right vs wrong areas {auc(pos, neg):.3f} ({len(pos)} / {len(neg)}); "
              f"score >= 90 at the truth {np.mean([r['truth'][k] >= 90 for r in rows]):.1%} vs another photo's truth "
              f"{np.mean([r['luck'][k] >= 90 for r in rows]):.1%}")
        fix = [r for r in rows if r["km"] >= 25 and any(a["km"] < 25 for a in r["areas"])]
        best_right = [max(a["score"][k] for a in r["areas"] if a["km"] < 25) for r in fix]
        print(f"    pipeline wrong but some area right: {len(fix)} photos ({100 * len(fix) / n_set:.1f} pts); right area above the "
              f"pipeline's answer {np.mean([b > r['chosen'][k] for b, r in zip(best_right, fix)]):.0%}, tied {np.mean([b == r['chosen'][k] for b, r in zip(best_right, fix)]):.0%}")
        best = None
        for thr in (80, 85, 90, 95, 100):
            for margin in (5, 10, 20):
                fixes = breaks = 0
                for r in rows:
                    if not r["areas"]:
                        continue
                    top = max(r["areas"], key=lambda a: a["score"][k])
                    if top["score"][k] >= thr and top["score"][k] - r["chosen"][k] >= margin:
                        fixes += r["km"] >= 25 and top["km"] < 25
                        breaks += r["km"] < 25 and top["km"] >= 25
                if best is None or fixes - breaks > best[0]:
                    best = (fixes - breaks, thr, margin, fixes, breaks)
        print(f"    best in-sample switch (score >= T, beats the answer by >= M): T {best[1]}, M {best[2]}: {best[3]} fixes / {best[4]} breaks "
              f"= {100 * best[0] / n_set:+.2f} pts (upper bound)")


def rescore_chosen(tag: str, names: Names) -> None:
    """Recompute the pipeline answer's score in a saved run (runs before the float32 fix scored it 0) and report again."""

    z = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz")
    chosen = z[KM[tag] + "|center"]
    photos = json.loads(Path(f"artifacts/photo_sets/{tag}/photos.json").read_text(encoding="utf-8"))
    obs = json.loads(Path("artifacts/agent", OBS[tag]).read_text(encoding="utf-8"))
    path = OUT / f"reverse_text_{tag}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    for r in rows:
        full, words = strings((obs.get(photos[r["i"]]["image_id"]) or {}).get("parsed") or {})
        r["chosen"] = score(full, words, names(*chosen[r["i"]]))
    path.write_text(json.dumps(rows), encoding="utf-8")
    report(tag, rows, len(photos))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=["commons26dev_setft_own", "dev3k_setft_own"])
    ap.add_argument("--rescore-chosen", action="store_true", help="fix a saved run's answer scores (float32 bug) and report")
    args = ap.parse_args()
    names = Names(MapsSearch())
    for tag in args.tags:
        (rescore_chosen if args.rescore_chosen else run)(tag, names)


if __name__ == "__main__":
    main()
