# Geolocation as hypothesis testing (HYPOTHESIS_PLAN.md), E2: hand-built comparators over the hypotheses of `hypotheses.py`. Each photo has up to
# K Pinpoint areas (+ clue places on dev3k), each searched with Pinpoint restricted to a ~25 km box; a comparator picks one hypothesis and the
# answer is its restricted top-1. Evidence per hypothesis and baselines:
#   area 1 (restricted top-1 of the best-supported area) ~ lesson 68's agreement;  max restricted top-1 score (the rule that failed before);
#   restricted score mass sum exp(s / T) (unnormalised, so comparable across hypotheses);  PIGEON-style: hypothesis prior (pool support) x
#   softmax of the restricted top-1 score;  mass x support^a / density^b (popularity-corrected, distinct photographers);  Vo et al. 2017 KDE
#   over the top-100 neighbours (deep_votes, neighbours alone).
# Parameters are chosen on one half of the photos and counted on the other; on train when its hypotheses exist. dev sets only.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.hypothesis_compare features --tag dev3k   (CPU; caches features)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.hypothesis_compare report --tag dev3k

"""features: per hypothesis: restricted top-1 / top-3 scores, score mass at T in (0.1, 0.2, 0.5), number of restricted candidates, distinct
          photographers among them, gallery density (index photos within the box), pool support, source, rank; whether its restricted top-1
          is < 25 / 1 km from the truth -> artifacts/agent/hypothesis_features_{tag}{_clues}.npz.
report:   % < 25 km (and < 1 km) per comparator vs Pinpoint's top-1 and the hypothesis oracle."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

OUT = Path("artifacts/agent")
PHOTO_SETS = Path("artifacts/photo_sets")
AUTHORS = OUT / "mp16_author_by_cache_row.npy"
BOX_KM = 25.0


def _cache_authors() -> np.ndarray:
    """Author id per MP16 embedding-cache row (cached)."""

    if AUTHORS.exists():
        return np.load(AUTHORS)
    from .strategy_search import MP16_CSV, MP16_EMBED

    vocab: dict[str, int] = {}
    with MP16_CSV.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        col = next(reader).index("AUTHOR")
        by_csv = np.asarray([vocab.setdefault(r[col], len(vocab)) for r in reader], np.int64)
    out = by_csv[np.fromfile(MP16_EMBED / "row_index.i64.bin", dtype=np.int64)]
    np.save(AUTHORS, out)
    return out


def _paths(tag: str, clues: bool) -> tuple[Path, Path]:
    s = "_clues" if clues else ""
    return OUT / f"hypotheses_{tag}{s}.npz", OUT / f"hypothesis_features_{tag}{s}.npz"


def features(tag: str, clues: bool) -> None:
    from scipy.spatial import cKDTree

    from ..models.pinpoint_reranker import _discover_index, _unit_xyz
    from .strategy_search import _haversine_km

    src, dst = _paths(tag, clues)
    z = dict(np.load(src))
    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    truth = np.asarray([e["truth"] for e in photos], np.float64)
    index_dir = _discover_index()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    count = manifest["shapes"]["raw_image_embeddings"][0]
    index_latlon = np.array(np.memmap(index_dir / manifest["files"]["latlon_deg"], dtype=np.float32, mode="r", shape=(count, 2)), np.float64)
    cache_rows = np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
    authors = _cache_authors()[cache_rows]
    tree = cKDTree(_unit_xyz(index_latlon))
    density = tree.query_ball_point(_unit_xyz(z["center"]), 2 * math.sin(BOX_KM / 6371.0 / 2), return_length=True)
    s = z["scores"]
    valid = z["rows"] >= 0
    n = len(s)
    f = {"photo": z["photo"], "source": z["source"], "rank": z["rank"], "support": z["support"], "density": np.asarray(density, np.float64),
         "n_cand": valid.sum(1), "top1": np.where(valid[:, 0], s[:, 0], -np.inf),
         "top3": np.asarray([s[i, valid[i]][:3].mean() if valid[i].any() else -np.inf for i in range(n)])}
    for t in (0.1, 0.2, 0.5):
        f[f"mass_{t}"] = np.where(valid, np.exp(np.where(valid, s, 0) / t), 0).sum(1)
    f["photographers"] = np.asarray([len(set(authors[z["rows"][i, valid[i]]].tolist())) for i in range(n)])
    km = np.full(n, np.inf)
    for i in range(n):
        if valid[i, 0]:
            km[i] = _haversine_km(*truth[z["photo"][i]], z["latlon"][i, :1].astype(np.float64))[0]
    f["km"] = km
    f["base_km"] = np.asarray([_haversine_km(*e["truth"], np.asarray([e["pool"][0]]))[0] for e in photos])
    np.savez(dst, **f)
    print(f"{n} hypotheses for {len(photos)} photos -> {dst}")


def report(tag: str, clues: bool) -> None:
    from .strategy_search import _bootstrap

    _, path = _paths(tag, clues)
    f = dict(np.load(path))
    n_photos = len(f["base_km"])
    groups = [np.flatnonzero(f["photo"] == p) for p in range(n_photos)]
    base = f["base_km"] < 25
    ok = f["km"] < 25
    oracle = np.asarray([ok[g].any() if len(g) else False for g in groups])

    def evaluate(score: np.ndarray, members: Sequence[int]) -> np.ndarray:
        out = np.empty(len(members))
        for j, p in enumerate(members):
            g = groups[p]
            out[j] = ok[g[np.argmax(score[g])]] if len(g) and np.isfinite(score[g]).any() else base[p]
        return out

    lg = lambda x: np.log(np.maximum(x, 1e-12))
    comparators: dict[str, tuple[Callable[..., np.ndarray], list[tuple]]] = {
        "area 1 (best pool support)": (lambda: -f["rank"] - 10 * (f["source"] != 0), [()]),
        "max restricted top-1 score": (lambda: f["top1"], [()]),
        "max restricted top-3 mean": (lambda: f["top3"], [()]),
        "restricted score mass": (lambda t: f[f"mass_{t}"], [(0.1,), (0.2,), (0.5,)]),
        "PIGEON-style: support x softmax(top-1 / tau)": (lambda tau: lg(f["support"]) + f["top1"] / tau, [(0.02,), (0.05,), (0.1,), (0.2,)]),
        "mass x support^a / density^b x photographers^c": (
            lambda t, a, b, c: lg(f[f"mass_{t}"]) + a * lg(f["support"]) - b * lg(f["density"] + 1) + c * lg(f["photographers"]),
            list(itertools.product((0.2,), (0.0, 0.5, 1.0, 2.0), (0.0, 0.1, 0.25, 0.5), (0.0, 0.5, 1.0)))),
    }
    print(f"{tag}{' (+ clue hypotheses)' if clues else ''}: {n_photos} photos, {len(f['photo'])} hypotheses; Pinpoint < 25 km {base.mean():.1%}; "
          f"hypothesis oracle {oracle.mean():.1%}")
    fold = np.random.default_rng(0).permutation(n_photos) % 2
    for name, (fn, grid) in comparators.items():
        chosen = np.empty(n_photos)
        picks = []
        for half in (0, 1):
            fit, held = np.flatnonzero(fold != half), np.flatnonzero(fold == half)
            g = max(grid, key=lambda g: evaluate(fn(*g), fit).mean())
            picks.append(g)
            chosen[held] = evaluate(fn(*g), held)
        c = _bootstrap(chosen - base)
        print(f"  {name:48s} < 25 km {chosen.mean():.1%} ({100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]); "
              f"fixes {int(((chosen == 1) & ~base).sum())}, breaks {int(((chosen == 0) & base).sum())}" + (f"; chosen {picks}" if grid != [()] else ""))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("features", "report"))
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--clues", action="store_true")
    args = parser.parse_args(argv)
    features(args.tag, args.clues) if args.cmd == "features" else report(args.tag, args.clues)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
