# Agreement among Pinpoint's own candidates (archive/TEXT_SEARCH_PLAN.md, locatability follow-up): Pinpoint's reranker scores each candidate on its own,
# so it never uses "5 of my 12 candidates are in the same area". Re-score each candidate by the score mass of all candidates within R km of it
# (softmax of the reranker scores at temperature T), optionally mixed with its own score, and measure the change vs Pinpoint's top-1 on dev3k
# (settings chosen on one half of the photos, counted on the other). dev sets only.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.pool_agreement --tag dev3k                 (grid, held-out choice)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.pool_agreement --fixed --tags train dev3k   (R 25 km, T 0.2, both picks)

"""Per photo: candidates (Pinpoint's pool, best first) with reranker scores; support_j = sum_k softmax(s / T)_k * [d(j, k) < R]; new score
= log support_j + lam * softmax(s / T)_j; the pick is the argmax. Reports every setting on all photos (optimistic) and the held-out choice."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Sequence

import numpy as np

PHOTO_SETS = Path("artifacts/photo_sets")


def fixed(tags: Sequence[str], radius: float = 25.0, temp: float = 0.2) -> None:
    """The setting chosen on train (R 25 km, T 0.2, lam 0) on each set: pick = the most supported candidate, or the best-scored candidate of the
    most supported area (keeps Pinpoint's fine choice inside the area), at 25 km and 1 km."""

    from .pinpoint_reference import load_pools
    from .strategy_search import _bootstrap, _haversine_km

    pools = load_pools("mp16")
    for tag in tags:
        rows = {"support": ([], []), "best in area": ([], [])}
        base25, base1 = [], []
        for e in json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8")):
            valid = pools["valid"][e["index"]]
            s = pools["scores"][e["index"]][valid].astype(np.float64)
            ll = np.asarray(e["pool"], np.float64)
            d = np.stack([_haversine_km(*p, ll) for p in ll])
            km = _haversine_km(*e["truth"], ll)
            w = np.exp((s - s.max()) / temp)
            w /= w.sum()
            c = int(np.argmax((d < radius) @ w))
            members = np.flatnonzero(d[c] < radius)
            for name, j in (("support", c), ("best in area", int(members[np.argmax(s[members])]))):
                rows[name][0].append(km[j] < 25)
                rows[name][1].append(km[j] < 1)
            base25.append(km[0] < 25)
            base1.append(km[0] < 1)
        b25, b1 = np.asarray(base25, float), np.asarray(base1, float)
        print(f"{tag}: {len(b25)} photos; Pinpoint < 25 km {b25.mean():.1%}, < 1 km {b1.mean():.1%}")
        for name, (a25, a1) in rows.items():
            a25, a1 = np.asarray(a25, float), np.asarray(a1, float)
            c25, c1 = _bootstrap(a25 - b25), _bootstrap(a1 - b1)
            print(f"  {name:13s} < 25 km {100 * c25[0]:+.1f} [{100 * c25[1]:+.1f}, {100 * c25[2]:+.1f}] (fixes {int(((a25 == 1) & (b25 == 0)).sum())}, "
                  f"breaks {int(((a25 == 0) & (b25 == 1)).sum())}); < 1 km {100 * c1[0]:+.1f} [{100 * c1[1]:+.1f}, {100 * c1[2]:+.1f}]")


def main(argv: Sequence[str] | None = None) -> int:
    from .pinpoint_reference import load_pools
    from .strategy_search import _bootstrap, _haversine_km

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--fixed", action="store_true", help="evaluate the train-chosen setting on --tags")
    parser.add_argument("--tags", nargs="*", default=("train", "dev3k"))
    args = parser.parse_args(argv)
    if args.fixed:
        fixed(args.tags)
        return 0
    photos = json.loads((PHOTO_SETS / args.tag / "photos.json").read_text(encoding="utf-8"))
    pools = load_pools("mp16")
    data = []
    for e in photos:
        valid = pools["valid"][e["index"]]
        s = pools["scores"][e["index"]][valid].astype(np.float64)
        ll = np.asarray(e["pool"], np.float64)
        d = np.stack([_haversine_km(*p, ll) for p in ll])
        right = _haversine_km(*e["truth"], ll) < 25
        data.append((s, d, right))
    base = np.asarray([r[0] for _, _, r in data], float)
    oracle = np.asarray([r.any() for _, _, r in data], float)
    print(f"{args.tag}: {len(photos)} photos; Pinpoint < 25 km {base.mean():.1%}, pool oracle {oracle.mean():.1%}")

    def picks(radius: float, temp: float, lam: float, idx) -> np.ndarray:
        out = []
        for i in idx:
            s, d, right = data[i]
            w = np.exp((s - s.max()) / temp)
            w /= w.sum()
            score = np.log((d < radius) @ w) + lam * w
            out.append(right[int(np.argmax(score))])
        return np.asarray(out, float)

    grid = list(itertools.product((5.0, 25.0, 100.0, 300.0), (0.05, 0.1, 0.2, 0.5), (0.0, 1.0, 3.0, 10.0)))
    allidx = np.arange(len(data))
    results = sorted(((picks(*g, allidx).mean() - base.mean(), g) for g in grid), reverse=True)
    print("  best settings on all photos (optimistic):")
    for delta, (r, t, lam) in results[:6]:
        print(f"    R {r:5.0f} km, T {t:4.2f}, lam {lam:4.1f}: {100 * delta:+.1f} pts")
    fold = np.random.default_rng(0).permutation(len(data)) % 2
    chosen = np.zeros(len(data))
    for half in (0, 1):
        fit, held = allidx[fold != half], allidx[fold == half]
        g = max(grid, key=lambda g: picks(*g, fit).mean() - base[fit].mean())
        chosen[held] = picks(*g, held)
        print(f"  half {half}: chosen R {g[0]:.0f} km, T {g[1]}, lam {g[2]}")
    c = _bootstrap(chosen - base)
    print(f"  held-out: < 25 km {chosen.mean():.1%} ({100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]); fixes {int(((chosen == 1) & (base == 0)).sum())}, "
          f"breaks {int(((chosen == 0) & (base == 1)).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
