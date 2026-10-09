# Building on lesson 68 (agreement among Pinpoint's candidates): two cheap variants measured on train (35k, the setting source) and dev3k.
#   multiscale: coarse to fine, as an expert narrows down: the country with the most softmax score mass, then the region with the most mass
#               inside it, then lesson 68's agreement pick among the candidates of that region (vs agreement at 25 km alone).
#   gated:      the clue judge's override (an accepted read-text clue answers with its own coordinate) only when the agreement area's support
#               is below s: a clue pointing away from a confident area is probably wrong (dev3k only: the judge was trained on train).
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.area_rules multiscale --tags train dev3k
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.area_rules gated --tag dev3k

"""Labels are the Overture (country, region) of each pool candidate (labels.json); unlabelled candidates form their own group. Settings:
pinpoint_tool.RADIUS_KM / TEMP (chosen on train). Changes are vs Pinpoint's top-1, % within 25 km with a bootstrap interval; gated thresholds
are chosen on one half of the photos and counted on the other."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Sequence

import numpy as np

PHOTO_SETS = Path("artifacts/photo_sets")


def _load(tag: str) -> list[dict]:
    from .pinpoint_reference import load_pools
    from .strategy_search import _haversine_km

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    labels = {p["image_id"]: p for p in json.loads((PHOTO_SETS / tag / "labels.json").read_text(encoding="utf-8"))}
    pools = load_pools("mp16")
    out = []
    for e in photos:
        valid = pools["valid"][e["index"]]
        pool_labels = labels[e["image_id"]]["pool"]
        out.append({"image_id": e["image_id"], "scores": pools["scores"][e["index"]][valid].astype(np.float64), "latlon": e["pool"],
                    "km": _haversine_km(*e["truth"], np.asarray(e["pool"])),
                    "country": [c.get("country") or f"?{i}" for i, c in enumerate(pool_labels)],
                    "region": [f"{c.get('country')}|{c.get('region')}" if c.get("country") and c.get("region") else f"?{i}" for i, c in enumerate(pool_labels)]})
    return out


def _report(name: str, rows: list[dict], picks: list[int]) -> None:
    from .strategy_search import _bootstrap

    gain = np.asarray([int(r["km"][j] < 25) - int(r["km"][0] < 25) for r, j in zip(rows, picks)], float)
    g1 = np.asarray([int(r["km"][j] < 1) - int(r["km"][0] < 1) for r, j in zip(rows, picks)], float)
    c, c1 = _bootstrap(gain), _bootstrap(g1)
    print(f"  {name:44s} < 25 km {100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}] (fixes {int((gain > 0).sum())}, breaks {int((gain < 0).sum())}); "
          f"< 1 km {100 * c1[0]:+.1f}")


def multiscale(tags: Sequence[str]) -> None:
    from .pinpoint_tool import TEMP, pick

    for tag in tags:
        rows = _load(tag)
        print(f"{tag}: {len(rows)} photos; Pinpoint < 25 km {np.mean([r['km'][0] < 25 for r in rows]):.1%}")
        variants = collections.defaultdict(list)
        for r in rows:
            s = r["scores"]
            w = np.exp((s - s.max()) / TEMP)
            w /= w.sum()
            variants["agreement at 25 km (lesson 68)"].append(pick(s, r["latlon"], "support"))
            for level in ("country", "region"):
                mass = collections.Counter()
                for g, x in zip(r[level], w):
                    mass[g] += x
                top = mass.most_common(1)[0][0]
                idx = [i for i, g in enumerate(r[level]) if g == top]
                sub = pick(s[idx], [r["latlon"][i] for i in idx], "support")
                variants[f"top {level} by mass, then agreement inside"].append(idx[sub])
                variants[f"top {level} by mass, then Pinpoint's best inside"].append(max(idx, key=lambda i: s[i]))
            # country, then region inside it, then agreement inside that
            cmass = collections.Counter()
            for g, x in zip(r["country"], w):
                cmass[g] += x
            ci = [i for i, g in enumerate(r["country"]) if g == cmass.most_common(1)[0][0]]
            rmass = collections.Counter()
            for i in ci:
                rmass[r["region"][i]] += w[i]
            ri = [i for i in ci if r["region"][i] == rmass.most_common(1)[0][0]]
            variants["country -> region -> agreement"].append(ri[pick(s[ri], [r["latlon"][i] for i in ri], "support")])
        for name, picks in variants.items():
            _report(name, rows, picks)


def gated(tag: str, judge: str = "ftvsall4b") -> None:
    from .clue_judge import _joined
    from .pinpoint_tool import areas
    from .strategy_search import _bootstrap

    rows = {r["image_id"]: r for r in _load(tag)}
    first = {}
    for k, r in rows.items():
        a = areas(r["scores"], r["latlon"])[0]
        first[k] = (r["km"][a.founder], a.support)
    clues = [c for c in _joined(tag, judge, "4b") if c["differs"] and not c["source"].startswith("query")]
    ids = sorted(rows)
    fold = {k: i % 2 for i, k in enumerate(np.random.default_rng(0).permutation(ids))}
    base = {k: rows[k]["km"][0] for k in ids}

    def answer(t: float, smax: float, keep: set[str]) -> dict[str, float]:
        out = {k: first[k][0] for k in keep}
        best: dict[str, tuple[float, float]] = {}
        for c in clues:
            k = c["image_id"]
            if k in keep and c["p_yes"] >= t and first[k][1] < smax and (k not in best or c["p_yes"] > best[k][0]):
                best[k] = (c["p_yes"], c["km"])
        out |= {k: v[1] for k, v in best.items()}
        return out

    grid = [(t, s) for t in (0.3, 0.5, 0.7, 0.8, 0.9) for s in (0.3, 0.5, 0.7, 0.9, 1.01)]
    chosen: dict[str, float] = {}
    for half in (0, 1):
        fit = {k for k in ids if fold[k] != half}
        held = {k for k in ids if fold[k] == half}
        g = max(grid, key=lambda g: sum(int(d < 25) for d in answer(*g, fit).values()))
        print(f"  half {half}: judge >= {g[0]}, area support < {g[1]}")
        chosen |= answer(*g, held)
    gain = np.asarray([int(chosen[k] < 25) - int(base[k] < 25) for k in ids], float)
    c = _bootstrap(gain)
    print(f"{tag}: agreement + clue judge gated by area support (thresholds held out): < 25 km {100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}] "
          f"(fixes {int((gain > 0).sum())}, breaks {int((gain < 0).sum())}); ungated +2.6 (pinpoint_tool combine)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("multiscale", "gated"))
    parser.add_argument("--tags", nargs="*", default=("train", "dev3k"))
    parser.add_argument("--tag", default="dev3k")
    args = parser.parse_args(argv)
    multiscale(args.tags) if args.cmd == "multiscale" else gated(args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
