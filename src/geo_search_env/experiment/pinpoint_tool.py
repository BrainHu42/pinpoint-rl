# Pinpoint as the agent's retrieval tool (TEXT_SEARCH_PLAN.md target design + lesson 68): instead of a ranked list of single candidates, the
# tool returns areas, each backed by the candidates within RADIUS_KM of its best-supported member and weighted by their reranker score mass
# (softmax at TEMP). Area 1's pick is lesson 68's agreement pick (+1.0 pts on train over Pinpoint's top-1). `--restrict` style use (a country /
# region / box) goes through pinpoint_constrained / models.pinpoint_reranker. dev sets only for evaluation.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.pinpoint_tool show --tag dev3k --n 3     (what the agent would read)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.pinpoint_tool combine --tag dev3k     (agreement + the clue judge)

"""areas(): greedy: the candidate with the most support (score mass within RADIUS_KM) founds area 1 with all candidates within RADIUS_KM of it;
those are removed and the next area is founded from the rest. Each area: founding candidate, best-scored member, support, member count,
place label. text(): the agent-readable summary. pick(): the area-1 answer ("support": the founding candidate, lesson 68's best setting;
"best": Pinpoint's best-scored member of area 1, which keeps 1 km accuracy)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

PHOTO_SETS = Path("artifacts/photo_sets")
RADIUS_KM, TEMP = 25.0, 0.2  # chosen on train (lesson 68)


@dataclass
class Area:
    founder: int  # candidate index (Pinpoint order, 0 = its top-1)
    best: int  # best-scored member
    members: list[int]
    support: float  # share of the pool's softmax score mass inside the area
    label: str = ""


def _km(latlon: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(latlon[:, 0]), np.radians(latlon[:, 1])
    h = (np.sin((lat[:, None] - lat[None]) / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None]) * np.sin((lon[:, None] - lon[None]) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def areas(scores: Sequence[float], latlon: Sequence[Sequence[float]], labels: Sequence[str] | None = None,
          radius: float = RADIUS_KM, temp: float = TEMP) -> list[Area]:
    s = np.asarray(scores, np.float64)
    w = np.exp((s - s.max()) / temp)
    w /= w.sum()
    near = _km(np.asarray(latlon, np.float64)) < radius
    alive = np.ones(len(s), bool)
    out = []
    while alive.any():
        support = (near & alive[None]) @ w
        support[~alive] = -1
        founder = int(np.argmax(support))
        members = [int(j) for j in np.flatnonzero(near[founder] & alive)]
        best = max(members, key=lambda j: s[j])
        out.append(Area(founder, best, members, float(w[members].sum()), labels[best] if labels else ""))
        alive[members] = False
    return out


def pick(scores: Sequence[float], latlon: Sequence[Sequence[float]], mode: str = "support") -> int:
    a = areas(scores, latlon)[0]
    return a.founder if mode == "support" else a.best


def text(scores: Sequence[float], latlon: Sequence[Sequence[float]], labels: Sequence[str], top: int = 4) -> str:
    """What the agent reads from the Pinpoint tool."""

    out = areas(scores, latlon, labels)
    lines = [f"pinpoint(): {len(scores)} visually similar geotagged photos, grouped into {len(out)} areas (share of retrieval confidence):"]
    for i, a in enumerate(out[:top], 1):
        la, lo = latlon[a.best]
        lines.append(f"  {i}. {a.label or '?'} ({la:.3f}, {lo:.3f}): {a.support:.0%}, {len(a.members)} of {len(scores)} candidates")
    return "\n".join(lines)


def combine(tag: str, judge: str = "ftvsall4b") -> None:
    """Agreement pick (both modes) combined with the fine-tuned clue judge on read-text clues (an accepted clue whose region differs from
    Pinpoint's top-1 region answers with its own coordinate); threshold chosen on one half of the photos and counted on the other."""

    from .clue_judge import _joined
    from .pinpoint_reference import load_pools
    from .strategy_search import _bootstrap, _haversine_km

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    pools = load_pools("mp16")
    km = {e["image_id"]: _haversine_km(*e["truth"], np.asarray(e["pool"])) for e in photos}
    base = {k: v[0] for k, v in km.items()}
    agree = {}
    for mode in ("support", "best"):
        agree[mode] = {}
        for e in photos:
            valid = pools["valid"][e["index"]]
            agree[mode][e["image_id"]] = km[e["image_id"]][pick(pools["scores"][e["index"]][valid], e["pool"], mode)]
    clues = [r for r in _joined(tag, judge, "4b") if r["differs"] and not r["source"].startswith("query")]
    ids = sorted(base)
    fold = {k: i % 2 for i, k in enumerate(np.random.default_rng(0).permutation(ids))}

    def judged(t: float, keep: set[str]) -> dict[str, float]:
        out: dict[str, tuple[float, float]] = {}
        for r in clues:
            if r["p_yes"] >= t and r["image_id"] in keep and (r["image_id"] not in out or r["p_yes"] > out[r["image_id"]][0]):
                out[r["image_id"]] = (r["p_yes"], r["km"])
        return {k: v[1] for k, v in out.items()}

    def show(name: str, answer: dict[str, float]) -> None:
        gain = np.asarray([int(answer[k] < 25) - int(base[k] < 25) for k in ids], float)
        c = _bootstrap(gain)
        print(f"  {name:52s} < 25 km {100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]; fixes {int((gain > 0).sum())}, breaks {int((gain < 0).sum())}")

    print(f"{tag}: {len(ids)} photos; Pinpoint < 25 km {np.mean([base[k] < 25 for k in ids]):.1%}")
    for mode in ("support", "best"):
        show(f"agreement ({mode})", agree[mode])
    for start_name, start in (("Pinpoint", base), ("agreement (support)", agree["support"]), ("agreement (best)", agree["best"])):
        answer = dict(start)
        for half in (0, 1):
            fit = {k for k in ids if fold[k] != half}
            held = {k for k in ids if fold[k] == half}
            t = max((0.3, 0.5, 0.7, 0.8, 0.9), key=lambda t: sum(int(d < 25) - int(start[k] < 25) for k, d in judged(t, fit).items()))
            answer |= judged(t, held)
        show(f"{start_name} + clue judge (threshold held out)", answer)


def main(argv: Sequence[str] | None = None) -> int:
    from .pinpoint_reference import load_pools

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("show", "combine"))
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args(argv)
    if args.cmd == "combine":
        combine(args.tag)
        return 0
    photos = json.loads((PHOTO_SETS / args.tag / "photos.json").read_text(encoding="utf-8"))
    labels = {p["image_id"]: p for p in json.loads((PHOTO_SETS / args.tag / "labels.json").read_text(encoding="utf-8"))}
    pools = load_pools("mp16")
    for e in photos[:args.n]:
        valid = pools["valid"][e["index"]]
        names = [", ".join(x for x in (c.get("locality"), c.get("region"), c.get("country")) if x) for c in labels[e["image_id"]]["pool"]]
        print(e["image_id"], "truth", e["truth"])
        print(text(pools["scores"][e["index"]][valid], e["pool"], names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
