# Geolocation as hypothesis testing (HYPOTHESIS_PLAN.md), E1: the hypothesis oracle. Per photo, K area hypotheses: Pinpoint's agreement areas
# (pinpoint_tool.areas, by support) plus, optionally, places of accepted read-text clues (fine-tuned judge). For each (photo, hypothesis),
# Pinpoint's retrieval + reranker restricted to a box of BOX_KM around the hypothesis. How often does some hypothesis's restricted top-1 land
# < 25 km from the truth (K = 1..5), against Pinpoint, agreement and the pool oracle? dev sets only.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.hypotheses run --tag dev3k [--k 5] [--clues]   (GPU, one index pass)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.hypotheses oracle --tag dev3k

"""run:    hypotheses and their restricted results -> artifacts/agent/hypotheses_{tag}.npz: per (photo, hypothesis) the photo position, source
        (0 Pinpoint area, 1 clue), area rank, centre, area support, restricted candidates (lat / lon, reranker scores, 9 retrieval features,
        index rows).
oracle: % < 25 km when the best hypothesis is picked (its restricted top-1, or any restricted candidate), by K, with and without clue
        hypotheses; Pinpoint, agreement and the pool oracle for reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
PHOTO_SETS = Path("artifacts/photo_sets")
SFT_ROOT = Path("artifacts/sft")
NVME_INDEX = Path("/opt/dlami/nvme/pinpoint/attention_retrieval_index")
BOX_KM = 25.0
JUDGE = "ftvsall4b"


def _hypotheses(tag: str, k: int, clues: bool) -> list[dict]:
    from .pinpoint_reference import load_pools
    from .pinpoint_tool import areas

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    pools = load_pools("mp16")
    accepted: dict[str, list[tuple[float, float, float]]] = {}
    if clues:
        from .clue_judge import _joined

        for r in _joined(tag, JUDGE, "4b"):
            if r["differs"] and not r["source"].startswith("query") and r["p_yes"] >= 0.5:
                accepted.setdefault(r["image_id"], []).append((r["p_yes"], r["lat"], r["lon"]))
    out = []
    for p, e in enumerate(photos):
        valid = pools["valid"][e["index"]]
        for rank, a in enumerate(areas(pools["scores"][e["index"]][valid], e["pool"])[:k]):
            out.append({"photo": p, "source": 0, "rank": rank, "support": a.support, "center": e["pool"][a.founder]})
        for rank, (pr, la, lo) in enumerate(sorted(accepted.get(e["image_id"], []), reverse=True)[:2]):
            out.append({"photo": p, "source": 1, "rank": rank, "support": pr, "center": [la, lo]})
    return out


def run(tag: str, k: int, clues: bool) -> None:
    from ..models.pinpoint_reranker import PinpointReranker
    from .strategy_search import load_world

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    hyps = _hypotheses(tag, k, clues)
    queries = json.loads((SFT_ROOT / "queries.json").read_text(encoding="utf-8"))
    world = load_world(mp16_queries=[queries[e["index"]] for e in photos])
    reranker = PinpointReranker(device="cuda", index_dir=NVME_INDEX if (NVME_INDEX / "manifest.json").exists() else None)
    p = np.asarray([h["photo"] for h in hyps])
    centre = np.asarray([h["center"] for h in hyps], np.float64)
    dlat = BOX_KM / 111.0
    dlon = dlat / np.maximum(np.cos(np.radians(centre[:, 0])), 0.05)
    box = np.stack([centre[:, 0] - dlat, centre[:, 0] + dlat, centre[:, 1] - dlon, centre[:, 1] + dlon], 1)
    print(f"{tag}: {len(photos)} photos, {len(hyps)} hypotheses ({sum(h['source'] == 1 for h in hyps)} from clues)", flush=True)
    found = reranker.rank(world.query_embeddings[p], query_author=world.query_author[p], index_author=world.mp16["author"][reranker.cache_indices],
                          query_box=box)
    suffix = "_clues" if clues else ""
    np.savez(OUT / f"hypotheses_{tag}{suffix}.npz", photo=p, source=np.asarray([h["source"] for h in hyps]), rank=np.asarray([h["rank"] for h in hyps]),
             support=np.asarray([h["support"] for h in hyps]), center=centre, latlon=found.latlon, scores=found.scores, features=found.features,
             rows=found.rows)
    print(f"-> {OUT / f'hypotheses_{tag}{suffix}.npz'}", flush=True)
    oracle(tag, clues)


def oracle(tag: str, clues: bool = False) -> None:
    from .pinpoint_tool import pick
    from .pinpoint_reference import load_pools
    from .strategy_search import _haversine_km

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    z = dict(np.load(OUT / f"hypotheses_{tag}{'_clues' if clues else ''}.npz"))
    pools = load_pools("mp16")
    truth = np.asarray([e["truth"] for e in photos])
    n = len(photos)
    base = np.asarray([_haversine_km(*e["truth"], np.asarray([e["pool"][0]]))[0] < 25 for e in photos])
    pool_oracle = np.asarray([(_haversine_km(*e["truth"], np.asarray(e["pool"])) < 25).any() for e in photos])
    agree = np.asarray([_haversine_km(*e["truth"], np.asarray([e["pool"][pick(pools["scores"][e["index"]][pools["valid"][e["index"]]], e["pool"])]]))[0] < 25
                        for e in photos])
    top1_ok = np.zeros(len(z["photo"]), bool)
    any_ok = np.zeros(len(z["photo"]), bool)
    for i, ph in enumerate(z["photo"]):
        ll = z["latlon"][i]
        ok = np.isfinite(ll[:, 0])
        if ok.any():
            d = _haversine_km(*truth[ph], ll[ok])
            top1_ok[i] = d[0] < 25
            any_ok[i] = (d < 25).any()
    print(f"{tag}: {n} photos; Pinpoint < 25 km {base.mean():.1%}, agreement {agree.mean():.1%}, pool oracle {pool_oracle.mean():.1%}")
    for k in range(1, int(z["rank"].max()) + 2):
        for name, src in (("Pinpoint areas", (0,)), ("+ clue places", (0, 1))):
            if name.startswith("+") and not (z["source"] == 1).any():
                continue
            sel = np.isin(z["source"], src) & ((z["source"] == 1) | (z["rank"] < k))
            o_top = np.zeros(n, bool)
            o_any = np.zeros(n, bool)
            np.logical_or.at(o_top, z["photo"][sel], top1_ok[sel])
            np.logical_or.at(o_any, z["photo"][sel], any_ok[sel])
            print(f"  K = {k} {name:15s}: hypothesis oracle (restricted top-1) {o_top.mean():.1%}; (any restricted candidate) {o_any.mean():.1%}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("run", "oracle"))
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--clues", action="store_true")
    args = parser.parse_args(argv)
    run(args.tag, args.k, args.clues) if args.cmd == "run" else oracle(args.tag, args.clues)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
