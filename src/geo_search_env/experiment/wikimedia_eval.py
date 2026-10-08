# The wikimedia final-test photos through the same pipeline as im2gps3k / yfcc4k: candidate pool, reranker order, baseline and oracle numbers.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.wikimedia_eval candidates   (GPU; ~10-30 min: neighbours, region head, pool)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.wikimedia_eval report       (CPU; writes artifacts/query_evidence/wikimedia/dev.json)

"""candidates: the benchmark pipeline's candidates for every wikimedia photo (Pinpoint GPS gallery + raw SigLIP2 with the region prior, clustered at 1 km), ranked
              by the one-step reranker fitted on the im2gps3k / yfcc4k tune halves (as sft_data.candidates does for MP16 photos). Nothing is trained on wikimedia.
report:       reranker top-1, oracle over its top 8 and over the whole pool at 1 / 25 / 200 / 750 / 2500 km on all photos and on the density-balanced subset
              (test_balanced.csv), the share of near-duplicates of a gallery photo, and `dev.json` (same fields as the other photo sets, tag `wikimedia`)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ..data.benchmarks import DATA_ROOT, compute_metrics
from .query_evidence import BENCH_ROOT, ROOT
from .sft_data import BENCHMARK_NAMES, _haversine_km, fit_reranker
from .strategy_search import _knn_distribution, _search_candidates, load_world, neighbor_cache, region_head

OUT = Path("artifacts/wikimedia")
TAG = "wikimedia"
THRESHOLDS = (1, 25, 200, 750, 2500)
TOPK = 8


def candidates() -> None:
    bench_world = load_world()
    score = fit_reranker(bench_world, BENCH_ROOT)
    del bench_world
    coarse_report = json.loads((BENCH_ROOT / "coarse.json").read_text(encoding="utf-8"))
    head = max(("head_linear", "head_mlp"), key=lambda name: sum(coarse_report[f"{b}/tune"][name]["region_mass"] for b in BENCHMARK_NAMES))
    gps_name = coarse_report["selected"]["knn_mp16_gps"]
    tau, k = float(gps_name.split("tau=")[1].split("|")[0]), int(gps_name.split("k=")[1])

    OUT.mkdir(parents=True, exist_ok=True)
    world = load_world(with_pinpoint=True, benchmarks=("wikimedia",))
    n = len(world.queries)
    print(f"{n} wikimedia photos; same-photographer matches with MP16: {int((world.query_author >= 0).sum())}", flush=True)
    (OUT / "queries.json").write_text(json.dumps(world.queries) + "\n", encoding="utf-8")
    if not (OUT / "neighbors.npz").exists():
        np.savez(OUT / "neighbors.npz", **neighbor_cache(world, chunk=16_384))
    cache = dict(np.load(OUT / "neighbors.npz"))
    if not (OUT / "region_head.npz").exists():
        region_head(OUT, world=world)
    heads = dict(np.load(OUT / "region_head.npz"))
    head_predictions = [{int(r): float(p) for r, p in zip(heads[f"{head}_regions"][q], heads[f"{head}_probs"][q])} for q in range(n)]
    gps_votes = [_knn_distribution(world, cache, "mp16_gps", q, k, tau) for q in range(n)]
    coords, valid, one_shot = _search_candidates(world, cache, head_predictions, gps_votes)
    ranking = np.argsort(-score(one_shot, valid), axis=1)
    distance = np.full(valid.shape, np.inf)
    for q in range(n):
        distance[q, valid[q]] = _haversine_km(*world.query_latlon[q], coords[q, valid[q]])
    pinpoint_top1 = world.mp16["latlon"][cache["mp16_gps_idx"][:, 0]]  # Pinpoint's GPS-gallery top-1 per photo, same-photographer rows excluded
    np.savez(OUT / "candidates.npz", coords=coords, valid=valid, one_shot=one_shot, ranking=ranking, distance=distance, latlon=world.query_latlon, pinpoint_top1=pinpoint_top1)
    print(f"saved candidates for {n} queries -> {OUT / 'candidates.npz'}")


def report() -> None:
    queries = json.loads((OUT / "queries.json").read_text(encoding="utf-8"))
    saved = np.load(OUT / "candidates.npz")
    coords, valid, ranking, distance, truth = saved["coords"], saved["valid"], saved["ranking"], saved["distance"], saved["latlon"]
    cache = np.load(OUT / "neighbors.npz")
    with (DATA_ROOT / "wikimedia" / "test_balanced.csv").open(encoding="utf-8", newline="") as stream:
        balanced_ids = {row["IMG_ID"].strip() for row in csv.DictReader(stream)}
    balanced = np.asarray([q["image_id"] in balanced_ids for q in queries])
    near_duplicate = np.maximum(cache["mp16_raw_sim"][:, 0], cache["osv_raw_sim"][:, 0]) >= 0.95

    ordered = np.take_along_axis(np.where(valid, distance, np.inf), ranking, axis=1)  # distances in reranker order
    pinpoint = np.asarray([_haversine_km(*truth[q], saved["pinpoint_top1"][q][None, :])[0] for q in range(len(queries))])
    sets = {"all": np.ones(len(queries), bool), "balanced subset": balanced}
    result: dict[str, dict] = {}
    print(f"{len(queries)} photos; balanced subset {int(balanced.sum())}; near-duplicates of a gallery photo (cosine >= 0.95): {near_duplicate.mean():.1%}; "
          f"mean pool size {valid.sum(1).mean():.1f}\n")
    print(f"{'':34s}" + "".join(f"{t:>7d} km" for t in THRESHOLDS))
    for label, sel in sets.items():
        rows = {
            "Pinpoint GPS top-1 (author-filtered)": [(pinpoint[sel] < t).mean() for t in THRESHOLDS],
            "reranker top-1": [(ordered[sel, 0] < t).mean() for t in THRESHOLDS],
            "oracle, reranker's top 8": [(ordered[sel, :TOPK].min(1) < t).mean() for t in THRESHOLDS],
            "oracle, whole pool": [(ordered[sel].min(1) < t).mean() for t in THRESHOLDS],
        }
        result[label] = {k: [float(x) for x in v] for k, v in rows.items()} | {"n": int(sel.sum())}
        print(f"{label} (n={int(sel.sum())})")
        for name, values in rows.items():
            print(f"  {name:32s}" + "".join(f"{100 * v:9.1f}%" for v in values))
    keep = ~near_duplicate
    print(f"\nwithout near-duplicates (n={int(keep.sum())}): reranker top-1 " + " / ".join(f"{100 * (ordered[keep, 0] < t).mean():.1f}" for t in THRESHOLDS))
    result["no near-duplicate"] = {"reranker top-1": [float((ordered[keep, 0] < t).mean()) for t in THRESHOLDS], "n": int(keep.sum())}
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    dev = [
        {
            "index": q, "image_id": queries[q]["image_id"], "path": queries[q]["path"], "benchmark": "wikimedia", "balanced": bool(balanced[q]),
            "truth": truth[q].tolist(), "candidates": [],
            "pool": [[float(coords[q, c, 0]), float(coords[q, c, 1])] for c in ranking[q] if valid[q, c]],  # every pooled candidate, best reranker rank first
        }
        for q in range(len(queries))
    ]
    (ROOT / TAG).mkdir(parents=True, exist_ok=True)
    (ROOT / TAG / "dev.json").write_text(json.dumps(dev) + "\n", encoding="utf-8")
    print(f"\ndev.json: {len(dev)} photos -> {ROOT / TAG / 'dev.json'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("candidates", "report"))
    args = parser.parse_args(argv)
    {"candidates": candidates, "report": report}[args.node]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
