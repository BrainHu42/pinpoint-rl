# Does a region prior turn the raw neighbours' deep recall into candidates? Region-diversified pools on dev / val (CPU, cached neighbours).
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.region_prior

"""Region = MP16-Pro (state, country), as in strategy_search. The region distribution is the MLP region head on frozen SigLIP2 (query
photographers held out), optionally mixed with Pinpoint's GPS-gallery kNN votes. All candidates come from the cached top-1000 raw
neighbours (MP16 + OSV-5M, same-photographer rows excluded), clustered at 1 km.

Pools compared at the same number of candidates:
current:   the pipeline's pool (Pinpoint + prior-weighted photo matching, reranker order, ~17).
raw prior: photo-match clusters with similarity + lambda * log prior (the pipeline's raw source, lambda = 0.01; larger lambdas too).
regional:  clusters inside each of the top regions separately, interleaved by P(region) / (rank in region + 1) ** beta.
oracle region: clusters inside the true region only (a perfect region prior; ceiling).
Also: the current pool plus K extra candidates from each source (oracle gain over the pool at matched budget).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .query_evidence import BENCH_ROOT, ROOT, SFT_ROOT
from .stage1_eval import _bootstrap
from .strategy_search import BENCHMARK_NAMES, EARTH_KM, PRIOR_FLOOR, _knn_distribution, _pool, _xyz, load_world
from .wiki_backend import _km

OUT = ROOT / "region"
THRESHOLDS = (1.0, 25.0, 200.0)
BUDGETS = (10, 17, 30)
EXTRA = (5, 10, 20)
LAMBDAS = (0.01, 0.03, 0.1)
BETAS = (0.5, 1.0, 2.0)
REGIONS = 10
PER_REGION = 30
DEPTH = max(max(BUDGETS), max(EXTRA) + 20)


def _region_distribution(heads, q: int, gps: dict[int, float] | None, mix: float) -> dict[int, float]:
    head = {int(r): float(p) for r, p in zip(heads["head_mlp_regions"][q], heads["head_mlp_probs"][q])}
    if gps is None or mix == 0:
        return head
    keys = set(head) | set(gps)
    return {r: (1 - mix) * head.get(r, 0.0) + mix * gps.get(r, 0.0) for r in keys}


def _regional(hits: list[tuple[float, float, float, int]], prior: dict[int, float], beta: float) -> list[tuple[float, float]]:
    ranked = sorted(((p, r) for r, p in prior.items() if r >= 0), reverse=True)[:REGIONS]
    keyed = []
    for p, r in ranked:
        for i, c in enumerate(_pool([h[:3] for h in hits if h[3] == r], limit=PER_REGION)):
            keyed.append((math.log(p + 1e-9) - beta * math.log(i + 1), c))
    keyed.sort(key=lambda x: -x[0])
    return _dedupe([c for _, c in keyed])


def _dedupe(points: list[tuple[float, float]], km: float = 1.0) -> list[tuple[float, float]]:
    cos = math.cos(km / EARTH_KM)
    out: list[tuple[float, float]] = []
    xyz: list[np.ndarray] = []
    for p in points:
        v = _xyz(np.asarray(p))
        if any(v @ u >= cos for u in xyz):
            continue
        out.append(p)
        xyz.append(v)
    return out


def _extend(base: list[tuple[float, float]], extra: list[tuple[float, float]], k: int, km: float = 1.0) -> list[tuple[float, float]]:
    """`base` plus the first `k` points of `extra` not within `km` of anything already chosen."""

    return _dedupe(list(base) + list(extra))[: len(base) + k] if base else extra[:k]


def _hit(points, truth, t) -> bool:
    return bool(len(points)) and bool((_km(np.asarray(points), *truth) < t).any())


def main() -> None:
    dev = json.loads((ROOT / "dev" / "dev.json").read_text(encoding="utf-8"))
    world = load_world(mp16_queries=[{"row": e["row"], "image_id": e["image_id"]} for e in dev])
    excluded = set(json.loads((ROOT / "val" / "exclude.json").read_text(encoding="utf-8")))
    coarse = json.loads((BENCH_ROOT / "coarse.json").read_text(encoding="utf-8"))
    gps_name = coarse["selected"]["knn_mp16_gps"]
    tau, k_gps = float(gps_name.split("tau=")[1].split("|")[0]), int(gps_name.split("k=")[1])
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    for tag, root in (("dev", SFT_ROOT), ("val", BENCH_ROOT)):
        photos = [e for e in json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8")) if e["image_id"] not in excluded]
        with np.load(root / "neighbors.npz") as saved:
            cache = {k: saved[k] for k in ("mp16_raw_idx", "mp16_raw_sim", "osv_raw_idx", "osv_raw_sim", "mp16_gps_idx", "mp16_gps_sim")}
        heads = dict(np.load(root / "region_head.npz"))
        truth = np.asarray([e["truth"] for e in photos])
        truth_region = world.region_grid.lookup(truth[:, 0], truth[:, 1])
        pools: dict[str, list[list[tuple[float, float]]]] = {}
        region_rank = {"head": [], "head+gps": []}
        for n, e in enumerate(photos):
            q = e["index"]
            hits = []
            for name, gallery in (("mp16_raw", world.mp16), ("osv_raw", world.osv)):
                idx, sim = cache[f"{name}_idx"][q], cache[f"{name}_sim"][q]
                keep = np.isfinite(sim)
                xy, labels = gallery["latlon"][idx[keep]], gallery["region"][idx[keep]]
                hits += [(float(a), float(b), float(s), int(r)) for (a, b), s, r in zip(xy, sim[keep], labels)]
            gps = _knn_distribution(world, cache, "mp16_gps", q, k_gps, tau)
            priors = {"head": _region_distribution(heads, q, None, 0.0), "head+gps": _region_distribution(heads, q, gps, 0.5)}
            for key, prior in priors.items():
                ranked = [r for r, _ in sorted(prior.items(), key=lambda x: -x[1]) if r >= 0]
                region_rank[key].append(ranked.index(int(truth_region[n])) + 1 if truth_region[n] in ranked else 999)
            arms: dict[str, list[tuple[float, float]]] = {"current": [tuple(p) for p in e["pool"]]}
            head = priors["head"]
            for lam in LAMBDAS:
                arms[f"raw prior lambda={lam}"] = _pool([(a, b, s + lam * math.log(head.get(r, 0.0) + PRIOR_FLOOR)) for a, b, s, r in hits], limit=DEPTH)
            for key, prior in priors.items():
                for beta in BETAS:
                    arms[f"regional {key} beta={beta}"] = _regional(hits, prior, beta)[:DEPTH]
            arms["oracle region"] = _pool([h[:3] for h in hits if h[3] == truth_region[n]], limit=DEPTH)
            for arm, points in arms.items():
                pools.setdefault(arm, []).append(points)
            if (n + 1) % 200 == 0:
                print(f"{tag} {n + 1}/{len(photos)}", flush=True)

        entry: dict[str, Any] = {"n": len(photos)}
        for key, ranks in region_rank.items():
            r = np.asarray(ranks)
            entry[f"true region rank ({key})"] = {f"top{k}": float((r <= k).mean()) for k in (1, 3, 5, 10)}
        current_hit = {t: np.asarray([_hit(p, truth[i], t) for i, p in enumerate(pools["current"])]) for t in THRESHOLDS}
        entry["pool oracle"] = {f"<{t:g} km": float(current_hit[t].mean()) for t in THRESHOLDS}
        entry["reranker top-1"] = {f"<{t:g} km": float(np.mean([_hit(p[:1], truth[i], t) for i, p in enumerate(pools["current"])])) for t in THRESHOLDS}
        entry["alone"], entry["extra"] = {}, {}
        for arm, rows in pools.items():
            entry["alone"][arm] = {f"@{b}": {f"<{t:g} km": float(np.mean([_hit(p[:b], truth[i], t) for i, p in enumerate(rows)])) for t in THRESHOLDS} for b in BUDGETS}
            if arm == "current":
                continue
            entry["extra"][arm] = {}
            for k in EXTRA:
                hit = {t: np.asarray([_hit(_extend(pools["current"][i], p, k), truth[i], t) for i, p in enumerate(rows)]) for t in THRESHOLDS}
                entry["extra"][arm][f"+{k}"] = {f"<{t:g} km": list(_bootstrap((hit[t] & ~current_hit[t]).astype(float))) for t in THRESHOLDS}
        report[tag] = entry
        (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        print(f"\n== {tag} (n={len(photos)}) ==")
        for key in region_rank:
            print(f"  true region in top-1/3/5/10 of {key}: " + " / ".join(f"{v:.0%}" for v in entry[f"true region rank ({key})"].values()))
        print("  reranker top-1 <1/25/200 km: " + " / ".join(f"{v:.1%}" for v in entry["reranker top-1"].values()))
        print("  pool oracle    <1/25/200 km: " + " / ".join(f"{v:.1%}" for v in entry["pool oracle"].values()))
        print(f"  {'pool alone (oracle <1 / <25 / <200 km)':40s}" + "".join(f"{'@' + str(b):>22s}" for b in BUDGETS))
        for arm, v in entry["alone"].items():
            print(f"  {arm:40s}" + "".join(f"{' / '.join(f'{x:.1%}' for x in v[f'@{b}'].values()):>22s}" for b in BUDGETS))
        print(f"  {'current pool + K extra: gain <25 km [95% CI]':40s}" + "".join(f"{'+' + str(k):>22s}" for k in EXTRA))
        for arm, v in entry["extra"].items():
            print(f"  {arm:40s}" + "".join(f"{'{:+.1f} [{:+.1f},{:+.1f}]'.format(*(100 * x for x in v[f'+{k}']['<25 km'])):>22s}" for k in EXTRA))


if __name__ == "__main__":
    main()
