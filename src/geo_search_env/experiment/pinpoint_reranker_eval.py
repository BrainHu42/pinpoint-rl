# Pinpoint's attention reranker (the submission's full model) on im2gps3k, yfcc4k and wikimedia: the baseline, run as in prior work (no photographer filter).
# Usage: PYTHONPATH=src:/home/brian/workspace/pinpoint-submission/submission/src \
#          /home/brian/workspace/pinpoint-submission/submission/.venv/bin/python -m geo_search_env.experiment.pinpoint_reranker_eval {run,parity,report}

"""run: top-1 and the 12 reranked candidates for every benchmark photo without the photographer filter (the baseline, comparable to prior work), and
for im2gps3k / yfcc4k also with it (secondary; no wikimedia photographer is in MP16) (GPU, ~17 GB, ~15 min).
parity: the unfiltered run against the submission's own code (`original_unfiltered.npz`, made by its unchanged inference path).
report: % within 1 / 25 / 200 / 750 / 2500 km per benchmark, then on the 3,713 eval-half photos used for every past decision (placeholders dropped),
next to the old one-step reranker's top-1 and both pool oracles. The one-step pool was built with the photographer filter, which only matters on yfcc4k.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data.benchmarks import compute_metrics, geodesic_km, load_benchmark


OUT = Path("artifacts/pinpoint_reranker")
QE = Path("artifacts/query_evidence")
BENCHMARKS = ("im2gps3k", "yfcc4k", "wikimedia")
RADII = (1, 25, 200, 750, 2500)


def run() -> None:
    from ..models.pinpoint_reranker import PinpointReranker, query_author_ids

    reranker = PinpointReranker()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in BENCHMARKS:
        bench = load_benchmark(name)
        authors = query_author_ids(bench.authors, reranker.vocab)
        print(f"{name}: {len(bench.image_ids)} photos, {int((authors >= 0).sum())} by an MP16 photographer", flush=True)
        modes = {"unfiltered": None} if name == "wikimedia" else {"unfiltered": None, "filtered": authors}
        for mode, query_authors in modes.items():
            result = reranker.predict(bench.embeddings, query_authors)
            np.savez(OUT / f"{name}_{mode}.npz", ids=np.asarray(bench.image_ids), truth=bench.latlon, **result)
            print(f"  {mode}: saved", flush=True)


def parity() -> None:
    reference = np.load(OUT / "original_unfiltered.npz")
    for name in ("im2gps3k", "yfcc4k"):
        ours = np.load(OUT / f"{name}_unfiltered.npz")
        assert list(ours["ids"]) == list(reference[f"{name}_ids"]), f"{name}: photo order differs"
        same_set = np.mean([set(a) == set(b) for a, b in zip(ours["idx"], reference[f"{name}_idx"])])
        km = geodesic_km(ours["pred"].astype(np.float64), reference[f"{name}_pred"].astype(np.float64))
        print(f"{name}: same 12 candidates {same_set:.2%}, same top-1 (< 1 m) {np.mean(km < 0.001):.2%}, top-1 moved > 1 km {np.mean(km > 1):.2%}")


def _pct(distances: np.ndarray) -> list[float]:
    return [100 * float(np.mean(distances < r)) for r in RADII]


def _row(label: str, n: int, values: list[float]) -> str:
    return f"  {label:44s} n={n:5d} " + " ".join(f"{v:5.1f}" for v in values)


def report() -> None:
    results: dict[str, dict] = {}
    print("whole benchmarks, % within " + " / ".join(f"{r} km" for r in RADII))
    runs = {}
    for name in BENCHMARKS:
        for mode in ("filtered", "unfiltered"):
            path = OUT / f"{name}_{mode}.npz"
            if not path.exists():
                continue
            d = runs[name, mode] = np.load(path)
            m = compute_metrics(d["pred"], d["truth"])
            results[f"{name}/{mode}"] = m
            print(_row(f"{name} attention reranker, {mode}", len(d["pred"]), [100 * m[f"Under_{r}_km"] for r in RADII]) + f"  median {m['Median_km_error']:.0f} km")

    for tag, names in (("full", ("im2gps3k", "yfcc4k")), ("wikimedia", ("wikimedia",))):
        photos = json.loads((QE / tag / "dev.json").read_text(encoding="utf-8"))
        if tag == "full":
            placeholder = {p["image_id"] for p in json.loads((QE / "multi_exemplar_full_pairs.json").read_text(encoding="utf-8")) if p["placeholder"]}
            photos = [p for p in photos if p["image_id"] not in placeholder]
        rows: dict[str, list[float]] = {}
        old_top1, old_pool, truth = [], [], []
        for p in photos:
            truth.append(json.loads(p["truth"]) if isinstance(p["truth"], str) else p["truth"])
            pool = np.asarray(json.loads(p["pool"]) if isinstance(p["pool"], str) else p["pool"], dtype=np.float64)
            old_top1.append(pool[0])
            old_pool.append(geodesic_km(pool, np.repeat(np.asarray(truth[-1], dtype=np.float64)[None], len(pool), 0)).min())
        truth = np.asarray(truth, dtype=np.float64)
        for mode in ("unfiltered", "filtered"):
            if not all((name, mode) in runs for name in names):
                continue
            lookup = {(name, image_id): k for name in names for k, image_id in enumerate(runs[name, mode]["ids"])}
            top1, best = [], []
            for p, t in zip(photos, truth):
                d, k = runs[p["benchmark"], mode], lookup[p["benchmark"], p["image_id"]]
                mask = np.isfinite(d["scores"][k])
                top1.append(d["pred"][k])
                best.append(geodesic_km(d["cand_latlon"][k][mask].astype(np.float64), np.repeat(t[None], mask.sum(), 0)).min())
            label = "" if mode == "unfiltered" else ", photographer-filtered"
            rows[f"attention reranker top-1{label}"] = _pct(geodesic_km(np.asarray(top1, dtype=np.float64), truth))
            rows[f"oracle over its 12{label}"] = _pct(np.asarray(best))
        rows["one-step reranker top-1 (filtered)"] = _pct(geodesic_km(np.asarray(old_top1, dtype=np.float64), truth))
        rows["oracle over the one-step pool (filtered)"] = _pct(np.asarray(old_pool))
        label = "3,713 eval-half photos (placeholders dropped)" if tag == "full" else "all wikimedia photos"
        print(f"\n{label}, % within " + " / ".join(f"{r} km" for r in RADII))
        for k, v in rows.items():
            print(_row(k, len(truth), v))
        results[f"{tag}/compare"] = {"n": len(truth), **rows}
    (OUT / "report.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "parity", "report"))
    args = parser.parse_args()
    {"run": run, "parity": parity, "report": report}[args.command]()


if __name__ == "__main__":
    main()
