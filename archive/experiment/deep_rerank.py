# Does the comparator turn deeper candidate recall into top-1? Re-rank the top 50 location clusters of the raw neighbours instead of the ~17-candidate pool.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.deep_rerank build     (CPU)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.deep_rerank judge     (vLLM serving the merged comparator on :8765, 2 images per prompt)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.deep_rerank report

"""Candidates: the top-1000 raw neighbours of each dev / val photo from MP16 and OSV-5M (merged by SigLIP2 similarity, same-photographer rows already
excluded), clustered greedily at 1 km; each cluster is represented by its best-similarity photo (the exemplar) and keeps its similarity rank. The first
DEPTH clusters are scored by the comparator (query + that photo -> same place?). A cluster within 1 km of a pool candidate also carries that candidate's
reranker rank (else POOL_MISSING).

report: how often a right-place cluster (within 25 km of the truth) exists among the first K clusters, top-1 within 1 / 25 / 200 km by raw similarity rank,
by the comparator alone, and by a 5-fold cross-validated learned combiner (similarity rank, similarity, comparator, reranker rank) over dev + val, against the
original reranker's top-1.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .exemplar_judge import _jpeg, _p_same
from .query_evidence import BENCH_ROOT, ROOT, SFT_ROOT, MP16Images, _osv_path
from .stage1_eval import _bootstrap
from .strategy_search import MP16_EMBED, OSV_EMBED, EARTH_KM, _xyz, load_world
from .wiki_backend import _km

DEPTH = 50
CLUSTER_KM = 1.0
POOL_MISSING = 20
OUT = ROOT / "deep"


def build() -> None:
    world = load_world(mp16_queries=[{"row": e["row"], "image_id": e["image_id"]} for e in json.loads((ROOT / "dev" / "dev.json").read_text(encoding="utf-8"))])
    gallery = {"mp16": world.mp16["latlon"], "osv": world.osv["latlon"]}
    ids = {"mp16": (MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines(), "osv": (OSV_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines()}
    excluded = set(json.loads((ROOT / "val" / "exclude.json").read_text(encoding="utf-8")))
    cos = math.cos(CLUSTER_KM / EARTH_KM)
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, cache_root in (("dev", SFT_ROOT), ("val", BENCH_ROOT)):
        photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
        with np.load(cache_root / "neighbors.npz") as saved:
            cache = {k: saved[k] for k in ("mp16_raw_idx", "mp16_raw_sim", "osv_raw_idx", "osv_raw_sim")}
        out = []
        for e in photos:
            if e["image_id"] in excluded:
                continue
            rows, sims, corpus = [], [], []
            for c in ("mp16", "osv"):
                idx, sim = cache[f"{c}_raw_idx"][e["index"]], cache[f"{c}_raw_sim"][e["index"]]
                keep = np.isfinite(sim)
                rows.append(idx[keep]); sims.append(sim[keep]); corpus += [c] * int(keep.sum())
            rows, sims, corpus = np.concatenate(rows), np.concatenate(sims), np.asarray(corpus)
            order = np.argsort(-sims)
            centres = np.zeros((0, 3))
            clusters: list[dict[str, Any]] = []
            pool = np.asarray(e["pool"])
            pool_xyz = _xyz(pool)
            for o in order:
                c = str(corpus[o])
                lat, lon = (float(x) for x in gallery[c][rows[o]])
                point = _xyz(np.asarray([lat, lon]))  # a single (lat, lon) gives a 3-vector
                if len(centres) and (centres @ point >= cos).any():
                    continue
                centres = np.vstack([centres, point])
                near = np.flatnonzero(pool_xyz @ point >= cos)
                clusters.append({"corpus": c, "id": ids[c][rows[o]], "lat": lat, "lon": lon, "sim": float(sims[o]), "pool_rank": int(near.min()) if len(near) else POOL_MISSING})
                if len(clusters) == DEPTH:
                    break
            d = _km(np.asarray([[x["lat"], x["lon"]] for x in clusters]), *e["truth"])
            out.append({"image_id": e["image_id"], "path": e.get("path"), "benchmark": e.get("benchmark", "mp16"), "truth": e["truth"], "reranker_km": float(_km(pool[:1], *e["truth"])[0]),
                        "clusters": clusters, "km": [float(x) for x in d]})
        (OUT / f"{tag}.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
        print(f"{tag}: {len(out)} photos, {np.mean([len(p['clusters']) for p in out]):.1f} clusters per photo", flush=True)


def judge(server: str) -> None:
    images = MP16Images()
    folders = sorted(p for p in Path("/data/hf/datasets/osv5m/images/train").iterdir() if p.is_dir())

    def read(corpus: str, image_id: str) -> bytes | None:
        if corpus == "mp16":
            return images.read(image_id)
        path = _osv_path(image_id, folders)
        return path.read_bytes() if path else None

    for tag in ("dev", "val"):
        photos = json.loads((OUT / f"{tag}.json").read_text(encoding="utf-8"))
        jobs = [(i, k) for i, p in enumerate(photos) for k in range(len(p["clusters"]))]

        def run(job: tuple[int, int]) -> float | None:
            i, k = job
            p = photos[i]
            neighbour = read(p["clusters"][k]["corpus"], p["clusters"][k]["id"])
            query = Path(p["path"]).read_bytes() if p["path"] else images.read(p["image_id"])
            if neighbour is None or query is None:
                return None
            try:
                return _p_same(server, _jpeg(query), _jpeg(neighbour))
            except Exception:
                return None

        with ThreadPoolExecutor(48) as pool:
            scores = list(pool.map(run, jobs))
        for p in photos:
            for c in p["clusters"]:
                c["p_same"] = None
        for (i, k), s in zip(jobs, scores):
            photos[i]["clusters"][k]["p_same"] = s
        (OUT / f"{tag}_scored.json").write_text(json.dumps(photos) + "\n", encoding="utf-8")
        print(f"{tag}: {len(jobs)} comparisons, {sum(s is None for s in scores)} failed", flush=True)


def _features(p: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    cl = p["clusters"]
    n = len(cl)
    raw = np.asarray([np.nan if c["p_same"] is None else c["p_same"] for c in cl])
    prob = np.where(np.isnan(raw), 0.5, np.clip(raw, 0.02, 0.98))
    logit = np.log(prob / (1 - prob))
    sim = np.asarray([c["sim"] for c in cl])
    rank = np.arange(n, dtype=float)
    pool = np.asarray([c["pool_rank"] for c in cl], dtype=float)
    judge_order = (-logit).argsort().argsort().astype(float)
    f = np.stack([np.log1p(rank), sim, sim - sim.max(), logit, logit - logit.max(), np.log1p(judge_order), pool, (pool < POOL_MISSING), (pool == 0), np.log1p(pool)], axis=1)
    return f.astype(np.float32), np.asarray(p["km"])


def report() -> None:
    import torch

    photos = []
    for tag in ("dev", "val"):
        for p in json.loads((OUT / f"{tag}_scored.json").read_text(encoding="utf-8")):
            p["tag"] = tag
            photos.append(p)
    n = len(photos)
    F = np.zeros((n, DEPTH, 10), dtype=np.float32)
    D = np.full((n, DEPTH), 1e5)
    for i, p in enumerate(photos):
        f, d = _features(p)
        F[i, : len(f)], D[i, : len(d)] = f, d
    valid = D < 9e4
    tags = np.asarray([p["tag"] for p in photos])
    reranker = np.asarray([p["reranker_km"] for p in photos])
    print(f"{n} photos (dev {int((tags == 'dev').sum())}, val {int((tags == 'val').sum())}); clusters scored per photo: up to {DEPTH}\n")
    print("Does a right-place cluster (within 25 km of the truth) exist among the first K similarity-ranked clusters?")
    for label, sel in (("dev", tags == "dev"), ("val", tags == "val")):
        print(f"  {label}: " + ", ".join(f"K={k}: {(D[sel][:, :k] < 25).any(1).mean():.0%}" for k in (1, 5, 10, 25, 50)) + f"; reranker pool (~17): {(np.asarray([p['reranker_km'] for p in photos])[sel] < 25).mean():.0%} (its top-1)")

    reward = np.where(valid, (D < 25) * 1.0 + (D < 1) * 0.5, 0.0).astype(np.float32)
    mean, std = F[valid].mean(0), F[valid].std(0) + 1e-6
    X, M, R = torch.as_tensor((F - mean) / std), torch.as_tensor(valid), torch.as_tensor(reward)

    def cv(cols: list[int]) -> dict[float, np.ndarray]:
        folds = np.array_split(np.random.default_rng(0).permutation(n), 5)
        hits = {t: np.zeros(n) for t in (1.0, 25.0, 200.0)}
        for seed in range(3):
            for f in folds:
                torch.manual_seed(seed)
                net = torch.nn.Sequential(torch.nn.Linear(len(cols), 32), torch.nn.GELU(), torch.nn.Linear(32, 1))
                opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-2)
                idx = torch.as_tensor(np.setdiff1d(np.arange(n), f))
                for _ in range(300):
                    loss = -(torch.softmax(net(X[idx][..., cols]).squeeze(-1).masked_fill(~M[idx], float("-inf")), -1) * R[idx]).sum(-1).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    pick = net(X[f][..., cols]).squeeze(-1).masked_fill(~M[f], float("-inf")).argmax(1).numpy()
                for t in hits:
                    hits[t][f] += (D[f, pick] < t) / 3
        return hits

    comparator_pick = np.where(valid, F[..., 3], -1e9).argmax(1)
    systems = {
        "original reranker top-1": {t: (reranker < t).astype(float) for t in (1.0, 25.0, 200.0)},
        "raw similarity, best cluster": {t: (D[:, 0] < t).astype(float) for t in (1.0, 25.0, 200.0)},
        "comparator alone over 50 clusters": {t: (D[np.arange(n), comparator_pick] < t).astype(float) for t in (1.0, 25.0, 200.0)},
        "learned: similarity features only": cv([0, 1, 2]),
        "learned: similarity + comparator": cv([0, 1, 2, 3, 4, 5]),
        "learned: similarity + reranker rank": cv([0, 1, 2, 6, 7, 8, 9]),
        "learned: similarity + comparator + reranker rank": cv(list(range(10))),
    }
    base = systems["original reranker top-1"]
    print(f"\n{'top-1 within 1 / 25 / 200 km':52s}  all                    dev   val   (<25 km change vs the original reranker [95% CI], all photos)")
    for name, hits in systems.items():
        cells = " ".join(f"{hits[t].mean():6.1%}" for t in (1.0, 25.0, 200.0))
        ci = _bootstrap(hits[25.0] - base[25.0])
        print(f"  {name:50s} {cells}   {hits[25.0][tags == 'dev'].mean():5.1%} {hits[25.0][tags == 'val'].mean():5.1%}   {ci[0]:+.1%} [{ci[1]:+.1%}, {ci[2]:+.1%}]")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("build", "judge", "report"))
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    args = parser.parse_args(argv)
    {"build": build, "judge": lambda: judge(args.server), "report": report}[args.node]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
