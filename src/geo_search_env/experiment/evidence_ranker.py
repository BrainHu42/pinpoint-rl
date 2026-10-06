# Is the evidence worth anything to a learned chooser? Refit the pipeline's candidate reranker with and without evidence features.
# Usage: .venv/bin/python -m geo_search_env.experiment.evidence_ranker select                (once: MP16 train photo list)
#        .venv/bin/python -m geo_search_env.experiment.stage1_eval places --tag train      (vLLM on :8765)
#        .venv/bin/python -m geo_search_env.experiment.evidence_ranker retrieve            (GPU: SigLIP2 text + Wikipedia dense)
#        .venv/bin/python -m geo_search_env.experiment.evidence_ranker fit                 (GPU, a few minutes)

"""The pipeline's per-candidate MLP reranker (listwise expected-reward objective, `strategy_search._fit_selector`) refit on Pinpoint's
held-out MP16 train photos (bucket 99, training photographers), with and without evidence features, evaluated on `dev` (MP16 val
photographers) and `val` (benchmark eval halves).

Evidence = the 4B's three named places per photo (prompt v2) searched in SigLIP2 photo space (one MP16 + one OSV photo per query)
or in Wikipedia (bge dense, two articles per query). Per candidate: log1p of the number of results within 1 / 25 / 200 km, plus
log1p of the number of results the photo has. Configs: A original features; B + SigLIP2; C + Wikipedia; D + both; E = D with the
evidence shuffled across photos within each set (control: gains must come from content, not capacity). Three seeds.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Sequence

import numpy as np

from .query_evidence import BENCH_ROOT, OVERLAY_THRESHOLD, ROOT, SFT_ROOT, _load, _save
from .stage1_eval import _bootstrap, _retrieve
from .strategy_search import EARTH_KM, MP16_EMBED, _geoguessr, _haversine_km, _xyz


TAG = "train"
RADII_KM = (1.0, 25.0, 200.0)
SEEDS = (0, 1, 2)
THRESHOLDS = (1.0, 25.0, 200.0)
EVIDENCE = ("siglip", "dense")


def select_train() -> None:
    """Held-out MP16 train-split photos without burned-in GPS (the same pool the SFT data used)."""

    queries = json.loads((SFT_ROOT / "queries.json").read_text(encoding="utf-8"))
    scores = json.loads((SFT_ROOT / "overlay_scores.json").read_text(encoding="utf-8"))
    latlon = np.fromfile(MP16_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2).astype(np.float64)
    photos = [
        {"index": i, "image_id": q["image_id"], "row": q["row"], "truth": latlon[q["row"]].tolist()}
        for i, q in enumerate(queries)
        if q["group"] == "held_out" and q["split"] == "train" and scores.get(q["image_id"]) is not None and scores[q["image_id"]] < OVERLAY_THRESHOLD
    ]
    (ROOT / TAG).mkdir(parents=True, exist_ok=True)
    _save(TAG, "dev.json", photos)
    print(f"{len(photos)} training photos")


def retrieve_train() -> None:
    photos = _load(TAG, "dev.json")
    generated = _load(TAG, "places.json")
    texts = [(m, q) for m, e in enumerate(photos) for q in generated[e["image_id"]]["queries"]]
    print(f"{len(photos)} photos, {len(texts)} queries", flush=True)
    results = _retrieve(TAG, "v2", photos, generated, texts, backends=EVIDENCE)
    (ROOT / TAG / "results.json").write_text(json.dumps({k: results[k] for k in EVIDENCE}) + "\n", encoding="utf-8")


def evidence_features(coords: np.ndarray, valid: np.ndarray, points: list[list[tuple[float, float]]]) -> np.ndarray:
    """(photos, candidates, 4): log1p of the results within each radius of a candidate, and of the photo's result count."""

    out = np.zeros(valid.shape + (len(RADII_KM) + 1,), dtype=np.float32)
    cosines = [math.cos(r / EARTH_KM) for r in RADII_KM]
    for q, pts in enumerate(points):
        out[q, :, -1] = np.log1p(len(pts))
        if not pts:
            continue
        dots = _xyz(coords[q]) @ _xyz(np.asarray(pts, dtype=np.float64)).T
        for r, c in enumerate(cosines):
            out[q, :, r] = np.log1p((dots >= c).sum(1))
    return out * valid[..., None]


def load_set(tag: str) -> dict[str, Any]:
    photos = _load(tag, "dev.json")
    index = [e["index"] for e in photos]
    bench = "path" in photos[0]
    saved = dict(np.load((BENCH_ROOT / "search_features.npz") if bench else (SFT_ROOT / "candidates.npz")))
    coords, valid, one_shot = saved["coords"][index], saved["valid"][index], saved["one_shot"][index]
    results = json.loads((ROOT / tag / "results.json").read_text(encoding="utf-8"))
    truth = np.asarray([e["truth"] for e in photos])
    distance = np.full(valid.shape, np.inf)
    for q in range(len(photos)):
        distance[q, valid[q]] = _haversine_km(*truth[q], coords[q, valid[q]])
    return {
        "photos": photos, "coords": coords, "valid": valid, "one_shot": one_shot, "distance": distance,
        "evidence": {name: evidence_features(coords, valid, [[tuple(p) for p in pts] for pts in results[name]]) for name in EVIDENCE},
        "reranker": np.asarray([_haversine_km(*truth[q], np.asarray([e["pool"][0]]))[0] if "pool" in e else np.inf for q, e in enumerate(photos)]),
    }


def _reward(distance: np.ndarray, valid: np.ndarray) -> np.ndarray:
    finite = np.where(valid, distance, 1e5)
    return np.where(valid, _geoguessr(finite) / 5000.0 + 0.5 * (finite < 25) + 0.5 * (finite < 1), 0.0)


def fit_scores(features: np.ndarray, valid: np.ndarray, reward: np.ndarray, train: np.ndarray, seed: int, steps: int = 1_500) -> np.ndarray:
    """`strategy_search._fit_selector` (same MLP, optimizer and listwise objective) on the GPU; scores for every row."""

    import torch

    torch.manual_seed(seed)
    mean, std = features[train][valid[train]].mean(0), features[train][valid[train]].std(0) + 1e-6
    x = torch.as_tensor((features - mean) / std, dtype=torch.float32, device="cuda")
    mask, r = torch.as_tensor(valid, device="cuda"), torch.as_tensor(reward, dtype=torch.float32, device="cuda")
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[-1], 64), torch.nn.GELU(), torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 1)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    index = torch.as_tensor(np.flatnonzero(train), device="cuda")
    for _ in range(steps):
        logits = model(x[index]).squeeze(-1).masked_fill(~mask[index], float("-inf"))
        loss = -(torch.softmax(logits, dim=-1) * r[index]).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return model(x).squeeze(-1).masked_fill(~mask, float("-inf")).cpu().numpy()


def fit() -> None:
    sets = {name: load_set(name) for name in ("train", "dev", "val")}
    sizes = {name: len(s["photos"]) for name, s in sets.items()}
    print(f"photos: {sizes}", flush=True)
    join = lambda key: np.concatenate([sets[n][key] for n in sets])
    valid, distance, one_shot = join("valid"), join("distance"), join("one_shot")
    reward = _reward(distance, valid)
    bounds = np.cumsum([0, *sizes.values()])
    rows = {name: np.arange(bounds[i], bounds[i + 1]) for i, name in enumerate(sets)}
    train = np.zeros(len(valid), dtype=bool)
    train[rows["train"]] = True
    evidence = {name: np.concatenate([sets[n]["evidence"][name] for n in sets]) for name in EVIDENCE}
    rng = np.random.default_rng(0)
    shuffled = {name: evidence[name].copy() for name in EVIDENCE}
    order = {n: rng.permutation(len(rows[n])) for n in sets}  # one permutation per set, shared by both backends
    for name in EVIDENCE:
        for n in sets:
            shuffled[name][rows[n]] = evidence[name][rows[n]][order[n]]
    configs = {
        "A original features": one_shot,
        "B + SigLIP2 evidence": np.concatenate([one_shot, evidence["siglip"]], axis=-1),
        "C + Wikipedia evidence": np.concatenate([one_shot, evidence["dense"]], axis=-1),
        "D + both": np.concatenate([one_shot, evidence["siglip"], evidence["dense"]], axis=-1),
        "E + both, shuffled": np.concatenate([one_shot, shuffled["siglip"], shuffled["dense"]], axis=-1),
    }
    hits: dict[str, dict[float, np.ndarray]] = {}  # config -> threshold -> per-photo hit rate over seeds, all rows
    for name, features in configs.items():
        picked = [np.take_along_axis(distance, fit_scores(features, valid, reward, train, seed).argmax(1)[:, None], 1)[:, 0] for seed in SEEDS]
        hits[name] = {t: np.mean([p < t for p in picked], axis=0) for t in THRESHOLDS}
        print(f"fitted {name}", flush=True)

    report: dict[str, Any] = {"photos": sizes, "seeds": list(SEEDS), "evidence features": f"log1p count within {RADII_KM} km + log1p result count"}
    groups = {"dev": rows["dev"], "val": rows["val"]}
    benchmark = np.asarray([e["benchmark"] for e in sets["val"]["photos"]])
    groups |= {f"val {b}": rows["val"][benchmark == b] for b in ("im2gps3k", "yfcc4k")}
    for g, idx in groups.items():
        which = "dev" if g == "dev" else "val"
        original = sets[which]["reranker"][idx - rows[which][0]]
        print(f"\n{g} (n={len(idx)})   top-1 accuracy {'<1 km':>7s} {'<25 km':>7s} {'<200 km':>8s}   change vs A (<1 / <25 / <200 km); <25 km 95% CI")
        print(f"  {'original reranker (benchmark-tune fit)':40s} " + " ".join(f"{np.mean(original < t):7.1%}" for t in THRESHOLDS))
        report[g] = {"original reranker": {f"<{int(t)} km": float(np.mean(original < t)) for t in THRESHOLDS}}
        for name in configs:
            entry = {f"<{int(t)} km": float(hits[name][t][idx].mean()) for t in THRESHOLDS}
            line = f"  {name:40s} " + " ".join(f"{entry[f'<{int(t)} km']:7.1%}" for t in THRESHOLDS)
            if not name.startswith("A"):
                delta = {t: float(hits[name][t][idx].mean() - hits["A original features"][t][idx].mean()) for t in THRESHOLDS}
                ci = _bootstrap(hits[name][25.0][idx] - hits["A original features"][25.0][idx])
                entry |= {"change vs A": {f"<{int(t)} km": v for t, v in delta.items()}, "<25 km CI": ci}
                line += "   " + " ".join(f"{delta[t]:+6.1%}" for t in THRESHOLDS) + f"   [{ci[1]:+.1%}, {ci[2]:+.1%}]"
            report[g][name] = entry
            print(line)
    (ROOT / "evidence_ranker.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("select", "retrieve", "fit"))
    args = parser.parse_args(argv)
    {"select": select_train, "retrieve": retrieve_train, "fit": fit}[args.node]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
