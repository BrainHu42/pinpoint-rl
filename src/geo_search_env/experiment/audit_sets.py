# Audit of the train / dev / val photo sets: leakage, near-duplicates, label quality, unlocatable photos, representativeness, clustering.
# Usage: .venv/bin/python -m geo_search_env.experiment.audit_sets   (CPU; needs the train/dev/val photo lists and the neighbour caches)

"""Checks, each printed with the numbers that decide it:

1 photographers  overlap between train, dev and val photographers; photos per photographer
2 duplicates     near-duplicate (SigLIP2 cosine >= 0.95) photos between train and dev / val, inside each set, and against the gallery
                 (top raw neighbour after same-photographer exclusion), with the reranker top-1 accuracy on them
3 labels         share of truths on a coarse coordinate grid, and photos sharing the exact truth coordinate in the MP16 gallery
4 placeholders   photos that look like Flickr's "photo is no longer available" image (embedding match to a known one)
5 countries      top countries of each set
6 clustering     standard error of the reranker top-1 hit rate by photo vs by photographer cluster
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .query_evidence import ROOT, SFT_ROOT, BENCH_ROOT
from .strategy_search import load_world
from .wiki_backend import _km


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def cluster_se(hit: np.ndarray, cluster: np.ndarray) -> tuple[float, float]:
    """Standard error of the mean hit rate by photo, and by cluster (cluster-mean estimator)."""

    by_photo = hit.std(ddof=1) / np.sqrt(len(hit))
    ids = np.unique(cluster)
    sums = np.asarray([hit[cluster == c].sum() for c in ids])
    sizes = np.asarray([(cluster == c).sum() for c in ids])
    ratio = hit.sum() / len(hit)
    residual = sums - ratio * sizes
    by_cluster = np.sqrt(len(ids) / (len(ids) - 1) * (residual**2).sum()) / len(hit)
    return float(by_photo), float(by_cluster)


def main() -> int:
    train = json.loads((ROOT / "train" / "dev.json").read_text(encoding="utf-8"))
    dev = json.loads((ROOT / "dev" / "dev.json").read_text(encoding="utf-8"))
    val = json.loads((ROOT / "val" / "dev.json").read_text(encoding="utf-8"))
    mp16 = load_world(mp16_queries=[{"row": e["row"], "image_id": e["image_id"]} for e in train + dev])
    bench = load_world()
    n_train = len(train)
    author = {"train": mp16.query_author[:n_train], "dev": mp16.query_author[n_train:], "val": bench.query_author[[e["index"] for e in val]]}
    emb = {"train": normalize(mp16.query_embeddings[:n_train]), "dev": normalize(mp16.query_embeddings[n_train:]),
           "val": normalize(bench.query_embeddings[[e["index"] for e in val]])}
    truth = {"train": np.asarray([e["truth"] for e in train]), "dev": np.asarray([e["truth"] for e in dev]), "val": np.asarray([e["truth"] for e in val])}
    sets = {"dev": dev, "val": val}
    top1 = {k: np.asarray([_km(np.asarray(e["pool"][:1]), *e["truth"])[0] < 25 for e in v]) for k, v in sets.items()}
    top1_1km = {k: np.asarray([_km(np.asarray(e["pool"][:1]), *e["truth"])[0] < 1 for e in v]) for k, v in sets.items()}

    print("1. PHOTOGRAPHERS")
    for name, a in author.items():
        known = a[a >= 0]
        counts = Counter(known.tolist())
        print(f"  {name}: {len(a)} photos, {len(counts)} distinct known photographers, {int((a < 0).sum())} unknown; max photos per photographer {max(counts.values()) if counts else 0}; "
              f"photographers with >= 3 photos {sum(v >= 3 for v in counts.values())}")
    tr = set(author["train"][author["train"] >= 0].tolist())
    for name in ("dev", "val"):
        shared = [x for x in author[name].tolist() if x >= 0 and x in tr]
        print(f"  {name} photos whose photographer also has TRAIN photos: {len(shared)} of {len(author[name])}")
    print(f"  dev photographers also in val: {len(set(author['dev'][author['dev'] >= 0].tolist()) & set(author['val'][author['val'] >= 0].tolist()))}")

    print("\n2. NEAR-DUPLICATES (cosine >= 0.95)")
    for name in ("dev", "val"):
        sim = emb[name] @ emb["train"].T
        best = sim.max(1)
        within = emb[name] @ emb[name].T
        np.fill_diagonal(within, -1)
        print(f"  {name} vs TRAIN photos: max cosine >= 0.95 for {int((best >= 0.95).sum())}, >= 0.90 for {int((best >= 0.90).sum())} of {len(best)}; "
              f"inside {name}: pairs >= 0.95 {int((within >= 0.95).sum() // 2)}, photos with a duplicate {int((within.max(1) >= 0.95).sum())}")
        path = (SFT_ROOT if name == "dev" else BENCH_ROOT) / "neighbors.npz"
        with np.load(path) as saved:
            rows = [e["index"] for e in sets[name]]
            raw = np.maximum(saved["mp16_raw_sim"][rows, 0], saved["osv_raw_sim"][rows, 0])
        (ROOT / name / "near_duplicate.json").write_text(json.dumps([e["image_id"] for e, r in zip(sets[name], raw) if r >= 0.95]) + "\n", encoding="utf-8")
        for t in (0.95, 0.90):
            m = raw >= t
            print(f"  {name} top raw neighbour in the retrieval gallery >= {t}: {int(m.sum())} photos; reranker top-1 <25 km on them {top1[name][m].mean() if m.any() else float('nan'):.0%} "
                  f"(others {top1[name][~m].mean():.0%}); <1 km {top1_1km[name][m].mean() if m.any() else float('nan'):.0%} (others {top1_1km[name][~m].mean():.0%})")

    print("\n3. LABELS")
    latlon = mp16.mp16["latlon"]
    key = lambda p: (np.round(p[:, 0] * 1e5).astype(np.int64) + 9_000_000) * 40_000_000 + (np.round(p[:, 1] * 1e5).astype(np.int64) + 18_000_000)
    gallery_keys = np.sort(key(latlon))
    for name in ("train", "dev", "val"):
        t = truth[name]
        coarse2 = np.mean((np.abs(t - np.round(t, 2)) < 1e-4).all(1))
        coarse1 = np.mean((np.abs(t - np.round(t, 1)) < 1e-4).all(1))
        k = key(t)
        count = np.searchsorted(gallery_keys, k, "right") - np.searchsorted(gallery_keys, k, "left")
        print(f"  {name}: on a 0.01 deg grid {coarse2:.1%} (chance ~0.04%), on a 0.1 deg grid {coarse1:.1%}; MP16 gallery photos at the exact truth coordinate: "
              f"median {int(np.median(count))}, >= 5 for {np.mean(count >= 5):.0%}, >= 50 for {np.mean(count >= 50):.0%}")
        if name in top1:
            m = count >= 5
            print(f"      reranker top-1 <1 km where >= 5 gallery photos share the coordinate: {top1_1km[name][m].mean() if m.any() else float('nan'):.0%} (n={int(m.sum())}) vs {top1_1km[name][~m].mean():.0%} elsewhere")

    print("\n4. PLACEHOLDERS (Flickr 'photo is no longer available')")
    ref = next((e for e in val if abs(e["truth"][0] - 52.99) < 0.01 and abs(e["truth"][1] + 3.19) < 0.01), None)
    if ref is not None:
        r = emb["val"][[e["image_id"] for e in val].index(ref["image_id"])]
        for name in ("train", "dev", "val"):
            s = emb[name] @ r
            print(f"  {name}: cosine to the known placeholder >= 0.95: {int((s >= 0.95).sum())}, >= 0.90: {int((s >= 0.90).sum())} of {len(s)}")
        (ROOT / "val" / "exclude.json").write_text(json.dumps([e["image_id"] for e, s in zip(val, emb["val"] @ r) if s >= 0.95]) + "\n", encoding="utf-8")

    print("\n5. COUNTRIES (share of photos)")
    names = {v: k for k, v in mp16.vocab["country"].items()}
    for name in ("train", "dev", "val"):
        ids = mp16.country_grid.lookup(truth[name][:, 0], truth[name][:, 1])
        top = Counter(names.get(int(i), "unknown") for i in ids).most_common(7)
        print(f"  {name}: " + ", ".join(f"{c} {n / len(ids):.0%}" for c, n in top))

    print("\n6. CLUSTERING (standard error of the reranker top-1 <25 km hit rate)")
    for name in ("dev", "val"):
        a = author[name].copy()
        unknown = a < 0
        a[unknown] = 10_000_000 + np.arange(unknown.sum())  # unknown photographers each count as their own cluster (a lower bound)
        s_photo, s_cluster = cluster_se(top1[name].astype(float), a)
        print(f"  {name}: by photo {s_photo:.4f}, by photographer {s_cluster:.4f} (design effect {(s_cluster / s_photo) ** 2:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
