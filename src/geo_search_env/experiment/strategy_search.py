# Shared world loading and gallery search for the benchmarks and MP16 queries, plus coarse (region / country) predictors.
# Usage: PYTHONPATH=src python -m geo_search_env.experiment.strategy_search {neighbors,region_head,coarse} --root artifacts/strategy_search

"""Galleries, queries and neighbour caches on the Flickr benchmarks; region heads and kNN votes as coarse predictors.

Galleries: MP16-Pro images (raw SigLIP2), Pinpoint's MP16 GPS gallery, and OSV-5M train images (raw SigLIP2).
Gallery rows by the same Flickr photographer as a query are excluded. Region = MP16-Pro (state, country),
projected onto queries, OSV images and VLM guesses by a majority-label grid built from MP16 coordinates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from ..paths import DATA_ROOT, HF_HOME
from ..data.benchmarks import load_benchmark
from .coarse_filter import UNKNOWN, LabelGrid


MP16_EMBED = DATA_ROOT / "mp16-embed/siglip2-giant-opt-patch16-384"
MP16_CSV = HF_HOME / "datasets/MP16-Pro/metadata/MP16_Pro_filtered.csv"
OSV_EMBED = DATA_ROOT / "osv5m-embed/siglip2-giant-opt-patch16-384"
BENCHMARK_NAMES = ("im2gps3k", "yfcc4k")
NEIGHBORS = 1_000
GPS_NEIGHBORS = 500
EARTH_KM = 6371.0088
THRESHOLDS_KM = (1.0, 25.0, 200.0, 750.0)


def _memmap_gallery(root: Path) -> dict[str, np.ndarray]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    count, dim = manifest["shapes"]["embeddings"]
    return {
        "embeddings": np.memmap(root / manifest["files"]["embeddings"], dtype=np.float16, mode="r", shape=(count, dim)),
        "latlon": np.fromfile(root / manifest["files"]["latlon_deg"], dtype=np.float32).reshape(count, 2).astype(np.float64),
        "row_index": np.fromfile(root / manifest["files"]["row_index"], dtype=np.int64),
    }


# PINPOINT_KEEP_OWN=1: benchmark queries carry no photographer, so no stage excludes the query photographer's own index photos (the original
# Pinpoint protocol, lesson 54's "kept" rows; the user asked for it on the test set, 2026-10-15). Default: excluded (our rule). For MP16 queries
# (train / dev3k, which are in the index) it drops only the query photo itself, as Pinpoint's training leave-one-out does (lesson 108).
KEEP_OWN = os.environ.get("PINPOINT_KEEP_OWN") == "1"


def load_world(*, with_pinpoint: bool = False, mp16_queries: Sequence[dict[str, Any]] | None = None, benchmarks: Sequence[str] | None = None) -> SimpleNamespace:
    """Galleries, shared admin labels and both benchmarks' queries (with a stable tune/eval split).

    With `mp16_queries` (dicts with "row", a position in MP16 embedding order), those MP16 photos are the queries instead; `benchmarks` picks other benchmark
    names than the default pair (e.g. ("wikimedia",)).
    """

    print("loading MP16-Pro metadata", flush=True)
    vocab: dict[str, dict[str, int]] = {"country": {}, "region": {}, "author": {}}
    authors, countries, regions = [], [], []
    with MP16_CSV.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        a, s, c = header.index("AUTHOR"), header.index("state"), header.index("country")
        for row in reader:
            authors.append(vocab["author"].setdefault(row[a], len(vocab["author"])))
            countries.append(vocab["country"].setdefault(row[c], len(vocab["country"])) if row[c] else UNKNOWN)
            region = f"{row[s]}|{row[c]}" if row[c] else ""
            regions.append(vocab["region"].setdefault(region, len(vocab["region"])) if region else UNKNOWN)
    mp16 = _memmap_gallery(MP16_EMBED)
    order = mp16["row_index"]
    mp16["author"] = np.asarray(authors, dtype=np.int64)[order]
    if KEEP_OWN:  # MP16 queries are in the index: each photo becomes its own "author", so exclusion drops only the query photo itself
        mp16["author"] = np.arange(len(order), dtype=np.int64)
    mp16["country"] = np.asarray(countries, dtype=np.int64)[order]
    mp16["region"] = np.asarray(regions, dtype=np.int64)[order]
    country_grid = LabelGrid(mp16["latlon"][:, 0], mp16["latlon"][:, 1], mp16["country"])
    region_grid = LabelGrid(mp16["latlon"][:, 0], mp16["latlon"][:, 1], mp16["region"])
    region_country = np.full(len(vocab["region"]), UNKNOWN, dtype=np.int64)
    known = (mp16["region"] != UNKNOWN) & (mp16["country"] != UNKNOWN)
    region_country[mp16["region"][known]] = mp16["country"][known]

    if (OSV_EMBED / "manifest.json").exists():
        osv = _memmap_gallery(OSV_EMBED)
    else:  # the OSV-5M embeddings were deleted (2026-10-12, disk space); Pinpoint's own OSV support cache lives in its index
        osv = {"embeddings": np.zeros((0, 1536), np.float16), "latlon": np.zeros((0, 2)), "row_index": np.zeros(0, np.int64)}
    osv["region"] = region_grid.lookup(osv["latlon"][:, 0], osv["latlon"][:, 1])
    osv["author"] = np.full(len(osv["latlon"]), -2, dtype=np.int64)  # never equal to a query author

    queries: list[dict[str, Any]] = []
    embeddings, latlon = [], []
    if mp16_queries is not None:
        rows = np.asarray([q["row"] for q in mp16_queries], dtype=np.int64)
        queries = [dict(q, benchmark="mp16", author=int(mp16["author"][q["row"]])) for q in mp16_queries]
        embeddings.append(np.asarray(mp16["embeddings"][rows], dtype=np.float32))
        latlon.append(mp16["latlon"][rows])
    for name in (benchmarks or BENCHMARK_NAMES) if mp16_queries is None else ():
        bench = load_benchmark(name)
        for i, image_id in enumerate(bench.image_ids):
            digest = int(hashlib.sha256(f"{name}:{image_id}".encode()).hexdigest()[:8], 16)
            queries.append({
                "benchmark": name,
                "image_id": image_id,
                "path": str(bench.image_paths[i]),
                "author": -1 if KEEP_OWN else vocab["author"].get(bench.authors[i], -1),
                "split": (bench.splits[i] if bench.splits and bench.splits[i] else "tune" if digest % 2 == 0 else "eval"),  # commons26: its dev / test
            })
        embeddings.append(bench.embeddings)
        latlon.append(bench.latlon)
    latlon = np.concatenate(latlon)
    world = SimpleNamespace(
        vocab=vocab,
        mp16=mp16,
        osv=osv,
        country_grid=country_grid,
        region_grid=region_grid,
        region_country=region_country,
        queries=queries,
        query_embeddings=np.concatenate(embeddings),
        query_latlon=latlon,
        query_country=country_grid.lookup(latlon[:, 0], latlon[:, 1]),
        query_region=region_grid.lookup(latlon[:, 0], latlon[:, 1]),
        query_author=np.asarray([q["author"] for q in queries], dtype=np.int64),
    )
    if with_pinpoint:
        from ..models.pinpoint import PinpointRetrievalBaseline

        world.baseline = PinpointRetrievalBaseline(device="auto")
    return world


def _stream_topk(array, queries, query_author, gallery_author, top_k: int, *, normalize: bool, chunk: int = 65_536):
    """Exact top-k over a disk-backed gallery for all queries at once, excluding same-author rows."""

    import torch
    import torch.nn.functional as F

    device = queries.device
    q_author = torch.as_tensor(query_author, device=device)
    best_values = torch.full((len(queries), 0), float("-inf"), device=device)
    best_indices = torch.zeros((len(queries), 0), dtype=torch.long, device=device)
    for start in range(0, len(array), chunk):
        block = torch.as_tensor(np.asarray(array[start : start + chunk], dtype=np.float16), device=device)
        if normalize:
            block = F.normalize(block.float(), dim=-1).half()
        scores = (queries @ block.T).float()
        authors = torch.as_tensor(gallery_author[start : start + chunk], device=device)
        scores.masked_fill_(authors.unsqueeze(0) == q_author.unsqueeze(1), float("-inf"))
        values, indices = torch.topk(scores, min(top_k, scores.shape[1]), dim=1)
        best_values = torch.cat((best_values, values), dim=1)
        best_indices = torch.cat((best_indices, indices + start), dim=1)
        keep = torch.topk(best_values, min(top_k, best_values.shape[1]), dim=1).indices
        best_values, best_indices = torch.gather(best_values, 1, keep), torch.gather(best_indices, 1, keep)
        if (start // chunk) % 16 == 0:
            print(f"  {start + len(block)}/{len(array)}", flush=True)
    return best_indices.cpu().numpy().astype(np.int32), best_values.cpu().numpy().astype(np.float32)


def neighbors(root: Path) -> None:
    """Cache every query's top neighbours in each gallery (same-photographer rows excluded)."""

    world = load_world(with_pinpoint=True)
    started = time.time()
    results = neighbor_cache(world, unfiltered_pinpoint=True)
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / "neighbors.npz", **results)
    (root / "queries.json").write_text(json.dumps(world.queries) + "\n", encoding="utf-8")
    print(f"neighbors cached in {time.time() - started:.0f}s", flush=True)


def neighbor_cache(world, *, unfiltered_pinpoint: bool = False, chunk: int = 65_536) -> dict[str, np.ndarray]:
    """Top neighbours of `world`'s queries in MP16 raw, Pinpoint's MP16 GPS gallery and OSV raw (needs `with_pinpoint`)."""

    import torch
    import torch.nn.functional as F

    device = torch.device("cuda")
    raw = F.normalize(torch.as_tensor(world.query_embeddings, device=device), dim=-1).half()
    projected = torch.as_tensor(world.baseline.project_image_embeddings(world.query_embeddings, source="mp16"), device=device).half()
    index_dir = world.baseline.index_dir
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    count, dim = manifest["shapes"]["gps_embeddings"]
    gps = np.memmap(index_dir / manifest["files"]["gps_embeddings"], dtype=np.float16, mode="r", shape=(count, dim))
    # Everything downstream indexes MP16 by our embedding-cache row. A version-6 index (model.zip) covers a subset of
    # MP16 in Pinpoint's own cache order, so map its rows through the MP16 CSV row into ours.
    csv_rows = world.baseline.mp16_csv_rows
    if csv_rows is None:
        gps_rows = np.arange(count)
        if count != len(world.mp16["author"]):
            raise ValueError(f"Pinpoint index has {count} rows, MP16 embedding cache {len(world.mp16['author'])}")
    else:
        ours = np.full(max(int(csv_rows.max()), int(world.mp16["row_index"].max())) + 1, -1, dtype=np.int64)
        ours[world.mp16["row_index"]] = np.arange(len(world.mp16["row_index"]))
        gps_rows = ours[csv_rows]
        if (gps_rows < 0).any():
            raise ValueError(f"{int((gps_rows < 0).sum())} Pinpoint index photos are missing from the MP16 embedding cache")
    gps_latlon = np.fromfile(index_dir / manifest["files"]["latlon_deg"], dtype=np.float32).reshape(count, 2)
    if not np.allclose(gps_latlon[::997], world.mp16["latlon"][gps_rows[::997]], atol=1e-4):
        raise ValueError("Pinpoint index rows do not line up with the MP16 embedding cache")
    gps_author = world.mp16["author"][gps_rows]

    results: dict[str, np.ndarray] = {}
    for name, array, queries, top_k, normalize, gallery_author in (
        ("mp16_raw", world.mp16["embeddings"], raw, NEIGHBORS, True, world.mp16["author"]),
        ("mp16_gps", gps, projected, GPS_NEIGHBORS, False, gps_author),
        ("osv_raw", world.osv["embeddings"], raw, NEIGHBORS, True, world.osv["author"]),
    ):
        print(f"gallery {name}", flush=True)
        results[f"{name}_idx"], results[f"{name}_sim"] = _stream_topk(array, queries, world.query_author, gallery_author, top_k, normalize=normalize, chunk=chunk)
    if unfiltered_pinpoint:
        # Unfiltered Pinpoint top-1, to compare against the submission's reported benchmark.
        no_author = np.full(len(world.query_author), -3, dtype=np.int64)
        results["mp16_gps_unfiltered_idx"], results["mp16_gps_unfiltered_sim"] = _stream_topk(gps, projected, no_author, gps_author, 1, normalize=False)
    for key in ("mp16_gps_idx", "mp16_gps_unfiltered_idx"):
        if key in results:
            results[key] = gps_rows[results[key]].astype(np.int32)
    return results


def region_head(root: Path, *, world=None, epochs: int = 2, block: int = 8_192, blocks_per_buffer: int = 128, batch: int = 8_192) -> None:
    """Linear and one-hidden-layer region classifiers on frozen MP16 SigLIP2 embeddings (query photographers held out)."""

    import torch
    import torch.nn.functional as F

    world = world or load_world()
    device = torch.device("cuda")
    regions = world.mp16["region"].copy()
    regions[np.isin(world.mp16["author"], world.query_author[world.query_author >= 0])] = UNKNOWN
    n_regions, dim = len(world.vocab["region"]), world.mp16["embeddings"].shape[1]
    models = {
        "head_linear": torch.nn.Linear(dim, n_regions),
        "head_mlp": torch.nn.Sequential(torch.nn.Linear(dim, 2048), torch.nn.GELU(), torch.nn.Linear(2048, n_regions)),
    }
    optimizers = {name: torch.optim.AdamW(m.to(device).parameters(), lr=1e-3, weight_decay=1e-4) for name, m in models.items()}
    starts = np.arange(0, len(regions), block)
    total_steps = epochs * math.ceil(len(regions) / batch) + len(starts)
    schedulers = {name: torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=total_steps) for name, opt in optimizers.items()}
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        order = rng.permutation(starts)
        for group in range(0, len(order), blocks_per_buffer):
            chosen = sorted(order[group : group + blocks_per_buffer])
            x = np.concatenate([np.asarray(world.mp16["embeddings"][s : s + block]) for s in chosen])
            y = np.concatenate([regions[s : s + block] for s in chosen])
            known = y != UNKNOWN
            x = F.normalize(torch.as_tensor(x[known], device=device).float(), dim=-1) * 30.0
            y = torch.as_tensor(y[known], device=device)
            permutation = torch.randperm(len(y), device=device)
            losses = {name: [] for name in models}
            for i in range(0, len(y), batch):
                take = permutation[i : i + batch]
                for name, model in models.items():
                    loss = F.cross_entropy(model(x[take]), y[take])
                    optimizers[name].zero_grad(set_to_none=True)
                    loss.backward()
                    optimizers[name].step()
                    schedulers[name].step()
                    losses[name].append(loss.item())
            print(f"epoch {epoch} buffer {group // blocks_per_buffer} " + " ".join(f"{n}={np.mean(v):.3f}" for n, v in losses.items()), flush=True)
    queries = F.normalize(torch.as_tensor(world.query_embeddings, device=device), dim=-1) * 30.0
    outputs: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for name, model in models.items():
            top = torch.topk(torch.softmax(model(queries), dim=-1), 50, dim=-1)
            outputs[f"{name}_regions"], outputs[f"{name}_probs"] = top.indices.cpu().numpy(), top.values.cpu().numpy()
    np.savez_compressed(root / "region_head.npz", **outputs)


def _gallery(world, name: str) -> dict[str, np.ndarray]:
    return world.osv if name.startswith("osv") else world.mp16


def _knn_distribution(world, cache, name: str, q: int, k: int, tau: float) -> dict[int, float]:
    sims = cache[f"{name}_sim"][q, :k]
    labels = _gallery(world, name)["region"][cache[f"{name}_idx"][q, :k]]
    weights = np.exp((sims - sims.max()) / tau)
    votes: dict[int, float] = {}
    for label, w in zip(labels, weights):
        votes[int(label)] = votes.get(int(label), 0.0) + float(w)
    total = sum(votes.values())
    return {label: w / total for label, w in votes.items()}


def _vlm_distribution(world, hypothesis: dict[str, Any] | None) -> dict[int, float]:
    if not hypothesis or not hypothesis["guesses"]:
        return {UNKNOWN: 1.0}
    g = np.asarray(hypothesis["guesses"], dtype=np.float64)
    votes: dict[int, float] = {}
    for label, p in zip(world.region_grid.lookup(g[:, 0], g[:, 1]), hypothesis["probabilities"]):
        votes[int(label)] = votes.get(int(label), 0.0) + p
    return votes


def _load_vlm(root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("vlm_*.jsonl")):
        model = path.stem.removeprefix("vlm_")
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
            out.setdefault(model, {})[row["episode_id"]] = row
    return out


def _distribution_metrics(world, predictions: Sequence[dict[int, float]], members: Sequence[int]) -> dict[str, float]:
    totals = dict(region_top1=0.0, region_top5=0.0, region_mass=0.0, country_top1=0.0, country_top3=0.0, country_mass=0.0)
    confidence, correct = [], []
    for q in members:
        prediction, truth, truth_country = predictions[q], int(world.query_region[q]), int(world.query_country[q])
        ranked = sorted(prediction.items(), key=lambda item: -item[1])
        countries: dict[int, float] = {}
        for region, p in ranked:
            key = int(world.region_country[region]) if region >= 0 else UNKNOWN
            countries[key] = countries.get(key, 0.0) + p
        ranked_countries = [c for c, _ in sorted(countries.items(), key=lambda item: -item[1])]
        top = [r for r, _ in ranked]
        totals["region_top1"] += top[:1] == [truth]
        totals["region_top5"] += truth in top[:5]
        totals["region_mass"] += prediction.get(truth, 0.0)
        totals["country_top1"] += ranked_countries[:1] == [truth_country]
        totals["country_top3"] += truth_country in ranked_countries[:3]
        totals["country_mass"] += countries.get(truth_country, 0.0)
        confidence.append(ranked[0][1])
        correct.append(top[:1] == [truth])
    n = len(members)
    result = {key: value / n for key, value in totals.items()}
    confident = np.argsort(confidence)[::-1][: n // 2]
    result["region_top1_confident_half"] = float(np.mean([correct[i] for i in confident]))
    result["n"] = n
    return result


def _members(world, benchmark: str, split: str, subset: set[str] | None = None) -> list[int]:
    return [
        i for i, q in enumerate(world.queries)
        if q["benchmark"] == benchmark and (split == "all" or q["split"] == split)
        and (subset is None or f"{q['benchmark']}:{q['image_id']}" in subset)
    ]


def coarse(root: Path) -> None:
    """Region/country accuracy of kNN votes per gallery, trained heads and zero-shot VLMs."""

    world = load_world()
    cache = np.load(root / "neighbors.npz")
    n = len(world.queries)
    predictors: dict[str, list[dict[int, float]]] = {}
    for gallery in ("mp16_raw", "osv_raw", "mp16_gps"):
        for tau in (0.01, 0.03, 0.1):
            for k in (10, 100):
                predictors[f"knn_{gallery}|tau={tau}|k={k}"] = [_knn_distribution(world, cache, gallery, q, k, tau) for q in range(n)]
    if (root / "region_head.npz").exists():
        heads = np.load(root / "region_head.npz")
        for name in ("head_linear", "head_mlp"):
            predictors[name] = [{int(r): float(p) for r, p in zip(heads[f"{name}_regions"][q], heads[f"{name}_probs"][q])} for q in range(n)]
    vlm = _load_vlm(root)
    vlm_subsets = {model: set(rows) for model, rows in vlm.items()}
    for model, rows in vlm.items():
        predictors[f"vlm_{model}"] = [_vlm_distribution(world, rows.get(f"{q['benchmark']}:{q['image_id']}")) for q in world.queries]

    report: dict[str, Any] = {"selected": {}}
    for benchmark in BENCHMARK_NAMES:
        for split in ("tune", "eval"):
            report[f"{benchmark}/{split}"] = {name: _distribution_metrics(world, rows, _members(world, benchmark, split)) for name, rows in predictors.items() if not name.startswith("vlm_")}
    # Hyperparameters chosen on the pooled tune splits.
    for name in predictors:
        family = name.split("|")[0]
        if family.startswith("vlm_"):
            report["selected"][family] = name
            continue
        score = sum(report[f"{b}/tune"][name]["region_mass"] for b in BENCHMARK_NAMES)
        if family not in report["selected"] or score > sum(report[f"{b}/tune"][report["selected"][family]]["region_mass"] for b in BENCHMARK_NAMES):
            report["selected"][family] = name
    # VLMs ran on a subset: compare every selected predictor on exactly that subset.
    for model, subset in vlm_subsets.items():
        for benchmark in BENCHMARK_NAMES:
            members = _members(world, benchmark, "all", subset)
            if members:
                report[f"{benchmark}/vlm_subset_{model}"] = {name: _distribution_metrics(world, predictors[name], members) for name in report["selected"].values()}
    (root / "coarse.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for key in [k for k in report if "/" in k and not k.endswith("/tune")]:
        print(f"\n{key}")
        print(f"  {'predictor':34s} regTop1 regTop5 regMass ctyTop1 ctyTop3 regTop1@confHalf")
        for family, name in report["selected"].items():
            if name in report[key]:
                m = report[key][name]
                print(f"  {family:34s} {m['region_top1']:6.0%} {m['region_top5']:7.0%} {m['region_mass']:7.2f} {m['country_top1']:7.0%} {m['country_top3']:7.0%} {m['region_top1_confident_half']:10.0%}   n={m['n']}")


def _haversine_km(latitude: float, longitude: float, coordinates: np.ndarray) -> np.ndarray:
    lat1, lon1 = math.radians(latitude), math.radians(longitude)
    lat2, lon2 = np.radians(coordinates[:, 0]), np.radians(coordinates[:, 1])
    h = np.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _geoguessr(distances: np.ndarray) -> np.ndarray:
    return np.round(5000 * np.exp(-distances / 1492.7))


def _xyz(latlon: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(latlon[..., 0]), np.radians(latlon[..., 1])
    return np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)


def _bootstrap(delta: np.ndarray, resamples: int = 2000) -> tuple[float, float, float]:
    """Mean of a paired per-photo difference and its bootstrap 95% interval."""

    rng = np.random.default_rng(0)
    means = delta[rng.integers(0, len(delta), (resamples, len(delta)))].mean(1)
    return float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("neighbors", "region_head", "coarse"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/strategy_search"))
    args = parser.parse_args(argv)
    {"neighbors": neighbors, "region_head": region_head, "coarse": coarse}[args.node](args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
