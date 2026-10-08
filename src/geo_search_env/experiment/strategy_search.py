# Exploratory tree search over agent training setups on im2gps3k / yfcc4k, one cheap offline proxy per node.
# Usage: PYTHONPATH=src python -m geo_search_env.experiment.strategy_search {neighbors,region_head,coarse,pools,search} --root artifacts/strategy_search

"""Offline strategy search on the Flickr benchmarks: coarse predictors, candidate pools and local-search ceilings.

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
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from ..data.benchmarks import compute_metrics, load_benchmark
from .coarse_filter import UNKNOWN, LabelGrid


MP16_EMBED = Path("/data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384")
MP16_CSV = Path("/data/hf/datasets/MP16-Pro/metadata/MP16_Pro_filtered.csv")
OSV_EMBED = Path("/data/pinpoint/osv5m-embed/siglip2-giant-opt-patch16-384")
BENCHMARK_NAMES = ("im2gps3k", "yfcc4k")
NEIGHBORS = 1_000
GPS_NEIGHBORS = 500
EARTH_KM = 6371.0088
THRESHOLDS_KM = (1.0, 25.0, 200.0, 750.0)
POOL_BUDGETS = (1, 5, 10, 25, 50)
PRIOR_LAMBDAS = (0.0, 0.001, 0.003, 0.01, 0.03)
PRIOR_FLOOR = 0.01


def _memmap_gallery(root: Path) -> dict[str, np.ndarray]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    count, dim = manifest["shapes"]["embeddings"]
    return {
        "embeddings": np.memmap(root / manifest["files"]["embeddings"], dtype=np.float16, mode="r", shape=(count, dim)),
        "latlon": np.fromfile(root / manifest["files"]["latlon_deg"], dtype=np.float32).reshape(count, 2).astype(np.float64),
        "row_index": np.fromfile(root / manifest["files"]["row_index"], dtype=np.int64),
    }


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
    mp16["country"] = np.asarray(countries, dtype=np.int64)[order]
    mp16["region"] = np.asarray(regions, dtype=np.int64)[order]
    country_grid = LabelGrid(mp16["latlon"][:, 0], mp16["latlon"][:, 1], mp16["country"])
    region_grid = LabelGrid(mp16["latlon"][:, 0], mp16["latlon"][:, 1], mp16["region"])
    region_country = np.full(len(vocab["region"]), UNKNOWN, dtype=np.int64)
    known = (mp16["region"] != UNKNOWN) & (mp16["country"] != UNKNOWN)
    region_country[mp16["region"][known]] = mp16["country"][known]

    osv = _memmap_gallery(OSV_EMBED)
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
                "author": vocab["author"].get(bench.authors[i], -1),
                "split": "tune" if digest % 2 == 0 else "eval",
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

    results: dict[str, np.ndarray] = {}
    for name, array, queries, top_k, normalize, gallery_author in (
        ("mp16_raw", world.mp16["embeddings"], raw, NEIGHBORS, True, world.mp16["author"]),
        ("mp16_gps", gps, projected, GPS_NEIGHBORS, False, world.mp16["author"]),
        ("osv_raw", world.osv["embeddings"], raw, NEIGHBORS, True, world.osv["author"]),
    ):
        print(f"gallery {name}", flush=True)
        results[f"{name}_idx"], results[f"{name}_sim"] = _stream_topk(array, queries, world.query_author, gallery_author, top_k, normalize=normalize, chunk=chunk)
    if unfiltered_pinpoint:
        # Unfiltered Pinpoint top-1, to compare against the submission's reported benchmark.
        no_author = np.full(len(world.query_author), -3, dtype=np.int64)
        results["mp16_gps_unfiltered_idx"], results["mp16_gps_unfiltered_sim"] = _stream_topk(gps, projected, no_author, world.mp16["author"], 1, normalize=False)
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


def _pool(rows: Sequence[tuple[float, float, float]], limit: int = 50, radius_km: float = 1.0) -> list[tuple[float, float]]:
    """Cluster (lat, lon, score) hits, highest score first, into distinct 1 km location proposals.

    Vectorized equivalent of cluster_candidate_locations: a hit seeds a new cluster unless an earlier seed lies
    within the radius, joins the earliest such seed, and clusters are score-weighted spherical centroids.
    """

    if not rows:
        return []
    array = np.asarray(rows, dtype=np.float64)
    array = array[np.argsort(-array[:, 2], kind="stable")]
    lat, lon = np.radians(array[:, 0]), np.radians(array[:, 1])
    xyz = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=1)
    near = xyz @ xyz.T >= math.cos(radius_km / EARTH_KM)
    covered = np.zeros(len(array), dtype=bool)
    seeds: list[int] = []
    position = 0
    while len(seeds) < limit:
        uncovered = np.flatnonzero(~covered[position:])
        if not len(uncovered):
            break
        seed = position + int(uncovered[0])
        seeds.append(seed)
        covered |= near[seed]
        position = seed + 1
    membership = near[:, seeds]
    assigned = membership.any(axis=1)
    cluster = np.where(assigned, membership.argmax(axis=1), -1)
    proposals = []
    for c, seed in enumerate(seeds):
        members = np.flatnonzero(cluster == c)
        scores = array[members, 2]
        weights = np.exp(np.clip((scores - scores.max()) / 0.05, -20.0, 0.0))
        centre = (xyz[members] * np.maximum(weights, 1e-6)[:, None]).sum(axis=0)
        norm = np.linalg.norm(centre)
        if norm <= 1e-12:
            proposals.append((float(array[seed, 0]), float(array[seed, 1])))
            continue
        centre /= norm
        proposals.append((math.degrees(math.atan2(centre[2], math.hypot(centre[0], centre[1]))), math.degrees(math.atan2(centre[1], centre[0]))))
    return proposals


def _hits(world, cache, name: str, q: int, limit: int | None = None) -> list[tuple[float, float, float, int]]:
    gallery = _gallery(world, name)
    idx, sim = cache[f"{name}_idx"][q][:limit], cache[f"{name}_sim"][q][:limit]
    finite = np.isfinite(sim)
    xy, labels = gallery["latlon"][idx[finite]], gallery["region"][idx[finite]]
    return [(float(a), float(b), float(s), int(r)) for (a, b), s, r in zip(xy, sim[finite], labels)]


def build_pools(world, cache, head_predictions, head: str, lambdas: Sequence[float]) -> list[dict[str, list[tuple[float, float]]]]:
    pools: list[dict[str, list[tuple[float, float]]]] = []
    for q in range(len(world.queries)):
        truth_region = int(world.query_region[q])
        mp16_raw, osv_raw, gps = (_hits(world, cache, name, q) for name in ("mp16_raw", "osv_raw", "mp16_gps"))
        arms = {
            "pinpoint_gps": _pool([h[:3] for h in gps]),
            "mp16_raw": _pool([h[:3] for h in mp16_raw]),
            "osv_raw": _pool([h[:3] for h in osv_raw]),
        }
        combined = mp16_raw + osv_raw
        arms["mp16+osv_raw"] = _pool([h[:3] for h in combined])
        arms["mp16+osv_raw|oracle_region"] = _pool([h[:3] for h in combined if h[3] == truth_region])
        if head_predictions is not None:
            log_prior = np.log(np.asarray([head_predictions[q].get(h[3], 0.0) for h in combined]) + PRIOR_FLOOR)
            for lam in lambdas:
                if lam > 0:
                    arms[f"mp16+osv_raw|{head}|lambda={lam}"] = _pool([(a, b, s + lam * lp) for (a, b, s, _), lp in zip(combined, log_prior)])
        pools.append(arms)
        if (q + 1) % 1000 == 0:
            print(f"pools {q + 1}/{len(world.queries)}", flush=True)
    return pools


def pools(root: Path) -> None:
    """Candidate-location recall and top-1 benchmark metrics per candidate pool."""

    world = load_world()
    cache = np.load(root / "neighbors.npz")
    coarse_report = json.loads((root / "coarse.json").read_text(encoding="utf-8"))
    heads = np.load(root / "region_head.npz") if (root / "region_head.npz").exists() else None
    head = max(("head_linear", "head_mlp"), key=lambda name: sum(coarse_report[f"{b}/tune"][name]["region_mass"] for b in BENCHMARK_NAMES)) if heads is not None else None
    head_predictions = [{int(r): float(p) for r, p in zip(heads[f"{head}_regions"][q], heads[f"{head}_probs"][q])} for q in range(len(world.queries))] if heads is not None else None
    all_pools = build_pools(world, cache, head_predictions, head, PRIOR_LAMBDAS)

    arms = list(all_pools[0])
    distances = {arm: np.full((len(world.queries), max(POOL_BUDGETS)), np.inf) for arm in arms}
    for q, pool in enumerate(all_pools):
        for arm, locations in pool.items():
            if locations:
                d = _haversine_km(*world.query_latlon[q], np.asarray(locations))[: max(POOL_BUDGETS)]
                distances[arm][q, : len(d)] = d
    best_so_far = {arm: np.minimum.accumulate(d, axis=1) for arm, d in distances.items()}

    unfiltered = world.mp16["latlon"][cache["mp16_gps_unfiltered_idx"][:, 0]]
    summary: dict[str, Any] = {"head": head}
    for benchmark in BENCHMARK_NAMES:
        for split in ("tune", "eval", "all"):
            members = _members(world, benchmark, split)
            entry: dict[str, Any] = {"n": len(members)}
            entry["pinpoint_gps_unfiltered_top1"] = compute_metrics(unfiltered[members], world.query_latlon[members])
            for arm in arms:
                top1 = np.asarray([all_pools[q][arm][0] if all_pools[q][arm] else (0.0, 0.0) for q in members])
                entry[arm] = {
                    "top1": compute_metrics(top1, world.query_latlon[members]),
                    "recall": {f"le_{t:g}km@{b}": float((best_so_far[arm][members, b - 1] <= t).mean()) for b in POOL_BUDGETS for t in THRESHOLDS_KM},
                }
            summary[f"{benchmark}/{split}"] = entry
    # Choose the prior strength on tune (oracle-free objective: recall within 25 km at 10 candidates).
    prior_arms = [arm for arm in arms if "|lambda=" in arm] + ["mp16+osv_raw"]
    summary["selected_prior_arm"] = max(prior_arms, key=lambda arm: sum(summary[f"{b}/tune"][arm]["recall"]["le_25km@10"] for b in BENCHMARK_NAMES))
    (root / "pools.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for benchmark in BENCHMARK_NAMES:
        entry = summary[f"{benchmark}/eval"]
        print(f"\n{benchmark}/eval n={entry['n']}  (top-1 accuracy <1/25/200/750 km | candidate recall <1/25 km @10 and @50)")
        m = entry["pinpoint_gps_unfiltered_top1"]
        print(f"  {'pinpoint_gps (no author filter)':44s} top1 " + " ".join(f"{m[f'Under_{t}_km']:5.1%}" for t in (1, 25, 200, 750)))
        for arm in arms:
            m, r = entry[arm]["top1"], entry[arm]["recall"]
            print(f"  {arm:44s} top1 " + " ".join(f"{m[f'Under_{t}_km']:5.1%}" for t in (1, 25, 200, 750)) + " | " + " ".join(f"{r[k]:5.1%}" for k in ("le_1km@10", "le_25km@10", "le_1km@50", "le_25km@50")))
    print("selected prior arm:", summary["selected_prior_arm"])


FUSION_SOURCES = ("pinpoint_gps", "mp16_raw", "osv_raw")


def _fusion_features(world, cache) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-query candidate coordinates [Q, S, 2] and selector features from the cached neighbour lists."""

    top = {
        "pinpoint_gps": world.mp16["latlon"][cache["mp16_gps_idx"][:, 0]],
        "mp16_raw": world.mp16["latlon"][cache["mp16_raw_idx"][:, 0]],
        "osv_raw": world.osv["latlon"][cache["osv_raw_idx"][:, 0]],
    }
    candidates = np.stack([top[s] for s in FUSION_SOURCES], axis=1)
    features, names = [], []

    def add(name: str, values: np.ndarray) -> None:
        names.append(name)
        features.append(values.astype(np.float64))

    for key in ("mp16_gps", "mp16_raw", "osv_raw"):
        sims = cache[f"{key}_sim"]
        add(f"{key}_sim1", sims[:, 0])
        add(f"{key}_margin", sims[:, 0] - sims[:, 9])
    for key, gallery in (("mp16_raw", world.mp16), ("mp16_gps", world.mp16)):
        # Agreement among the top 10 hits: how many sit within 1 km / 25 km of the first.
        coords = gallery["latlon"][cache[f"{key}_idx"][:, :10]]
        for radius in (1.0, 25.0):
            add(f"{key}_top10_within_{radius:g}km", np.asarray([(_haversine_km(*c[0], c[1:]) <= radius).sum() for c in coords]))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        d = np.asarray([_haversine_km(*a, b[None])[0] for a, b in zip(candidates[:, i], candidates[:, j])])
        add(f"log_km_{FUSION_SOURCES[i]}_{FUSION_SOURCES[j]}", np.log1p(d))
    return candidates, np.stack(features, axis=1), names


def _geoguessr(distances: np.ndarray) -> np.ndarray:
    return np.round(5000 * np.exp(-distances / 1492.7))


def fusion(root: Path) -> None:
    """How much a learned selector between coarse (Pinpoint) and fine (raw image kNN) answers could gain, vs rules and oracle."""

    import torch

    world = load_world()
    cache = np.load(root / "neighbors.npz")
    candidates, features, names = _fusion_features(world, cache)
    distance = np.stack([[_haversine_km(*world.query_latlon[q], candidates[q, s][None])[0] for s in range(len(FUSION_SOURCES))] for q in range(len(world.queries))])
    tune = np.asarray([q["split"] == "tune" for q in world.queries])
    benchmark = np.asarray([q["benchmark"] for q in world.queries])

    choices: dict[str, np.ndarray] = {s: np.full(len(world.queries), i) for i, s in enumerate(FUSION_SOURCES)}
    choices["oracle(pinpoint|mp16_raw)"] = np.argmin(distance[:, :2], axis=1)
    choices["oracle(all three)"] = np.argmin(distance, axis=1)

    # Threshold rules tuned on the tune split to maximise GeoGuessr score.
    def tuned(rule_values: np.ndarray, grid: np.ndarray, *, above: bool) -> np.ndarray:
        best, best_score = None, -1.0
        for threshold in grid:
            pick = np.where((rule_values >= threshold) if above else (rule_values <= threshold), 1, 0)
            score = _geoguessr(distance[tune, :][np.arange(tune.sum()), pick[tune]]).mean()
            if score > best_score:
                best, best_score = pick, score
        return best

    raw_sim = features[:, names.index("mp16_raw_sim1")]
    agree = features[:, names.index("log_km_pinpoint_gps_mp16_raw")]
    choices["rule: raw if sim>=t"] = tuned(raw_sim, np.quantile(raw_sim, np.linspace(0, 1, 101)), above=True)
    choices["rule: raw if near pinpoint"] = tuned(agree, np.log1p(np.asarray([0.5, 1, 5, 25, 100, 200, 500, 1000])), above=False)

    # Supervised selector: softmax regression over the three sources, reward-weighted (target = best GeoGuessr source).
    x = torch.as_tensor((features - features[tune].mean(0)) / (features[tune].std(0) + 1e-9), dtype=torch.float32)
    reward = torch.as_tensor(_geoguessr(distance) / 5000.0, dtype=torch.float32)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 32), torch.nn.GELU(), torch.nn.Linear(32, len(FUSION_SOURCES)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    tune_idx = torch.as_tensor(np.flatnonzero(tune))
    for _ in range(2_000):
        # Expected reward under the policy: the policy-gradient objective an RL learner would optimise, in closed form.
        probabilities = torch.softmax(model(x[tune_idx]), dim=-1)
        loss = -(probabilities * reward[tune_idx]).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        choices["learned selector (expected-reward)"] = model(x).argmax(-1).numpy()

    report: dict[str, Any] = {"features": names}
    for name in BENCHMARK_NAMES:
        for split in ("eval", "all"):
            members = np.flatnonzero((benchmark == name) & (~tune if split == "eval" else True))
            report[f"{name}/{split}"] = {
                arm: compute_metrics(candidates[members, pick[members]], world.query_latlon[members]) | {"picked_raw": float((pick[members] != 0).mean())}
                for arm, pick in choices.items()
            }
    (root / "fusion.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name in BENCHMARK_NAMES:
        print(f"\n{name}/eval   <1km   <25km  <200km <750km <2500km  GeoGuessr  non-pinpoint picks")
        for arm, m in report[f"{name}/eval"].items():
            print(f"  {arm:38s}" + "".join(f"{m[f'Under_{t}_km']:7.1%}" for t in (1, 25, 200, 750, 2500)) + f"  {m['Geoguessr_score']:7.0f}  {m['picked_raw']:6.0%}")


SEARCH_PER_SOURCE = 10
SEARCH_MAX_CANDIDATES = 2 * SEARCH_PER_SOURCE
SEARCH_PRIOR_LAMBDA = 0.01  # selected on tune by the pools node
VERIFY_RADIUS_KM = (1.0, 5.0)
VERIFY_NEAREST = 3_000
VERIFY_BUDGETS = (1, 3, 5, 10, 20)
ONE_SHOT_FEATURES = [
    "pinpoint_inv_rank", "raw_inv_rank",
    *[f"{key}_{stat}" for key in ("mp16_raw", "osv_raw", "mp16_gps") for stat in ("maxsim_1km_rel", "log_n_1km", "log_n_25km")],
    "log_head_prior", "log_gps_vote_prior", "log_km_to_pinpoint_top1", "log_km_to_raw_top1",
]
VERIFY_FEATURES = ["verify_max_1km_rel", "verify_top5_1km_rel", "verify_log_n_1km", "verify_max_5km_rel", "verify_top5_5km_rel", "verify_log_n_5km"]


def _xyz(latlon: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(latlon[..., 0]), np.radians(latlon[..., 1])
    return np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)


def _search_candidates(world, cache, head_predictions, gps_votes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interleave Pinpoint and prior-weighted photo-matching candidates, dedupe within 1 km; compute one-shot features."""

    n = len(world.queries)
    coords = np.zeros((n, SEARCH_MAX_CANDIDATES, 2))
    valid = np.zeros((n, SEARCH_MAX_CANDIDATES), dtype=bool)
    features = np.zeros((n, SEARCH_MAX_CANDIDATES, len(ONE_SHOT_FEATURES)), dtype=np.float32)
    cos_1km, cos_25km = math.cos(1.0 / EARTH_KM), math.cos(25.0 / EARTH_KM)
    for q in range(n):
        mp16_raw, osv_raw, gps = (_hits(world, cache, name, q) for name in ("mp16_raw", "osv_raw", "mp16_gps"))
        pinpoint = _pool([h[:3] for h in gps])[:SEARCH_PER_SOURCE]
        combined = mp16_raw + osv_raw
        log_prior = np.log(np.asarray([head_predictions[q].get(h[3], 0.0) for h in combined]) + PRIOR_FLOOR)
        raw = _pool([(a, b, s + SEARCH_PRIOR_LAMBDA * lp) for (a, b, s, _), lp in zip(combined, log_prior)])[:SEARCH_PER_SOURCE]
        chosen: list[tuple[float, float]] = []
        ranks: list[list[int]] = []  # [pinpoint rank, raw rank], 0 = absent
        for i in range(SEARCH_PER_SOURCE):
            for source, pool in ((0, pinpoint), (1, raw)):
                if i >= len(pool):
                    continue
                point = _xyz(np.asarray(pool[i]))
                existing = [j for j, c in enumerate(chosen) if point @ _xyz(np.asarray(c)) >= cos_1km]
                if existing:
                    if ranks[existing[0]][source] == 0:
                        ranks[existing[0]][source] = i + 1
                    continue
                chosen.append(pool[i])
                ranks.append([i + 1 if source == 0 else 0, i + 1 if source == 1 else 0])
        k = len(chosen)
        coords[q, :k], valid[q, :k] = chosen, True
        cand_xyz = _xyz(coords[q, :k])
        f = features[q, :k]
        rank = np.asarray(ranks, dtype=np.float32)
        f[:, 0], f[:, 1] = np.where(rank[:, 0] > 0, 1 / np.maximum(rank[:, 0], 1), 0), np.where(rank[:, 1] > 0, 1 / np.maximum(rank[:, 1], 1), 0)
        column = 2
        for key in ("mp16_raw", "osv_raw", "mp16_gps"):
            hits = {"mp16_raw": mp16_raw, "osv_raw": osv_raw, "mp16_gps": gps}[key]
            hit_xyz = _xyz(np.asarray([h[:2] for h in hits]))
            sims = np.asarray([h[2] for h in hits])
            dots = cand_xyz @ hit_xyz.T
            within_1, within_25 = dots >= cos_1km, dots >= cos_25km
            best = np.where(within_1, sims[None], -np.inf).max(axis=1)
            f[:, column] = np.where(np.isfinite(best), best - sims.max(), sims.min() - sims.max())
            f[:, column + 1], f[:, column + 2] = np.log1p(within_1.sum(1)), np.log1p(within_25.sum(1))
            column += 3
        regions = world.region_grid.lookup(coords[q, :k, 0], coords[q, :k, 1])
        f[:, column] = np.log([head_predictions[q].get(int(r), 0.0) + PRIOR_FLOOR for r in regions])
        f[:, column + 1] = np.log([gps_votes[q].get(int(r), 0.0) + PRIOR_FLOOR for r in regions])
        for offset, anchor in ((2, pinpoint[0] if pinpoint else chosen[0]), (3, raw[0] if raw else chosen[0])):
            f[:, column + offset] = np.log1p(_haversine_km(*anchor, coords[q, :k]))
        if (q + 1) % 1000 == 0:
            print(f"candidates {q + 1}/{n}", flush=True)
    return coords, valid, features


def _verify_features(world, coords: np.ndarray, valid: np.ndarray, batch: int = 256) -> np.ndarray:
    """Simulated local search: photo-match the query against the nearest MP16 images within 5 km of each candidate."""

    import torch
    import torch.nn.functional as F
    from scipy.spatial import cKDTree

    print("building MP16 spatial index and loading embeddings into RAM", flush=True)
    tree = cKDTree(_xyz(world.mp16["latlon"]))
    embeddings = np.fromfile(MP16_EMBED / "embeddings.f16.bin", dtype=np.float16).reshape(world.mp16["embeddings"].shape)
    device = torch.device("cuda")
    queries = F.normalize(torch.as_tensor(world.query_embeddings, device=device), dim=-1)
    chord_5km = 2 * math.sin(VERIFY_RADIUS_KM[1] / EARTH_KM / 2)
    cos_1km = math.cos(VERIFY_RADIUS_KM[0] / EARTH_KM)
    out = np.zeros(valid.shape + (len(VERIFY_FEATURES),), dtype=np.float32)
    for start in range(0, len(coords), batch):
        stop = min(start + batch, len(coords))
        points = _xyz(coords[start:stop].reshape(-1, 2))
        _, neighbours = tree.query(points, k=VERIFY_NEAREST, distance_upper_bound=chord_5km, workers=-1)
        neighbours = neighbours.reshape(stop - start, SEARCH_MAX_CANDIDATES, VERIFY_NEAREST)
        for q in range(start, stop):
            rows = neighbours[q - start]
            present = (rows < len(embeddings)) & valid[q][:, None]
            safe = np.where(present, rows, 0)
            present &= world.mp16["author"][safe] != world.query_author[q]
            flat = np.unique(safe[present])
            if not len(flat):
                continue
            block = F.normalize(torch.as_tensor(embeddings[flat], device=device).float(), dim=-1)
            sims = (block @ queries[q]).cpu().numpy()
            position = np.searchsorted(flat, safe)
            sim_matrix = np.where(present, sims[np.clip(position, 0, len(flat) - 1)], -np.inf)
            within_1 = present & ((_xyz(world.mp16["latlon"][safe]) * _xyz(coords[q])[:, None]).sum(-1) >= cos_1km)
            reference = float(sims.max())
            for column, mask in ((0, within_1), (3, present)):
                masked = np.where(mask, sim_matrix, -np.inf)
                top5 = -np.sort(-masked, axis=1)[:, :5]
                count = mask.sum(1)
                best = top5[:, 0]
                out[q, :, column] = np.where(count > 0, best - reference, -1.0)
                finite = np.where(np.isfinite(top5), top5, np.nan)
                with np.errstate(invalid="ignore"):
                    out[q, :, column + 1] = np.where(count > 0, np.nanmean(finite, axis=1) - reference, -1.0)
                out[q, :, column + 2] = np.log1p(count)
        print(f"verify {stop}/{len(coords)}", flush=True)
    return out


def _train_selector(features: np.ndarray, valid: np.ndarray, reward: np.ndarray, train: np.ndarray, *, steps: int = 1_500, seed: int = 0) -> np.ndarray:
    """Per-candidate MLP scorer with a listwise expected-reward objective; returns scores for every query."""

    return _fit_selector(features, valid, reward, train, steps=steps, seed=seed)(features, valid)


def _fit_selector(features: np.ndarray, valid: np.ndarray, reward: np.ndarray, train: np.ndarray, *, steps: int = 1_500, seed: int = 0):
    """As `_train_selector`, but returns the fitted scorer so it can rank other queries' candidates."""

    import torch

    torch.manual_seed(seed)
    mean = features[train][valid[train]].mean(0)
    std = features[train][valid[train]].std(0) + 1e-6
    x = torch.as_tensor((features - mean) / std, dtype=torch.float32)
    mask = torch.as_tensor(valid)
    r = torch.as_tensor(reward, dtype=torch.float32)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[-1], 64), torch.nn.GELU(), torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    index = torch.as_tensor(np.flatnonzero(train))
    for _ in range(steps):
        logits = model(x[index]).squeeze(-1).masked_fill(~mask[index], float("-inf"))
        loss = -(torch.softmax(logits, dim=-1) * r[index]).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def score(other_features: np.ndarray, other_valid: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            other = torch.as_tensor((other_features - mean) / std, dtype=torch.float32)
            return model(other).squeeze(-1).masked_fill(~torch.as_tensor(other_valid), float("-inf")).numpy()

    return score


def search(root: Path) -> None:
    """Does multi-step verification beat a one-step reranker? Compare selectors with and without simulated local search."""

    world = load_world()
    cache = np.load(root / "neighbors.npz")
    heads = np.load(root / "region_head.npz")
    coarse_report = json.loads((root / "coarse.json").read_text(encoding="utf-8"))
    head = max(("head_linear", "head_mlp"), key=lambda name: sum(coarse_report[f"{b}/tune"][name]["region_mass"] for b in BENCHMARK_NAMES))
    feature_path = root / "search_features.npz"
    if feature_path.exists():
        saved = np.load(feature_path)
        coords, valid, one_shot, verify = saved["coords"], saved["valid"], saved["one_shot"], saved["verify"]
    else:
        n = len(world.queries)
        head_predictions = [{int(r): float(p) for r, p in zip(heads[f"{head}_regions"][q], heads[f"{head}_probs"][q])} for q in range(n)]
        gps_name = coarse_report["selected"]["knn_mp16_gps"]
        tau, k = float(gps_name.split("tau=")[1].split("|")[0]), int(gps_name.split("k=")[1])
        gps_votes = [_knn_distribution(world, cache, "mp16_gps", q, k, tau) for q in range(n)]
        coords, valid, one_shot = _search_candidates(world, cache, head_predictions, gps_votes)
        verify = _verify_features(world, coords, valid)
        np.savez(feature_path, coords=coords, valid=valid, one_shot=one_shot, verify=verify)

    n = len(world.queries)
    distance = np.full(valid.shape, np.inf)
    for q in range(n):
        distance[q, valid[q]] = _haversine_km(*world.query_latlon[q], coords[q, valid[q]])
    # GeoGuessr alone barely separates 1 km from 25 km, so street and city hits get explicit bonuses.
    finite = np.where(valid, distance, 1e5)
    reward = np.where(valid, _geoguessr(finite) / 5000.0 + 0.5 * (finite < 25) + 0.5 * (finite < 1), 0.0)
    tune = np.asarray([q["split"] == "tune" for q in world.queries])
    benchmark = np.asarray([q["benchmark"] for q in world.queries])

    picks: dict[str, np.ndarray] = {
        "pinpoint top-1": np.argmax(one_shot[..., 0] == 1.0, axis=1),
        "photo match + prior top-1": np.argmax(one_shot[..., 1] == 1.0, axis=1),
        f"oracle over pool (<= {SEARCH_MAX_CANDIDATES})": np.argmin(distance, axis=1),
    }
    one_shot_scores = _train_selector(one_shot, valid, reward, tune)
    picks["one-step reranker"] = np.argmax(one_shot_scores, axis=1)
    order = np.argsort(-one_shot_scores, axis=1)
    # Verify the reranker's top-B candidates; B = pool size is the full-information ceiling for any verification policy.
    for budget in VERIFY_BUDGETS:
        verified = np.zeros(valid.shape, dtype=bool)
        np.put_along_axis(verified, order[:, :budget], True, axis=1)
        verified &= valid
        combined = np.concatenate((one_shot, np.where(verified[..., None], verify, 0.0), verified[..., None].astype(np.float32)), axis=-1)
        picks[f"verify top-{budget} then rerank"] = np.argmax(_train_selector(combined, valid, reward, tune), axis=1)
        print(f"trained verification selectors for budget {budget}", flush=True)

    report: dict[str, Any] = {"head": head, "features": ONE_SHOT_FEATURES + VERIFY_FEATURES}
    for name in BENCHMARK_NAMES:
        members = np.flatnonzero((benchmark == name) & ~tune)
        report[f"{name}/eval"] = {
            arm: compute_metrics(coords[members, pick[members]], world.query_latlon[members]) for arm, pick in picks.items()
        }
    (root / "search.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name in BENCHMARK_NAMES:
        print(f"\n{name}/eval n={int(((benchmark == name) & ~tune).sum())}   <1km   <25km  <200km <750km <2500km  GeoGuessr")
        for arm, m in report[f"{name}/eval"].items():
            print(f"  {arm:40s}" + "".join(f"{m[f'Under_{t}_km']:7.1%}" for t in (1, 25, 200, 750, 2500)) + f"  {m['Geoguessr_score']:7.0f}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("neighbors", "region_head", "coarse", "pools", "fusion", "search"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/strategy_search"))
    args = parser.parse_args(argv)
    {"neighbors": neighbors, "region_head": region_head, "coarse": coarse, "pools": pools, "fusion": fusion, "search": search}[args.node](args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
