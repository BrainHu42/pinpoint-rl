# MP16 queries for supervised fine-tuning, with retrieval candidates built exactly as for the benchmarks.
# Usage: .venv/bin/python -m geo_search_env.experiment.sft_data {pool,candidates,leak_check,overlay,dataset} --root artifacts/sft

"""SFT query pool, test-time candidates for each query, and a check that Pinpoint's training data does not leak.

Pinpoint's contrastive retriever trained on MP16 images with md5(image_id) % 100 < 99; bucket 99 was its validation
split. `leak_check` compares candidate quality on bucket-99 photos against a sample of photos Pinpoint trained on:
if the trained-on photos get clearly better candidates, training on them would teach the model to over-trust
retrieval. Benchmark photographers and near-duplicates of benchmark photos are never used as queries.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import threading
import urllib.request
from typing import Any, Sequence

import numpy as np

from ..data.benchmarks import compute_metrics
from .pivot_diagnostics import SFT_EVIDENCE_PROMPT, SFT_PROMPT, SFT_RETRIEVAL_PROMPT, candidate_evidence, format_options
from .strategy_search import (
    BENCHMARK_NAMES,
    MP16_CSV,
    MP16_EMBED,
    THRESHOLDS_KM,
    _haversine_km,
    _knn_distribution,
    _search_candidates,
    _xyz,
    load_world,
    neighbor_cache,
    region_head,
)
from .verifiers import MP16_ROOT, _place_names, _read_mp16_image, fit_reranker, reranker_ranking


BENCH_ROOT = Path("artifacts/strategy_search")
HELD_OUT_BUCKET = 99  # Pinpoint's contrastive retriever validated on this md5 bucket (val_pct=1) and trained on the rest
TRAINED_ON_SAMPLE = 5_000
NEAR_DUPLICATE_SIM = 0.95  # SigLIP2 cosine to any benchmark photo
VAL_AUTHOR_BUCKETS = 10  # md5(author) % 10 == 0 -> validation photographers, for this pool and any later expansion
TOP_CANDIDATES = 10
NAME_SEARCH = 16  # nearest MP16 rows searched for a candidate's place name, skipping the query's photographer
COPY_CANDIDATE_KM = 1.0  # within this, the target copies the candidate's coordinates (same reward, easier to learn)
MATCH_CANDIDATE_KM = 25.0  # the target's "Candidate: N" names the nearest shown candidate within this distance
OVERLAY_PROMPT = (
    "Does this photo have GPS coordinates (latitude/longitude numbers) printed or overlaid on it as text, "
    "for example by a camera or GPS app? Answer yes or no."
)
OVERLAY_SPOT_CHECK = 60  # highest-scoring images saved for a manual look when choosing the cut-off
# Chosen by viewing the top 60 on the held-out pool: the 8 true overlays score >= 0.659; below are date stamps and
# watermarks, plus one real harbour sign with coordinates (0.628), which is scene content and stays.
OVERLAY_THRESHOLD = 0.64


class MP16Images:
    """Thread-safe JPEG bytes by image id from the MP16-Pro tar shards (each thread seeks its own file handles)."""

    def __init__(self) -> None:
        with (MP16_ROOT / "metadata" / "tar_index.pkl").open("rb") as stream:
            self.index = pickle.load(stream)
        self.parts = sorted(MP16_ROOT.glob("mp-16-images[0-9][0-9]"))
        self.chunk = self.parts[0].stat().st_size
        self.local = threading.local()

    def read(self, image_id: str) -> bytes | None:
        if not hasattr(self.local, "handles"):
            self.local.handles = [part.open("rb") for part in self.parts]
        return _read_mp16_image(image_id, self.index, self.local.handles, self.chunk)


def _md5_bucket(text: str, buckets: int) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % buckets


def pool(root: Path, *, seed: int = 0) -> None:
    """Pinpoint-held-out MP16 photos plus a sample of photos it trained on, minus benchmark photographers and near-duplicates."""

    import torch
    import torch.nn.functional as F

    world = load_world()
    image_ids = (MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines()
    author_names = np.empty(len(world.vocab["author"]), dtype=object)
    for name, index in world.vocab["author"].items():
        author_names[index] = name
    print("hashing image ids", flush=True)
    bucket = np.fromiter((_md5_bucket(i, 100) for i in image_ids), dtype=np.int16, count=len(image_ids))
    val_author = np.asarray([_md5_bucket(a, VAL_AUTHOR_BUCKETS) == 0 for a in author_names])[world.mp16["author"]]
    eligible = ~np.isin(world.mp16["author"], world.query_author[world.query_author >= 0])
    held_out = np.flatnonzero(eligible & (bucket == HELD_OUT_BUCKET))
    rng = np.random.default_rng(seed)
    trained_on = np.sort(rng.choice(np.flatnonzero(eligible & (bucket < HELD_OUT_BUCKET) & ~val_author), TRAINED_ON_SAMPLE, replace=False))

    rows = np.concatenate((held_out, trained_on))
    device = torch.device("cuda")
    bench = F.normalize(torch.as_tensor(world.query_embeddings, device=device), dim=-1).half()
    near = np.concatenate([
        (F.normalize(torch.as_tensor(np.asarray(world.mp16["embeddings"][rows[i : i + 8192]]), device=device).float(), dim=-1).half() @ bench.T)
        .max(dim=1).values.float().cpu().numpy()
        for i in range(0, len(rows), 8192)
    ])
    keep = near < NEAR_DUPLICATE_SIM
    queries = [
        {
            "row": int(r),
            "image_id": image_ids[r],
            "group": "held_out" if i < len(held_out) else "trained_on",
            "split": "val" if val_author[r] else "train",
        }
        for i, r in enumerate(rows) if keep[i]
    ]
    summary = {
        "held_out_bucket": HELD_OUT_BUCKET,
        "excluded_benchmark_photographer_images": int((~eligible).sum()),
        "near_duplicates_dropped": {g: int((~keep[s]).sum()) for g, s in (("held_out", slice(0, len(held_out))), ("trained_on", slice(len(held_out), None)))},
        "queries": {g: {s: sum(q["group"] == g and q["split"] == s for q in queries) for s in ("train", "val")} for g in ("held_out", "trained_on")},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "queries.json").write_text(json.dumps(queries) + "\n", encoding="utf-8")
    (root / "pool.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def candidates(root: Path) -> None:
    """The benchmark pipeline's 20 candidates per MP16 query, ranked by the reranker fitted on the benchmark tune halves."""

    from scipy.spatial import cKDTree

    bench_world = load_world()
    score = fit_reranker(bench_world, BENCH_ROOT)
    del bench_world
    coarse_report = json.loads((BENCH_ROOT / "coarse.json").read_text(encoding="utf-8"))
    head = max(("head_linear", "head_mlp"), key=lambda name: sum(coarse_report[f"{b}/tune"][name]["region_mass"] for b in BENCHMARK_NAMES))
    gps_name = coarse_report["selected"]["knn_mp16_gps"]
    tau, k = float(gps_name.split("tau=")[1].split("|")[0]), int(gps_name.split("k=")[1])

    queries = json.loads((root / "queries.json").read_text(encoding="utf-8"))
    world = load_world(with_pinpoint=True, mp16_queries=queries)
    if not (root / "neighbors.npz").exists():
        np.savez(root / "neighbors.npz", **neighbor_cache(world, chunk=16_384))
    cache = dict(np.load(root / "neighbors.npz"))  # NpzFile re-reads an array on every key access
    if not (root / "region_head.npz").exists():
        region_head(root, world=world)  # query photographers held out of the region prior
    heads = dict(np.load(root / "region_head.npz"))
    n = len(queries)
    head_predictions = [{int(r): float(p) for r, p in zip(heads[f"{head}_regions"][q], heads[f"{head}_probs"][q])} for q in range(n)]
    gps_votes = [_knn_distribution(world, cache, "mp16_gps", q, k, tau) for q in range(n)]
    coords, valid, one_shot = _search_candidates(world, cache, head_predictions, gps_votes)
    ranking = np.argsort(-score(one_shot, valid), axis=1)
    distance = np.full(valid.shape, np.inf)
    for q in range(n):
        distance[q, valid[q]] = _haversine_km(*world.query_latlon[q], coords[q, valid[q]])

    print("naming candidates", flush=True)
    tree = cKDTree(_xyz(world.mp16["latlon"]))
    _, nearest = tree.query(_xyz(coords.reshape(-1, 2)), k=NAME_SEARCH, workers=-1)
    nearest = nearest.reshape(n, coords.shape[1], NAME_SEARCH)
    other = world.mp16["author"][nearest] != world.query_author[:, None, None]
    first = np.where(other.any(-1), other.argmax(-1), 0)  # all-same-photographer neighbourhoods fall back to the nearest row
    name_row = np.where(valid, world.mp16["row_index"][np.take_along_axis(nearest, first[..., None], -1)[..., 0]], -1)
    np.savez(root / "candidates.npz", coords=coords, valid=valid, one_shot=one_shot, ranking=ranking, distance=distance, name_row=name_row)
    print(f"saved candidates for {n} queries", flush=True)


def _candidate_quality(coords, valid, one_shot, ranking, distance, truth, members) -> dict[str, Any]:
    top = np.take_along_axis(np.where(valid, distance, np.inf), ranking[:, :TOP_CANDIDATES], axis=1)[members]
    entry: dict[str, Any] = {"n": int(len(members))}
    for name, pick in (
        ("reranker top-1", ranking[:, 0]),
        ("pinpoint top-1", np.argmax(one_shot[..., 0] == 1.0, axis=1)),
        ("photo match + prior top-1", np.argmax(one_shot[..., 1] == 1.0, axis=1)),
    ):
        m = compute_metrics(coords[members, pick[members]], truth[members])
        entry[name] = {f"Under_{int(t)}_km": m[f"Under_{int(t)}_km"] for t in THRESHOLDS_KM}
    entry[f"top-{TOP_CANDIDATES} recall"] = {f"Under_{int(t)}_km": float((top < t).any(1).mean()) for t in THRESHOLDS_KM}
    return entry


def leak_check(root: Path) -> None:
    """Candidate quality on photos Pinpoint held out vs photos it trained on, next to the benchmark eval halves."""

    report: dict[str, Any] = {}
    bench_world = load_world()
    coords, valid, distance, ranking = reranker_ranking(bench_world, BENCH_ROOT)
    one_shot = np.load(BENCH_ROOT / "search_features.npz")["one_shot"]
    for name in BENCHMARK_NAMES:
        members = np.asarray([i for i, q in enumerate(bench_world.queries) if q["benchmark"] == name and q["split"] == "eval"])
        report[f"{name} eval"] = _candidate_quality(coords, valid, one_shot, ranking, distance, bench_world.query_latlon, members)
    queries = json.loads((root / "queries.json").read_text(encoding="utf-8"))
    rows = np.asarray([q["row"] for q in queries])
    truth = bench_world.mp16["latlon"][rows]
    saved = np.load(root / "candidates.npz")
    for group in ("held_out", "trained_on"):
        members = np.asarray([i for i, q in enumerate(queries) if q["group"] == group])
        report[f"MP16 {group}"] = _candidate_quality(saved["coords"], saved["valid"], saved["one_shot"], saved["ranking"], saved["distance"], truth, members)
    (root / "leak_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    columns = ("reranker top-1", "pinpoint top-1", "photo match + prior top-1", f"top-{TOP_CANDIDATES} recall")
    print(f"{'':20s} {'n':>6s}  " + "  ".join(f"{c:>27s}" for c in columns))
    print(f"{'':20s} {'':6s}  " + "  ".join(f"{'<1km <25km <200km <750km':>27s}" for _ in columns))
    for arm, entry in report.items():
        print(f"{arm:20s} {entry['n']:6d}  " + "  ".join(
            f"{'':3s}" + " ".join(f"{entry[c][f'Under_{int(t)}_km']:6.1%}" for t in THRESHOLDS_KM) for c in columns
        ))


def _p_yes(server: str, image: bytes) -> float:
    body = {
        "model": "vlm", "temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}},
            {"type": "text", "text": OVERLAY_PROMPT},
        ]}],
    }
    request = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        top = json.loads(response.read())["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "yes")


def overlay(root: Path, *, server: str, workers: int = 64) -> None:
    """P(yes) that each held-out query shows burned-in GPS coordinates, from the base VLM served at `server`."""

    queries = [q for q in json.loads((root / "queries.json").read_text(encoding="utf-8")) if q["group"] == "held_out"]
    images = MP16Images()

    def score(q: dict[str, Any]) -> tuple[str, float | None]:
        data = images.read(q["image_id"])
        return q["image_id"], _p_yes(server, data) if data else None

    scores: dict[str, float | None] = {}
    with ThreadPoolExecutor(workers) as pool:
        for done, (image_id, p) in enumerate(pool.map(score, queries), start=1):
            scores[image_id] = p
            if done % 2000 == 0:
                print(f"  overlay {done}/{len(queries)}", flush=True)
    (root / "overlay_scores.json").write_text(json.dumps(scores) + "\n", encoding="utf-8")
    ranked = sorted((p, i) for i, p in scores.items() if p is not None)[::-1]
    spot = root / "overlay_spot_check"
    spot.mkdir(exist_ok=True)
    for rank, (p, image_id) in enumerate(ranked[:OVERLAY_SPOT_CHECK]):
        (spot / f"{rank:02d}_p{p:.3f}_{image_id}").write_bytes(images.read(image_id))
    print(f"unreadable {sum(p is None for p in scores.values())}; P(yes) >= 0.5: {sum(p >= 0.5 for p, _ in ranked)}, >= 0.1: {sum(p >= 0.1 for p, _ in ranked)}")


def _mp16_labels(csv_rows: set[int]) -> dict[int, tuple[str, str, str]]:
    """(country, region, city) per MP16-Pro CSV row; city falls back to county, missing fields are 'unknown'."""

    labels: dict[int, tuple[str, str, str]] = {}
    with MP16_CSV.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        city, county, state, country = (header.index(c) for c in ("city", "county", "state", "country"))
        for i, row in enumerate(reader):
            if i in csv_rows:
                labels[i] = (row[country] or "unknown", row[state] or "unknown", row[city] or row[county] or "unknown")
    return labels


def build_example(
    options: list[tuple[str, float, float]], truth: tuple[float, float], labels: tuple[str, str, str],
    evidence: list[tuple[int, float | None, int]] | None = None, template: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Prompt, target and bookkeeping for one query from its shown candidates (name, lat, lon) in display order."""

    shown = np.asarray([(lat, lon) for _, lat, lon in options])
    distance = _haversine_km(*truth, shown) if len(shown) else np.asarray([])
    nearest = int(np.argmin(distance)) if len(distance) else -1
    near_km = float(distance[nearest]) if nearest >= 0 else math.inf
    if near_km <= COPY_CANDIDATE_KM:
        lat, lon = float(shown[nearest, 0]), float(shown[nearest, 1])  # exactly as displayed after rounding below
    else:
        lat, lon = truth
    candidate = str(nearest + 1) if near_km <= MATCH_CANDIDATE_KM else "none"
    country, region, city = labels
    target = (
        f"Country: {country}. Region: {region}. City: {city}. Candidate: {candidate}\n"
        f'```json\n{{"lat": {lat:.3f}, "lon": {lon:.3f}}}\n```'
    )
    case = "copy <1km" if near_km <= COPY_CANDIDATE_KM else "candidate 1-25km" if candidate != "none" else "no candidate <25km"
    prompt = (template or (SFT_PROMPT if evidence is None else SFT_EVIDENCE_PROMPT)).format(options=format_options(options, evidence))
    return prompt, target, {"nearest_candidate_km": near_km, "case": case}


def dataset(root: Path, *, overlay_threshold: float = OVERLAY_THRESHOLD, variant: str = "sft") -> None:
    """JSONL of {image_id, split, prompt, target, meta} for the held-out pool, minus burned-in-GPS photos.

    Variants: `sft` (reranker top-10), `sft_evidence` (plus each candidate's retrieval support) and `sft_retrieval`
    (no reranker: all pooled candidates in pool order, with support). meta keeps the reranker's top-1 as a reference.
    """

    evidence = variant != "sft"

    queries = json.loads((root / "queries.json").read_text(encoding="utf-8"))
    saved = np.load(root / "candidates.npz")
    coords, valid, ranking, name_row = saved["coords"], saved["valid"], saved["ranking"], saved["name_row"]
    overlay_scores = json.loads((root / "overlay_scores.json").read_text(encoding="utf-8"))
    row_index = np.fromfile(MP16_EMBED / "row_index.i64.bin", dtype=np.int64)
    latlon = np.fromfile(MP16_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2).astype(np.float64)
    members = [i for i, q in enumerate(queries) if q["group"] == "held_out"]
    if variant == "sft_retrieval":
        shown = {i: [int(c) for c in np.flatnonzero(valid[i])] for i in members}
    else:
        shown = {i: [int(c) for c in ranking[i, :TOP_CANDIDATES] if valid[i, c]] for i in members}
    names = _place_names({int(name_row[i, c]) for i in members for c in shown[i]})
    labels = _mp16_labels({int(row_index[queries[i]["row"]]) for i in members})
    if evidence:
        world = load_world(mp16_queries=queries)
        with np.load(root / "neighbors.npz") as saved:
            cache = {k: saved[k] for k in saved.files}

    counts: dict[str, dict[str, int]] = {}
    dropped = {"burned-in GPS": 0, "unreadable": 0}
    name = variant
    with (root / f"{name}.jsonl").open("w", encoding="utf-8") as out:
        for i in members:
            q = queries[i]
            p = overlay_scores.get(q["image_id"])
            if p is None or p >= overlay_threshold:
                dropped["unreadable" if p is None else "burned-in GPS"] += 1
                continue
            options = [(names[int(name_row[i, c])], float(coords[i, c, 0]), float(coords[i, c, 1])) for c in shown[i]]
            truth = (float(latlon[q["row"], 0]), float(latlon[q["row"], 1]))
            support = candidate_evidence(world, cache, i, coords[i, shown[i]]) if evidence else None
            template = SFT_RETRIEVAL_PROMPT if variant == "sft_retrieval" else None
            prompt, target, meta = build_example(options, truth, labels[int(row_index[q["row"]])], support, template)
            meta["reranker_top1"] = [float(coords[i, ranking[i, 0], 0]), float(coords[i, ranking[i, 0], 1])]
            out.write(json.dumps({"image_id": q["image_id"], "split": q["split"], "prompt": prompt, "target": target, "meta": meta | {"lat": truth[0], "lon": truth[1]}}, ensure_ascii=False) + "\n")
            counts.setdefault(q["split"], {}).setdefault(meta["case"], 0)
            counts[q["split"]][meta["case"]] += 1
    summary = {"overlay_threshold": overlay_threshold, "dropped": dropped, "cases": counts}
    (root / f"{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("pool", "candidates", "leak_check", "overlay", "dataset"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/sft"))
    parser.add_argument("--server", default="http://127.0.0.1:8765", help="OpenAI-compatible server with the base VLM (overlay)")
    parser.add_argument("--variant", choices=("sft", "sft_evidence", "sft_retrieval"), default="sft", help="dataset: prompt variant (see `dataset`)")
    parser.add_argument("--overlay-threshold", type=float, default=OVERLAY_THRESHOLD, help="drop photos with P(burned-in GPS) at or above this")
    args = parser.parse_args(argv)
    if args.node == "overlay":
        overlay(args.root, server=args.server)
    elif args.node == "dataset":
        dataset(args.root, overlay_threshold=args.overlay_threshold, variant=args.variant)
    else:
        {"pool": pool, "candidates": candidates, "leak_check": leak_check}[args.node](args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
