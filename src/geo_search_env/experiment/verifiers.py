# Test whether verifiers with information beyond SIGLIP similarity can pick the right candidate the reranker misses.
# Usage: PYTHONPATH=src python -m geo_search_env.experiment.verifiers --root artifacts/strategy_search  (needs the llama-server from vlm_hypotheses running)

"""Geometric (SIFT + RANSAC) and VLM-knowledge verification of the one-step reranker's top candidates.

Runs on a stratified subset of the benchmark queries. Thresholds are tuned on the subset's tune half and
reported on its eval half, so the verdict on multi-step verification does not reuse the data it was tuned on.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
import json
import math
from pathlib import Path
import pickle
import re
import urllib.error
import urllib.request
from typing import Any, Sequence

import numpy as np

from ..data.benchmarks import compute_metrics
from .strategy_search import (
    BENCHMARK_NAMES,
    EARTH_KM,
    MP16_CSV,
    MP16_EMBED,
    _geoguessr,
    _fit_selector,
    _haversine_km,
    _xyz,
    load_world,
)


MP16_ROOT = Path("/data/hf/datasets/MP16-Pro")
SUBSET_PER_CELL = 150  # per benchmark x split
TOP_CANDIDATES = 5
REFERENCES_PER_CANDIDATE = 3
REFERENCE_SEARCH = 300
SIFT_MAX_SIDE = 1024
VLM_PROMPT = """You are an expert at geolocating photos. Which of these candidate places is this photo from?
{options}
Use any readable text, landmarks, architecture, vegetation, terrain and other cues. Candidates may be wrong; give each a probability (they must sum to 1).
Finish with exactly one JSON block: ```json
{{"probabilities": [p1, p2, ...]}}
```"""


def _read_mp16_image(image_id: str, tar_index, handles, chunk_size: int) -> bytes | None:
    info = tar_index.get(image_id)
    if info is None:
        return None
    part, local = divmod(info.offset_data, chunk_size)
    if part >= len(handles):
        return None
    handles[part].seek(local)
    if local + info.size <= chunk_size:
        return handles[part].read(info.size)
    first = handles[part].read(chunk_size - local)
    handles[part + 1].seek(0)
    return first + handles[part + 1].read(info.size - len(first))


def _sift_inliers(task: tuple[bytes, list[bytes | None]]) -> list[int]:
    """RANSAC fundamental-matrix inliers between the query and each reference (0 when unmatched)."""

    import cv2

    def decode(data: bytes):
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        scale = SIFT_MAX_SIDE / max(image.shape)
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image

    query_bytes, references = task
    sift = cv2.SIFT_create(nfeatures=2_000)
    query = decode(query_bytes)
    if query is None:
        return [0] * len(references)
    q_points, q_desc = sift.detectAndCompute(query, None)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    out = []
    for data in references:
        image = decode(data) if data else None
        if image is None or q_desc is None or len(q_points) < 8:
            out.append(0)
            continue
        r_points, r_desc = sift.detectAndCompute(image, None)
        if r_desc is None or len(r_points) < 8:
            out.append(0)
            continue
        pairs = [m for m, n in (p for p in matcher.knnMatch(q_desc, r_desc, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(pairs) < 8:
            out.append(0)
            continue
        src = np.float32([q_points[m.queryIdx].pt for m in pairs])
        dst = np.float32([r_points[m.trainIdx].pt for m in pairs])
        fundamental, mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, 3.0, 0.99)
        # On failure OpenCV returns F=None with an uninitialised mask, so only a fitted F yields inliers.
        out.append(int((mask > 0).sum()) if fundamental is not None and mask is not None else 0)
    return out


def _ask_vlm(server: str, image_path: str, options: list[str]) -> tuple[list[float], str]:
    text = VLM_PROMPT.format(options="\n".join(f"{i + 1}. {o}" for i, o in enumerate(options)))
    body = {
        "model": "gemma-4-26b-a4b",
        "temperature": 0.0,
        "max_tokens": 1200,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(Path(image_path).read_bytes()).decode()}},
            {"type": "text", "text": text},
        ]}],
    }
    request = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            answer = json.loads(response.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as error:
        message = json.loads(error.read())["error"]["message"]
        answer = message.split("<channel|>", 1)[-1]
    block = answer[answer.rfind("probabilities"):] if "probabilities" in answer else ""
    values = [float(v) for v in re.findall(r"\d*\.?\d+", block)][: len(options)]
    if len(values) != len(options) or sum(values) <= 0:
        return [1 / len(options)] * len(options), answer
    return [v / sum(values) for v in values], answer


def _place_names(rows: set[int]) -> dict[int, str]:
    """'city, state, country' for MP16-Pro CSV rows."""

    names: dict[int, str] = {}
    with MP16_CSV.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        columns = [header.index(c) for c in ("city", "county", "state", "country")]
        for i, row in enumerate(reader):
            if i in rows:
                city, county, state, country = (row[c] for c in columns)
                names[i] = ", ".join(part for part in (city or county, state, country) if part) or "unknown"
    return names


def reranker_ranking(world, root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Candidate coords/validity/truth distances from the search node, and the one-step reranker's full ranking."""

    coords, valid, distance, one_shot, score = _reranker(world, root)
    return coords, valid, distance, np.argsort(-score(one_shot, valid), axis=1)


def fit_reranker(world, root: Path):
    """The one-step reranker fitted on the benchmark tune halves, as a scorer of (one-shot features, validity)."""

    return _reranker(world, root)[-1]


def _reranker(world, root: Path):
    saved = np.load(root / "search_features.npz")
    coords, valid, one_shot = saved["coords"], saved["valid"], saved["one_shot"]
    distance = np.full(valid.shape, np.inf)
    for q in range(len(world.queries)):
        distance[q, valid[q]] = _haversine_km(*world.query_latlon[q], coords[q, valid[q]])
    finite = np.where(valid, distance, 1e5)
    reward = np.where(valid, _geoguessr(finite) / 5000.0 + 0.5 * (finite < 25) + 0.5 * (finite < 1), 0.0)
    tune = np.asarray([q["split"] == "tune" for q in world.queries])
    return coords, valid, distance, one_shot, _fit_selector(one_shot, valid, reward, tune)


def study_subset(world, per_cell: int = SUBSET_PER_CELL, seed: int = 0) -> list[int]:
    """Stratified per benchmark x split sample, shared by the verifier and pivot diagnostics."""

    rng = np.random.default_rng(seed)
    subset: list[int] = []
    for benchmark in BENCHMARK_NAMES:
        for split in ("tune", "eval"):
            members = [i for i, q in enumerate(world.queries) if q["benchmark"] == benchmark and q["split"] == split]
            subset.extend(sorted(rng.choice(members, per_cell, replace=False).tolist()))
    return subset


def run(root: Path, *, server: str, per_cell: int = SUBSET_PER_CELL, seed: int = 0) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    world = load_world()
    n = len(world.queries)
    coords, valid, distance, ranking = reranker_ranking(world, root)
    order = ranking[:, :TOP_CANDIDATES]
    subset = study_subset(world, per_cell, seed)

    print("selecting reference photos near each candidate", flush=True)
    tree = cKDTree(_xyz(world.mp16["latlon"]))
    image_ids = (MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines()
    chord = 2 * math.sin(1.0 / EARTH_KM / 2)
    references: dict[int, list[list[int]]] = {}
    for q in subset:
        query = world.query_embeddings[q] / np.linalg.norm(world.query_embeddings[q])
        per_candidate = []
        for c in order[q]:
            if not valid[q, c]:
                per_candidate.append([])
                continue
            _, rows = tree.query(_xyz(coords[q, c]), k=REFERENCE_SEARCH, distance_upper_bound=chord)
            rows = rows[rows < len(image_ids)]
            rows = rows[world.mp16["author"][rows] != world.query_author[q]]
            if not len(rows):
                per_candidate.append([])
                continue
            embeddings = np.asarray(world.mp16["embeddings"][np.sort(rows)], dtype=np.float32)
            sims = embeddings @ query / np.linalg.norm(embeddings, axis=1)
            per_candidate.append(np.sort(rows)[np.argsort(-sims)[:REFERENCES_PER_CANDIDATE]].tolist())
        references[q] = per_candidate

    print("reading reference photos from MP16 shards", flush=True)
    tar_index = pickle.load((MP16_ROOT / "metadata" / "tar_index.pkl").open("rb"))
    parts = sorted(MP16_ROOT.glob("mp-16-images[0-9][0-9]"))
    handles = [p.open("rb") for p in parts]
    chunk = parts[0].stat().st_size
    tasks, layout = [], []
    for q in subset:
        query_bytes = Path(world.queries[q]["path"]).read_bytes()
        for c, rows in enumerate(references[q]):
            tasks.append((query_bytes, [_read_mp16_image(image_ids[r], tar_index, handles, chunk) for r in rows]))
            layout.append((q, c))
    del tar_index
    print(f"SIFT verification on {len(tasks)} query-candidate pairs", flush=True)
    inliers = np.zeros((n, TOP_CANDIDATES))
    with ProcessPoolExecutor(max_workers=16) as pool:
        for (q, c), counts in zip(layout, pool.map(_sift_inliers, tasks, chunksize=8)):
            inliers[q, c] = max(counts, default=0)

    print("VLM verification", flush=True)
    nearest_rows = {}
    for q in subset:
        for c in order[q]:
            if valid[q, c]:
                _, row = tree.query(_xyz(coords[q, c]))
                nearest_rows[(q, int(c))] = int(world.mp16["row_index"][row])
    names = _place_names(set(nearest_rows.values()))
    vlm = np.zeros((n, TOP_CANDIDATES))
    raw_answers: dict[int, str] = {}

    def ask(q: int):
        options = [f"{names[nearest_rows[(q, int(c))]]} (lat {coords[q, c, 0]:.3f}, lon {coords[q, c, 1]:.3f})" for c in order[q] if valid[q, c]]
        return q, _ask_vlm(server, world.queries[q]["path"], options)

    with ThreadPoolExecutor(4) as pool:
        for done, (q, (probabilities, answer)) in enumerate(pool.map(ask, subset), start=1):
            vlm[q, : len(probabilities)] = probabilities
            raw_answers[q] = answer
            if done % 100 == 0:
                print(f"  VLM {done}/{len(subset)}", flush=True)

    return _evaluate(world, subset, order, coords, valid, distance, inliers, vlm, raw_answers, root)


def _evaluate(world, subset, order, coords, valid, distance, inliers, vlm, raw_answers, root: Path) -> dict[str, Any]:
    sub = np.asarray(subset)
    top_distance = np.take_along_axis(np.where(valid, distance, np.inf), order, axis=1)
    benchmark = np.asarray([world.queries[q]["benchmark"] for q in range(len(world.queries))])
    tune = np.asarray([world.queries[q]["split"] == "tune" for q in range(len(world.queries))])

    def rule(scores: np.ndarray, threshold: float) -> np.ndarray:
        best = np.argmax(scores, axis=1)
        return np.where(scores[np.arange(len(scores)), best] >= threshold, best, 0)

    def objective(pick: np.ndarray, members: np.ndarray) -> float:
        d = top_distance[members, pick[members]]
        return float((d < 1).sum() + (d < 25).sum())

    arms: dict[str, np.ndarray] = {"one-step reranker": np.zeros(len(world.queries), dtype=int)}
    tuned: dict[str, float] = {}
    tune_members = sub[tune[sub]]
    for name, scores, grid in (
        ("SIFT inliers", inliers, [15, 20, 30, 40, 60, 80, 120, 200, 1e9]),
        ("VLM probability", vlm, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]),
    ):
        tuned[name] = max(grid, key=lambda t: objective(rule(scores, t), tune_members))
        arms[f"{name} >= {tuned[name]:g}, else reranker"] = rule(scores, tuned[name])
    sift_pick, vlm_pick = arms[next(a for a in arms if a.startswith("SIFT"))], arms[next(a for a in arms if a.startswith("VLM"))]
    sift_fired = inliers[np.arange(len(inliers)), np.argmax(inliers, axis=1)] >= tuned["SIFT inliers"]
    arms["SIFT, then VLM, else reranker"] = np.where(sift_fired, sift_pick, vlm_pick)

    report: dict[str, Any] = {"thresholds": tuned, "subset_size": len(subset)}
    for name in (*BENCHMARK_NAMES, "both"):
        members = sub[~tune[sub] & ((benchmark[sub] == name) if name != "both" else True)]
        entry: dict[str, Any] = {"n": int(len(members))}
        entry["top5 contains <1km"] = float((top_distance[members] < 1).any(1).mean())
        entry["top5 contains <25km"] = float((top_distance[members] < 25).any(1).mean())
        for radius in (1, 25):
            has = members[(top_distance[members] < radius).any(1) & ~(top_distance[members] < radius).all(1)]
            entry[f"when top5 has a <{radius}km and a wrong candidate (n={len(has)})"] = {
                "reranker ranks correct first": float((top_distance[has, 0] < radius).mean()) if len(has) else None,
                "SIFT ranks correct first": float((top_distance[has, np.argmax(inliers[has], 1)] < radius).mean()) if len(has) else None,
                "VLM ranks correct first": float((top_distance[has, np.argmax(vlm[has], 1)] < radius).mean()) if len(has) else None,
            }
        for arm, pick in arms.items():
            chosen = coords[members, order[members, pick[members]]]
            entry[arm] = compute_metrics(chosen, world.query_latlon[members]) | {"changed_vs_reranker": float((pick[members] != 0).mean())}
        report[name] = entry
    (root / "verifiers.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez(root / "verifiers_scores.npz", subset=sub, order=order[sub], inliers=inliers[sub], vlm=vlm[sub])
    (root / "verifiers_vlm_answers.json").write_text(json.dumps({world.queries[q]["image_id"]: a for q, a in raw_answers.items()}) + "\n", encoding="utf-8")
    for name in (*BENCHMARK_NAMES, "both"):
        entry = report[name]
        print(f"\n{name} eval subset n={entry['n']}: top-5 contains <1km {entry['top5 contains <1km']:.0%}, <25km {entry['top5 contains <25km']:.0%}")
        for key, value in entry.items():
            if key.startswith("when"):
                print(f"  {key}: " + ", ".join(f"{k} {v:.0%}" for k, v in value.items() if v is not None))
        for arm in arms:
            m = entry[arm]
            print(f"  {arm:34s}" + "".join(f"{m[f'Under_{t}_km']:7.1%}" for t in (1, 25, 200, 750)) + f"  changed {m['changed_vs_reranker']:.0%}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/strategy_search"))
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--per-cell", type=int, default=SUBSET_PER_CELL, help="queries per benchmark x split")
    args = parser.parse_args(argv)
    run(args.root, server=args.server, per_cell=args.per_cell)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
