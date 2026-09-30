# Diagnostics for the two RL pivots: active region search (B) and RL fine-tuning of a VLM geolocator (A).
# Usage: PYTHONPATH=src python -m geo_search_env.experiment.pivot_diagnostics {region_search,vlm_sampling} --root artifacts/strategy_search

"""Is there headroom an RL policy could learn?

region_search (B): retrieval restricted to the region head's predicted regions over the full galleries, versus
unrestricted retrieval, the true region, and an oracle that picks the best of the predicted regions.

vlm_sampling (A): a VLM given the photo and the reranker's candidates; greedy vs single-sample vs best-of-N accuracy.
GRPO-style RL mostly concentrates probability on answers the model can already sample, so best-of-N bounds it.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

import numpy as np

from ..data.benchmarks import compute_metrics
from .coarse_filter import UNKNOWN
from .strategy_search import BENCHMARK_NAMES, EARTH_KM, _geoguessr, _haversine_km, _knn_distribution, _pool, _train_selector, _xyz, load_world
from .verifiers import _place_names, reranker_ranking, study_subset


REGION_TOP_K = (1, 3, 5)
REGION_FEATURES = ["head_prob", "head_rank", "gps_vote_mass", "log_hits", "max_sim", "top10_mean_sim", "max_sim_minus_best_region", "pool_size"]
RETRIEVE = 1_000
POOL_BUDGETS = (1, 10, 50)
THRESHOLDS_KM = (1.0, 25.0, 200.0, 750.0)
VLM_CANDIDATES = 10
EVIDENCE_TOP = 100  # per candidate: support among the 100 most similar photos and among Pinpoint's top 100
VLM_SAMPLES = 8
VLM_TEMPERATURE = 1.0


def _stream_region_topk(array, queries, query_author, gallery_author, gallery_region, allowed_regions, top_k: int, chunk: int = 32_768):
    """Top-k per query over a disk-backed gallery, restricted to each query's allowed region set (same author excluded)."""

    import torch
    import torch.nn.functional as F

    device = queries.device
    q_author = torch.as_tensor(query_author, device=device)
    allowed = torch.as_tensor(allowed_regions, device=device)  # [Q, R]
    best_values = torch.full((len(queries), 0), float("-inf"), device=device)
    best_indices = torch.zeros((len(queries), 0), dtype=torch.long, device=device)
    for start in range(0, len(array), chunk):
        block = F.normalize(torch.as_tensor(np.asarray(array[start : start + chunk], dtype=np.float16), device=device).float(), dim=-1).half()
        scores = (queries @ block.T).float()
        regions = torch.as_tensor(gallery_region[start : start + chunk], device=device)
        keep = (regions.view(1, 1, -1) == allowed.unsqueeze(-1)).any(1) & (regions.view(1, -1) != UNKNOWN)
        keep &= torch.as_tensor(gallery_author[start : start + chunk], device=device).unsqueeze(0) != q_author.unsqueeze(1)
        scores.masked_fill_(~keep, float("-inf"))
        values, indices = torch.topk(scores, min(top_k, scores.shape[1]), dim=1)
        best_values, best_indices = torch.cat((best_values, values), 1), torch.cat((best_indices, indices + start), 1)
        keep_top = torch.topk(best_values, min(top_k, best_values.shape[1]), dim=1).indices
        best_values, best_indices = torch.gather(best_values, 1, keep_top), torch.gather(best_indices, 1, keep_top)
        if (start // chunk) % 32 == 0:
            print(f"  {start + len(block)}/{len(array)}", flush=True)
    return best_indices.cpu().numpy(), best_values.cpu().numpy()


def region_search(root: Path) -> None:
    import torch
    import torch.nn.functional as F

    world = load_world()
    cache = np.load(root / "neighbors.npz")
    heads = np.load(root / "region_head.npz")
    predicted = heads["head_mlp_regions"][:, : max(REGION_TOP_K)]
    truth = world.query_region[:, None]
    device = torch.device("cuda")
    queries = F.normalize(torch.as_tensor(world.query_embeddings, device=device), dim=-1).half()
    hits: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {"predicted": [], "truth": []}
    for name, allowed in (("predicted", predicted), ("truth", truth)):
        for gallery in (world.mp16, world.osv):
            print(f"region-restricted retrieval: {name}", flush=True)
            idx, sim = _stream_region_topk(gallery["embeddings"], queries, world.query_author, gallery["author"], gallery["region"], allowed, RETRIEVE)
            hits[name].append((gallery["latlon"][idx], sim, gallery["region"][idx]))

    def rows(parts, q: int, regions: set[int] | None = None) -> list[tuple[float, float, float]]:
        out = []
        for latlon, sim, region in parts:
            finite = np.isfinite(sim[q])
            for (a, b), s, r in zip(latlon[q][finite], sim[q][finite], region[q][finite]):
                if regions is None or int(r) in regions:
                    out.append((float(a), float(b), float(s)))
        return out

    cached = [(world.mp16["latlon"][cache["mp16_raw_idx"]], cache["mp16_raw_sim"], world.mp16["region"][cache["mp16_raw_idx"]]),
              (world.osv["latlon"][cache["osv_raw_idx"]], cache["osv_raw_sim"], world.osv["region"][cache["osv_raw_idx"]])]
    n = len(world.queries)
    coarse_report = json.loads((root / "coarse.json").read_text(encoding="utf-8"))
    gps_name = coarse_report["selected"]["knn_mp16_gps"]
    tau, k_votes = float(gps_name.split("tau=")[1].split("|")[0]), int(gps_name.split("k=")[1])
    slots = max(REGION_TOP_K)
    region_features = np.zeros((n, slots, len(REGION_FEATURES)), dtype=np.float32)
    region_valid = np.zeros((n, slots), dtype=bool)
    region_top1 = np.zeros((n, slots, 2))
    arms = ["unrestricted"] + [f"head top-{k} regions" for k in REGION_TOP_K] + ["best of head top-5 (oracle choice)", "true region"]
    best_so_far = {arm: np.full((n, max(POOL_BUDGETS)), np.inf) for arm in arms}
    top1 = {arm: np.zeros((n, 2)) for arm in arms}
    for q in range(n):
        pools = {"unrestricted": _pool(rows(cached, q)), "true region": _pool(rows(hits["truth"], q))}
        for k in REGION_TOP_K:
            pools[f"head top-{k} regions"] = _pool(rows(hits["predicted"], q, {int(r) for r in predicted[q, :k]}))
        region_rows = [rows(hits["predicted"], q, {int(r)}) for r in predicted[q, :slots]]
        per_region = [_pool(r) for r in region_rows]
        votes = _knn_distribution(world, cache, "mp16_gps", q, k_votes, tau)
        best_overall = max((max(h[2] for h in r) for r in region_rows if r), default=0.0)
        for i, (r_rows, pool) in enumerate(zip(region_rows, per_region)):
            if not pool:
                continue
            sims = np.sort([h[2] for h in r_rows])[::-1]
            region_valid[q, i], region_top1[q, i] = True, pool[0]
            region_features[q, i] = (
                heads["head_mlp_probs"][q, i], i, votes.get(int(predicted[q, i]), 0.0), np.log1p(len(r_rows)),
                sims[0], sims[:10].mean(), sims[0] - best_overall, len(pool),
            )
        scored = [(min(_haversine_km(*world.query_latlon[q], np.asarray(p[:10])), default=np.inf) if p else np.inf, i) for i, p in enumerate(per_region)]
        pools["best of head top-5 (oracle choice)"] = per_region[min(scored)[1]] if per_region else []
        for arm, pool in pools.items():
            if pool:
                d = np.minimum.accumulate(_haversine_km(*world.query_latlon[q], np.asarray(pool[: max(POOL_BUDGETS)])))
                best_so_far[arm][q, : len(d)] = d
                best_so_far[arm][q, len(d):] = d[-1]
                top1[arm][q] = pool[0]
        if (q + 1) % 1000 == 0:
            print(f"pools {q + 1}/{n}", flush=True)

    report: dict[str, Any] = {}
    for benchmark in BENCHMARK_NAMES:
        members = np.asarray([i for i, x in enumerate(world.queries) if x["benchmark"] == benchmark and x["split"] == "eval"])
        entry: dict[str, Any] = {"n": len(members)}
        for k in REGION_TOP_K:
            entry[f"true region in head top-{k}"] = float((predicted[members, :k] == truth[members]).any(1).mean())
        for arm in arms:
            entry[arm] = {
                "top1": compute_metrics(top1[arm][members], world.query_latlon[members]),
                "recall": {f"le_{t:g}km@{b}": float((best_so_far[arm][members, b - 1] <= t).mean()) for b in POOL_BUDGETS for t in (1.0, 25.0)},
            }
        report[benchmark] = entry
    (root / "pivot_region_search.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez(root / "pivot_region_pools.npz", features=region_features, valid=region_valid, top1=region_top1, predicted=predicted)
    for benchmark in BENCHMARK_NAMES:
        entry = report[benchmark]
        print(f"\n{benchmark}/eval n={entry['n']}: true region in head top-1/3/5 = " + ", ".join(f"{entry[f'true region in head top-{k}']:.0%}" for k in REGION_TOP_K))
        print(f"  {'arm':38s} top-1 <1km  <25km <200km <750km | recall <1km@10 <25km@10 <1km@50 <25km@50")
        for arm in arms:
            m, r = entry[arm]["top1"], entry[arm]["recall"]
            print(f"  {arm:38s}" + "".join(f"{m[f'Under_{int(t)}_km']:7.1%}" for t in THRESHOLDS_KM) + " |" + "".join(f"{r[k]:9.1%}" for k in ("le_1km@10", "le_25km@10", "le_1km@50", "le_25km@50")))


PROMPT = """Geolocate this photo. A retrieval system proposed these candidate locations, best first (all of them may be wrong):
{options}
Use readable text, landmarks, architecture, vegetation, terrain and any other cues. You may choose a candidate or give different coordinates.
Think briefly (under 120 words), then end with exactly one JSON block: ```json
{{"lat": 0.0, "lon": 0.0}}
```"""
# The fine-tuned model's prompt: same evidence, answer in the SFT target format instead of free reasoning.
SFT_PROMPT = """Geolocate this photo. A retrieval system proposed these candidate locations, best first (all of them may be wrong):
{options}
Use readable text, landmarks, architecture, vegetation, terrain and any other cues. You may choose a candidate or give different coordinates.
Answer in exactly this format:
Country: <country>. Region: <region>. City: <city>. Candidate: <number or none>
```json
{{"lat": 0.0, "lon": 0.0}}
```"""
# The SFT prompt plus each candidate's retrieval evidence (see `candidate_evidence`).
SFT_EVIDENCE_PROMPT = SFT_PROMPT.replace(
    "(all of them may be wrong):\n",
    "(all of them may be wrong).\n"
    f"Each shows how many of the {EVIDENCE_TOP} most similar database photos were taken within 1 km of it (with the best "
    f"similarity, 0-1), and how many of the retrieval system's own top {EVIDENCE_TOP} matches are within 1 km:\n",
)
# No reranker: all pooled candidates in the order retrieval produced them, with their evidence.
SFT_RETRIEVAL_PROMPT = SFT_EVIDENCE_PROMPT.replace(
    "these candidate locations, best first (all of them may be wrong).",
    "these candidate locations, in the order its two search methods produced them, not sorted by confidence (all of them may be wrong).",
)
PROMPTS = {"default": PROMPT, "sft": SFT_PROMPT, "sft-evidence": SFT_EVIDENCE_PROMPT, "sft-retrieval": SFT_RETRIEVAL_PROMPT}
EVIDENCE_PROMPTS = ("sft-evidence", "sft-retrieval")


def candidate_evidence(world, cache: dict[str, np.ndarray], q: int, coords: np.ndarray) -> list[tuple[int, float | None, int]]:
    """Per candidate [K, 2]: (similar photos within 1 km, their best similarity, Pinpoint matches within 1 km).

    Similar photos are the EVIDENCE_TOP highest raw-SigLIP matches across MP16 and OSV-5M; Pinpoint matches are its
    top EVIDENCE_TOP in the MP16 GPS gallery. Same-photographer rows are already excluded from the cache.
    """

    near = lambda latlon: _xyz(np.asarray(coords, dtype=np.float64)) @ _xyz(latlon).T >= math.cos(1.0 / EARTH_KM)
    sim = np.concatenate((cache["mp16_raw_sim"][q], cache["osv_raw_sim"][q]))
    latlon = np.concatenate((world.mp16["latlon"][cache["mp16_raw_idx"][q]], world.osv["latlon"][cache["osv_raw_idx"][q]]))
    top = np.argsort(-sim)[:EVIDENCE_TOP]
    top = top[np.isfinite(sim[top])]
    photos = near(latlon[top])
    best = np.where(photos, sim[top][None], -np.inf).max(axis=1)
    gps = cache["mp16_gps_idx"][q, :EVIDENCE_TOP][np.isfinite(cache["mp16_gps_sim"][q, :EVIDENCE_TOP])]
    pinpoint = near(world.mp16["latlon"][gps]).sum(axis=1)
    return [(int(n), float(b) if n else None, int(p)) for n, b, p in zip(photos.sum(axis=1), best, pinpoint)]


def format_options(candidates: Sequence[tuple[str, float, float]], evidence: Sequence[tuple[int, float | None, int]] | None = None) -> str:
    """Numbered candidate lines, best first, from (place name, lat, lon) and optional `candidate_evidence`."""

    lines = [f"{rank}. {name} ({lat:.3f}, {lon:.3f})" for rank, (name, lat, lon) in enumerate(candidates, start=1)]
    if evidence is not None:
        lines = [
            f"{line} - similar photos {n}/{EVIDENCE_TOP}" + (f" (best {best:.2f})" if best is not None else "") + f", retrieval {p}/{EVIDENCE_TOP}"
            for line, (n, best, p) in zip(lines, evidence, strict=True)
        ]
    return "\n".join(lines)


def _ask(
    server: str, model: str, api_key: str | None, image_path: str, prompt: str, temperature: float, seed: int, thinking: bool = False, n: int = 1,
) -> list[tuple[float, float] | None]:
    """`n` parsed answers from one request (n > 1 lets a local vLLM encode the image once for all samples)."""

    if thinking and temperature == 0.0:
        # Qwen's guidance: greedy decoding in thinking mode loops; its "greedy" answer uses the recommended sampler.
        temperature = 0.6
    body = {
        "model": model,
        "temperature": temperature,
        **({"top_p": 0.95, "top_k": 20} if thinking else {}),
        "seed": seed,
        **({"n": n} if n > 1 else {}),
        # Reasoning (remote models, or local ones with thinking on) spends tokens before the answer.
        "max_tokens": 14000 if thinking else 8000 if api_key else 600,  # 14k fits a 16k server slot
        **({} if api_key else {"chat_template_kwargs": {"enable_thinking": thinking}}),
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(Path(image_path).read_bytes()).decode()}},
            {"type": "text", "text": prompt},
        ]}],
    }
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})}
    request = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode(), headers)
    answers = None
    for attempt in range(4):  # remote APIs rate-limit and occasionally fail transiently
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                answers = [choice["message"]["content"] or "" for choice in json.loads(response.read())["choices"]]
            break
        except urllib.error.HTTPError as error:
            if api_key and error.code in (408, 429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(5 * 2**attempt)
                continue
            answers = [json.loads(error.read()).get("error", {}).get("message", "").split("<channel|>", 1)[-1]] * n
            break
        except Exception:
            if attempt < 3:
                time.sleep(5 * 2**attempt)
                continue
            return [None] * n
    return [None] * n if answers is None else [parse_coordinates(a) for a in answers]


def parse_coordinates(answer: str) -> tuple[float, float] | None:
    """The last "lat"/"lon" values in a model answer, or None if absent or out of range."""

    lat = re.findall(r"lat\W*?(-?\d+(?:\.\d+)?)", answer)
    lon = re.findall(r"lon\W*?(-?\d+(?:\.\d+)?)", answer)
    if not lat or not lon or abs(float(lat[-1])) > 90 or abs(float(lon[-1])) > 180:
        return None
    return float(lat[-1]), float(lon[-1])


def vlm_sampling(
    root: Path, *, server: str, model: str = "vlm", api_key_env: str | None = None, workers: int | None = None,
    thinking: bool = False, samples: int = VLM_SAMPLES, per_benchmark: int | None = None, prompt_name: str = "default",
    temperature: float = VLM_TEMPERATURE, full_eval: bool = False,
) -> None:
    from scipy.spatial import cKDTree

    world = load_world()
    coords, valid, distance, ranking = reranker_ranking(world, root)
    subset = [q for q in (range(len(world.queries)) if full_eval else study_subset(world)) if world.queries[q]["split"] == "eval"]
    if per_benchmark is not None:
        subset = [q for b in BENCHMARK_NAMES for q in [q for q in subset if world.queries[q]["benchmark"] == b][:per_benchmark]]
    if prompt_name == "sft-retrieval":  # every pooled candidate, in pool order
        shown = {q: [int(c) for c in np.flatnonzero(valid[q])] for q in subset}
    else:
        shown = {q: [int(c) for c in ranking[q, :VLM_CANDIDATES] if valid[q, c]] for q in subset}
    tree = cKDTree(_xyz(world.mp16["latlon"]))
    nearest = {}
    for q in subset:
        for c in shown[q]:
            nearest[(q, c)] = int(world.mp16["row_index"][tree.query(_xyz(coords[q, c]))[1]])
    names = _place_names(set(nearest.values()))

    api_key = os.environ[api_key_env] if api_key_env else None
    workers = workers or (4 if api_key else 64)  # remote APIs rate-limit; a local vLLM needs many requests in flight
    jobs = []  # (query, tags, prompt, temperature, seed): one request answers every tag
    if prompt_name in EVIDENCE_PROMPTS:
        with np.load(root / "neighbors.npz") as saved:  # NpzFile re-reads an array on every key access
            cache = {k: saved[k] for k in ("mp16_raw_idx", "mp16_raw_sim", "osv_raw_idx", "osv_raw_sim", "mp16_gps_idx", "mp16_gps_sim")}
    for q in subset:
        options = format_options(
            [(names[nearest[(q, c)]], coords[q, c, 0], coords[q, c, 1]) for c in shown[q]],
            candidate_evidence(world, cache, q, coords[q, shown[q]]) if prompt_name in EVIDENCE_PROMPTS else None,
        )
        prompt = PROMPTS[prompt_name].format(options=options)
        jobs.append((q, ["greedy"], prompt, 0.0, 0))
        tags = [f"sample{i}" for i in range(samples)]
        if api_key:  # OpenAI-compatible remote APIs may not support n > 1
            jobs.extend((q, [tag], prompt, temperature, i + 1) for i, tag in enumerate(tags))
        elif tags:
            jobs.append((q, tags, prompt, temperature, 1))
    answers: dict[int, dict[str, tuple[float, float] | None]] = {q: {} for q in subset}
    started = time.time()

    def run(job):
        q, tags, prompt, temperature, seed = job
        return _ask(server, model, api_key, world.queries[q]["path"], prompt, temperature, seed, thinking, n=len(tags))

    with ThreadPoolExecutor(workers) as pool:
        for done, ((q, tags, *_), parsed) in enumerate(zip(jobs, pool.map(run, jobs)), start=1):
            answers[q].update(zip(tags, parsed))
            if done % 300 == 0 or (len(jobs) < 900 and done % 25 == 0):
                print(f"  VLM {done}/{len(jobs)}", flush=True)
    print(f"VLM requests took {time.time() - started:.0f}s", flush=True)

    reranker = {q: tuple(coords[q, ranking[q, 0]]) for q in subset}
    fallback = {q: tuple(coords[q, shown[q][0]]) for q in subset}  # unparseable answers fall back to the first shown candidate

    def point(q: int, tag: str) -> tuple[float, float]:
        return answers[q].get(tag) or fallback[q]

    report: dict[str, Any] = {"model_server": server, "model": model, "thinking": thinking, "samples": samples, "temperature": temperature, "prompt": prompt_name}
    for name in (*BENCHMARK_NAMES, "both"):
        members = [q for q in subset if name == "both" or world.queries[q]["benchmark"] == name]
        truth = world.query_latlon[members]
        candidate_xy = [coords[q, shown[q]] for q in members]
        entry: dict[str, Any] = {"n": len(members), "unparsed_rate": float(np.mean([answers[q].get(t) is None for q in members for t in answers[q]]))}
        entry["reranker top-1"] = compute_metrics(np.asarray([reranker[q] for q in members]), truth)
        entry["VLM greedy"] = compute_metrics(np.asarray([point(q, "greedy") for q in members]), truth)
        for t in THRESHOLDS_KM:
            entry.setdefault("shown candidates (oracle)", {})[f"Under_{int(t)}_km"] = float(np.mean([(_haversine_km(*tr, c) < t).any() for tr, c in zip(truth, candidate_xy)]))
        if samples:
            sampled = np.asarray([[point(q, f"sample{i}") for i in range(samples)] for q in members])  # [M, S, 2]
            sample_distance = np.stack([_haversine_km(*t, s) for t, s in zip(truth, sampled)])
            # Self-consistency: the sample with the most other samples within 25 km.
            consensus = []
            for s in sampled:
                pairwise = np.stack([_haversine_km(*p, s) for p in s])
                consensus.append(s[np.argmax((pairwise < 25).sum(1))])
            entry[f"VLM self-consistency ({samples})"] = compute_metrics(np.asarray(consensus), truth)
            for t in THRESHOLDS_KM:
                entry.setdefault("VLM mean single sample", {})[f"Under_{int(t)}_km"] = float((sample_distance < t).mean())
                entry.setdefault(f"VLM best of {samples} (oracle)", {})[f"Under_{int(t)}_km"] = float((sample_distance < t).any(1).mean())
                entry.setdefault(f"best of {samples} or shown candidates (oracle)", {})[f"Under_{int(t)}_km"] = float(np.mean([
                    (sample_distance[i] < t).any() or (_haversine_km(*truth[i], candidate_xy[i]) < t).any() for i in range(len(members))
                ]))
            off = [(_haversine_km(*sampled[i, j], candidate_xy[i]) > 25).all() for i in range(len(members)) for j in range(samples) if sample_distance[i, j] < 25]
            entry["share of <25km samples that are >25km from every candidate"] = float(np.mean(off)) if off else 0.0
        report[name] = entry
    suffix = "" if model == "vlm" else "_" + re.sub(r"[^A-Za-z0-9.-]+", "_", model)
    suffix += "" if prompt_name == "default" else f"_{prompt_name}-prompt"
    suffix += "" if temperature == VLM_TEMPERATURE else f"_t{temperature:g}"
    suffix += "_full-eval" if full_eval else ""
    (root / f"pivot_vlm_sampling{suffix}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (root / f"pivot_vlm_answers{suffix}.json").write_text(json.dumps({world.queries[q]["image_id"]: answers[q] for q in subset}) + "\n", encoding="utf-8")
    for name in (*BENCHMARK_NAMES, "both"):
        entry = report[name]
        print(f"\n{name} eval {'half' if full_eval else 'subset'} n={entry['n']} (unparsed {entry['unparsed_rate']:.1%})   <1km  <25km <200km <750km")
        for arm, value in entry.items():
            if isinstance(value, dict):
                print(f"  {arm:32s}" + "".join(f"{value[f'Under_{int(t)}_km']:7.1%}" for t in THRESHOLDS_KM))
        if "share of <25km samples that are >25km from every candidate" in entry:
            print(f"  correct (<25km) samples that are off every candidate: {entry['share of <25km samples that are >25km from every candidate']:.0%}")


def region_selector(root: Path) -> None:
    """Can the region choice be learned from region-restricted retrieval evidence? Train on tune, report on eval."""

    world = load_world()
    saved = np.load(root / "pivot_region_pools.npz")
    features, valid, top1 = saved["features"], saved["valid"], saved["top1"]
    n = len(world.queries)
    distance = np.full(valid.shape, 1e5)
    for q in range(n):
        if valid[q].any():
            distance[q, valid[q]] = _haversine_km(*world.query_latlon[q], top1[q, valid[q]])
    reward = np.where(valid, _geoguessr(distance) / 5000.0 + 0.5 * (distance < 25) + 0.5 * (distance < 1), 0.0)
    tune = np.asarray([q["split"] == "tune" for q in world.queries])
    picks = {
        "head top-1 region": np.zeros(n, dtype=int),
        "learned region choice": np.argmax(_train_selector(features, valid, reward, tune), axis=1),
        "oracle region choice (top-5)": np.argmin(distance, axis=1),
    }
    report: dict[str, Any] = {"features": REGION_FEATURES}
    for benchmark in BENCHMARK_NAMES:
        members = np.asarray([i for i, x in enumerate(world.queries) if x["benchmark"] == benchmark and x["split"] == "eval"])
        report[benchmark] = {arm: compute_metrics(top1[members, pick[members]], world.query_latlon[members]) for arm, pick in picks.items()}
    (root / "pivot_region_selector.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for benchmark in BENCHMARK_NAMES:
        print(f"\n{benchmark}/eval top-1 of chosen region   <1km  <25km <200km <750km  GeoGuessr")
        for arm, m in report[benchmark].items():
            print(f"  {arm:30s}" + "".join(f"{m[f'Under_{int(t)}_km']:7.1%}" for t in THRESHOLDS_KM) + f"  {m['Geoguessr_score']:7.0f}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("region_search", "region_selector", "vlm_sampling"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/strategy_search"))
    parser.add_argument("--server", default="http://127.0.0.1:8765", help="OpenAI-compatible base URL (e.g. https://openrouter.ai/api)")
    parser.add_argument("--model", default="vlm", help="model id sent to the server")
    parser.add_argument("--api-key-env", help="environment variable holding the API key, for remote servers")
    parser.add_argument("--workers", type=int, help="concurrent requests (default: 64 local, 4 remote)")
    parser.add_argument("--thinking", action="store_true", help="let local reasoning models think before answering")
    parser.add_argument("--samples", type=int, default=VLM_SAMPLES, help="temperature samples per query (0 = greedy only)")
    parser.add_argument("--per-benchmark", type=int, help="limit the eval subset to the first N queries per benchmark")
    parser.add_argument("--temperature", type=float, default=VLM_TEMPERATURE, help="sampling temperature for the samples")
    parser.add_argument("--full-eval", action="store_true", help="every eval-half query instead of the 300-query study subset")
    parser.add_argument("--prompt", choices=tuple(PROMPTS), default="default", help="sft: the fine-tuned model's answer format")
    args = parser.parse_args(argv)
    if args.node == "region_search":
        region_search(args.root)
    elif args.node == "region_selector":
        region_selector(args.root)
    else:
        vlm_sampling(args.root, server=args.server, model=args.model, api_key_env=args.api_key_env, workers=args.workers, thinking=args.thinking, samples=args.samples, per_benchmark=args.per_benchmark, prompt_name=args.prompt, temperature=args.temperature, full_eval=args.full_eval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
