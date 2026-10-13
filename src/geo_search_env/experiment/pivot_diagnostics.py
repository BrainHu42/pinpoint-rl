# Prompts for a VLM choosing among retrieval candidates, and a benchmark harness for any such chooser (base model, SFT runs, API models).
# Usage: PYTHONPATH=src python -m geo_search_env.experiment.pivot_diagnostics vlm_sampling [--full-eval] [--prompt sft]   (vLLM on :8765)

"""vlm_sampling: a VLM given the photo and Pinpoint's candidates (its reranker's top-10, `pinpoint_reference bench`); greedy vs
single-sample vs best-of-N accuracy against the reranker top-1. GRPO-style RL mostly concentrates probability on answers the model
can already sample, so best-of-N bounds it.
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
from .strategy_search import BENCHMARK_NAMES, EARTH_KM, _haversine_km, _xyz, load_world


THRESHOLDS_KM = (1.0, 25.0, 200.0, 750.0)
VLM_CANDIDATES = 10
EVIDENCE_TOP = 100  # per candidate: support among the 100 most similar photos and among Pinpoint's top 100
VLM_SAMPLES = 8
VLM_TEMPERATURE = 1.0
SUBSET_PER_CELL = 150  # per benchmark x split
OUT = Path("artifacts/vlm_sampling")


def study_subset(world, per_cell: int = SUBSET_PER_CELL, seed: int = 0) -> list[int]:
    """Stratified per benchmark x split sample."""

    rng = np.random.default_rng(seed)
    subset: list[int] = []
    for benchmark in BENCHMARK_NAMES:
        for split in ("tune", "eval"):
            members = [i for i, q in enumerate(world.queries) if q["benchmark"] == benchmark and q["split"] == split]
            subset.extend(sorted(rng.choice(members, per_cell, replace=False).tolist()))
    return subset


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
# No reranker: all pool candidates in the order retrieval produced them, with their evidence. (The wording predates Pinpoint's
# three searches; kept as the sft-*-retrieval models were trained on it.)
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
    from .pinpoint_reference import load_pools, retrieval_order
    from .sft_data import _place_names

    world = load_world()
    pools = load_pools("bench")
    if [q["image_id"] for q in pools["queries"]] != [q["image_id"] for q in world.queries]:
        raise ValueError("Pinpoint's benchmark pools are out of date; rerun `pinpoint_reference bench`")
    coords, valid, csv_rows = pools["latlon"], pools["valid"], pools["csv_rows"]
    subset = [q for q in (range(len(world.queries)) if full_eval else study_subset(world)) if world.queries[q]["split"] == "eval"]
    if per_benchmark is not None:
        subset = [q for b in BENCHMARK_NAMES for q in [q for q in subset if world.queries[q]["benchmark"] == b][:per_benchmark]]
    if prompt_name == "sft-retrieval":  # every pool candidate, in retrieval order
        order = retrieval_order(pools["features"], valid)
        shown = {q: [int(c) for c in order[q] if valid[q, c]] for q in subset}
    else:
        shown = {q: [int(c) for c in np.flatnonzero(valid[q])[:VLM_CANDIDATES]] for q in subset}
    names = _place_names({int(csv_rows[q, c]) for q in subset for c in shown[q]})

    api_key = os.environ[api_key_env] if api_key_env else None
    workers = workers or (4 if api_key else 64)  # remote APIs rate-limit; a local vLLM needs many requests in flight
    jobs = []  # (query, tags, prompt, temperature, seed): one request answers every tag
    if prompt_name in EVIDENCE_PROMPTS:
        with np.load(root / "neighbors.npz") as saved:  # NpzFile re-reads an array on every key access
            cache = {k: saved[k] for k in ("mp16_raw_idx", "mp16_raw_sim", "osv_raw_idx", "osv_raw_sim", "mp16_gps_idx", "mp16_gps_sim")}
    for q in subset:
        options = format_options(
            [(names[int(csv_rows[q, c])], coords[q, c, 0], coords[q, c, 1]) for c in shown[q]],
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

    reranker = {q: tuple(coords[q, 0]) for q in subset}
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
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"pivot_vlm_sampling{suffix}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (OUT / f"pivot_vlm_answers{suffix}.json").write_text(json.dumps({world.queries[q]["image_id"]: answers[q] for q in subset}) + "\n", encoding="utf-8")
    for name in (*BENCHMARK_NAMES, "both"):
        entry = report[name]
        print(f"\n{name} eval {'half' if full_eval else 'subset'} n={entry['n']} (unparsed {entry['unparsed_rate']:.1%})   <1km  <25km <200km <750km")
        for arm, value in entry.items():
            if isinstance(value, dict):
                print(f"  {arm:32s}" + "".join(f"{value[f'Under_{int(t)}_km']:7.1%}" for t in THRESHOLDS_KM))
        if "share of <25km samples that are >25km from every candidate" in entry:
            print(f"  correct (<25km) samples that are off every candidate: {entry['share of <25km samples that are >25km from every candidate']:.0%}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("vlm_sampling",))
    parser.add_argument("--root", type=Path, default=Path("artifacts/strategy_search"), help="benchmark neighbour cache (evidence prompts)")
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
    vlm_sampling(args.root, server=args.server, model=args.model, api_key_env=args.api_key_env, workers=args.workers, thinking=args.thinking, samples=args.samples, per_benchmark=args.per_benchmark, prompt_name=args.prompt, temperature=args.temperature, full_eval=args.full_eval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
