# Stage 1 of the plan: does the model's search retrieve evidence that raises the oracle accuracy? No chooser in the loop.
# Usage: .venv/bin/python -m geo_search_env.experiment.stage1_eval places --tag dev      (vLLM on :8765)
#        ~/.venvs/sft/bin/python -m geo_search_env.experiment.stage1_eval evaluate --tag dev      (GPU; PYTHONPATH=src)

"""Oracle accuracy of (shown candidates + retrieved evidence), the ceiling for any stage-2 chooser.

places:   the model sees only the photo and names three specific places, which are the search queries (prompt below).
evaluate: budget per photo is six results. Backends: SigLIP2 photo search (one MP16 + one OSV-5M photo per query),
          Wikipedia BM25 and Wikipedia dense (two articles per query). Control: the query photo's own whole-image search
          (three per corpus). Near-duplicates of the query photo (cosine >= 0.95) and same-photographer rows are excluded.
          Reports the % of photos with a result within 1 / 25 / 200 km of the truth when the pooled candidates (everything the
          pipeline found, ~17 per photo, not only the reranker's top-10) are added, the gain over the pool-only oracle,
          and paired bootstrap intervals.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

import numpy as np

from .query_evidence import (
    BENCH_ROOT, NEAR_DUPLICATE_SIM, SFT_ROOT, TEXT_TOKENS, MP16Images, _chat, _load, _parallel, _photo, _save, parse_queries,
)
from .query_headroom import _features, _model, _search
from .strategy_search import load_world
from .wiki_backend import Wiki, _km


PLACES_PROMPT = """Where was this photo taken? Use any readable text.
Name the 3 most specific places it could be (landmark, venue, building, park or street, with city and country), most likely first.
Answer: ```json
{{"queries": ["", "", ""]}}
```"""
# v2 as first run repeated one place three times (24% of dev photos) and often left the empty template: ask for different
# places and use placeholders in the template.
PROMPTS = {
    "v2": PLACES_PROMPT,
    "v2c": """Where was this photo taken? Use any readable text.
Name 3 different specific places it could be (landmark, venue, building, park or street, with city and country), most likely first.
Answer: ```json
{{"queries": ["", "", ""]}}
```""",
    # v2b (placeholders in the template) made 44% of dev photos answer in prose and run out of tokens before the JSON.
    "v2b": """Where was this photo taken? Use any readable text.
Name 3 different specific places it could be (landmark, venue, building, park or street, with city and country), most likely first.
Answer: ```json
{{"queries": ["<place 1>", "<place 2>", "<place 3>"]}}
```""",
}
THRESHOLDS = (1.0, 25.0, 200.0)  # street, city, region
RESULTS_PER_QUERY = 2  # SigLIP: one MP16 + one OSV photo; Wikipedia: two articles
WHOLE_PER_CORPUS = 3  # equal budget (6 results) for the whole-image control


def _suffix(variant: str) -> str:
    return "" if variant == "v2" else f"_{variant}"


def places(tag: str, server: str, variant: str = "v2") -> None:
    dev = _load(tag, "dev.json")
    images = MP16Images()
    raws = _parallel(lambda e: _chat(server, [_photo(images, e), {"type": "text", "text": PROMPTS[variant]}], 150), dev)
    out = {e["image_id"]: {"raw": raw, "queries": parse_queries(raw)} for e, raw in zip(dev, raws)}
    _save(tag, f"places{_suffix(variant)}.json", out)
    counts = np.bincount([len(v["queries"]) for v in out.values()], minlength=4)
    print(f"queries per photo 0/1/2/3: {counts.tolist()}")


def _bootstrap(delta: np.ndarray, resamples: int = 2000) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    means = delta[rng.integers(0, len(delta), (resamples, len(delta)))].mean(1)
    return float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(tag: str, variant: str = "v2") -> None:
    import torch

    dev = _load(tag, "dev.json")
    generated = _load(tag, f"places{_suffix(variant)}.json")
    bench = "path" in dev[0]
    world = load_world() if bench else load_world(mp16_queries=[{"row": e["row"], "image_id": e["image_id"]} for e in dev])
    position = [e["index"] if bench else m for m, e in enumerate(dev)]
    with np.load((BENCH_ROOT if bench else SFT_ROOT) / "neighbors.npz") as saved:
        initial = {k: saved[k][[e["index"] for e in dev]] for k in ("mp16_raw_idx", "mp16_raw_sim", "osv_raw_idx", "osv_raw_sim")}

    texts = [(m, q) for m, e in enumerate(dev) for q in generated[e["image_id"]]["queries"]]
    model, processor = _model()
    embeddings = []
    for start in range(0, len(texts), 128):
        tokens = processor(text=[t[1].lower() for t in texts[start : start + 128]], return_tensors="pt", padding="max_length",
                           max_length=TEXT_TOKENS, truncation=True).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            embeddings.append(_features(model.get_text_features(**tokens)).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    found = _search(world, np.concatenate(embeddings), world.query_author[[position[m] for m, _ in texts]])
    wiki = Wiki()
    dense = wiki.dense([q for _, q in texts], k=2 * RESULTS_PER_QUERY)
    bm25 = wiki.bm25_many([q for _, q in texts], k=2 * RESULTS_PER_QUERY)

    def photo_results(m: int, corpus: str, rows: np.ndarray, sims: np.ndarray, count: int, taken: set) -> list[tuple[float, float]]:
        gallery = world.mp16 if corpus == "mp16" else world.osv
        q_emb = world.query_embeddings[position[m]] / np.linalg.norm(world.query_embeddings[position[m]])
        out: list[tuple[float, float]] = []
        for row, sim in zip(rows, sims):
            if len(out) == count:
                break
            if not np.isfinite(sim) or (corpus, int(row)) in taken:
                continue
            g = np.asarray(gallery["embeddings"][int(row)], dtype=np.float32)
            if float(g @ q_emb / np.linalg.norm(g)) >= NEAR_DUPLICATE_SIM:
                continue
            taken.add((corpus, int(row)))
            out.append((float(gallery["latlon"][int(row)][0]), float(gallery["latlon"][int(row)][1])))
        return out

    results: dict[str, list[list[tuple[float, float]]]] = {k: [[] for _ in dev] for k in ("whole", "siglip", "bm25", "dense")}
    taken_whole = [set() for _ in dev]
    for m in range(len(dev)):
        for corpus in ("mp16", "osv"):
            results["whole"][m] += photo_results(m, corpus, initial[f"{corpus}_raw_idx"][m], initial[f"{corpus}_raw_sim"][m], WHOLE_PER_CORPUS, taken_whole[m])
    matched: list[list[tuple[float, float]]] = [[] for _ in dev]  # whole-image results, one per corpus per query this photo got
    queries_per_photo = np.bincount([m for m, _ in texts], minlength=len(dev))
    for m in range(len(dev)):
        taken = set()
        for corpus in ("mp16", "osv"):
            matched[m] += photo_results(m, corpus, initial[f"{corpus}_raw_idx"][m], initial[f"{corpus}_raw_sim"][m], int(queries_per_photo[m]), taken)
    results["whole (matched)"] = matched
    taken_siglip = [set() for _ in dev]
    seen = {"bm25": [set() for _ in dev], "dense": [set() for _ in dev]}
    for j, (m, _) in enumerate(texts):
        for corpus in ("mp16", "osv"):
            results["siglip"][m] += photo_results(m, corpus, found[f"{corpus}_idx"][j], found[f"{corpus}_sim"][j], 1, taken_siglip[m])
        for name, hits in (("bm25", bm25[j]), ("dense", dense[j])):
            fresh = [h for h in hits if h["id"] not in seen[name][m]][:RESULTS_PER_QUERY]
            seen[name][m].update(h["id"] for h in fresh)
            results[name][m] += [(h["lat"], h["lon"]) for h in fresh]
    results["siglip+dense (12)"] = [a + b for a, b in zip(results["siglip"], results["dense"])]

    truth = np.asarray([e["truth"] for e in dev])
    km = lambda pts, m: float(_km(pts, *truth[m]).min()) if len(pts) else np.inf
    shown = np.asarray([km([c[1:] for c in e["candidates"]], m) for m, e in enumerate(dev)])  # reranker top-10
    pooled = np.asarray([km(e.get("pool") or [c[1:] for c in e["candidates"]], m) for m, e in enumerate(dev)])  # every candidate the pipeline pooled
    top1 = np.asarray([km([e["candidates"][0][1:]], m) for m, e in enumerate(dev)])
    oracle = {name: np.minimum(pooled, [km(pts, m) for m, pts in enumerate(res)]) for name, res in results.items()}  # baseline: the whole pool
    cities = [{c[0].split(",")[0].lower() for c in e["candidates"]} for e in dev]
    named = np.mean([any(c in q.lower() for c in cities[m]) for m, q in texts]) if texts else 0.0

    groups = {"all": np.ones(len(dev), dtype=bool)}
    if bench:
        groups |= {b: np.asarray([e["benchmark"] == b for e in dev]) for b in ("im2gps3k", "yfcc4k")}
    pool_size = float(np.mean([len(e.get("pool") or e["candidates"]) for e in dev]))
    report: dict[str, Any] = {"tag": tag, "variant": variant, "n": len(dev), "queries per photo": len(texts) / len(dev), "pool size": pool_size,
                              "queries naming a shown candidate's city": float(named)}
    print(f"tag {tag} variant {variant}: n={len(dev)}, {len(texts) / len(dev):.2f} queries per photo, pool {pool_size:.1f} candidates per photo, "
          f"{named:.0%} of queries name a shown candidate's city")
    cols = "<1 km  <25 km <200 km"
    for g, mask in groups.items():
        print(f"\n{g} (n={mask.sum()})              ORACLE {cols}   GAIN over pool {cols}   gain <25 km [95% CI]   vs matched whole-image <25 km [CI]")
        rows = {"reranker top-1": top1, "reranker top-10": shown, "pooled candidates (baseline)": pooled, **{f"+ {k}": v for k, v in oracle.items()}}
        report[g] = {}
        for name, d in rows.items():
            entry = {f"<{int(t)} km": float((d[mask] < t).mean()) for t in THRESHOLDS}
            line = f"  {name:28s} " + " ".join(f"{entry[f'<{int(t)} km']:6.1%}" for t in THRESHOLDS)
            if name.startswith("+ "):
                gains = {f"<{int(t)} km": float((d[mask] < t).mean() - (pooled[mask] < t).mean()) for t in THRESHOLDS}
                gain = _bootstrap((d[mask] < 25).astype(float) - (pooled[mask] < 25))
                control = _bootstrap((d[mask] < 25).astype(float) - (oracle["whole (matched)"][mask] < 25))
                entry |= {"gain": gains, "gain <25 km CI": gain, "vs whole (matched) <25 km": control}
                line += "   " + " ".join(f"{gains[f'<{int(t)} km']:+6.1%}" for t in THRESHOLDS)
                line += f"   {gain[0]:+6.1%} [{gain[1]:+.1%}, {gain[2]:+.1%}]   {control[0]:+6.1%} [{control[1]:+.1%}, {control[2]:+.1%}]"
            report[g][name] = entry
            print(line)
    _save(tag, f"stage1{_suffix(variant)}.json", report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("places", "evaluate"))
    parser.add_argument("--tag", default="dev")
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--variant", choices=tuple(PROMPTS), default="v2")
    args = parser.parse_args(argv)
    places(args.tag, args.server, args.variant) if args.node == "places" else evaluate(args.tag, args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
