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
    BENCH_ROOT, ROOT, NEAR_DUPLICATE_SIM, SFT_ROOT, TEXT_TOKENS, MP16Images, _chat, _load, _parallel, _photo, _save, parse_queries,
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
    "v2-27b": PLACES_PROMPT,  # same prompt, answered by the Qwen3.6-27B (llama.cpp on :8766); separate output files
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


def places(tag: str, server: str, variant: str = "v2", max_tokens: int | None = None) -> None:
    """`max_tokens` defaults to 150 (the 4B answers with the JSON straight away) and 700 for the 27B, which writes an analysis first."""

    dev = _load(tag, "dev.json")
    images = MP16Images()
    budget = max_tokens or (700 if variant.endswith("27b") else 150)
    raws = _parallel(lambda e: _chat(server, [_photo(images, e), {"type": "text", "text": PROMPTS[variant]}], budget), dev)
    out = {e["image_id"]: {"raw": raw, "queries": parse_queries(raw)} for e, raw in zip(dev, raws)}
    _save(tag, f"places{_suffix(variant)}.json", out)
    counts = np.bincount([len(v["queries"]) for v in out.values()], minlength=4)
    print(f"queries per photo 0/1/2/3: {counts.tolist()}")


def _bootstrap(delta: np.ndarray, resamples: int = 2000) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    means = delta[rng.integers(0, len(delta), (resamples, len(delta)))].mean(1)
    return float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


SUPPORT_KM = 25.0  # a result supports a candidate if it lies this close to it
WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def _informativeness(dev, results, truth, groups, tag, variant, report) -> None:
    """Does the evidence tell a chooser which candidate to trust, even when it adds no new candidate?

    support(c) = results within 25 km of candidate c. Reported on photos whose pool holds a correct candidate (within 25 km):
    how often the evidence touches the pool, its support on correct vs wrong candidates, the within-photo AUC of support
    for separating them (reference: the reranker's rank), and top-1 accuracy of score = -rank + w * support, with w fitted
    on `dev` and reused on other tags (one parameter, no learned model).
    """

    pools = [np.asarray(e.get("pool") or [c[1:] for c in e["candidates"]], dtype=float) for e in dev]
    dist = [_km(p, *truth[m]) for m, p in enumerate(pools)]  # km from each pooled candidate (reranker order) to the truth
    correct = [d < SUPPORT_KM for d in dist]
    answerable = np.asarray([c.any() for c in correct])
    arms = ("whole", "whole (matched)", "siglip", "bm25", "dense", "siglip+dense (12)")
    support = {a: [np.asarray([(_km(pts, *c) < SUPPORT_KM).sum() if len(pts) else 0 for c in p]) for p, pts in zip(pools, results[a])] for a in arms}

    def auc(score: np.ndarray, ok: np.ndarray) -> float:
        if not ok.any() or ok.all():
            return np.nan
        a, b = score[ok][:, None], score[~ok][None, :]
        return float((a > b).mean() + 0.5 * (a == b).mean())

    def hit(w: float, a: str, t: float) -> np.ndarray:
        return np.asarray([d[int(np.argmax(-np.arange(len(d)) + w * s))] < t for d, s in zip(dist, support[a])])

    weights: dict[str, float] = {}
    if tag != "dev":
        try:
            weights = json.loads((ROOT / "dev" / f"stage1{_suffix(variant)}.json").read_text(encoding="utf-8")).get("combiner w", {})
        except FileNotFoundError:
            pass
    fitted_on = "dev" if weights or tag == "dev" else "this set"
    for a in arms:
        if a not in weights:
            weights[a] = max(WEIGHTS, key=lambda w: (hit(w, a, 25.0).mean(), -w))
    report["combiner w"], report["combiner w fitted on"] = weights, fitted_on
    report["informativeness"] = {}
    base = {t: hit(0.0, "siglip", t) for t in THRESHOLDS}  # w = 0 is the reranker top-1
    for g, mask in groups.items():
        ans = mask & answerable
        rank_auc = float(np.nanmean([auc(-np.arange(len(d)).astype(float), c) for d, c, k in zip(dist, correct, ans) if k]))
        print(f"\n{g}: EVIDENCE INFORMATIVENESS on {ans.sum()} photos whose pool holds a correct candidate (<25 km); w fitted on {fitted_on}; "
              f"reranker-rank AUC {rank_auc:.3f}")
        print(f"  {'arm':20s} touches pool | support>0: correct / wrong cand. | AUC   | top-1 with evidence (w)  <1 km <25 km <200 km |  d<25 km [95% CI]  | fixed/broke (25 km)")
        report["informativeness"][g] = {"rank AUC": rank_auc}
        for a in arms:
            touches = float(np.mean([s.sum() > 0 for s, k in zip(support[a], ans) if k]))
            on_right = float(np.mean([(s[c] > 0).mean() for s, c, k in zip(support[a], correct, ans) if k]))
            on_wrong = float(np.mean([(s[~c] > 0).mean() for s, c, k in zip(support[a], correct, ans) if k and (~c).any()]))
            a_auc = float(np.nanmean([auc(s.astype(float), c) for s, c, k in zip(support[a], correct, ans) if k]))
            w = weights[a]
            tops = {t: hit(w, a, t)[mask] for t in THRESHOLDS}
            delta = _bootstrap(tops[25.0].astype(float) - base[25.0][mask])
            fixed, broke = int((tops[25.0] & ~base[25.0][mask]).sum()), int((~tops[25.0] & base[25.0][mask]).sum())
            report["informativeness"][g][a] = {"touches pool": touches, "support>0 correct": on_right, "support>0 wrong": on_wrong, "AUC": a_auc, "w": w,
                                               "top-1": {f"<{int(t)} km": float(v.mean()) for t, v in tops.items()}, "delta <25 km": delta,
                                               "fixed": fixed, "broke": broke}
            print(f"  {a:20s} {touches:11.0%} | {on_right:9.0%} / {on_wrong:5.0%}        | {a_auc:.3f} | w={w:5.2f}   "
                  + " ".join(f"{tops[t].mean():6.1%}" for t in THRESHOLDS) + f"   {delta[0]:+6.1%} [{delta[1]:+.1%}, {delta[2]:+.1%}]   {fixed}/{broke}")
        print(f"  reranker top-1 (w=0)  {'':41s}" + " ".join(f"{base[t][mask].mean():6.1%}" for t in THRESHOLDS))


def _search_batched(world, embeddings: np.ndarray, author: np.ndarray, batch: int = 8_000) -> dict[str, np.ndarray]:
    """`_search` in query batches: the score matrix of one pass is queries x 16k gallery rows x 4 bytes."""

    parts = [_search(world, embeddings[s : s + batch], author[s : s + batch]) for s in range(0, len(embeddings), batch)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _retrieve(
    tag: str, variant: str, dev: list[dict[str, Any]], generated: dict[str, Any], texts: list[tuple[int, str]],
    backends: Sequence[str] = ("whole", "siglip", "bm25", "dense"),
) -> dict[str, list[list[tuple[float, float]]]]:
    """Coordinates of each arm's results per photo (GPU: SigLIP2 text embeddings, gallery search, bge dense search).

    `backends` limits the arms computed (the others stay empty): "whole" (also fills "whole (matched)"), "siglip", "bm25", "dense".
    """

    import torch

    bench = "path" in dev[0]
    world = load_world() if bench else load_world(mp16_queries=[{"row": e["row"], "image_id": e["image_id"]} for e in dev])
    position = [e["index"] if bench else m for m, e in enumerate(dev)]
    initial = {}
    if "whole" in backends:
        with np.load((BENCH_ROOT if bench else SFT_ROOT) / "neighbors.npz") as saved:
            initial = {k: saved[k][[e["index"] for e in dev]] for k in ("mp16_raw_idx", "mp16_raw_sim", "osv_raw_idx", "osv_raw_sim")}

    found: dict[str, np.ndarray] = {}
    if "siglip" in backends:
        model, processor = _model()
        embeddings = []
        for start in range(0, len(texts), 128):
            tokens = processor(text=[t[1].lower() for t in texts[start : start + 128]], return_tensors="pt", padding="max_length",
                               max_length=TEXT_TOKENS, truncation=True).to("cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                embeddings.append(_features(model.get_text_features(**tokens)).float().cpu().numpy())
        del model
        torch.cuda.empty_cache()
        found = _search_batched(world, np.concatenate(embeddings), world.query_author[[position[m] for m, _ in texts]])
    dense = bm25 = [[] for _ in texts]
    if "dense" in backends or "bm25" in backends:
        wiki = Wiki(dense="dense" in backends)
        if "dense" in backends:
            dense = wiki.dense([q for _, q in texts], k=2 * RESULTS_PER_QUERY)
        if "bm25" in backends:
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
    matched: list[list[tuple[float, float]]] = [[] for _ in dev]  # whole-image results, one per corpus per query this photo got
    queries_per_photo = np.bincount([m for m, _ in texts], minlength=len(dev))
    for m in range(len(dev) if "whole" in backends else 0):
        for corpus in ("mp16", "osv"):
            results["whole"][m] += photo_results(m, corpus, initial[f"{corpus}_raw_idx"][m], initial[f"{corpus}_raw_sim"][m], WHOLE_PER_CORPUS, taken_whole[m])
        taken = set()
        for corpus in ("mp16", "osv"):
            matched[m] += photo_results(m, corpus, initial[f"{corpus}_raw_idx"][m], initial[f"{corpus}_raw_sim"][m], int(queries_per_photo[m]), taken)
    results["whole (matched)"] = matched
    taken_siglip = [set() for _ in dev]
    seen = {"bm25": [set() for _ in dev], "dense": [set() for _ in dev]}
    for j, (m, _) in enumerate(texts):
        for corpus in ("mp16", "osv") if found else ():
            results["siglip"][m] += photo_results(m, corpus, found[f"{corpus}_idx"][j], found[f"{corpus}_sim"][j], 1, taken_siglip[m])
        for name, hits in (("bm25", bm25[j]), ("dense", dense[j])):
            fresh = [h for h in hits if h["id"] not in seen[name][m]][:RESULTS_PER_QUERY]
            seen[name][m].update(h["id"] for h in fresh)
            results[name][m] += [(h["lat"], h["lon"]) for h in fresh]
    results["siglip+dense (12)"] = [a + b for a, b in zip(results["siglip"], results["dense"])]
    return results


def evaluate(tag: str, variant: str = "v2") -> None:
    dev = _load(tag, "dev.json")
    generated = _load(tag, f"places{_suffix(variant)}.json")
    texts = [(m, q) for m, e in enumerate(dev) for q in generated[e["image_id"]]["queries"]]
    bench = "path" in dev[0]
    cache = ROOT / tag / f"results{_suffix(variant)}.json"  # retrieved coordinates; used only if newer than the photo list and the queries
    inputs = (ROOT / tag / "dev.json", ROOT / tag / f"places{_suffix(variant)}.json")
    if cache.exists() and all(cache.stat().st_mtime > p.stat().st_mtime for p in inputs):
        results = {k: [[tuple(p) for p in pts] for pts in v] for k, v in json.loads(cache.read_text(encoding="utf-8")).items()}
    else:
        results = _retrieve(tag, variant, dev, generated, texts)
        cache.write_text(json.dumps(results) + "\n", encoding="utf-8")

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
    _informativeness(dev, results, truth, groups, tag, variant, report)
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
