# Can the base 4B model write search queries that fetch useful evidence? Fixed loop, no training (QUERY_EVIDENCE_PLAN.md).
# Usage: .venv/bin/python -m geo_search_env.experiment.query_evidence {select,generate,retrieve,answer,report} --tag dev --n 1000
#        (generate and answer need vLLM on :8765, retrieve needs the GPU; scripts/query_evidence.sh runs them all)

"""Stage 0 initial answer, stage 1 search queries, stage 2 revised answer after six evidence photos.

Dev photos: Pinpoint-held-out MP16 val photos (bucket 99, validation photographers, no burned-in GPS), shown the reranker's
top-10 candidates as text. Evidence photos come from MP16 + OSV-5M (SigLIP2 giant), same-photographer and near-duplicate
(cosine >= 0.95 to the query photo) rows excluded. Arms, all with six evidence photos and one prompt:

revise:  no new evidence, just ask again
whole:   top-3 per corpus for the query photo's own embedding
caption: top-3 per corpus for one general caption
visual:  one photo per corpus for each of three queries describing what is visible
geo:     the same for three queries that each name a possible place
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import time
import urllib.request
from typing import Any, Sequence

import numpy as np

from .evidence_test import _image_part, _thumbnail
from .pivot_diagnostics import PROMPT, format_options, parse_coordinates
from .query_headroom import _model, _features, _search
from .sft_data import OVERLAY_THRESHOLD, MP16Images
from .strategy_search import MP16_EMBED, OSV_EMBED, _haversine_km, load_world
from .verifiers import _place_names


SFT_ROOT = Path("artifacts/sft")
BENCH_ROOT = Path("artifacts/strategy_search")
ROOT = Path("artifacts/query_evidence")
OSV_IMAGES = Path("/data/hf/datasets/osv5m/images/train")
CANDIDATES = 10
PER_CORPUS = 3  # evidence photos per corpus in the whole-image and caption arms; the query arms take one per query
TEXT_TOKENS = 64
NEAR_DUPLICATE_SIM = 0.95
ARMS = ("revise", "whole", "caption", "visual", "geo")
THRESHOLDS = (1.0, 25.0)

VISUAL_PROMPT = """Where was this photo taken? Candidates:
{options}
Write 3 short search queries (max 8 words each) describing distinctive things you see. No place names.
Answer: ```json
{{"queries": ["", "", ""]}}
```"""
GEO_PROMPT = """Where was this photo taken? Candidates:
{options}
Write 3 short search queries (max 8 words each), each naming a different possible place and one thing visible there.
Answer: ```json
{{"queries": ["", "", ""]}}
```"""
CAPTION_PROMPT = "Describe this photo in one short search query (max 12 words)."
ANSWER_FORMAT = """Think briefly (under 120 words), then end with exactly one JSON block: ```json
{{"lat": <latitude>, "lon": <longitude>}}
```"""
REVISE_PROMPT = "Where was this photo taken? Candidates:\n{options}\nYour earlier answer was ({lat:.3f}, {lon:.3f}). Give your final answer.\n" + ANSWER_FORMAT
EVIDENCE_PROMPT = "Where was the first photo taken? Candidates:\n{options}\nPhotos 2-{last} are search results, taken at: {places}. They may be wrong.\n" + ANSWER_FORMAT


def parse_queries(answer: str) -> list[str]:
    """Distinct non-empty strings from the answer's "queries" list (at most three), in order."""

    match = re.search(r'"queries"\s*:\s*\[(.*?)\]', answer, re.DOTALL)
    found = [s.strip() for s in re.findall(r'"((?:[^"\\]|\\.)*)"', match.group(1))] if match else []
    return list(dict.fromkeys(s for s in found if s))[:3]


def _path(tag: str, name: str) -> Path:
    return ROOT / tag / name


def _load(tag: str, name: str) -> Any:
    return json.loads(_path(tag, name).read_text(encoding="utf-8"))


def _save(tag: str, name: str, value: Any) -> None:
    _path(tag, name).write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def select_bench(tag: str, n: int) -> None:
    """Validation photos: `n` // 2 random eval-half photos from each of im2gps3k and yfcc4k (fixed seed), with reranker top-10 candidates.

    Entries carry the benchmark image `path`; "index" is the position in the benchmark caches. Burned-in GPS photos are not filtered
    (no overlay scores exist for the benchmarks).
    """

    from scipy.spatial import cKDTree

    from .strategy_search import BENCHMARK_NAMES, _xyz
    from .verifiers import reranker_ranking

    world = load_world()
    coords, valid, _, ranking = reranker_ranking(world, BENCH_ROOT)
    rng = np.random.default_rng(0)
    picked: list[int] = []
    for name in BENCHMARK_NAMES:
        members = [q for q, x in enumerate(world.queries) if x["benchmark"] == name and x["split"] == "eval"]
        picked += sorted(int(q) for q in rng.permutation(members)[: n // 2])
    shown = {q: [int(c) for c in ranking[q, :CANDIDATES] if valid[q, c]] for q in picked}
    tree = cKDTree(_xyz(world.mp16["latlon"]))
    name_row = {}
    for q in picked:  # place name of the nearest MP16 photo to each candidate, skipping the query's photographer
        for c in shown[q]:
            _, found = tree.query(_xyz(coords[q, c]), k=16)
            other = [int(r) for r in found if world.mp16["author"][r] != world.query_author[q]]
            name_row[(q, c)] = int(world.mp16["row_index"][other[0] if other else found[0]])
    names = _place_names(set(name_row.values()))
    val = [
        {
            "index": q, "image_id": world.queries[q]["image_id"], "path": world.queries[q]["path"], "benchmark": world.queries[q]["benchmark"],
            "truth": world.query_latlon[q].tolist(),
            "candidates": [[names[name_row[(q, c)]], float(coords[q, c, 0]), float(coords[q, c, 1])] for c in shown[q]],
            "pool": [[float(coords[q, c, 0]), float(coords[q, c, 1])] for c in ranking[q] if valid[q, c]],  # every pooled candidate, best reranker rank first
        }
        for q in picked
    ]
    (ROOT / tag).mkdir(parents=True, exist_ok=True)
    _save(tag, "dev.json", val)
    print(f"{len(val)} validation photos: " + ", ".join(f"{b} {sum(v['benchmark'] == b for v in val)}" for b in BENCHMARK_NAMES))


def select(tag: str, n: int) -> None:
    """`n` random dev photos (the first `n` of one fixed shuffle, so a pilot is a prefix of the full set) with their candidates."""

    queries = json.loads((SFT_ROOT / "queries.json").read_text(encoding="utf-8"))
    scores = json.loads((SFT_ROOT / "overlay_scores.json").read_text(encoding="utf-8"))
    eligible = [
        i for i, q in enumerate(queries)
        if q["group"] == "held_out" and q["split"] == "val" and scores.get(q["image_id"]) is not None and scores[q["image_id"]] < OVERLAY_THRESHOLD
    ]
    picked = np.random.default_rng(0).permutation(eligible)[:n]
    saved = np.load(SFT_ROOT / "candidates.npz")
    coords, valid, ranking, name_row = saved["coords"], saved["valid"], saved["ranking"], saved["name_row"]
    latlon = np.fromfile(MP16_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2).astype(np.float64)
    shown = {int(i): [int(c) for c in ranking[i, :CANDIDATES] if valid[i, c]] for i in picked}
    names = _place_names({int(name_row[i, c]) for i, cs in shown.items() for c in cs})
    dev = [
        {
            "index": int(i), "image_id": queries[i]["image_id"], "row": queries[i]["row"], "truth": latlon[queries[i]["row"]].tolist(),
            "candidates": [[names[int(name_row[i, c])], float(coords[i, c, 0]), float(coords[i, c, 1])] for c in shown[int(i)]],
            "pool": [[float(coords[i, c, 0]), float(coords[i, c, 1])] for c in ranking[i] if valid[i, c]],  # every pooled candidate, best reranker rank first
        }
        for i in sorted(picked)
    ]
    (ROOT / tag).mkdir(parents=True, exist_ok=True)
    _save(tag, "dev.json", dev)
    print(f"{len(dev)} dev photos of {len(eligible)} eligible")


def _chat(server: str, content: list[dict[str, Any]], max_tokens: int) -> str:
    body = {
        "model": "vlm", "temperature": 0.0, "seed": 0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": content}],
    }
    request = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.loads(response.read())["choices"][0]["message"]["content"] or ""
        except Exception:
            time.sleep(5 * 2**attempt)
    return ""


def _photo(images: MP16Images, entry: dict[str, Any]) -> dict[str, Any]:
    data = Path(entry["path"]).read_bytes() if "path" in entry else images.read(entry["image_id"])
    return _image_part(base64.b64encode(data).decode())


def _options(entry: dict[str, Any]) -> str:
    return format_options([tuple(c) for c in entry["candidates"]])


def _parallel(function, items, workers: int = 64) -> list:
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(function, items))


def generate(tag: str, server: str) -> None:
    """Stage 0 (the default prompt with the candidates) and stage 1 (visual, geographic and caption queries) for every dev photo."""

    dev = _load(tag, "dev.json")
    images = MP16Images()
    jobs = [(e, kind) for e in dev for kind in ("initial", "visual", "geo", "caption")]

    def run(job):
        entry, kind = job
        prompt = {
            "initial": PROMPT.format(options=_options(entry)), "visual": VISUAL_PROMPT.format(options=_options(entry)),
            "geo": GEO_PROMPT.format(options=_options(entry)), "caption": CAPTION_PROMPT,
        }[kind]
        raw = _chat(server, [_photo(images, entry), {"type": "text", "text": prompt}], 600 if kind == "initial" else 150)
        return {"raw": raw}

    out: dict[str, dict[str, Any]] = {e["image_id"]: {} for e in dev}
    for (entry, kind), result in zip(jobs, _parallel(run, jobs)):
        if kind == "initial":
            result["answer"] = parse_coordinates(result["raw"])
        elif kind == "caption":
            result["queries"] = [result["raw"].strip().split("\n")[0].strip(' "`')] if result["raw"].strip() else []
        else:
            result["queries"] = parse_queries(result["raw"])
        out[entry["image_id"]][kind] = result
    _save(tag, "generated.json", out)
    for kind in ("initial", "visual", "geo", "caption"):
        bad = sum(not (r[kind].get("answer") if kind == "initial" else r[kind]["queries"]) for r in out.values())
        print(f"{kind}: {bad}/{len(dev)} unparsed or empty")


def _osv_path(image_id: str, folders: list[Path]) -> Path | None:
    return next((p for d in folders if (p := d / f"{image_id}.jpg").exists()), None)


class Evidence:
    """Thumbnails of gallery photos by (corpus, row); None when the photo cannot be read."""

    def __init__(self) -> None:
        self.ids = {"mp16": (MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines(),
                    "osv": (OSV_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines()}
        self.mp16 = MP16Images()
        self.folders = sorted(p for p in OSV_IMAGES.iterdir() if p.is_dir())

    def thumbnail(self, corpus: str, row: int) -> str | None:
        image_id = self.ids[corpus][row]
        try:
            if corpus == "mp16":
                data = self.mp16.read(image_id)
            else:
                path = _osv_path(image_id, self.folders)
                data = path.read_bytes() if path else None
            return _thumbnail(data) if data else None
        except Exception:
            return None


def retrieve(tag: str) -> None:
    """Embed every generated query, search both galleries, and pick the six evidence photos of each arm."""

    import torch

    dev = _load(tag, "dev.json")
    generated = _load(tag, "generated.json")
    bench = "path" in dev[0]  # validation photos come from the benchmarks, dev photos from MP16
    world = load_world() if bench else load_world(mp16_queries=[{"row": e["row"], "image_id": e["image_id"]} for e in dev])
    position = [e["index"] if bench else m for m, e in enumerate(dev)]  # row of each photo in world.query_*
    with np.load((BENCH_ROOT if bench else SFT_ROOT) / "neighbors.npz") as saved:
        initial = {k: saved[k][[e["index"] for e in dev]] for k in ("mp16_raw_idx", "mp16_raw_sim", "osv_raw_idx", "osv_raw_sim")}
    evidence = Evidence()

    texts: list[tuple[int, str, str]] = []  # (dev position, arm, query)
    for m, e in enumerate(dev):
        g = generated[e["image_id"]]
        texts += [(m, "caption", q) for q in g["caption"]["queries"]]
        texts += [(m, arm, q) for arm in ("visual", "geo") for q in g[arm]["queries"]]
    model, processor = _model()
    embeddings, lengths = [], []
    for start in range(0, len(texts), 128):
        batch = [t[2].lower() for t in texts[start : start + 128]]
        lengths += [len(ids) for ids in processor.tokenizer(batch).input_ids]
        tokens = processor(text=batch, return_tensors="pt", padding="max_length", max_length=TEXT_TOKENS, truncation=True).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            embeddings.append(_features(model.get_text_features(**tokens)).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    found = _search(world, np.concatenate(embeddings), world.query_author[[position[t[0]] for t in texts]])
    print(f"query token counts: max {max(lengths)}, truncated {sum(n > TEXT_TOKENS for n in lengths)}/{len(lengths)}", flush=True)

    def ranked(source: dict[str, np.ndarray], j: int, prefix: str) -> dict[str, list[tuple[int, float]]]:
        return {c: [(int(r), float(s)) for r, s in zip(source[f"{prefix}{c}_idx"][j], source[f"{prefix}{c}_sim"][j]) if np.isfinite(s)]
                for c in ("mp16", "osv")}

    def take(m: int, corpus: str, hits: list[tuple[int, float]], count: int, taken: set, query: str | None) -> list[dict[str, Any]]:
        e, gallery, cards = dev[m], world.mp16 if corpus == "mp16" else world.osv, []
        q_emb = world.query_embeddings[position[m]] / np.linalg.norm(world.query_embeddings[position[m]])
        shown = np.asarray([c[1:] for c in e["candidates"]])
        for row, sim in hits:
            if len(cards) == count:
                break
            g_emb = np.asarray(gallery["embeddings"][row], dtype=np.float32)
            if (row_key := (corpus, row)) in taken or float(g_emb @ q_emb / np.linalg.norm(g_emb)) >= NEAR_DUPLICATE_SIM:
                continue
            if evidence.thumbnail(corpus, row) is None:
                continue
            taken.add(row_key)
            lat, lon = (float(x) for x in gallery["latlon"][row])
            cards.append({
                "corpus": corpus, "row": row, "image_id": evidence.ids[corpus][row], "lat": lat, "lon": lon, "sim": sim, "query": query,
                "in_initial": bool(row in initial[f"{corpus}_raw_idx"][m]),
                "near_shown_km": float(_haversine_km(lat, lon, shown).min()) if len(shown) else None,
            })
        return cards

    texts_by = {(m, arm): [] for m in range(len(dev)) for arm in ("caption", "visual", "geo")}
    for j, (m, arm, query) in enumerate(texts):
        texts_by[(m, arm)].append((j, query))
    packages: dict[str, dict[str, Any]] = {}
    for m, e in enumerate(dev):
        whole = {c: list(zip(initial[f"{c}_raw_idx"][m].tolist(), initial[f"{c}_raw_sim"][m].tolist())) for c in ("mp16", "osv")}
        whole = {c: [(r, s) for r, s in hits if np.isfinite(s)] for c, hits in whole.items()}
        package: dict[str, list[dict[str, Any]]] = {}
        taken: set = set()
        package["whole"] = [c for corpus in ("mp16", "osv") for c in take(m, corpus, whole[corpus], PER_CORPUS, taken, None)]
        for arm in ("caption", "visual", "geo"):
            taken, cards = set(), []
            for j, query in texts_by[(m, arm)]:
                hits = ranked(found, j, "")
                if arm == "caption":
                    cards += [c for corpus in ("mp16", "osv") for c in take(m, corpus, hits[corpus], PER_CORPUS, taken, query)]
                else:
                    cards += [c for corpus in ("mp16", "osv") for c in take(m, corpus, hits[corpus], 1, taken, query)]
            package[arm] = cards
        packages[e["image_id"]] = package
    _save(tag, "retrieved.json", packages)
    for arm in ("whole", "caption", "visual", "geo"):
        sizes = [len(p[arm]) for p in packages.values()]
        overlap = np.mean([c["in_initial"] for p in packages.values() for c in p[arm]])
        print(f"{arm}: {np.mean(sizes):.2f} cards per photo, {sum(s < 6 for s in sizes)} short; {overlap:.0%} already in the initial top-1000")


def answer(tag: str, server: str) -> None:
    """Stage 2: ask again with each arm's evidence (or none), same prompt and decoding."""

    dev = _load(tag, "dev.json")
    generated = _load(tag, "generated.json")
    packages = _load(tag, "retrieved.json")
    images, evidence = MP16Images(), Evidence()
    thumbs: dict[tuple[str, int], str | None] = {}

    def content(entry: dict[str, Any], arm: str) -> list[dict[str, Any]]:
        parts = [_photo(images, entry)]
        cards = [] if arm == "revise" else packages[entry["image_id"]][arm]
        if not cards:
            earlier = generated[entry["image_id"]]["initial"]["answer"] or entry["candidates"][0][1:]
            return parts + [{"type": "text", "text": REVISE_PROMPT.format(options=_options(entry), lat=earlier[0], lon=earlier[1])}]
        for c in cards:
            key = (c["corpus"], c["row"])
            if key not in thumbs:
                thumbs[key] = evidence.thumbnail(*key)
            parts.append(_image_part(thumbs[key]))
        places = "; ".join(f"{k} ({c['lat']:.3f}, {c['lon']:.3f})" for k, c in enumerate(cards, start=2))
        return parts + [{"type": "text", "text": EVIDENCE_PROMPT.format(options=_options(entry), last=len(cards) + 1, places=places)}]

    jobs = [(e, arm) for e in dev for arm in ARMS]
    raws = _parallel(lambda job: _chat(server, content(*job), 600), jobs)
    out: dict[str, dict[str, Any]] = {e["image_id"]: {} for e in dev}
    for (entry, arm), raw in zip(jobs, raws):
        out[entry["image_id"]][arm] = {"raw": raw, "answer": parse_coordinates(raw)}
    _save(tag, "answers.json", out)
    for arm in ARMS:
        print(f"{arm}: {sum(r[arm]['answer'] is None for r in out.values())}/{len(dev)} unparsed")


def _paired(better: np.ndarray, base: np.ndarray, *, resamples: int = 2000) -> tuple[float, float, float]:
    """Mean difference in hit rate and its 95% bootstrap interval over queries."""

    rng = np.random.default_rng(0)
    diff = better.astype(float) - base.astype(float)
    means = diff[rng.integers(0, len(diff), (resamples, len(diff)))].mean(1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(tag: str) -> None:
    dev = _load(tag, "dev.json")
    generated = _load(tag, "generated.json")
    answers = _load(tag, "answers.json")
    packages = _load(tag, "retrieved.json")
    truth = np.asarray([e["truth"] for e in dev])
    top1 = np.asarray([e["candidates"][0][1:] for e in dev])
    initial = np.asarray([generated[e["image_id"]]["initial"]["answer"] or tuple(e["candidates"][0][1:]) for e in dev])
    points = {"none": initial, "reranker top-1": top1}
    for arm in ARMS:
        points[arm] = np.asarray([answers[e["image_id"]][arm]["answer"] or tuple(initial[m]) for m, e in enumerate(dev)])
    distance = {k: _haversine_all(v, truth) for k, v in points.items()}
    hit = {k: {t: d < t for t in THRESHOLDS} for k, d in distance.items()}
    shown = [np.asarray([c[1:] for c in e["candidates"]]) for e in dev]
    report: dict[str, Any] = {"n": len(dev)}
    print(f"n={len(dev)}            <1 km   <25 km   fixed  broke   vs none (25 km)        vs whole (25 km)")
    for k in points:
        row = {f"<{int(t)} km": float(hit[k][t].mean()) for t in THRESHOLDS}
        if k in ARMS:
            row["fixed"], row["broke"] = int((hit[k][25.0] & ~hit["none"][25.0]).sum()), int((~hit[k][25.0] & hit["none"][25.0]).sum())
            row["vs none"], row["vs whole"] = (_paired(hit[k][25.0], hit["none"][25.0]), _paired(hit[k][25.0], hit["whole"][25.0]))
            row["unparsed"] = float(np.mean([answers[e["image_id"]][k]["answer"] is None for e in dev]))
            row["placeholder"] = float(np.mean([answers[e["image_id"]][k]["answer"] == [0.0, 0.0] for e in dev]))
            print(f"  {k:14s} {row['<1 km']:6.1%} {row['<25 km']:7.1%} {row['fixed']:6d} {row['broke']:6d}   "
                  f"{row['vs none'][0]:+.1%} [{row['vs none'][1]:+.1%}, {row['vs none'][2]:+.1%}]   {row['vs whole'][0]:+.1%} [{row['vs whole'][1]:+.1%}, {row['vs whole'][2]:+.1%}]   unparsed {row['unparsed']:.0%}, 0/0 {row['placeholder']:.0%}")
        else:
            print(f"  {k:14s} {row['<1 km']:6.1%} {row['<25 km']:7.1%}")
        report[k] = row
    oracle = {"shown candidates": [_haversine_all(s, truth[m:m + 1]).min() for m, s in enumerate(shown)]}
    for arm in ("whole", "caption", "visual", "geo"):
        evidence = [np.asarray([[c["lat"], c["lon"]] for c in packages[e["image_id"]][arm]]).reshape(-1, 2) for e in dev]
        oracle[f"shown + {arm} evidence"] = [_haversine_all(np.vstack([s, ev]), truth[m:m + 1]).min() for m, (s, ev) in enumerate(zip(shown, evidence))]
    print("candidate coverage (oracle)   <1 km   <25 km")
    for k, d in oracle.items():
        d = np.asarray(d)
        report[k] = {f"<{int(t)} km": float((d < t).mean()) for t in THRESHOLDS}
        print(f"  {k:28s} {(d < 1).mean():6.1%} {(d < 25).mean():7.1%}")
    for arm in ("visual", "geo", "caption"):
        queries = [generated[e["image_id"]][arm]["queries"] for e in dev]
        report[f"{arm} queries"] = {"mean per photo": float(np.mean([len(q) for q in queries])), "empty": int(sum(not q for q in queries))}
    _save(tag, "report.json", report)


def _haversine_all(points: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Distance in km from each point to the matching truth row (or to the one truth row)."""

    truth = np.broadcast_to(truth, points.shape)
    lat1, lon1, lat2, lon2 = (np.radians(a) for a in (points[:, 0], points[:, 1], truth[:, 0], truth[:, 1]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("select", "generate", "retrieve", "answer", "report"))
    parser.add_argument("--tag", default="dev", help="output folder under artifacts/query_evidence")
    parser.add_argument("--n", type=int, default=1000, help="select: number of photos")
    parser.add_argument("--source", choices=("mp16", "bench"), default="mp16", help="select: MP16 dev photos or benchmark eval-half validation photos")
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    args = parser.parse_args(argv)
    if args.node == "select":
        (select_bench if args.source == "bench" else select)(args.tag, args.n)
    elif args.node == "generate":
        generate(args.tag, args.server)
    elif args.node == "retrieve":
        retrieve(args.tag)
    elif args.node == "answer":
        answer(args.tag, args.server)
    else:
        report(args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
