# Go/no-go checks for query-rewriting retrieval: how much could crop and text queries add over whole-image retrieval?
# Usage: .venv/bin/python -m geo_search_env.experiment.query_headroom {crops,text}

"""Oracle headroom of alternative retrieval queries on the benchmark eval halves (same-photographer rows excluded).

crops: SigLIP2 embeddings of 10 fixed crops per eval photo (a 3x3 grid of half-size crops + a centre 2/3 crop), each
       searched in MP16 + OSV-5M. Compared at equal budget with the whole image's top-10 distinct locations, and as an
       addition to the candidates the model sees today.
text:  SigLIP2 text embeddings of the place names Gemini named with high confidence, searched the same way, compared
       with GeoNames exact-name lookup and with whole-image retrieval on the same photos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .strategy_search import THRESHOLDS_KM, _haversine_km, _pool, _stream_topk, load_world
from .verifiers import reranker_ranking


ROOT = Path("artifacts/strategy_search")
MODEL = "google/siglip2-giant-opt-patch16-384"
TOP_K = 100
CROPS = [(x, y, 0.5, 0.5) for y in (0.0, 0.25, 0.5) for x in (0.0, 0.25, 0.5)] + [(1 / 6, 1 / 6, 2 / 3, 2 / 3)]  # (left, top, w, h)


def _model():
    import torch
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    return model, AutoProcessor.from_pretrained(MODEL)


def _features(output):
    return output if not hasattr(output, "pooler_output") else output.pooler_output


def _search(world, embeddings: np.ndarray, author: np.ndarray) -> dict[str, np.ndarray]:
    """Top-K raw matches of each (unnormalized) embedding in MP16 and OSV, best first, as (lat, lon, sim)."""

    import torch
    import torch.nn.functional as F

    queries = F.normalize(torch.as_tensor(embeddings, device="cuda").float(), dim=-1).half()
    out = {}
    for name, gallery in (("mp16", world.mp16), ("osv", world.osv)):
        print(f"searching {name} for {len(queries)} queries", flush=True)
        out[f"{name}_idx"], out[f"{name}_sim"] = _stream_topk(gallery["embeddings"], queries, author, gallery["author"], TOP_K, normalize=True, chunk=16_384)
    return out


def _hits(world, found: dict[str, np.ndarray], i: int) -> list[tuple[float, float, float]]:
    rows = []
    for name, gallery in (("mp16", world.mp16), ("osv", world.osv)):
        sim, idx = found[f"{name}_sim"][i], found[f"{name}_idx"][i]
        keep = np.isfinite(sim)
        rows += [(float(a), float(b), float(s)) for (a, b), s in zip(gallery["latlon"][idx[keep]], sim[keep])]
    return sorted(rows, key=lambda r: -r[2])


def _rates(distance: np.ndarray) -> dict[str, float]:
    return {f"<{int(t)} km": float((distance < t).mean()) for t in THRESHOLDS_KM[:2]}


def crops(*, batch: int = 64) -> None:
    import torch
    from PIL import Image

    world = load_world()
    members = [i for i, q in enumerate(world.queries) if q["split"] == "eval"]
    cache_path = ROOT / "query_headroom_crops.npz"
    if cache_path.exists():
        found = dict(np.load(cache_path))
    else:
        model, processor = _model()
        embeddings = np.zeros((len(members), len(CROPS) + 1, model.config.vision_config.hidden_size), dtype=np.float32)
        for start in range(0, len(members), batch):
            images = []
            for q in members[start : start + batch]:
                image = Image.open(world.queries[q]["path"]).convert("RGB")
                w, h = image.size
                images.append(image)  # whole image first, as a check against the cached query embedding
                images += [image.crop((int(x * w), int(y * h), int((x + cw) * w), int((y + ch) * h))) for x, y, cw, ch in CROPS]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                pixels = processor(images=images, return_tensors="pt")["pixel_values"].cuda()
                feats = _features(model.get_image_features(pixel_values=pixels)).float().cpu().numpy()
            embeddings[start : start + len(images) // (len(CROPS) + 1)] = feats.reshape(-1, len(CROPS) + 1, feats.shape[-1])
            if (start // batch) % 10 == 0:
                print(f"  embedded {start + batch}/{len(members)}", flush=True)
        whole = embeddings[:, 0]
        cached = world.query_embeddings[members]
        cos = (whole * cached).sum(1) / (np.linalg.norm(whole, axis=1) * np.linalg.norm(cached, axis=1))
        print(f"whole-image re-embedding vs cached query embedding: cosine median {np.median(cos):.4f}, min {cos.min():.4f}", flush=True)
        author = np.repeat(world.query_author[members], len(CROPS))
        found = _search(world, embeddings[:, 1:].reshape(-1, embeddings.shape[-1]), author)
        np.savez(cache_path, **found)

    coords, valid, _, _ = reranker_ranking(world, ROOT)
    with np.load(ROOT / "neighbors.npz") as saved:
        cache = {k: saved[k] for k in ("mp16_raw_idx", "mp16_raw_sim", "osv_raw_idx", "osv_raw_sim")}
    report: dict[str, Any] = {}
    for bench in ("im2gps3k", "yfcc4k", "both"):
        rows = [(m, q) for m, q in enumerate(members) if bench == "both" or world.queries[q]["benchmark"] == bench]
        d: dict[str, list] = {k: [] for k in ("whole top-1", "centre crop top-1", "whole top-10 (oracle)", "10 crops top-1 (oracle)",
                                              "shown candidates (oracle)", "shown + 10 crops (oracle)", "shown + whole top-10 (oracle)")}
        for m, q in rows:
            truth = world.query_latlon[q]
            whole_rows = []
            for name, gallery in (("mp16_raw", world.mp16), ("osv_raw", world.osv)):
                sim, idx = cache[f"{name}_sim"][q, :TOP_K], cache[f"{name}_idx"][q, :TOP_K]
                keep = np.isfinite(sim)
                whole_rows += [(float(a), float(b), float(s)) for (a, b), s in zip(gallery["latlon"][idx[keep]], sim[keep])]
            whole = np.asarray(_pool(whole_rows)[:10])
            crop_top = np.asarray([_hits(world, found, m * len(CROPS) + c)[0][:2] for c in range(len(CROPS))])
            shown = coords[q, valid[q]]
            km = lambda pts: _haversine_km(*truth, np.asarray(pts))
            d["whole top-1"].append(km(whole[:1])[0])
            d["centre crop top-1"].append(km(crop_top[-1:])[0])
            d["whole top-10 (oracle)"].append(km(whole).min())
            d["10 crops top-1 (oracle)"].append(km(crop_top).min())
            d["shown candidates (oracle)"].append(km(shown).min())
            d["shown + 10 crops (oracle)"].append(min(km(shown).min(), km(crop_top).min()))
            d["shown + whole top-10 (oracle)"].append(min(km(shown).min(), km(whole).min()))
        report[bench] = {"n": len(rows), **{k: _rates(np.asarray(v)) for k, v in d.items()}}
        print(f"\n{bench} n={len(rows)}          <1 km   <25 km")
        for k, v in report[bench].items():
            if k != "n":
                print(f"  {k:32s} {v['<1 km']:6.1%} {v['<25 km']:7.1%}")
    (ROOT / "query_headroom_crops.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def text() -> None:
    import torch

    world = load_world()
    labels = json.loads((ROOT / "llm_advantage_labels.json").read_text(encoding="utf-8"))
    gazetteer = {r["image_id"]: r for r in json.loads((ROOT / "llm_advantage_gazetteer.json").read_text(encoding="utf-8"))["rows"]}
    index = {q["image_id"]: i for i, q in enumerate(world.queries)}
    named = [(k, v["named_place"]) for k, v in labels.items() if v and v.get("named_place") and v.get("confidence") == "high"]
    queries = [index[k] for k, _ in named]
    model, processor = _model()
    embeddings = []
    for start in range(0, len(named), 64):
        texts = [f"a photo of {name}" for _, name in named[start : start + 64]]
        tokens = processor(text=texts, return_tensors="pt", padding="max_length", max_length=64, truncation=True).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            embeddings.append(_features(model.get_text_features(**tokens)).float().cpu().numpy())
    found = _search(world, np.concatenate(embeddings), world.query_author[queries])
    with np.load(ROOT / "neighbors.npz") as saved:
        cache = {k: saved[k] for k in ("mp16_raw_idx", "mp16_raw_sim", "osv_raw_idx", "osv_raw_sim")}
    d: dict[str, list] = {k: [] for k in ("text top-1", "text top-1 cluster", "text top-10 clusters (oracle)", "GeoNames most populous",
                                          "GeoNames best match (oracle)", "whole-image top-1 cluster")}
    for i, ((image_id, name), q) in enumerate(zip(named, queries)):
        truth = world.query_latlon[q]
        rows = _hits(world, found, i)
        clusters = _pool(rows)[:10]
        whole_rows = []
        for key, gallery in (("mp16_raw", world.mp16), ("osv_raw", world.osv)):
            sim, idx = cache[f"{key}_sim"][q, :TOP_K], cache[f"{key}_idx"][q, :TOP_K]
            keep = np.isfinite(sim)
            whole_rows += [(float(a), float(b), float(s)) for (a, b), s in zip(gallery["latlon"][idx[keep]], sim[keep])]
        km = lambda pts: _haversine_km(*truth, np.asarray(pts))
        d["text top-1"].append(km([rows[0][:2]])[0])
        d["text top-1 cluster"].append(km(clusters[:1])[0])
        d["text top-10 clusters (oracle)"].append(km(clusters).min())
        d["GeoNames most populous"].append(gazetteer[image_id]["pick_km"])
        d["GeoNames best match (oracle)"].append(gazetteer[image_id]["oracle_km"])
        d["whole-image top-1 cluster"].append(km(_pool(whole_rows)[:1])[0])
    report = {"n": len(named), **{k: _rates(np.asarray(v)) for k, v in d.items()}}
    (ROOT / "query_headroom_text.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nhigh-confidence named places n={len(named)}   <1 km   <25 km")
    for k, v in report.items():
        if k != "n":
            print(f"  {k:32s} {v['<1 km']:6.1%} {v['<25 km']:7.1%}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("crops", "text"))
    args = parser.parse_args(argv)
    crops() if args.node == "crops" else text()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
