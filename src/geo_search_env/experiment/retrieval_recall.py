# Recall of Pinpoint's three retrieval searches (RETRIEVAL_PLAN.md step 1): where the nearest right index photo ranks in each modality.
# Usage: .venv/bin/python -m geo_search_env.experiment.retrieval_recall ranks --photo-set dev3k --tag released
#        .venv/bin/python -m geo_search_env.experiment.retrieval_recall report --tag released

"""ranks: for each query and each search (contrastive image-image, raw SigLIP2 cosine, image-to-GPS, as models.pinpoint_reranker._search, fp16
scores over the resident index, original protocol: nothing excluded), the exact rank of the best-scoring index photo within 1 / 25 km of the truth
= the number of index photos scoring strictly higher. Computed without truncation: the best right score per query is taken over the index rows
within the radius, then one streaming pass counts the rows above it. Also with the query photographer's own photos removed (rank among the
others), and the rank of the best own photo at any distance (own photo in the pool). recall@K of a modality = P(rank < K); a union pool
(k_image, k_raw, k_gps) has a right candidate iff some modality's rank < its k. Saved to artifacts/retrieval/recall_{photo_set}_{tag}.npz.

report: recall@K per modality and for union pools (released 8 / 2 / 2, ours 32 / 8 / 8, deeper), rank quantiles, sliced by own photo in the
32 / 8 / 8 pool, gallery density near the truth (index photos within 25 km) and scene (MP16-Pro indoor / natural / urban)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

OUT = Path("artifacts/retrieval")
MODALITIES = ("image", "raw_image", "gps")
# rank variants per modality: best right photo within 1 / 25 km, the same among photos not by the query's photographer, the best own photo
VARIANTS = ("r1", "r25", "r1_other", "r25_other", "own")
RADII_KM = (1.0, 25.0)
EARTH_KM = 6371.0088


def _xyz(latlon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(latlon[:, 0]), np.radians(latlon[:, 1])
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)


def _mp16_meta(query_rows: np.ndarray):
    """Author code per MP16 embedding row, and (indoor, natural, urban) probabilities of the query rows."""

    from .strategy_search import MP16_CSV, MP16_EMBED

    row_index = np.fromfile(MP16_EMBED / "row_index.i64.bin", dtype=np.int64)
    need = {int(r): i for i, r in enumerate(row_index[query_rows])}
    vocab: dict[str, int] = {}
    authors, scene = [], np.zeros((len(query_rows), 3), np.float32)
    with MP16_CSV.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        a = header.index("AUTHOR")
        probs = [header.index(k) for k in ("Prob_indoor", "Prob_natural", "Prob_urban")]
        for r, row in enumerate(reader):
            authors.append(vocab.setdefault(row[a], len(vocab)))
            if r in need:
                scene[need[r]] = [float(row[k] or 0) for k in probs]
    return np.asarray(authors, dtype=np.int64)[row_index], scene


def ranks(photo_set: str, tag: str, index_dir: str | None = None, chunk: int = 131_072, query_batch: int = 512) -> None:
    import torch
    from scipy.spatial import cKDTree

    from ..models.pinpoint_reranker import PinpointReranker
    from .pinpoint_set import NVME_INDEX
    from .strategy_search import MP16_EMBED

    photos = json.loads((Path("artifacts/photo_sets") / photo_set / "photos.json").read_text(encoding="utf-8"))
    query_rows = np.asarray([e["row"] for e in photos], dtype=np.int64)
    truth = np.asarray([e["truth"] for e in photos], dtype=np.float64)
    mp16_author, scene = _mp16_meta(query_rows)
    query_author = mp16_author[query_rows]
    embeddings = np.asarray(np.memmap(MP16_EMBED / "embeddings.f16.bin", dtype=np.float16, mode="r").reshape(-1, 1536)[query_rows], dtype=np.float32)

    model = PinpointReranker(device="cuda", index_dir=Path(index_dir) if index_dir else NVME_INDEX if (NVME_INDEX / "manifest.json").exists() else None)
    index_author = mp16_author[model.cache_indices]
    latlon = np.asarray(model.latlon, dtype=np.float64)
    print(f"{len(photos)} queries; index {len(latlon):,} rows; loading it onto the GPU", flush=True)
    model.load_resident()
    res, device = model._resident, model.device
    raw, projected = model.encode_queries(embeddings)
    vectors = {"image": projected.half(), "raw_image": raw.half(), "gps": projected.half()}

    def block(m: str, idx: torch.Tensor) -> torch.Tensor:
        if m == "raw_image":
            r = res["raw"][idx].float() * res["scales"][idx, None]
            torch.nan_to_num_(r, nan=0.0, posinf=0.0, neginf=0.0)
            return r.div_(r.norm(dim=1, keepdim=True).clamp_(min=1e-8)).half()
        return res[m][idx]

    # best score per query, modality and variant over the rows that qualify (-inf: none; its rank is set to the index size)
    tree = cKDTree(_xyz(latlon))
    q_xyz = _xyz(truth)
    near = {km: tree.query_ball_point(q_xyz, 2 * np.sin(km / EARTH_KM / 2)) for km in RADII_KM}
    own_rows = {}
    order = np.argsort(index_author, kind="stable")
    sorted_author = index_author[order]
    for a in np.unique(query_author):
        lo, hi = np.searchsorted(sorted_author, [a, a + 1])
        own_rows[int(a)] = order[lo:hi]
    best = np.full((len(photos), len(MODALITIES), len(VARIANTS)), -np.inf, dtype=np.float32)
    count_near = np.zeros((len(photos), 2), np.int64)
    count_own = np.zeros((len(photos), 3), np.int64)  # own photos in the index: anywhere, within 1 km, within 25 km
    with torch.inference_mode():
        for q in range(len(photos)):
            mine = own_rows[int(query_author[q])]
            sets = {}
            for km, key in zip(RADII_KM, ("r1", "r25")):
                rows = np.asarray(near[km][q], dtype=np.int64)
                other = rows[index_author[rows] != query_author[q]]
                sets[key], sets[key + "_other"] = rows, other
                count_near[q, RADII_KM.index(km)] = len(rows)
                count_own[q, 1 + RADII_KM.index(km)] = len(rows) - len(other)
            sets["own"] = mine
            count_own[q, 0] = len(mine)
            for v, key in enumerate(VARIANTS):
                rows = sets[key]
                if not len(rows):
                    continue
                for m_i, m in enumerate(MODALITIES):
                    top = -np.inf
                    for s in range(0, len(rows), 65_536):
                        idx = torch.as_tensor(rows[s : s + 65_536], device=device)
                        top = max(top, float((block(m, idx) @ vectors[m][q]).max()))
                    best[q, m_i, v] = top
            if q % 500 == 0:
                print(f"  best right scores {q}/{len(photos)}", flush=True)

    # one pass: rows strictly above each threshold (the "_other" variants count only rows not by the query's photographer)
    rank = np.zeros_like(best, dtype=np.int64)
    thr = torch.as_tensor(best, device=device).half()  # compared in fp16, as the scores
    q_author = torch.as_tensor(query_author, device=device)
    other_v = torch.as_tensor([v.endswith("_other") for v in VARIANTS], device=device)
    count = len(latlon)
    counts = torch.zeros(best.shape, dtype=torch.long, device=device)
    with torch.inference_mode():
        for start in range(0, count, chunk):
            idx = torch.arange(start, min(start + chunk, count), device=device)
            authors = torch.as_tensor(index_author[start : start + chunk], device=device)
            for m_i, m in enumerate(MODALITIES):
                b = block(m, idx)
                for qs in range(0, len(photos), query_batch):
                    scores = vectors[m][qs : qs + query_batch] @ b.T  # [B, chunk] fp16
                    mine = authors.unsqueeze(0) == q_author[qs : qs + query_batch].unsqueeze(1)
                    for v in range(len(VARIANTS)):
                        above = scores > thr[qs : qs + query_batch, m_i, v : v + 1]
                        if other_v[v]:
                            above &= ~mine
                        counts[qs : qs + query_batch, m_i, v] += above.sum(1)
            if (start // chunk) % 8 == 0:
                print(f"  index {min(start + chunk, count):,}/{count:,}", flush=True)
    rank = counts.cpu().numpy()
    rank[~np.isfinite(best)] = count
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"recall_{photo_set}_{tag}.npz"
    np.savez(dst, rank=rank, best=best, count_near=count_near, count_own=count_own, scene=scene, truth=truth.astype(np.float32),
             image_id=np.asarray([e["image_id"] for e in photos]), modalities=np.asarray(MODALITIES), variants=np.asarray(VARIANTS))
    print(f"-> {dst}", flush=True)


def report(photo_set: str, tag: str) -> None:
    z = np.load(OUT / f"recall_{photo_set}_{tag}.npz")
    rank, count_near, count_own, scene = z["rank"], z["count_near"], z["count_own"], z["scene"]
    v = {k: i for i, k in enumerate(VARIANTS)}
    lines: list[str] = []

    def show(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    def union(r, ks):  # r [Q, 3] ranks of one variant
        return (r < np.asarray(ks)[None]).any(1)

    own_pool = union(rank[:, :, v["own"]], (32, 8, 8))
    dens = count_near[:, 1]
    labels = np.asarray(["indoor", "natural", "urban"])[np.argmax(scene, 1)]
    slices = [("all", np.ones(len(rank), bool)), ("own photo in 32/8/8 pool", own_pool), ("no own photo in pool", ~own_pool),
              ("photographer has no index photo", count_own[:, 0] == 0), ("own index photo within 25 km", count_own[:, 2] > 0)]
    slices += [(f"index photos within 25 km {lo}-{hi}", (dens >= lo) & (dens < hi)) for lo, hi in ((0, 100), (100, 1000), (1000, 10000), (10000, 10 ** 9))]
    slices += [(f"scene {k}", labels == k) for k in ("indoor", "natural", "urban")]
    ks = (1, 2, 8, 32, 128, 512, 1000, 4096)
    unions = (("8/2/2", (8, 2, 2)), ("32/8/8", (32, 8, 8)), ("64/64/64", (64,) * 3), ("256/256/256", (256,) * 3), ("1000/1000/1000", (1000,) * 3))
    show(f"{photo_set} ({len(rank)} photos), retriever {tag}, original protocol; % of photos with a right index photo in the top K")
    for variant, label in (("", "own photos kept"), ("_other", "own photos removed")):
        for km in ("1", "25"):
            key = f"r{km}{variant}"
            show(f"\n== within {km} km, {label}: per modality recall@K for K = {' / '.join(map(str, ks))}; union pools; median / p75 / p90 rank")
            r = rank[:, :, v[key]]
            for lab, m in slices:
                if m.sum() < 30:
                    continue
                cells = []
                for mi, name in enumerate(MODALITIES):
                    cells.append(f"{name} " + " ".join(f"{100 * np.mean(r[m, mi] < k):4.1f}" for k in ks))
                show(f"  {lab:34s} {100 * m.mean():5.1f}% of photos (n {m.sum()}), {np.mean(count_near[m, int(km == '25')] > 0) * 100:.1f}% have a right index photo")
                for c in cells:
                    show(f"      {c}")
                show("      union " + "  ".join(f"{u} {100 * union(r[m], k).mean():4.1f}" for u, k in unions) +
                     f"  best-modality rank median / p75 / p90 " + " / ".join(f"{np.quantile(r[m].min(1), p):.0f}" for p in (0.5, 0.75, 0.9)))
                show("      rank median / p75 / p90: " + "; ".join(f"{name} " + " / ".join(f"{np.quantile(r[m, mi], p):.0f}" for p in (0.5, 0.75, 0.9))
                                                         for mi, name in enumerate(MODALITIES)))
    # which modality finds the right photo first, within 25 km, own kept
    r = rank[:, :, v["r25"]]
    first = np.argmin(r, 1)
    show("\n== within 25 km, own kept: the modality with the best rank: " + ", ".join(f"{n} {100 * np.mean(first == i):.1f}%" for i, n in enumerate(MODALITIES)))
    for name_i, name in enumerate(MODALITIES):
        alone = (r[:, name_i] < 32) & (np.delete(r, name_i, 1) >= 32).all(1)
        show(f"  found in top 32 only by {name}: {100 * alone.mean():.1f}%")
    (OUT / f"recall_{photo_set}_{tag}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare(photo_set: str, tags: list[str]) -> None:
    """Headline recall per retriever: image search @8 / 32 / 128, union 32 / 8 / 8 and 256 x 3, at 25 and 1 km; all photos and no own photo
    in the (released) 32 / 8 / 8 pool, so every tag is sliced on the same photos."""

    v = {k: i for i, k in enumerate(VARIANTS)}
    base = np.load(OUT / f"recall_{photo_set}_{tags[0]}.npz")["rank"]
    no_own = ~(base[:, :, v["own"]] < np.asarray((32, 8, 8))[None]).any(1)
    print(f"{photo_set}: % with a right photo; slice 'no own' = no own photo in {tags[0]}'s 32 / 8 / 8 pool ({100 * no_own.mean():.1f}%)")
    print(f"{'tag':14s} {'km':>3s} {'slice':7s} | image@8 @32 @128 | raw@32 | gps@32 | 32/8/8 | 256x3")
    for km in ("25", "1"):
        for tag in tags:
            r = np.load(OUT / f"recall_{photo_set}_{tag}.npz")["rank"][:, :, v[f"r{km}"]]
            for label, m in (("all", np.ones(len(r), bool)), ("no own", no_own)):
                cell = lambda mi, k: 100 * np.mean(r[m, mi] < k)
                print(f"{tag:14s} {km:>3s} {label:7s} | {cell(0, 8):5.1f} {cell(0, 32):5.1f} {cell(0, 128):5.1f} | {cell(1, 32):5.1f} | {cell(2, 32):5.1f} | "
                      f"{100 * (r[m] < np.asarray((32, 8, 8))[None]).any(1).mean():5.1f} | {100 * (r[m] < 256).any(1).mean():5.1f}")


def top1(photo_set: str, tag: str, targets: list[int], index_dir: str | None = None) -> None:
    """The reranker's own pick (PINPOINT_RERANKER_CKPT, default released) over a pool of `targets` (image, raw, GPS) per photo, original
    protocol: % within 1 / 25 km and the pool oracle. Saved to artifacts/retrieval/top1_{photo_set}_{tag}.npz."""

    from ..models.pinpoint_reranker import PinpointReranker
    from .pinpoint_set import NVME_INDEX
    from .strategy_search import MP16_EMBED

    photos = json.loads((Path("artifacts/photo_sets") / photo_set / "photos.json").read_text(encoding="utf-8"))
    rows = np.asarray([e["row"] for e in photos], dtype=np.int64)
    truth = np.asarray([e["truth"] for e in photos], dtype=np.float64)
    emb = np.asarray(np.memmap(MP16_EMBED / "embeddings.f16.bin", dtype=np.float16, mode="r").reshape(-1, 1536)[rows], dtype=np.float32)
    model = PinpointReranker(device="cuda", index_dir=Path(index_dir) if index_dir else NVME_INDEX if (NVME_INDEX / "manifest.json").exists() else None)
    model.widen(targets)
    found = model.rank(emb)

    def km(ll):
        a, b = np.radians(truth), np.radians(ll)
        h = np.sin((b[..., 0] - a[:, None, 0]) / 2) ** 2 + np.cos(a[:, None, 0]) * np.cos(b[..., 0]) * np.sin((b[..., 1] - a[:, None, 1]) / 2) ** 2
        return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

    d = np.where(found.valid, km(found.latlon), np.inf)
    lines = [f"{photo_set}, {tag}, pool {'/'.join(map(str, targets))} ({found.valid.sum(1).mean():.1f} candidates): top-1 < 1 / 25 km "
             f"{100 * np.mean(d[:, 0] < 1):.1f} / {100 * np.mean(d[:, 0] < 25):.1f}; pool oracle {100 * np.mean(d.min(1) < 1):.1f} / "
             f"{100 * np.mean(d.min(1) < 25):.1f}"]
    print(lines[0], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"top1_{photo_set}_{tag}.npz", rows=found.rows, latlon=found.latlon.astype(np.float32), scores=found.scores.astype(np.float32),
             truth=truth.astype(np.float32))
    with open(OUT / "top1.txt", "a", encoding="utf-8") as fh:
        fh.write(lines[0] + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("ranks", "report"):
        s = sub.add_parser(name)
        s.add_argument("--photo-set", default="dev3k")
        s.add_argument("--tag", default="released")
    sub.choices["ranks"].add_argument("--index-dir", help="a retrained retriever's index (with PINPOINT_CONTRASTIVE_CKPT set to its checkpoint)")
    c = sub.add_parser("compare")
    c.add_argument("--photo-set", default="dev3k")
    c.add_argument("tags", nargs="+")
    t = sub.add_parser("top1")
    t.add_argument("--photo-set", default="dev3k")
    t.add_argument("--tag", required=True)
    t.add_argument("--targets", type=int, nargs=3, default=[32, 8, 8])
    t.add_argument("--index-dir")
    a = p.parse_args(argv)
    if a.cmd == "top1":
        top1(a.photo_set, a.tag, a.targets, a.index_dir)
    elif a.cmd == "compare":
        compare(a.photo_set, a.tags)
    else:
        ranks(a.photo_set, a.tag, a.index_dir) if a.cmd == "ranks" else report(a.photo_set, a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
