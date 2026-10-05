# Near-miss photos: the reranker's top-1 is within 25 km of the truth but not within 1 km. How many are there, can the gallery place them within 1 km, and do
# simple local refinement rules of the top-1 coordinate (no model) get them there?
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.near_miss ceiling   (CPU, a few min)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.near_miss rules     (CPU)

"""ceiling: on the 3,713 benchmark eval-half photos (placeholders dropped) and the 1,000 MP16 dev photos, the share of near-miss photos that have a gallery photo
         (MP16 not by the query's photographer, OSV) within 1 km of the truth, and that has it among the cached top neighbours of the query.
rules:   replace the top-1 coordinate by a location read from the cached neighbours within RADIUS_KM of it: the most similar photo's location, or the mode of a
         similarity-weighted kernel density. Settings are chosen on the MP16 dev photos; the benchmark photos are only scored. Reported as the share within 1 km
         overall, among near-misses and among photos already within 1 km (how many a rule breaks)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .query_evidence import ROOT
from .strategy_search import MP16_EMBED, OSV_EMBED, _haversine_km, _memmap_gallery

RADIUS_KM = 25.0
BENCH_CACHE, SFT_CACHE = Path("artifacts/strategy_search"), Path("artifacts/sft")
EARTH_KM = 6371.0088


def _top1_photos(tag: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Photos of a set (placeholders dropped), the reranker's top-1 coordinate and the truth, both [n, 2]."""

    photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
    if tag == "full":
        placeholder = {p["image_id"] for p in json.loads((ROOT / "multi_exemplar_full_pairs.json").read_text(encoding="utf-8")) if p["placeholder"]}
        photos = [e for e in photos if e["image_id"] not in placeholder]
    return photos, np.asarray([e["pool"][0] for e in photos]), np.asarray([e["truth"] for e in photos])


def _neighbour_points(cache, q: int, mp16: np.ndarray, osv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Locations and cosine similarities of query q's cached raw neighbours (MP16 + OSV), most similar first."""

    pts = np.concatenate((mp16[cache["mp16_raw_idx"][q]], osv[cache["osv_raw_idx"][q]]))
    sims = np.concatenate((cache["mp16_raw_sim"][q], cache["osv_raw_sim"][q]))
    order = np.argsort(-sims)
    return pts[order], sims[order]


def _kernel_mode(points: np.ndarray, sims: np.ndarray, temperature: float, bandwidth_km: float) -> np.ndarray:
    """The neighbour location with the highest similarity-weighted kernel density (weights exp((sim - max) / temperature))."""

    w = np.exp((sims - sims.max()) / temperature)
    d = np.stack([_haversine_km(*p, points) for p in points])
    density = (np.exp(-0.5 * (d / bandwidth_km) ** 2) * w[None, :]).sum(1)
    return points[int(np.argmax(density))]


def ceiling() -> None:
    from scipy.spatial import cKDTree

    from .strategy_search import _xyz, load_world

    world = load_world()
    mp16, osv = world.mp16["latlon"], world.osv["latlon"]
    chord = 2 * np.sin(1.0 / EARTH_KM / 2)
    tree_mp16, tree_osv = cKDTree(_xyz(mp16)), cKDTree(_xyz(osv))
    index_of = {q["image_id"]: i for i, q in enumerate(world.queries)}
    cache = np.load(BENCH_CACHE / "neighbors.npz")
    cache = {k: cache[k] for k in ("mp16_raw_idx", "mp16_raw_sim", "osv_raw_idx", "osv_raw_sim", "mp16_gps_idx")}
    photos, top, truth = _top1_photos("full")
    d_top = np.asarray([_haversine_km(*truth[i], top[i][None, :])[0] for i in range(len(photos))])
    near_miss, exact = (d_top < RADIUS_KM) & (d_top >= 1), d_top < 1
    print(f"{len(photos)} benchmark photos: top-1 within 1 km {exact.mean():.1%}, near-miss (1-25 km) {near_miss.mean():.1%} (n={int(near_miss.sum())}), "
          f"beyond 25 km {(d_top >= RADIUS_KM).mean():.1%}")
    in_gallery, in_neighbours, in_gps = [], [], []
    for i, e in enumerate(photos):
        q = index_of[e["image_id"]]
        author = int(world.query_author[q])
        near_rows = [r for r in tree_mp16.query_ball_point(_xyz(truth[i]), chord) if world.mp16["author"][r] != author]
        in_gallery.append(bool(near_rows) or bool(tree_osv.query_ball_point(_xyz(truth[i]), chord)))
        pts, _ = _neighbour_points(cache, q, mp16, osv)
        in_neighbours.append(bool((_haversine_km(*truth[i], pts) < 1).any()))
        in_gps.append(bool((_haversine_km(*truth[i], mp16[cache["mp16_gps_idx"][q]]) < 1).any()))
    in_gallery, in_neighbours, in_gps = map(np.asarray, (in_gallery, in_neighbours, in_gps))
    print("\nshare of photos with a gallery photo within 1 km of the truth / within 1 km among the cached top neighbours (raw MP16 + OSV, 2,000) / in Pinpoint's top-500 GPS rows:")
    for label, sel in (("all", np.ones(len(photos), bool)), ("near-miss", near_miss), ("already < 1 km", exact), ("top-1 beyond 25 km", d_top >= RADIUS_KM)):
        print(f"  {label:20s} n={int(sel.sum()):5d}  gallery {in_gallery[sel].mean():.1%}   cached neighbours {in_neighbours[sel].mean():.1%}   GPS rows {in_gps[sel].mean():.1%}")


KERNELS = [(t, h) for t in (0.01, 0.03, 0.1) for h in (0.3, 1.0)]
CANDIDATE_NEIGHBOURS = 200  # nearest-by-similarity neighbours inside the radius that a rule looks at


def _refine(top: np.ndarray, pts: np.ndarray, sims: np.ndarray) -> dict[str, np.ndarray]:
    """Refined coordinate per rule for one photo, from the neighbours within RADIUS_KM of its top-1 coordinate."""

    inside = _haversine_km(*top, pts) < RADIUS_KM
    out = {"top-1 (baseline)": top}
    if not inside.any():
        return out | {"most similar within 25 km": top} | {f"kernel mode T={t} h={h}": top for t, h in KERNELS}
    p, s = pts[inside][:CANDIDATE_NEIGHBOURS], sims[inside][:CANDIDATE_NEIGHBOURS]
    out["most similar within 25 km"] = p[0]
    for t, h in KERNELS:
        out[f"kernel mode T={t} h={h}"] = _kernel_mode(p, s, t, h)
    return out


def rules() -> None:
    mp16, osv = _memmap_gallery(MP16_EMBED)["latlon"], _memmap_gallery(OSV_EMBED)["latlon"]
    results: dict[str, dict[str, dict[str, float]]] = {}
    for tag, cache_root in (("dev", SFT_CACHE), ("full", BENCH_CACHE)):
        photos, top, truth = _top1_photos(tag)
        saved = np.load(cache_root / "neighbors.npz")
        cache = {k: saved[k] for k in ("mp16_raw_idx", "mp16_raw_sim", "osv_raw_idx", "osv_raw_sim")}
        refined: dict[str, list[np.ndarray]] = {}
        for i, e in enumerate(photos):
            pts, sims = _neighbour_points(cache, e["index"], mp16, osv)
            for rule, point in _refine(top[i], pts, sims).items():
                refined.setdefault(rule, []).append(point)
        d_top = np.asarray([_haversine_km(*truth[i], top[i][None, :])[0] for i in range(len(photos))])
        near_miss, exact = (d_top < RADIUS_KM) & (d_top >= 1), d_top < 1
        print(f"\n{tag}: {len(photos)} photos; top-1 < 1 km {exact.mean():.1%}, near-miss {near_miss.mean():.1%} (n={int(near_miss.sum())})")
        print(f"  {'rule':32s} {'< 1 km':>8s} {'< 25 km':>8s} {'fixes a near-miss':>18s} {'breaks an exact':>16s}")
        results[tag] = {}
        for rule, points in refined.items():
            d = np.asarray([_haversine_km(*truth[i], np.asarray(points[i])[None, :])[0] for i in range(len(photos))])
            results[tag][rule] = {"<1 km": float((d < 1).mean()), "<25 km": float((d < 25).mean()), "fixed": float((near_miss & (d < 1)).sum()), "broken": float((exact & (d >= 1)).sum())}
            r = results[tag][rule]
            print(f"  {rule:32s} {r['<1 km']:8.1%} {r['<25 km']:8.1%} {int(r['fixed']):10d} of {int(near_miss.sum())} {int(r['broken']):9d} of {int(exact.sum())}")
    chosen = max((r for r in results["dev"] if r != "top-1 (baseline)"), key=lambda r: results["dev"][r]["<1 km"])
    print(f"\nchosen on dev (best < 1 km): {chosen}: benchmark photos < 1 km {results['full'][chosen]['<1 km']:.1%} vs baseline {results['full']['top-1 (baseline)']['<1 km']:.1%}")
    (ROOT / "near_miss_rules.json").write_text(json.dumps({"chosen": chosen, "results": results}, indent=2) + "\n", encoding="utf-8")


MAX_NEIGHBOURS, MAX_CANDIDATES = 300, 24
FEATURES = ["top-1", "log n<1km", "log n<5km", "w<0.5km", "w<1km", "w<3km", "best sim gap", "log first rank", "log km to top-1", "osv share", "log gps<1km", "gps sim gap"]
LOCAL_CACHE = Path("artifacts/near_miss")


def _km_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from .strategy_search import _xyz

    return EARTH_KM * np.arccos(np.clip(_xyz(a) @ _xyz(b).T, -1.0, 1.0))


def _local_candidates(top: np.ndarray, mp16: np.ndarray, osv: np.ndarray, cache, q: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Candidate coordinates [c, 2] (the top-1 first, then 1 km clusters of the neighbours within RADIUS_KM of it) and their features [c, F]."""

    pts = np.concatenate((mp16[cache["mp16_raw_idx"][q]], osv[cache["osv_raw_idx"][q]]))
    sims = np.concatenate((cache["mp16_raw_sim"][q], cache["osv_raw_sim"][q]))
    src = np.concatenate((np.zeros(len(cache["mp16_raw_idx"][q])), np.ones(len(cache["osv_raw_idx"][q]))))
    order = np.argsort(-sims)
    pts, sims, src = pts[order], sims[order], src[order]
    to_top = _km_matrix(top[None], pts)[0]
    inside = np.flatnonzero(to_top < RADIUS_KM)[:MAX_NEIGHBOURS]
    if not len(inside):
        return None
    p, s, o, dt = pts[inside], sims[inside], src[inside], to_top[inside]
    seeds: list[int] = []
    pairwise = _km_matrix(p, p)
    for i in range(len(p)):
        if dt[i] >= 1.0 and all(pairwise[i, j] >= 1.0 for j in seeds):
            seeds.append(i)
            if len(seeds) == MAX_CANDIDATES - 1:
                break
    coords = np.concatenate((top[None], p[seeds]))
    dm = _km_matrix(coords, p)  # [c, neighbours]
    w = np.exp((s - s.max()) / 0.03)
    w /= w.sum()
    gps_pts, gps_sims = mp16[cache["mp16_gps_idx"][q]], cache["mp16_gps_sim"][q]
    gm = _km_matrix(coords, gps_pts)
    f = np.zeros((len(coords), len(FEATURES)), dtype=np.float32)
    f[0, 0] = 1.0
    within1, within5 = dm < 1.0, dm < 5.0
    f[:, 1], f[:, 2] = np.log1p(within1.sum(1)), np.log1p(within5.sum(1))
    f[:, 3], f[:, 4], f[:, 5] = (w * (dm < 0.5)).sum(1), (w * within1).sum(1), (w * (dm < 3.0)).sum(1)
    best = np.where(within1, s[None], -np.inf).max(1)
    f[:, 6] = np.where(np.isfinite(best), best - s.max(), s.min() - s.max())
    f[:, 7] = np.log1p(np.where(within1.any(1), within1.argmax(1), len(p)))
    f[:, 8] = np.log1p(_km_matrix(coords, top[None])[:, 0])
    f[:, 9] = np.where(within1.any(1), (within1 * o[None]).sum(1) / np.maximum(within1.sum(1), 1), 0.0)
    gw = gm < 1.0
    f[:, 10] = np.log1p(gw.sum(1))
    gbest = np.where(gw, gps_sims[None], -np.inf).max(1)
    f[:, 11] = np.where(np.isfinite(gbest), gbest - gps_sims.max(), gps_sims.min() - gps_sims.max())
    return coords, f


def _local_set(tag: str) -> dict[str, np.ndarray]:
    """Candidates, features and truth distances per photo of a set ('train' = bucket-99 train photos of the SFT pool, 'dev', 'full'), cached."""

    path = LOCAL_CACHE / f"{tag}.npz"
    if path.exists():
        return dict(np.load(path))
    mp16, osv = _memmap_gallery(MP16_EMBED)["latlon"], _memmap_gallery(OSV_EMBED)["latlon"]
    if tag == "train":
        queries = json.loads((SFT_CACHE / "queries.json").read_text(encoding="utf-8"))
        saved = np.load(SFT_CACHE / "candidates.npz")
        pool_coords, pool_ranking = saved["coords"], saved["ranking"]  # NpzFile re-reads an array on every key access: load once
        latlon = np.fromfile(MP16_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2).astype(np.float64)
        idx = [i for i, q in enumerate(queries) if q["group"] == "held_out" and q["split"] == "train"]
        top = np.asarray([pool_coords[i, pool_ranking[i, 0]] for i in idx])
        truth = latlon[[queries[i]["row"] for i in idx]]
        cache_root = SFT_CACHE
    else:
        photos, top, truth = _top1_photos(tag)
        idx = [e["index"] for e in photos]
        cache_root = SFT_CACHE if tag == "dev" else BENCH_CACHE
    saved_n = np.load(cache_root / "neighbors.npz")
    cache = {k: saved_n[k] for k in ("mp16_raw_idx", "mp16_raw_sim", "osv_raw_idx", "osv_raw_sim", "mp16_gps_idx", "mp16_gps_sim")}
    n = len(idx)
    coords, feats, dist = np.zeros((n, MAX_CANDIDATES, 2)), np.zeros((n, MAX_CANDIDATES, len(FEATURES)), np.float32), np.full((n, MAX_CANDIDATES), 1e5)
    for k, q in enumerate(idx):
        out = _local_candidates(top[k], mp16, osv, cache, q)
        if out is None:  # no neighbour near the top-1: the top-1 is the only candidate
            out = (top[k][None], np.zeros((1, len(FEATURES)), np.float32))
            out[1][0, 0] = 1.0
        c, f = out
        coords[k, : len(c)], feats[k, : len(c)] = c, f
        dist[k, : len(c)] = _km_matrix(truth[k][None], c)[0]
        if (k + 1) % 5000 == 0:
            print(f"  {tag} {k + 1}/{n}", flush=True)
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, coords=coords, feats=feats, dist=dist, top=top, truth=truth)
    return dict(np.load(path))


def _fit_listwise(F: np.ndarray, D: np.ndarray, valid: np.ndarray, seed: int, steps: int = 400):
    import torch

    reward = np.where(valid, (D < 1.0) * 1.0 + (D < 5.0) * 0.25, 0.0).astype(np.float32)
    mean, std = F[valid].mean(0), F[valid].std(0) + 1e-6
    X, M, R = torch.as_tensor((F - mean) / std), torch.as_tensor(valid), torch.as_tensor(reward)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(F.shape[-1], 32), torch.nn.GELU(), torch.nn.Linear(32, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(steps):
        loss = -(torch.softmax(net(X).squeeze(-1).masked_fill(~M, float("-inf")), -1) * R).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def score(F_new: np.ndarray, valid_new: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return net(torch.as_tensor((F_new - mean) / std)).squeeze(-1).masked_fill(~torch.as_tensor(valid_new), float("-inf")).numpy()

    return score


def local_rerank() -> None:
    sets = {tag: _local_set(tag) for tag in ("train", "dev", "full")}
    valid = {t: s["dist"] < 9e4 for t, s in sets.items()}
    scorers = [_fit_listwise(sets["train"]["feats"], sets["train"]["dist"], valid["train"], seed) for seed in range(3)]
    print(f"train photos {len(sets['train']['dist'])}; mean candidates {valid['train'].sum(1).mean():.1f}; "
          f"a candidate within 1 km of the truth exists for {(sets['train']['dist'].min(1) < 1).mean():.1%} (top-1 alone {(sets['train']['dist'][:, 0] < 1).mean():.1%})")
    report: dict[str, dict] = {}
    for tag in ("dev", "full"):
        d, v = sets[tag]["dist"], valid[tag]
        score = sum(s(sets[tag]["feats"], v) for s in scorers)
        pick = np.argmax(score, axis=1)
        chosen = d[np.arange(len(d)), pick]
        base = d[:, 0]
        near_miss, exact = (base < RADIUS_KM) & (base >= 1), base < 1
        row = {"n": int(len(d)), "top-1 < 1 km": float((base < 1).mean()), "local rerank < 1 km": float((chosen < 1).mean()), "top-1 < 25 km": float((base < 25).mean()),
               "local rerank < 25 km": float((chosen < 25).mean()), "oracle over local candidates < 1 km": float((d.min(1) < 1).mean()),
               "fixed near-misses": int((near_miss & (chosen < 1)).sum()), "near-misses": int(near_miss.sum()), "broken exact": int((exact & (chosen >= 1)).sum()), "exact": int(exact.sum())}
        from .stage1_eval import _bootstrap

        row["change < 1 km [95% CI]"] = list(_bootstrap((chosen < 1).astype(float) - (base < 1)))
        report[tag] = row
        print(f"\n{tag} (n={row['n']}): top-1 < 1 km {row['top-1 < 1 km']:.1%} -> local rerank {row['local rerank < 1 km']:.1%}  change {100 * row['change < 1 km [95% CI]'][0]:+.1f} "
              f"[{100 * row['change < 1 km [95% CI]'][1]:+.1f}, {100 * row['change < 1 km [95% CI]'][2]:+.1f}]; < 25 km {row['top-1 < 25 km']:.1%} -> {row['local rerank < 25 km']:.1%}; "
              f"oracle over local candidates < 1 km {row['oracle over local candidates < 1 km']:.1%}; fixes {row['fixed near-misses']} of {row['near-misses']} near-misses, "
              f"breaks {row['broken exact']} of {row['exact']} exact")
    (ROOT / "near_miss_local_rerank.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def exemplar_pairs(tag: str = "dev") -> None:
    """One exemplar (a gallery photo within 1 km of the candidate, not by the query's photographer, most similar to the query) per local candidate of every
    photo of `tag`, in the pairs format that `multi_exemplar judge` reads (tag `nearmiss_<tag>`)."""

    import math

    from scipy.spatial import cKDTree

    from .multi_exemplar import _exemplars, _pairs_path
    from .strategy_search import EARTH_KM as R, _xyz, load_world

    local = _local_set(tag)
    photos, _, _ = _top1_photos(tag) if tag != "train" else (None, None, None)
    world = load_world(mp16_queries=[{"row": e["row"], "image_id": e["image_id"]} for e in photos]) if tag == "dev" else load_world()
    ids = (MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines()
    tree = cKDTree(_xyz(world.mp16["latlon"]))
    chord = 2 * math.sin(1.0 / R / 2)
    out = []
    for m, e in enumerate(photos):
        pos = m if tag == "dev" else e["index"]
        q = world.query_embeddings[pos] / np.linalg.norm(world.query_embeddings[pos])
        author = int(world.query_author[pos])
        found = [_exemplars(world, tree, ids, chord, q, author, c)[:1] if local["dist"][m, k] < 9e4 else [] for k, c in enumerate(local["coords"][m])]
        out.append({"image_id": e["image_id"], "path": e.get("path"), "exemplars": found})
    _pairs_path(f"nearmiss_{tag}").write_text(json.dumps(out) + "\n", encoding="utf-8")
    counts = [len(x) for p in out for x in p["exemplars"]]
    print(f"{len(out)} photos, {sum(c > 0 for c in counts)} local candidates with an exemplar of {int((local['dist'] < 9e4).sum())}")


def exemplar_report(name: str, tag: str = "dev") -> None:
    """How well a comparator's score on the local candidates' exemplars separates a candidate within 1 km of the truth from one 1-25 km away (all within the
    photo), whether its top local candidate beats the top-1, and a cross-validated listwise combiner with the 12 local features."""

    from .multi_exemplar import _logit, _scores_path
    from .stage1_eval import _bootstrap

    local = _local_set(tag)
    scored = json.loads(_scores_path(name, f"nearmiss_{tag}").read_text(encoding="utf-8"))
    d, feats = local["dist"], local["feats"]
    valid = d < 9e4
    logit = np.zeros(d.shape)
    has = np.zeros(d.shape, bool)
    for m, p in enumerate(scored):
        for k, xs in enumerate(p["p_same"]):
            if xs and xs[0] is not None:
                logit[m, k], has[m, k] = _logit(np.asarray([xs[0]]))[0], True
    pos, neg = (d < 1.0) & valid & has, (d >= 1.0) & (d < RADIUS_KM) & valid & has
    wins = ties = pairs = 0
    for m in range(len(d)):
        a, b = logit[m][pos[m]], logit[m][neg[m]]
        if len(a) and len(b):
            wins += int((a[:, None] > b[None, :]).sum()); ties += int((a[:, None] == b[None, :]).sum()); pairs += len(a) * len(b)
    print(f"{name} on {tag}: {int(has.sum())} scored candidates; within-photo AUC (< 1 km vs 1-25 km), {pairs} pairs: {(wins + 0.5 * ties) / max(pairs, 1):.3f}")
    base = d[:, 0]
    near_miss, exact = (base < RADIUS_KM) & (base >= 1), base < 1
    pick = np.argmax(np.where(valid & has, logit, -np.inf), axis=1)
    chosen = d[np.arange(len(d)), pick]
    print(f"  comparator's own top local candidate: < 1 km {(chosen < 1).mean():.1%} vs top-1 {(base < 1).mean():.1%}; fixes {int((near_miss & (chosen < 1)).sum())} of "
          f"{int(near_miss.sum())} near-misses, breaks {int((exact & (chosen >= 1)).sum())} of {int(exact.sum())} exact")
    # cross-validated listwise combiner over the local features + the comparator logit (5 folds, 3 seeds)
    F = np.concatenate((feats, logit[..., None].astype(np.float32), (logit - np.where(valid & has, logit, -np.inf).max(1, keepdims=True))[..., None].astype(np.float32)), axis=-1)
    folds = np.array_split(np.random.default_rng(0).permutation(len(d)), 5)
    for label, cols in (("local features", list(range(feats.shape[-1]))), ("local features + comparator", list(range(F.shape[-1])))):
        hit = np.zeros(len(d))
        for seed in range(3):
            for f in folds:
                train = np.setdiff1d(np.arange(len(d)), f)
                score = _fit_listwise(F[train][..., cols], d[train], valid[train], seed)
                p = np.argmax(score(F[f][..., cols], valid[f]), axis=1)
                hit[f] += (d[f, p] < 1) / 3
        ci = _bootstrap(hit - (base < 1))
        print(f"  CV combiner, {label:28s}: < 1 km {hit.mean():.1%} vs top-1 {(base < 1).mean():.1%}, change {100 * ci[0]:+.1f} [{100 * ci[1]:+.1f}, {100 * ci[2]:+.1f}]")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("ceiling", "rules", "local-rerank", "exemplar-pairs", "exemplar-report"))
    parser.add_argument("--tag", default="dev")
    parser.add_argument("--name", default="comparator-b")
    args = parser.parse_args(argv)
    if args.node == "exemplar-pairs":
        exemplar_pairs(args.tag)
    elif args.node == "exemplar-report":
        exemplar_report(args.name, args.tag)
    else:
        {"ceiling": ceiling, "rules": rules, "local-rerank": local_rerank}[args.node]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
