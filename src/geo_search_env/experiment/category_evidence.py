# Do the kinds of places near a candidate (lighthouse, stadium, beach...) match what the photo shows? Overture place categories as map evidence.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.category_evidence build    (CPU, ~10 min: category counts near every pool candidate)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.category_evidence photo    (GPU, SigLIP2 zero-shot: P(category visible) per photo)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.category_evidence report

"""Map side: Overture places (release 2026-09-23.1, /data/pinpoint/overture/places.sqlite) of VISUAL categories, counted within 1 and 5 km of every
pooled candidate of the dev / val photos. Photo side: SigLIP2 giant zero-shot, P(visible) = sigmoid(scale * cos(photo, "a photo of <text>") + bias),
from the cached photo embeddings.

report (category weights: per-photo softmax over the SigLIP2 logits, since its sigmoid P is ~0 everywhere), on each photo's top-8 candidates (the comparator's set; placeholders dropped):
- match(candidate) = sum over categories of P(visible) * idf * [a place of that category within R km], idf = log(candidates / candidates with one).
- screen: on "choosing" photos (a candidate < 25 km from the truth in the top 8, reranker top-1 wrong), how often the best right candidate's match
  beats the wrong top-1's (ties reported apart), and the within-photo AUC of match for right vs wrong candidates on all photos.
- gate: a cross-validated combiner (5 folds, 3 seeds) of rank + comparator + category features against rank + comparator (lesson 33).
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from typing import Any, Sequence

import numpy as np

from .query_evidence import ROOT
from .stage1_eval import _bootstrap
from .strategy_search import EARTH_KM, _xyz

OUT = ROOT / "category"
DB = "/data/pinpoint/overture/places.sqlite"
RADII_KM = (1.0, 5.0)
TOPK = 8
VISUAL = {  # Overture category -> what a photo of it shows
    "christian_place_of_worship": "a church", "muslim_place_of_worship": "a mosque", "hindu_place_of_worship": "a hindu temple",
    "buddhist_place_of_worship": "a buddhist temple", "jewish_place_of_worship": "a synagogue", "historic_site": "a historic building",
    "castle": "a castle", "fort": "a fortress", "monument": "a monument", "sculpture_statue": "a statue", "public_fountain": "a fountain",
    "museum": "a museum", "art_gallery": "an art gallery", "library": "a library", "stadium_arena": "a stadium", "sport_field": "a sports field",
    "golf_course": "a golf course", "swimming_pool": "a swimming pool", "skating_rink": "an ice rink", "skate_park": "a skate park",
    "amusement_park": "an amusement park", "zoo": "a zoo", "aquarium": "an aquarium", "park": "a park", "garden": "a garden",
    "playground": "a playground", "public_plaza": "a town square", "market": "a market", "farmers_market": "a farmers market",
    "shopping_mall": "a shopping mall", "beach": "a beach", "lake": "a lake", "river": "a river", "mountain": "a mountain",
    "waterfall": "a waterfall", "forest": "a forest", "island": "an island", "canyon": "a canyon", "hot_springs": "hot springs",
    "national_park": "a national park", "nature_reserve": "a nature reserve", "campground": "a campsite", "marina": "a marina with boats",
    "pier": "a pier", "lighthouse": "a lighthouse", "bridge": "a bridge", "canal": "a canal", "dam": "a dam", "airport": "an airport",
    "train_station": "a train station", "farm": "a farm", "winery": "a vineyard", "cemetery": "a cemetery", "college_university": "a university campus",
    "theatre_venue": "a theatre", "music_venue": "a concert", "casino": "a casino", "resort": "a resort", "hotel": "a hotel",
    "military_site": "a military site", "street_art": "street art", "scenic_viewpoint": "a scenic viewpoint",
}
CATS = list(VISUAL)


def _photos() -> list[dict[str, Any]]:
    excluded = set(json.loads((ROOT / "val" / "exclude.json").read_text(encoding="utf-8")))
    return [dict(e, tag=tag) for tag in ("dev", "val") for e in json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8")) if e["image_id"] not in excluded]


def build() -> None:
    from scipy.spatial import cKDTree

    photos = _photos()
    points = np.asarray([c for e in photos for c in e["pool"]], dtype=np.float64)
    tree = cKDTree(_xyz(points))
    counts = np.zeros((len(RADII_KM), len(points), len(CATS)), dtype=np.int32)
    chords = [2 * math.sin(r / EARTH_KM / 2) for r in RADII_KM]
    index = {c: i for i, c in enumerate(CATS)}
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cursor = db.execute(f"SELECT lat, lon, cat FROM places WHERE cat IN ({','.join('?' * len(CATS))})", CATS)
    done = 0
    while rows := cursor.fetchmany(1_000_000):
        latlon = np.asarray([(r[0], r[1]) for r in rows], dtype=np.float64)
        cat = np.asarray([index[r[2]] for r in rows])
        xyz = _xyz(latlon)
        for k, chord in enumerate(chords):
            for place, near in enumerate(tree.query_ball_point(xyz, chord, workers=-1)):
                if near:
                    np.add.at(counts[k], (np.asarray(near), cat[place]), 1)
        done += len(rows)
        print(f"{done} places", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "counts.npz", counts=counts, offsets=np.cumsum([0] + [len(e["pool"]) for e in photos]))
    totals = db.execute(f"SELECT cat, count(*) FROM places WHERE cat IN ({','.join('?' * len(CATS))}) GROUP BY cat", CATS).fetchall()
    (OUT / "totals.json").write_text(json.dumps(dict(totals), indent=2) + "\n", encoding="utf-8")


def photo() -> None:
    import torch
    import torch.nn.functional as F

    from .query_headroom import _features, _model
    from .strategy_search import load_world

    photos = _photos()
    world = load_world()  # benchmark queries: "index" of val photos; dev photos' "row" is their MP16 embedding row
    images = np.stack([np.asarray(world.mp16["embeddings"][e["row"]], dtype=np.float32) if e["tag"] == "dev" else world.query_embeddings[e["index"]] for e in photos])
    model, processor = _model()
    tokens = processor(text=[f"a photo of {VISUAL[c]}" for c in CATS], return_tensors="pt", padding="max_length", max_length=64, truncation=True).to("cuda")
    with torch.inference_mode():
        text = F.normalize(_features(model.get_text_features(**tokens)).float(), dim=-1)
        cos = F.normalize(torch.as_tensor(images, device="cuda"), dim=-1) @ text.T
        p = torch.sigmoid(cos * model.logit_scale.exp().float() + model.logit_bias.float()).cpu().numpy()
    np.save(OUT / "p_visible.npy", p)
    top = p.mean(0).argsort()[::-1]
    print("mean P(visible), most common: " + ", ".join(f"{CATS[i]} {p[:, i].mean():.2f}" for i in top[:12]))
    print("share of photos with P > 0.5 per category: " + ", ".join(f"{CATS[i]} {(p[:, i] > 0.5).mean():.1%}" for i in top[:12]))


def _cv(D: np.ndarray, F: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Top-1 within 25 km per photo of a 5-fold, 3-seed cross-validated listwise combiner over the features F (as in exemplar_judge.topk_cv)."""

    import torch

    reward = np.where(valid, (D < 25) * 1.0 + (D < 1) * 0.5, 0.0).astype(np.float32)
    mean, std = F[valid].mean(0), F[valid].std(0) + 1e-6
    X, M, R = torch.as_tensor((F - mean) / std), torch.as_tensor(valid), torch.as_tensor(reward)
    folds = np.array_split(np.random.default_rng(0).permutation(len(D)), 5)
    hits = np.zeros(len(D))
    for seed in range(3):
        for f in folds:
            torch.manual_seed(seed)
            net = torch.nn.Sequential(torch.nn.Linear(F.shape[-1], 16), torch.nn.GELU(), torch.nn.Linear(16, 1))
            opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-2)
            idx = torch.as_tensor(np.setdiff1d(np.arange(len(D)), f))
            for _ in range(300):
                loss = -(torch.softmax(net(X[idx]).squeeze(-1).masked_fill(~M[idx], float("-inf")), -1) * R[idx]).sum(-1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pick = net(X[f]).squeeze(-1).masked_fill(~M[f], float("-inf")).argmax(1).numpy()
            hits[f] += (D[f, pick] < 25) / 3
    return hits


def report() -> None:
    from .exemplar_judge import _load_topk, _logit

    photos = _photos()
    judged = _load_topk("comparator-a")
    assert len(judged) == len(photos)
    saved = np.load(OUT / "counts.npz")
    counts, offsets = saved["counts"], saved["offsets"]
    p = np.load(OUT / "p_visible.npy").astype(np.float64)
    logit = np.log(p) - np.log1p(-p)  # SigLIP2's sigmoid puts nearly every P below 0.01; its per-photo ranking is sensible, so use a softmax
    p_vis = np.exp(logit - logit.max(1, keepdims=True))
    p_vis /= p_vis.sum(1, keepdims=True)
    present = counts > 0
    idf = np.log(present.shape[1] / (present.sum(1) + 1.0))  # [radius, category]
    tags = np.asarray([e["tag"] for e in photos])

    D, base_f, cat_f = [], [], []
    match = {r: [] for r in RADII_KM}
    for i, (e, j) in enumerate(zip(photos, judged)):
        n = len(j["dist"])
        rows = slice(offsets[i], offsets[i] + n)
        lg = _logit(j["p"])
        rank = np.arange(n, dtype=float)
        order = (-lg).argsort().argsort().astype(float)
        f = np.stack([rank, (rank == 0).astype(float), (rank < 3).astype(float), lg, lg * rank, lg - lg.max(), order, (order == 0).astype(float), lg * (rank == 0)], axis=1)
        feats = []
        for k, r in enumerate(RADII_KM):
            m = (present[k, rows] * p_vis[i] * idf[k]).sum(1)
            match[r].append(m)
            feats += [m, m - m.max(), m - m[0], (m == m.max()).astype(float)]
        # contrast: categories the photo shows that only this candidate has nearby (among the top 8)
        unique = present[0, rows] & (present[0, rows].sum(0, keepdims=True) == 1)
        feats.append((unique * p_vis[i] * idf[0]).sum(1))
        D.append(np.pad(j["dist"], (0, TOPK - n), constant_values=1e5))
        base_f.append(np.pad(f, ((0, TOPK - n), (0, 0))))
        cat_f.append(np.pad(np.stack(feats, axis=1), ((0, TOPK - n), (0, 0))))
    D, base_f, cat_f = np.asarray(D), np.asarray(base_f, dtype=np.float32), np.asarray(cat_f, dtype=np.float32)
    valid = D < 9e4

    out: dict[str, Any] = {}
    print(f"{len(D)} photos (dev {int((tags == 'dev').sum())}, val {int((tags == 'val').sum())}), top-{TOPK} candidates, P(visible) from SigLIP2 zero-shot")
    for r in RADII_KM:
        wins = ties = losses = 0
        aucs = []
        for d, m in zip(D, match[r]):
            d = d[: len(m)]
            right, wrong = m[d < 25], m[d >= 25]
            if len(right) and len(wrong):
                a, b = right[:, None], wrong[None, :]
                aucs.append((a > b).mean() + 0.5 * (a == b).mean())
            if d[0] >= 25 and (d < 25).any():
                best = m[d < 25].max()
                wins += best > m[0]; ties += best == m[0]; losses += best < m[0]
        n_choose = wins + ties + losses
        out[f"{r:g} km"] = {"choosing photos": int(n_choose), "right beats wrong top-1": int(wins), "tie": int(ties), "wrong top-1 beats right": int(losses), "within-photo AUC": float(np.mean(aucs))}
        # the mechanism of lesson 21: is the photo's most likely category near the wrong top-1 as often as near the right candidate?
        k = RADII_KM.index(r)
        near_right = near_wrong = 0
        for i, (d, m) in enumerate(zip(D, match[r])):
            d = d[: len(m)]
            if d[0] >= 25 and (d < 25).any():
                c = int(p_vis[i].argmax())
                near_wrong += present[k, offsets[i], c]
                near_right += present[k, offsets[i] + int(np.flatnonzero(d < 25)[0]), c]
        out[f"{r:g} km"]["photo's top category near the right / wrong top-1"] = [int(near_right), int(near_wrong)]
        print(f"  radius {r:g} km: the photo's most likely category lies near the best-ranked right candidate in {near_right} and near the wrong top-1 in {near_wrong} of them")
        print(f"  radius {r:g} km: choosing photos {n_choose}: right candidate's match beats the wrong top-1's {wins} ({wins / n_choose:.0%}), tie {ties} ({ties / n_choose:.0%}), "
              f"loses {losses} ({losses / n_choose:.0%}); within-photo AUC right vs wrong {np.mean(aucs):.3f} (n={len(aucs)} photos)")

    base = (D[:, 0] < 25).astype(float)
    shuffled = cat_f[np.random.default_rng(0).permutation(len(cat_f))]
    shuffled = np.where(valid[..., None], shuffled, 0.0)  # shuffled photos may have fewer candidates
    configs = {
        "rank + comparator": base_f,
        "rank + categories": np.concatenate((base_f[..., :3], cat_f), -1),
        "rank + comparator + categories": np.concatenate((base_f, cat_f), -1),
        "rank + comparator + shuffled categories": np.concatenate((base_f, shuffled), -1),
    }
    results = {}
    for name, F in configs.items():
        h = _cv(D, F, valid)
        results[name] = h
        ci = _bootstrap(h - base)
        out[name] = {"top-1 <25 km": float(h.mean()), "vs reranker": ci}
        print(f"  {name:42s} top-1 <25 km {h.mean():.1%}  vs reranker {ci[0]:+.1%} [{ci[1]:+.1%}, {ci[2]:+.1%}]  "
              f"dev {h[tags == 'dev'].mean() - base[tags == 'dev'].mean():+.1%}  val {h[tags == 'val'].mean() - base[tags == 'val'].mean():+.1%}")
    ci = _bootstrap(results["rank + comparator + categories"] - results["rank + comparator"])
    out["categories beyond rank + comparator"] = ci
    print(f"  reranker top-1 <25 km {base.mean():.1%}; categories beyond rank + comparator {ci[0]:+.1%} [{ci[1]:+.1%}, {ci[2]:+.1%}]")
    (OUT / "report.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("build", "photo", "report"))
    args = parser.parse_args(argv)
    {"build": build, "photo": photo, "report": report}[args.node]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
