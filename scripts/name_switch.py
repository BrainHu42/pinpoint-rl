"""Name-restricted search as a candidate generator (lesson 79): when should the pipeline's answer be replaced by the best name candidate?
Baseline: the merged pipeline's answer (all areas zoomed, gated pick; train: 5-fold out-of-fold answer heads, dev3k: trained on all of
train). Candidate: per photo, the name query (name_search.py) whose restricted top-1 has the highest reranker score. A gain classifier
(read_agent.fit_value, -1 / 0 / +1 at 25 km, out of fold on train) decides; the threshold is chosen on train CV and applied to dev3k.

  PYTHONPATH=src .venv/bin/python scripts/name_switch.py
"""

import json
from pathlib import Path

import numpy as np
import torch

from geo_search_env.experiment.hypothesis_agent import SOURCES, _paths, _sub, fit_answer, load
from geo_search_env.experiment.name_checks import photo_names
from geo_search_env.experiment.read_agent import fit_value
from geo_search_env.experiment.strategy_search import _bootstrap


def km_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(a[:, 0]), np.radians(b[:, 0])
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(b[:, 1] - a[:, 1]) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def slot_points(tag: str, n: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """[photo, slot, 2] restricted top-1 of each area token and [photo, slot] its reranker score, in hypothesis_agent.load's slot order."""

    f = np.load(_paths(tag, "setft_a")[0])
    zm = np.load(_paths(tag, "setft_a")[1])
    rows = np.flatnonzero(np.isin(f["source"], SOURCES))
    s = np.where(zm["rows"] >= 0, zm["scores"], -np.inf)
    point = np.where(np.isfinite(s.max(1))[:, None], zm["latlon"][np.arange(len(s)), s.argmax(1)], zm["center"])
    out, score = np.full((n, h, 2), np.nan), np.full((n, h), np.nan)
    seen = np.zeros(n, int)
    for r in rows:
        p = f["photo"][r]
        out[p, seen[p]] = point[r]
        score[p, seen[p]] = s[r].max() if np.isfinite(s[r].max()) else np.nan
        seen[p] += 1
    return out, score


def answers(net, d, photos):
    """km of the answer, the answer head's top probability, and the chosen slot (all areas zoomed, gated pick)."""

    km, prob, pick = [], [], []
    with torch.no_grad():
        for chunk in np.array_split(np.asarray(photos), max(1, len(photos) // 1024)):
            b = _sub(d, torch.as_tensor(chunk, device="cuda"))
            logits = net.answer_logits(b, b["mask"])
            p = logits.argmax(1)
            km.append(b["km_zoom"].gather(1, p[:, None]).squeeze(1).cpu().numpy())
            prob.append(torch.softmax(logits, 1).max(1).values.cpu().numpy())
            pick.append(p.cpu().numpy())
    return np.concatenate(km), np.concatenate(prob), np.concatenate(pick)


def candidates(tag: str, n: int):
    z = np.load(f"artifacts/agent/name_search_{tag}_setft_a.npz")
    score = np.where(np.isfinite(z["latlon"][:, 0, 0]), z["scores"][:, 0], -np.inf)
    best = np.full(n, -1)
    for i in np.argsort(score):
        best[z["photo"][i]] = i
    has = best >= 0
    b = np.maximum(best, 0)
    ll = z["latlon"][b, 0].astype(np.float64)
    truth = z["truth"].astype(np.float64)
    km = np.where(has & np.isfinite(ll[:, 0]), km_between(truth, np.nan_to_num(ll)), np.inf)
    n_q = np.bincount(z["photo"], minlength=n)
    # the 4B clue judge's P(Yes) "this name is evidence of where the photo was taken", where the name went through text search (-1: not judged)
    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_setft_a" / "photos.json").read_text(encoding="utf-8"))
    pos = {p["image_id"]: i for i, p in enumerate(photos)}
    judged: dict[tuple[int, str], float] = {}
    for r in json.loads(Path(f"artifacts/agent/clue_judge_{tag}_4b_4b.json").read_text(encoding="utf-8")):
        if r["image_id"] in pos:
            key = (pos[r["image_id"]], r["query"].strip().lower())
            judged[key] = max(judged.get(key, 0.0), r["p_yes"])
    names = photo_names(tag)
    p_yes = np.full(n, -1.0)
    for i in np.flatnonzero(best >= 0):
        q = names[i]["read"] + names[i]["recall"]
        vals = [judged[(i, x)] for x in q if (i, x) in judged]
        p_yes[i] = max(vals) if vals else -1.0
    s2 = np.where(np.isfinite(z["scores"][b, 1]), z["scores"][b, 1], z["scores"][b, 0] - 5)
    feats = np.stack([z["scores"][b, 0], z["scores"][b, 0] - s2, np.log1p(z["count"][b]), np.log1p(z["n_allowed"][b]), z["kind"][b],
                      np.log1p(n_q), p_yes], 1)
    return has & np.isfinite(ll[:, 0]), km, ll, feats


def report(name, base, cand, switch):
    km = np.where(switch, cand, base)
    for t in (25, 1):
        c = _bootstrap((km < t).astype(float) - (base < t))
        print(f"  {name}: switch {switch.mean():.1%}; at {t} km {100 * np.mean(base < t):.1f} -> {100 * np.mean(km < t):.1f} "
              f"({100 * c[0]:+.2f} [{100 * c[1]:+.2f}, {100 * c[2]:+.2f}])")


tr = load("train", "setft_a")
dv = load("dev3k", "setft_a", stats=tr["stats"])
n = tr["n"]
folds = np.random.default_rng(0).permutation(n) % 5
base_km, prob, pick = np.zeros(n), np.zeros(n), np.zeros(n, int)
for k in range(5):
    net = fit_answer(tr, np.flatnonzero(folds != k), seed=10 * k)
    held = np.flatnonzero(folds == k)
    base_km[held], prob[held], pick[held] = answers(net, tr, held)
net = fit_answer(tr, np.arange(n), seed=100)
dv_km, dv_prob, dv_pick = answers(net, dv, np.arange(dv["n"]))

rows = {}
for tag, d, bkm, pr, pk in (("train", tr, base_km, prob, pick), ("dev3k", dv, dv_km, dv_prob, dv_pick)):
    has, ckm, cll, feats = candidates(tag, d["n"])
    pts, pscore = slot_points(tag, d["n"], d["mask"].shape[1])
    chosen = pts[np.arange(d["n"]), pk]
    agree = np.log1p(km_between(np.nan_to_num(chosen), np.nan_to_num(cll)))
    chosen_score = np.nan_to_num(pscore[np.arange(d["n"]), pk])
    best_area = np.nan_to_num(np.nanmax(np.where(np.isnan(pscore), -np.inf, pscore), 1), neginf=0.0)
    gap = np.stack([feats[:, 0] - chosen_score, feats[:, 0] - best_area, chosen_score], 1)  # same reranker: name candidate vs the pipeline's
    x = np.concatenate([feats, pr[:, None], agree[:, None], gap], 1).astype(np.float32)
    gain = (ckm < 25).astype(int) - (bkm < 25).astype(int)
    rows[tag] = (has, bkm, ckm, x, gain)
    print(f"\n{tag}: {d['n']} photos, pipeline {100 * np.mean(bkm < 25):.1f}% < 25 km; with a name candidate {has.mean():.1%}; "
          f"switching would help {np.mean(has & (gain > 0)):.1%}, hurt {np.mean(has & (gain < 0)):.1%} of photos")

has, bkm, ckm, x, gain = rows["train"]
idx = np.flatnonzero(has)
value = np.full(n, -np.inf)
for k in range(5):
    fit, held = idx[folds[idx] != k], idx[folds[idx] == k]
    value[held] = fit_value(x[fit], gain[fit], seed=k)(x[held])
FEATURES = ["name score", "name margin", "log matches", "log allowed", "recalled", "log name queries", "judge P(Yes)", "answer prob",
            "log km name-chosen",
            "name - chosen score", "name - best area score", "chosen score"]


def auc(score, pos, neg):
    a, b = score[pos], score[neg]
    ranks = np.argsort(np.argsort(np.concatenate([a, b])))
    return (ranks[: len(a)].sum() - len(a) * (len(a) - 1) / 2) / (len(a) * len(b))


helps, hurts = has & (gain > 0), has & (gain < 0)
print(f"\ntrain, separating switch-helps ({helps.sum()}) from switch-hurts ({hurts.sum()}), AUC (0.5 = no signal):")
print("  " + ", ".join(f"{f} {auc(x[:, j], helps, hurts):.2f}" for j, f in enumerate(FEATURES)) + f"; learned value {auc(value, helps, hurts):.2f}")
best_t, best_gain = None, -1.0
for t in np.quantile(value[idx], np.linspace(0.5, 0.995, 60)):
    g = np.mean(np.where(value > t, ckm, bkm) < 25) - np.mean(bkm < 25)
    if g > best_gain:
        best_t, best_gain = t, g
print(f"\ntrain CV (threshold chosen here, E[gain] > {best_t:.3f}):")
report("learned", bkm, ckm, value > best_t)
report("oracle", bkm, ckm, has & (gain > 0))

has_d, bkm_d, ckm_d, x_d, gain_d = rows["dev3k"]
v = fit_value(x[idx], gain[idx], seed=100)(x_d)
v = np.where(has_d, v, -np.inf)
print("\ndev3k (models and threshold from train):")
report("learned", bkm_d, ckm_d, v > best_t)
report("oracle", bkm_d, ckm_d, has_d & (gain_d > 0))
