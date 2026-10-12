"""Zoom agent (PINPOINT_SET_PLAN.md step 5, first version): a policy trained with RL chooses which area hypotheses to zoom into under a
budget of B restricted searches; the comparator then picks among the zoomed areas and answers with its agreement pick.

Environment, from cached tool results (hypothesis_compare features; each hypothesis's restricted search already ran offline):
  - before zooming the policy sees each hypothesis's pre-zoom evidence: source (Pinpoint area / clue place / 4B place), area rank,
    pool support, score mass, photographers, clue-search statistics, distance to Pinpoint's top-1, set-layer mass;
  - zooming reveals the restricted-search features; the comparator (hypothesis_learn, out-of-fold on train) scores each zoomed
    hypothesis and the best one answers. Reward: 1 if that answer is within 25 km.
Policy: per-hypothesis MLP with the photo's mean as context (DeepSets), B picks without replacement (Gumbel top-B); REINFORCE with the
greedy picks' reward as baseline (self-critical). Compared with fixed rules at the same budget: Pinpoint's top-B areas, and the best
fixed mix of areas and clue / 4B places.

  python -m geo_search_env.experiment.zoom_agent [--budgets 1 2 3]
"""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np

from . import hypothesis_learn as hl
from .area_model import FEATURES


def _pre_zoom(f: dict) -> np.ndarray:
    lg = lambda x: np.log(np.maximum(x, 1e-12))
    photo = f["photo"]
    cols = []
    for k in ("mass_0.2", "support", "photographers", "density"):
        x = lg(f[k])
        mx = np.full(photo.max() + 1, -np.inf)
        np.maximum.at(mx, photo, x)
        cols += [x, x - mx[photo]]
    cols += [(f["rank"] == r).astype(float) for r in range(5)]
    cols += [(f["source"] == s).astype(float) for s in (0, 1, 2)]
    cols += [np.log1p(f[k]) if k.endswith("_n") else f[k] for k in hl.CLUE_KEYS]
    cols += [np.log1p(f["d_top1"]), np.log(np.maximum(f["set_mass"], 1e-6)), f["set_top_in"], np.log1p(f["d_settop"])]
    return np.nan_to_num(np.stack(cols, 1).astype(np.float32))


def _chooser_scores(f: dict, groups, folds: np.ndarray | None, fit_photos: np.ndarray | None, ensemble: int = 3) -> np.ndarray:
    """Comparator log-probability of each hypothesis given the photo's full hypothesis set (out-of-fold when folds is given)."""

    import torch

    X, g = hl._matrix(f, ("vlm",), True)
    ok = f["km"] < 25
    out = np.zeros(len(X))

    def score(models, mu, sd, photos):
        rows = np.concatenate([g[p] for p in photos if len(g[p])])
        seg = np.concatenate([np.full(len(g[p]), j) for j, p in enumerate(photos) if len(g[p])])
        x = torch.as_tensor((X[rows] - mu) / sd, dtype=torch.float32)
        segt = torch.as_tensor(seg)
        with torch.no_grad():
            s = sum(m.scores(x, segt, len(photos)) for m in models) / len(models)
        out[rows] = s.numpy()

    if folds is not None:
        for k in range(5):
            fits = [hl._fit(X, g, ok, np.flatnonzero(folds != k), seed=10 * k + j) for j in range(ensemble)]
            score([m for m, _, _ in fits], fits[0][1], fits[0][2], np.flatnonzero(folds == k))
        return out, None
    fits = [hl._fit(X, g, ok, fit_photos, seed=j) for j in range(ensemble)]
    return out, fits


def _apply_chooser(fits, f, groups) -> np.ndarray:
    import torch

    X, g = hl._matrix(f, ("vlm",), True)
    out = np.zeros(len(X))
    photos = np.arange(len(g))
    rows = np.concatenate([g[p] for p in photos if len(g[p])])
    seg = np.concatenate([np.full(len(g[p]), j) for j, p in enumerate(photos) if len(g[p])])
    mu, sd = fits[0][1], fits[0][2]
    with torch.no_grad():
        s = sum(m.scores(torch.as_tensor((X[rows] - mu) / sd, dtype=torch.float32), torch.as_tensor(seg), len(photos)) for m, _, _ in fits) / len(fits)
    out[rows] = s.numpy()
    return out


def _reward(chosen: list[np.ndarray], score: np.ndarray, ok: np.ndarray, base_ok: np.ndarray, photos) -> np.ndarray:
    return np.asarray([ok[c[np.argmax(score[c])]] if len(c) else base_ok[p] for c, p in zip(chosen, photos)], float)


class _Policy:
    def __init__(self, dim: int, seed: int) -> None:
        import torch

        torch.manual_seed(seed)
        h = 64
        self.enc = torch.nn.Sequential(torch.nn.Linear(dim, h), torch.nn.GELU(), torch.nn.Linear(h, h), torch.nn.GELU())
        self.head = torch.nn.Sequential(torch.nn.Linear(2 * h, h), torch.nn.GELU(), torch.nn.Linear(h, 1))

    def parameters(self):
        return list(self.enc.parameters()) + list(self.head.parameters())

    def logits(self, x, seg, n_seg):
        import torch

        e = self.enc(x)
        ctx = (torch.zeros(n_seg, e.shape[1]).index_add_(0, seg, e) / torch.bincount(seg, minlength=n_seg).clamp(min=1).unsqueeze(1))[seg]
        return self.head(torch.cat([e, ctx], 1)).squeeze(1)


def _pick(logits: np.ndarray, groups, photos, budget: int, rng=None) -> list[np.ndarray]:
    out = []
    for p in photos:
        r = groups[p]
        l = logits[r] + (rng.gumbel(size=len(r)) if rng is not None else 0.0)
        out.append(r[np.argsort(-l)[:budget]])
    return out


def _train_policy(P, groups, score, ok, base_ok, photos, budget, seed, epochs=15):
    import torch

    mu, sd = P.mean(0), P.std(0) + 1e-6
    pol = _Policy(P.shape[1], seed)
    opt = torch.optim.AdamW(pol.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    photos = np.asarray([p for p in photos if len(groups[p]) > budget])
    for _ in range(epochs):
        for b in np.array_split(rng.permutation(photos), max(1, len(photos) // 256)):
            n_h = max(len(groups[p]) for p in b)
            idx = np.full((len(b), n_h), -1)
            for j, p in enumerate(b):
                idx[j, :len(groups[p])] = groups[p]
            m = idx >= 0
            rows = idx[m]
            seg = np.nonzero(m)[0]
            flat = pol.logits(torch.as_tensor((P[rows] - mu) / sd), torch.as_tensor(seg), len(b))
            logits = torch.full((len(b), n_h), -1e9)
            logits[torch.as_tensor(m)] = flat
            ln = logits.detach().numpy()
            sampled_pos = np.argsort(-(ln + rng.gumbel(size=ln.shape)), 1)[:, :budget]
            greedy_pos = np.argsort(-ln, 1)[:, :budget]
            take = lambda pos: [idx[j, pos[j]] for j in range(len(b))]
            adv = _reward(take(sampled_pos), score, ok, base_ok, b) - _reward(take(greedy_pos), score, ok, base_ok, b)
            # log-probability of the sampled ordered picks (Plackett-Luce)
            avail = torch.as_tensor(m).clone()
            logp = torch.zeros(len(b))
            for k in range(budget):
                c = torch.as_tensor(sampled_pos[:, k])
                logp = logp + logits.gather(1, c[:, None]).squeeze(1) - torch.logsumexp(logits.masked_fill(~avail, -1e9), 1)
                avail = avail.scatter(1, c[:, None], False)
            loss = -(torch.as_tensor(adv, dtype=torch.float32) * logp).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return pol, mu, sd


def _policy_logits(pol, mu, sd, P, groups) -> np.ndarray:
    import torch

    photos = [p for p in range(len(groups)) if len(groups[p])]
    rows = np.concatenate([groups[p] for p in photos])
    seg = np.concatenate([np.full(len(groups[p]), j) for j, p in enumerate(photos)])
    out = np.zeros(len(P))
    with torch.no_grad():
        out[rows] = pol.logits(torch.as_tensor((P[rows] - mu) / sd), torch.as_tensor(seg), len(photos)).numpy()
    return out


def _fixed_pick(priority: np.ndarray, f: dict, groups, photos, budget: int, n_places: int | None = None) -> list[np.ndarray]:
    out = []
    for p in photos:
        r = groups[p]
        if n_places is None:
            out.append(r[np.argsort(-priority[r])[:budget]])
            continue
        places = r[f["source"][r] > 0]
        places = places[np.argsort(-priority[places])][:n_places] if n_places else places[:0]
        areas = r[f["source"][r] == 0]
        areas = areas[np.argsort(f["rank"][areas])][: budget - len(places)]
        out.append(np.concatenate([areas, places]))
    return out


def run(budgets: Sequence[int], seeds: int = 3) -> None:
    from .strategy_search import _bootstrap

    tr, dv = dict(np.load(FEATURES["train"])), dict(np.load(FEATURES["dev3k"]))
    data = {}
    for tag, f in (("train", tr), ("dev3k", dv)):
        groups = [np.flatnonzero(f["photo"] == p) for p in range(len(f["base_km"]))]
        data[tag] = (f, groups, _pre_zoom(f), f["km_agree"] < 25, f["base_km"] < 25)
    n = len(data["train"][1])
    folds = np.random.default_rng(0).permutation(n) % 5
    print("comparator scores (train out-of-fold, dev3k from all of train)", flush=True)
    s_tr, _ = _chooser_scores(tr, data["train"][1], folds, None)
    _, fits = _chooser_scores(tr, data["train"][1], None, np.arange(n))
    s_dv = _apply_chooser(fits, dv, data["dev3k"][1])
    scores = {"train": s_tr, "dev3k": s_dv}
    print(f"hypotheses per photo: train {len(tr['photo']) / n:.2f}, dev3k {len(dv['photo']) / len(data['dev3k'][1]):.2f}")
    for budget in budgets:
        print(f"\n== budget {budget} zoom(s)", flush=True)
        results = {}
        for tag in ("train", "dev3k"):
            f, groups, P, ok, base_ok = data[tag]
            photos = np.arange(len(groups))
            res = {"all hypotheses (no budget)": _reward([g for g in groups], scores[tag], ok, base_ok, photos)}
            res[f"Pinpoint top-{budget} areas"] = _reward(_fixed_pick(None, f, groups, photos, budget, 0), scores[tag], ok, base_ok, photos)
            for n_places in range(1, budget + 1):
                prio = np.log(np.maximum(f["support"], 1e-12)) + (f["source"] == 1)
                res[f"{budget - n_places} areas + {n_places} places"] = _reward(_fixed_pick(prio, f, groups, photos, budget, n_places), scores[tag], ok, base_ok, photos)
            res["oracle (any hypothesis right)"] = np.asarray([ok[g].any() if len(g) else base_ok[p] for g, p in zip(groups, photos)], float)
            results[tag] = res
        # RL policy: train CV (policy per fold on the out-of-fold comparator scores) and dev3k (policy from all of train)
        f, groups, P, ok, base_ok = data["train"]
        cv_logits = np.zeros(len(P))
        for k in range(5):
            pols = [_train_policy(P, groups, s_tr, ok, base_ok, np.flatnonzero(folds != k), budget, seed=10 * k + j) for j in range(seeds)]
            held = [g for p, g in enumerate(groups) if folds[p] == k]
            lg = sum(_policy_logits(pl, mu, sd, P, groups) for pl, mu, sd in pols) / seeds
            rows = np.concatenate([g for g in held if len(g)])
            cv_logits[rows] = lg[rows]
        results["train"]["RL zoom policy"] = _reward(_pick(cv_logits, groups, np.arange(len(groups)), budget), s_tr, ok, base_ok, np.arange(len(groups)))
        pols = [_train_policy(P, groups, s_tr, ok, base_ok, np.arange(len(groups)), budget, seed=100 + j) for j in range(seeds)]
        fd, gd, Pd, okd, based = data["dev3k"]
        lg = sum(_policy_logits(pl, mu, sd, Pd, gd) for pl, mu, sd in pols) / seeds
        results["dev3k"]["RL zoom policy"] = _reward(_pick(lg, gd, np.arange(len(gd)), budget), s_dv, okd, based, np.arange(len(gd)))
        for tag in ("train", "dev3k"):
            base = data[tag][4].astype(float)
            ref = results[tag][f"Pinpoint top-{budget} areas"]
            print(f" {tag} (Pinpoint {base.mean():.1%} < 25 km)")
            for name, r in results[tag].items():
                d = _bootstrap(r - base)
                dr = _bootstrap(r - ref)
                print(f"   {name:30s} {r.mean():.1%} ({100 * d[0]:+.1f} vs Pinpoint; {100 * dr[0]:+.1f} [{100 * dr[1]:+.1f}, {100 * dr[2]:+.1f}] vs top-{budget} areas)", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budgets", type=int, nargs="*", default=(1, 2, 3))
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args(argv)
    run(args.budgets, args.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
