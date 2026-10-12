# Geolocation as hypothesis testing (HYPOTHESIS_PLAN.md), E3: a learned set-level comparator over a photo's hypotheses (Pinpoint areas, each
# searched with Pinpoint restricted to a ~25 km box; hypothesis_compare features). A per-hypothesis MLP with a set context (DeepSets: mean of the
# photo's hypothesis embeddings) scores the hypotheses jointly; listwise softmax loss towards the hypotheses whose restricted top-1 is < 25 km
# from the truth (GPS labels only). 5-fold CV on train (35k bucket-99 photos), then trained on all of train and applied to dev3k. dev sets only.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.hypothesis_learn --ablate photographers support
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.hypothesis_learn --e4 --ablate clues        (E4)

"""Features per hypothesis: log score mass at T 0.1 / 0.2 / 0.5, restricted top-1 and top-3 scores, log pool support, area rank (one-hot),
log distinct photographers, log restricted candidates, log gallery density, and each of the main ones minus the photo's maximum (relative).
Only Pinpoint-area hypotheses (source 0); clue hypotheses are E4. Reported: % < 25 km (and < 1 km) when the top-scored hypothesis's restricted
top-1 is the answer, vs Pinpoint's top-1, with the agreement reference and the E2 hand comparator."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
FEATURES = {"train": OUT / "hypothesis_features_train.npz", "dev3k": OUT / "hypothesis_features_dev3k_clues.npz"}
FEATURES_E4 = {"train": OUT / "hypothesis_features_train_e4.npz", "dev3k": OUT / "hypothesis_features_dev3k_e4.npz"}
CLUE_KEYS = [f"{kind}_{stat}" for kind in ("read", "recall") for stat in ("n", "p", "share", "rare")]
RESTRICTED_KEYS = [f"rf_{j}" for j in range(9)] + ["r_support", "r_areas", "r_spread"]
GROUPS = {"clues": [], "restricted": [], "vlm": [], "dist": [], "set": [], "mass": ["mass_0.1", "mass_0.2", "mass_0.5"], "scores": ["top1", "top3"], "support": ["support"], "rank": ["rank"],
          "photographers": ["photographers"], "candidates": ["n_cand"], "density": ["density"]}


def _matrix(f: dict, drop: Sequence[str], e4: bool = False, extra: bool = False) -> tuple[np.ndarray, list[np.ndarray]]:
    keep = np.ones(len(f["source"]), bool) if e4 else f["source"] == 0
    photo = f["photo"][keep]
    lg = lambda x: np.log(np.maximum(x, 1e-12))
    cols, names = [], []
    for g, keys in GROUPS.items():
        if g in drop:
            continue
        for k in keys:
            x = f[k][keep].astype(np.float64)
            if k == "rank":
                cols += [(x == r).astype(float) for r in range(10)]
                continue
            x = lg(x) if k not in ("top1", "top3") else np.where(np.isfinite(x), x, -1.0)
            cols.append(x)
            if k in ("mass_0.2", "top1", "support", "photographers"):
                mx = np.full(int(photo.max()) + 1, -np.inf)
                np.maximum.at(mx, photo, x)
                cols.append(x - mx[photo])
    if e4:
        cols += [(f["source"][keep] == src).astype(float) for src in (1, 2)]  # clue place, VLM place
        if "clues" not in drop:
            cols += [np.log1p(f[k][keep]) if k.endswith("_n") else f[k][keep] for k in CLUE_KEYS]
    if e4 and "dist" not in drop and "d_top1" in f:
        cols.append(np.log1p(f["d_top1"][keep]))
    if e4 and "vlm" not in drop and "vlm_p" in f and np.isfinite(f["vlm_p"]).any():
        v = f["vlm_p"][keep]
        cols += [np.where(np.isfinite(v), v, 0.5), np.isfinite(v).astype(float)]
    if e4 and "set" not in drop and "set_mass" in f:  # set-aware Pinpoint (pinpoint_set)
        cols += [np.log(np.maximum(f["set_mass"][keep], 1e-6)), f["set_top_in"][keep], np.log1p(f["d_settop"][keep])]
        if "setr_max" in f:  # the set model inside each hypothesis: best and log-sum-exp logit, raw and minus the photo's maximum
            for k in ("setr_max", "setr_lse"):
                x = f[k][keep].astype(np.float64)
                mx = np.full(int(photo.max()) + 1, -np.inf)
                np.maximum.at(mx, photo, x)
                cols += [x, x - mx[photo]]
    if extra and "restricted" not in drop:
        cols += [np.log1p(f[k][keep]) if k == "r_spread" else f[k][keep] for k in RESTRICTED_KEYS]
    X = np.stack(cols, 1)
    groups = [np.flatnonzero(photo == p) for p in range(len(f["base_km"]))]
    return X, groups


class _Model:
    """Per-hypothesis encoder + set context: the mean of the photo's hypotheses (DeepSets), or with attn one round of attention in which each
    hypothesis attends to the photo's other hypotheses."""

    attn = False

    def __init__(self, dim: int, seed: int) -> None:
        import torch

        torch.manual_seed(seed)
        h = 64
        self.enc = torch.nn.Sequential(torch.nn.Linear(dim, h), torch.nn.GELU(), torch.nn.Linear(h, h), torch.nn.GELU())
        self.qkv = torch.nn.Linear(h, 3 * h)
        self.head = torch.nn.Sequential(torch.nn.Linear(2 * h, h), torch.nn.GELU(), torch.nn.Linear(h, 1))

    def parameters(self):
        return list(self.enc.parameters()) + list(self.head.parameters()) + (list(self.qkv.parameters()) if self.attn else [])

    def scores(self, x, seg, n_seg):
        import torch

        e = self.enc(x)
        if not self.attn:
            ctx = (torch.zeros(n_seg, e.shape[1]).index_add_(0, seg, e) / torch.bincount(seg, minlength=n_seg).clamp(min=1).unsqueeze(1))[seg]
        else:
            q, k, v = self.qkv(e).chunk(3, dim=1)
            logits = (q @ k.T) / k.shape[1] ** 0.5
            logits = logits.masked_fill(seg.unsqueeze(0) != seg.unsqueeze(1), float("-inf"))
            ctx = torch.softmax(logits, dim=1) @ v
        return self.head(torch.cat([e, ctx], 1)).squeeze(1)


def _fit(X: np.ndarray, groups: list[np.ndarray], ok: np.ndarray, photos: np.ndarray, seed: int = 0, epochs: int = 30):
    import torch

    mu, sd = X.mean(0), X.std(0) + 1e-6
    usable = [p for p in photos if len(groups[p]) and ok[groups[p]].any() and not ok[groups[p]].all()]
    model = _Model(X.shape[1], seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for b in np.array_split(rng.permutation(usable), max(1, len(usable) // 256)):
            rows = np.concatenate([groups[p] for p in b])
            seg = np.concatenate([np.full(len(groups[p]), j) for j, p in enumerate(b)])
            x = torch.as_tensor((X[rows] - mu) / sd, dtype=torch.float32)
            segt = torch.as_tensor(seg)
            s = model.scores(x, segt, len(b))
            target = torch.as_tensor(ok[rows], dtype=torch.float32)
            m = torch.full((len(b),), -1e9).scatter_reduce(0, segt, s, "amax")
            lse = torch.log(torch.zeros(len(b)).index_add_(0, segt, torch.exp(s - m[segt]))) + m
            pos = torch.log(torch.zeros(len(b)).index_add_(0, segt, torch.exp(s - m[segt]) * target)) + m
            loss = (lse - pos).mean()  # -log P(a right hypothesis)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, mu, sd


def _predict(model, mu, sd, X, groups, photos) -> dict[int, int]:
    """Top-scored hypothesis per photo. model may be a list of models (an ensemble: their per-photo softmax probabilities are averaged)."""

    import torch

    models = model if isinstance(model, list) else [model]
    out = {}
    with torch.no_grad():
        for b in np.array_split(np.asarray(photos), max(1, len(photos) // 2048)):
            b = [p for p in b if len(groups[p])]
            if not b:
                continue
            rows = np.concatenate([groups[p] for p in b])
            seg = np.concatenate([np.full(len(groups[p]), j) for j, p in enumerate(b)])
            x = torch.as_tensor((X[rows] - mu) / sd, dtype=torch.float32)
            segt = torch.as_tensor(seg)
            s = np.zeros(len(rows))
            for m in models:
                raw = m.scores(x, segt, len(b))
                mx = torch.full((len(b),), -1e9).scatter_reduce(0, segt, raw, "amax")
                ex = torch.exp(raw - mx[segt])
                s += (ex / torch.zeros(len(b)).index_add_(0, segt, ex)[segt]).numpy()
            start = 0
            for p in b:
                n = len(groups[p])
                out[p] = int(groups[p][int(np.argmax(s[start:start + n]))])
                start += n
    return out


def _evaluate(name: str, f: dict, groups, picks: dict[int, int], e4: bool = False, agree: bool | str = False) -> None:
    """agree: False = the chosen hypothesis's restricted top-1; True = its agreement pick; "set" = set-aware Pinpoint's top pick when it is
    inside the chosen hypothesis, else the agreement pick."""

    from .strategy_search import _bootstrap

    keep = np.arange(len(f["source"])) if e4 else np.flatnonzero(f["source"] == 0)
    if agree == "setr":
        km = f["km_setr"][keep]
    elif agree == "set":
        km = np.where(f["set_top_in"] > 0, f["set_km"][f["photo"]], f["km_agree"])[keep]
    else:
        km = (f["km_agree"] if agree else f["km"])[keep]
    base = f["base_km"]
    chosen25 = np.asarray([km[picks[p]] < 25 if p in picks else base[p] < 25 for p in range(len(base))], float)
    chosen1 = np.asarray([km[picks[p]] < 1 if p in picks else base[p] < 1 for p in range(len(base))], float)
    c, c1 = _bootstrap(chosen25 - (base < 25)), _bootstrap(chosen1 - (base < 1))
    print(f"  {name:58s} < 25 km {chosen25.mean():.1%} ({100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]); "
          f"fixes {int(((chosen25 == 1) & (base >= 25)).sum())}, breaks {int(((chosen25 == 0) & (base < 25)).sum())}; < 1 km {100 * c1[0]:+.1f}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drop", nargs="*", default=(), help="feature groups left out of every run (e.g. vlm)")
    parser.add_argument("--ablate", nargs="*", default=(), help="feature groups to drop in extra runs: " + ", ".join(GROUPS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--e4", action="store_true", help="Pinpoint areas + clue hypotheses + clue features (ablation group: clues)")
    parser.add_argument("--extra", action="store_true", help="+ restricted-retrieval features (ablation group: restricted); needs --e4")
    parser.add_argument("--attn", action="store_true", help="attention between a photo's hypotheses instead of the mean")
    parser.add_argument("--agree", action="store_true", help="answer with the agreement pick inside the chosen hypothesis")
    parser.add_argument("--ensemble", type=int, default=1, help="models per fit, softmax-averaged")
    parser.add_argument("--extra-train", action="store_true", help="add the 35k MP16 bucket 0-98 photos (train_extra) to every training fit")
    parser.add_argument("--target-agree", action="store_true", help="train towards hypotheses whose agreement pick (not top-1) is < 25 km")
    parser.add_argument("--save", type=Path, help="save the comparator(s) trained on all of train (seed 0)")
    parser.add_argument("--dump", type=Path, help="save per-photo dev3k distances (seed 0) for analysis")
    parser.add_argument("--variant", default="", help="feature-file suffix after _e4 (e.g. more10, vlmhyps, more10_vlmhyps)")
    args = parser.parse_args(argv)
    _Model.attn = args.attn
    paths = ({k: v.with_name(v.stem + f"_{args.variant}.npz") for k, v in FEATURES_E4.items()} if args.variant else FEATURES_E4) if args.e4 else FEATURES
    tr, dv = dict(np.load(paths["train"])), dict(np.load(paths["dev3k"]))
    print(f"train: {len(tr['base_km'])} photos, Pinpoint < 25 km {np.mean(tr['base_km'] < 25):.1%} (agreement +1.0, hand comparator +1.6); "
          f"dev3k: Pinpoint {np.mean(dv['base_km'] < 25):.1%} (agreement +2.0, hand comparator +2.2)")
    for ablated in [()] + [(g,) for g in args.ablate]:
        drop = tuple(args.drop) + ablated
        Xt, gt = _matrix(tr, drop, args.e4, args.extra)
        Xd, gd = _matrix(dv, drop, args.e4, args.extra)
        okt = (tr["km_agree" if args.target_agree else "km"] if args.e4 else tr["km"][tr["source"] == 0]) < 25
        n_train = len(gt)
        if args.extra_train:  # appended after train's photos; used for fitting only, never evaluated
            ex = dict(np.load(FEATURES_E4["train"].with_name("hypothesis_features_train_extra_e4" + (f"_{args.variant}" if args.variant else "") + ".npz")))
            Xe, ge = _matrix(ex, drop, args.e4, args.extra)
            if Xe.shape[1] < Xt.shape[1]:  # train has VLM-plausibility columns the extra photos lack: pad as "missing"
                Xe = np.concatenate([Xe, np.tile([0.5, 0.0], (len(Xe), 1))[:, :Xt.shape[1] - Xe.shape[1]]], 1)
            off = len(Xt)
            Xt = np.concatenate([Xt, Xe])
            gt = gt + [g + off for g in ge]
            okt = np.concatenate([okt, ex["km"] < 25])
            print(f"  + {len(ge)} extra training photos (buckets 0-98)")
        label = "all features" if not ablated else f"without {ablated[0]}"
        # 5-fold CV on train (held-out photos), seed-averaged by voting the first seed's picks (seeds vary the init only)
        folds = np.concatenate([np.random.default_rng(0).permutation(n_train) % 5, np.full(len(gt) - n_train, -1)])
        cv = {}
        for k in range(5):
            fit, held = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            fits = [_fit(Xt, gt, okt, fit, seed=10 * k + j) for j in range(args.ensemble)]
            cv |= _predict([m for m, _, _ in fits], fits[0][1], fits[0][2], Xt, gt, held)
        print(f" {label}")
        _evaluate("train, 5-fold cross-validation", tr, gt, cv, args.e4)
        if args.agree:
            _evaluate("train, 5-fold CV, agreement pick inside the hypothesis", tr, gt, cv, args.e4, agree=True)
            if "set_km" in tr:
                _evaluate("train, 5-fold CV, set-model pick inside the hypothesis", tr, gt, cv, args.e4, agree="set")
            if "km_setr" in tr:
                _evaluate("train, 5-fold CV, set model run inside the hypothesis", tr, gt, cv, args.e4, agree="setr")
        for seed in range(args.seeds if not ablated else 1):
            fits = [_fit(Xt, gt, okt, np.arange(len(gt)), seed=100 * seed + j) for j in range(args.ensemble)]  # train (+ extra)
            model, mu, sd = [m for m, _, _ in fits], fits[0][1], fits[0][2]
            if args.save and seed == 0 and not ablated:
                import torch

                torch.save({"models": [{"enc": m.enc.state_dict(), "qkv": m.qkv.state_dict(), "head": m.head.state_dict()} for m in model],
                            "mu": mu, "sd": sd, "dim": Xt.shape[1], "attn": _Model.attn, "args": vars(args)}, args.save)
                print(f"  saved {len(model)} comparator(s) -> {args.save}")
            picks = _predict(model, mu, sd, Xd, gd, np.arange(len(gd)))
            if args.dump and seed == 0 and not ablated:
                keep = np.arange(len(dv["source"])) if args.e4 else np.flatnonzero(dv["source"] == 0)
                km = np.asarray([dv["km"][keep][picks[p]] if p in picks else dv["base_km"][p] for p in range(len(gd))])
                kma = np.asarray([dv["km_agree"][keep][picks[p]] if p in picks and "km_agree" in dv else km[p] for p in range(len(gd))])
                np.savez(args.dump, km=km, km_agree=kma, base_km=dv["base_km"])
                print(f"  per-photo dev3k km -> {args.dump}")
            _evaluate(f"dev3k, trained on all of train (seed {seed})", dv, gd, picks, args.e4)
            if args.agree:
                _evaluate(f"dev3k, agreement pick inside (seed {seed})", dv, gd, picks, args.e4, agree=True)
                if "set_km" in dv:
                    _evaluate(f"dev3k, set-model pick inside (seed {seed})", dv, gd, picks, args.e4, agree="set")
                if "km_setr" in dv:
                    _evaluate(f"dev3k, set model run inside (seed {seed})", dv, gd, picks, args.e4, agree="setr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
