# GLDv2 landmark matches as answer-head columns (lesson R10): per area, the best GLDv2 photo match (MegaLoc, full index) within 25 km of the
# area's centre, the log count of the query's top-2,048 hits there, and whether there are any. Visible in every state (combined_head._with_cols).
# Usage: PYTHONPATH=src .venv/bin/python scripts/gldv2_head.py [--seeds 3] [--cut 0.9] [--text]
#   dev3k: the lesson-111 agent trained on MP16 train, base vs + GLDv2 columns (needs gldv2_index search --set train / dev3k)
#   commons26 dev: uploader folds as combined_head cv (weight 3), +commons vs +commons + GLDv2 columns (needs --set commons26dev)
#   --cut: hits more similar than this are dropped as near-duplicates (Flickr photos copied to Commons can be the query itself)
import argparse
import collections
import math
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import gldv2_pilot as p  # noqa: E402
from geo_search_env.experiment import action_agent as aa  # noqa: E402
from geo_search_env.experiment import combined_head as ch  # noqa: E402

COLS = ["gldv2 best within 25 km", "gldv2 hits within 25 km (log)", "gldv2 any within 25 km", "gldv2 best within 25 km - top"]


def _lm_table() -> np.ndarray:
    ids, ll = p._landmarks()
    tab = np.full((ids.max() + 1, 2), np.nan)
    tab[ids] = ll
    return tab


def area_cols(name: str, d: dict, cut: float, tab: np.ndarray, chunk: int = 512) -> np.ndarray:
    import torch

    z = np.load(p.OUT / f"search_{name}.npz")
    S = z["photo_sim"].astype(np.float32)
    S = np.where(S > cut, -2.0, S)
    hit_ll = tab[z["photo_lm"]]  # [n, 2048, 2]
    centre = d["center"].cpu().numpy().astype(np.float64)
    mask = d["mask"].cpu().numpy()
    n, h = mask.shape
    assert len(S) == n, (name, len(S), n)
    out = np.zeros((n, h, len(COLS)), np.float32)
    top = S.max(1)
    for s in range(0, n, chunk):
        c = torch.as_tensor(centre[s : s + chunk], device="cuda")  # [b, h, 2]
        q = torch.as_tensor(hit_ll[s : s + chunk], device="cuda")  # [b, k, 2]
        la1, lo1 = torch.deg2rad(c[..., 0])[:, :, None], torch.deg2rad(c[..., 1])[:, :, None]
        la2, lo2 = torch.deg2rad(q[..., 0])[:, None, :], torch.deg2rad(q[..., 1])[:, None, :]
        a = torch.sin((la2 - la1) / 2) ** 2 + torch.cos(la1) * torch.cos(la2) * torch.sin((lo2 - lo1) / 2) ** 2
        km = 2 * 6371.0 * torch.asin(torch.sqrt(a.clamp(0, 1)))  # [b, h, k]
        near = (km < 25) & torch.isfinite(km)
        sim = torch.as_tensor(S[s : s + chunk], device="cuda")[:, None, :].expand_as(km)
        best = torch.where(near, sim, torch.full_like(sim, -1.0)).max(-1).values
        cnt = near.sum(-1).float()
        anyn = (cnt > 0).float()
        floor = torch.as_tensor(S[s : s + chunk, -1], device="cuda")[:, None].expand_as(best)  # no hit: the 2,048th similarity
        best = torch.where(anyn > 0, best, floor)
        tp = torch.as_tensor(top[s : s + chunk], device="cuda")[:, None]
        out[s : s + chunk] = torch.stack([best, torch.log1p(cnt), anyn, best - tp], -1).cpu().numpy()
    return out * mask[..., None]


def _norm(a: np.ndarray, m: np.ndarray, ref: tuple | None = None):
    mu, sd = ref if ref else (a[m].mean(0), a[m].std(0) + 1e-6)
    return (a - mu) / sd * m[..., None], (mu, sd)


def _line(name, key, vs, base):
    ok, ok1 = np.mean([v < 25 for v in vs], 0), np.mean([v < 1 for v in vs], 0)
    diff = ok - base
    per_seed = ", ".join(f"{100 * np.mean(v < 25):.1f}" for v in vs)
    return (f"   {name:24s} {key:38s} < 25 km {ok.mean():6.1%} ({100 * diff.mean():+.2f}, SE {100 * diff.std() / math.sqrt(len(diff)):.2f}; "
            f"fixes {np.sum(diff > 0):.0f}, breaks {np.sum(diff < 0):.0f}; seeds {per_seed})  < 1 km {ok1.mean():5.1%}")


def _text_cells(a, tr, cm, x_tr, x_cm, lines) -> None:
    """commons26 dev, uploader folds: lesson 140's +commons+text cell (text columns + text-place tokens, weight 3) with / without GLDv2."""

    text, photos = ch.Text(), ch._photos()
    n_tr, n_c = tr["n"], cm["n"]
    out = collections.defaultdict(dict)
    for cell in ("+commons+text", "+commons+text+gldv2"):
        tr_c, cm_c = (ch._with_cols(tr, x_tr), ch._with_cols(cm, x_cm)) if cell.endswith("gldv2") else (tr, cm)
        for k in range(5):
            cols, tokens = ch._text_inputs(ch._with_text(cm, None, None), text, text.probs((k,)), photos)
            mix = ch._concat(ch._with_text(tr_c, None, [[] for _ in range(n_tr)]), ch._with_text(cm_c, cols, tokens))
            fit_idx = np.concatenate([np.arange(n_tr), np.repeat(n_tr + np.flatnonzero(text.fold != k), a.weight)])
            held = np.flatnonzero(text.fold == k)
            for s in range(a.seeds):
                net, pols = ch._fit(mix, fit_idx, 7 + 100 * s + k)
                for key, v in ch._keep(aa.evaluate(net, pols, mix, n_tr + held, cm["base_km"][held])).items():
                    out[cell].setdefault((key, s), np.zeros(n_c))[held] = v
        print(f"commons26 {cell} done", flush=True)
    lines.append(f"\n commons26 dev (uploader folds, weight {a.weight}, text columns + tokens): released Pinpoint {np.mean(cm['base_km'] < 25):.1%}")
    agg = {cell: collections.defaultdict(list) for cell in out}
    for cell in out:
        for (key, s), v in out[cell].items():
            agg[cell][key].append(v)
    for cell in agg:
        for key, vs in agg[cell].items():
            lines.append(_line(cell, key, vs, np.mean([v < 25 for v in agg["+commons+text"][key]], 0)))
    text_ = "\n".join(lines)
    print(text_, flush=True)
    (p.OUT / f"head_gldv2_text_cut{a.cut:g}.txt").write_text(text_ + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cut", type=float, default=0.9)
    ap.add_argument("--weight", type=int, default=3)
    ap.add_argument("--text", action="store_true", help="commons26 only: lesson 140's +commons+text cell with and without the GLDv2 columns")
    a = ap.parse_args()
    aa.CKPT_TAG, aa.ONE_KM = ch.CKPT, 0.5
    tab = _lm_table()
    tr = aa._data("train")
    dv = aa._data("dev3k", stats=tr["stats"])
    cm = aa._data(ch.TAG, stats=tr["stats"])
    cm["base_km"] = aa._released_own_km(ch.TAG, ch.CKPT)
    m_tr, m_dv, m_cm = (x["mask"].cpu().numpy() for x in (tr, dv, cm))
    x_tr, ref = _norm(area_cols("train", tr, a.cut, tab), m_tr)
    x_dv, _ = _norm(area_cols("dev3k", dv, a.cut, tab), m_dv, ref)
    x_cm, _ = _norm(area_cols("commons26dev", cm, a.cut, tab), m_cm, ref)
    lines = [f"GLDv2 columns in the answer head (near-duplicate cut {a.cut}, {a.seeds} seeds; accuracy averaged over seeds)"]
    if a.text:
        return _text_cells(a, tr, cm, x_tr, x_cm, lines)
    # dev3k: MP16 train only
    res = collections.defaultdict(dict)
    for name, (tr_v, dv_v) in {"base": (tr, dv), "+gldv2": (ch._with_cols(tr, x_tr), ch._with_cols(dv, x_dv))}.items():
        for s in range(a.seeds):
            net, pols = ch._fit(tr_v, np.arange(tr["n"]), 7 + 100 * s)
            for key, v in ch._keep(aa.evaluate(net, pols, dv_v, np.arange(dv["n"]), dv["base_km"])).items():
                res[name].setdefault(key, []).append(v)
        print(f"dev3k {name} done", flush=True)
    lines.append(f"\n dev3k (pipeline trained on MP16 train): released Pinpoint {np.mean(dv['base_km'] < 25):.1%}")
    for name in res:
        for key, vs in res[name].items():
            lines.append(_line(name, key, vs, np.mean([v < 25 for v in res["base"][key]], 0)))
    # commons26 dev: uploader folds, MP16 train + the other folds x weight
    text = ch.Text()
    n_tr, n_c = tr["n"], cm["n"]
    out = collections.defaultdict(dict)
    for cell in ("+commons", "+commons+gldv2"):
        if cell == "+commons":
            mix = ch._concat(tr, cm)
        else:
            mix = ch._concat(ch._with_cols(tr, x_tr), ch._with_cols(cm, x_cm))
        for k in range(5):
            fit_idx = np.concatenate([np.arange(n_tr), np.repeat(n_tr + np.flatnonzero(text.fold != k), a.weight)])
            held = np.flatnonzero(text.fold == k)
            for s in range(a.seeds):
                net, pols = ch._fit(mix, fit_idx, 7 + 100 * s + k)
                for key, v in ch._keep(aa.evaluate(net, pols, mix, n_tr + held, cm["base_km"][held])).items():
                    out[cell].setdefault((key, s), np.zeros(n_c))[held] = v
        print(f"commons26 {cell} done", flush=True)
    lines.append(f"\n commons26 dev (uploader folds, weight {a.weight}): released Pinpoint {np.mean(cm['base_km'] < 25):.1%}")
    agg = {cell: collections.defaultdict(list) for cell in out}
    for cell in out:
        for (key, s), v in out[cell].items():
            agg[cell][key].append(v)
    for cell in agg:
        for key, vs in agg[cell].items():
            lines.append(_line(cell, key, vs, np.mean([v < 25 for v in agg["+commons"][key]], 0)))
    text_ = "\n".join(lines)
    print(text_, flush=True)
    (p.OUT / f"head_gldv2_cut{a.cut:g}.txt").write_text(text_ + "\n")


if __name__ == "__main__":
    main()
