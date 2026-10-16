# MegaLoc with more photos per area (the user's go-ahead after lesson 96: "verification is the gap"). Lesson 93's MegaLoc columns see only the
# zoom file's 12 restricted-search photos per area; lesson 95 found that where the same place is in the index, Pinpoint's areas already hold it.
# Gate (dev3k): per area the N = 64 index photos within 25 km of its centre nearest the query in raw SigLIP2 (photographer excluded), MegaLoc on
# them (reusing lesson 93's and 95's embeddings), then
#   A. area choice: right vs wrong area AUC and the conditional logit (megaloc_gate's) with 12 vs 12 + 64 photos per area;
#   B. the point inside an area: MegaLoc's best photo vs the restricted search's top-1 at 1 / 25 km (lesson 91's local chooser was at chance).
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.megaloc_area shortlist --tag dev3k
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.megaloc_area embed --tag dev3k
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.megaloc_area report --tag dev3k

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .refine_headroom import _hav

OUT = Path("artifacts/agent")
CKPT_TAG = "setft_a"
N = 64
RADIUS_KM = 25.0


def _paths(tag: str):
    return OUT / f"megaloc_area_{tag}_{CKPT_TAG}.npz", OUT / f"megaloc_area_emb_{tag}_{CKPT_TAG}.npy"


def _zoom(tag: str):
    return np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{CKPT_TAG}.npz")


def shortlist(tag: str) -> None:
    import torch

    from ..tools.pinpoint import Pinpoint

    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
    zm = _zoom(tag)
    tool = Pinpoint()
    rr = tool.reranker
    q_author = tool.mp16_author(np.asarray([e["row"] for e in photos]))
    A = tool.index_author()
    views = np.load(OUT / f"crops_{tag}_{CKPT_TAG}.npy", mmap_mode="r")
    q, _ = rr.encode_queries(np.asarray(views[:, 0], np.float32))
    q = q.half()
    H = len(zm["photo"])
    rows = np.full((H, N), -1, np.int64)
    sims = np.full((H, N), -np.inf, np.float32)
    n_near = np.zeros(H, np.int64)
    with torch.inference_mode():
        for h in range(H):
            p = int(zm["photo"][h])
            near = tool.rows_near(zm["center"][h][None], RADIUS_KM)
            near = near[A[near] != q_author[p]]
            n_near[h] = len(near)
            if not len(near):
                continue
            s = rr._dequantize(0, rows=near) @ q[p]
            v, i = torch.topk(s, min(N, len(near)))
            rows[h, :len(v)], sims[h, :len(v)] = near[i.cpu().numpy()], v.float().cpu().numpy()
            if h % 1000 == 0:
                print(f"  {h} / {H} areas ({len(near)} photos within {RADIUS_KM:.0f} km)", flush=True)
    valid = rows >= 0
    latlon = np.where(valid[..., None], np.asarray(rr.latlon)[np.where(valid, rows, 0)], np.nan).astype(np.float32)
    meta, _ = _paths(tag)
    np.savez(meta, rows=rows, sims=sims, latlon=latlon, n_near=n_near,
             cache_rows=np.where(valid, rr.cache_indices[np.where(valid, rows, 0)], -1))
    print(f"-> {meta}: {H} areas, median {np.median(n_near):.0f} photos within {RADIUS_KM:.0f} km, {len(np.unique(rows[valid]))} distinct")


def _cache() -> np.ndarray:
    """Index row -> MP16 embedding-cache row."""

    from ..models.pinpoint_reranker import _discover_index

    index_dir = _discover_index()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    return np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)


def _known(tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(MP16 cache rows already embedded by lessons 93 / 95, their source file id 0 / 1, row in that file), sorted by cache row."""

    from .megaloc_deep import _paths as deep_paths
    from .megaloc_gate import _paths as gate_paths

    gm = np.load(gate_paths(tag)[0])
    nq = int(gm["n_queries"])
    cache = _cache()
    g_cache = cache[gm["rows"]]
    d_cache = np.load(OUT / f"megaloc_deep_uniq_{tag}_{CKPT_TAG}.npy") if deep_paths(tag)[1].exists() else np.empty(0, np.int64)
    keys = np.concatenate([g_cache, d_cache])
    src = np.concatenate([np.zeros(len(g_cache), np.int64), np.ones(len(d_cache), np.int64)])
    pos = np.concatenate([nq + np.arange(len(g_cache)), nq + np.arange(len(d_cache))])
    keys, first = np.unique(keys, return_index=True)
    return keys, src[first], pos[first]


def embed(tag: str, batch: int = 64, workers: int = 3) -> None:
    import torch
    from torch.utils.data import DataLoader

    from .megaloc_gate import _image_ids, _model

    meta, emb = _paths(tag)
    z = np.load(meta)
    zm = _zoom(tag)
    need = np.unique(np.concatenate([z["rows"][z["rows"] >= 0], zm["rows"][zm["rows"] >= 0]]))  # index rows
    cache = _cache()
    keys, _, _ = _known(tag)
    missing = need[~np.isin(cache[need], keys)]
    print(f"{tag}: {len(need)} area photos, {len(need) - len(missing)} already embedded, {len(missing)} to embed", flush=True)
    ids, nq, dataset_cls = _image_ids(tag, missing)
    model = _model()
    out = np.lib.format.open_memmap(emb, mode="w+", dtype=np.float16, shape=(len(missing), 8448))
    with torch.inference_mode():
        for k, (idx, x) in enumerate(DataLoader(dataset_cls(ids[nq:]), batch_size=batch, num_workers=workers)):
            out[idx.numpy()] = model(x.cuda()).half().cpu().numpy()
            if k % 500 == 0:
                print(f"  {k * batch} / {len(missing)}", flush=True)
    out.flush()
    np.save(OUT / f"megaloc_area_keys_{tag}_{CKPT_TAG}.npy", cache[missing])
    print(f"-> {emb}")


class _Lookup:
    """MegaLoc embedding of any MP16 cache row embedded by lessons 93 / 95 or by embed above."""

    def __init__(self, tag: str) -> None:
        from .megaloc_deep import _paths as deep_paths
        from .megaloc_gate import _paths as gate_paths

        k0, s0, p0 = _known(tag)
        k1 = np.load(OUT / f"megaloc_area_keys_{tag}_{CKPT_TAG}.npy")
        self.keys = np.concatenate([k0, k1])
        self.src = np.concatenate([s0, np.full(len(k1), 2)])
        self.pos = np.concatenate([p0, np.arange(len(k1))])
        o = np.argsort(self.keys)
        self.keys, self.src, self.pos = self.keys[o], self.src[o], self.pos[o]
        self.files = [np.load(gate_paths(tag)[1], mmap_mode="r"), np.load(deep_paths(tag)[1], mmap_mode="r"), np.load(_paths(tag)[1], mmap_mode="r")]
        self.nq = int(np.load(gate_paths(tag)[0])["n_queries"])

    def queries(self) -> np.ndarray:
        return np.asarray(self.files[0][: self.nq], np.float32)

    def get(self, cache_rows: np.ndarray) -> np.ndarray:
        i = np.searchsorted(self.keys, cache_rows)
        assert np.all(self.keys[i] == cache_rows)
        out = np.zeros((len(cache_rows), 8448), np.float32)
        for s in range(3):
            m = self.src[i] == s
            if m.any():
                p = self.pos[i[m]]
                o = np.argsort(p)
                block = np.asarray(self.files[s][p[o]], np.float32)
                out[np.flatnonzero(m)[o]] = block
        return out


def _sims(tag: str):
    """Per area the MegaLoc similarity of the zoom file's 12 photos and of the N shortlist photos (NaN where empty)."""

    path = OUT / f"megaloc_area_sims_{tag}_{CKPT_TAG}.npz"
    if path.exists():
        s = np.load(path)
        return s["zoom"], s["deep"]
    cache = _cache()
    z = np.load(_paths(tag)[0])
    zm = _zoom(tag)
    look = _Lookup(tag)
    q = look.queries()
    out = {}
    for name, rows in (("zoom", zm["rows"]), ("deep", z["rows"])):
        sims = np.full(rows.shape, np.nan, np.float32)
        h_idx, k_idx = np.nonzero(rows >= 0)
        for s in range(0, len(h_idx), 100_000):
            h, k = h_idx[s:s + 100_000], k_idx[s:s + 100_000]
            e = look.get(cache[rows[h, k]])
            sims[h, k] = (e * q[zm["photo"][h]]).sum(1)
            print(f"  {name}: {s + len(h)} / {len(h_idx)}", flush=True)
        out[name] = sims
    np.savez(path, **out)
    return out["zoom"], out["deep"]


def _agg(sims: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    srt = np.sort(np.where(np.isnan(sims), -np.inf, sims), 1)[:, -3:]
    best = np.where(np.isfinite(srt[:, -1]), srt[:, -1], -1.0)
    with np.errstate(invalid="ignore"):
        top3 = np.nan_to_num(np.nanmean(np.where(np.isfinite(srt), srt, np.nan), 1), nan=-1.0)
    return best, top3


def _rel(v: np.ndarray, photo: np.ndarray) -> np.ndarray:
    out = np.zeros_like(v)
    for p in np.unique(photo):
        J = np.flatnonzero(photo == p)
        for k, j in enumerate(J):
            o = np.delete(v[J], k)
            out[j] = v[j] - (o.max() if len(o) else v[j])
    return out


def _auc(a: np.ndarray, b: np.ndarray) -> float:
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def _clogit(X: np.ndarray, photo: np.ndarray, right: np.ndarray) -> float:
    """megaloc_gate's conditional logit: % of photos whose picked area is right (< 25 km), 2-fold x 3 seeds."""

    import torch

    P = photo.max() + 1
    H = np.bincount(photo).max()
    slot = np.zeros(len(photo), int)
    seen = np.zeros(P, int)
    for j, q in enumerate(photo):
        slot[j], seen[q] = seen[q], seen[q] + 1
    X = ((X - X.mean(0)) / (X.std(0) + 1e-6)).astype(np.float32)
    T = np.zeros((P, H, X.shape[1]), np.float32)
    T[photo, slot] = X
    M = np.zeros((P, H), bool)
    M[photo, slot] = True
    Y = np.zeros((P, H), bool)
    Y[photo, slot] = right
    has = Y.any(1)
    hits = np.zeros(P)
    Xt, Mt, Yt = torch.tensor(T), torch.tensor(M), torch.tensor(Y)
    for seed in range(3):
        folds = np.random.default_rng(seed).permutation(P) % 2
        for k in range(2):
            tr, te = (folds != k) & has, folds == k
            w = torch.zeros(X.shape[1], requires_grad=True)
            opt = torch.optim.LBFGS([w], max_iter=200)

            def closure():
                opt.zero_grad()
                lp = torch.log_softmax((Xt[tr] @ w).masked_fill(~Mt[tr], -1e9), 1)
                loss = -torch.logsumexp(lp.masked_fill(~Yt[tr], -1e9), 1).mean() + 1e-3 * (w ** 2).sum()
                loss.backward()
                return loss

            opt.step(closure)
            s = (Xt[te] @ w.detach()).masked_fill(~Mt[te], -1e9)
            hits[te] += Y[te][np.arange(te.sum()), s.argmax(1).numpy()] / 3
    return float(hits.mean())


def report(tag: str) -> None:
    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
    truth = np.asarray([p["truth"] for p in photos], np.float64)
    zm = _zoom(tag)
    z = np.load(_paths(tag)[0])
    s_zoom, s_deep = _sims(tag)
    s_all = np.concatenate([s_zoom, s_deep], 1)
    ll_all = np.concatenate([zm["latlon"], z["latlon"]], 1).astype(np.float64)
    t = truth[zm["photo"]]
    km_all = _hav(t[:, 0, None], t[:, 1, None], ll_all[..., 0], ll_all[..., 1])
    km_all = np.where(np.isnan(ll_all[..., 0]), np.inf, km_all)

    # A. area choice
    f = np.load(OUT / f"hypothesis_features_{tag}_{CKPT_TAG}_e4_set.npz")
    keep = np.isin(f["source"], (0, 1))
    assert keep.sum() == len(zm["photo"]) and np.all(f["photo"][keep] == zm["photo"])
    photo, right, rank = f["photo"][keep], f["km_agree"][keep] < 25, f["rank"][keep]
    P = photo.max() + 1
    groups = {}
    for name, s in (("12 zoom photos", s_zoom), (f"{N} SigLIP2-nearest", s_deep), (f"12 + {N}", s_all)):
        b, t3 = _agg(s)
        groups[name] = np.stack([b, t3, _rel(b, photo), _rel(t3, photo)], 1)
    choice = np.asarray([right[photo == q].any() and not right[photo == q].all() for q in range(P)])
    top_wrong = np.asarray([right[photo == q].any() and not right[photo == q][np.argmin(rank[photo == q])] for q in range(P)])
    print(f"{tag}: {len(photo)} areas, {P} photos; median {np.median(z['n_near']):.0f} index photos within {RADIUS_KM:.0f} km of an area centre")
    print(f"A. right vs wrong areas, AUC of the best-similarity margin (photos with both: {choice.sum()}; top area wrong: {top_wrong.sum()})")
    for name, g in groups.items():
        print(f"   {name:22s} best {_auc(g[choice[photo] & right, 0], g[choice[photo] & ~right, 0]):.3f}  "
              f"margin {_auc(g[choice[photo] & right, 2], g[choice[photo] & ~right, 2]):.3f}  "
              f"margin, top wrong {_auc(g[top_wrong[photo] & right, 2], g[top_wrong[photo] & ~right, 2]):.3f}")
    base = np.stack([np.minimum(rank, 8), rank == 0, f["source"][keep], f["support"][keep], f["mass_0.2"][keep], f["photographers"][keep]], 1)
    clue = np.stack([f[k][keep] for k in ("read_n", "read_p", "read_share", "read_rare", "recall_n", "recall_p", "recall_share", "recall_rare")], 1)
    print("   conditional logit (< 25 km, 2-fold x 3 seeds):")
    print(f"     Pinpoint features + clues              {_clogit(np.concatenate([base, clue], 1), photo, right):.2%}")
    for name, g in groups.items():
        print(f"     + MegaLoc, {name:26s}  {_clogit(np.concatenate([base, clue, g], 1), photo, right):.2%}")
    print(f"     + MegaLoc, 12 and {N} as separate groups   {_clogit(np.concatenate([base, clue, groups['12 zoom photos'], groups[f'{N} SigLIP2-nearest']], 1), photo, right):.2%}")

    # B. the point inside an area: on Pinpoint's top area (rank 0) of each photo and on right areas
    top_area = (zm["source"] == 0) & (zm["rank"] == 0)
    arg_ml = np.argmax(np.where(np.isnan(s_all), -np.inf, s_all), 1)
    km_ml = km_all[np.arange(len(km_all)), arg_ml]
    km_zoom1 = km_all[:, 0]
    km_deep1 = km_all[:, 12]
    sim_best = np.nanmax(s_all, 1)
    srt = np.sort(np.where(np.isnan(s_all), -np.inf, s_all), 1)
    gap = srt[:, -1] - srt[:, -2]
    print("\nB. point inside an area, % within 1 / 25 km of the truth")
    for label, sel in (("Pinpoint's top area (all photos)", top_area), ("areas within 25 km of the truth", right)):
        print(f"   {label} ({sel.sum()}):")
        for thr in (1, 25):
            print(f"     < {thr:2d} km: restricted top-1 {np.mean(km_zoom1[sel] < thr):.1%}; SigLIP2-nearest top-1 {np.mean(km_deep1[sel] < thr):.1%}; "
                  f"MegaLoc best {np.mean(km_ml[sel] < thr):.1%}; any of the {12 + N} {np.mean((km_all[sel] < thr).any(1)):.1%}")
    # a switch on the top area: MegaLoc's best photo when its similarity (or margin) clears t, else the restricted top-1; t chosen on one half
    sel = np.flatnonzero(top_area)
    rng = np.random.default_rng(0)
    print("   switch to MegaLoc's best photo on the top area when its similarity / margin >= t (t chosen on one half, 1 km), net change of the other half:")
    for feat_name, v in (("similarity", sim_best), ("margin", gap)):
        res = {1: [], 25: []}
        for seed in range(5):
            half = rng.permutation(len(sel)) % 2
            for k in range(2):
                tr, te = sel[half != k], sel[half == k]
                grid = np.quantile(v[tr], np.linspace(0, 0.99, 100))
                gain = [np.mean(np.where(v[tr] >= g, km_ml[tr] < 1, km_zoom1[tr] < 1)) for g in grid]
                g = grid[int(np.argmax(gain))]
                for thr in (1, 25):
                    res[thr].append(np.mean(np.where(v[te] >= g, km_ml[te] < thr, km_zoom1[te] < thr)) - np.mean(km_zoom1[te] < thr))
        print(f"     {feat_name:10s}: 1 km {100 * np.mean(res[1]):+.2f} pts, 25 km {100 * np.mean(res[25]):+.2f} pts (of photos)")
    print("   MegaLoc best similarity on the top area, by bucket: photos, MegaLoc best < 1 km, restricted top-1 < 1 km")
    for lo, hi in ((-1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 2)):
        m = top_area & (sim_best >= lo) & (sim_best < hi)
        if m.any():
            print(f"     [{lo:4.1f}, {hi:3.1f}): {m.sum():5d}  {np.mean(km_ml[m] < 1):6.1%}  {np.mean(km_zoom1[m] < 1):6.1%}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("shortlist", "embed", "report"))
    parser.add_argument("--tag", default="dev3k")
    args = parser.parse_args(argv)
    {"shortlist": shortlist, "embed": embed, "report": report}[args.cmd](args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
