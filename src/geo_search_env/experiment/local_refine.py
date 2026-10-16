# MegaLoc local refinement of the agent's answer point (lesson 111, the user: "add MegaLoc local refinement at 1 km"). Every area offers two
# answer points (hypothesis_agent: the global pool's best candidate near the area, and the zoomed kernel peak). Around each point P, a local
# chooser picks among P and the global pool's candidates within RADIUS_KM of P, from the reranker score / rank, agreement nearby, distance
# from P and MegaLoc same-place similarity (lesson 107: MegaLoc carries signal between neighbours where it fails globally). Listwise, trained
# on a train subset with the thresholds met as the reward; applied to the eval sets' area points, which action_agent --refine answers with.
# Usage: PYTHONPATH=src PINPOINT_KEEP_OWN=1 .venv/bin/python -m geo_search_env.experiment.local_refine embed --tag train --limit 12000
#        ... embed --tag dev3k;  ... fit --limit 12000 --apply dev3k

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .refine_headroom import _hav

OUT = Path("artifacts/agent")
CKPT_TAG = "setft_own"
RADIUS_KM = 25.0
THRESHOLDS = (1.0, 25.0)
MAX_C = 48
NAMES = ["is_P", "score_gap", "rank", "in_pool", "log_move", "n1", "mass1", "mass0.25", "ml", "ml_rel", "ml_nb1", "ml_nb1_rel", "ml_dup", "n5"]


def _photos(tag: str) -> list[dict]:
    """The photo set with each pool's reranker scores attached (the pool is stored sorted by score, as in the reference pools)."""

    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
    z = np.load(Path("artifacts/pinpoint_reference") / f"{'bench' if tag == 'full' else 'mp16'}_{CKPT_TAG}.npz")
    scores = z["scores"]
    for e in photos:
        s = scores[e["index"]]
        e["scores"] = s[np.isfinite(s)][: len(e["pool"])]
    return photos


def _cache() -> np.ndarray:
    from .megaloc_area import _cache as cache

    return cache()


def _points(tag: str, photos: list[dict]):
    """Per area (feature-file order): the zoom file, the area's photo, the unzoomed point (pool position, -1 = the centre) and the zoomed point
    (zoom-file candidate slot, -1 = none), as hypothesis_agent places them (gated rule)."""

    from .area_model import _paths
    from .hypothesis_agent import _kernel_km, _unzoomed_km

    feat_path, zoom_path = _paths(tag, CKPT_TAG)
    hphoto = np.load(feat_path)["photo"]
    zm = np.load(zoom_path)
    _, un = _unzoomed_km({"photo": hphoto, "center": zm["center"]}, photos, return_pick=True)
    _, zp = _kernel_km(zm, np.arange(len(hphoto)), np.asarray([e["truth"] for e in photos], np.float64), gate=True, return_pick=True)
    return zm, hphoto, un, zp


def _subset(n: int, limit: int | None) -> np.ndarray:
    return np.sort(np.random.default_rng(0).permutation(n)[:limit]) if limit else np.arange(n)


def _sources(tag: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """(cache-row keys, embeddings) of the MegaLoc caches holding this set's MP16 photos (lessons 93-97, 107, 111)."""

    names = [("local_megaloc", CKPT_TAG)] + ([("pool_megaloc", "setft_own"), ("refine_megaloc", "setft_a_r25")] if tag == "dev3k" else [])
    out = []
    for name, suffix in names:
        kp = OUT / f"{name}_keys_{tag}_{suffix}.npy"
        if kp.exists():
            out.append((np.load(kp), np.load(OUT / f"{name}_emb_{tag}_{suffix}.npy", mmap_mode="r")))
    return out


class _Emb:
    def __init__(self, tag: str):
        self.src = [(k[np.argsort(k)], np.argsort(k), e) for k, e in _sources(tag)]
        self.look = None
        if tag == "dev3k":
            from .megaloc_area import _Lookup

            self.look = _Lookup("dev3k")

    def keys(self) -> np.ndarray:
        ks = [sk for sk, _, _ in self.src] + ([self.look.keys] if self.look is not None else [])
        return np.unique(np.concatenate(ks)) if ks else np.zeros(0, np.int64)

    def get(self, keys: np.ndarray) -> np.ndarray:
        out = np.zeros((len(keys), 8448), np.float32)
        done = np.zeros(len(keys), bool)
        for sk, o, e in self.src:
            i = np.clip(np.searchsorted(sk, keys), 0, max(len(sk) - 1, 0))
            m = ~done & (sk[i] == keys)
            if m.any():
                pos = o[i[m]]
                order = np.argsort(pos)
                out[np.flatnonzero(m)[order]] = np.asarray(e[pos[order]], np.float32)
                done |= m
        if self.look is not None and (~done).any():
            m = ~done & np.isin(keys, self.look.keys)
            if m.any():
                out[m] = self.look.get(keys[m])
            done |= m
        assert done.all(), f"{(~done).sum()} candidates without a MegaLoc embedding"
        return out


def embed(tag: str, limit: int | None, batch: int = 64, workers: int = 3) -> None:
    """MegaLoc embeddings of every pool candidate and zoomed point of the (subset of) photos that no earlier cache holds."""

    import torch
    from torch.utils.data import DataLoader

    from . import megaloc_gate

    photos = _photos(tag)
    zm, hphoto, _, zp = _points(tag, photos)
    keep = _subset(len(photos), limit)
    cache = _cache()
    sel = np.isin(hphoto, keep) & (zp >= 0)
    need = np.unique(np.concatenate([np.asarray(photos[i]["pool_rows"]) for i in keep] + [zm["rows"][np.flatnonzero(sel), zp[sel]]]))
    need = need[~np.isin(cache[need], _Emb(tag).keys())]
    print(f"{tag}: {len(keep)} photos, {len(need)} candidate photos to embed", flush=True)
    megaloc_gate.CKPT_TAG = CKPT_TAG
    ids, nq, dataset_cls = megaloc_gate._image_ids(tag, need)
    model = megaloc_gate._model()
    emb_path = OUT / f"local_megaloc_emb_{tag}_{CKPT_TAG}.npy"
    out = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16, shape=(len(need), 8448))
    with torch.inference_mode():
        for k, (idx, x) in enumerate(DataLoader(dataset_cls(ids[nq:]), batch_size=batch, num_workers=workers)):
            out[idx.numpy()] = model(x.cuda()).half().cpu().numpy()
            if k % 500 == 0:
                print(f"  {k * batch} / {len(need)}", flush=True)
    out.flush()
    np.save(OUT / f"local_megaloc_keys_{tag}_{CKPT_TAG}.npy", cache[need])
    print(f"-> {emb_path}")


def lists(tag: str, limit: int | None = None):
    """Per answer option (area x {unzoomed, zoomed}): its local candidate list. Returns x [O, C, F], valid, km [O, C], the option's area row
    and kind (0 unzoomed, 1 zoomed)."""

    photos = _photos(tag)
    zm, hphoto, un, zp = _points(tag, photos)
    keep = np.zeros(len(photos), bool)
    keep[_subset(len(photos), limit)] = True
    cache = _cache()
    emb = _Emb(tag)
    q = np.asarray(np.load(OUT / f"megaloc_queries_{tag}_{CKPT_TAG}.npy"), np.float32)
    opts = []  # (area row, kind, point, zoom-file row of the point or -1)
    for h, p in enumerate(hphoto):
        if keep[p] and un[h] >= 0:
            opts.append((h, 0, np.asarray(photos[p]["pool"][un[h]], np.float64), -1))
        if keep[p] and zp[h] >= 0:
            opts.append((h, 1, zm["latlon"][h, zp[h]].astype(np.float64), int(zm["rows"][h, zp[h]])))
    zrows = np.unique([o[3] for o in opts if o[3] >= 0])
    zsim_emb = dict(zip(zrows.tolist(), emb.get(cache[zrows]))) if len(zrows) else {}
    sims = {}
    x = np.zeros((len(opts), MAX_C, len(NAMES)), np.float32)
    valid = np.zeros((len(opts), MAX_C), bool)
    km = np.full((len(opts), MAX_C), np.inf)
    for j, (h, kind, P, zrow) in enumerate(opts):
        p = int(hphoto[h])
        e = photos[p]
        if p not in sims:
            sims[p] = emb.get(cache[np.asarray(e["pool_rows"])]) @ q[p]
        pool, s = np.asarray(e["pool"], np.float64), np.asarray(e["scores"], np.float64)
        idx = np.flatnonzero(_hav(P[0], P[1], pool[:, 0], pool[:, 1]) < RADIUS_KM)[: MAX_C - 1]
        ll, sc, sm, rank, inpool = pool[idx], s[idx], sims[p][idx], idx.astype(np.float64), np.ones(len(idx))
        is_p = _hav(P[0], P[1], ll[:, 0], ll[:, 1]) < 1e-3
        if zrow >= 0 and not is_p.any():  # the zoomed point is a restricted candidate outside the global pool
            ll = np.vstack([ll, P[None]])
            sc, sm = np.r_[sc, np.nan], np.r_[sm, float(zsim_emb[zrow] @ q[p])]
            rank, inpool, is_p = np.r_[rank, 46.0], np.r_[inpool, 0.0], np.r_[is_p, True]
        if is_p.sum() > 1:  # several pool photos at the point's coordinate: the best-ranked one stands for it
            is_p[np.flatnonzero(is_p)[1:]] = False
        c = len(ll)
        pair = _hav(ll[:, None, 0], ll[:, None, 1], ll[None, :, 0], ll[None, :, 1])
        soft = np.where(np.isfinite(sc), np.exp((np.nan_to_num(sc, nan=-1e9) - s.max()) / 0.2), 0.0)
        others = ~np.eye(c, dtype=bool)
        nb = np.where((pair < 1.0) & others, sm[None, :], -1.0).max(1)
        cols = [is_p, np.nan_to_num(sc - s.max(), nan=-5.0), np.minimum(rank, 46) / 46, inpool, np.log1p(_hav(P[0], P[1], ll[:, 0], ll[:, 1])),
                ((pair < 1.0) & others).sum(1), (pair < 1.0) @ soft, (pair < 0.25) @ soft, sm, sm - sm.max(), nb, nb - nb.max(),
                sm >= 0.5, np.log1p((pair < 5.0).sum(1))]
        x[j, :c] = np.stack(cols, 1)
        valid[j, :c] = True
        km[j, :c] = _hav(e["truth"][0], e["truth"][1], ll[:, 0], ll[:, 1])
    return x, valid, km, np.asarray([o[0] for o in opts]), np.asarray([o[1] for o in opts]), photos, hphoto, un


def fit(apply: Sequence[str], limit: int | None, hidden: int = 32, gap: float = 0.5) -> None:
    import torch

    from .area_model import _paths
    from .hypothesis_agent import _unzoomed_km

    xt, vt, kt, _, _, _, _, _ = lists("train", limit)
    m = xt[vt].mean(0), xt[vt].std(0) + 1e-6
    norm = lambda x, v: torch.tensor(np.where(v[..., None], (x - m[0]) / m[1], 0.0).astype(np.float32))
    is_p = lambda x: x[..., 0] > 0.5
    at_p = lambda x, k: np.where(is_p(x), k, np.inf).min(1)
    print(f"train: {len(xt)} answer options, {vt.sum(1).mean():.1f} local candidates; the point < 1 km {np.mean(at_p(xt, kt) < 1):.1%}, "
          f"list oracle {np.mean((kt < 1).any(1)):.1%}", flush=True)

    def train_net(h, seed):
        torch.manual_seed(seed)
        d = xt.shape[-1]
        net = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Linear(h, 1)) if h else torch.nn.Linear(d, 1)
        X, M = norm(xt, vt), torch.tensor(vt)
        R = torch.tensor((sum((kt < t).astype(np.float32) for t in THRESHOLDS) * vt).astype(np.float32))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        rng = np.random.default_rng(seed)
        for _ in range(60):
            for b in np.array_split(rng.permutation(len(X)), max(1, len(X) // 4096)):
                opt.zero_grad()
                lp = torch.log_softmax(net(X[b]).squeeze(-1).masked_fill(~M[b], -1e9), 1)
                (-(lp.exp() * R[b]).sum(1).mean()).backward()
                opt.step()
        return net

    nets = {h: [train_net(h, s) for s in range(3)] for h in (0, hidden)}

    def choose(ns, x, v):
        with torch.no_grad():
            return np.mean([torch.log_softmax(n(norm(x, v)).squeeze(-1).masked_fill(~torch.tensor(v), -1e9), 1).numpy() for n in ns], 0)

    for tag in apply:
        x, v, k, area, kind, photos, hphoto, un = lists(tag)
        base = at_p(x, k)
        print(f"\n{tag}: {len(x)} answer options; at the point < 1 km {np.mean(base < 1):.1%}, < 25 km {np.mean(base < 25):.1%}; list oracle "
              f"< 1 km {np.mean((k < 1).any(1)):.1%}  (net change over all options, 3-seed ensembles trained on train)")
        refined = None
        for h, ns in nets.items():
            lp = choose(ns, x, v)
            for g in (0.0, 0.5, 1.0):
                move = lp.max(1) - np.where(is_p(x), lp, -np.inf).max(1) > g
                kk = np.where(move, k[np.arange(len(k)), lp.argmax(1)], base)
                print(f"  {'linear' if not h else f'MLP {h}':7s} gap > {g}: moves {move.mean():.0%}; < 1 km {100 * (np.mean(kk < 1) - np.mean(base < 1)):+.2f}, "
                      f"< 25 km {100 * (np.mean(kk < 25) - np.mean(base < 25)):+.2f}")
                if h == hidden and g == gap:
                    refined = kk
        zm = np.load(_paths(tag, CKPT_TAG)[1])
        ku = _unzoomed_km({"photo": hphoto, "center": zm["center"]}, photos)  # areas without a pool candidate nearby keep the centre
        kz = np.full(len(hphoto), np.inf)
        ku[area[kind == 0]], kz[area[kind == 1]] = refined[kind == 0], refined[kind == 1]
        dst = OUT / f"local_refine_{tag}_{CKPT_TAG}.npz"
        np.savez(dst, km_unzoom=ku, km_zoom=kz)
        print(f"  -> {dst} (MLP {hidden}, gap > {gap}, chosen before looking at {tag})")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("embed", "fit"))
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--limit", type=int, help="a random subset of this many train photos")
    parser.add_argument("--apply", nargs="*", default=("dev3k",), help="fit: photo sets to refine (full = test set, only when the user asks)")
    args = parser.parse_args(argv)
    if args.cmd == "embed":
        embed(args.tag, args.limit if args.tag == "train" else None)
    else:
        fit(args.apply, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
