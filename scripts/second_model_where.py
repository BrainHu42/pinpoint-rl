"""Where does the second image models' gain come from? (dev3k; the gate's stack of PE-Core-G + DINOv3-7B columns on the answer head's option
logits, second_model_gate.py). Every photo-seed whose pick changes is a fix (head wrong -> right < 25 km), a break, or neutral; fixes minus
breaks (pts of dev3k, averaged over the 3 seeds) split by: same area / other area, the head's miss distance, the new area's source and rank,
whether the query's photographer has a photo in the new / old area (the original protocol keeps them in the index), scene (MP16's S3
label), continent, and how close the best new-model match is (near-duplicate or not).

  PYTHONPATH=src .venv/bin/python scripts/second_model_where.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import second_model_gate as smg  # noqa: E402

TAG, CKPT, MODELS = "dev3k", "setft_own", ("pe_g", "dino_7b")


def picks(X: np.ndarray, valid: np.ndarray, km: np.ndarray, seeds: int = 3) -> np.ndarray:
    """smg.clogit, returning the chosen option per seed (seeds, n)."""

    import torch

    X = np.where(valid[..., None], X, 0.0)
    mu, sd = X[valid].mean(0), X[valid].std(0) + 1e-6
    Xt = torch.tensor(np.where(valid[..., None], (X - mu) / sd, 0.0), dtype=torch.float32)
    Mt = torch.tensor(valid)
    out = np.zeros((seeds, len(X)), int)
    for seed in range(seeds):
        folds = np.random.default_rng(seed).permutation(len(X)) % 2
        for k in range(2):
            tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            w = torch.zeros(X.shape[-1], requires_grad=True)
            w.data[0] = 1.0
            opt = torch.optim.LBFGS([w], max_iter=300)

            def closure():
                opt.zero_grad()
                lp = torch.log_softmax((Xt[tr] @ w).masked_fill(~Mt[tr], -1e9), 1)
                loss = 1e-3 * (w ** 2).sum()
                for r, wt in ((25.0, 1.0), (1.0, 0.5)):
                    ok = torch.tensor((km[tr] < r) & valid[tr])
                    use = ok.any(1)
                    if use.any():
                        loss = loss - wt * torch.logsumexp(lp.masked_fill(~ok, -1e9), 1)[use].mean()
                loss.backward()
                return loss

            opt.step(closure)
            out[seed, te] = (Xt[te] @ w.detach()).masked_fill(~Mt[te], -1e9).numpy().argmax(1)
    return out


def authors(tag: str, ckpt: str, zm) -> tuple[list[str], dict[int, str], dict[str, dict]]:
    """Query authors (photo-set order), index row -> author, and the query's MP16 metadata (scene, continent)."""

    from geo_search_env.models.pinpoint_reranker import _discover_index
    from geo_search_env.experiment.strategy_search import MP16_CSV, MP16_EMBED, _memmap_gallery

    photos = json.loads(Path(f"artifacts/photo_sets/{tag}_{ckpt}/photos.json").read_text(encoding="utf-8"))
    index_dir = _discover_index()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    cache = np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
    row_index = _memmap_gallery(MP16_EMBED)["row_index"]
    rows = np.unique(zm["rows"][zm["rows"] >= 0])
    csv_row = {int(row_index[cache[r]]): int(r) for r in rows}
    qids = {p["image_id"] for p in photos}
    by_row, meta = {}, {}
    with MP16_CSV.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        head = next(reader)
        col = {c: i for i, c in enumerate(head)}
        for i, r in enumerate(reader):
            if i in csv_row:
                by_row[csv_row[i]] = r[col["AUTHOR"]]
            if r[0] in qids:
                meta[r[0]] = {"author": r[col["AUTHOR"]], "s3": r[col["S3_Label"]], "continent": r[col["continent"]] or "?",
                              "indoor": float(r[col["Prob_indoor"]] or 0)}
    return [meta.get(p["image_id"], {}).get("author", "") for p in photos], by_row, {p["image_id"]: meta.get(p["image_id"], {}) for p in photos}


def main() -> None:
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{TAG}_{CKPT}.npz")
    f = np.load(Path("artifacts/agent") / f"hypothesis_features_{TAG}_{CKPT}_e4_set.npz")
    logits, km, hyp = smg.head_options(TAG, CKPT)
    valid = np.isfinite(logits)
    n, H2 = km.shape
    h = H2 // 2
    head = np.where(valid, logits, -np.inf).argmax(1)
    cols = {m: smg.columns(m, TAG, CKPT, zm) for m in MODELS}
    X = np.concatenate([np.where(valid, logits, 0.0)[..., None]] +
                       [np.where((hyp >= 0)[..., None], c[np.maximum(hyp, 0)], 0.0) for c in cols.values()], -1)
    P = picks(X, valid, km)
    q_author, row_author, meta = authors(TAG, CKPT, zm)
    photos = json.loads(Path(f"artifacts/photo_sets/{TAG}_{CKPT}/photos.json").read_text(encoding="utf-8"))

    def own(p: int, opt: int) -> bool:  # the query's photographer has a photo among the option's area photos
        j = hyp[p, opt]
        return j >= 0 and any(row_author.get(int(r)) == q_author[p] for r in zm["rows"][j] if r >= 0)

    agg = defaultdict(lambda: defaultdict(float))
    S = len(P)
    hk = km[np.arange(n), head]
    for s in range(S):
        for p in range(n):
            o = P[s, p]
            if o == head[p]:
                continue
            new, old = float(km[p, o]), float(hk[p])
            d = int(new < 25) - int(old < 25)
            d1 = int(new < 1) - int(old < 1)
            j = hyp[p, o]
            m = meta[photos[p]["image_id"]]
            keys = {
                "switch": "same area, other point" if o % h == head[p] % h else "other area",
                "head's miss": "head right (< 25 km)" if old < 25 else ("25-200 km" if old < 200 else ("200-750 km" if old < 750 else "> 750 km")),
                "new area source": {0: "Pinpoint area", 1: "clue place", 2: "street"}.get(int(zm["source"][j]), "?"),
                "new area rank": f"rank {min(int(f['rank'][j]), 3)}{'+' if f['rank'][j] >= 3 else ''}",
                "photographer in new area": str(own(p, o)),
                "photographer in head's area": str(own(p, head[p])),
                "scene (MP16 S3)": {"0.0": "indoor", "1.0": "natural", "2.0": "urban"}.get(m.get("s3"), "?"),
                "continent": m.get("continent", "?"),
                "best PE-G cosine, new area": ("> 0.9" if cols["pe_g"][j, 0] > 0.9 else "0.8-0.9" if cols["pe_g"][j, 0] > 0.8 else "< 0.8"),
            }
            for g, v in keys.items():
                agg[g, v]["fix"] += (d > 0) / S
                agg[g, v]["break"] += (d < 0) / S
                agg[g, v]["d1"] += d1 / S
                agg[g, v]["n"] += 1 / S
    base = smg.clogit(np.where(valid, logits, 0.0)[..., None], valid, km)
    tot = (km[np.arange(n)[None], P] < 25).mean() - (base < 25).mean()
    print(f"\n{TAG}: stack of {'+'.join(MODELS)} on the head: {100 * tot:+.2f} pts < 25 km; changed picks per seed {np.mean(P != head):.1%} of photos")
    print(f"  {'':44s} {'changed':>8s} {'fixes':>7s} {'breaks':>7s} {'net pts':>8s} {'net 1 km':>9s}")
    last = None
    for (g, v), a in sorted(agg.items(), key=lambda kv: (kv[0][0], -(kv[1]["fix"] - kv[1]["break"]))):
        if g != last:
            print(f"  {g}")
            last = g
        print(f"    {v:42s} {a['n']:8.1f} {a['fix']:7.1f} {a['break']:7.1f} {100 * (a['fix'] - a['break']) / n:+8.2f} {100 * a['d1'] / n:+9.2f}")
    has_own = np.asarray([any(own(p, o) for o in range(h) if valid[p, o]) for p in range(n)])
    print(f"  photos whose photographer appears in some candidate area: {has_own.mean():.1%}")



def without_own() -> None:
    """The same stack with the query photographer's photos dropped from every area's new-model columns (head unchanged): the gain that
    does not come from recognising the photographer's other photos."""

    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{TAG}_{CKPT}.npz")
    q_author, row_author, _ = authors(TAG, CKPT, zm)
    masked = dict(zm)
    rows = zm["rows"].copy()
    for j, p in enumerate(zm["photo"]):
        for k, r in enumerate(rows[j]):
            if r >= 0 and row_author.get(int(r)) == q_author[p]:
                rows[j, k] = -1
    masked["rows"] = rows
    print(f"  area photos by the query's photographer: {np.mean((zm['rows'] >= 0) & (rows < 0)) / np.mean(zm['rows'] >= 0):.1%}")
    logits, km, hyp = smg.head_options(TAG, CKPT)
    valid = np.isfinite(logits)
    base = smg.clogit(np.where(valid, logits, 0.0)[..., None], valid, km)
    for label, z in (("all area photos", zm), ("photographer's photos dropped", masked)):
        cols = [smg.columns(m, TAG, CKPT, z) for m in MODELS]
        X = np.concatenate([np.where(valid, logits, 0.0)[..., None]] + [np.where((hyp >= 0)[..., None], c[np.maximum(hyp, 0)], 0.0) for c in cols], -1)
        res = smg.clogit(X, valid, km)
        print(f"  {label:32s}: vs head {100 * ((res < 25).mean() - (base < 25).mean()):+.2f} / {100 * ((res < 1).mean() - (base < 1).mean()):+.2f} pts (< 25 / 1 km)")


if __name__ == "__main__":
    without_own() if "--without-own" in sys.argv else main()
