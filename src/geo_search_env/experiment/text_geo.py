# Text-to-location statistics for read text (lesson 92, the user chose to build on the clue-reading success). The pipeline uses read text
# only as names (text_search + the clue judge). Here every word the 4B reads becomes soft evidence for where it is written: from the 48M
# geotagged YFCC100M titles + tags (yfcc.db FTS5, rowid = geo.npz row + 1), counts of distinct (photographer, 0.25-degree cell) pairs per
# word; per hypothesis area, a naive-Bayes log-likelihood ratio sum_w log P(w | area) / P(w) with P(w | area) smoothed towards P(w), at 25 and
# 200 km. Language, regional vocabulary and names in one model. The query photo's photographer is removed from every count (leak guard).
# Gate (dev3k): separation of right / wrong areas and gain over the existing clue features in a conditional logit.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.text_geo build --tag dev3k
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.text_geo gate --tag dev3k

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
CKPT_TAG = "setft_a"
CELL = 0.25  # degrees
NLAT, NLON = int(180 / CELL), int(360 / CELL)
RADII = (25.0, 200.0)
ALPHA = 20.0  # pseudo-count towards the word's global rate
CAP = 2_000_000  # FTS matches kept per word (random sample beyond)
CLIP = (-3.0, 6.0)  # per-word log ratio
YFCC_DB = Path("/opt/dlami/nvme/text_search/yfcc.db")


def tokens(texts: Sequence[str]) -> list[str]:
    out = []
    for t in texts:
        t = "".join(c for c in unicodedata.normalize("NFKD", t.casefold()) if not unicodedata.combining(c))
        out += [w for w in re.findall(r"\w+", t) if len(w) >= 3 and not w.isdigit()]
    return sorted(set(out))


def _cells(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    i = np.clip(((lat + 90) / CELL).astype(np.int64), 0, NLAT - 1)
    j = np.clip(((lon + 180) / CELL).astype(np.int64), 0, NLON - 1)
    return i * NLON + j


def _window(lat: float, lon: float, km: float) -> list[tuple[int, int]]:
    """Cell-id ranges (one per latitude row) covering a km box around a point."""

    dlat = km / 111.0
    dlon = km / (111.0 * max(np.cos(np.radians(lat)), 0.05))
    i0, i1 = int(np.clip((lat - dlat + 90) / CELL, 0, NLAT - 1)), int(np.clip((lat + dlat + 90) / CELL, 0, NLAT - 1))
    j0, j1 = int(np.clip((lon - dlon + 180) / CELL, 0, NLON - 1)), int(np.clip((lon + dlon + 180) / CELL, 0, NLON - 1))
    return [(i * NLON + j0, i * NLON + j1) for i in range(i0, i1 + 1)]


def _paths(tag: str, reader: str = "4b") -> tuple[Path, Path]:
    sfx = "" if reader == "4b" else f"_{reader}"
    return OUT / f"textgeo_{tag}_{CKPT_TAG}{sfx}.npz", OUT / f"textgeo_words_{tag}{sfx}.json"


def build(tag: str, reader: str = "4b", specific: int = 0) -> None:
    """specific > 0: only words with at most that many (cell, photographer) pairs, positive evidence only (variant "_spec")."""
    from .region_evidence import _owns

    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
    obs = json.loads((OUT / f"observe_{tag}_{reader}.json").read_text(encoding="utf-8"))
    words = [tokens(((obs.get(e["image_id"]) or {}).get("parsed") or {}).get("text") or []) for e in photos]
    vocab = sorted({w for ws in words for w in ws})
    print(f"{tag}: read text on {np.mean([bool(w) for w in words]):.1%} of {len(photos)} photos, {len(vocab)} distinct words", flush=True)

    g = np.load("data/yfcc100m/geo.npz")
    lat, lon, uid = g["lat"], g["lon"], g["uid"]
    cell = _cells(lat.astype(np.float64), lon.astype(np.float64))
    uids = {u: i for i, u in enumerate(json.loads(Path("data/yfcc100m/geo_uid_vocab.json").read_text(encoding="utf-8")))}
    owns = _owns([{"image_id": e["image_id"]} for e in photos])
    q_uid = np.asarray([uids.get(owns.get(e["image_id"], (0, ""))[1], -1) for e in photos])
    print(f"  query photographers found in YFCC: {np.mean(q_uid >= 0):.1%}", flush=True)
    key = np.unique(cell * 300_000 + uid.astype(np.int64))  # distinct (cell, photographer) pairs
    total_pairs = len(key)
    bg = np.bincount(key // 300_000, minlength=NLAT * NLON).astype(np.float64).reshape(NLAT, NLON)
    prefix = np.zeros((NLAT + 1, NLON + 1))
    prefix[1:, 1:] = bg.cumsum(0).cumsum(1)
    del key
    print(f"  background: {total_pairs} (cell, photographer) pairs", flush=True)

    con = sqlite3.connect(f"file:{YFCC_DB}?mode=ro", uri=True)
    rng = np.random.default_rng(0)
    per_word: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for k, w in enumerate(vocab):
        try:
            rows = np.fromiter((r[0] for r in con.execute("SELECT rowid FROM titles WHERE titles MATCH ?", (f'"{w}"',))), np.int64)
        except sqlite3.OperationalError:
            rows = np.zeros(0, np.int64)
        scale = 1.0
        if len(rows) > CAP:
            scale = len(rows) / CAP
            rows = rng.choice(rows, CAP, replace=False)
        idx = rows - 1
        pk = np.unique(cell[idx] * 300_000 + uid[idx].astype(np.int64))
        per_word[w] = (pk // 300_000, (pk % 300_000).astype(np.int32), scale)  # sorted by cell
        if k % 2000 == 0:
            print(f"  words {k} / {len(vocab)}", flush=True)

    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{CKPT_TAG}.npz")
    centers, hyp_photo = zm["center"], zm["photo"]
    n = len(hyp_photo)
    cols = {f"{k}_{int(r)}": np.zeros(n, np.float32) for r in RADII for k in ("llr", "found")}
    cols["n_words"] = np.zeros(n, np.float32)
    for h in range(n):
        p = hyp_photo[h]
        ws = [w for w in words[p] if len(per_word[w][0]) and (not specific or len(per_word[w][0]) * per_word[w][2] <= specific)]
        cols["n_words"][h] = len(ws)
        if not ws:
            continue
        for r in RADII:
            win = _window(float(centers[h, 0]), float(centers[h, 1]), r)
            N = sum(prefix[(a // NLON) + 1, (b % NLON) + 1] - prefix[a // NLON, (b % NLON) + 1] - prefix[(a // NLON) + 1, a % NLON]
                    + prefix[a // NLON, a % NLON] for a, b in win)
            llr, found = 0.0, 0
            for w in ws:
                c, u, scale = per_word[w]
                pw = len(c) * scale / total_pairs
                lo = np.searchsorted(c, [a for a, _ in win])
                hi = np.searchsorted(c, [b + 1 for _, b in win])
                nw = 0
                for a, b in zip(lo, hi):
                    if b > a:
                        nw += int(b - a) - int((u[a:b] == q_uid[p]).sum())
                nw *= scale
                found += nw > 0
                llr += float(np.clip(np.log((nw + ALPHA * pw) / (N + ALPHA)) - np.log(pw), 0.0 if specific else CLIP[0], CLIP[1]))
            cols[f"llr_{int(r)}"][h], cols[f"found_{int(r)}"][h] = llr, found
        if h % 3000 == 0:
            print(f"  hypotheses {h} / {n}", flush=True)
    dst, wdst = _paths(tag, reader + ("_spec" if specific else ""))
    np.savez(dst, photo=hyp_photo, source=zm["source"], **cols)
    wdst.write_text(json.dumps(words, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"-> {dst}")


def gate(tag: str, reader: str = "4b") -> None:
    import torch

    z = np.load(_paths(tag, reader)[0])
    f = np.load(OUT / f"hypothesis_features_{tag}_{CKPT_TAG}_e4_set.npz")
    assert np.array_equal(z["photo"], f["photo"])
    keep = np.isin(f["source"], (0, 1))
    photo, right = f["photo"][keep], f["km_agree"][keep] < 25
    P = photo.max() + 1
    has_text = np.bincount(photo, weights=z["n_words"][keep] > 0, minlength=P) > 0

    def rel(v):
        out = np.zeros_like(v)
        for q in range(P):
            J = np.flatnonzero(photo == q)
            for k, j in enumerate(J):
                o = np.delete(v[J], k)
                out[j] = v[j] - (o.max() if len(o) else v[j])
        return out

    tg = {k: z[k][keep].astype(np.float64) for k in ("llr_25", "llr_200", "found_25", "found_200")}
    tg |= {f"{k}_rel": rel(v) for k, v in list(tg.items())}
    choice = np.asarray([right[photo == q].any() and not right[photo == q].all() for q in range(P)])
    sel = (choice & has_text)[photo]
    print(f"{tag}: {has_text.sum()} photos with read words; {(choice & has_text).sum()} of them with a right and a wrong hypothesis")
    for k, v in tg.items():
        a, b = v[sel & right], v[sel & ~right]
        auc = (a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()
        print(f"  {k:14s} AUC right vs wrong areas {auc:.3f}")
    clue = ("read_n", "read_p", "read_share", "read_rare", "recall_n", "recall_p", "recall_share", "recall_rare")
    for k in clue[:4]:
        v = f[k][keep].astype(np.float64)
        a, b = v[sel & right], v[sel & ~right]
        print(f"  existing {k:10s} AUC {(a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean():.3f}")

    rank = f["rank"][keep].astype(np.float32)
    base = np.stack([np.minimum(rank, 8), rank == 0, f["source"][keep], f["support"][keep], f["mass_0.2"][keep], f["photographers"][keep]], 1)
    clue_x = np.stack([f[k][keep] for k in clue], 1)
    text_x = np.stack(list(tg.values()), 1)
    H = np.bincount(photo).max()
    slot = np.zeros(len(photo), int)
    seen = np.zeros(P, int)
    for j, q in enumerate(photo):
        slot[j] = seen[q]
        seen[q] += 1

    def run(X, label):
        X = ((X - X.mean(0)) / (X.std(0) + 1e-6)).astype(np.float32)
        T = np.zeros((P, H, X.shape[1]), np.float32)
        T[photo, slot] = X
        M = np.zeros((P, H), bool)
        M[photo, slot] = True
        Y = np.zeros((P, H), bool)
        Y[photo, slot] = right
        has = Y.any(1)
        hits = np.zeros(P)
        for seed in range(3):
            folds = np.random.default_rng(seed).permutation(P) % 2
            for k in range(2):
                tr, te = (folds != k) & has, folds == k
                w = torch.zeros(X.shape[1], requires_grad=True)
                Xt, Mt, Yt = torch.tensor(T), torch.tensor(M), torch.tensor(Y)
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
        print(f"  {label:32s} all photos {hits.mean():.2%}, photos with read words {hits[has_text].mean():.2%}")

    print("conditional logit, pick right (< 25 km), 2-fold x 3 seeds:")
    run(base, "Pinpoint features")
    run(np.concatenate([base, clue_x], 1), "+ existing clue features")
    run(np.concatenate([base, text_x], 1), "+ text statistics")
    run(np.concatenate([base, clue_x, text_x], 1), "+ both")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("build", "gate"))
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--specific", type=int, default=0, help="build: keep only words with <= this many (cell, photographer) pairs")
    parser.add_argument("--reader", default="4b", help="observe_{tag}_{reader}.json (4b; 9bcrop: 9B text with 4 crops)")
    args = parser.parse_args(argv)
    build(args.tag, args.reader, args.specific) if args.cmd == "build" else gate(args.tag, args.reader)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
