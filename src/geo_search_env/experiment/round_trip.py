# Round-trip verification (lesson 90): every evidence source so far asked "does this area look like the photo?", which Pinpoint's look-alike
# areas pass by construction. Here the reverse question: seen from an area's photo, is the query special? For each candidate photo r of each
# hypothesis (the zoom file's 12 restricted-search photos), the reverse rank = how many index photos r is closer to than to the query, in
# Pinpoint's contrastive retrieval space (256-d image embeddings). A same-place match ranks the query near the top from its side too; a generic
# look-alike (beach, sunset, nave) has thousands of photos closer. Photos by the query's photographer never count (benchmark rule); a variant
# also drops r's own photographer (bursts of one place).
# Gate (dev3k): does the reverse rank separate right from wrong areas better than forward similarity (lesson 89: AUC 0.68)?
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.round_trip ranks --tag dev3k
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.round_trip report --tag dev3k

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
CKPT_TAG = "setft_a"


def _path(tag: str) -> Path:
    return OUT / f"roundtrip_{tag}_{CKPT_TAG}.npz"


def ranks(tag: str, batch: int = 2048, chunk: int = 512_000) -> None:
    import torch

    from ..tools.pinpoint import Pinpoint

    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{CKPT_TAG}.npz")
    tool = Pinpoint()
    rr = tool.reranker
    index_author = tool.index_author()
    q_author = tool.mp16_author(np.asarray([e["row"] for e in photos]))
    views = np.load(OUT / f"crops_{tag}_{CKPT_TAG}.npy", mmap_mode="r")
    _, q_emb = rr.encode_queries(np.asarray(views[:, 0], np.float32))  # [P, 256] unit, the photo's own contrastive embedding
    dev = rr.device
    E = torch.as_tensor(np.asarray(rr.image), device=dev)  # [N, 256] fp16, ~2 GB
    A = torch.as_tensor(index_author, device=dev)
    rows = zm["rows"]
    h, k = np.nonzero(rows >= 0)
    r = rows[h, k]
    p = zm["photo"][h]
    print(f"{tag}: {len(r)} (hypothesis, photo) pairs over {len(np.unique(r))} index photos; index {len(E)}", flush=True)
    sim = np.zeros(len(r), np.float32)
    rev = np.zeros(len(r), np.int64)  # index photos closer to r than the query (query's photographer and r itself excluded)
    rev_own = np.zeros(len(r), np.int64)  # ... also excluding r's photographer
    with torch.inference_mode():
        for s in range(0, len(r), batch):
            rb = torch.as_tensor(r[s:s + batch], device=dev)
            er = E[rb]  # [b, 256]
            qa = torch.as_tensor(q_author[p[s:s + batch]], device=dev)
            ra = A[rb]
            thr = (er.float() * q_emb[torch.as_tensor(p[s:s + batch], device=dev)]).sum(1)
            sim[s:s + batch] = thr.cpu().numpy()
            c1 = torch.zeros(len(rb), dtype=torch.long, device=dev)
            c2 = torch.zeros_like(c1)
            for c in range(0, len(E), chunk):
                sc = (er @ E[c:c + chunk].T).float()  # [b, chunk]
                closer = sc > thr[:, None]
                closer &= A[c:c + chunk][None, :] != qa[:, None]
                own = A[c:c + chunk][None, :] == ra[:, None]
                c1 += closer.sum(1)
                c2 += (closer & ~own).sum(1)
            c1 -= 1  # r itself (similarity 1 > thr) is always counted
            rev[s:s + batch] = c1.cpu().numpy()
            rev_own[s:s + batch] = c2.cpu().numpy()
            if (s // batch) % 20 == 0:
                print(f"  {s + len(rb)} / {len(r)}", flush=True)
    np.savez(_path(tag), hyp=h, slot=k, row=r, photo=p, sim=sim, rev=rev, rev_own=rev_own)
    print(f"-> {_path(tag)}; reverse rank median {np.median(rev):.0f}, < 10 on {np.mean(rev < 10):.1%} of pairs")


def per_hypothesis(tag: str) -> tuple[np.ndarray, list[str]]:
    """Columns per hypothesis (rows of the zoom / feature files): min / median log reverse rank, how many photos have the query in their
    top 10 / 100, forward similarity (max) for comparison, both reverse-rank variants."""

    z = np.load(_path(tag))
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{CKPT_TAG}.npz")
    n = len(zm["photo"])
    cols, names = [], []
    for key in ("rev", "rev_own"):
        lr = np.log1p(z[key]).astype(np.float32)
        mn = np.full(n, np.log1p(4e6), np.float32)
        np.minimum.at(mn, z["hyp"], lr)
        med = np.full(n, np.log1p(4e6), np.float32)
        order = np.argsort(z["hyp"], kind="stable")
        hs, ls = z["hyp"][order], lr[order]
        starts = np.flatnonzero(np.r_[True, hs[1:] != hs[:-1]])
        for a, b in zip(starts, np.r_[starts[1:], len(hs)]):
            med[hs[a]] = np.median(ls[a:b])
        t10 = np.bincount(z["hyp"], weights=z[key] < 10, minlength=n).astype(np.float32)
        t100 = np.bincount(z["hyp"], weights=z[key] < 100, minlength=n).astype(np.float32)
        cols += [mn, med, t10, t100]
        names += [f"{key}_min", f"{key}_median", f"{key}_top10", f"{key}_top100"]
    fwd = np.full(n, -1.0, np.float32)
    np.maximum.at(fwd, z["hyp"], z["sim"])
    cols.append(fwd)
    names.append("fwd_sim_max")
    return np.stack(cols, 1), names


def report(tag: str) -> None:
    x, names = per_hypothesis(tag)
    f = np.load(OUT / f"hypothesis_features_{tag}_{CKPT_TAG}_e4_set.npz")
    keep = np.isin(f["source"], (0, 1))
    photo, rank, right, x = f["photo"][keep], f["rank"][keep], f["km_agree"][keep] < 25, x[keep]
    choice = np.asarray([right[photo == q].any() and not right[photo == q].all() for q in range(photo.max() + 1)])
    sel = choice[photo]
    print(f"{tag}: {choice.sum()} photos with a right and a wrong hypothesis")
    # within-photo margin over the best other hypothesis (higher = better for the area), so sign-flip rank-like columns
    sign = np.asarray([-1.0 if ("min" in n or "median" in n) else 1.0 for n in names])
    for k, n in enumerate(names):
        v = sign[k] * x[:, k]
        a, b = v[sel & right], v[sel & ~right]
        auc = (a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()
        hits = [right[J][np.argmax(v[J])] for J in (np.flatnonzero(photo == q) for q in np.flatnonzero(choice))]
        print(f"  {n:16s} AUC {auc:.3f}; argmax pick right {np.mean(hits):.1%}")
    hits = [right[J][np.argmin(rank[J])] for J in (np.flatnonzero(photo == q) for q in np.flatnonzero(choice))]
    print(f"  {'rank (top area)':16s} pick right {np.mean(hits):.1%}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("ranks", "report"))
    parser.add_argument("--tag", default="dev3k")
    args = parser.parse_args(argv)
    if args.cmd == "ranks":
        ranks(args.tag)
    else:
        report(args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
