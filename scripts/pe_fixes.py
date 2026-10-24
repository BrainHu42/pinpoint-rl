"""Same place or looks alike? (PE_PLAN exploration, the core question: knowledge beyond visual similarity, measured as reranker fixes.)
Reads `pe_retrieval.py rerank --combo`'s dump (artifacts/retrieval/b1/combo_dump_dev3k.npz: today's setft_own pool, the reranker's scores,
each candidate's raw PE-G / SigLIP2 / contrastive similarity to the query and its trained-PE-G similarity) and asks:
1. where reranker + lambda x trained PE-G fixes the reranker (or breaks it): is its pick less alike than the reranker's pick (raw PE-G,
   SigLIP2), and where does it rank by raw similarity within the pool?
2. matched pairs: among a pool's candidate pairs (one < 25 km from the truth, one > 25 km) that look equally alike (raw similarity within
   0.01), how often each similarity ranks the right one higher. ~50% = appearance only; well above = place identity beyond appearance.

  .venv/bin/python scripts/pe_fixes.py [--set dev3k]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.radians(a), np.radians(b)
    h = np.sin((b[..., 0] - a[..., 0]) / 2) ** 2 + np.cos(a[..., 0]) * np.cos(b[..., 0]) * np.sin((b[..., 1] - a[..., 1]) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev3k")
    z = np.load(Path("artifacts/retrieval/b1") / f"combo_dump_{ap.parse_args().set}.npz")
    v = z["rows"] >= 0
    q = np.arange(len(v))
    d = km(z["truth"][:, None, :], z["latlon"].astype(np.float64))  # [Q, C] km of each candidate
    lam = np.where(z["fold"] == 0, z["lam_trained"][0], z["lam_trained"][1])[:, None]
    r = np.where(v, z["score"], -np.inf).argmax(1)
    c = np.where(v, z["score"] + lam * z["cos_trained"], -np.inf).argmax(1)
    right_r, right_c = d[q, r] < 25, d[q, c] < 25
    sims = {"raw PE-G": z["cos_raw"], "SigLIP2": z["sim_siglip2"], "contrastive": z["sim_contrastive"], "trained PE-G": z["cos_trained"]}
    print(f"{len(q)} photos; reranker {100 * right_r.mean():.2f}%, + trained PE-G {100 * right_c.mean():.2f}% (< 25 km); "
          f"fixes {int((right_c & ~right_r).sum())}, breaks {int((right_r & ~right_c).sum())}")

    def rank(s: np.ndarray, j: np.ndarray) -> np.ndarray:  # 1 = the most similar candidate in the pool
        s = np.where(v, s, -np.inf)
        return 1 + (s > s[q, j][:, None]).sum(1)

    for name, sel in (("fixes (reranker wrong, + trained PE-G right)", right_c & ~right_r),
                      ("breaks (reranker right, + trained PE-G wrong)", right_r & ~right_c)):
        print(f"\n{name}: {int(sel.sum())} photos; the new pick vs the reranker's pick")
        for k, s in sims.items():
            new, old = s[q, c][sel], s[q, r][sel]
            print(f"  {k:13s} median {np.median(new):.3f} vs {np.median(old):.3f}; the new pick is less alike for {100 * np.mean(new < old):.0f}%; "
                  f"its rank in the pool by {k}: median {np.median(rank(s, c)[sel]):.0f} (reranker's pick {np.median(rank(s, r)[sel]):.0f})")
        print(f"  distance of the new pick {np.median(d[q, c][sel]):.1f} km vs the reranker's {np.median(d[q, r][sel]):.0f} km (medians)")

    # candidates the projection saw in training (B1 gallery, the train half: author hash parity 1, dev3k photographers out; pe_retrieval trained)
    import hashlib

    g = np.load("artifacts/retrieval/b1/gallery.npz")
    half = np.asarray([hashlib.md5(a.encode()).digest()[0] & 1 for a in g["authors"]], bool)
    seen_rows = set(g["rows"][half].tolist())  # conservative: the train half's dev3k-photographer photos (left out) count as seen too
    seen = np.vectorize(lambda r: r in seen_rows)(np.where(v, z["rows"], -1)) & v
    print(f"\ncandidates the projection trained on: {100 * seen[v].mean():.1f}% of pool slots")

    # 2. matched pairs within each pool: right (< 25 km) vs wrong (> 25 km) candidates that look equally alike
    rng = np.random.default_rng(0)
    for match, unseen in (("raw PE-G", False), ("SigLIP2", False), ("raw PE-G", True), ("SigLIP2", True)):
        wins = {k: [] for k in sims}
        for i in range(len(q)):
            ok = np.flatnonzero(v[i] & ~seen[i]) if unseen else np.flatnonzero(v[i])
            R, W = ok[d[i, ok] < 25], ok[d[i, ok] >= 25]
            if not len(R) or not len(W):
                continue
            m = sims[match][i]
            close = np.abs(m[R][:, None] - m[W][None, :]) < 0.01
            pr, pw = np.nonzero(close)
            if len(pr) > 20:
                keep = rng.choice(len(pr), 20, replace=False)
                pr, pw = pr[keep], pw[keep]
            for k, s in sims.items():
                wins[k].extend((s[i, R[pr]] > s[i, W[pw]]).astype(float) + 0.5 * (s[i, R[pr]] == s[i, W[pw]]))
        n = len(wins["trained PE-G"])
        print(f"\nmatched pairs{' (only candidates the projection never saw)' if unseen else ''} (right vs wrong candidate, same pool, {match} "
              f"similarity within 0.01; {n:,} pairs, <= 20 per photo): "
              "% where each similarity ranks the right one higher")
        print("  " + "   ".join(f"{k} {100 * np.mean(w):.1f}%" for k, w in wins.items()))


if __name__ == "__main__":
    main()
