"""Probe A (user, 2026-10-24; explore stage): does an evidence source's independence from the proposer predict what it adds? Thesis: the
system proposes right places far more often than it can pick them (V7, V9, S1-S3), and only evidence independent of retrieval's similarity
picks them (PE-G +2.4, text AUC 0.94; look-alike checks ~0.6-0.79). If so, a source's added value should follow its independence from the
head, not its strength alone.

Per source with per-area columns on dev3k (MegaLoc, SigLIP2's own pooled embedding = the retriever's model, PE-G, PE-G + track R's trained
projection, PE-L, DINOv3-L / 7B, the PE-G region probe), on the answer head of the current pipeline (second_model_gate.head_options):
  strength      AUC right vs wrong areas over all photos with both (the source's best-cosine column alone)
  independence  1 - mean within-photo Spearman between the source's area scores and the head's area logits (photos with >= 3 areas)
  complement    AUC right vs wrong areas on photos where the head's pick is wrong and a right area exists
  pick          the source's own top area right on those photos (%)
  gain          + its 4 columns in second_model_gate.clogit over the head logit, pts < 25 km (2-fold x 3 seeds, paired SE)
Then rank correlations across sources: gain vs strength, independence, complement.

  PYTHONPATH=src .venv/bin/python scripts/independence_probe.py [--tag dev3k]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import second_model_gate as smg  # noqa: E402

COLS = ("pe_gtr", "siglip_final", "pe_probe")  # sources only cached as column files (artifacts/value/{m}_cols_{tag}_{ckpt}.npz)
RAW = ("pe_g", "pe_l", "dino_l", "dino_7b")  # sources with cached embeddings: columns from second_model_gate.columns


def _auc(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a) or not len(b):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([a, b]))) + 1.0
    return (r[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def run(tag: str, ckpt: str) -> None:
    zm = np.load(Path("artifacts/pinpoint_set") / f"zoom_{tag}_{ckpt}.npz")
    logits, km, hyp = smg.head_options(tag, ckpt)
    valid = np.isfinite(logits)
    n, h2 = km.shape
    h = h2 // 2
    head_km = km[np.arange(n), np.where(valid, logits, -np.inf).argmax(1)]
    # per area slot: right (either point < 25 km), the head's area logit (max over unzoomed / zoomed)
    slot = hyp[:, :h]
    has = slot >= 0
    right = np.minimum(km[:, :h], km[:, h:]) < 25
    hl = np.maximum(np.where(valid[:, :h], logits[:, :h], -np.inf), np.where(valid[:, h:], logits[:, h:], -np.inf))
    has &= np.isfinite(hl)
    hw = (head_km >= 25) & (right & has).any(1)
    print(f"{tag}: {n} photos; head {np.mean(head_km < 25):.2%} < 25 km; head wrong with a right area {hw.mean():.1%}", flush=True)

    sources = {"megaloc": np.load(Path("artifacts/agent") / f"megaloc_{tag}_{ckpt}.npz")["x"]}
    for m in COLS:
        f = np.load(smg.OUT / f"{m}_cols_{tag}_{ckpt}.npz")
        assert len(f["x"]) == len(zm["photo"]), m
        sources[m] = f["x"]
    for m in RAW:
        if smg._paths(m, tag, ckpt)[0].exists():
            sources[m] = smg.columns(m, tag, ckpt, zm)

    base = smg.clogit(np.where(valid, logits, 0.0)[..., None], valid, km)
    print(f"  stacked head only {np.mean(base < 25):.2%} < 25 km\n", flush=True)
    print(f"  {'source':13s} {'strength':>8s} {'indep':>6s} {'compl':>6s} {'pick':>6s} {'gain':>7s} {'(SE)':>6s}")
    rows = []
    for name, cols in sources.items():
        s = np.where(has, cols[np.maximum(slot, 0), 0], np.nan)
        both = has.any(1) & (right & has).any(1) & (~right & has).any(1)
        strength = _auc(s[both][(right & has)[both]], s[both][(~right & has)[both]])
        compl = _auc(s[hw][(right & has)[hw]], s[hw][(~right & has)[hw]])
        rho = [_spearman(s[p, has[p]], hl[p, has[p]]) for p in range(n) if has[p].sum() >= 3]
        indep = 1 - np.nanmean(rho)
        pick = np.mean([right[p, np.nanargmax(np.where(has[p], s[p], np.nan))] for p in np.flatnonzero(hw)])
        X = np.concatenate([np.where(valid, logits, 0.0)[..., None], np.where((hyp >= 0)[..., None], cols[np.maximum(hyp, 0)], 0.0)], -1)
        res = smg.clogit(X, valid, km)
        diff = (res < 25).mean(0) - (base < 25).mean(0)
        gain, se = 100 * diff.mean(), 100 * diff.std() / np.sqrt(n)
        rows.append((name, strength, indep, compl, pick, gain))
        print(f"  {name:13s} {strength:8.3f} {indep:6.3f} {compl:6.3f} {100 * pick:5.1f}% {gain:+7.2f} {se:6.2f}", flush=True)
    a = np.array([r[1:] for r in rows], float)
    print(f"\n  across {len(rows)} sources, Spearman of gain with: strength {_spearman(a[:, 4], a[:, 0]):+.2f}, independence "
          f"{_spearman(a[:, 4], a[:, 1]):+.2f}, complement {_spearman(a[:, 4], a[:, 2]):+.2f}, pick {_spearman(a[:, 4], a[:, 3]):+.2f}")
    np.savez(smg.OUT / f"independence_{tag}_{ckpt}.npz", names=[r[0] for r in rows], stats=a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dev3k")
    ap.add_argument("--ckpt", default="setft_own")
    args = ap.parse_args()
    run(args.tag, args.ckpt)


if __name__ == "__main__":
    main()
