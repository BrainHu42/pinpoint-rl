# FLICKR_PLAN.md step 1: can dense same-era Flickr photos at a candidate place tell the right candidate from the look-alike the pipeline
# chose? Worked back from the truth on dev3k's close calls, with YFCC100M (a Creative Commons subset of Flickr, data/yfcc100m/geo.npz) as an
# offline lower bound for what the live Flickr API returns at a point. No API calls: photos come from Flickr's static image CDN.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.spot_audit {shortlist,fetch,report}

"""Close calls: dev3k errors (viz/errors.py tiers, artifacts/viz/traces/dev3k) where a right place is on the table: 1 policy, 2 coin flip,
4 ranked low (a right area; its centre is a pool photo) and 5 in pool (the nearest retrieval candidate within 25 km). Per photo, three points:
  truth  the true GPS (ceiling: the agent never knows it)
  right  the right area's centre / nearest right candidate (what the agent could check)
  wrong  the pipeline's answer
Reference photos per point: the --k nearest YFCC100M photos within --radius km (Flickr accuracy >= 13, no videos), never the query photo or
any photo by its photographer; each flagged as in Pinpoint's index (MP16) or new. Score of a point: best MegaLoc cosine between the query
and its references, after dropping near-duplicates of the query (cosine >= DUP; counted separately).

shortlist: points and their references -> artifacts/spot_audit/shortlist.npz
fetch:     downloads each unique reference once (resumable), MegaLoc-embeds it -> artifacts/spot_audit/megaloc.npz
report:    right vs wrong (and truth vs wrong): paired win rate (AUC), clean wins / losses at a threshold, by number of references, source
           (all / index only / new only) and tier -> artifacts/spot_audit/report.txt"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from .density_probe import GEO, MIN_ACCURACY, _haversine, _index_rows, _mp16_photo_ids

OUT = Path("artifacts/spot_audit")
TRACES = Path("artifacts/viz/traces/dev3k")
QUERIES = Path("artifacts/density_probe/queries.npz")  # dev3k photo ids and photographer uids (YFCC vocabulary)
MEGALOC_Q = Path("artifacts/agent/megaloc_queries_dev3k_setft_own.npy")
CLOSE_TIERS = (1, 2, 4, 5)
KINDS = ("truth", "right", "wrong")
DUP = 0.9  # MegaLoc cosine at which a reference counts as a copy of the query
KS = (5, 10, 25, 50)
CDN_PAUSE = 120.0  # seconds every thread waits after a 403 / 429 from the image CDN, doubling on repeats
MEGALOC_DIM = 8448
THRESH = 0.3  # a "clean" match (lesson 138: covered views matched at 0.47-0.58; random pairs ~0.03)


def _points() -> tuple[list[dict], np.ndarray]:
    errors = json.loads((TRACES / "errors.json").read_text(encoding="utf-8"))["errors"]
    pts, rows = [], []
    for e in errors:
        if e["tier"] not in CLOSE_TIERS:
            continue
        t = json.loads((TRACES / f"{e['i']}.json").read_text(encoding="utf-8"))
        final = t["agent"]["0"]["final"]
        if e.get("right_slot") is not None and e["tier"] != 5:
            right = tuple(t["areas"][e["right_slot"]]["center"])
        else:  # in pool (or a policy fix without a right area slot): the nearest right retrieval candidate
            near = [c for c in t["retrained"]["candidates"] if c["km"] < 25]
            if not near:
                continue
            c = min(near, key=lambda c: c["km"])
            right = (c["lat"], c["lon"])
        for kind, (la, lo) in zip(KINDS, ((t["truth"]["lat"], t["truth"]["lon"]), right, (final["lat"], final["lon"]))):
            pts.append({"i": e["i"], "tier": e["tier"], "kind": kind, "lat": float(la), "lon": float(lo)})
            rows.append(e["i"])
    return pts, np.asarray(rows)


def shortlist(k: int, radius: float) -> None:
    pts, _ = _points()
    q = dict(np.load(QUERIES))
    dev = np.flatnonzero(q["set"] == "dev3k")
    q_pid, q_uid = q["photo_id"][dev], q["uid"][dev]  # photo-set order
    g = np.load(GEO)
    g = {key: g[key] for key in ("photo_id", "uid", "lat", "lon", "video", "accuracy", "server", "secret")}
    sel = np.flatnonzero((g["video"] == 0) & (g["accuracy"] >= MIN_ACCURACY))
    lat, lon = g["lat"][sel], g["lon"][sel]
    size = 0.02
    cell = (np.floor(lat / size).astype(np.int64) + 10000) * 100000 + (np.floor(lon / size).astype(np.int64) + 20000)
    o = np.argsort(cell, kind="stable")
    cell_s = cell[o]
    index_pid = np.sort(_mp16_photo_ids()[_index_rows()])
    refs = {"point": [], "photo_id": [], "server": [], "secret": [], "dist": [], "in_index": []}
    for p_idx, p in enumerate(pts):
        ci, cj = int(np.floor(p["lat"] / size)), int(np.floor(p["lon"] / size))
        si = int(np.ceil(radius / (111 * size))) + 1
        sj = int(np.ceil(radius / (111 * size * max(np.cos(np.radians(p["lat"])), 0.05)))) + 1
        parts = []
        for di in range(-si, si + 1):
            for dj in range(-sj, sj + 1):
                key = (ci + di + 10000) * 100000 + (cj + dj + 20000)
                parts.append(o[np.searchsorted(cell_s, key): np.searchsorted(cell_s, key, "right")])
        idx = sel[np.concatenate(parts)]
        uid = q_uid[p["i"]]
        idx = idx[(g["photo_id"][idx] != q_pid[p["i"]]) & ((g["uid"][idx] != uid) | (uid < 0))]
        d = _haversine(p["lat"], p["lon"], g["lat"][idx], g["lon"][idx])
        keep = d < radius
        idx, d = idx[keep], d[keep]
        top = np.argsort(d, kind="stable")[:k]
        idx, d = idx[top], d[top]
        pid = g["photo_id"][idx]
        j = np.clip(np.searchsorted(index_pid, pid), 0, len(index_pid) - 1)
        refs["point"] += [p_idx] * len(idx)
        refs["photo_id"] += pid.tolist()
        refs["server"] += g["server"][idx].tolist()
        refs["secret"] += g["secret"][idx].tolist()
        refs["dist"] += d.tolist()
        refs["in_index"] += (index_pid[j] == pid).tolist()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "shortlist.npz", **{f"p_{key}": np.asarray([p[key] for p in pts]) for key in ("i", "tier", "kind", "lat", "lon")},
             **{key: np.asarray(v) for key, v in refs.items()}, k=k, radius=radius)
    n = np.bincount(np.asarray(refs["point"], np.int64), minlength=len(pts))
    print(f"{len(pts) // 3} photos, {len(pts)} points; {len(refs['point']):,} references ({len(set(refs['photo_id'])):,} unique), "
          f"in the index {100 * np.mean(refs['in_index']):.0f}%")
    for kind in KINDS:
        m = np.asarray([p["kind"] == kind for p in pts])
        print(f"  {kind}: median {np.median(n[m]):.0f} references, none {100 * (n[m] == 0).mean():.1f}%, full ({k}) {100 * (n[m] >= k).mean():.1f}%")


class Throttled(Exception):
    pass


_pause_until = [0.0]


def _get(args) -> bytes | None:
    """One CDN image: bytes, None if gone (404 / 410 / Flickr's 'unavailable' placeholder). HTTP 403 / 429 is throttling, not absence: all
    threads pause and retry (the CDN answered every request with 403 / 429 for a while after a burst on 2026-10-24)."""

    import urllib.error
    import urllib.request

    pid, server, secret = args
    url = f"https://live.staticflickr.com/{server}/{pid}_{secret.decode() if isinstance(secret, bytes) else secret}.jpg"
    for attempt in range(8):
        wait = _pause_until[0] - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pinpoint-rl research (brainhu42@gmail.com)"}),
                                        timeout=30) as r:
                return None if "photo_unavailable" in r.geturl() else r.read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                return None
            if e.code in (403, 429):
                _pause_until[0] = max(_pause_until[0], time.time() + CDN_PAUSE * 2 ** min(attempt, 4))
                continue
            time.sleep(2 ** attempt)
        except OSError:
            time.sleep(2 ** attempt)
    raise Throttled(url)


def fetch(workers: int, batch: int) -> None:
    import torch

    from .clue_verify import _megaloc_tf
    from .megaloc_gate import _model

    s = np.load(OUT / "shortlist.npz")
    ids, first = np.unique(s["photo_id"], return_index=True)
    server, secret = s["server"][first], s["secret"][first]
    part = OUT / "megaloc_part.npz"
    emb, done = None, 0
    if part.exists():
        p = np.load(part)
        emb, done = p["emb"], int(p["done"])
        print(f"resuming at {done:,}/{len(ids):,}")
    model, tf = _model(), _megaloc_tf()
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for a in range(done, len(ids), batch):
            chunk = range(a, min(a + batch, len(ids)))
            blobs = list(pool.map(_get, [(ids[i], server[i], secret[i]) for i in chunk]))  # Throttled ends the run; rerun resumes
            xs = [tf(b) for b in blobs]
            ok = [n for n, x in enumerate(xs) if x is not None]
            if emb is None:
                emb = np.full((len(ids), MEGALOC_DIM), np.nan, np.float16)
            if ok:
                with torch.inference_mode():
                    emb[[chunk[n] for n in ok]] = model(torch.stack([xs[n] for n in ok]).cuda()).float().cpu().numpy()
            if (a - done) // batch % 20 == 0 or chunk[-1] == len(ids) - 1:
                np.savez(part, emb=emb, done=chunk[-1] + 1)
                rate = (chunk[-1] + 1 - done) / (time.time() - t0)
                print(f"  {chunk[-1] + 1:,}/{len(ids):,}, {rate:.0f}/s, gone {100 * np.isnan(emb[:chunk[-1] + 1, 0]).mean():.1f}%", flush=True)
    np.savez(OUT / "megaloc.npz", ids=ids, emb=emb)
    part.unlink(missing_ok=True)
    print(f"embedded {np.isfinite(emb[:, 0]).sum():,}/{len(ids):,} ({time.time() - t0:.0f}s)")


def _paired(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) over pairs (ties half); points without references score -1."""

    return float(np.mean((a > b) + 0.5 * (a == b)))


def report() -> None:
    s = dict(np.load(OUT / "shortlist.npz"))
    if (OUT / "megaloc.npz").exists():
        m = np.load(OUT / "megaloc.npz")
        ids, emb, partial = m["ids"], m["emb"], ""
    else:  # a fetch in progress: references not fetched yet count as missing (fewer per point, alike for every kind of point)
        m = np.load(OUT / "megaloc_part.npz")
        ids, emb = np.unique(s["photo_id"]), m["emb"]
        partial = f" PARTIAL: {int(m['done']):,} of {len(ids):,} references fetched;"
    q = np.load(MEGALOC_Q).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    e = emb.astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    pos = np.searchsorted(ids, s["photo_id"])
    ref_emb = e[pos]
    sim = np.einsum("nd,nd->n", ref_emb, q[s["p_i"][s["point"]]])  # NaN where the photo is gone
    rank = np.zeros(len(s["point"]), np.int64)  # order within its point (nearest first)
    for p in np.unique(s["point"]):
        r = np.flatnonzero(s["point"] == p)
        rank[r] = np.arange(len(r))
    ok = np.isfinite(sim)
    dup = ok & (sim >= DUP)
    n_pts = len(s["p_i"])
    lines = [f"Spot audit, dev3k close calls (tiers {CLOSE_TIERS}): {n_pts // 3} photos;{partial} references k <= {int(s['k'])} within "
             f"{float(s['radius']):g} km; fetched {100 * ok.mean():.1f}%; near-duplicates of the query (>= {DUP}) dropped: "
             f"{int(dup.sum())} ({len(np.unique(s['point'][dup]))} points)"]

    def score(k: int, source: str) -> np.ndarray:
        use = ok & ~dup & (rank < k)
        if source == "index":
            use &= s["in_index"].astype(bool)
        elif source == "new":
            use &= ~s["in_index"].astype(bool)
        best = np.full(n_pts, -1.0)
        np.maximum.at(best, s["point"][use], sim[use])
        return best

    kind = s["p_kind"]
    tiers = s["p_tier"][kind == "truth"]
    lines.append(f"  P(score at A > score at wrong pick), paired; clean = best >= {THRESH} and above the other point's best")
    lines.append(f"  {'k':>3} {'source':<6} {'right>wrong':>11} {'truth>wrong':>11} {'right clean win':>15} {'wrong clean win':>15} "
                 f"{'right >= thr':>12} {'wrong >= thr':>12}")
    for k in KS:
        for source in ("all", "index", "new"):
            b = score(k, source)
            t, r, w = b[kind == "truth"], b[kind == "right"], b[kind == "wrong"]
            lines.append(f"  {k:>3} {source:<6} {_paired(r, w):11.3f} {_paired(t, w):11.3f} {100 * ((r >= THRESH) & (r > w)).mean():14.1f}% "
                         f"{100 * ((w >= THRESH) & (w > r)).mean():14.1f}% {100 * (r >= THRESH).mean():11.1f}% {100 * (w >= THRESH).mean():11.1f}%")
    b = score(max(KS), "all")
    r, w = b[kind == "right"], b[kind == "wrong"]
    lines.append(f"  by tier (k {max(KS)}, all): tier n right>wrong right-clean-win wrong-clean-win")
    for tier in CLOSE_TIERS:
        mt = tiers == tier
        lines.append(f"    {tier} {int(mt.sum()):4d} {_paired(r[mt], w[mt]):.3f} {100 * ((r[mt] >= THRESH) & (r[mt] > w[mt])).mean():5.1f}% "
                     f"{100 * ((w[mt] >= THRESH) & (w[mt] > r[mt])).mean():5.1f}%")
    n_refs = np.bincount(s["point"][ok & ~dup], minlength=n_pts)
    lines.append(f"  references kept per point, median: truth {np.median(n_refs[kind == 'truth']):.0f}, right "
                 f"{np.median(n_refs[kind == 'right']):.0f}, wrong {np.median(n_refs[kind == 'wrong']):.0f}")
    for lo_, hi_ in ((0, 10), (10, 30), (30, 10**9)):
        mr = (n_refs[kind == "right"] >= lo_) & (n_refs[kind == "right"] < hi_)
        lines.append(f"    right point with {lo_}-{hi_ if hi_ < 10**9 else '':} refs: {int(mr.sum())} photos, right>wrong {_paired(r[mr], w[mr]):.3f}")
    # The wrong pick sits in denser places, which tilts best-of-k towards it: compare with the same number of kept references at both points
    # (the nearest k of each), and within a tighter radius.
    keep = ok & ~dup
    per_point = [[] for _ in range(n_pts)]
    for p, v, d in zip(s["point"][keep], sim[keep], s["dist"][keep]):
        per_point[p].append((d, v))
    per_point = [sorted(x) for x in per_point]
    lines.append("  equal reference counts (both points >= k kept; best of each point's nearest k), and within RADIUS km:")
    for radius in (float(s["radius"]), 0.25):
        for k in (5, 10, 25):
            rr, ww, tt = [], [], []
            for j in np.flatnonzero(kind == "truth"):  # points come in truth, right, wrong order per photo
                t_, r_, w_ = ([v for d, v in per_point[j + x] if d < radius][:k] for x in range(3))
                if len(r_) >= k and len(w_) >= k:
                    rr.append(max(r_)); ww.append(max(w_)); tt.append(max(t_) if len(t_) >= k else np.nan)
            if not rr:
                continue
            rr, ww, tt = map(np.asarray, (rr, ww, tt))
            mt = np.isfinite(tt)
            lines.append(f"    radius {radius:g} km, k {k:>2}: n {len(rr):3d}  right>wrong {_paired(rr, ww):.3f}  truth>wrong "
                         f"{_paired(tt[mt], ww[mt]) if mt.any() else float('nan'):.3f} (n {int(mt.sum())})  >= {THRESH}: right "
                         f"{100 * (rr >= THRESH).mean():.1f}% wrong {100 * (ww >= THRESH).mean():.1f}%  median best right {np.median(rr):.3f} "
                         f"wrong {np.median(ww):.3f}")
    text = "\n".join(lines)
    print(text)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("shortlist", "fetch", "report"))
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args(argv)
    if args.cmd == "shortlist":
        shortlist(args.k, args.radius)
    elif args.cmd == "fetch":
        fetch(args.workers, args.batch)
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
