"""Map checks (features_near) on the merged pipeline's areas: can the features the 4B clue reader sees in a photo (observe.py 'features', e.g.
church, river, tram stop) tell the right area from wrong ones? Ceiling before any agent: per photo, each observed feature is looked up
(maps_search.FeatureIndex, OSM boxes) within r km of the true location, of the right area's restricted top-1 and of every wrong area's
restricted top-1 (Pinpoint areas, source 0; wrong = more than 25 km from the truth). Control: the same lookups for features the 4B did not
report, so the gap shows what the observation adds beyond how densely a place is mapped.

  python -m geo_search_env.experiment.map_checks ceiling [--tags dev3k train] [--radii 1 3]
  python -m geo_search_env.experiment.map_checks features [--tags dev3k train]   (per-area columns for hypothesis_agent.load(extra=...))

features: for every area of the merged pipeline (Pinpoint areas and clue places, rows of hypothesis_agent's feature / zoom files), checked at
its restricted top-1: 'checks' = for the WORDS most often observed on train, +1 present / -1 absent / 0 not observed within 1 and 3 km, plus
'summary' = the share of observed features present and their number; 'density' = log OSM objects within 1 and 3 km
-> artifacts/agent/mapcheck_{tag}_setft_a.npz (x, columns, groups).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .maps_search import FeatureIndex
from .strategy_search import _haversine_km

OUT = Path("artifacts/agent")
CKPT_TAG = "setft_a"


def _areas(tag: str) -> dict:
    """Per photo: truth, observed feature words, and per Pinpoint area its rank and restricted top-1."""

    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
    observed = json.loads((OUT / f"observe_{tag}_4b.json").read_text(encoding="utf-8"))
    z = np.load(OUT / f"hypotheses_{tag}_{CKPT_TAG}.npz")
    keep = z["source"] == 0
    scores = np.where(z["rows"] >= 0, z["scores"], -np.inf)[keep]
    top = z["latlon"][keep][np.arange(keep.sum()), scores.argmax(1)].astype(np.float64)
    truth = np.asarray([p["truth"] for p in photos], np.float64)
    words = [[w for w in ((observed.get(p["image_id"]) or {}).get("parsed") or {}).get("features") or []] for p in photos]
    return {"truth": truth, "words": words, "photo": z["photo"][keep], "rank": z["rank"][keep], "top": top}


def ceiling(tags: list[str], radii: list[float]) -> None:
    index = FeatureIndex()
    vocab = set(index.words)
    for tag in tags:
        a = _areas(tag)
        truth, photo = a["truth"], a["photo"]
        km = np.asarray([_haversine_km(*truth[p], a["top"][[i]])[0] for i, p in enumerate(photo)])
        right = km < 25
        seen = [sorted({w for w in ws if w in vocab}) for ws in a["words"]]
        n_obs = sum(bool(s) for s in seen)
        print(f"\n{tag}: {len(truth)} photos, {n_obs} ({n_obs / len(truth):.0%}) with an observed feature the map tool knows; "
              f"{len(photo)} Pinpoint areas, {right.mean():.0%} right", flush=True)
        words = sorted(vocab & {w for s in seen for w in s})
        col = {w: j for j, w in enumerate(words)}
        for r in radii:
            at_truth, _ = index.near(truth[:, 0], truth[:, 1], r, words)
            at_area, _ = index.near(a["top"][:, 0], a["top"][:, 1], r, words)
            dens_truth, dens_area = index.density(truth[:, 0], truth[:, 1], r), index.density(a["top"][:, 0], a["top"][:, 1], r)
            obs = np.zeros((len(truth), len(words)), bool)
            for i, s in enumerate(seen):
                obs[i, [col[w] for w in s]] = True
            pres_t, pres_a = at_truth > 0, at_area > 0
            print(f"\n  r = {r:g} km: feature present near ... (observed in the photo / not observed, as control)")
            print(f"  {'feature':16s} {'n obs':>6s}  {'truth':>13s}  {'right area':>13s}  {'wrong area':>13s}")
            rows = []
            for w, j in col.items():
                o = obs[:, j]
                if o.sum() < 20:
                    continue
                oa = o[photo]
                cells = [(pres_t[o, j].mean(), pres_t[~o, j].mean()), (pres_a[oa & right, j].mean(), pres_a[~oa & right, j].mean()),
                         (pres_a[oa & ~right, j].mean(), pres_a[~oa & ~right, j].mean())]
                rows.append((o.sum(), w, cells))
            for n, w, cells in sorted(rows, reverse=True)[:25]:
                print(f"  {w:16s} {n:6d}  " + "  ".join(f"{100 * x:5.1f} / {100 * y:5.1f}" for x, y in cells))
            # elimination ceiling: photos whose top Pinpoint area is wrong while another area is right
            fail = np.zeros(len(photo), bool)  # area's top-1 lacks a feature the photo shows
            for i, p in enumerate(photo):
                if seen[p]:
                    fail[i] = not pres_a[i, [col[w] for w in seen[p]]].all()
            helps = hurts = n_target = n_obs_t = 0
            for p in range(len(truth)):
                idx = np.flatnonzero(photo == p)
                if not len(idx) or not seen[p]:
                    continue
                first = idx[np.argmin(a["rank"][idx])]
                good = idx[right[idx]]
                if right[first] or not len(good):
                    continue
                n_target += 1
                best_good = good[np.argmin(a["rank"][good])]
                helps += bool(fail[first] and not fail[best_good])
                hurts += bool(fail[best_good] and not fail[first])
            truth_fail = np.mean([not pres_t[p, [col[w] for w in seen[p]]].all() for p in range(len(truth)) if seen[p]])
            print(f"  truth lacks an observed feature: {truth_fail:.0%} of photos with one (the check's false-alarm rate at the true spot)")
            print(f"  right area's top-1 fails a check: {fail[right & np.asarray([bool(seen[p]) for p in photo])].mean():.0%};"
                  f" wrong area's: {fail[~right & np.asarray([bool(seen[p]) for p in photo])].mean():.0%}")
            print(f"  target photos (top area wrong, a right area exists, a feature observed): {n_target} ({n_target / len(truth):.1%} of photos); "
                  f"check separates them right {helps} ({helps / len(truth):.1%}), wrong {hurts} ({hurts / len(truth):.1%})")
            print(f"  OSM objects within r (density): truth median {np.median(dens_truth):.0f}, right areas {np.median(dens_area[right]):.0f}, "
                  f"wrong areas {np.median(dens_area[~right]):.0f}", flush=True)


RADII = (1.0, 3.0)
N_WORDS = 30


def features(tags: list[str]) -> None:
    from .hypothesis_agent import _paths

    index = FeatureIndex()
    vocab = set(index.words)
    counts: dict[str, int] = {}
    for ws in _areas("train")["words"]:
        for w in set(ws) & vocab:
            counts[w] = counts.get(w, 0) + 1
    words = sorted(counts, key=lambda w: (-counts[w], w))[:N_WORDS]
    print(f"checked words (most observed on train): {', '.join(words)}", flush=True)
    for tag in tags:
        a = _areas(tag)
        zm = np.load(_paths(tag, CKPT_TAG)[1])
        s = np.where(zm["rows"] >= 0, zm["scores"], -np.inf)
        point = zm["latlon"][np.arange(len(s)), s.argmax(1)].astype(np.float64)
        point = np.where(np.isfinite(s.max(1))[:, None], point, zm["center"])
        obs = np.zeros((len(a["truth"]), len(words)), np.float32)
        obs_all = np.zeros(len(a["truth"]), np.float32)
        for i, ws in enumerate(a["words"]):
            ws = set(ws) & vocab
            obs_all[i] = len(ws)
            obs[i] = [w in ws for w in words]
        o = obs[zm["photo"]]
        cols, names = [], []
        for r in RADII:
            count, _ = index.near(point[:, 0], point[:, 1], r, words)
            cols.append(o * np.where(count > 0, 1.0, -1.0))
            names += [f"{w}@{r:g}" for w in words]
            # share of all observed (vocabulary) features present, not only the WORDS columns
            share = np.zeros(len(point), np.float32)
            for p in np.flatnonzero(obs_all):
                ws = sorted(set(a["words"][p]) & vocab)
                idx = np.flatnonzero(zm["photo"] == p)
                c, _ = index.near(point[idx, 0], point[idx, 1], r, ws)
                share[idx] = (c > 0).mean(1)
            cols.append(share[:, None])
            names.append(f"share@{r:g}")
        cols.append(np.log1p(obs_all[zm["photo"]])[:, None])
        names.append("n_obs")
        for r in RADII:
            cols.append(np.log1p(index.density(point[:, 0], point[:, 1], r)).astype(np.float32)[:, None])
            names.append(f"density@{r:g}")
        x = np.concatenate(cols, 1).astype(np.float32)
        groups = np.asarray(["density" if n.startswith("density") else "summary" if n.startswith(("share", "n_obs")) else "checks" for n in names])
        out = OUT / f"mapcheck_{tag}_{CKPT_TAG}.npz"
        np.savez(out, x=x, columns=np.asarray(names), groups=groups, photo=zm["photo"], source=zm["source"])
        print(f"{tag}: {len(x)} areas x {x.shape[1]} columns -> {out}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("ceiling", "features"))
    parser.add_argument("--tags", nargs="+", default=["dev3k", "train"])
    parser.add_argument("--radii", nargs="+", type=float, default=[1.0, 3.0])
    args = parser.parse_args(argv)
    if args.cmd == "ceiling":
        ceiling(args.tags, args.radii)
    else:
        features(args.tags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
