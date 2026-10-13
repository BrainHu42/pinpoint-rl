"""Names inside areas (maps_search over OSM): are the names a photo shows or evokes found near a candidate area? A name read in the photo
('Calle Mayor', a shop sign) is useless worldwide but nearly decisive inside one 25 km area, and retrieval can't see it.

Names per photo, the clue reader's outputs: 'read' = observe.py text + place_names (what is written in the photo), 'recall' = the 4B's
search queries (queries_{tag}_4b.json; landmarks it recognises). Kept: 3-60 characters with a letter.

lookup:   every distinct name once, phrase match over all OSM name variants (maps_search osm.db, FTS5), in parallel processes. Names with more
          than MAX_HITS matches are only counted (too common; the rows are in spatial order, so a capped fetch would be regionally biased)
          -> artifacts/agent/name_hits.npz (names, count, coordinates by offset).
features: per area (rows of hypothesis_agent's feature / zoom files), at its restricted top-1, for read and recalled names separately:
          names matched within 1 and 5 km, the specificity of the rarest matched name (-log worldwide matches), the share of the photo's
          found names matched within 5 km -> artifacts/agent/namecheck_{tag}_setft_a.npz (groups names_read, names_recall; hypothesis_agent.load(extra=...)).
ceiling:  on photos whose top Pinpoint area is wrong while another is right, how often a name matches near the right area's top-1 and
          not near the top area's, and the reverse.

  python -m geo_search_env.experiment.name_checks lookup [--tags dev3k train] [--jobs 4]
  python -m geo_search_env.experiment.name_checks features [--tags dev3k train]
  python -m geo_search_env.experiment.name_checks ceiling [--tags dev3k train]
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

OUT = Path("artifacts/agent")
CKPT_TAG = "setft_a"
HITS = OUT / "name_hits.npz"
MAX_HITS = 20_000
KINDS = ("read", "recall")
RADII = (1.0, 5.0)


def photo_names(tag: str) -> list[dict[str, list[str]]]:
    """Per photo of {tag}_setft_a (photos.json order): {'read': [...], 'recall': [...]}, lower-cased, de-duplicated."""

    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
    recall = json.loads((OUT / f"queries_{tag}_4b.json").read_text(encoding="utf-8"))
    observed = json.loads((OUT / f"observe_{tag}_4b.json").read_text(encoding="utf-8"))
    keep = lambda xs: sorted({x.strip().lower() for x in xs if isinstance(x, str) and 3 <= len(x.strip()) <= 60 and any(c.isalpha() for c in x)})
    out = []
    for p in photos:
        parsed = (observed.get(p["image_id"]) or {}).get("parsed") or {}
        out.append({"read": keep((parsed.get("text") or []) + (parsed.get("place_names") or [])), "recall": keep(recall.get(p["image_id"]) or [])})
    return out


_con = None


def _lookup(name: str) -> tuple[str, int, np.ndarray]:
    import sqlite3

    from .maps_search import _osm_path, _phrase

    global _con
    if _con is None:
        _con = sqlite3.connect(f"file:{_osm_path()}?mode=ro", uri=True)
    phrase = _phrase(name)
    if phrase is None:
        return name, 0, np.empty((0, 2), np.float32)
    n = _con.execute("SELECT count(*) FROM (SELECT rowid FROM names WHERE names MATCH ? LIMIT ?)", (phrase, MAX_HITS + 1)).fetchone()[0]
    if n == 0 or n > MAX_HITS:
        return name, n, np.empty((0, 2), np.float32)
    rows = _con.execute("SELECT p.lat, p.lon FROM names JOIN places p ON p.rowid = names.rowid WHERE names MATCH ?", (phrase,)).fetchall()
    return name, len(rows), np.asarray(rows, np.float32).reshape(-1, 2)


def lookup(tags: list[str], jobs: int) -> None:
    names = sorted({x for tag in tags for ph in photo_names(tag) for kind in KINDS for x in ph[kind]})
    print(f"{len(names):,} distinct names; {jobs} processes", flush=True)
    found: dict[str, tuple[int, np.ndarray]] = {}
    with Pool(jobs) as pool:
        for i, (name, n, xy) in enumerate(pool.imap_unordered(_lookup, names, chunksize=64)):
            found[name] = (n, xy)
            if (i + 1) % 10_000 == 0:
                print(f"  {i + 1:,} names, {sum(v[0] > 0 for v in found.values()):,} found", flush=True)
    order = sorted(found)
    lengths = np.asarray([len(found[k][1]) for k in order])
    np.savez(HITS, names=np.asarray(order), count=np.asarray([found[k][0] for k in order]), offset=np.concatenate([[0], np.cumsum(lengths)]),
             xy=np.concatenate([found[k][1] for k in order]) if len(order) else np.empty((0, 2), np.float32))
    n = np.asarray([found[k][0] for k in order])
    print(f"-> {HITS}: found {np.mean(n > 0):.0%} of names ({np.mean(n > MAX_HITS):.1%} too common), {lengths.sum():,} coordinates", flush=True)


class Hits:
    def __init__(self) -> None:
        z = np.load(HITS)
        self.index = {k: i for i, k in enumerate(z["names"].tolist())}
        self.count, self.offset, self.xy = z["count"], z["offset"], z["xy"]

    def get(self, name: str) -> tuple[int, np.ndarray]:
        i = self.index.get(name)
        if i is None:
            return 0, np.empty((0, 2), np.float32)
        return int(self.count[i]), self.xy[self.offset[i]: self.offset[i + 1]]


def _area_points(tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """photo, source, restricted top-1 (centre if the search came back empty) of every area row (hypothesis_agent's zoom file)."""

    from .hypothesis_agent import _paths

    zm = np.load(_paths(tag, CKPT_TAG)[1])
    s = np.where(zm["rows"] >= 0, zm["scores"], -np.inf)
    point = zm["latlon"][np.arange(len(s)), s.argmax(1)].astype(np.float64)
    return zm["photo"], zm["source"], np.where(np.isfinite(s.max(1))[:, None], point, zm["center"])


def _km(points: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """[points, matches] great-circle km."""

    p1, p2 = np.radians(points[:, :1]), np.radians(xy[None, :, 0].astype(np.float64))
    dl = np.radians(xy[None, :, 1].astype(np.float64) - points[:, 1:])
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def area_matches(tag: str, hits: Hits) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Per area row: for each kind, the nearest-match km of each of the photo's names (as columns of the features)."""

    names = photo_names(tag)
    photo, source, point = _area_points(tag)
    cols = {f"{k}_{s}": np.zeros(len(photo), np.float32) for k in KINDS for s in ("n1", "n5", "spec", "share5")}
    by_photo: dict[int, list[int]] = {}
    for i, p in enumerate(photo):
        by_photo.setdefault(int(p), []).append(i)
    for p, idx in by_photo.items():
        idx = np.asarray(idx)
        for kind in KINDS:
            found = [(n, xy) for n, xy in (hits.get(x) for x in names[p][kind]) if len(xy)]
            if not found:
                continue
            near = np.stack([_km(point[idx], xy).min(1) for _, xy in found], 1)  # [areas, names]
            spec = np.asarray([-np.log(n) for n, _ in found])
            cols[f"{kind}_n1"][idx] = (near <= RADII[0]).sum(1)
            cols[f"{kind}_n5"][idx] = (near <= RADII[1]).sum(1)
            cols[f"{kind}_spec"][idx] = np.where(near <= RADII[1], spec[None] + np.log(MAX_HITS) + 1, 0).max(1)  # > 0 when matched
            cols[f"{kind}_share5"][idx] = (near <= RADII[1]).mean(1)
    return cols, photo, source


def features(tags: list[str]) -> None:
    hits = Hits()
    for tag in tags:
        cols, photo, source = area_matches(tag, hits)
        names = list(cols)
        x = np.stack([cols[k] for k in names], 1).astype(np.float32)
        x[:, [i for i, k in enumerate(names) if k.endswith(("_n1", "_n5"))]] = np.log1p(x[:, [i for i, k in enumerate(names) if k.endswith(("_n1", "_n5"))]])
        out = OUT / f"namecheck_{tag}_{CKPT_TAG}.npz"
        np.savez(out, x=x, columns=np.asarray(names), groups=np.asarray([f"names_{k.split('_')[0]}" for k in names]), photo=photo, source=source)
        print(f"{tag}: {len(x)} areas x {x.shape[1]} columns -> {out}; areas with a name within 5 km: "
              f"read {np.mean(cols['read_n5'] > 0):.1%}, recall {np.mean(cols['recall_n5'] > 0):.1%}", flush=True)


def ceiling(tags: list[str]) -> None:
    from .strategy_search import _haversine_km

    hits = Hits()
    for tag in tags:
        cols, photo, source = area_matches(tag, hits)
        photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
        truth = np.asarray([p["truth"] for p in photos], np.float64)
        from .hypothesis_agent import _paths

        _, _, point = _area_points(tag)
        rank = np.load(_paths(tag, CKPT_TAG)[1])["rank"]
        km = np.asarray([_haversine_km(*truth[p], point[[i]])[0] for i, p in enumerate(photo)])
        right = km < 25
        names = photo_names(tag)
        print(f"\n{tag}: {len(truth)} photos; with a read name {np.mean([bool(n['read']) for n in names]):.0%}, a recalled name "
              f"{np.mean([bool(n['recall']) for n in names]):.0%}")
        for kind in KINDS + ("any",):
            m5 = (cols["read_n5"] > 0) | (cols["recall_n5"] > 0) if kind == "any" else cols[f"{kind}_n5"] > 0
            pin = source == 0
            print(f"  {kind:6s}: a name within 5 km of the area's top-1: right areas {m5[pin & right].mean():.1%}, wrong areas {m5[pin & ~right].mean():.1%}")
            helps = hurts = targets = 0
            for p in range(len(truth)):
                idx = np.flatnonzero((photo == p) & (source == 0))
                if not len(idx):
                    continue
                first = idx[np.argmin(rank[idx])]
                good = idx[right[idx]]
                if right[first] or not len(good):
                    continue
                targets += 1
                helps += bool(m5[good].any() and not m5[first])
                hurts += bool(m5[first] and not m5[good].any())
            print(f"          targets (top area wrong, a right area exists) {targets} ({targets / len(truth):.1%}): a name near a right area and not the "
                  f"top one {helps} ({helps / len(truth):.1%} of photos), the reverse {hurts} ({hurts / len(truth):.1%})", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("lookup", "features", "ceiling"))
    parser.add_argument("--tags", nargs="+", default=["dev3k", "train"])
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    if args.cmd == "lookup":
        lookup(args.tags, args.jobs)
    elif args.cmd == "features":
        features(args.tags)
    else:
        ceiling(args.tags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
