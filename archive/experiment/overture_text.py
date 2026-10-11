# Does text read from a photo match named places near the pipeline's candidates? Overture Maps places (81M POIs) as the name index.
# Usage: ~/.venvs/sft/bin/python -m geo_search_env.experiment.overture_text build      (PYTHONPATH=src; ~1 h, ~10 GB)
#        .venv/bin/python -m geo_search_env.experiment.overture_text screen --variant text-9b

"""Candidate-conditioned name check.

build:  /data/pinpoint/overture/places/*.parquet (release 2026-09-23.1, theme=places) -> SQLite `places` (name, lat, lon, confidence,
        category) with an FTS5 index over the names.
screen: for every dev / val photo with transcribed text (`places_<variant>.json`), find the places whose names contain all the
        tokens of a string; a pooled candidate is *supported* if a match lies within RADIUS_KM of it. Reports, on photos with
        strings, how often the right candidates vs the wrong ones are supported (and AUC), the "choosing" photos, the global
        evidence of specific strings (few matches), and the one-parameter combiner top-1 (w fitted on dev), against the reranker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
from typing import Any, Sequence

import numpy as np

from .query_evidence import ROOT
from .stage1_eval import _bootstrap, _suffix
from .wiki_backend import _km


PLACES = Path("/data/pinpoint/overture/places")
DB = Path("/data/pinpoint/overture/places.sqlite")
RADIUS_KM = 3.0
MAX_MATCHES = 300_000  # a string matching more places than this ("hotel") says nothing about where
SPECIFIC_MAX = 50  # a string with at most this many matches is "specific": its places are global evidence
STOP = {"www", "com", "http", "https", "net", "org", "the", "and", "de", "la", "le", "of"}
WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens without accents; URL and article words dropped."""

    folded = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return [t for t in re.findall(r"[^\W_]+", folded) if t not in STOP]


def usable(toks: list[str]) -> bool:
    """At least one alphabetic token of three letters or more (drops "73", "A10", "AF 593")."""

    return any(len(t) >= 3 and not t.isdigit() for t in toks)


def build() -> None:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    DB.parent.mkdir(parents=True, exist_ok=True)
    DB.unlink(missing_ok=True)
    db = sqlite3.connect(DB)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("CREATE TABLE places (id INTEGER PRIMARY KEY, name TEXT, lat REAL, lon REAL, conf REAL, cat TEXT)")
    total = 0
    for path in sorted(PLACES.glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=250_000, columns=["names", "bbox", "confidence", "basic_category", "operating_status"]):
            name = pc.struct_field(batch.column("names"), "primary")
            status = batch.column("operating_status")  # null for ~99.5% of rows (unknown); `or_`/`and_` would propagate it, so use Kleene logic
            ok = pc.and_kleene(pc.is_valid(name), pc.or_kleene(pc.is_null(status), pc.equal(status, "open")))
            ok = pc.fill_null(ok, False)
            batch = batch.filter(ok)
            if not batch.num_rows:
                continue
            rows = zip(
                pc.struct_field(batch.column("names"), "primary").to_pylist(), pc.struct_field(batch.column("bbox"), "ymin").to_pylist(),
                pc.struct_field(batch.column("bbox"), "xmin").to_pylist(), batch.column("confidence").to_pylist(), batch.column("basic_category").to_pylist(),
            )
            db.executemany("INSERT INTO places (name, lat, lon, conf, cat) VALUES (?, ?, ?, ?, ?)", rows)
            total += batch.num_rows
        db.commit()
        print(f"{path.name}: {total} places", flush=True)
    print("building the full-text index", flush=True)
    db.execute("CREATE VIRTUAL TABLE places_fts USING fts5(name, content='places', content_rowid='id', tokenize='unicode61 remove_diacritics 2')")
    db.execute("INSERT INTO places_fts (rowid, name) SELECT id, name FROM places")
    db.commit()
    print(f"{total} places indexed in {DB}")


def matches(db: sqlite3.Connection, text: str) -> tuple[np.ndarray, list[str]] | None:
    """(lat, lon, conf) of the places whose names contain every token of `text`, and a few of their names; None if unusable or too generic."""

    toks = tokens(text)
    if not usable(toks):
        return None
    query = " AND ".join(f'"{t}"' for t in dict.fromkeys(toks))
    rows = db.execute("SELECT p.lat, p.lon, p.conf, p.name FROM places_fts f JOIN places p ON p.id = f.rowid WHERE places_fts MATCH ? LIMIT ?",
                      (query, MAX_MATCHES + 1)).fetchall()
    if len(rows) > MAX_MATCHES:
        return None
    return np.asarray([r[:3] for r in rows], dtype=np.float64).reshape(-1, 3), [r[3] for r in rows[:3]]


def _auc(score: np.ndarray, ok: np.ndarray) -> float:
    if not ok.any() or ok.all():
        return float("nan")
    a, b = score[ok][:, None], score[~ok][None, :]
    return float((a > b).mean() + 0.5 * (a == b).mean())


def screen(variant: str, min_conf: float = 0.0) -> None:
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    weights: dict[str, float] = {}
    out: dict[str, Any] = {}
    for tag in ("dev", "val"):
        dev = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
        strings = json.loads((ROOT / tag / f"places{_suffix(variant)}.json").read_text(encoding="utf-8"))
        n = len(dev)
        has_text = np.zeros(n, bool)
        found_any, specific_any = np.zeros(n, bool), np.zeros(n, bool)
        pool_hit, top1_hit, global_hit = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
        supports: list[np.ndarray] = []
        dist: list[np.ndarray] = []
        examples: list[str] = []
        for m, e in enumerate(dev):
            pool, truth = np.asarray(e["pool"]), e["truth"]
            d_pool = _km(pool, *truth)
            dist.append(d_pool)
            pool_hit[m], top1_hit[m] = d_pool.min() < 25, d_pool[0] < 25
            support = np.zeros(len(pool))
            global_pts = []
            for s in strings[e["image_id"]]["queries"]:
                has_text[m] = True
                found = matches(db, s)
                if found is None:
                    continue
                pts, names = found
                pts = pts[pts[:, 2] >= min_conf]
                if not len(pts):
                    continue
                found_any[m] = True
                for c in range(len(pool)):
                    support[c] += float((_km(pts[:, :2], *pool[c]) < RADIUS_KM).any())
                if len(pts) <= SPECIFIC_MAX:
                    specific_any[m] = True
                    global_pts.append(pts[:, :2])
                    if len(examples) < 12 and _km(pts[:, :2], *truth).min() < 25:
                        examples.append(f"{s!r} -> {names[0]!r} ({len(pts)} matches, {_km(pts[:, :2], *truth).min():.1f} km from truth; pool best {d_pool.min():.0f} km)")
            supports.append(support)
            if global_pts:
                global_hit[m] = _km(np.vstack(global_pts), *truth).min() < 25
        applies = has_text
        choosing = pool_hit & ~top1_hit
        correct = [d < 25 for d in dist]
        sup_any = np.asarray([s.sum() > 0 for s in supports])

        def rate(mask: np.ndarray, of: np.ndarray) -> float:
            return float(of[mask].mean()) if mask.any() else float("nan")

        on_right = [(s[c] > 0).mean() for s, c in zip(supports, correct) if c.any() and s.sum() > 0]
        on_wrong = [(s[~c] > 0).mean() for s, c in zip(supports, correct) if c.any() and (~c).any() and s.sum() > 0]
        auc = np.nanmean([_auc(s, c) for s, c in zip(supports, correct) if c.any() and s.sum() > 0] or [np.nan])
        sup_right = np.asarray([bool(c.any() and (s[c] > 0).any()) for s, c in zip(supports, correct)])
        sup_top1 = np.asarray([bool(s[0] > 0) for s in supports])

        def top1(w: float) -> np.ndarray:
            return np.asarray([d[int(np.argmax(-np.arange(len(d)) + w * s))] < 25 for d, s in zip(dist, supports)])

        if "w" not in weights:
            weights["w"] = max(WEIGHTS, key=lambda w: (top1(w).mean(), -w))
        base, withw = top1(0.0), top1(weights["w"])
        delta = _bootstrap(withw.astype(float) - base)
        report = {
            "photos with text": int(applies.sum()), "with a name match": int((found_any & applies).sum()), "with a specific match": int(specific_any.sum()),
            "support>0 on correct / wrong candidates (photos with any support)": [float(np.mean(on_right)), float(np.mean(on_wrong))], "AUC": float(auc),
            "choosing & supported: n": int((choosing & sup_any).sum()),
            "choosing & supported: backs right": rate(choosing & sup_any, sup_right), "choosing & supported: backs wrong top-1": rate(choosing & sup_any, sup_top1),
            "global specific evidence hit | pool hit": rate(specific_any & pool_hit, global_hit), "global specific evidence hit | pool miss": rate(specific_any & ~pool_hit, global_hit),
            "misses recovered (specific)": int((global_hit & ~pool_hit).sum()), "w": weights["w"], "combiner top-1 <25 km": float(withw.mean()),
            "reranker top-1 <25 km": float(base.mean()), "change [95% CI]": list(delta), "fixed": int((withw & ~base).sum()), "broke": int((~withw & base).sum()),
        }
        out[tag] = report
        print(f"\n{tag} / {variant}: radius {RADIUS_KM:g} km, min confidence {min_conf:g}")
        print(f"  photos with text {report['photos with text']}; a name match for {report['with a name match']}; a specific match (<= {SPECIFIC_MAX} places) for {report['with a specific match']}")
        r = report["support>0 on correct / wrong candidates (photos with any support)"]
        print(f"  on photos with any support: a correct candidate is supported {r[0]:.0%} vs a wrong one {r[1]:.0%} of the time; AUC {report['AUC']:.3f}")
        print(f"  choosing photos with support (n={report['choosing & supported: n']}): backs a right candidate {report['choosing & supported: backs right']:.0%}, backs the wrong top-1 {report['choosing & supported: backs wrong top-1']:.0%}")
        print(f"  specific strings as global evidence within 25 km of truth: {report['global specific evidence hit | pool hit']:.0%} when the pool holds the answer, "
              f"{report['global specific evidence hit | pool miss']:.0%} when it misses (recovers {report['misses recovered (specific)']})")
        print(f"  combiner (w={weights['w']:g}, fitted on dev) top-1 <25 km {report['combiner top-1 <25 km']:.1%} vs reranker {report['reranker top-1 <25 km']:.1%}: "
              f"{delta[0]:+.1%} [{delta[1]:+.1%}, {delta[2]:+.1%}], fixed/broke {report['fixed']}/{report['broke']}")
        for line in examples[:8]:
            print("    e.g.", line)
    (ROOT / f"overture_screen_{variant}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("build", "screen"))
    parser.add_argument("--variant", default="text-9b")
    parser.add_argument("--min-conf", type=float, default=0.0)
    args = parser.parse_args(argv)
    build() if args.node == "build" else screen(args.variant, args.min_conf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
