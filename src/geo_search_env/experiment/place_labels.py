# Fine-grained place labels for MP16 photos from Overture Maps divisions (release 2026-09-23.1; /data/pinpoint/overture/divisions, downloaded with
# `aws s3 cp --no-sign-request --recursive s3://overturemaps-us-west-2/release/2026-09-23.1/theme=divisions/ .`): for each photo, the smallest land polygon of
# each subtype (country, region, county, locality, macrohood, neighborhood, microhood) that contains its coordinates.
# Usage: PYTHONPATH=src ~/.venvs/overture/bin/python -m geo_search_env.experiment.place_labels pilot [--sample 200000]   (CPU; ~/.venvs/overture = duckdb, shapely, pyarrow)

"""pilot: label a random sample of MP16 photos and report coverage, area per level, and label agreement between near-duplicate photos by different
        photographers (cosine >= DUP_SIM in SigLIP2 space, within 2 km), the label noise from GPS error and camera position."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np

DIVISIONS = Path("/data/pinpoint/overture/divisions/type=division_area")
MP16_EMBED = Path("/data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384")
OUT = Path("artifacts/place_labels")
LEVELS = ("country", "region", "county", "locality", "macrohood", "neighborhood", "microhood")
DUP_SIM = 0.95


class Labeller:
    """Overture land polygons loaded once into DuckDB; label() maps points to the smallest containing polygon per level."""

    def __init__(self, threads: int = 32) -> None:
        import duckdb

        self.con = duckdb.connect()
        self.con.sql(f"INSTALL spatial; LOAD spatial; SET threads={threads}")
        # per division: English name, population, and the region / country / locality of its own hierarchy (the last such entry: the nearest)
        self.con.sql(f"""CREATE TABLE divs AS SELECT id, coalesce(names.common['en'], names."primary") AS en, population AS pop,
                         list_filter(hierarchies[1], x -> x.subtype = 'region')[-1].division_id AS region_id,
                         list_filter(hierarchies[1], x -> x.subtype = 'country')[-1].division_id AS country_id,
                         list_filter(hierarchies[1], x -> x.subtype = 'locality')[-1].division_id AS locality_id
                         FROM '{DIVISIONS.parent}/type=division/*.parquet'""")
        self.con.sql(f"""CREATE TABLE polys AS SELECT a.subtype, a.division_id, coalesce(a.names.common['en'], a.names."primary") AS name, a.geometry AS geom,
                         coalesce(ST_Area_Spheroid(ST_FlipCoordinates(a.geometry)) / 1e6, 'inf'::DOUBLE) AS km2, coalesce(d.pop, 0) AS pop
                         FROM '{DIVISIONS}/*.parquet' a LEFT JOIN divs d ON a.division_id = d.id WHERE a.class = 'land' AND a.subtype IN {LEVELS}""")

    def label(self, latlon: np.ndarray) -> dict[str, list]:
        """Per level: (name, area in km^2) of the smallest containing land polygon, or (None, nan)."""

        import pyarrow as pa

        self.con.register("pts_arrow", pa.table({"i": np.arange(len(latlon)), "lat": latlon[:, 0], "lon": latlon[:, 1]}))
        self.con.sql("CREATE OR REPLACE TABLE pts AS SELECT i, ST_Point(lon, lat) AS geom FROM pts_arrow")
        rows = self.con.sql("SELECT pts.i, polys.subtype, polys.name, polys.km2 FROM pts JOIN polys ON ST_Contains(polys.geom, pts.geom)").fetchall()
        self.con.unregister("pts_arrow")
        out = {lv: [(None, float("nan"))] * len(latlon) for lv in LEVELS}
        for i, sub, name, km2 in rows:
            cur = out[sub][i]
            if cur[0] is None or (km2 is not None and km2 < cur[1]):
                out[sub][i] = (name, km2)
        return out


    def canonical(self, latlon: np.ndarray) -> list[dict[str, str | None]]:
        """One consistent chain per point: {country, region, locality, neighborhood}. The city is the containing locality with the largest population
        (Chicago, not the overlapping South Chicago township that Overture's own hierarchy names as the Loop's parent; New York, not Manhattan); without
        one, the locality in the finest polygon's hierarchy. The finest level is the smallest containing neighborhood (else macrohood), kept only if it is
        smaller than the city's polygon and not the city's name. Region and country come from the city's hierarchy."""

        import pyarrow as pa

        self.con.register("pts_arrow", pa.table({"i": np.arange(len(latlon)), "lat": latlon[:, 0], "lon": latlon[:, 1]}))
        self.con.sql("CREATE OR REPLACE TABLE pts AS SELECT i, ST_Point(lon, lat) AS geom FROM pts_arrow")
        # all of the selection runs inside DuckDB (all threads); the keys order neighborhoods before macrohoods, and cities by population, then area
        rows = self.con.sql("""
            WITH m AS (SELECT pts.i, p.subtype, p.division_id AS did, p.km2, p.pop FROM pts JOIN polys p ON ST_Contains(p.geom, pts.geom)),
            loc AS (SELECT i, arg_max(did, pop * 1e7 + least(km2, 1e6)) AS loc, max_by(km2, pop * 1e7 + least(km2, 1e6)) AS loc_km2
                    FROM m WHERE subtype = 'locality' GROUP BY i),
            fine AS (SELECT i, arg_min(did, (subtype <> 'neighborhood')::INT * 1e12 + km2) AS fine,
                            min_by(km2, (subtype <> 'neighborhood')::INT * 1e12 + km2) AS fine_km2
                     FROM m WHERE subtype IN ('neighborhood', 'macrohood') GROUP BY i),
            reg AS (SELECT i, arg_min(did, km2) AS reg FROM m WHERE subtype = 'region' GROUP BY i),
            ctry AS (SELECT i, arg_min(did, km2) AS ctry FROM m WHERE subtype = 'country' GROUP BY i),
            b AS (SELECT pts.i, coalesce(loc.loc, fd.locality_id) AS city, coalesce(loc.loc_km2, 'inf'::DOUBLE) AS city_km2, fine.fine, fine.fine_km2,
                         fd.en AS fine_en, fd.region_id AS fine_region, fd.country_id AS fine_country, reg.reg, ctry.ctry
                  FROM pts LEFT JOIN loc USING (i) LEFT JOIN fine USING (i) LEFT JOIN reg USING (i) LEFT JOIN ctry USING (i)
                  LEFT JOIN divs fd ON fd.id = fine.fine),
            k AS (SELECT b.*, cd.en AS city_en, cd.region_id AS city_region, cd.country_id AS city_country,
                         b.fine IS NOT NULL AND b.fine_km2 < b.city_km2 AND b.fine_en IS DISTINCT FROM cd.en AND regexp_matches(coalesce(b.fine_en, ''), '\\pL')
                         AS keep_fine
                  FROM b LEFT JOIN divs cd ON cd.id = b.city)
            SELECT k.i, co.en, re.en, k.city_en, CASE WHEN k.keep_fine THEN k.fine_en END
            FROM k LEFT JOIN divs re ON re.id = coalesce(k.city_region, CASE WHEN k.keep_fine THEN k.fine_region END, k.reg)
                   LEFT JOIN divs co ON co.id = coalesce(k.city_country, CASE WHEN k.keep_fine THEN k.fine_country END, k.ctry)
            ORDER BY k.i""").fetchall()
        self.con.unregister("pts_arrow")
        return [{"country": c, "region": r, "locality": l, "neighborhood": n} for _, c, r, l, n in rows]


CANONICAL = ("country", "region", "locality", "neighborhood")


def label(latlon: np.ndarray, threads: int = 16) -> dict[str, list]:
    return Labeller(threads).label(latlon)


def label_all(chunk: int = 1_000_000) -> None:
    """Labels for every candidate (and the truth) of the dev / val / full photo sets, then for all MP16 photos (chunked), written under OUT."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    labeller = Labeller()
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in ("dev", "val", "full"):
        photos = json.loads((Path("artifacts/query_evidence") / tag / "dev.json").read_text(encoding="utf-8"))
        pts = np.asarray([c for e in photos for c in e["pool"]] + [e["truth"] for e in photos], dtype=np.float64)
        names = labeller.canonical(pts)
        out, k = [], 0
        for e in photos:
            out.append({"image_id": e["image_id"], "pool": names[k : k + len(e["pool"])]})
            k += len(e["pool"])
        for e, o in zip(photos, out):
            o["truth"] = names[k]
            k += 1
        (OUT / f"candidates_{tag}.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
        print(f"{tag}: {len(photos)} photos, {len(pts)} points labelled", flush=True)
    latlon = np.fromfile(MP16_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2).astype(np.float64)
    writer = None
    start = time.time()
    for lo in range(0, len(latlon), chunk):
        lab = labeller.canonical(latlon[lo : lo + chunk])
        table = pa.table({"row": np.arange(lo, lo + len(lab))} | {lv: pa.array([x[lv] for x in lab], pa.string()) for lv in CANONICAL})
        writer = writer or pq.ParquetWriter(OUT / "mp16.parquet", table.schema)
        writer.write_table(table)
        print(f"  mp16 {lo + len(lab)}/{len(latlon)} ({time.time() - start:.0f} s)", flush=True)
    writer.close()


def pilot(sample: int) -> None:
    latlon = np.fromfile(MP16_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2).astype(np.float64)
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(len(latlon), sample, replace=False))
    start = time.time()
    labels = label(latlon[rows])
    print(f"labelled {sample} photos in {time.time() - start:.0f} s")
    print(f"\n{'level':14s} {'coverage':>9s} {'distinct':>9s} {'median km2':>11s} {'p90 km2':>9s}")
    for lv in LEVELS:
        names = [x[0] for x in labels[lv]]
        km2 = np.asarray([x[1] for x in labels[lv]], float)
        cov = np.mean([n is not None for n in names])
        ok = np.isfinite(km2)
        print(f"{lv:14s} {cov:9.1%} {len(set(n for n in names if n)):9d} {np.median(km2[ok]) if ok.any() else float('nan'):11.1f} "
              f"{np.quantile(km2[ok], 0.9) if ok.any() else float('nan'):9.1f}")
    # finest available label per photo
    finest = [next((lv for lv in LEVELS[::-1] if labels[lv][k][0]), None) for k in range(sample)]
    print("\nfinest label available: " + ", ".join(f"{lv} {np.mean([f == lv for f in finest]):.1%}" for lv in LEVELS[::-1] + (None,)))
    ids = np.asarray((MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines())
    examples = rng.choice(sample, 20, replace=False)
    print("\nexamples (microhood | neighborhood | macrohood | locality | region | country):")
    for k in examples:
        print("  " + " | ".join(labels[lv][k][0] or "-" for lv in LEVELS[::-1][:4] + ("region", "country")))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pilot.json").write_text(json.dumps({"rows": rows.tolist(), "labels": {lv: [x[0] for x in labels[lv]] for lv in LEVELS}}) + "\n", encoding="utf-8")
    agreement(rows, labels, latlon)


def agreement(rows: np.ndarray, labels: dict, latlon: np.ndarray) -> None:
    """Near-duplicate pairs (cosine >= DUP_SIM, different photographers, within 2 km) among the sample; share whose labels agree at each level."""

    import csv

    from scipy.spatial import cKDTree

    authors = {}
    with open("/data/hf/datasets/MP16-Pro/metadata/MP16_Pro_filtered.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            authors[r["IMG_ID"]] = r["AUTHOR"]
    ids = (MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines()
    manifest = json.loads((MP16_EMBED / "manifest.json").read_text(encoding="utf-8"))
    emb = np.memmap(MP16_EMBED / manifest["files"]["embeddings"], dtype=np.float16, mode="r", shape=tuple(manifest["shapes"]["embeddings"]))
    E = np.asarray(emb[rows], dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    ll = latlon[rows]
    xyz = np.stack((np.cos(np.radians(ll[:, 0])) * np.cos(np.radians(ll[:, 1])), np.cos(np.radians(ll[:, 0])) * np.sin(np.radians(ll[:, 1])), np.sin(np.radians(ll[:, 0]))), -1)
    pairs = cKDTree(xyz).query_pairs(2 * np.sin(2.0 / 6371.0088 / 2), output_type="ndarray")
    sims = np.concatenate([np.einsum("ij,ij->i", E[p[:, 0]], E[p[:, 1]]) for p in np.array_split(pairs, max(1, len(pairs) // 200_000))])
    a = np.asarray([authors.get(ids[r], "") for r in rows])
    keep = (sims >= DUP_SIM) & (a[pairs[:, 0]] != a[pairs[:, 1]])
    pairs = pairs[keep]
    km = 6371.0088 * np.arccos(np.clip(np.einsum("ij,ij->i", xyz[pairs[:, 0]], xyz[pairs[:, 1]]), -1, 1))
    print(f"\nnear-duplicate pairs by different photographers within 2 km (cosine >= {DUP_SIM}): {len(pairs)}; distance between their GPS: median {np.median(km) * 1000:.0f} m, "
          f"p75 {np.quantile(km, 0.75) * 1000:.0f} m, p90 {np.quantile(km, 0.9) * 1000:.0f} m")
    for lv in LEVELS:
        both = [(labels[lv][i][0], labels[lv][j][0]) for i, j in pairs if labels[lv][i][0] and labels[lv][j][0]]
        if both:
            print(f"  {lv:14s} both labelled {len(both):6d}: same label {np.mean([x == y for x, y in both]):.1%}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("pilot", "agreement", "label-all"))
    parser.add_argument("--sample", type=int, default=200_000)
    args = parser.parse_args(argv)
    if args.node == "label-all":
        label_all()
    elif args.node == "pilot":
        pilot(args.sample)
    else:  # rerun the agreement check on the saved pilot labels
        saved = json.loads((OUT / "pilot.json").read_text(encoding="utf-8"))
        latlon = np.fromfile(MP16_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2).astype(np.float64)
        agreement(np.asarray(saved["rows"]), {lv: [(n, 0.0) for n in names] for lv, names in saved["labels"].items()}, latlon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
