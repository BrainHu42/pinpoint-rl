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


def label(latlon: np.ndarray, threads: int = 16) -> dict[str, list]:
    """Per level: (name, area in km^2) of the smallest containing land polygon, or (None, nan)."""

    import duckdb
    import pyarrow as pa

    con = duckdb.connect()
    con.sql(f"INSTALL spatial; LOAD spatial; SET threads={threads}")
    con.register("pts_arrow", pa.table({"i": np.arange(len(latlon)), "lat": latlon[:, 0], "lon": latlon[:, 1]}))
    con.sql("CREATE TABLE pts AS SELECT i, ST_Point(lon, lat) AS geom FROM pts_arrow")
    con.sql(f"""CREATE TABLE polys AS SELECT subtype, coalesce(names.common['en'], names."primary") AS name, geometry AS geom,
                ST_Area_Spheroid(ST_FlipCoordinates(geometry)) / 1e6 AS km2
                FROM '{DIVISIONS}/*.parquet' WHERE class = 'land' AND subtype IN {LEVELS}""")
    rows = con.sql("""SELECT pts.i, polys.subtype, polys.name, polys.km2 FROM pts JOIN polys ON ST_Contains(polys.geom, pts.geom)""").fetchall()
    out = {lv: [(None, float("nan"))] * len(latlon) for lv in LEVELS}
    for i, sub, name, km2 in rows:
        cur = out[sub][i]
        if cur[0] is None or (km2 is not None and km2 < cur[1]):
            out[sub][i] = (name, km2)
    return out


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
    parser.add_argument("node", choices=("pilot", "agreement"))
    parser.add_argument("--sample", type=int, default=200_000)
    args = parser.parse_args(argv)
    if args.node == "pilot":
        pilot(args.sample)
    else:  # rerun the agreement check on the saved pilot labels
        saved = json.loads((OUT / "pilot.json").read_text(encoding="utf-8"))
        latlon = np.fromfile(MP16_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2).astype(np.float64)
        agreement(np.asarray(saved["rows"]), {lv: [(n, 0.0) for n in names] for lv, names in saved["labels"].items()}, latlon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
