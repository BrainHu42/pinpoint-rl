# Candidate-side attributes from offline maps, for every pooled candidate of the train / dev / val photos.
# Usage: PYTHONPATH=src ~/.venvs/geo/bin/python -m geo_search_env.experiment.candidate_attributes build   (standalone: numpy, scipy, rasterio, pyarrow)

"""Per candidate location (lat, lon):

climate:   WorldClim 2.1 (2.5 arc-min) mean annual temperature, temperature seasonality, annual precipitation, precipitation
           seasonality (nodata filled from the nearest valid cell)
terrain:   ETOPO 2022 (60 arc-sec) elevation (signed log), ruggedness (std of the 5x5 neighbourhood, log), land flag
coast:     log distance to the nearest coastline pixel (land next to ocean in ETOPO)
urbanness: log of the Overture places count in the 0.02 degree cell and in its 3x3 neighbourhood
country:   ISO code of the nearest GeoNames populated place (languages come from GeoNames countryInfo, see `country_languages`)

Writes artifacts/query_evidence/attributes/candidates_<tag>.npz with `num` (photos, candidates, 10) and `country` (photos, candidates).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

GEO = Path("/data/pinpoint/geo")
OVERTURE = Path("/data/pinpoint/overture/places")
GEONAMES = Path("/data/pinpoint/geonames")
ART = Path("artifacts")
OUT = ART / "query_evidence" / "attributes"
EARTH_KM = 6371.0088
FEATURES = ("temp", "temp_season", "precip", "precip_season", "elev", "rugged", "land", "coast", "poi_cell", "poi_6km")
CELL = 0.02  # degrees, the Overture count grid


def xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack((np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)), axis=-1)


class Raster:
    """A global lat/lon GeoTIFF held in memory; `at` returns the value of the cell containing each point."""

    def __init__(self, path: Path, fill_nearest: bool = False) -> None:
        import rasterio
        from scipy.ndimage import distance_transform_edt

        with rasterio.open(path) as dataset:
            self.array = dataset.read(1).astype(np.float32)
            self.transform, nodata = dataset.transform, dataset.nodata
        bad = ~np.isfinite(self.array) | (self.array <= -1e30)
        if nodata is not None:
            bad |= self.array == nodata
        self.bad_share = float(bad.mean())
        if fill_nearest and bad.any():
            self.array = self.array[tuple(distance_transform_edt(bad, return_distances=False, return_indices=True))]

    def index(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t, (h, w) = self.transform, self.array.shape
        return np.clip(((lat - t.f) / t.e).astype(int), 0, h - 1), np.clip(((lon - t.c) / t.a).astype(int), 0, w - 1)

    def at(self, lat: np.ndarray, lon: np.ndarray, array: np.ndarray | None = None) -> np.ndarray:
        row, col = self.index(lat, lon)
        return (self.array if array is None else array)[row, col]


def poi_density() -> np.ndarray:
    """Overture places per 0.02 degree cell (rows north to south, columns west to east)."""

    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    rows, cols = int(180 / CELL), int(360 / CELL)
    flat: list[np.ndarray] = []
    for path in sorted(OVERTURE.glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1_000_000, columns=["bbox"]):
            lat = pc.struct_field(batch.column("bbox"), "ymin").to_numpy(zero_copy_only=False)
            lon = pc.struct_field(batch.column("bbox"), "xmin").to_numpy(zero_copy_only=False)
            r = np.clip(((90 - lat) / CELL).astype(np.int64), 0, rows - 1)
            c = np.clip(((lon + 180) / CELL).astype(np.int64), 0, cols - 1)
            flat.append((r * cols + c).astype(np.uint32))
        print(f"  poi grid: {path.name}", flush=True)
    counts = np.bincount(np.concatenate(flat), minlength=rows * cols).reshape(rows, cols)
    return counts.astype(np.float32)


def country_index() -> tuple["cKDTree", np.ndarray]:  # noqa: F821
    """KD-tree over GeoNames populated places and their country codes."""

    from scipy.spatial import cKDTree

    lat, lon, code = [], [], []
    with (GEONAMES / "allCountries.txt").open("r", encoding="utf-8") as stream:
        for line in stream:
            f = line.split("\t", 9)
            if f[6] == "P" and f[8]:
                lat.append(float(f[4])); lon.append(float(f[5])); code.append(f[8])
    print(f"  {len(lat)} populated places", flush=True)
    return cKDTree(xyz(np.asarray(lat), np.asarray(lon))), np.asarray(code)


def country_languages() -> dict[str, list[str]]:
    """ISO country code -> language codes (primary subtag) from GeoNames countryInfo."""

    out: dict[str, list[str]] = {}
    for line in (GEONAMES / "countryInfo.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            f = line.split("\t")
            out[f[0]] = [x.split("-")[0].lower() for x in f[15].split(",") if x]
    return out


def candidate_sets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(coords, valid) of the pooled candidates for each photo set, in the photo lists' order."""

    sets = {}
    for tag in ("train", "dev", "val"):
        photos = json.loads((OUT.parent / tag / "dev.json").read_text(encoding="utf-8"))
        index = [e["index"] for e in photos]
        path = ART / "strategy_search" / "search_features.npz" if "path" in photos[0] else ART / "sft" / "candidates.npz"
        saved = dict(np.load(path))
        sets[tag] = (saved["coords"][index], saved["valid"][index])
    return sets


def build() -> None:
    from scipy.ndimage import uniform_filter

    sets = candidate_sets()
    points = np.concatenate([coords[valid] for coords, valid in sets.values()])
    lat, lon = points[:, 0], points[:, 1]
    print(f"{len(points)} candidate points", flush=True)

    print("climate", flush=True)
    temp_raster = Raster(GEO / "worldclim" / "wc2.1_2.5m_bio_1.tif", fill_nearest=True)
    climate = [temp_raster.at(lat, lon)] + [Raster(GEO / "worldclim" / f"wc2.1_2.5m_bio_{n}.tif", fill_nearest=True).at(lat, lon) for n in (4, 12, 15)]

    print("terrain and coast", flush=True)
    etopo = Raster(GEO / "ETOPO_2022_v1_60s_N90W180_surface.tif")
    elev = etopo.array
    mean = uniform_filter(elev.astype(np.float64), size=5)
    rugged = np.sqrt(np.maximum(uniform_filter(elev.astype(np.float64) ** 2, size=5) - mean**2, 0)).astype(np.float32)
    land = elev > 0
    shore = land & ~(np.roll(land, 1, 0) & np.roll(land, -1, 0) & np.roll(land, 1, 1) & np.roll(land, -1, 1))
    rows, cols = np.nonzero(shore)
    t = etopo.transform
    from scipy.spatial import cKDTree

    coast_tree = cKDTree(xyz(t.f + (rows + 0.5) * t.e, t.c + (cols + 0.5) * t.a))
    chord, _ = coast_tree.query(xyz(lat, lon), workers=-1)
    coast_km = 2 * EARTH_KM * np.arcsin(np.minimum(chord / 2, 1.0))
    signed_log = lambda x: np.sign(x) * np.log1p(np.abs(x))
    del mean

    print("urbanness", flush=True)
    grid = poi_density()
    block = uniform_filter(grid, size=3) * 9
    r = np.clip(((90 - lat) / CELL).astype(int), 0, grid.shape[0] - 1)
    c = np.clip(((lon + 180) / CELL).astype(int), 0, grid.shape[1] - 1)

    num = np.stack([*climate, signed_log(etopo.at(lat, lon)), np.log1p(etopo.at(lat, lon, rugged)), (etopo.at(lat, lon) > 0).astype(np.float32),
                    np.log1p(coast_km), np.log1p(grid[r, c]), np.log1p(block[r, c])], axis=1).astype(np.float32)

    print("countries", flush=True)
    tree, codes = country_index()
    _, nearest = tree.query(xyz(lat, lon), workers=-1)
    country = codes[nearest]

    OUT.mkdir(parents=True, exist_ok=True)
    start = 0
    for tag, (coords, valid) in sets.items():
        count = int(valid.sum())
        n_out = np.zeros(valid.shape + (len(FEATURES),), dtype=np.float32)
        c_out = np.full(valid.shape, "", dtype="<U2")
        n_out[valid], c_out[valid] = num[start : start + count], country[start : start + count]
        np.savez(OUT / f"candidates_{tag}.npz", num=n_out, country=c_out)
        start += count
        print(f"saved {tag}: {valid.shape[0]} photos", flush=True)
    (OUT / "country_languages.json").write_text(json.dumps(country_languages()) + "\n", encoding="utf-8")
    print("feature means:", dict(zip(FEATURES, np.round(num.mean(0), 2).tolist())), flush=True)
    check = {"Paris": (48.86, 2.35), "Denver": (39.74, -104.99), "Dubai": (25.20, 55.27), "Reykjavik": (64.15, -21.94), "Singapore": (1.35, 103.82), "Kansas City": (39.10, -94.58)}
    for name, (la, lo) in check.items():  # known places: temperature, elevation, distance to coast, places in the cell
        la_a, lo_a = np.asarray([la]), np.asarray([lo])
        km = 2 * EARTH_KM * math.asin(min(coast_tree.query(xyz(la_a, lo_a))[0][0] / 2, 1))
        print(f"  sanity {name}: {temp_raster.at(la_a, lo_a)[0]:.1f} C, elev {etopo.at(la_a, lo_a)[0]:.0f} m, coast {km:.0f} km, "
              f"{grid[int((90 - la) / CELL), int((lo + 180) / CELL)]:.0f} places in the cell", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("build",))
    parser.parse_args(argv)
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
