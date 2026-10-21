"""Environmental map attributes on one global 2.5-arcminute grid (8640 x 4320, ~4.6 km at the equator) for the geodata check
(scripts/geodata_check.py): elevation and relief (WorldClim 2.1 30 s), climate (WorldClim 2.1 bioclim 2.5 m), land cover (EarthEnv consensus
1 km, 12 classes, % cover) and distance to the coast. Ocean cells are NaN. Writes data/geodata/grid/{name}.npy (float16, row 0 = 90 N).

  .venvs/geodata/bin/python -I scripts/geodata_grid.py data/geodata
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt

ROOT = Path(sys.argv[1])
OUT = ROOT / "grid"
RES = 1 / 24
W, H = 8640, 4320
DST = from_origin(-180, 90, RES, RES)
BIO = {"bio1": 1, "bio4": 4, "bio5": 5, "bio6": 6, "bio12": 12, "bio15": 15}
LC = {1: "lc_needleleaf", 2: "lc_evergreen_broadleaf", 3: "lc_deciduous_broadleaf", 4: "lc_mixed_trees", 5: "lc_shrubs", 6: "lc_herbaceous",
      7: "lc_cultivated", 8: "lc_flooded", 9: "lc_urban", 10: "lc_snow_ice", 11: "lc_barren", 12: "lc_water"}


def warp(path: Path, square: bool = False) -> np.ndarray:
    """A raster averaged onto the grid (ocean / nodata -> NaN); with `square`, the average of its square too."""

    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        outs = []
        for x in ([a, a * a] if square else [a]):
            dst = np.full((H, W), np.nan, np.float32)
            reproject(x, dst, src_transform=src.transform, src_crs=src.crs, dst_transform=DST, dst_crs="EPSG:4326",
                      src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.average)
            outs.append(dst)
    return outs if square else outs[0]


def save(name: str, a: np.ndarray, land: np.ndarray | None = None) -> None:
    if land is not None:
        a = np.where(land, a, np.nan)
    np.save(OUT / f"{name}.npy", a.astype(np.float16))
    print(name, f"{np.nanmin(a):.1f}..{np.nanmax(a):.1f}", flush=True)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    bio = ROOT / "worldclim_bio" / "x"
    land = None
    for name, k in BIO.items():
        a = warp(bio / f"wc2.1_2.5m_bio_{k}.tif")
        if land is None:
            land = np.isfinite(a)
            np.save(OUT / "land.npy", land)
        save(name, a, land)
    m, m2 = warp(ROOT / "worldclim_elev" / "x" / "wc2.1_30s_elev.tif", square=True)
    save("elev", m, land)
    save("relief", np.sqrt(np.maximum(m2 - m * m, 0)), land)
    del m, m2
    for k, name in LC.items():
        a = warp(ROOT / "earthenv_lc" / f"class_{k}.tif")
        save(name, np.nan_to_num(a, nan=0.0), land)
    # distance to the nearest ocean cell (great-circle, km), from the land mask
    _, (iy, ix) = distance_transform_edt(land, return_indices=True)
    lat = np.deg2rad(90 - (np.arange(H) + 0.5) * RES)[:, None]
    lon = np.deg2rad(-180 + (np.arange(W) + 0.5) * RES)[None, :]
    lat2, lon2 = np.deg2rad(90 - (iy + 0.5) * RES), np.deg2rad(-180 + (ix + 0.5) * RES)
    del iy, ix
    h = np.sin((lat2 - lat) / 2) ** 2 + np.cos(lat) * np.cos(lat2) * np.sin((lon2 - lon) / 2) ** 2
    save("coast_km", 2 * 6371 * np.arcsin(np.sqrt(np.clip(h, 0, 1))), land)


if __name__ == "__main__":
    main()
