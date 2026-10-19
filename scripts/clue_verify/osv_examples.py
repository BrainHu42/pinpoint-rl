"""Contact sheet: dev3k query | its 3 nearest OSV-5M images | its 2 nearest MP16 photos by other photographers (distance in m on each),
for random dev3k photos with an OSV image within 100 m -> artifacts/osv_refine/examples.jpg.

Usage: PYTHONPATH=src .venv/bin/python scripts/clue_verify/osv_examples.py
"""
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from geo_search_env.experiment.clue_verify import OSV_EMBED, _haversine
from geo_search_env.experiment.density_probe import _index_rows
from geo_search_env.experiment.osv_remote import INDEX, read
from geo_search_env.experiment.sft_data import MP16Images
from geo_search_env.experiment.strategy_search import MP16_CSV, MP16_EMBED, _memmap_gallery
from geo_search_env.models.pinpoint_reranker import _unit_xyz

N, H = 8, 220
n_osv = int(json.loads((OSV_EMBED / "manifest.json").read_text())["num_samples"])
osv_ll = np.fromfile(OSV_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(n_osv, 2).astype(np.float64)
osv_ids = (OSV_EMBED / "image_ids.txt").read_text().splitlines()
rows = _index_rows()
mp_ll = np.fromfile(MP16_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2)[rows].astype(np.float64)
authors = np.load("artifacts/agent/mp16_author_by_cache_row.npy")
row_index = _memmap_gallery(MP16_EMBED)["row_index"]
with MP16_CSV.open(encoding="utf-8", newline="") as stream:
    reader = csv.reader(stream)
    next(reader)
    csv_ids = [r[0] for r in reader]
zindex = json.loads(INDEX.read_text())
t_osv, t_mp = cKDTree(_unit_xyz(osv_ll)), cKDTree(_unit_xyz(mp_ll))
mp16 = MP16Images()
photos = json.loads(Path("artifacts/photo_sets/dev3k_setft_own/photos.json").read_text())
rng = np.random.default_rng(1)


def tile(data: bytes, label: str) -> Image.Image:
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im = im.resize((int(im.width * H / im.height), H))
    im = im.crop((0, 0, min(im.width, int(H * 1.5)), H))
    ImageDraw.Draw(im).rectangle((0, 0, 120, 18), fill="black")
    ImageDraw.Draw(im).text((4, 3), label, fill="white")
    return im


sheet_rows = []
for i in rng.permutation(len(photos)):
    if len(sheet_rows) == N:
        break
    p = photos[i]
    xyz = _unit_xyz(np.asarray([p["truth"]], np.float64))[0]
    d_o, j_o = t_osv.query(xyz, 3)
    km_o = _haversine(p["truth"][0], p["truth"][1], osv_ll[j_o, 0], osv_ll[j_o, 1])
    if km_o[0] > 0.1 or not all(osv_ids[j] in zindex for j in j_o):
        continue
    _, j_m = t_mp.query(xyz, 50)
    c = rows[j_m]
    j_m = j_m[(c != int(p["row"])) & (authors[c] != authors[int(p["row"])])][:2]
    km_m = _haversine(p["truth"][0], p["truth"][1], mp_ll[j_m, 0], mp_ll[j_m, 1])
    tiles = [tile(mp16.read(p["image_id"]), "query")]
    tiles += [tile(read(zindex[osv_ids[j]]), f"OSV {1000 * k:.0f} m") for j, k in zip(j_o, km_o)]
    tiles += [tile(mp16.read(csv_ids[row_index[rows[j]]]), f"MP16 {1000 * k:.0f} m") for j, k in zip(j_m, km_m)]
    sheet_rows.append(tiles)
w = max(sum(t.width for t in r) + 4 * len(r) for r in sheet_rows)
sheet = Image.new("RGB", (w, len(sheet_rows) * (H + 4)), "white")
for y, r in enumerate(sheet_rows):
    x = 0
    for t in r:
        sheet.paste(t, (x, y * (H + 4)))
        x += t.width + 4
sheet.save("artifacts/osv_refine/examples.jpg", quality=85)
print(f"-> artifacts/osv_refine/examples.jpg ({len(sheet_rows)} photos)")
