"""Blind reader test: export 40 random dev3k photos where none of the pipeline's areas is right (< 25 km), resized to the 4B's pixel budget.
Images go to reader/p00.jpg ...; the key (image id, truth) to reader_key/key.json, which is not opened until scoring."""
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from geo_search_env.experiment.hypothesis_agent import area_points
from geo_search_env.experiment.sft_data import MP16Images

tmp = Path("artifacts/agent/reader_test")  # images in reader/, key in reader_key/ (blind until scoring)
photos = json.loads(Path("artifacts/photo_sets/dev3k_setft_a/photos.json").read_text())
truth = np.asarray([p["truth"] for p in photos], np.float64)
photo, _, point = area_points("dev3k")
p1, p2 = np.radians(truth[photo, 0]), np.radians(point[:, 0])
h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(point[:, 1] - truth[photo, 1]) / 2) ** 2
km = 2 * 6371 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
any_right = np.zeros(len(photos), bool)
any_right[photo[km < 25]] = True
pick = np.random.default_rng(42).choice(np.flatnonzero(~any_right), 40, replace=False)
images = MP16Images()
(tmp / "reader").mkdir(exist_ok=True)
(tmp / "reader_key").mkdir(exist_ok=True)
key = []
for j, i in enumerate(pick):
    im = Image.open(io.BytesIO(images.read(photos[i]["image_id"]))).convert("RGB")
    scale = min(1.0, (786432 / (im.width * im.height)) ** 0.5)
    if scale < 1:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    im.save(tmp / "reader" / f"p{j:02d}.jpg", quality=92)
    key.append({"p": f"p{j:02d}", "index": int(i), "image_id": photos[i]["image_id"], "truth": photos[i]["truth"]})
(tmp / "reader_key" / "key.json").write_text(json.dumps(key))
print(f"exported {len(key)} photos")
