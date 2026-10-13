"""Blind contact sheets for the locatability study (scripts/locatability_score.py): a fixed random sample of dev3k photos, numbered, with no labels, truth or Pinpoint answer."""
import io
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

from geo_search_env.experiment.sft_data import MP16Images

OUT, N = sys.argv[1], int(sys.argv[2])
photos = json.load(open("artifacts/photo_sets/dev3k/photos.json"))
pick = sorted(np.random.default_rng(42).choice(len(photos), N, replace=False).tolist())
sample = [photos[i]["image_id"] for i in pick]
json.dump(sample, open(f"{OUT}/sample.json", "w"))
images = MP16Images()
for s in range(0, N, 6):
    sheet = Image.new("RGB", (3 * 520, 2 * 540), "white")
    draw = ImageDraw.Draw(sheet)
    for k, image_id in enumerate(sample[s:s + 6]):
        im = Image.open(io.BytesIO(images.read(image_id))).convert("RGB")
        im.thumbnail((520, 520))
        x, y = (k % 3) * 520, (k // 3) * 540
        sheet.paste(im, (x, y))
        draw.rectangle([x, y + 520, x + 520, y + 540], fill="black")
        draw.text((x + 4, y + 524), f"#{s + k}", fill="yellow")
    sheet.save(f"{OUT}/sheet_{s // 6:02d}.jpg", quality=88)
print(N, "photos ->", OUT)
