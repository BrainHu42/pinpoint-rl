# Qualitative look at "choosing" photos: what, if anything, tells the right candidate from the reranker's wrong top-1?
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.choosing_sheet    (CPU; writes artifacts/query_evidence/inspect/)

"""Choosing photo = a candidate < 25 km from the truth among the top 8, reranker top-1 >= 25 km off (placeholders dropped). Samples N per set
(dev, val) with a fixed seed. For each: a montage of the query photo, the exemplar of the best-ranked right candidate, and the exemplar of the
wrong top-1 (the MP16 photo within 1 km of each candidate most similar to the query, not by the query's photographer; as scored by the
comparator), plus a JSON row with names, distances, ranks and comparator P(same)."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .query_evidence import ROOT, MP16Images
from .wiki_backend import _km

OUT = ROOT / "inspect"
N = 15
TOPK = 8
HEIGHT = 380
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _panel(data: bytes, caption: str, font) -> Image.Image:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((max(1, round(image.width * HEIGHT / image.height)), HEIGHT))
    image = image.crop((0, 0, min(image.width, 560), HEIGHT))
    panel = Image.new("RGB", (image.width, HEIGHT + 44), "white")
    panel.paste(image, (0, 0))
    draw = ImageDraw.Draw(panel)
    for k, line in enumerate(caption.split("\n")[:2]):
        draw.text((4, HEIGHT + 2 + 20 * k), line[:70], fill="black", font=font)
    return panel


def main() -> None:
    images = MP16Images()
    font = ImageFont.truetype(FONT, 15) if Path(FONT).exists() else ImageFont.load_default()
    excluded = set(json.loads((ROOT / "val" / "exclude.json").read_text(encoding="utf-8")))
    scored = json.loads((ROOT / "exemplar_topk_scores_comparator-a.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag in ("dev", "val"):
        photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
        judged = [s for s in scored if s["tag"] == tag]
        choosing = []
        for e, s in zip(photos, judged):
            assert e["image_id"] == s["image_id"]
            if e["image_id"] in excluded:
                continue
            d = _km(np.asarray(e["pool"][:TOPK]), *e["truth"])
            if d[0] >= 25 and (d < 25).any():
                choosing.append((e, s, d))
        for e, s, d in [choosing[i] for i in sorted(np.random.default_rng(0).choice(len(choosing), N, replace=False))]:
            right = int(np.flatnonzero(d < 25)[0])
            name = lambda r: e["candidates"][r][0] if r < len(e["candidates"]) else "?"
            query = Path(e["path"]).read_bytes() if e.get("path") else images.read(e["image_id"])
            panels = [_panel(query, f"QUERY ({tag})\ntruth {e['truth'][0]:.3f}, {e['truth'][1]:.3f}", font)]
            for label, r in (("RIGHT", right), ("WRONG top-1", 0)):
                ex = s["exemplars"][r]
                caption = f"{label}: rank {r + 1}, {d[r]:.1f} km off, P(same) {s['p_same'][r]:.2f}\n{name(r)}"
                panels.append(_panel(images.read(ex) if ex else b"", caption, font) if ex else Image.new("RGB", (300, HEIGHT + 44), "grey"))
            sheet = Image.new("RGB", (sum(p.width for p in panels) + 20, HEIGHT + 44), "white")
            x = 0
            for p in panels:
                sheet.paste(p, (x, 0))
                x += p.width + 10
            file = OUT / f"{len(rows):02d}_{tag}_{Path(e['image_id']).stem}.jpg"
            sheet.save(file, quality=85)
            rows.append({"file": file.name, "tag": tag, "image_id": e["image_id"], "truth": e["truth"], "right_rank": right + 1, "right_km": float(d[right]),
                         "wrong_km": float(d[0]), "right_name": name(right), "wrong_name": name(0), "p_right": s["p_same"][right], "p_wrong": s["p_same"][0]})
        print(f"{tag}: {len(choosing)} choosing photos, sampled {N}")
    (OUT / "sheet.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
