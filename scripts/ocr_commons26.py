# Dedicated scene-text OCR for commons26 dev (lesson R12: the text chain is limited by what the 4B reads, not how it searches). PaddleOCR 3
# (PP-OCRv5 detection + recognition) in its own venv, one pass per script family; lines kept above MIN_SCORE, deduplicated per photo.
# -> artifacts/agent/ocr_commons26dev_paddle.json {image_id: [[text, score, lang], ...]}
# Usage: /home/brianhu/pinpoint-rl/.venvs/paddle/bin/python scripts/ocr_commons26.py [--limit N] [--langs ch latin ...]
import argparse
import json
import time
from pathlib import Path

PHOTOS = Path("artifacts/photo_sets/commons26dev_setft_own/photos.json")
IMAGES = Path("data/pinpoint/commons26/release/images")
OUT = Path("artifacts/agent/ocr_commons26dev_paddle.json")
# recognisers: "ch" = PP-OCRv6 (Chinese, Japanese, English and the Latin-script languages), then PP-OCRv5 per script: Cyrillic (ru),
# Korean, Arabic, Devanagari (hi), Thai, Greek
LANGS = ("ch", "ru", "korean", "ar", "hi", "th", "el")
MIN_SCORE = 0.5


def main() -> None:
    from paddleocr import PaddleOCR

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--langs", nargs="*", default=LANGS)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    photos = json.loads(PHOTOS.read_text())
    if a.limit:
        photos = photos[: a.limit]
    out = {e["image_id"]: [] for e in photos}
    for lang in a.langs:
        t0 = time.time()
        ocr = PaddleOCR(lang=lang, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
        for e in photos:
            try:
                res = ocr.predict(str(IMAGES / e["image_id"]))
            except Exception as ex:  # noqa: BLE001
                print(f"  {e['image_id']} {lang}: {ex}", flush=True)
                continue
            for r in res:
                for text, score in zip(r["rec_texts"], r["rec_scores"]):
                    if score >= MIN_SCORE and text.strip():
                        out[e["image_id"]].append([text.strip(), float(score), lang])
        print(f"{lang}: {len(photos)} photos in {time.time() - t0:.0f}s", flush=True)
        del ocr
    for k, v in out.items():  # the same text read by several recognisers: keep the most confident
        best = {}
        for text, score, lang in v:
            if text not in best or score > best[text][1]:
                best[text] = [text, score, lang]
        out[k] = sorted(best.values(), key=lambda x: -x[1])
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False))
    n = sum(1 for v in out.values() if v)
    print(f"-> {a.out}: text on {n} / {len(out)} photos, {sum(len(v) for v in out.values())} lines", flush=True)


if __name__ == "__main__":
    main()
