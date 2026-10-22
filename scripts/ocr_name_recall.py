# Does dedicated OCR read the place's name more often than the 4B? commons26 dev photos flagged as showing place text (benchmark.csv
# place_text = 1); the reference is the Commons file title (attribution.csv source URL, e.g. "Fireplace in Trinity House of Leith"): share whose
# specific title words (>= 4 letters, not generic, no digits) appear in what each reader produced, exactly (as a whole word)
# or approximately (a word within edit ratio >= 0.8, the near-misses literal search drops, e.g. OCR "BIACHE-STVAST" for Biache-Saint-Vaast).
# Usage: PYTHONPATH=src .venv/bin/python scripts/ocr_name_recall.py
import csv
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import text_refine_check as trc  # noqa: E402
from geo_search_env.experiment.text_search import norm  # noqa: E402


def main() -> None:
    bench = {r["IMG_ID"]: r for r in csv.DictReader(open("data/pinpoint/commons26/release/benchmark.csv", encoding="utf-8"))}
    photos = trc._photos()
    obs = json.loads(Path(f"artifacts/agent/observe_{trc.TAG}_4b.json").read_text(encoding="utf-8"))
    queries = json.loads(Path(f"artifacts/agent/queries_{trc.TAG}_4b.json").read_text(encoding="utf-8"))
    ocr = trc._ocr_phrases()
    readers = {
        "4B text": lambda k: (obs.get(k, {}).get("parsed") or {}).get("text") or [],
        "4B text + names + queries": lambda k: ((obs.get(k, {}).get("parsed") or {}).get("text") or [])
        + ((obs.get(k, {}).get("parsed") or {}).get("place_names") or []) + (queries.get(k) or []),
        "OCR": lambda k: ocr.get(k, []),
    }
    readers["4B all + OCR"] = lambda k: readers["4B text + names + queries"](k) + readers["OCR"](k)
    import urllib.parse

    title = {r["IMG_ID"]: urllib.parse.unquote(r["source"].rsplit("File:", 1)[-1]).rsplit(".", 1)[0].replace("_", " ")
             for r in csv.DictReader(open("data/pinpoint/commons26/release/attribution.csv", encoding="utf-8"))}
    rows = [(e["image_id"], title.get(e["image_id"], "")) for e in photos if bench[e["image_id"]]["place_text"] == "1"]
    lines = [f"title-word recall on the {len(rows)} commons26 dev photos flagged as showing place text (of {len(photos)})"]
    for name, get in readers.items():
        exact = approx = 0
        n = 0
        for k, b in rows:
            target = {w for w in norm(b).split() if len(w) >= 4 and w not in trc.GENERIC and not any(c.isdigit() for c in w)}
            words = set(" ".join(norm(str(x)) for x in get(k)).split())
            joined = " ".join(norm(str(x)) for x in get(k)).replace(" ", "")
            if not target:
                continue
            n += 1
            e = bool(target & words)
            a = e or any(t in joined or difflib.get_close_matches(t, words, 1, 0.8) for t in target)
            exact += e
            approx += a
        lines.append(f"  {name:28s} a title word read exactly {exact / n:.1%}, approximately {approx / n:.1%} (of {n} with title words)")
    text = "\n".join(lines)
    print(text)
    (trc.OUT / "ocr_name_recall.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
