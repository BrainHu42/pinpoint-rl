# KNOWLEDGE_PLAN.md step-1 check: does the knowledge-SFT model name the right country / region / city more often than the base 4B,
# especially on benchmark photos where the reranker (visual similarity) is wrong?
# Usage: scripts/knowledge_check.sh (serves each model, runs `answer`, then `report`)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.knowledge_check answer --name knowledge-a   (vLLM on :8765)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.knowledge_check report
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.knowledge_check rule --name knowledge-a   (after `answer --tag dev`; scripts/knowledge_ceiling.sh)

"""answer: greedy answer to the knowledge prompt for every photo of the full im2gps3k + yfcc4k eval halves (artifacts/query_evidence/full),
        optionally with a format example (`--hint`, for the base model, which wasn't trained on the format).
report: per level, the share of photos whose answer names the truth's Overture division (accent- and case-insensitive; a level counts as
        right when one name contains the other), on all photos and on those where the reranker top-1 is >= 25 km off; paired difference
        between each model and the base with a bootstrap 95% interval.
rule:   the model's answer as a candidate chooser: take the best-ranked of the reranker's first K pool candidates whose Overture labels match the answer down
        to a level (else keep the top-1). Variants (level x K) are compared on MP16 dev; all are then shown on the full eval halves, the dev choice marked.
        Change in top-1 < 1 / 25 / 200 km vs the reranker top-1, fixes (top-1 >= 25 km off, rule < 25 km) and breaks."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import unicodedata
import urllib.request
from typing import Sequence

import numpy as np

from ..paths import DATA_ROOT
from .knowledge_data import PROMPT
from .name_score import SEP
from .strategy_search import _haversine_km

PHOTOS = Path("artifacts/query_evidence/full/dev.json")
TRUTH = Path("artifacts/place_labels/candidates_full.json")
OUT = Path("artifacts/knowledge")
HINT = " For example: France > Île-de-France > Paris"
LEVELS = (("country", "country"), ("region", "region"), ("city", "locality"))


def _image(path: str) -> bytes:
    return (DATA_ROOT / Path(path).relative_to("/data/pinpoint")).read_bytes() if path.startswith("/data/pinpoint/") else Path(path).read_bytes()


def answer(name: str, hint: bool, tag: str = "full", server: str = "http://127.0.0.1:8765", workers: int = 32) -> None:
    """tag full: the benchmark eval halves (saved in OUT, read by `report`); any other tag: that MP16 photo set (saved in OUT / tag)."""

    from .sft_data import MP16Images

    photos = json.loads((PHOTOS if tag == "full" else PHOTOS.parents[1] / tag / "dev.json").read_text(encoding="utf-8"))
    images = MP16Images() if tag != "full" else None
    prompt = PROMPT + (HINT if hint else "")

    def run(e: dict) -> str | None:
        body = {"model": "vlm", "temperature": 0.0, "max_tokens": 60, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(_image(e["path"]) if e.get("path") else images.read(e["image_id"])).decode()}},
                    {"type": "text", "text": prompt}]}]}
        request = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.loads(response.read())["choices"][0]["message"]["content"]
        except Exception:
            return None

    with ThreadPoolExecutor(workers) as pool:
        answers = list(pool.map(run, photos))
    path = (OUT if tag == "full" else OUT / tag) / f"check_{name}{'_hint' if hint else ''}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({e["image_id"]: a for e, a in zip(photos, answers)}) + "\n", encoding="utf-8")
    print(f"{len(answers)} answers, {sum(a is None for a in answers)} failed -> {path}")


def _norm(text: str | None) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _right(said: str, truth: str | None) -> bool | None:
    if not truth:
        return None
    a, b = _norm(said), _norm(truth)
    return bool(a) and (a == b or a in b or b in a)


def report() -> None:
    from .stage1_eval import _bootstrap

    photos = json.loads(PHOTOS.read_text(encoding="utf-8"))
    truth = {p["image_id"]: p["truth"] for p in json.loads(TRUTH.read_text(encoding="utf-8"))}
    top1_km = np.asarray([_haversine_km(*e["truth"], np.asarray([e["candidates"][0][1:3]]))[0] for e in photos])
    wrong = top1_km >= 25
    runs = sorted(OUT.glob("check_*.json"))
    hits: dict[str, dict[str, np.ndarray]] = {}
    for path in runs:
        answers = json.loads(path.read_text(encoding="utf-8"))
        name = path.stem.removeprefix("check_")
        hits[name] = {}
        for level, key in LEVELS:
            out = np.full(len(photos), np.nan)
            for i, e in enumerate(photos):
                parts = [p.strip() for p in (answers.get(e["image_id"]) or "").split(">")]
                said = parts[[lv for lv, _ in LEVELS].index(level)] if len(parts) > [lv for lv, _ in LEVELS].index(level) else ""
                r = _right(said, truth[e["image_id"]].get(key))
                out[i] = np.nan if r is None else float(r)
            hits[name][level] = out
        fmt = np.mean([len((answers.get(e["image_id"]) or "").split(">")) >= 3 for e in photos])
        print(f"{name}: answers in 'a > b > c' form {fmt:.1%}")
    print(f"\nfull eval halves n={len(photos)}; reranker top-1 >= 25 km off on {wrong.sum()} ({wrong.mean():.1%})")
    for label, mask in (("all photos", np.ones(len(photos), bool)), ("reranker wrong (>= 25 km)", wrong), ("reranker right (< 25 km)", ~wrong)):
        print(f"\n{label}:")
        for level, _ in LEVELS:
            row = []
            for name, h in hits.items():
                v = h[level][mask]
                row.append(f"{name} {np.nanmean(v):.1%}")
            n = int((~np.isnan(next(iter(hits.values()))[level][mask])).sum())
            print(f"  {level:8s} (n={n}): " + "   ".join(row))
        base = next((n for n in hits if n.startswith("base")), None)
        for name in hits:
            if name.startswith("base") or base is None:
                continue
            for ref in [n for n in hits if n.startswith("base")]:
                diffs = []
                for level, _ in LEVELS:
                    a, b = hits[name][level][mask], hits[ref][level][mask]
                    ok = ~np.isnan(a) & ~np.isnan(b)
                    c = _bootstrap(a[ok] - b[ok])
                    diffs.append(f"{level} {100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]")
                print(f"  {name} - {ref}: " + "; ".join(diffs))


def _matches(parts: list[str], lab: dict, depth: int) -> bool:
    """The answer's first `depth` levels all name the candidate's divisions at those levels."""

    return len(parts) >= depth and all(_right(parts[d], lab.get(LEVELS[d][1])) for d in range(depth))


def _rule_hits(name: str, tag: str) -> tuple[dict[tuple[str, int], np.ndarray], np.ndarray]:
    """Per variant (level, K), the km from the chosen candidate to the truth per photo; and the reranker top-1's km."""

    photos = json.loads((PHOTOS if tag == "full" else PHOTOS.parents[1] / tag / "dev.json").read_text(encoding="utf-8"))
    labels = {p["image_id"]: p["pool"] for p in json.loads((TRUTH.parent / f"candidates_{tag}.json").read_text(encoding="utf-8"))}
    answers = json.loads(((OUT if tag == "full" else OUT / tag) / f"check_{name}.json").read_text(encoding="utf-8"))
    km = [_haversine_km(*e["truth"], np.asarray(e["pool"])) for e in photos]
    out = {}
    for level in ("country", "region", "city", "deepest"):
        for k in (3, 10, 100):
            chosen = np.zeros(len(photos))
            for i, e in enumerate(photos):
                parts = [x.strip() for x in (answers.get(e["image_id"]) or "").split(">")]
                pool = labels[e["image_id"]][:k]
                depths = (1, 2, 3) if level == "country" else (2,) if level == "region" else (3,) if level == "city" else (3, 2, 1)
                pick = 0
                for depth in (depths if level == "deepest" else depths[-1:]):
                    hit = next((j for j, lab in enumerate(pool) if _matches(parts, lab, depth)), None)
                    if hit is not None:
                        pick = hit
                        break
                chosen[i] = km[i][pick]
            out[(level, k)] = chosen
    return out, np.asarray([d[0] for d in km])


def rule(name: str) -> None:
    from .stage1_eval import _bootstrap

    chosen = None
    for tag in ("dev", "full"):
        hits, base = _rule_hits(name, tag)
        print(f"\n{tag} (n={len(base)}), {name}: reranker top-1 < 1 / 25 / 200 km {(base < 1).mean():.1%} / {(base < 25).mean():.1%} / {(base < 200).mean():.1%}")
        if chosen is None:  # pick on dev: best change at 25 km, then at 1 km
            chosen = max(hits, key=lambda v: ((hits[v] < 25).mean(), (hits[v] < 1).mean()))
        for v, d in hits.items():
            parts = []
            for km in (1, 25, 200):
                c = _bootstrap((d < km) * 1.0 - (base < km))
                parts.append(f"< {km} km {100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]")
            fixes, breaks = ((d < 25) & (base >= 25)).sum(), ((d >= 25) & (base < 25)).sum()
            label = f"{v[0]}, top {v[1] if v[1] < 100 else 'all'}"
            print(f"  {label:15s} switched {(d != base).mean():5.1%}: " + "; ".join(parts) + f"; fixes {fixes}, breaks {breaks}" + ("   <- chosen on dev" if v == chosen else ""))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("answer", "report", "rule"))
    parser.add_argument("--name", default="base")
    parser.add_argument("--hint", action="store_true", help="append a format example to the prompt")
    parser.add_argument("--tag", default="full", help="photo set: full (benchmark eval halves) or an MP16 set (dev, val)")
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    args = parser.parse_args(argv)
    if args.node == "answer":
        answer(args.name, args.hint, args.tag, args.server)
    elif args.node == "report":
        report()
    else:
        rule(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
