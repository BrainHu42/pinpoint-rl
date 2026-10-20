# Named-candidate contrast with a local VLM (lesson 105 with Sonnet: +0.94 pts at 25 km on dev3k; the user: no paid LLMs). The query photo
# and the head's top two areas named as "near <town> (lat, lon), country"; the model answers A or B. P(A) from the first-token logprobs,
# both orders averaged (the 4B is position-biased, lesson 99); follow the model when P(other area) >= a threshold.
# Pairs, names and the Sonnet answers: artifacts/agent/names_contrast_dev3k_sonnet.json (202 dev3k pairs decisive at 25 or 1 km).
# Usage: NAME=4b RUN="names_judge ask" bash scripts/clue_judge.sh      then  .venv/bin/python -m geo_search_env.experiment.names_judge report

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
SET = OUT / "names_contrast_dev3k_sonnet.json"

PROMPT = """You are an expert geolocator. This photo was taken within about 25 km of exactly one of two places:
A = {a}
B = {b}
Use text and language, brands, landmarks, events, architecture, landscape and terrain, vegetation, climate and what you know about each
place. Answer with the single letter A or B."""

REASON = """You are an expert geolocator. This photo was taken within about 25 km of exactly one of two places:
A = {a}
B = {b}
First name the single most decisive piece of evidence (text and language, brands, landmarks, events, architecture, landscape and
terrain, vegetation, climate, what you know about each place), then pick A or B and give a confidence from 0.5 (pure guess) to 1.0;
only go above 0.7 with specific evidence."""
THINK = """You are an expert geolocator. This photo was taken within about 25 km of exactly one of two places:
A = {a}
B = {b}
Think step by step: list what the photo shows that bears on location (text and language, brands, landmarks, events, architecture,
landscape and terrain, vegetation, climate), what you know about each place, and which observations favour A or B and how strongly.
Then pick A or B and give a confidence from 0.5 (pure guess) to 1.0; only go above 0.7 with specific evidence."""
THINK_SCHEMA = {"type": "json_schema", "json_schema": {"name": "pick", "schema": {
    "type": "object", "properties": {"reasoning": {"type": "string", "maxLength": 4000}, "answer": {"type": "string", "enum": ["A", "B"]},
                                     "confidence": {"type": "number"}}, "required": ["reasoning", "answer", "confidence"], "additionalProperties": False}}}
SCHEMA = {"type": "json_schema", "json_schema": {"name": "pick", "schema": {
    "type": "object", "properties": {"evidence": {"type": "string", "maxLength": 200}, "answer": {"type": "string", "enum": ["A", "B"]},
                                     "confidence": {"type": "number"}}, "required": ["evidence", "answer", "confidence"], "additionalProperties": False}}}


def _images(set_: dict):
    """Query JPEGs (base64 data URLs, long side 768) per example, re-read from MP16."""

    from PIL import Image

    from .name_score import _read
    from .sft_data import MP16Images

    photos = json.loads((Path("artifacts/photo_sets") / "dev3k_setft_a" / "photos.json").read_text(encoding="utf-8"))
    pr = json.loads((OUT / "contrast_pairs_dev3k.json").read_text(encoding="utf-8"))  # contrast_gate's pairs (lesson 99)
    images = MP16Images()
    out = {}
    for k in set_["key"]:
        im = Image.open(io.BytesIO(_read(photos[pr[k["pair"]]["photo"]], images))).convert("RGB")
        im.thumbnail((768, 768))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
        out[k["ex"]] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return out


def ask(name: str = "4b", server: str = "http://127.0.0.1:8765", workers: int = 16, reason: bool = False, think: bool = False) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from .name_score import _post

    set_ = json.loads(SET.read_text(encoding="utf-8"))
    cands = {m.group(1): (m.group(2), m.group(3)) for m in re.finditer(r"(ex\d+): A = (.*?); B = (.*)", set_["candidates"])}
    imgs = _images(set_)

    def run(job):
        ex, order = job
        a, b = cands[ex] if order == 0 else cands[ex][::-1]
        if reason or think:  # think: free step-by-step reasoning first (models without a thinking mode, e.g. Gemma 3)
            body = {"model": "vlm", "temperature": 0.0, "max_tokens": 1200 if think else 160, "response_format": THINK_SCHEMA if think else SCHEMA,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": imgs[ex]}},
                                                              {"type": "text", "text": (THINK if think else REASON).format(a=a, b=b)}]}]}
            try:
                js = json.loads(_post(server, "/v1/chat/completions", body)["choices"][0]["message"]["content"])
                c = min(1.0, max(0.5, float(js["confidence"])))
                return {"ex": ex, "order": order, "p_first": c if js["answer"] == "A" else 1 - c,
                        "evidence": js["reasoning" if think else "evidence"]}
            except Exception as e:  # noqa: BLE001
                return {"ex": ex, "order": order, "error": str(e)[:200]}
        body = {"model": "vlm", "temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": imgs[ex]}},
                                                          {"type": "text", "text": PROMPT.format(a=a, b=b)}]}]}
        try:
            top = _post(server, "/v1/chat/completions", body)["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except Exception as e:  # noqa: BLE001
            return {"ex": ex, "order": order, "error": str(e)[:200]}
        p = {"A": 0.0, "B": 0.0}
        for t in top:
            s = t["token"].strip().upper()
            if s in p:
                p[s] += math.exp(t["logprob"])
        pa = p["A"] / max(p["A"] + p["B"], 1e-9)
        return {"ex": ex, "order": order, "p_first": pa}  # P(the place shown as A in this order)

    jobs = [(ex, o) for ex in cands for o in (0, 1)]
    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(run, jobs))
    path = OUT / f"names_judge_dev3k_{name}.json"
    path.write_text(json.dumps(res), encoding="utf-8")
    print(f"-> {path}: {len(res)} answers, {sum('error' in r for r in res)} errors")


def pmi(name: str, server: str = "http://127.0.0.1:8765", workers: int = 16) -> None:
    """Score each named area instead of asking for a choice: log P(name | photo) - log P(name | no photo) of the place in "This photo was
    taken near <town>, <country>." (prompt logprobs, the assistant turn continued), and the same for "taken in <country>."."""

    from concurrent.futures import ThreadPoolExecutor

    from .name_score import _post

    set_ = json.loads(SET.read_text(encoding="utf-8"))
    cands = {m.group(1): (m.group(2), m.group(3)) for m in re.finditer(r"(ex\d+): A = (.*?); B = (.*)", set_["candidates"])}
    imgs = _images(set_)

    def lp(ex, image, text):  # summed prompt logprob of the assistant text, and its token count
        user = [{"type": "text", "text": "Where was this photo taken?"}]
        if image:
            user = [{"type": "image_url", "image_url": {"url": imgs[ex]}}] + user
        body = {"model": "vlm", "max_tokens": 1, "prompt_logprobs": 0, "add_generation_prompt": False, "continue_final_message": True,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": text}]}
        out = [next(iter(d.values()))["logprob"] for d in _post(server, "/v1/chat/completions", body)["prompt_logprobs"] if d]
        return out

    def run(job):
        ex, image = job
        res = {"ex": ex, "image": image}
        try:
            for level, prefix in (("place", "This photo was taken near"), ("country", "This photo was taken in")):
                n0 = len(lp(ex, image, prefix))
                for side, c in zip("AB", cands[ex]):
                    m = re.match(r"near (.*) \(.*?\), (.*)", c)
                    text = f"{prefix} {m.group(1)}, {m.group(2)}." if level == "place" else f"{prefix} {m.group(2)}."
                    res[f"{level}_{side}"] = float(sum(lp(ex, image, text)[n0:]))
        except Exception as e:  # noqa: BLE001
            res["error"] = str(e)[:200]
        return res

    jobs = [(ex, im) for ex in cands for im in (True, False)]
    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(run, jobs))
    path = OUT / f"names_judge_dev3k_{name}.json"
    path.write_text(json.dumps(res), encoding="utf-8")
    print(f"-> {path}: {len(res)} rows, {sum('error' in r for r in res)} errors")


def report_pmi(name: str) -> None:
    set_ = json.loads(SET.read_text(encoding="utf-8"))
    cands = {m.group(1): (m.group(2), m.group(3)) for m in re.finditer(r"(ex\d+): A = (.*?); B = (.*)", set_["candidates"])}
    res = {(r["ex"], r["image"]): r for r in json.loads((OUT / f"names_judge_dev3k_{name}.json").read_text(encoding="utf-8")) if "error" not in r}
    for level in ("place", "country"):
        for score in ("pmi", "photo only"):
            rows = []
            for k in set_["key"]:
                ex = k["ex"]
                if (ex, True) not in res or (ex, False) not in res:
                    continue
                if level == "country" and cands[ex][0].rsplit(", ", 1)[1] == cands[ex][1].rsplit(", ", 1)[1]:
                    continue
                s = {side: res[ex, True][f"{level}_{side}"] - (res[ex, False][f"{level}_{side}"] if score == "pmi" else 0) for side in "AB"}
                other = "B" if k["head"] == "A" else "A"
                rows.append((s[k["head"]] - s[other], k["km_head"], k["km_other"]))  # margin for the head's area, nats
            margin_report(np.asarray(rows), f"{name} {level}, {score}", "nats")


def margin_report(rows: np.ndarray, label: str, unit: str) -> None:
    """rows: (score margin for the head's area over the other, km of the head's area, km of the other) per pair; switches when the
    margin for the other area exceeds a threshold, fixes / breaks / net pts of dev3k (202 pairs drawn from ~500 decisive, lesson 105)."""

    print(f"=== {label}: {len(rows)} pairs")
    for thr in (25, 1):
        head_ok, other_ok = rows[:, 1] < thr, rows[:, 2] < thr
        dec = head_ok != other_ok
        acc = np.mean(np.where(rows[dec, 0] >= 0, head_ok[dec], other_ok[dec]))
        print(f"--- {thr} km: {dec.sum()} decisive, model right {acc:.1%} (head {head_ok[dec].mean():.1%}); "
              f"head-wrong {np.mean(rows[dec & ~head_ok, 0] < 0):.1%} right, head-right {np.mean(rows[dec & head_ok, 0] >= 0):.1%} right")
        for t in (0, 1, 2, 3, 4, 6):
            sw = rows[:, 0] < -t
            fx, br = int((sw & other_ok & ~head_ok).sum()), int((sw & head_ok & ~other_ok).sum())
            print(f"  follow when margin for the other area > {t} {unit}: switches {sw.sum()}, fixes {fx}, breaks {br}, "
                  f"net {100 * (fx - br) / 500 * 0.944:+.2f} pts of dev3k")


def report(name: str = "4b") -> None:
    set_ = json.loads(SET.read_text(encoding="utf-8"))
    res = json.loads((OUT / f"names_judge_dev3k_{name}.json").read_text(encoding="utf-8"))
    p = {}
    for r in res:
        if "p_first" in r:  # P(original A) from this order
            p.setdefault(r["ex"], []).append(r["p_first"] if r["order"] == 0 else 1 - r["p_first"])
    first = np.mean([r["p_first"] for r in res if "p_first" in r])
    print(f"{name}: mean P(first-shown place) {first:.2f} (0.5 = no position bias)")
    rows = []
    for k in set_["key"]:
        if len(p.get(k["ex"], [])) < 2:
            continue
        pa = float(np.mean(p[k["ex"]]))
        p_head = pa if k["head"] == "A" else 1 - pa
        rows.append((p_head, k["km_head"], k["km_other"]))
    rows = np.asarray(rows)
    for thr in (25, 1):
        head_ok, other_ok = rows[:, 1] < thr, rows[:, 2] < thr
        dec = head_ok != other_ok
        acc = np.mean(np.where(rows[dec, 0] >= 0.5, head_ok[dec], other_ok[dec]))
        print(f"--- {thr} km: {dec.sum()} decisive, model right {acc:.1%} (head {head_ok[dec].mean():.1%}); "
              f"head-wrong {np.mean(rows[dec & ~head_ok, 0] < 0.5):.1%} right, head-right {np.mean(rows[dec & head_ok, 0] >= 0.5):.1%} right")
        for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            sw = rows[:, 0] <= 1 - t
            fx, br = int((sw & other_ok & ~head_ok).sum()), int((sw & head_ok & ~other_ok).sum())
            print(f"  follow when P(other) >= {t}: switches {sw.sum()}, fixes {fx}, breaks {br}, net {100 * (fx - br) / 500 * 0.944:+.2f} pts of dev3k")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("ask", "report"))
    parser.add_argument("--name", default="4b")
    parser.add_argument("--tag", default="dev3k")
    args = parser.parse_args(argv)
    reason = args.name.endswith("_reason")  # NAME=27b_reason: evidence first, stated confidence
    think = args.name.endswith("_think")  # NAME=gemma27b_think: step-by-step reasoning first, stated confidence
    if args.name.endswith("_pmi"):  # NAME=27b_pmi: per-area name likelihood with vs without the photo, no choice asked
        pmi(args.name) if args.cmd == "ask" else report_pmi(args.name)
    else:
        ask(args.name, reason=reason, think=think) if args.cmd == "ask" else report(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
