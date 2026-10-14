# Gate for the query-RL plan (lesson 82): can the base 4B ever produce a place query the tools resolve near the truth? It sees the photo and
# Pinpoint's top areas and proposes up to 2 place queries or none, sampled K times (temperature 1); each query goes through tools.maps (top 3
# places) and text_search (top cluster; the photo and its photographer excluded). pass@K = any sample with a result < 25 km.
# Photos: the 302 fixable dev3k photos (reranker wrong, knowing the region would fix it) and 300 random dev3k photos (control).
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.query_passk sample     (VLM served at :8765)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.query_passk resolve
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.query_passk report

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
PHOTOS = Path("artifacts/photo_sets/dev3k_setft_a/photos.json")
K = 16
KM = 25.0
PROMPT = ("A visual search engine guesses this photo was taken near: {areas}.\n"
          "If the photo shows something a map or web search could find (a landmark, building, street, business, named place, or text naming a "
          "place), give up to 2 search queries for it, each as specific as you can (name plus town / country if you know them). If nothing in "
          'the photo is place-specific, answer none. Answer with JSON only: {{"queries": ["...", "..."]}} or {{"queries": []}}.')
BLIND = PROMPT.split("\n", 1)[1]  # the same question without Pinpoint's areas (does the 4B know anything beyond echoing them?)


def _photos() -> list[dict]:
    photos = json.loads(PHOTOS.read_text(encoding="utf-8"))
    fixable = {t["image_id"] for t in json.loads((OUT / "targets_dev3k.json").read_text(encoding="utf-8")) if t["fixable"]}
    rest = [i for i, e in enumerate(photos) if e["image_id"] not in fixable]
    control = set(np.random.default_rng(0).choice(rest, 300, replace=False).tolist())
    return [dict(e, group="fixable" if e["image_id"] in fixable else "control") for i, e in enumerate(photos) if e["image_id"] in fixable or i in control]


def _areas(e: dict, ts) -> str:
    labels = []
    for _, lat, lon in e["candidates"][:5]:
        lab = ts.label(lat, lon)
        if lab not in labels:
            labels.append(lab)
    return "; ".join(labels[:3])


def _file(variant: str, resolved: bool = False) -> Path:
    return OUT / f"query_passk_dev3k{'_' + variant if variant else ''}{'_resolved' if resolved else ''}.json"


def sample(variant: str = "", server: str = "http://127.0.0.1:8765", workers: int = 32) -> None:
    from .name_score import _post, _read
    from .sft_data import MP16Images
    from .text_search import TextSearch

    photos, images, ts = _photos(), MP16Images(), TextSearch(yfcc=False)

    def ask(e: dict) -> dict:
        body = {"model": "vlm", "temperature": 1.0, "top_p": 1.0, "n": K, "max_tokens": 120, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(_read(e, images)).decode()}},
                    {"type": "text", "text": BLIND.format() if variant == "blind" else PROMPT.format(areas=_areas(e, ts))}]}]}
        try:
            raw = [c["message"]["content"] for c in _post(server, "/v1/chat/completions", body)["choices"]]
        except Exception:  # noqa: BLE001
            raw = []
        out = []
        for r in raw:
            m = re.search(r"\{.*\}", r or "", re.S)
            try:
                out.append([str(q).strip() for q in (json.loads(m.group(0)).get("queries") or []) if str(q).strip()][:2] if m else [])
            except Exception:  # noqa: BLE001
                out.append([])
        return {"image_id": e["image_id"], "group": e["group"], "truth": e["truth"], "top1": e["candidates"][0][1:],
                "shown": [c[1:] for c in e["candidates"][:5]], "samples": out}

    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(ask, photos))
    _file(variant).write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(rows)} photos, {sum(not r['samples'] for r in rows)} failed -> {_file(variant)}")


def resolve(variant: str = "", workers: int = 16) -> None:
    from .region_evidence import _owns
    from .text_search import TextSearch, haversine_km
    from ..tools.maps import MapsSearch

    rows = json.loads(_file(variant).read_text(encoding="utf-8"))
    owns = _owns([{"image_id": r["image_id"]} for r in rows])
    ts, ms = TextSearch(), MapsSearch()
    jobs = sorted({(i, q) for i, r in enumerate(rows) for s in r["samples"] for q in s})
    print(f"{len(jobs)} distinct (photo, query) pairs", flush=True)

    def run(job):
        i, q = job
        t = np.asarray(rows[i]["truth"], np.float64)
        pts = []
        try:
            pts += [("maps", p.lat, p.lon) for p in ms.query(q)[:3]]
        except Exception:  # noqa: BLE001
            pass
        try:
            r = ts.search(q, exclude=owns.get(rows[i]["image_id"]))
            pts += [("text", c.lat, c.lon) for c in r.clusters[:1]]
        except Exception:  # noqa: BLE001
            pass
        km = [float(haversine_km(t[0], t[1], np.asarray([la]), np.asarray([lo]))[0]) for _, la, lo in pts]
        return {"src": [p[0] for p in pts], "km": km, "ll": [[round(la, 5), round(lo, 5)] for _, la, lo in pts]}

    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(run, jobs))
    out = {f"{i}\t{q}": v for (i, q), v in zip(jobs, res)}
    _file(variant, True).write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"-> {_file(variant, True)}")


def report(variant: str = "") -> None:
    from .text_search import haversine_km

    rows = json.loads(_file(variant).read_text(encoding="utf-8"))
    res = json.loads(_file(variant, True).read_text(encoding="utf-8"))
    print(f"variant: {variant or 'with areas'}")
    for group in ("fixable", "control"):
        idx = [i for i, r in enumerate(rows) if r["group"] == group]
        n_q, hit_any, hit_first, frac, top1_right, examples, novel = [], [], [], [], [], [], []
        for i in idx:
            r = rows[i]
            t = np.asarray(r["truth"], np.float64)
            top1_right.append(float(haversine_km(t[0], t[1], np.asarray([r["top1"][0]]), np.asarray([r["top1"][1]]))[0]) < KM)
            per_sample_any, per_sample_first = [], []
            shown = np.asarray(r["shown"], np.float64)
            novel_hit = False  # a result < 25 km of the truth and > 25 km from every area shown (not an echo of Pinpoint's pool)
            for s in r["samples"]:
                for q in s:
                    for k, (la, lo) in zip(res[f"{i}\t{q}"]["km"], res[f"{i}\t{q}"]["ll"]):
                        if k < KM and haversine_km(la, lo, shown[:, 0], shown[:, 1]).min() > KM:
                            novel_hit = True
                kms = [res[f"{i}\t{q}"]["km"] for q in s]
                per_sample_any.append(any(k < KM for ks in kms for k in ks))
                per_sample_first.append(any(ks and ks[0] < KM for ks in kms))  # the top result of a query (what an agent would take)
            n_q.append(np.mean([len(s) > 0 for s in r["samples"]]) if r["samples"] else 0.0)
            hit_any.append(any(per_sample_any))
            novel.append(novel_hit)
            hit_first.append(any(per_sample_first))
            frac.append(np.mean(per_sample_any) if per_sample_any else 0.0)
            if any(per_sample_any) and len(examples) < 8:
                good = next(q for s in r["samples"] for q in s if any(k < KM for k in res[f"{i}\t{q}"]["km"]))
                examples.append(good)
        n_q, hit_any, hit_first, frac, top1_right, novel = map(np.asarray, (n_q, hit_any, hit_first, frac, top1_right, novel))
        print(f"\n{group}: {len(idx)} photos, retrained-reranker top-1 right on {top1_right.mean():.1%}")
        print(f"  samples with any query: {n_q.mean():.1%}")
        print(f"  pass@{K} (any result of any query < {KM:.0f} km): {hit_any.mean():.1%}; top result only: {hit_first.mean():.1%}")
        print(f"  pass@1 (mean share of samples with a hit): {frac.mean():.1%}; photos hit in >= 25% of samples: {(frac >= 0.25).mean():.1%}")
        print(f"  pass@{K} where the top-1 is wrong: {hit_any[~top1_right].mean():.1%} of {(~top1_right).sum()}; "
              f"novel (> {KM:.0f} km from all 5 areas shown): {novel[~top1_right].mean():.1%}")
        print(f"  example hitting queries: {examples}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("sample", "resolve", "report"))
    parser.add_argument("--variant", default="", choices=("", "blind"), help="blind: the prompt without Pinpoint's areas")
    args = parser.parse_args(argv)
    {"sample": sample, "resolve": resolve, "report": report}[args.cmd](args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
