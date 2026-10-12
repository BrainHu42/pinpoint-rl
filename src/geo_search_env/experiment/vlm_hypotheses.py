# Does the VLM add hypotheses Pinpoint doesn't have? (HYPOTHESIS_PLAN.md, the check before training the agent, E5.) The base 4B proposes up to 3
# places for each photo from what it sees and knows; each is geocoded with the text-search databases and, if it lies more than BOX_KM from every
# existing hypothesis (Pinpoint areas, clue places), searched with Pinpoint restricted to a box around it. Measured: how much the hypothesis
# oracle rises (it bounds what an agent that proposes hypotheses could add). dev sets only.
# Usage: RUN="vlm_hypotheses propose" NAME=4b TAG=dev3k bash scripts/clue_judge.sh      (serves the 4B; proposals)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.vlm_hypotheses search --tag dev3k   (geocode + restricted search, GPU)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.vlm_hypotheses oracle --tag dev3k

"""propose: per photo, up to 3 places (most specific first) -> artifacts/agent/vlm_places_{tag}_{name}.json.
search:  geocoding: the first comma-separated part of each place searched by name (text_search.matches), keeping matches in the stated
         country when one is given; the heaviest 25 km cluster is the place. New ones -> hypotheses_{tag}_vlmhyps.npz (source 2).
oracle:  hypothesis oracle (restricted top-1 / any restricted candidate < 25 km) with Pinpoint areas, + clue places, + VLM places."""

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
PHOTO_SETS = Path("artifacts/photo_sets")
SFT_ROOT = Path("artifacts/sft")
NVME_INDEX = Path("/opt/dlami/nvme/pinpoint/attention_retrieval_index")
BOX_KM = 25.0
PROMPT = """You are an expert geolocator. Where was this photo most likely taken? Use everything you can see (text, landmarks, architecture,
vegetation, vehicles, people) and what you know. Give up to 3 different candidate places, most likely first, each as specific as you can
justify: "City, Country", or "Region, Country" if you can't name a city. Answer with JSON only: {"places": ["...", "..."]}."""


def propose(tag: str, name: str, server: str = "http://127.0.0.1:8765", workers: int = 32) -> None:
    from .name_score import _post, _read
    from .sft_data import MP16Images

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    images = MP16Images()

    def ask(e: dict) -> list[str]:
        image = base64.b64encode(_read(e, images)).decode()  # benchmark photos by path
        body = {"model": "vlm", "temperature": 0.0, "max_tokens": 120, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image}},
                                                          {"type": "text", "text": PROMPT}]}]}
        try:
            raw = _post(server, "/v1/chat/completions", body)["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", raw, re.S)
            return [str(x) for x in (json.loads(m.group(0)).get("places") or [])][:3] if m else []
        except Exception:  # noqa: BLE001
            return []

    with ThreadPoolExecutor(workers) as pool:
        answers = list(pool.map(ask, photos))
    path = OUT / f"vlm_places_{tag}_{name}.json"
    path.write_text(json.dumps({e["image_id"]: a for e, a in zip(photos, answers)}, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{tag} / {name}: places for {sum(bool(a) for a in answers)} of {len(photos)} photos -> {path}")


def _geocode(tool, place: str, name_to_cc: dict[str, str]) -> tuple[float, float] | None:
    from .text_search import haversine_km

    parts = [p.strip() for p in place.split(",") if p.strip()]
    if not parts:
        return None
    cc = name_to_cc.get(parts[-1].casefold()) if len(parts) > 1 else None
    _, places, _, _, _, _ = tool.matches(parts[0])
    pts = [(p[1], p[2]) for p in places if cc is None or p[3] == cc]
    if not pts and len(parts) > 1:  # e.g. "Region, Country": try the region part with the country, then the country itself
        _, places, _, _, _, _ = tool.matches(parts[-1])
        pts = [(p[1], p[2]) for p in places if cc is None or p[3] == cc]
    if not pts:
        return None
    a = np.asarray(pts, np.float64)
    support = [(haversine_km(la, lo, a[:, 0], a[:, 1]) < BOX_KM).sum() for la, lo in a]
    return tuple(a[int(np.argmax(support))])


def search(tag: str, name: str = "4b") -> None:
    from ..models.pinpoint_reranker import PinpointReranker
    from .strategy_search import _haversine_km, load_world
    from .text_search import TextSearch, _countries

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    proposals = json.loads((OUT / f"vlm_places_{tag}_{name}.json").read_text(encoding="utf-8"))
    existing: dict[int, list] = {}
    for f in (OUT / f"hypotheses_{tag}.npz", OUT / f"hypotheses_{tag}_clues.npz", OUT / f"hypotheses_{tag}_cluehyps.npz"):
        if f.exists():
            z = np.load(f)
            for ph, c in zip(z["photo"], z["center"]):
                existing.setdefault(int(ph), []).append(c)
    _, _, iso_names = _countries()
    name_to_cc = {v.casefold(): k for k, v in iso_names.items()} | {"usa": "US", "united states of america": "US", "uk": "GB", "england": "GB",
                                                                  "scotland": "GB", "wales": "GB", "russia": "RU", "south korea": "KR",
                                                                  "czech republic": "CZ", "vietnam": "VN", "taiwan": "TW"}
    tool = TextSearch(yfcc=False)
    hyps, n_geo = [], 0
    for p, e in enumerate(photos):
        kept: list = []
        for rank, place in enumerate(proposals.get(e["image_id"], [])):
            ll = _geocode(tool, place, name_to_cc)
            if ll is None:
                continue
            n_geo += 1
            here = np.asarray(existing.get(p, []) + kept, np.float64).reshape(-1, 2)
            if len(here) and (_haversine_km(ll[0], ll[1], here) < BOX_KM).any():
                continue
            kept.append(list(ll))
            hyps.append({"photo": p, "source": 2, "rank": rank, "support": 1.0 / (1 + rank), "center": list(ll)})
    print(f"{tag}: {n_geo} proposals geocoded, {len(hyps)} new hypotheses on {len({h['photo'] for h in hyps})} photos", flush=True)
    from .hypotheses import photo_world

    world, qrows = photo_world(photos)
    reranker = PinpointReranker(device="cuda", index_dir=NVME_INDEX if (NVME_INDEX / "manifest.json").exists() else None)
    pidx = np.asarray([h["photo"] for h in hyps])
    centre = np.asarray([h["center"] for h in hyps], np.float64)
    dlat = BOX_KM / 111.0
    dlon = dlat / np.maximum(np.cos(np.radians(centre[:, 0])), 0.05)
    box = np.stack([centre[:, 0] - dlat, centre[:, 0] + dlat, centre[:, 1] - dlon, centre[:, 1] + dlon], 1)
    found = reranker.rank(world.query_embeddings[qrows[pidx]], query_author=world.query_author[qrows[pidx]],
                          index_author=world.mp16["author"][reranker.cache_indices], query_box=box)
    np.savez(OUT / f"hypotheses_{tag}_vlmhyps.npz", photo=pidx, source=np.full(len(hyps), 2), rank=np.asarray([h["rank"] for h in hyps]),
             support=np.asarray([h["support"] for h in hyps]), center=centre, latlon=found.latlon, scores=found.scores, features=found.features,
             rows=found.rows)
    print(f"-> {OUT / f'hypotheses_{tag}_vlmhyps.npz'}")
    oracle(tag)


def oracle(tag: str) -> None:
    from .strategy_search import _haversine_km

    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    truth = np.asarray([e["truth"] for e in photos])
    base = np.asarray([_haversine_km(*e["truth"], np.asarray([e["pool"][0]]))[0] < 25 for e in photos])
    parts = []
    areas = OUT / f"hypotheses_{tag}.npz" if (OUT / f"hypotheses_{tag}.npz").exists() else OUT / f"hypotheses_{tag}_clues.npz"
    z = dict(np.load(areas))
    keep = z["source"] == 0
    parts.append(("Pinpoint areas (K = 5)", {k: v[keep] for k, v in z.items()}))
    for label, f in (("+ clue places", OUT / f"hypotheses_{tag}_cluehyps.npz"), ("+ VLM places", OUT / f"hypotheses_{tag}_vlmhyps.npz")):
        if f.exists():
            parts.append((label, dict(np.load(f))))
    top = np.zeros(len(photos), bool)
    anyc = np.zeros(len(photos), bool)
    print(f"{tag}: Pinpoint < 25 km {base.mean():.1%}")
    for label, h in parts:
        for i, ph in enumerate(h["photo"]):
            ll = h["latlon"][i]
            ok = np.isfinite(ll[:, 0])
            if ok.any():
                d = _haversine_km(*truth[ph], ll[ok])
                top[ph] |= d[0] < 25
                anyc[ph] |= (d < 25).any()
        print(f"  {label:24s} hypothesis oracle: restricted top-1 {top.mean():.1%}, any restricted candidate {anyc.mean():.1%} ({len(h['photo'])} hypotheses)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("propose", "search", "oracle"))
    parser.add_argument("--tag", default="dev3k")
    parser.add_argument("--name", default="4b")
    args = parser.parse_args(argv)
    {"propose": lambda: propose(args.tag, args.name), "search": lambda: search(args.tag, args.name), "oracle": lambda: oracle(args.tag)}[args.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
