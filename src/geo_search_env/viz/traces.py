"""Per-photo traces of the lesson 109 pipeline, built from its caches (photo set `full` = the test set, `dev3k`) and the saved agent (viz.agent).

One JSON per photo (artifacts/viz/traces/{set}/{i}.json) with every stage's inputs and outputs:
  released  - the released Pinpoint's 12 candidates (original protocol, photographer's photos kept);
  retrained - the retrained reranker's (setft_own) wide pool with set-layer probabilities;
  clues     - the 4B's reading (text, place names, features, country guess), its search phrases, text-search places and the clue judge's P(yes);
  areas     - candidate areas (Pinpoint agreement clusters + clue places): pre-zoom features, MegaLoc columns, the zoomed (restricted) search
              and both answer points (unzoomed: best global candidate inside; zoomed: the gated kernel pick);
  agent     - the RL policy's greedy steps (lambda 0 and 0.005) with action and answer probabilities, and fixed-tool answers.
Plus artifacts/viz/traces/{set}/index.json (one summary row per photo).

  PYTHONPATH=src .venv/bin/python -m geo_search_env.viz.traces --sets full dev3k
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .agent import CKPT_TAG, OUT

TRACES = OUT / "traces"
PRE_KEYS = ("support", "density", "read_n", "read_p", "read_share", "recall_n", "recall_p", "recall_share", "d_top1", "set_mass", "set_top_in",
            "d_settop")
POST_KEYS = ("top1", "top3", "mass_0.2", "n_cand", "photographers", "r_support", "r_areas", "r_spread")
ML_KEYS = ("best", "top3", "best_vs_others", "top3_vs_others")
FIXED = {"glance": (False, False, False), "read": (True, False, False), "recognise": (False, True, False), "zoom all": (False, False, True),
         "all tools": (True, True, True)}


def hav(a, b) -> np.ndarray:
    a, b = np.radians(np.asarray(a, np.float64)), np.radians(np.asarray(b, np.float64))
    h = np.sin((b[..., 0] - a[..., 0]) / 2) ** 2 + np.cos(a[..., 0]) * np.cos(b[..., 0]) * np.sin((b[..., 1] - a[..., 1]) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def r(x, nd=4):
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


class Gallery:
    """Attention-index row -> MP16 photo id, photographer and capture place (MP16 CSV), resolved for the rows asked for."""

    def __init__(self):
        from ..models.pinpoint_reranker import _discover_index
        from ..experiment.strategy_search import MP16_EMBED, _memmap_gallery

        index_dir = _discover_index()
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        self.cache = np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
        self.row_index = np.asarray(_memmap_gallery(MP16_EMBED)["row_index"])
        self.meta: dict[int, dict] = {}

    def resolve(self, rows) -> None:
        """Read the CSV rows of these index rows (one pass over the MP16 CSV)."""

        from ..experiment.strategy_search import MP16_CSV

        rows = np.unique(np.asarray(list(rows), np.int64))
        rows = rows[rows >= 0]
        want = {int(self.row_index[self.cache[x]]): int(x) for x in rows if int(x) not in self.meta}
        if not want:
            return
        with MP16_CSV.open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            head = next(reader)
            col = {k: head.index(k) for k in ("IMG_ID", "AUTHOR", "city", "state", "country")}
            for i, rec in enumerate(reader):
                if i in want:
                    place = ", ".join(x for x in (rec[col["city"]] or rec[col["state"]], rec[col["country"]]) if x)
                    self.meta[want[i]] = {"id": rec[col["IMG_ID"]], "author": rec[col["AUTHOR"]], "place": place}

    def cand(self, row: int, lat: float, lon: float, truth, **extra) -> dict:
        m = self.meta.get(int(row), {})
        out = {"row": int(row), "mp16": m.get("id"), "author": m.get("author"), "place": m.get("place", ""), "lat": r(lat, 5), "lon": r(lon, 5)}
        if truth is not None:
            out["km"] = r(hav(truth, [lat, lon]), 2)
        return out | {k: r(v) if isinstance(v, (float, np.floating)) else v for k, v in extra.items()}


def _clue_rows(tag: str) -> dict:
    """image_id -> observe / queries / evidence rows (+ judge P(yes), whether the pipeline used the row) for photo set tag."""

    from ..experiment.clue_judge import _items, _judge_path

    A = Path("artifacts/agent")
    load = lambda p: json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    obs, qs = load(A / f"observe_{tag}_4b.json"), load(A / f"queries_{tag}_4b.json")
    ev = load(A / f"region_evidence_{tag}_4b.json") or []
    judge = {(x["image_id"], x["query"]): x["p_yes"] for x in (load(_judge_path(tag, "4b", "4b")) or [])}
    used = {(x["image_id"], x["query"], x["source"]) for x in _items(tag, "4b")} if ev else set()
    out: dict[str, dict] = {}
    for k, v in obs.items():
        out.setdefault(k, {})["observe"] = v.get("parsed") or {}
    for k, v in qs.items():
        out.setdefault(k, {})["queries"] = v
    for x in ev:
        row = {"query": x["query"], "source": x["source"], "kind": x.get("kind"), "matches": x.get("matches"), "share": r(x.get("share", 0), 3),
               "lat": r(x["lat"], 5) if "lat" in x else None, "lon": r(x["lon"], 5) if "lon" in x else None,
               "region": x.get("region"), "p_yes": r(judge[(x["image_id"], x["query"])], 3) if (x["image_id"], x["query"]) in judge else None,
               "used": (x["image_id"], x["query"], x["source"]) in used}
        out.setdefault(x["image_id"], {}).setdefault("evidence", []).append(row)
    return out


def _released(tag: str, photos: list[dict]):
    """Per photo: the released Pinpoint's candidates (index rows, lat/lon, scores), original protocol, best first."""

    if tag == "dev3k":  # lesson 108's extraction with only the query photo excluded (dev3k photo set order)
        z = np.load("artifacts/pinpoint_set/mp16_32_8_8_dev3k_released_own.npz")
        ok = (z["rows"] >= 0) & (z["features"][..., 6:9] > 0).any(-1)
        rows, ll, sc = z["rows"], z["latlon"], np.where(ok, z["scores"], -np.inf)
        sel = np.arange(len(photos))
    else:
        z = np.load("artifacts/pinpoint_reference/bench_own.npz")
        sel = np.asarray([e["index"] for e in photos])
        rows, ll, sc = z["rows"], z["latlon"], np.where(z["rows"] >= 0, z["scores"], -np.inf)
    out = []
    for i in sel:
        o = np.argsort(-sc[i])[:12]
        o = o[np.isfinite(sc[i][o])]
        out.append((rows[i][o], ll[i][o], sc[i][o]))
    return out


def _unzoomed_pick(centers: np.ndarray, photo: np.ndarray, photos: list[dict]) -> np.ndarray:
    """Pool position of each area's unzoomed answer (hypothesis_agent._unzoomed_km): the best-scored global candidate within its BOX_KM, -1 = centre."""

    from ..experiment.hypothesis_agent import BOX_KM
    from ..experiment.strategy_search import _haversine_km

    out = np.full(len(photo), -1)
    for i, (ph, c) in enumerate(zip(photo, centers)):
        inside = np.flatnonzero(_haversine_km(c[0], c[1], np.asarray(photos[ph]["pool"], np.float64)) < BOX_KM)
        out[i] = inside[0] if len(inside) else -1
    return out


def _kernel_pick(zm, rows: np.ndarray) -> np.ndarray:
    """Candidate position of each zoomed area's answer under the gated soft-kernel rule (hypothesis_agent._kernel_km, gate=True), -1 = empty."""

    from ..experiment.hypothesis_agent import BOX_KM, GATE_KM, KERNEL_KM, KERNEL_T
    from ..experiment.pinpoint_set import _haversine_matrix

    picks = np.full(len(rows), -1)
    for c in range(0, len(rows), 8192):
        r = rows[c:c + 8192]
        valid = zm["rows"][r] >= 0
        s = np.where(valid, zm["scores"][r], -np.inf)
        best = np.where(valid.any(1, keepdims=True), s.max(1, keepdims=True), 0.0)
        w = np.where(valid, np.exp((s - best) / KERNEL_T), 0.0)
        lat = np.where(valid[..., None], zm["latlon"][r], 0.0)
        pair = _haversine_matrix(lat)
        dens = (w[:, None, :] * np.exp(-pair / KERNEL_KM)).sum(2)
        wb = np.where(valid, np.exp((s - best) / 0.2), 0.0)
        near = (pair < BOX_KM) & valid[:, None, :] & valid[:, :, None]
        founder = np.where(valid, (near * wb[:, None, :]).sum(2), -1).argmax(1)
        pick = np.where(near[np.arange(len(r)), founder], dens, -np.inf).argmax(1)  # within the founder's group, as the gated rule
        pick = np.where(pair[np.arange(len(r)), pick, founder] <= GATE_KM, pick, founder)
        picks[c:c + 8192] = np.where(valid.any(1), pick, -1)
    return picks


def build(tag: str, limit: int | None = None, *, d: dict | None = None, released: list | None = None, out_dir: Path | None = None,
          agent=None, gallery: "Gallery | None" = None) -> list[dict]:
    """Traces of photo set tag (the photo set {tag}_{CKPT_TAG} and its caches, relative to the working directory). The live worker (viz.live)
    passes its own agent batch d, released candidates, loaded agent and gallery; out_dir=False returns the traces without writing."""

    import torch

    from ..experiment import action_agent as aa
    from ..experiment.area_model import _paths
    from ..experiment.hypothesis_agent import SOURCES
    from ..experiment.pinpoint_tool import areas as agreement_areas
    from . import agent as va
    from .places import labels

    va._aa()
    net, pols, stats = agent or va.load_agent()
    d = d if d is not None else aa._data(tag, stats=stats, ckpt=CKPT_TAG)
    n = d["n"]
    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
    truth = np.asarray([e["truth"] for e in photos], np.float64)
    feat_path, zoom_path = _paths(tag, CKPT_TAG)
    f = dict(np.load(feat_path))
    zm = {k: v for k, v in np.load(zoom_path).items() if k != "hidden"}
    ml = np.load(Path("artifacts/agent") / f"megaloc_{tag}_{CKPT_TAG}.npz")["x"]
    sp = np.load(Path("artifacts/pinpoint_set") / f"set_probs_{tag}_{CKPT_TAG}.npz")
    set_probs = sp[f"{tag}_{CKPT_TAG}_probs"]
    clue_tag = f"{tag}_{CKPT_TAG}"
    clues = _clue_rows(clue_tag)
    released = released if released is not None else _released(tag, photos)

    # area rows in the agent's slot order (hypothesis_agent.load)
    rows = np.flatnonzero(np.isin(f["source"], SOURCES))
    slot_of = np.zeros(len(f["photo"]), int)
    seen = np.zeros(n, int)
    for j in rows:
        slot_of[j] = seen[f["photo"][j]]
        seen[f["photo"][j]] += 1
    by_photo: dict[int, list[int]] = {}
    for j in rows:
        by_photo.setdefault(int(f["photo"][j]), []).append(int(j))
    un_pick_of = dict(zip(rows.tolist(), _unzoomed_pick(zm["center"][rows], f["photo"][rows], photos).tolist()))
    kern_of = dict(zip(rows.tolist(), _kernel_pick(zm, rows).tolist()))

    # the agent: greedy traces for each lambda, fixed-tool answers, final km
    sel = np.arange(n) if limit is None else np.arange(min(n, limit))
    b = aa._sub(d, torch.as_tensor(sel, device="cuda"))
    steps = {lam: va.trace(net, pol, b) for lam, pol in pols.items()}
    fixed = {name: va.fixed_answer(net, b, *cfg) for name, cfg in FIXED.items()}
    kmz, kmu = d["km_zoom"][sel].cpu().numpy(), d["km_unzoom"][sel].cpu().numpy()

    g = gallery or Gallery()
    need = set()
    for i in sel:
        need.update(np.asarray(photos[i]["pool_rows"]).tolist())
        need.update(released[i][0].tolist())
        for j in by_photo.get(int(i), []):
            need.update(zm["rows"][j].tolist())
    print(f"{tag}: resolving {len(need)} gallery photos", flush=True)
    g.resolve(need)

    write = out_dir is not False
    out_dir = TRACES / tag if out_dir is None else out_dir
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
    index, traces = [], []
    mismatch = {"unzoomed": 0, "zoomed": 0}
    for i in sel:
        e = photos[i]
        t = truth[i]
        image = {"kind": "bench", "bench": e["benchmark"], "file": e["image_id"]} if "benchmark" in e else {"kind": "mp16", "id": e["image_id"]}
        rr, rl, rs = released[i]
        rel = [g.cand(x, *ll, t, score=s) for x, ll, s in zip(rr, rl, rs)]
        pool = [g.cand(x, *ll, t, p=set_probs[i][k]) for k, (x, ll) in enumerate(zip(e["pool_rows"], e["pool"]))]
        area_list = []
        for j in by_photo.get(int(i), []):
            valid = zm["rows"][j] >= 0
            cands = [g.cand(zm["rows"][j][k], *zm["latlon"][j][k], t, score=zm["scores"][j][k]) for k in np.flatnonzero(valid)]
            kp = kern_of[j]
            if kp < 0 and valid.any():  # empty kernel: the agreement founder (hypothesis_compare km_agree)
                kp = agreement_areas(zm["scores"][j][valid].astype(np.float64), zm["latlon"][j][valid].astype(np.float64))[0].founder
            up = un_pick_of[j]
            upt = e["pool"][up] if up >= 0 else zm["center"][j].tolist()
            slot = int(slot_of[j])
            area_list.append({
                "slot": slot, "source": "pinpoint" if f["source"][j] == 0 else "clue", "rank": int(f["rank"][j]),
                "center": [r(zm["center"][j][0], 5), r(zm["center"][j][1], 5)],
                "pre": {k: r(f[k][j]) for k in PRE_KEYS if k in f}, "post": {k: r(f[k][j]) for k in POST_KEYS if k in f},
                "megaloc": {k: r(x) for k, x in zip(ML_KEYS, ml[j])},
                "unzoomed": {"pool_index": int(up), "lat": r(upt[0], 5), "lon": r(upt[1], 5), "km": r(kmu[i, slot], 2)},
                "zoomed": {"cand_index": int(kp) if valid.any() else None, "km": r(kmz[i, slot], 2),
                           "lat": cands[int(kp)]["lat"] if valid.any() else None, "lon": cands[int(kp)]["lon"] if valid.any() else None},
                "candidates": cands,
            })
        area_list.sort(key=lambda a: a["slot"])
        for a in area_list:  # the answer points shown must be the ones the agent's km came from
            for kind in ("unzoomed", "zoomed"):
                pt = a[kind]
                if pt["lat"] is not None and pt["km"] is not None and abs(hav(t, [pt["lat"], pt["lon"]]) - pt["km"]) > max(0.05, 1e-3 * pt["km"]):
                    mismatch[kind] += 1
        h = len(area_list)
        agent = {}
        for lam, st in steps.items():
            sl = []
            for s in st[i if limit is None else int(np.flatnonzero(sel == i)[0])]:
                s = dict(s)
                s["zoomed"] = s["zoomed"][:h]
                s["answer"] = [r(x) for x in s["answer"][:h]]
                if "p_zoom" in s:
                    s["p_zoom"] = [r(x) for x in s["p_zoom"][:h]]
                    s["p_read"], s["p_recognise"], s["p_stop"] = r(s["p_read"]), r(s["p_recognise"]), r(s["p_stop"])
                sl.append(s)
            last = sl[-1]
            a = area_list[last["slot"]]
            zoomed = last["zoomed"][last["slot"]]
            pt = a["zoomed"] if zoomed else a["unzoomed"]
            agent[f"{lam:g}"] = {"steps": sl, "final": {"slot": last["slot"], "zoomed": zoomed, "lat": pt["lat"], "lon": pt["lon"], "km": pt["km"]}}
        k0 = int(np.flatnonzero(sel == i)[0])
        fx = {name: [r(x) for x in p[k0][:h]] for name, p in fixed.items()}
        pts = [[c["lat"], c["lon"]] for c in rel[:1] + pool[:1]] + [[a["center"][0], a["center"][1]] for a in area_list]
        pts += [[agent[k]["final"]["lat"], agent[k]["final"]["lon"]] for k in agent]
        lab = labels([t] + pts)
        trace = {
            "set": tag, "i": int(i), "image_id": e["image_id"], "benchmark": e.get("benchmark", tag), "image": image,
            "truth": {"lat": r(t[0], 5), "lon": r(t[1], 5), "place": lab[0]},
            "released": {"candidates": rel, "top1_place": lab[1]}, "retrained": {"candidates": pool, "top1_place": lab[2]},
            "clues": clues.get(e["image_id"], {}), "areas": area_list, "agent": agent, "fixed": fx,
        }
        for a, lb in zip(area_list, lab[3:3 + h]):
            a["place"] = lb
        for k, lb in zip(agent, lab[3 + h:]):
            agent[k]["final"]["place"] = lb
        traces.append(trace)
        if write:
            (out_dir / f"{int(i)}.json").write_text(json.dumps(trace, separators=(",", ":")), encoding="utf-8")
        st0 = agent["0"]["steps"]
        index.append({"i": int(i), "image_id": e["image_id"], "benchmark": trace["benchmark"], "truth_place": lab[0],
                      "released_km": rel[0]["km"] if rel else None, "retrained_km": pool[0]["km"] if pool else None,
                      "final_km": agent["0"]["final"]["km"], "final_km_005": agent["0.005"]["final"]["km"] if "0.005" in agent else None,
                      "actions": [s["action"] if s["action"] != "zoom" else f"zoom{s['slot']}" for s in st0[:-1]],
                      "final_source": area_list[agent["0"]["final"]["slot"]]["source"] if area_list else None, "n_areas": h,
                      "has_text": bool((trace["clues"].get("observe") or {}).get("text"))})
    if any(mismatch.values()):
        print(f"WARNING: answer points disagree with the agent's km: {mismatch}", flush=True)
    if not write:
        return traces
    (out_dir / "index.json").write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")
    km = np.asarray([x["final_km"] if x["final_km"] is not None else np.inf for x in index])
    rk = np.asarray([x["released_km"] if x["released_km"] is not None else np.inf for x in index])
    print(f"{tag}: {len(index)} traces -> {out_dir}; final < 25 / 1 km {np.mean(km < 25):.1%} / {np.mean(km < 1):.1%}, "
          f"released {np.mean(rk < 25):.1%} / {np.mean(rk < 1):.1%}", flush=True)
    return traces


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", default=["full", "dev3k"])
    parser.add_argument("--limit", type=int, help="first N photos only (plumbing)")
    args = parser.parse_args(argv)
    for tag in args.sets:
        build(tag, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
