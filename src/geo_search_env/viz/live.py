"""The lesson 109 pipeline on one uploaded photo, stage by stage, writing the same per-stage files the batch pipeline writes for a photo set
(under a synthetic tag, in a scratch working tree artifacts/viz/live_root) so the batch feature code and the agent loader run unchanged:

  SigLIP2 giant embedding -> released Pinpoint (12) and the retrained reranker's wide pool (setft_own, 32 8 8; set probabilities T 0.2)
  -> Pinpoint agreement areas (top 5) -> clue reading (Qwen3.5-4B over vLLM: observe + search phrases; text search; 4B clue judge; clue-place
  hypotheses, if the server is up) -> restricted searches in a 25 km box per hypothesis, re-scored for the reranker's CLS states (zoom)
  -> hypothesis_compare features -> MegaLoc similarities -> the saved answer head and RL policies (viz.agent) -> a trace (viz.traces).

No photographer exclusion (an uploaded photo has none). Without a true location the truth-dependent fields are computed against (0, 0) and
dropped from the trace. The worker runs as its own process (GPU, models resident):

  PYTHONPATH=src .venv/bin/python -m geo_search_env.viz.live worker            (started by `viz.server --live`)
  PYTHONPATH=src .venv/bin/python -m geo_search_env.viz.live run photo.jpg [--lat 48.85 --lon 2.29]   (one photo, prints the answer)

Clue reading needs the 4B served at $VIZ_VLM (default http://127.0.0.1:8766; scripts/viz_vlm.sh).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np

from ..paths import PROJECT_ROOT

REPO = PROJECT_ROOT
UPLOADS = REPO / "artifacts/viz/uploads"
LIVE_ROOT = REPO / "artifacts/viz/live_root"
CKPT_TAG = "setft_own"
CKPT = REPO / f"artifacts/pinpoint_train/{CKPT_TAG}/checkpoints/ckpt_final.pt"
VLM = os.environ.get("VIZ_VLM", "http://127.0.0.1:8766")
BOX_KM = 25.0
K_AREAS = 5


# ---------------------------------------------------------------- server side
class Worker:
    """Server-side handle: queues uploads as files and keeps one worker process (viz.live worker) running."""

    def __init__(self) -> None:
        UPLOADS.mkdir(parents=True, exist_ok=True)
        log = open(REPO / "artifacts/viz/live_worker.log", "a")
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
        cap = ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=16G", "-p", "MemorySwapMax=0"]  # 30 GB RAM, no swap
        self.proc = subprocess.Popen(cap + ["env", f"PYTHONPATH={REPO / 'src'}", sys.executable, "-m", "geo_search_env.viz.live", "worker"],
                                     cwd=REPO, env=env, stdout=log, stderr=log)

    def submit(self, data: bytes, name: str, truth: tuple[float, float] | None) -> str:
        job = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        d = UPLOADS / job
        d.mkdir(parents=True)
        (d / "upload.bin").write_bytes(data)
        (d / "status.json").write_text(json.dumps({"state": "queued", "message": "Queued…"}))
        (d / "request.json").write_text(json.dumps({"name": Path(name).name, "truth": truth}))
        return job

    def status(self, job: str) -> dict:
        d = UPLOADS / Path(job).name
        if self.proc.poll() is not None:
            return {"state": "error", "message": "the pipeline worker stopped (artifacts/viz/live_worker.log)"}
        try:
            s = json.loads((d / "status.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {"state": "queued", "message": "Queued…"}
        return s | {"id": job}


# ---------------------------------------------------------------- worker side
def _scratch() -> None:
    """The scratch working tree: shared inputs linked, per-upload outputs local, then chdir into it."""

    for sub in ("artifacts/agent", "artifacts/photo_sets", "artifacts/pinpoint_set", "artifacts/viz"):
        (LIVE_ROOT / sub).mkdir(parents=True, exist_ok=True)
    links = {"data": REPO / "data", "artifacts/place_labels": REPO / "artifacts/place_labels",
             "artifacts/agent/mp16_author_by_cache_row.npy": REPO / "artifacts/agent/mp16_author_by_cache_row.npy",
             "artifacts/viz/cities.npz": REPO / "artifacts/viz/cities.npz", "artifacts/viz/agent_setft_own.pt": REPO / "artifacts/viz/agent_setft_own.pt"}
    for rel, target in links.items():
        p = LIVE_ROOT / rel
        if p.is_symlink() and not p.exists():  # dangling (e.g. made from a since-deleted worktree)
            p.unlink()
        if not p.is_symlink() and not p.exists():
            p.symlink_to(target)
    os.chdir(LIVE_ROOT)


class Pipeline:
    def __init__(self, log=print) -> None:
        self.log = log
        _scratch()
        os.environ["PINPOINT_RERANKER_CKPT"] = str(CKPT)
        import torch

        from ..experiment import embed_cache
        from ..experiment.megaloc_gate import _model as megaloc_model
        from ..models import pinpoint_reranker as pr
        from . import agent as va
        from .traces import Gallery

        log("loading SigLIP2, rerankers, MegaLoc, agent…")
        self.siglip, self.processor = embed_cache._load_model()
        released_ckpt = pr.PINPOINT_ROOT / "exp/attention_reranker/checkpoints/ckpt_best.pt"
        pr.RERANKER_CHECKPOINT = released_ckpt
        self.released = pr.PinpointReranker(device="cuda")
        pr.RERANKER_CHECKPOINT = CKPT
        self.wide = pr.PinpointReranker(device="cuda")
        self.wide.widen((32, 8, 8))
        self.narrow = pr.PinpointReranker(device="cuda")
        self.wide._osv_support(np.zeros(1, np.int64))  # build the OSV support tree once and share it
        self.released._osv_tree = self.narrow._osv_tree = self.wide._osv_tree
        self.megaloc = megaloc_model()
        self.agent = va.load_agent()
        self.gallery = Gallery()
        self.torch = torch
        self._text = None
        self._labeller = None
        log("ready")

    # -- stages
    def embed(self, jpeg: bytes) -> np.ndarray:
        from ..experiment.embed_cache import _embed, _pixels

        return _embed(self.siglip, _pixels(self.processor, jpeg)[None]).astype(np.float32)[0]

    def wide_pool(self, emb: np.ndarray):
        m = self.wide
        raw, projected = m.encode_queries(emb[None])
        rows, ranks, flags = m._union(m._search(raw, projected, None, None))
        scores, _ = m._score(raw, projected, rows, ranks, flags)
        order = np.argsort(-scores, axis=1, kind="stable")
        rows, scores = np.take_along_axis(rows, order, 1)[0], np.take_along_axis(scores, order, 1)[0]
        valid = rows >= 0
        latlon = np.where(valid[:, None], np.asarray(m.latlon[np.where(valid, rows, 0)]), np.nan).astype(np.float32)
        return rows, latlon, scores.astype(np.float32), valid

    def vlm_up(self) -> bool:
        try:
            with urllib.request.urlopen(f"{VLM}/health", timeout=2):
                return True
        except Exception:  # noqa: BLE001
            return False

    def _chat(self, b64: str, text: str, max_tokens: int, logprobs: bool = False):
        body = {"model": "vlm", "temperature": 0.0, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                                                          {"type": "text", "text": text}]}]}
        if logprobs:
            body |= {"logprobs": True, "top_logprobs": 10}
        else:
            body["repetition_penalty"] = 1.05
        req = urllib.request.Request(f"{VLM}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["choices"][0]

    def clues(self, tag: str, image_id: str, jpeg: bytes, areas_centres: np.ndarray) -> list[dict]:
        """observe / queries / text search / judge files for photo set tag; returns the clue-place hypotheses (hypotheses.clue_hypotheses)."""

        from ..experiment import observe
        from ..experiment.clue_judge import _items, _judge_path, _prompt
        from ..experiment.region_evidence import _Labeller, _raw_phrases, evidence_path
        from ..experiment.strategy_search import _haversine_km
        from ..experiment.text_search import TextSearch

        A = Path("artifacts/agent")
        b64 = base64.b64encode(jpeg).decode()
        raw = self._chat(b64, observe.prompt(), 400)["message"]["content"]
        (A / f"observe_{tag}_4b.json").write_text(json.dumps({image_id: {"raw": raw, "parsed": observe.parse(raw)}}))
        qraw = self._chat(b64, observe.QUERY_PROMPT, 160)["message"]["content"]
        q = [str(x) for x in ((observe._salvage(qraw) or {}).get("queries") or []) if str(x).strip()][:5]
        (A / f"queries_{tag}_4b.json").write_text(json.dumps({image_id: q}))
        self.log("text search…")
        self._text = self._text or TextSearch()
        rows = []
        for query, src in _raw_phrases(tag, "4b").get(image_id, {}).items():
            r = self._text.search(query, exclude=(0, ""))
            row = {"image_id": image_id, "query": query, "source": src, "kind": r.kind, "hint": r.hint, "loose": r.loose,
                   "matches": int(sum(r.matches.values())), "words": len(query.split())}
            if r.clusters:
                c = r.clusters[0]
                row |= {"lat": c.lat, "lon": c.lon, "weight": c.weight, "share": c.share, "exact": c.exact, "places": c.places, "km": 0.0}
            rows.append(row)
        if any("lat" in r for r in rows):
            self._labeller = self._labeller or _Labeller()
            for r in rows:
                if "lat" in r:
                    r["region"] = self._labeller(r["lat"], r["lon"])
        evidence_path(tag, "4b").write_text(json.dumps(rows))
        self.log("clue judge…")
        judged = []
        for r in _items(tag, "4b"):
            top = self._chat(b64, _prompt(r), 1, logprobs=True)["logprobs"]["content"][0]["top_logprobs"]
            p = {"yes": 0.0, "no": 0.0}
            for t in top:
                if t["token"].strip().lower() in p:
                    p[t["token"].strip().lower()] += math.exp(t["logprob"])
            judged.append({"image_id": image_id, "query": r["query"], "p_yes": p["yes"] / (p["yes"] + p["no"]) if p["yes"] + p["no"] else None})
        _judge_path(tag, "4b", "4b").write_text(json.dumps(judged))
        # clue places: accepted (P(yes) >= 0.5) read clues more than BOX_KM from every area, at most 2, mutually >= BOX_KM apart
        p_yes = {x["query"]: x["p_yes"] or 0.0 for x in judged}
        cands = []
        for r in _items(tag, "4b"):
            if r["source"].startswith("query") or p_yes.get(r["query"], 0) < 0.5:
                continue
            if len(areas_centres) and (_haversine_km(r["lat"], r["lon"], areas_centres) < BOX_KM).any():
                continue
            cands.append((p_yes[r["query"]], r["lat"], r["lon"]))
        kept = []
        for pr_, la, lo in sorted(cands, reverse=True):
            if all(_haversine_km(la, lo, np.asarray([[a, b]]))[0] >= BOX_KM for _, a, b in kept):
                kept.append((pr_, la, lo))
        return [{"source": 1, "rank": k, "support": pr_, "center": [la, lo]} for k, (pr_, la, lo) in enumerate(kept[:2])]

    def megaloc_sims(self, jpeg: bytes, rows: np.ndarray) -> np.ndarray:
        from PIL import Image

        from ..experiment.megaloc_gate import SIZE
        from ..experiment.sft_data import MP16Images

        torch = self.torch
        mean, std = np.asarray([0.485, 0.456, 0.406], np.float32), np.asarray([0.229, 0.224, 0.225], np.float32)

        def tf(data):
            try:
                im = Image.open(io.BytesIO(data)).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            except Exception:  # noqa: BLE001
                return np.zeros((3, SIZE, SIZE), np.float32)
            return ((np.asarray(im, np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)

        need = np.unique(rows[rows >= 0])
        self.gallery.resolve(need)
        images = MP16Images()
        x = np.stack([tf(jpeg)] + [tf(images.read(self.gallery.meta[int(r)]["id"]) or b"") for r in need])
        with torch.inference_mode():
            e = torch.cat([self.megaloc(torch.as_tensor(x[i:i + 64], device="cuda")) for i in range(0, len(x), 64)]).half().float()
        q, g = e[0], e[1:]
        sims = np.full(rows.shape, np.nan, np.float32)
        h, k = np.nonzero(rows >= 0)
        sims[h, k] = (g[torch.as_tensor(np.searchsorted(need, rows[h, k]), device="cuda")] @ q).cpu().numpy()
        return sims

    # -- the whole photo
    def run(self, job_dir: Path, name: str, truth: tuple[float, float] | None, status=lambda m: None) -> dict:
        from PIL import Image, ImageOps

        from ..experiment import hypothesis_compare as hc
        from ..experiment import megaloc_gate
        from ..experiment.hypothesis_agent import load
        from ..experiment.pinpoint_tool import areas as agreement_areas
        from .traces import build

        torch = self.torch
        job = job_dir.name
        base = "u" + job.replace("-", "")
        tag = f"{base}_{CKPT_TAG}"
        A, P = Path("artifacts/agent"), Path("artifacts/pinpoint_set")
        im = ImageOps.exif_transpose(Image.open(io.BytesIO((job_dir / "upload.bin").read_bytes()))).convert("RGB")
        im.thumbnail((2048, 2048))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
        jpeg = buf.getvalue()
        (job_dir / "image.jpg").write_bytes(jpeg)
        t = list(truth) if truth else [0.0, 0.0]

        status("Embedding the photo (SigLIP2) and searching the index with the released and retrained rerankers…")
        emb = self.embed(jpeg)
        rel = self.released.rank(emb[None])
        rv = rel.rows[0] >= 0
        released = [(rel.rows[0][rv][:12], rel.latlon[0][rv][:12].astype(np.float32), rel.scores[0][rv][:12])]
        rows, latlon, scores, valid = self.wide_pool(emb)
        pool = latlon[valid].astype(float).tolist()
        photo = {"index": 0, "image_id": name, "path": str(job_dir / "image.jpg"), "truth": t, "pool": pool, "pool_rows": rows[valid].tolist(),
                 "pools": f"{base}_{CKPT_TAG}", "candidates": [["", a, b] for a, b in pool]}
        (Path("artifacts/photo_sets") / tag).mkdir(parents=True, exist_ok=True)
        (Path("artifacts/photo_sets") / tag / "photos.json").write_text(json.dumps([photo]))
        (Path("artifacts/photo_sets") / tag / "labels.json").write_text(json.dumps([{"image_id": name, "pool": [], "truth": {"region": "?"}}]))
        s = np.where(valid, scores, -np.inf)
        probs = np.exp((s - s.max()) / 0.2)
        np.savez(P / f"set_probs_{tag}.npz", **{f"{tag}_probs": (probs / probs.sum())[None], f"{tag}_latlon": latlon[None], f"{tag}_valid": valid[None]})

        hyps = [{"source": 0, "rank": r, "support": a.support, "center": pool[a.founder]}
                for r, a in enumerate(agreement_areas(scores[valid].astype(np.float64), pool)[:K_AREAS])]
        centres = np.asarray([h["center"] for h in hyps], np.float64)
        if self.vlm_up():
            status("Reading clues with the 4B (text, place names, search phrases), text search and clue judge…")
            hyps += self.clues(tag, name, jpeg, centres)
            clues_ok = True
        else:
            for f in (f"observe_{tag}_4b.json", f"queries_{tag}_4b.json"):
                (A / f).write_text("{}")
            clues_ok = False

        status(f"Zooming: Pinpoint restricted to {len(hyps)} areas…")
        centre = np.asarray([h["center"] for h in hyps], np.float64)
        dlat = BOX_KM / 111.0
        dlon = dlat / np.maximum(np.cos(np.radians(centre[:, 0])), 0.05)
        box = np.stack([centre[:, 0] - dlat, centre[:, 0] + dlat, centre[:, 1] - dlon, centre[:, 1] + dlon], 1)
        found = self.narrow.rank(np.repeat(emb[None], len(hyps), 0), query_box=box)
        src = np.asarray([h["source"] for h in hyps])
        for suffix, sel in (("", src == 0), ("_cluehyps", src == 1)):
            np.savez(A / f"hypotheses_{tag}{suffix}.npz", photo=np.zeros(sel.sum(), int), source=src[sel], rank=np.asarray([h["rank"] for h in hyps])[sel],
                     support=np.asarray([h["support"] for h in hyps])[sel], center=centre[sel], latlon=found.latlon[sel].astype(np.float32),
                     scores=found.scores[sel].astype(np.float32), features=found.features[sel].astype(np.float32), rows=found.rows[sel])
        # restricted re-scoring for the reranker's CLS states (pinpoint_set restricted, --model none)
        hc.EXTRA_HYPOTHESES[:] = ["cluehyps"]
        h = hc._merge_clue_hyps(tag)
        captured = []
        head = self.narrow.head.score_head

        class _Capture(torch.nn.Module):
            def forward(self, x):
                captured.append(x.detach().to(torch.float16).cpu())
                return head(x)

        self.narrow.head.score_head = _Capture()
        try:
            raw, projected = self.narrow.encode_queries(emb[None])
            hr, hf = h["rows"].astype(np.int64), h["features"].astype(np.float32)
            zscores = self.narrow._score(raw[[0] * len(hr)], projected[[0] * len(hr)], hr, hf[:, :, 3:6], hf[:, :, 6:9])[0]
        finally:
            self.narrow.head.score_head = head
        hidden = torch.cat(captured).numpy().reshape(hr.shape[0], hr.shape[1], -1)
        np.savez(P / f"zoom_{tag}.npz", photo=h["photo"], source=h["source"], rank=h["rank"], support=h["support"], center=h["center"], rows=hr,
                 latlon=h["latlon"].astype(np.float32), scores=zscores.astype(np.float32), hidden=hidden, features=hf.astype(np.float16))
        hc.SET_DUMP = str(P / f"set_probs_{tag}.npz")
        hc.features(tag, False, e4=True, extra_name="set")

        status("Recognising: MegaLoc similarity to every area's photos…")
        zm = np.load(P / f"zoom_{tag}.npz")
        sims = self.megaloc_sims(jpeg, zm["rows"])
        megaloc_gate.CKPT_TAG = CKPT_TAG
        megaloc_gate._write_columns(base, zm, sims)

        status("Running the RL agent…")
        net, pols, stats = self.agent
        d = load(base, CKPT_TAG, stats=stats, extra=("megaloc",))
        d["glance"] = torch.as_tensor((emb / np.linalg.norm(emb))[None], device="cuda")
        trace = build(base, d=d, released=released, out_dir=False, agent=self.agent, gallery=self.gallery)[0]
        trace |= {"set": "upload", "i": job, "image_id": name, "benchmark": "upload", "image": {"kind": "upload", "id": job},
                  "clues_available": clues_ok}
        for a, sm in zip(trace["areas"], _slot_rows(zm, sims)):
            for c, v in zip(a["candidates"], sm):
                c["megaloc"] = None if not np.isfinite(v) else round(float(v), 4)
        if not truth:
            trace["truth"] = None
            _drop_km(trace)
        (job_dir / "trace.json").write_text(json.dumps(trace, separators=(",", ":")))
        return trace


def _slot_rows(zm, sims) -> list[np.ndarray]:
    """The MegaLoc similarities of each area's valid candidates, in the trace's slot order (hypothesis_agent.load: merged row order)."""

    out = []
    for j in range(len(zm["photo"])):
        out.append(sims[j][zm["rows"][j] >= 0])
    return out


def _drop_km(x) -> None:
    if isinstance(x, dict):
        for k in list(x):
            if k == "km":
                x[k] = None
            else:
                _drop_km(x[k])
    elif isinstance(x, list):
        for v in x:
            _drop_km(v)


def worker() -> None:
    """Process queued uploads (artifacts/viz/uploads/*/request.json without a trace) one at a time, forever."""

    pipe = Pipeline(log=lambda m: print(time.strftime("%T"), m, flush=True))
    while True:
        jobs = sorted(p.parent for p in UPLOADS.glob("*/request.json") if not (p.parent / "trace.json").exists()
                      and json.loads((p.parent / "status.json").read_text()).get("state") != "error")
        if not jobs:
            time.sleep(1.0)
            continue
        d = jobs[0]
        req = json.loads((d / "request.json").read_text())
        st = lambda m, state="running": (d / "status.json").write_text(json.dumps({"state": state, "message": m}))
        try:
            print(time.strftime("%T"), f"job {d.name}", flush=True)
            start = time.time()
            trace = pipe.run(d, req["name"], tuple(req["truth"]) if req.get("truth") else None, status=st)
            fin = trace["agent"]["0"]["final"]
            st(f"Done in {time.time() - start:.0f} s: {fin.get('place')}", "done")
        except Exception as exc:  # noqa: BLE001 - report and keep serving
            import traceback

            traceback.print_exc()
            st(f"Failed: {exc!r}", "error")


def run_one(path: str, lat: float | None, lon: float | None) -> None:
    job = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = UPLOADS / job
    d.mkdir(parents=True)
    (d / "upload.bin").write_bytes(Path(path).read_bytes())
    truth = (lat, lon) if lat is not None and lon is not None else None
    (d / "request.json").write_text(json.dumps({"name": Path(path).name, "truth": truth}))
    pipe = Pipeline()
    t0 = time.time()
    trace = pipe.run(d, Path(path).name, truth, status=lambda m: print(time.strftime("%T"), m, flush=True))
    (d / "status.json").write_text(json.dumps({"state": "done", "message": "done"}))
    fin = trace["agent"]["0"]["final"]
    print(f"{time.time() - t0:.0f} s: answer {fin['lat']}, {fin['lon']} ({fin.get('place')}), km {fin.get('km')}; trace {d / 'trace.json'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("worker")
    r = sub.add_parser("run")
    r.add_argument("image")
    r.add_argument("--lat", type=float)
    r.add_argument("--lon", type=float)
    args = parser.parse_args(argv)
    if args.cmd == "worker":
        worker()
    else:
        run_one(args.image, args.lat, args.lon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
