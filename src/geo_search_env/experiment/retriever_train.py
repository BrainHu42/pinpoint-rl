# Fine-tune Pinpoint's contrastive retriever and rebuild its index (RETRIEVAL_PLAN.md steps 2-3).
# Runs in pinpoint-main's env (it imports pinpoint_train's GeoLightningModule):
#   PYTHONPATH=pinpoint_train/src:src data/pinpoint-main-src/pinpoint-main/.venv/bin/python -m geo_search_env.experiment.retriever_train \
#       train --run ret_ctrl --steps 5000 [--ii-weight 0.2]
#   ... retriever_train index --run ret_ctrl          (then retrieval_recall ranks --tag ret_ctrl, in .venv)
#   ... retriever_train check-index                   (encodes released rows with the released checkpoint, compares with the shipped index)

"""train: starts from the released ckpt_best.pt (its weights load strictly into this repo's GeoLightningModule) and keeps its losses: SigLIP
pairwise image-to-GPS per source, MP16 weight 1, OSV-5M 0.5, in-batch, ~2048 photos per source per step (the release: batch ~4096, uniform
mixing). The release's image-neighbourhood term is not in this code, so a run with --ii-weight 0 is the like-for-like control.
--ii-weight adds an image-to-image term on MP16 (lesson R1: the right photo is usually at image-search rank 10-100): each anchor's positive is
another training photo in its ~--ii-cell-km cube (distance only, no photographer information), SigLIP pairwise over anchors x positives with
its own scale / bias; off-diagonal pairs are down-weighted by distance as in the release (weight 0.02 within 1 km rising to 1 at 250 km).
Training rows: the index's MP16 rows (buckets 0-98; bucket 99 = our train / dev3k is never seen) and every OSV-5M train row. Training inputs are
GPU-resident int8 (MP16: the index's raw part; OSV: prepare-osv; tower output cosine 0.9997 vs fp16); the index is encoded from fp16. Checkpoints: artifacts/retrieval/ckpt/{run}/ckpt_{step}.pt and ckpt_final.pt (state_dict + hyper_parameters).

index: a new attention retrieval index for a checkpoint, artifacts/retrieval/index/{run}/: image (MP16 adapter) and GPS embeddings per index
row, the OSV support cache (OSV adapter); the raw SigLIP2 int8 part, latlon, cache indices and source ids are symlinks to the released index."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RELEASED = Path("data/pinpoint/pinpoint-submission/exp/contrastive_retrieval/checkpoints")
RELEASED_CKPT = RELEASED / "ckpt_best.pt"
RELEASED_INDEX = next((RELEASED / "attention_retrieval_index").iterdir())
DATA_ROOT = Path("/opt/dlami/nvme/retrieval")  # staged copies of data/pinpoint/{mp16,osv5m}-embed/siglip2-giant-opt-patch16-384
OUT = Path("artifacts/retrieval")
EARTH_KM = 6371.0088


def _module(state: dict | None = None):
    from pinpoint.contrastive_retrieval.lightning_module import GeoLightningModule

    checkpoint = torch.load(RELEASED_CKPT, map_location="cpu", weights_only=False)
    hp = dict(checkpoint["hyper_parameters"])
    accepted = inspect.signature(GeoLightningModule.__init__).parameters
    module = GeoLightningModule(**{k: v for k, v in hp.items() if k in accepted})
    module.load_state_dict(state or {k: v for k, v in checkpoint["state_dict"].items() if k != "best_val_median_km"}, strict=False)
    return module, hp


def _xyz(latlon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(latlon[:, 0]), np.radians(latlon[:, 1])
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)


def _cells(latlon: np.ndarray, cell_km: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows grouped by cube of side cell_km on the unit sphere's embedding: (order, start, size) with start / size per position in order."""

    q = np.floor(_xyz(latlon.astype(np.float64)) / (cell_km / EARTH_KM)).astype(np.int64) + (1 << 14)
    key = q[:, 0] | (q[:, 1] << 15) | (q[:, 2] << 30)
    order = np.argsort(key, kind="stable")
    k = key[order]
    first = np.r_[True, k[1:] != k[:-1]]
    group = np.cumsum(first) - 1
    starts = np.flatnonzero(first)
    sizes = np.diff(np.r_[starts, len(k)])
    return order, starts[group], sizes[group]


def _int8_rows(fp16: np.memmap, dst: Path) -> None:
    """fp16 [N, 1536] -> int8 rows with a per-row max-abs scale (as the index stores raw SigLIP2), for GPU-resident training data."""

    q = np.lib.format.open_memmap(dst.with_suffix(".i8.npy"), mode="w+", dtype=np.int8, shape=fp16.shape)
    scales = np.empty(len(fp16), np.float16)
    for s in range(0, len(fp16), 262_144):
        x = torch.as_tensor(np.asarray(fp16[s : s + 262_144]), device="cuda").float()
        scale = x.abs().amax(1).clamp(min=1e-8) / 127.0
        q[s : s + len(x)] = torch.round(x / scale[:, None]).clamp(-127, 127).to(torch.int8).cpu().numpy()
        scales[s : s + len(x)] = scale.cpu().numpy()
    q.flush()
    np.save(dst.with_suffix(".scale.npy"), scales)


def _resident(q: np.ndarray, scales: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(q.shape, dtype=torch.int8, device=device)
    for s in range(0, len(q), 1 << 20):
        out[s : s + (1 << 20)] = torch.as_tensor(np.asarray(q[s : s + (1 << 20)]), device=device)
    return out, torch.as_tensor(np.asarray(scales, dtype=np.float32), device=device)


def _haversine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[A, B] km between latlon rows (degrees)."""

    a, b = torch.deg2rad(a.float()), torch.deg2rad(b.float())
    dlat = b[None, :, 0] - a[:, None, 0]
    dlon = b[None, :, 1] - a[:, None, 1]
    h = torch.sin(dlat / 2) ** 2 + torch.cos(a[:, None, 0]) * torch.cos(b[None, :, 0]) * torch.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * torch.asin(torch.sqrt(h.clamp(0, 1)))


def _negative_weight(km: torch.Tensor, low: float = 0.02, dead: float = 1.0, full: float = 250.0) -> torch.Tensor:
    return low + (1 - low) * ((km - dead) / (full - dead)).clamp(0, 1)


def train(a) -> None:
    torch.manual_seed(a.seed)
    device = torch.device("cuda")
    module, hp = _module()
    module.to(device).train()
    ii_scale = torch.nn.Parameter(torch.tensor(math.log(10.0), device=device))
    ii_bias = torch.nn.Parameter(torch.tensor(-10.0, device=device))
    params = list(module.parameters()) + ([ii_scale, ii_bias] if a.ii_weight > 0 else [])
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    schedule = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - a.warmup) / max(1, a.steps - a.warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    # training data on the GPU: MP16 = the index's rows (its int8 raw SigLIP2), OSV-5M = all train rows (int8 copy made by prepare-osv)
    manifest = json.loads((RELEASED_INDEX / "manifest.json").read_text())
    count = int(manifest["num_samples"])
    mp16_rows = np.fromfile(RELEASED_INDEX / manifest["files"]["cache_indices"], dtype=np.int64)
    dev = {e["row"] for e in json.loads(Path("artifacts/photo_sets/dev3k/photos.json").read_text())}
    assert not dev & set(mp16_rows.tolist()), "dev3k photos in the training rows"
    mp16_q, mp16_s = _resident(np.memmap(RELEASED_INDEX / manifest["files"]["raw_image_embeddings"], dtype=np.int8, mode="r", shape=(count, 1536)),
                               np.memmap(RELEASED_INDEX / manifest["files"]["raw_image_scales"], dtype=np.float16, mode="r", shape=(count,)), device)
    osv_q, osv_s = _resident(np.load(DATA_ROOT / "osv5m/embeddings.i8.npy", mmap_mode="r"), np.load(DATA_ROOT / "osv5m/embeddings.scale.npy"), device)
    mp16_ll = np.fromfile(RELEASED_INDEX / manifest["files"]["latlon_deg"], dtype=np.float32).reshape(-1, 2)
    osv_ll = torch.as_tensor(np.fromfile(DATA_ROOT / "osv5m/latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2), device=device)
    order, cell_start, cell_size = (torch.as_tensor(x, device=device) for x in _cells(mp16_ll, a.ii_cell_km))
    position = torch.empty_like(order)
    position[order] = torch.arange(len(order), device=device)
    mp16_ll = torch.as_tensor(mp16_ll, device=device)
    per = a.batch // 2
    generator = torch.Generator(device=device).manual_seed(a.seed)

    def rows(q, scale, idx):
        return q[idx].float() * scale[idx, None]

    def batches():
        while True:
            anchor = torch.randint(0, count, (per,), device=device, generator=generator)
            pos = position[anchor]
            has = cell_size[pos] > 1
            j = cell_start[pos] + (torch.rand(per, device=device, generator=generator) * (cell_size[pos] - 1).clamp(min=1)).long()
            j = torch.where(j >= pos, j + 1, j).clamp(max=count - 1)  # skip the anchor itself
            partner = torch.where(has, order[j], anchor)
            o = torch.randint(0, len(osv_ll), (per,), device=device, generator=generator)
            yield (rows(mp16_q, mp16_s, anchor), mp16_ll[anchor], rows(mp16_q, mp16_s, partner), mp16_ll[partner], has,
                   rows(osv_q, osv_s, o), osv_ll[o])

    print(f"training data on the GPU: {torch.cuda.memory_allocated() / 2**30:.1f} GB; {(cell_size > 1).float().mean() * 100:.1f}% of MP16 rows "
          f"have a partner within their {a.ii_cell_km} km cell", flush=True)
    run_dir = OUT / "ckpt" / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(a), indent=1))
    log = open(run_dir / "train.log", "a")
    t0, sums = time.time(), {}
    for step, batch in enumerate(batches()):
        if step >= a.steps:
            break
        m_raw, m_ll, p_raw, p_ll, has, o_raw, o_ll = batch
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m_img = module.encode_image(m_raw)
            o_img = module.encode_osv_image(o_raw)
            loc = module.encode_location(torch.cat([m_ll, o_ll]).float())
            loss_m = module._siglip_pairwise_loss(module._compute_pair_logits(m_img.float(), loc[: len(m_ll)].float()))
            loss_o = module._siglip_pairwise_loss(module._compute_pair_logits(o_img.float(), loc[len(m_ll):].float()))
            loss = (loss_m * 1.0 + loss_o * float(hp["loss_weight_osv_image_loc"])) / (1.0 + float(hp["loss_weight_osv_image_loc"]))
            parts = {"mp16": loss_m.detach(), "osv": loss_o.detach()}
            if a.ii_weight > 0 and has.any():
                anchor, positive = m_img[has].float(), module.encode_image(p_raw[has]).float()
                logits = torch.exp(ii_scale).clamp(max=100.0) * anchor @ positive.T + ii_bias
                eye = torch.eye(len(anchor), device=device, dtype=torch.bool)
                weight = torch.where(eye, 1.0, _negative_weight(_haversine(m_ll[has], p_ll[has])))
                loss_ii = -(weight * F.logsigmoid(torch.where(eye, logits, -logits))).sum(1).mean()
                loss = loss + a.ii_weight * loss_ii
                parts["ii"] = loss_ii.detach()
                parts["has"] = has.float().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        if (step + 1) % a.log_every == 0:
            line = f"step {step + 1} " + " ".join(f"{k} {v / a.log_every:.4f}" for k, v in sums.items()) + \
                   f" lr {scheduler.get_last_lr()[0]:.2e} {(time.time() - t0) / (step + 1):.3f}s/step"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            sums = {}
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            name = "ckpt_final.pt" if step + 1 == a.steps else f"ckpt_{step + 1}.pt"
            state = {k: v.detach().cpu() for k, v in module.state_dict().items()}
            # Lightning's keys too, so the reranker trainer's GeoLightningModule.load_from_checkpoint takes it
            torch.save({"state_dict": state, "hyper_parameters": hp, "global_step": step + 1, "run": vars(a),
                        "pytorch-lightning_version": "2.0.0"}, run_dir / name)
    print(f"done in {time.time() - t0:.0f}s -> {run_dir}", flush=True)


@torch.inference_mode()
def _encode(fn, source: np.ndarray, rows: np.ndarray | None, count: int, dst: Path, batch: int = 65_536) -> None:
    out = np.lib.format.open_memmap(dst, mode="w+", dtype=np.float16, shape=(count, 256)) if dst.suffix == ".npy" else \
        np.memmap(dst, dtype=np.float16, mode="w+", shape=(count, 256))
    for s in range(0, count, batch):
        x = source[s : s + batch] if rows is None else source[rows[s : s + batch]]
        with torch.autocast("cuda", dtype=torch.float16):
            y = fn(torch.as_tensor(np.asarray(x), device="cuda").float())
        out[s : s + batch] = torch.nan_to_num(y.float()).cpu().numpy().astype(np.float16)
        if (s // batch) % 16 == 0:
            print(f"  {dst.name} {s + len(x):,}/{count:,}", flush=True)
    out.flush()


def _load_trained(run: str):
    checkpoint = torch.load(OUT / "ckpt" / run / "ckpt_final.pt", map_location="cpu", weights_only=False)
    module, _ = _module(checkpoint["state_dict"])
    return module.to("cuda").eval()


def index(a) -> None:
    module = _load_trained(a.run)
    dst = OUT / "index" / a.run
    (dst / "osv_support_cache").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RELEASED_INDEX / "manifest.json").read_text())
    for key in ("cache_indices", "latlon_deg", "raw_image_embeddings", "raw_image_scales", "source_ids"):
        link = dst / manifest["files"][key]
        if not link.exists():
            link.symlink_to((RELEASED_INDEX / manifest["files"][key]).resolve())
    count = int(manifest["num_samples"])
    rows = np.fromfile(RELEASED_INDEX / manifest["files"]["cache_indices"], dtype=np.int64)
    mp16 = np.memmap(DATA_ROOT / "mp16/embeddings.f16.bin", dtype=np.float16, mode="r").reshape(-1, 1536)
    latlon = np.fromfile(RELEASED_INDEX / manifest["files"]["latlon_deg"], dtype=np.float32).reshape(-1, 2)
    order = np.argsort(rows)  # read the MP16 memmap in row order, write back in index order
    tmp = dst / "image_sorted.f16.bin"
    _encode(module.encode_image, mp16, rows[order], count, tmp)
    sorted_img = np.memmap(tmp, dtype=np.float16, mode="r", shape=(count, 256))
    image = np.memmap(dst / manifest["files"]["image_embeddings"], dtype=np.float16, mode="w+", shape=(count, 256))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(count)
    for s in range(0, count, 1 << 20):
        image[s : s + (1 << 20)] = sorted_img[inverse[s : s + (1 << 20)]]
    image.flush()
    del sorted_img
    tmp.unlink()
    _encode(module.encode_location, latlon, None, count, dst / manifest["files"]["gps_embeddings"])
    osv_manifest = json.loads((RELEASED_INDEX / "osv_support_cache/manifest.json").read_text())
    osv = np.memmap(DATA_ROOT / "osv5m/embeddings.f16.bin", dtype=np.float16, mode="r").reshape(-1, 1536)
    _encode(module.encode_osv_image, osv, None, int(osv_manifest["num_samples"]), dst / "osv_support_cache" / osv_manifest["files"]["embeddings"])
    link = dst / "osv_support_cache" / osv_manifest["files"]["latlon_deg"]
    if not link.exists():
        link.symlink_to((RELEASED_INDEX / "osv_support_cache" / osv_manifest["files"]["latlon_deg"]).resolve())
    (dst / "osv_support_cache/manifest.json").write_text(json.dumps(dict(osv_manifest, checkpoint_path=str(OUT / "ckpt" / a.run / "ckpt_final.pt")), indent=2))
    (dst / "manifest.json").write_text(json.dumps(dict(manifest, checkpoint_path=str(OUT / "ckpt" / a.run / "ckpt_final.pt")), indent=2))
    print(f"-> {dst}", flush=True)


def check_index(a) -> None:
    """Released checkpoint on 4,096 released index rows and OSV rows vs the shipped embeddings."""

    module, _ = _module()
    module.to("cuda").eval()
    manifest = json.loads((RELEASED_INDEX / "manifest.json").read_text())
    count = int(manifest["num_samples"])
    rows = np.fromfile(RELEASED_INDEX / manifest["files"]["cache_indices"], dtype=np.int64)
    pick = np.sort(np.random.default_rng(0).choice(count, 4096, replace=False))
    mp16 = np.memmap(DATA_ROOT / "mp16/embeddings.f16.bin", dtype=np.float16, mode="r").reshape(-1, 1536)
    shipped = {k: np.memmap(RELEASED_INDEX / manifest["files"][k], dtype=np.float16, mode="r", shape=(count, 256)) for k in ("image_embeddings", "gps_embeddings")}
    latlon = np.fromfile(RELEASED_INDEX / manifest["files"]["latlon_deg"], dtype=np.float32).reshape(-1, 2)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        img = module.encode_image(torch.as_tensor(np.asarray(mp16[rows[pick]]), device="cuda").float()).float().cpu().numpy()
        gps = module.encode_location(torch.as_tensor(latlon[pick], device="cuda").float()).float().cpu().numpy()
        osv = np.memmap(DATA_ROOT / "osv5m/embeddings.f16.bin", dtype=np.float16, mode="r").reshape(-1, 1536)
        oshipped = np.memmap(RELEASED_INDEX / "osv_support_cache/embeddings.contrastive_osv.f16.bin", dtype=np.float16, mode="r").reshape(-1, 256)
        o = module.encode_osv_image(torch.as_tensor(np.asarray(osv[pick]), device="cuda").float()).float().cpu().numpy()
    for name, mine, ref in (("image", img, shipped["image_embeddings"][pick]), ("gps", gps, shipped["gps_embeddings"][pick]), ("osv", o, oshipped[pick])):
        ref = np.asarray(ref, dtype=np.float32)
        cos = (mine * ref).sum(1) / np.linalg.norm(mine, axis=1) / np.linalg.norm(ref, axis=1)
        print(f"{name}: cosine to shipped min {cos.min():.5f} mean {cos.mean():.5f}; max abs diff {np.abs(mine - ref).max():.4f}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--run", required=True)
    t.add_argument("--steps", type=int, default=5000)
    t.add_argument("--batch", type=int, default=4096)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--ii-weight", type=float, default=0.0)
    t.add_argument("--ii-cell-km", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--save-every", type=int, default=1000)
    i = sub.add_parser("index")
    i.add_argument("--run", required=True)
    sub.add_parser("check-index")
    sub.add_parser("prepare-osv")
    a = p.parse_args(argv)
    {"train": train, "index": index, "check-index": check_index,
     "prepare-osv": lambda _: _int8_rows(np.memmap(DATA_ROOT / "osv5m/embeddings.f16.bin", dtype=np.float16, mode="r").reshape(-1, 1536),
                                         DATA_ROOT / "osv5m/embeddings")}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
