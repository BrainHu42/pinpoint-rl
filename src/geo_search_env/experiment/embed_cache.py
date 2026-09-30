# Build the SigLIP2 image-embedding caches the retrieval code reads: MP16 / OSV-5M galleries and benchmark queries.
# Usage: .venv/bin/python -m geo_search_env.experiment.embed_cache {gallery,benchmark} NAME [--out DIR] [--compare DIR]

"""SigLIP2-giant image embeddings (raw `get_image_features`, fp16, not normalized) as flat binaries + manifest.json.

gallery {mp16,osv5m}: embeddings.f16.bin [N, 1536], latlon_deg.f32.bin and latlon_norm.f32.bin [N, 2],
    row_index.i64.bin [N] (row in the metadata CSV) and image_ids.txt. Rows come in the order of the Pinpoint cache
    builder the rest of the code was run on: worker w of W embeds CSV rows with row % W == w, and the DataLoader
    interleaves their batches round-robin. `--workers`/`--batch-size` default to the layout of our caches (4 x 32 for
    MP16, 4 x 64 for OSV-5M); other values give the same rows in a different order, which invalidates `artifacts/`.
benchmark {im2gps3k,yfcc4k}: embeddings.f16.bin and image_ids.txt in CSV order, plus failures.json.

--limit N builds only the first N rows. --compare DIR then checks the new cache against an existing one (same ids in
the same order, per-row cosine), e.g. `gallery mp16 --limit 2048 --out /tmp/x --compare <MP16_EMBED>`.
Full gallery builds took ~11 h (MP16) and ~13 h (OSV-5M) for our caches, bound by JPEG decoding.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import pickle
from typing import Iterator, Sequence

import numpy as np

from ..data.benchmarks import BENCHMARKS, DATA_ROOT, EMBEDDING_KEY
from .strategy_search import MP16_CSV, MP16_EMBED, OSV_EMBED
from .verifiers import MP16_ROOT, _read_mp16_image


MODEL = "google/siglip2-giant-opt-patch16-384"
OSV_ROOT = Path("/data/hf/datasets/osv5m")
GALLERY_LAYOUT = {"mp16": (4, 32), "osv5m": (4, 64)}  # (workers, batch size) that produced our caches
GALLERY_OUT = {"mp16": MP16_EMBED, "osv5m": OSV_EMBED}


def _valid_latlon(row: dict[str, str], lat_key: str, lon_key: str) -> tuple[float, float] | None:
    try:
        lat, lon = float(row[lat_key]), float(row[lon_key])
    except (KeyError, TypeError, ValueError):
        return None
    return (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 180 else None


def _pixels(processor, data: bytes | None):
    from PIL import Image

    if data is None:
        return None
    try:
        return processor(images=Image.open(io.BytesIO(data)).convert("RGB"), return_tensors="pt")["pixel_values"][0]
    except Exception:  # undecodable photos are skipped, as in the Pinpoint builder
        return None


def _osv_paths() -> dict[str, Path]:
    root = OSV_ROOT / "images" / "train"
    return {p.stem: p for shard in sorted(root.iterdir()) if shard.is_dir() for p in shard.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}}


def _gallery_rows(dataset: str):
    """Iterable of (pixels, row, image_id, lat, lon) in Pinpoint's per-worker order (see module docstring)."""

    import torch
    from torch.utils.data import IterableDataset, get_worker_info

    class Rows(IterableDataset):
        def __init__(self, processor, osv_paths: dict[str, Path] | None) -> None:
            self.processor, self.osv_paths = processor, osv_paths

        def __iter__(self) -> Iterator[tuple]:
            worker = get_worker_info()
            wid, workers = (worker.id, worker.num_workers) if worker else (0, 1)
            if dataset == "mp16":
                with (MP16_ROOT / "metadata" / "tar_index.pkl").open("rb") as stream:
                    index = pickle.load(stream)
                parts = sorted(MP16_ROOT.glob("mp-16-images[0-9][0-9]"))
                handles, chunk = [p.open("rb") for p in parts], parts[0].stat().st_size
                csv_path, id_key, lat_key, lon_key = MP16_CSV, "IMG_ID", "LAT", "LON"
                read = lambda image_id: _read_mp16_image(image_id, index, handles, chunk)  # noqa: E731
            else:
                index = self.osv_paths
                csv_path, id_key, lat_key, lon_key = OSV_ROOT / "train.csv", "id", "latitude", "longitude"
                read = lambda image_id: index[image_id].read_bytes()  # noqa: E731
            with csv_path.open("r", encoding="utf-8", newline="") as stream:
                for row, record in enumerate(csv.DictReader(stream)):
                    if row % workers != wid:
                        continue
                    image_id, latlon = record.get(id_key), _valid_latlon(record, lat_key, lon_key)
                    if not image_id or image_id not in index or latlon is None:
                        continue
                    try:
                        data = read(image_id)
                    except OSError:
                        data = None
                    pixels = _pixels(self.processor, data)
                    if pixels is not None:
                        yield pixels, row, image_id, latlon[0], latlon[1]

    def collate(batch):
        pixels, rows, ids, lats, lons = zip(*batch)
        return torch.stack(pixels), np.asarray(rows, dtype=np.int64), list(ids), np.asarray([lats, lons], dtype=np.float32).T

    return Rows, collate


def _load_model():
    import torch
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    return model, AutoProcessor.from_pretrained(MODEL)  # PIL resize (no torchvision); Pinpoint used the torchvision one


def _embed(model, pixels) -> np.ndarray:
    import torch

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out = model.get_image_features(pixel_values=pixels.cuda(non_blocking=True))
    out = out if not hasattr(out, "pooler_output") else out.pooler_output
    return out.float().cpu().numpy().astype(np.float16)


def _refuse_overwrite(out: Path, overwrite: bool) -> None:
    if (out / "manifest.json").exists() and not overwrite:
        raise SystemExit(f"{out} already holds a cache; pass --overwrite to replace it")
    out.mkdir(parents=True, exist_ok=True)


def gallery(dataset: str, out: Path, *, workers: int, batch_size: int, limit: int, overwrite: bool) -> None:
    from torch.utils.data import DataLoader

    _refuse_overwrite(out, overwrite)
    model, processor = _load_model()
    osv_paths = None
    if dataset == "osv5m":
        print("indexing OSV-5M train images", flush=True)
        osv_paths = _osv_paths()
    rows, collate = _gallery_rows(dataset)
    # fork: workers inherit the OSV path index instead of unpickling it; they never touch CUDA
    loader = DataLoader(rows(processor, osv_paths), batch_size=batch_size, num_workers=workers, collate_fn=collate,
                        pin_memory=True, multiprocessing_context="fork" if workers else None)
    written, dim = 0, None
    with (
        (out / "embeddings.f16.bin").open("wb") as emb,
        (out / "latlon_deg.f32.bin").open("wb") as deg,
        (out / "latlon_norm.f32.bin").open("wb") as norm,
        (out / "row_index.i64.bin").open("wb") as row_index,
        (out / "image_ids.txt").open("w", encoding="utf-8") as ids_out,
    ):
        for batch, (pixels, rows_, ids, latlon) in enumerate(loader):
            take = len(ids) if not limit else min(len(ids), limit - written)
            features = _embed(model, pixels[:take])
            dim = features.shape[1]
            features.tofile(emb)
            latlon[:take].tofile(deg)
            (latlon[:take] / np.float32([90, 180])).astype(np.float32).tofile(norm)
            rows_[:take].tofile(row_index)
            ids_out.write("".join(f"{i}\n" for i in ids[:take]))
            written += take
            if batch % 1000 == 0:
                print(f"  {written:,} embedded", flush=True)
            if limit and written >= limit:
                break
    _write_manifest(out, {
        "format": "flat_binary_memmap_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "source_split": "train",
        "model_name": MODEL,
        "normalize_embeddings": False,
        "num_workers": workers,
        "batch_size": batch_size,
        "num_samples": written,
        "embedding_dim": dim,
        "files": {"embeddings": "embeddings.f16.bin", "latlon_deg": "latlon_deg.f32.bin", "latlon_norm": "latlon_norm.f32.bin",
                  "row_index": "row_index.i64.bin", "image_ids": "image_ids.txt"},
        "dtypes": {"embeddings": "float16", "latlon_deg": "float32", "latlon_norm": "float32", "row_index": "int64", "image_ids": "utf-8 lines"},
        "shapes": {"embeddings": [written, dim], "latlon_deg": [written, 2], "latlon_norm": [written, 2], "row_index": [written]},
    })
    print(f"wrote {written:,} rows to {out}", flush=True)


def benchmark(name: str, out: Path, *, batch_size: int, limit: int, overwrite: bool) -> None:
    import torch

    spec = BENCHMARKS[name]
    _refuse_overwrite(out, overwrite)
    model, processor = _load_model()
    with (DATA_ROOT / spec.csv_relpath).open("r", encoding="utf-8", newline="") as stream:
        ids = [r["IMG_ID"].strip() for r in csv.DictReader(stream) if r.get("IMG_ID", "").strip() and _valid_latlon(r, "LAT", "LON")]
    ids = ids[:limit] if limit else ids
    kept, failures, chunks = [], {}, []
    for start in range(0, len(ids), batch_size):
        pixels = []
        for image_id in ids[start : start + batch_size]:
            path = DATA_ROOT / spec.image_root_relpath / image_id
            p = _pixels(processor, path.read_bytes() if path.exists() else None)
            if p is None:
                failures[image_id] = "missing" if not path.exists() else "undecodable"
            else:
                kept.append(image_id)
                pixels.append(p)
        if pixels:
            chunks.append(_embed(model, torch.stack(pixels)))
    embeddings = np.concatenate(chunks)
    embeddings.tofile(out / "embeddings.f16.bin")
    (out / "image_ids.txt").write_text("".join(f"{i}\n" for i in kept), encoding="utf-8")
    (out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    _write_manifest(out, {
        "backbone_model_name": MODEL,
        "dataset_name": name,
        "dtype": "float16",
        "embedding_dim": int(embeddings.shape[1]),
        "files": {"embeddings": "embeddings.f16.bin", "failures": "failures.json", "image_ids": "image_ids.txt"},
        "num_embeddings": len(kept),
        "version": 1,
    })
    print(f"wrote {len(kept):,} rows to {out} ({len(failures)} failures)", flush=True)


def _write_manifest(out: Path, manifest: dict) -> None:
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_cache(root: Path) -> tuple[list[str], np.ndarray]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    ids = (root / manifest["files"]["image_ids"]).read_text(encoding="utf-8").splitlines()
    dim = manifest["embedding_dim"]
    return ids, np.memmap(root / manifest["files"]["embeddings"], dtype=np.float16, mode="r", shape=(len(ids), dim))


def compare(new: Path, reference: Path) -> bool:
    """Same ids in the same order as the reference's first rows, and embeddings that match to fp16/kernel noise."""

    ids, emb = _read_cache(new)
    ref_ids, ref_emb = _read_cache(reference)
    same_order = ids == ref_ids[: len(ids)]
    print(f"ids: {len(ids):,} new rows, {'same order as' if same_order else 'DIFFERENT from'} the reference's first rows")
    if not same_order:
        return False
    a, b = np.asarray(emb, dtype=np.float32), np.asarray(ref_emb[: len(ids)], dtype=np.float32)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    print(f"cosine to reference: median {np.median(cos):.4f}, min {cos.min():.4f}")
    return bool(cos.min() >= 0.99)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=("gallery", "benchmark"))
    parser.add_argument("name", help="gallery: mp16 | osv5m; benchmark: " + " | ".join(BENCHMARKS))
    parser.add_argument("--out", type=Path, help="default: the path the code reads (strategy_search / benchmarks.py)")
    parser.add_argument("--workers", type=int, help="gallery DataLoader workers (default: our caches' layout)")
    parser.add_argument("--batch-size", type=int, help="default: our caches' layout for galleries, 64 for benchmarks")
    parser.add_argument("--limit", type=int, default=0, help="build only the first N rows (0 = all)")
    parser.add_argument("--compare", type=Path, help="existing cache to check the new one against")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.kind == "gallery":
        if args.name not in GALLERY_LAYOUT:
            parser.error(f"unknown gallery {args.name}")
        workers, batch_size = GALLERY_LAYOUT[args.name]
        out = args.out or GALLERY_OUT[args.name]
        gallery(args.name, out, workers=args.workers if args.workers is not None else workers,
                batch_size=args.batch_size or batch_size, limit=args.limit, overwrite=args.overwrite)
    else:
        if args.name not in BENCHMARKS:
            parser.error(f"unknown benchmark {args.name}")
        spec = BENCHMARKS[args.name]
        out = args.out or DATA_ROOT / Path(spec.image_root_relpath).parent / "image_embeddings" / EMBEDDING_KEY
        benchmark(args.name, out, batch_size=args.batch_size or 64, limit=args.limit, overwrite=args.overwrite)
    return 0 if args.compare is None or compare(out, args.compare) else 1


if __name__ == "__main__":
    raise SystemExit(main())
