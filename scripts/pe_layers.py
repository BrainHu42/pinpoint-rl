"""PE-core-G at SigLIP2's resolution, with intermediate layers (PE_PLAN.md step 5). Track V's gate found PE-G's per-area columns add +0.79
(+1.37 in the trained head, V5) at 448 px. Two questions in one pass over the same dev3k photos (queries + the zoom file's area photos):
  - resolution: PE-G's pooled embedding at 378 px (27 x 27 patches, the multiple of 14 closest to SigLIP2's 384) vs track V's 448 px;
  - layers: mean-pooled patch tokens of blocks 32 / 38 / 44 / 47 (1-based, the paper's best layers for tracking / detection / VQA) vs the
    pooled output, all at 378 px.

embed (.venvs/sft, timm on PYTHONPATH): -> artifacts/retrieval/gate/pe_g378_dev3k_setft_own.npy (pooled, unit norm, gate format) and
        pe_g378_raw_l{L}.npy (mean-pooled residual stream, float32); resumable.
post:   each layer centred on the area photos' mean (the residual stream has a large shared component) and unit-normed ->
        pe_g378_l{L}_dev3k_setft_own.npy (gate format).
Then: scripts/siglip_gate.py gate --models pe_g pe_g378 pe_g378_l32 ... (and solo).

  PYTHONPATH=<timm>:<value-agent>/src .venvs/sft/bin/python scripts/pe_layers.py embed
  PYTHONPATH=<value-agent>/src .venv/bin/python scripts/pe_layers.py post
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import numpy as np

OUT = Path("artifacts/retrieval/gate")
VALUE = Path("artifacts/value")
MODEL = "vit_pe_core_gigantic_patch14_448.fb"
SIZE = 378
LAYERS = (32, 38, 44, 47)  # 1-based, of 50


def embed(tag: str, ckpt: str, batch: int = 64, workers: int = 4) -> None:
    os.environ.setdefault("HF_HOME", str(Path("data/hf").resolve()))
    os.environ["HF_HUB_OFFLINE"] = "1"
    import timm
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from geo_search_env.experiment import megaloc_gate
    from geo_search_env.experiment.sft_data import MP16Images

    megaloc_gate.CKPT_TAG = ckpt
    ref = np.load(VALUE / f"pe_g_{tag}_{ckpt}_keys.npz")  # track V's rows, so the gate lines up
    need = ref["rows"]
    ids, nq, _ = megaloc_gate._image_ids(tag, need)
    assert nq == int(ref["n_queries"])
    net = timm.create_model(MODEL, pretrained=True, img_size=SIZE).cuda().eval().to(torch.bfloat16)
    cfg = timm.data.resolve_data_config(model=net) | {"input_size": (3, SIZE, SIZE)}
    tf = timm.data.create_transform(**cfg)

    class Images(Dataset):
        images = None

        def __len__(self) -> int:
            return len(ids)

        def __getitem__(self, i: int):
            if self.images is None:
                self.images = MP16Images()
            try:
                key = ids[i]
                data = Path(key).read_bytes() if key.startswith("data/pinpoint/") or key.startswith("/") else self.images.read(key)
                return i, tf(Image.open(io.BytesIO(data)).convert("RGB")), True
            except Exception:  # noqa: BLE001
                return i, torch.zeros(3, SIZE, SIZE), False

    OUT.mkdir(parents=True, exist_ok=True)
    pooled_path, key_path, prog = OUT / f"pe_g378_{tag}_{ckpt}.npy", OUT / f"pe_g378_{tag}_{ckpt}_keys.npz", OUT / f"pe_g378_{tag}_{ckpt}.progress"
    raw_paths = [OUT / f"pe_g378_raw_l{layer}_{tag}_{ckpt}.npy" for layer in LAYERS]
    n = len(ids)
    if prog.exists():
        pooled = np.lib.format.open_memmap(pooled_path, mode="r+")
        raws = [np.lib.format.open_memmap(p, mode="r+") for p in raw_paths]
        ok = np.load(key_path)["ok"]
        start = int(prog.read_text())
    else:
        pooled = np.lib.format.open_memmap(pooled_path, mode="w+", dtype=np.float16, shape=(n, 1280))
        raws = [np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=(n, 1536)) for p in raw_paths]
        ok = np.zeros(n, bool)
        start = 0
        np.savez(key_path, n_queries=nq, rows=need, ok=ok)
    print(f"pe_g378 {tag}: {nq} queries + {len(need)} area photos at {SIZE} px, layers {LAYERS}; starting at {start}", flush=True)
    done = start
    sub = torch.utils.data.Subset(Images(), list(range(start, n)))
    with torch.inference_mode():
        for b, (idx, x, good) in enumerate(DataLoader(sub, batch_size=batch, num_workers=workers)):
            final, inter = net.forward_intermediates(x.cuda().to(torch.bfloat16), indices=[layer - 1 for layer in LAYERS], output_fmt="NLC")
            i = idx.numpy()
            pooled[i] = torch.nn.functional.normalize(net.forward_head(final).float(), dim=-1).cpu().numpy().astype(np.float16)
            for raw, t in zip(raws, inter):
                raw[i] = t.float().mean(1).cpu().numpy()
            ok[i] = good.numpy()
            done += len(i)
            if b % 200 == 199 or done == n:
                pooled.flush()
                for raw in raws:
                    raw.flush()
                np.savez(key_path, n_queries=nq, rows=need, ok=ok)
                prog.write_text(str(done))
                print(f"  {done:,} / {n:,}", flush=True)
    print(f"-> {pooled_path} + {len(raws)} layers: {np.mean(~ok):.2%} unreadable", flush=True)


def post(tag: str, ckpt: str) -> None:
    k = np.load(OUT / f"pe_g378_{tag}_{ckpt}_keys.npz")
    nq, ok = int(k["n_queries"]), k["ok"]
    for layer in LAYERS:
        x = np.load(OUT / f"pe_g378_raw_l{layer}_{tag}_{ckpt}.npy").astype(np.float32)
        area = np.arange(len(x)) >= nq
        x -= x[area & ok].mean(0)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        name = f"pe_g378_l{layer}_{tag}_{ckpt}"
        np.save(OUT / f"{name}.npy", x.astype(np.float16))
        np.savez(OUT / f"{name}_keys.npz", n_queries=nq, rows=k["rows"], ok=ok)
        print(f"-> {OUT / name}.npy", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("node", choices=("embed", "post"))
    ap.add_argument("--tag", default="dev3k")
    ap.add_argument("--ckpt", default="setft_own")
    args = ap.parse_args()
    embed(args.tag, args.ckpt) if args.node == "embed" else post(args.tag, args.ckpt)


if __name__ == "__main__":
    main()
