"""PE_PLAN step 10: does R17/R18's trained PE-G similarity add to the answer head that already has PE-G columns (V5 + probe, R16)?
Track V's PE-G embeddings of each tag's queries and area photos (448 px, artifacts/value/pe_g_{tag}_setft_own.npy) go through the projection
trained on B1's gallery (378 px; on the same photos the projected 378 / 448 vectors agree at median cosine 0.977, similarities correlate 0.95)
-> a new model "pe_gtr" in second_model_gate's format (embeddings in artifacts/retrieval/pe_gtr/, linked into artifacts/value/), then its
per-area columns (best / top-3 / both relative to the photo's other areas) and the trained-head test with track V's head loop.

  PYTHONPATH=<value-agent>/src:<value-agent>/scripts .venv/bin/python scripts/pe_trained_cols.py cols
  PYTHONPATH=<value-agent>/src:<value-agent>/scripts .venv/bin/python scripts/pe_trained_cols.py head
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

CKPT, TAGS = "setft_own", ("train", "dev3k", "commons26dev")
OUT = Path("artifacts/retrieval/pe_gtr")
VALUE = Path("artifacts/value")


def cols() -> None:
    import second_model_gate as smg
    import torch
    import torch.nn.functional as F

    head = torch.nn.Sequential(torch.nn.Linear(1280, 2048), torch.nn.GELU(), torch.nn.Linear(2048, 256)).cuda()
    head.load_state_dict(torch.load("artifacts/retrieval/b1/trained_heads.pt")["pe_g"])
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in TAGS:
        src, keys = smg._paths("pe_g", tag, CKPT)
        e = np.load(src, mmap_mode="r")
        dst = OUT / f"pe_gtr_{tag}_{CKPT}.npy"
        out = np.lib.format.open_memmap(dst, mode="w+", dtype=np.float16, shape=(len(e), 256))
        with torch.no_grad():
            for a in range(0, len(e), 65536):
                x = F.normalize(torch.tensor(np.asarray(e[a:a + 65536], np.float32)).cuda(), dim=-1)
                out[a:a + 65536] = F.normalize(head(x), dim=-1).cpu().numpy().astype(np.float16)
        out.flush()
        os.link(keys, OUT / f"pe_gtr_{tag}_{CKPT}_keys.npz") if not (OUT / f"pe_gtr_{tag}_{CKPT}_keys.npz").exists() else None
        for f in (dst, OUT / f"pe_gtr_{tag}_{CKPT}_keys.npz"):
            link = VALUE / f.name
            if not link.exists():
                link.symlink_to(f.resolve())
        smg.write("pe_gtr", tag, CKPT, photos="artifacts/value/train_informative_setft_own.npy" if tag == "train" else "")
        print(f"{tag}: {len(e):,} vectors projected", flush=True)


def head() -> None:
    import second_model_gate as smg

    smg.head(["pe_g", "siglip_final", "pe_probe", "pe_gtr"], CKPT, tags=("dev3k", "commons26dev"),
             configs=["none", "pe_g+siglip_final+pe_probe", "pe_g+siglip_final+pe_probe+pe_gtr"])


def single() -> None:
    """Each image model's columns alone in the same head (the user's question: PE-G only vs SigLIP2 only), and PE-G's three column kinds
    without SigLIP2."""
    import second_model_gate as smg

    smg.head(["pe_g", "siglip_final", "pe_probe", "pe_gtr"], CKPT, tags=("dev3k", "commons26dev"),
             configs=["siglip_final", "pe_g", "pe_gtr", "pe_g+pe_probe+pe_gtr"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("node", choices=("cols", "head", "single"))
    {"cols": cols, "head": head, "single": single}[ap.parse_args().node]()


if __name__ == "__main__":
    main()
