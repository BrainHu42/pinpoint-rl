"""Pinpoint's attention reranker (the submission's full model), with same-photographer gallery rows excluded.

Loads the submission's own code and checkpoint (run with the submission's `.venv` python and its `src` importable) and
replaces one method: the exact top-k search over the MP16 retrieval index also drops rows whose Flickr photographer is
the query's. Everything else (candidate union of 8 image + 2 raw-image + 2 GPS neighbours, OSV support token, attention
scoring) is the submission's code path. With no query authors given it reproduces the submission's benchmark output.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import types
from typing import Sequence

import numpy as np


PINPOINT_ROOT = Path("/home/brian/workspace/pinpoint-submission/submission")
CHECKPOINT = PINPOINT_ROOT / "exp/attention_reranker/checkpoints/ckpt_best.pt"
MP16_EMBED = Path("/data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384")
MP16_CSV = Path("/data/hf/datasets/MP16-Pro/metadata/MP16_Pro_filtered.csv")
NO_AUTHOR = -1  # queries without a known photographer; never equal to a gallery author id (those are >= 0)


def mp16_author_ids() -> tuple[np.ndarray, dict[str, int]]:
    """Photographer id per MP16 embedding-cache row, and the photographer -> id vocabulary."""

    vocab: dict[str, int] = {}
    with MP16_CSV.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        column = next(reader).index("AUTHOR")
        by_csv_row = np.asarray([vocab.setdefault(row[column], len(vocab)) for row in reader], dtype=np.int64)
    row_index = np.fromfile(MP16_EMBED / "row_index.i64.bin", dtype=np.int64)
    return by_csv_row[row_index], vocab


def query_author_ids(authors: Sequence[str], vocab: dict[str, int]) -> np.ndarray:
    return np.asarray([vocab.get(a, NO_AUTHOR) if a.strip() else NO_AUTHOR for a in authors], dtype=np.int64)


class PinpointReranker:
    """`predict(embeddings, query_authors)` -> top-1 lat/lon, the candidates' index rows, lat/lon and rerank scores."""

    def __init__(self, *, device: str = "cuda", inference_dtype: str = "bfloat16", checkpoint: Path = CHECKPOINT) -> None:
        import torch
        from pinpoint.attention_reranker.inference import AttentionInterpolateGeoModel

        self.torch = torch
        self.model = AttentionInterpolateGeoModel(checkpoint_path=checkpoint, device=device, inference_dtype=inference_dtype)
        module = self.model.geo_module
        module._ensure_retrieval_ready()
        index_dir = Path(module.retrieval_index.index_dir)
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        cache_rows = np.fromfile(index_dir / manifest["files"]["cache_indices"], dtype=np.int64)
        if [s["cache_root"] for s in manifest["sources"]] != [str(MP16_EMBED)]:
            raise ValueError(f"expected an MP16-only retrieval index, got {manifest['sources']}")
        authors, self.vocab = mp16_author_ids()
        self.gallery_author = torch.as_tensor(authors[cache_rows], device=module.device)
        self._query_author = None
        module._topk_from_matrix = types.MethodType(_topk_excluding_authors(self), module)

    def predict(self, embeddings: np.ndarray, query_authors: np.ndarray | None = None, *, batch: int = 1) -> dict[str, np.ndarray]:
        # One photo at a time, as the submission evaluates: batched bf16 search changes ~25% of candidate sets and ~4% of top-1s.
        torch = self.torch
        module = self.model.geo_module
        out: dict[str, list[np.ndarray]] = {"pred": [], "idx": [], "cand_latlon": [], "scores": []}
        for start in range(0, len(embeddings), batch):
            x = self.model._prepare_embeddings(embeddings[start : start + batch])
            authors = None if query_authors is None else query_authors[start : start + batch]
            self._query_author = None if authors is None else torch.as_tensor(authors, device=module.device)
            with torch.inference_mode():
                result = module(x, query_indices=None, apply_leave_one_out=False)
            top = torch.argmax(result["rerank_scores"], dim=-1, keepdim=True)
            latlon = result["candidate_latlon_deg"].to(torch.float32)
            out["pred"].append(module._gather_candidates(latlon, top)[:, 0].cpu().numpy())
            out["idx"].append(result["top_indices"].cpu().numpy())
            out["cand_latlon"].append(latlon.cpu().numpy())
            out["scores"].append(result["rerank_scores"].to(torch.float32).cpu().numpy())
        self._query_author = None
        return {k: np.concatenate(v) for k, v in out.items()}


def _topk_excluding_authors(owner: PinpointReranker):
    """The submission's `_topk_from_matrix` (exact chunked top-k), plus a mask of the current batch's same-author rows."""

    torch = owner.torch

    def _topk_from_matrix(self, *, query, retrieval_mat, k, query_indices, query_exclusion_latlon_deg):
        num_rows = int(retrieval_mat.shape[0])
        k = min(int(k), num_rows)
        if query_indices is not None or query_exclusion_latlon_deg is not None:
            raise ValueError("leave-one-out search is a training feature; not supported here")
        q_author = owner._query_author
        best_scores = torch.full((query.shape[0], k), -float("inf"), device=query.device, dtype=query.dtype)
        best_indices = torch.zeros((query.shape[0], k), device=query.device, dtype=torch.long)
        chunk_size = min(int(self.retrieval_chunk_size), num_rows)
        for start in range(0, num_rows, chunk_size):
            end = min(start + chunk_size, num_rows)
            chunk_scores = torch.matmul(query, retrieval_mat[start:end].T)
            if q_author is not None:
                same = owner.gallery_author[start:end].unsqueeze(0) == q_author.unsqueeze(1)
                chunk_scores.masked_fill_(same, -float("inf"))
            chunk_top_scores, chunk_top_local = torch.topk(chunk_scores, k=min(k, end - start), dim=-1)
            merged_scores = torch.cat((best_scores, chunk_top_scores), dim=-1)
            merged_indices = torch.cat((best_indices, chunk_top_local + int(start)), dim=-1)
            best_scores, keep = torch.topk(merged_scores, k=k, dim=-1)
            best_indices = torch.gather(merged_indices, dim=-1, index=keep)
        return best_indices, best_scores

    return _topk_from_matrix
