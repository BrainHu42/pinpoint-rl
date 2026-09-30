import numpy as np
import pytest

torch = pytest.importorskip("torch")

from geo_search_env.experiment.strategy_search import _stream_topk


def test_same_photographer_rows_never_retrieved_even_when_they_match_best():
    rng = np.random.default_rng(0)
    gallery = rng.normal(size=(6, 8)).astype(np.float16)
    query = gallery[[2]].astype(np.float32)  # identical to gallery row 2, which the query's photographer took
    gallery_author = np.asarray([1, 1, 7, 1, 7, 1])
    idx, sim = _stream_topk(
        gallery, torch.nn.functional.normalize(torch.as_tensor(query), dim=-1).half(), np.asarray([7]),
        gallery_author, top_k=3, normalize=True, chunk=4,  # chunks split the gallery, as on the real disk-backed arrays
    )
    kept = idx[0][np.isfinite(sim[0])]
    assert len(kept) == 3
    assert not np.isin(kept, [2, 4]).any()
