# Frozen Pinpoint Retrieval Baseline

The feasibility pilot uses the submission repository's contrastive retrieval
model as the immutable upstream baseline. The port preserves this inference
path:

```text
query image
  -> google/siglip2-giant-opt-patch16-384
  -> checkpoint mp16 image adapter and shared projection tower
  -> exact dot-product search over the frozen MP16 GPS index
  -> ranked coordinate candidates
```

The rank-1 coordinate is the no-search baseline answer. Higher-ranked
candidates are public starting anchors for the search controller. Their cosine
similarities are mapped from `[-1, 1]` to `[0, 1]` for the existing candidate
confidence field; this value is not a calibrated probability.

```python
from geo_search_env import PinpointImageEmbedder, PinpointRetrievalBaseline

embedder = PinpointImageEmbedder(device="cuda")
retriever = PinpointRetrievalBaseline(device="cuda")

query_embedding = embedder.embed_path("query.jpg")
initial_candidates = retriever.initial_candidates(query_embedding, top_k=5)
```

Install the optional inference dependencies with:

```shell
pip install -e '.[retrieval]'
```

By default, the port uses the frozen checkpoint and version-3 retrieval index
under `/home/brian/workspace/pinpoint-submission/submission`.
Explicit paths may be supplied for a released experiment bundle.

The port loads only the checkpoint's inference-time image tower. It reuses the
already-built GPS embedding index and performs exact search in bounded chunks,
avoiding the original implementation's full multi-gigabyte GPU copy. On the
current artifacts, its encoded query vector and top-five coordinates and
scores were numerically identical to the original implementation.

For a valid held-out result, pilot query images must be absent from the frozen
checkpoint's training data. The MP16 index is part of the frozen retrieval
baseline only; the RL agent cannot query it as an evidence source.
