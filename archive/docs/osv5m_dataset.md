# OSV-5M Dataset Access

`OSV5MDataset` joins the extracted OSV-5M image tree with the cached SigLIP2
embedding store by cache row. Its defaults point to the local research stores:

```text
raw images: /data/hf/datasets/osv5m
embeddings: /data/pinpoint/osv5m-embed/siglip2-giant-opt-patch16-384
```

```python
from geo_search_env import OSV5MDataset

dataset = OSV5MDataset()
sample = dataset[0]

print(sample.image_id)
print(sample.image_path)
print(sample.embedding.shape)
print(sample.latitude, sample.longitude)
```

Each sample exposes the source image ID, cache and source-CSV row indices,
latitude and longitude, raw image path, and a read-only NumPy view of the
cached embedding. Pass `load_image_bytes=True` to include the original encoded
image bytes. `embedding_at(index)` accesses embeddings without resolving or
reading raw images.

The loader validates the cache format, shapes, dtypes, file sizes, source
split, and extracted image tree. Large binary arrays remain read-only memory
maps. Image IDs are indexed lazily with bounded memory instead of being kept as
millions of Python strings.

The loader deliberately does not reproduce the submission repository's
ID-hash train/validation partition. This experiment requires geographic,
same-sequence, same-capture, and near-duplicate grouping before assigning
splits. Those audited split indices should be layered over the cache-row index
rather than generated inside this loader.
