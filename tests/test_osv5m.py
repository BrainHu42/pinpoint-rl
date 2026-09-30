import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from geo_search_env import OSV5MDataset


class OSV5MDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.raw_root = root / "raw"
        self.cache_root = root / "cache"
        shard = self.raw_root / "images" / "train" / "00"
        shard.mkdir(parents=True)
        self.cache_root.mkdir()
        (shard / "image-a.jpg").write_bytes(b"raw-a")
        (shard / "image-b.jpg").write_bytes(b"raw-b")
        (self.cache_root / "image_ids.txt").write_text("image-a\nimage-b\n", encoding="utf-8")
        np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float16).tofile(
            self.cache_root / "embeddings.f16.bin"
        )
        np.asarray([[10, 20], [-30, 40]], dtype=np.float32).tofile(
            self.cache_root / "latlon_deg.f32.bin"
        )
        np.asarray([7, 11], dtype=np.int64).tofile(self.cache_root / "row_index.i64.bin")
        manifest = {
            "format": "flat_binary_memmap_v1",
            "num_samples": 2,
            "embedding_dim": 3,
            "source_split": "train",
            "files": {
                "embeddings": "embeddings.f16.bin",
                "image_ids": "image_ids.txt",
                "latlon_deg": "latlon_deg.f32.bin",
                "row_index": "row_index.i64.bin",
            },
            "dtypes": {"embeddings": "float16", "row_index": "int64"},
            "shapes": {"embeddings": [2, 3], "latlon_deg": [2, 2], "row_index": [2]},
        }
        (self.cache_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_joins_raw_image_and_embedding_by_cache_row(self):
        dataset = OSV5MDataset(self.raw_root, self.cache_root, load_image_bytes=True)
        self.assertEqual(len(dataset), 2)
        self.assertIsNone(dataset._embeddings)
        sample = dataset[1]
        self.assertEqual(sample.image_id, "image-b")
        self.assertEqual(sample.source_row_index, 11)
        self.assertEqual((sample.latitude, sample.longitude), (-30.0, 40.0))
        self.assertEqual(sample.image_path.name, "image-b.jpg")
        self.assertEqual(sample.image_bytes, b"raw-b")
        np.testing.assert_array_equal(sample.embedding, np.asarray([4, 5, 6], dtype=np.float16))

    def test_embedding_access_does_not_resolve_or_read_an_image(self):
        dataset = OSV5MDataset(self.raw_root, self.cache_root)
        np.testing.assert_array_equal(
            dataset.embedding_at(0), np.asarray([1, 2, 3], dtype=np.float16)
        )
        self.assertEqual(dataset.image_id_at(-1), "image-b")

    def test_rejects_truncated_binary_cache(self):
        (self.cache_root / "embeddings.f16.bin").write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "file size mismatch"):
            OSV5MDataset(self.raw_root, self.cache_root)


if __name__ == "__main__":
    unittest.main()
