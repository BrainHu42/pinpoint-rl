"""Released commons26 images must not carry the GPS (or any EXIF) they were uploaded with, and stripping must not change pixels."""

import io

import numpy as np
from PIL import Image

from geo_search_env.experiment.commons_bench import _strip_metadata


def test_strip_removes_gps_and_keeps_pixels():
    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (32, 48, 3), dtype=np.uint8))
    exif = Image.Exif()
    exif.get_ifd(0x8825)[2] = (48.0, 51.0, 24.0)  # GPSLatitude
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", exif=exif, comment=b"Eiffel Tower, Paris")
    original = buffer.getvalue()
    assert Image.open(io.BytesIO(original)).getexif().get_ifd(0x8825)

    stripped = _strip_metadata(original)

    assert len(Image.open(io.BytesIO(stripped)).getexif()) == 0
    assert b"Eiffel" not in stripped
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(original))), np.asarray(Image.open(io.BytesIO(stripped))))
