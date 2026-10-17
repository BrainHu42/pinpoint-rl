# Read single OSV-5M images straight from the Hugging Face zip shards with HTTP range requests (the shards are ~2.5 GB each; we need a few
# thousand images). Usage: .venv/bin/python -m geo_search_env.experiment.osv_remote index  (writes artifacts/clue_verify/osv_zip_index.json)

"""Remote zip access for OSV-5M (huggingface.co/datasets/osv5m/osv5m, images/train/NN.zip).

index: reads every shard's central directory (a few MB each) and writes image id -> (shard, local header offset, compressed size,
       method) to artifacts/clue_verify/osv_zip_index.json.
read(image_id): the JPEG bytes, from one range request (local header + data).
"""

from __future__ import annotations

import io
import json
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://huggingface.co/datasets/osv5m/osv5m/resolve/main/images/train/"
INDEX = Path("artifacts/clue_verify/osv_zip_index.json")


def _get(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "pinpoint-rl"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile to parse the central directory)."""

    def __init__(self, url: str) -> None:
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "pinpoint-rl"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.geturl()  # the CDN location after redirects

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        data = _get(self.url, self.pos, min(self.pos + n, self.size) - 1)
        self.pos += len(data)
        return data


def index(shards: list[str]) -> None:
    out = json.loads(INDEX.read_text()) if INDEX.exists() else {}
    done = {v[0] for v in out.values()}
    for shard in shards:
        if shard in done:
            continue
        with zipfile.ZipFile(RangeFile(BASE + shard)) as z:
            for info in z.infolist():
                stem = Path(info.filename).stem
                if info.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    out[stem] = [shard, info.header_offset, info.compress_size, info.compress_type]
        print(f"{shard}: {len(out):,} images indexed", flush=True)
        INDEX.parent.mkdir(parents=True, exist_ok=True)
        INDEX.write_text(json.dumps(out))


_URLS: dict[str, str] = {}


def read(entry) -> bytes | None:
    shard, offset, size, method = entry
    url = _URLS.get(shard) or _URLS.setdefault(shard, RangeFile(BASE + shard).url)
    try:
        raw = _get(url, offset, offset + 30 + 1024 + size)  # local header (30 + name + extra) + data; extra rarely > 1 KB
    except Exception:
        return None
    n_name, n_extra = struct.unpack("<HH", raw[26:30])
    data = raw[30 + n_name + n_extra: 30 + n_name + n_extra + size]
    if method == zipfile.ZIP_DEFLATED:
        import zlib
        data = zlib.decompress(data, -15)
    return data


def shards() -> list[str]:
    with urllib.request.urlopen("https://huggingface.co/api/datasets/osv5m/osv5m/tree/main/images/train", timeout=60) as r:
        return sorted(Path(e["path"]).name for e in json.load(r) if e["path"].endswith(".zip"))


if __name__ == "__main__":
    if sys.argv[1:] == ["index"]:
        index(shards())
