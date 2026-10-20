"""The pipeline viewer: a small web app over the traces of viz.traces (and uploaded photos run through viz.live).

  PYTHONPATH=src .venv/bin/python -m geo_search_env.viz.server [--port 8800] [--live]
  then open http://localhost:8800 (from a laptop: ssh -L 8800:localhost:8800 <this machine>)

By default it serves the cached traces only (no GPU, starts in well under a second; the upload button is hidden, earlier uploads stay
browsable). --live starts the upload worker (GPU; viz.live, and scripts/viz_vlm.sh for clue reading) and shows the upload button.

Routes: / (app), /api/sets, /api/index/{set}, /api/errors/{set} (error triage, viz.errors), /api/featured, /api/trace/{set}/{i},
/img/{bench|mp16|upload}/..., POST /api/upload, /api/job/{id}.
"""

from __future__ import annotations

import argparse
import io
import json
import mimetypes
import threading
import urllib.parse
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence

from .agent import OUT
from .traces import TRACES

STATIC = Path(__file__).parent / "static"
FEATURED = Path(__file__).parent / "featured.json"
UPLOADS = OUT / "uploads"
SET_NAMES = {"full": "Test (im2gps3k + yfcc4k eval halves)", "dev3k": "Dev (MP16 dev3k)", "upload": "Your photos"}

_images = None
_lock = threading.Lock()
LIVE = None  # viz.live.Worker when --live


def _mp16():
    global _images
    with _lock:
        if _images is None:
            from ..experiment.sft_data import MP16Images

            _images = MP16Images()
    return _images


@lru_cache(maxsize=4096)
def _thumb(kind: str, key: str, size: int) -> bytes | None:
    from PIL import Image, ImageOps

    if kind == "mp16":
        data = _mp16().read(key)
    elif kind == "bench":
        bench, name = key.split("/", 1)
        path = Path("data/pinpoint") / bench / "images" / Path(name).name
        data = path.read_bytes() if path.exists() else None
    elif kind == "upload":
        path = UPLOADS / Path(key).name / "image.jpg"
        data = path.read_bytes() if path.exists() else None
    else:
        return None
    if data is None:
        return None
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    if size:
        im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _sets() -> list[dict]:
    out = []
    for name, title in SET_NAMES.items():
        if name == "upload":
            n = len(list(UPLOADS.glob("*/trace.json"))) if UPLOADS.exists() else 0
            if n or LIVE is not None:
                out.append({"name": name, "title": title, "n": n, "live": LIVE is not None})
        elif (TRACES / name / "index.json").exists():
            out.append({"name": name, "title": title, "n": len(json.loads((TRACES / name / "index.json").read_text()))})
    return out


def _upload_index() -> list[dict]:
    rows = []
    for p in sorted(UPLOADS.glob("*/trace.json"), key=lambda p: p.stat().st_mtime, reverse=True) if UPLOADS.exists() else []:
        t = json.loads(p.read_text())
        fin = t["agent"].get("0", {}).get("final", {})
        rows.append({"i": t["i"], "image_id": t["image_id"], "benchmark": "upload", "truth_place": t["truth"].get("place") if t["truth"] else "",
                     "released_km": (t["released"]["candidates"] or [{}])[0].get("km"), "retrained_km": (t["retrained"]["candidates"] or [{}])[0].get("km"),
                     "final_km": fin.get("km"), "final_place": fin.get("place"), "actions": [], "n_areas": len(t["areas"])})
    return rows


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, body: bytes, ctype: str = "application/json", status: int = 200, cache: bool = False):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200):
        self._send(json.dumps(obj).encode(), status=status)

    def do_GET(self):  # noqa: N802
        url = urllib.parse.urlparse(self.path)
        parts = [urllib.parse.unquote(p) for p in url.path.strip("/").split("/") if p]
        q = urllib.parse.parse_qs(url.query)
        try:
            if not parts:
                return self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if parts[0] == "static" and len(parts) == 2:
                path = STATIC / Path(parts[1]).name
                if path.exists():
                    return self._send(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            if parts[:2] == ["api", "sets"]:
                return self._json(_sets())
            if parts[:2] == ["api", "featured"]:
                return self._json(json.loads(FEATURED.read_text()) if FEATURED.exists() else [])
            if parts[:2] == ["api", "errors"] and len(parts) == 3:
                from .errors import load

                return self._json(load(Path(parts[2]).name))
            if parts[:2] == ["api", "index"] and len(parts) == 3:
                if parts[2] == "upload":
                    return self._json(_upload_index())
                return self._send((TRACES / Path(parts[2]).name / "index.json").read_bytes())
            if parts[:2] == ["api", "trace"] and len(parts) == 4:
                if parts[2] == "upload":
                    return self._send((UPLOADS / Path(parts[3]).name / "trace.json").read_bytes())
                return self._send((TRACES / Path(parts[2]).name / f"{int(parts[3])}.json").read_bytes())
            if parts[:2] == ["api", "job"] and len(parts) == 3:
                return self._json(LIVE.status(parts[2]) if LIVE else {"state": "error", "message": "live mode is off"})
            if parts[0] == "img" and len(parts) >= 3:
                body = _thumb(parts[1], "/".join(parts[2:]), int(q.get("s", ["0"])[0]))
                if body is not None:
                    return self._send(body, "image/jpeg", cache=True)
            self._json({"error": "not found"}, 404)
        except FileNotFoundError:
            self._json({"error": "not found"}, 404)
        except Exception as exc:  # keep serving
            self._json({"error": repr(exc)}, 500)

    def do_POST(self):  # noqa: N802
        url = urllib.parse.urlparse(self.path)
        if url.path != "/api/upload":
            return self._json({"error": "not found"}, 404)
        if LIVE is None:
            return self._json({"error": "uploads need the live worker: restart the server with --live"}, HTTPStatus.SERVICE_UNAVAILABLE)
        q = urllib.parse.parse_qs(url.query)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        lat, lon = q.get("lat", [""])[0], q.get("lon", [""])[0]
        truth = (float(lat), float(lon)) if lat and lon else None
        job = LIVE.submit(body, q.get("name", ["upload.jpg"])[0], truth)
        self._json({"job": job})


def main(argv: Sequence[str] | None = None) -> int:
    global LIVE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--live", action="store_true", help="accept uploads and run them through the pipeline (GPU, viz.live)")
    args = parser.parse_args(argv)
    if args.live:
        from .live import Worker

        LIVE = Worker()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"pipeline viewer on http://{args.host}:{args.port}{' (live uploads on)' if LIVE else ''}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
