# A photo set restricted to one split, with every cached pipeline file of the source set filtered to it (no model runs), so that a pipeline
# variant can be built and evaluated on, e.g., commons26 dev alone without touching its test photos.
# Usage: .venv/bin/python scripts/subset_photo_set.py --src commons26_setft_own --dst commons26dev_setft_own --split dev
import argparse
import json
from pathlib import Path

import numpy as np

DIRS = (Path("artifacts/agent"), Path("artifacts/pinpoint_set"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--split", required=True)
    a = ap.parse_args()
    photos = json.loads((Path("artifacts/photo_sets") / a.src / "photos.json").read_text(encoding="utf-8"))
    keep = np.asarray([i for i, e in enumerate(photos) if e.get("split") == a.split])
    n = len(photos)
    new_pos = np.full(n, -1)
    new_pos[keep] = np.arange(len(keep))
    ids = {photos[i]["image_id"] for i in keep}
    out = Path("artifacts/photo_sets") / a.dst
    out.mkdir(parents=True, exist_ok=True)
    (out / "photos.json").write_text(json.dumps([photos[i] for i in keep]), encoding="utf-8")
    print(f"{a.dst}: {len(keep)} of {n} photos ({a.split})")
    for d in DIRS:
        for f in sorted(d.glob(f"*{a.src}*")):
            if "partial" in f.name:
                continue
            g = d / f.name.replace(a.src, a.dst)
            if f.suffix == ".npz":
                z = dict(np.load(f))
                if "photo" in z:  # per hypothesis (photo column) and per photo arrays
                    m = new_pos[z["photo"]] >= 0
                    nh = len(z["photo"])
                    z = {k: (v[m] if v.ndim and len(v) == nh else v[keep] if v.ndim and len(v) == n else v) for k, v in z.items()}
                    z["photo"] = new_pos[z["photo"]]
                else:  # per photo arrays (set_probs: keys prefixed with the tag)
                    z = {k.replace(a.src, a.dst): (v[keep] if v.ndim and len(v) == n else v) for k, v in z.items()}
                np.savez(g, **z)
            elif f.suffix == ".npy":
                v = np.load(f)
                assert len(v) == n, f
                np.save(g, v[keep])
            elif f.suffix == ".json":
                rows = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(rows, dict):  # keyed by image id
                    g.write_text(json.dumps({k: v for k, v in rows.items() if k in ids}), encoding="utf-8")
                else:
                    assert all("image_id" in r for r in rows), f
                    g.write_text(json.dumps([r for r in rows if r["image_id"] in ids]), encoding="utf-8")
            else:
                continue
            print(f"  {f} -> {g}")


if __name__ == "__main__":
    main()
