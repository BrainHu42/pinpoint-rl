"""Region expert screen, step 1 (user, 2026-10-24): do geo-pretrained models break the ~70% within 750 km ceiling of frozen-embedding region
readouts (lesson V14)? Zero-shot on dev3k only; bar (pre-declared): region accuracy on area-oracle misses >= 5 pts above the B2 PE-G
expert's 50.1% within 750 km, or right on >= 25% of the misses where that expert is > 750 km off.

  StreetCLIP (geolocal/StreetCLIP: CLIP ViT-L/14-336 trained on street-view images, not MP16): zero-shot over GeoNames first-level regions
      ("A Street View photo in {region}, {country}.") and countries ("A Street View photo in {country}."); prediction = the top region's
      GeoNames point, and the top region inside the top country (hierarchical).
  GeoCLIP (pip geoclip: CLIP ViT-L/14 image encoder + GPS encoder over a 100k-coordinate gallery): top-1 GPS. Trained on MP16, which dev3k
      is drawn from: an upper bound, not a clean result.

GPU: one model at a time, capped at GPU_GB (default 6 GB, torch.cuda.set_per_process_memory_fraction), small batches, so it fits beside
track R's B2 embed (~11 GB).

  PYTHONPATH=src .venv/bin/python scripts/geo_zeroshot.py dump                        (dev3k JPEGs -> IMG_DIR; admin1 table)
  HF_HOME=data/hf .venvs/sft/bin/python scripts/geo_zeroshot.py streetclip
  .venvs/geoclip/bin/python scripts/geo_zeroshot.py geoclip
  PYTHONPATH=src .venv/bin/python scripts/geo_zeroshot.py report
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

OUT = Path("artifacts/value")
IMG_DIR = Path(os.environ.get("GEO_ZS_IMAGES", "artifacts/value/dev3k_images"))
ADMIN1 = OUT / "admin1.npz"
GPU_GB = float(os.environ.get("GPU_GB", "6"))
STREETCLIP = "geolocal/StreetCLIP"


def _ids() -> list[str]:
    return [p["image_id"] for p in json.loads(Path("artifacts/photo_sets/dev3k_setft_own/photos.json").read_text(encoding="utf-8"))]


def _cap_gpu() -> None:
    import torch

    torch.cuda.set_per_process_memory_fraction(min(1.0, GPU_GB * 2**30 / torch.cuda.get_device_properties(0).total_memory))


def dump() -> None:
    from geo_search_env.experiment.sft_data import MP16Images

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    images = MP16Images()
    for i, k in enumerate(_ids()):
        f = IMG_DIR / f"{i:04d}.jpg"
        if not f.exists():
            f.write_bytes(images.read(k))
    countries = {}
    for line in Path("data/pinpoint/geonames/countryInfo.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            p = line.split("\t")
            countries[p[0]] = p[4]
    name, cc, lat, lon = [], [], [], []
    with Path("data/pinpoint/geonames/allCountries.txt").open(encoding="utf-8") as f:
        for line in f:
            p = line.split("\t")
            if p[6] == "A" and p[7] == "ADM1" and p[8] in countries:
                name.append(p[2] or p[1])
                cc.append(countries[p[8]])
                lat.append(float(p[4]))
                lon.append(float(p[5]))
    np.savez(ADMIN1, name=np.asarray(name), country=np.asarray(cc), lat=np.asarray(lat), lon=np.asarray(lon))
    print(f"{len(_ids())} images -> {IMG_DIR}; {len(name)} first-level regions in {len(set(cc))} countries -> {ADMIN1}", flush=True)


def streetclip(batch: int = 32) -> None:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    _cap_gpu()
    a = np.load(ADMIN1)
    regions = [f"A Street View photo in {n}, {c}." for n, c in zip(a["name"], a["country"])]
    cnames = sorted(set(a["country"]))
    model = CLIPModel.from_pretrained(STREETCLIP, torch_dtype=torch.float16).cuda().eval()
    proc = CLIPProcessor.from_pretrained(STREETCLIP)

    def text(prompts):
        out = []
        for s in range(0, len(prompts), 256):
            t = proc(text=prompts[s:s + 256], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                e = model.text_projection(model.text_model(**t).pooler_output)
            out.append(torch.nn.functional.normalize(e.float(), dim=-1))
        return torch.cat(out)

    tr, tc = text(regions), text([f"A Street View photo in {c}." for c in cnames])
    files = sorted(IMG_DIR.glob("*.jpg"))
    img = []
    for s in range(0, len(files), batch):
        x = proc(images=[Image.open(f).convert("RGB") for f in files[s:s + batch]], return_tensors="pt").to("cuda")
        with torch.no_grad():
            img.append(torch.nn.functional.normalize(model.visual_projection(model.vision_model(pixel_values=x["pixel_values"].half()).pooler_output).float(), dim=-1))
    img = torch.cat(img)
    scale = model.logit_scale.exp().float().detach()
    lr = torch.log_softmax(scale * img @ tr.T, 1).cpu().numpy()
    lc = torch.log_softmax(scale * img @ tc.T, 1).cpu().numpy()
    np.savez(OUT / "zeroshot_streetclip_dev3k.npz", region_logp=lr.astype(np.float32), country_logp=lc.astype(np.float32),
             countries=np.asarray(cnames))
    print(f"streetclip: {len(files)} images, {len(regions)} region and {len(cnames)} country prompts; peak GPU "
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)


def geoclip(batch: int = 32) -> None:
    import torch
    from geoclip import GeoCLIP

    _cap_gpu()
    model = GeoCLIP().cuda().eval()
    files = sorted(IMG_DIR.glob("*.jpg"))
    gps, prob = [], []
    for f in files:
        g, p = model.predict(str(f), top_k=5)
        gps.append(np.asarray(g, np.float32))
        prob.append(np.asarray(p, np.float32))
    np.savez(OUT / "zeroshot_geoclip_dev3k.npz", gps=np.stack(gps), prob=np.stack(prob))
    print(f"geoclip: {len(files)} images; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)


def report() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import coarse_signals as cs
    import hedge_search as hs

    photos = hs._photos("dev3k")
    truth = np.asarray([e["truth"] for e in photos], np.float64)
    miss = ~hs._area_oracle("dev3k")
    t_country = cs._country(truth)
    e = np.load(OUT / "region_expert_b2_8192.npz")
    guesses = {"PE-G expert (B2, 8,192 cells)": e["cells"].astype(np.float64)[e["logp"].argmax(1)],
               "region probe": np.load("artifacts/retrieval/probe/cells.npy").astype(np.float64)[np.load(OUT / "probe_logp_dev3k.npy").argmax(1)]}
    a = np.load(ADMIN1)
    pts = np.stack([a["lat"], a["lon"]], 1).astype(np.float64)
    if (OUT / "zeroshot_streetclip_dev3k.npz").exists():
        z = np.load(OUT / "zeroshot_streetclip_dev3k.npz")
        guesses["StreetCLIP, top region"] = pts[z["region_logp"].argmax(1)]
        top_c = z["countries"][z["country_logp"].argmax(1)]
        hier = np.where(a["country"][None, :] == top_c[:, None], z["region_logp"], -np.inf).argmax(1)
        guesses["StreetCLIP, region in top country"] = pts[hier]
        sc_country = np.asarray([cs._norm(c) for c in top_c])
    if (OUT / "zeroshot_geoclip_dev3k.npz").exists():
        guesses["GeoCLIP top-1 (trained on MP16: upper bound)"] = np.load(OUT / "zeroshot_geoclip_dev3k.npz")["gps"][:, 0].astype(np.float64)
    ok_e = hs._km(truth, guesses["PE-G expert (B2, 8,192 cells)"]) < 750
    print(f"dev3k: right within 25 / 200 / 750 / 2500 km, % of all photos | % of area-oracle misses ({miss.mean():.1%}); "
          f"right on misses where the PE-G expert is > 750 km off ({(miss & ~ok_e).mean() / miss.mean():.1%} of misses)")
    for name, g in guesses.items():
        d = hs._km(truth, g)
        cells = "  ".join(f"{100 * np.mean(d < r):5.1f} | {100 * np.mean((d < r)[miss]):5.1f}" for r in (25, 200, 750, 2500))
        print(f"   {name:46s} {cells}   rescue {100 * np.mean((d < 750)[miss & ~ok_e]):5.1f}")
    if (OUT / "zeroshot_streetclip_dev3k.npz").exists():
        e_c = cs._country(guesses["PE-G expert (B2, 8,192 cells)"])
        print(f"   country right: StreetCLIP {100 * np.mean(sc_country == t_country):.1f} | {100 * np.mean((sc_country == t_country)[miss]):.1f}; "
              f"PE-G expert's top cell {100 * np.mean(e_c == t_country):.1f} | {100 * np.mean((e_c == t_country)[miss]):.1f}")
    best = max((n for n in guesses if "StreetCLIP" in n), key=lambda n: np.mean((hs._km(truth, guesses[n]) < 750)[miss]), default=None)
    if best:
        acc = 100 * np.mean((hs._km(truth, guesses[best]) < 750)[miss])
        res = 100 * np.mean((hs._km(truth, guesses[best]) < 750)[miss & ~ok_e])
        base = 100 * np.mean(ok_e[miss])
        print(f"   bar ({best}): misses within 750 km {acc:.1f} vs PE-G {base:.1f} + 5 -> {'met' if acc >= base + 5 else 'not met'}; "
              f"rescue {res:.1f} vs 25 -> {'met' if res >= 25 else 'not met'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("node", choices=("dump", "streetclip", "geoclip", "report"))
    args = ap.parse_args()
    {"dump": dump, "streetclip": streetclip, "geoclip": geoclip, "report": report}[args.node]()


if __name__ == "__main__":
    main()
