# Usage: PYTHONPATH=src .venv/bin/python scripts/scene_zeroshot.py  (lesson R5; writes artifacts/retrieval/skyline_zeroshot_dev3k.npy)
# Zero-shot sizing: how many dev3k photos show terrain skylines, and how the pipeline does on them.
import json, sys
import numpy as np, torch
sys.path.insert(0, "src")
from geo_search_env.experiment.embed_cache import MODEL
from transformers import AutoModel, AutoProcessor

photos = json.loads(open("artifacts/photo_sets/dev3k/photos.json").read())
rows = np.asarray([e["row"] for e in photos])
emb = np.memmap("data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384/embeddings.f16.bin", dtype=np.float16, mode="r").reshape(-1, 1536)
x = torch.nn.functional.normalize(torch.as_tensor(np.asarray(emb[rows], dtype=np.float32)), dim=-1)
model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
proc = AutoProcessor.from_pretrained(MODEL)
prompts = {
    "mountain skyline": "a photo of mountains with a clear skyline against the sky",
    "hills on horizon": "a landscape photo with hills on the horizon",
    "flat horizon": "a photo of a flat landscape or sea horizon",
    "city street": "a photo of a city street with buildings",
    "indoor": "a photo taken indoors",
    "people closeup": "a close-up photo of people",
    "object closeup": "a close-up photo of an object or food",
}
with torch.inference_mode():
    t = proc(text=list(prompts.values()), padding="max_length", max_length=64, return_tensors="pt").to("cuda")
    out = model.get_text_features(**t)
    out = out if not hasattr(out, "pooler_output") else out.pooler_output
    txt = torch.nn.functional.normalize(out.float().cpu(), dim=-1)
logits = (x @ txt.T) * model.logit_scale.exp().float().cpu() + model.logit_bias.float().cpu()
p = torch.sigmoid(logits).detach().numpy()
lab = np.asarray(list(prompts))[p.argmax(1)]
A = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz")
km = A["dev3k|RL policy, lambda 0"]
print(f"pipeline dev3k < 25 km {100 * np.mean(km < 25):.1f}")
for k in prompts:
    m = lab == k
    print(f"{k:18s} {100 * m.mean():5.1f}% of photos | pipeline < 25 km {100 * np.mean(km[m] < 25):5.1f} | wrong photos {100 * np.mean(m & (km >= 25)):4.1f} pts")
for th in (0.1, 0.3, 0.5):
    m = p[:, 0] > th
    print(f"P(mountain skyline) > {th}: {100 * m.mean():.1f}% of photos, pipeline {100 * np.mean(km[m] < 25):.1f}, wrong {100 * np.mean(m & (km >= 25)):.1f} pts")
np.save("artifacts/retrieval/skyline_zeroshot_dev3k.npy", p)
