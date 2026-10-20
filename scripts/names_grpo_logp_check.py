"""Do the trainer's model (transformers) and vLLM agree on the judge's first token? P(A) for 12 dev pairs (order 0) from the HF base model vs
the zero-shot vLLM answers (artifacts/agent/names_grpo_base.jsonl).

Usage: HF_HUB_OFFLINE=1 PYTHONPATH=src .venvs/grpo/bin/python scripts/names_grpo_logp_check.py
"""
import base64
import io
import json

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from geo_search_env.experiment.names_grpo import BASE, PROMPT, _decisive, _image_url
from geo_search_env.experiment.sft_data import MP16Images
from geo_search_env.experiment.sft_train import MAX_PIXELS

proc = AutoProcessor.from_pretrained(BASE, max_pixels=MAX_PIXELS)
model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16).cuda().eval()
ids = [proc.tokenizer.encode(x, add_special_tokens=False) for x in ("A", "B")]
print("token ids A / B:", ids)
vllm = {r["photo"]: r["p_top"][0] for r in map(json.loads, open("artifacts/agent/names_grpo_base.jsonl")) if r["split"] == "dev" and "p_top" in r}
images = MP16Images()
for r in [r for r in _decisive("dev") if r["photo"] in vllm][:12]:
    url = _image_url(r, images)
    im = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT.format(a=r["names"][0], b=r["names"][1])}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    x = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**x).logits[0, -1].float()
    p = torch.softmax(logits, -1)
    pa, pb = p[ids[0][0]].item(), p[ids[1][0]].item()
    print(f"photo {r['photo']:5d}: HF P(A) {pa / (pa + pb):.3f} (A+B mass {pa + pb:.2f}), vLLM {vllm[r['photo']]:.3f}")
print("prompt tail:", repr(text[-120:]))
