# Setup

How to rebuild the environments, models, datasets and caches for this repo from scratch, written so a person or a
coding agent can follow it end to end. See `README.md` for what the project is and `LEARNINGS.md` for results.

## 1. Hardware and disk

- One NVIDIA GPU with ≥ 32 GB (developed on an RTX 5090, driver 595, CUDA 13 wheels). SFT peaks at ~17–27 GB,
  GRPO with colocated vLLM at ~26 GB; vLLM evaluation servers take 70% of the GPU.
- Linux, Python 3.11 and 3.12, [uv](https://github.com/astral-sh/uv) ≥ 0.10.
- Disk (all under `/data` by default):

  | What | Size |
  |---|---|
  | MP16-Pro images + metadata | 371 GB |
  | OSV-5M (only needed to build its embeddings) | 251 GB |
  | MP16 / OSV-5M SigLIP2 embeddings | 37 GB / 22 GB |
  | im2gps3k + yfcc4k (images + embeddings) | 1 GB |
  | Pinpoint checkpoint + retrieval index | 3.2 GB |
  | Model weights (Qwen3.5-4B, SigLIP2 giant) | ~20 GB |
  | GeoNames dump | 1.7 GB |
  | Our caches and results (`artifacts/`) | ~1.5 GB |
  | Each SFT/GRPO run (adapter + merged model) | ~9 GB |

## 2. Paths the code assumes

Most paths are module constants, not configuration. Either recreate this layout (symlinks are fine) or edit the
constants listed here.

| Path | Used for | Defined in |
|---|---|---|
| `/data/hf` (`HF_HOME`), `/data/hf/hub` | Hugging Face cache; Qwen3.5-4B snapshot | `experiment/sft_train.py` (`BASE_MODEL`) |
| `/data/hf/datasets/MP16-Pro` | MP16 metadata CSV, tar shards, `metadata/tar_index.pkl` | `strategy_search.MP16_CSV`, `verifiers.MP16_ROOT` |
| `/data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384` | MP16 gallery embeddings | `strategy_search.MP16_EMBED` |
| `/data/pinpoint/osv5m-embed/siglip2-giant-opt-patch16-384` | OSV-5M gallery embeddings | `strategy_search.OSV_EMBED` |
| `/data/pinpoint/{im2gps3k,yfcc4k}` | benchmark CSVs, images, embeddings | `data/benchmarks.py` (`DATA_ROOT`) |
| `/data/pinpoint/geonames` | GeoNames dump | `llm_advantage.py`, `evidence_test.py` |
| `/data/pinpoint/sft` | SFT/GRPO checkpoints | `sft_train.RUNS` |
| `$PINPOINT_ROOT` | Pinpoint checkpoint + retrieval index | `models/pinpoint.py` |
| `~/.venvs/{vllm,sft,grpo}` | serving and training environments | `scripts/*.sh` |

Copy `.env.example` to `.env`, fill it in, and load it with `set -a; . ./.env; set +a` before running anything.

## 3. Python environments

Four environments, because training, serving and analysis pin different torch versions.

```bash
# (a) Repo env: analysis, retrieval caches, dataset building, evaluation clients. Python 3.11, torch 2.14 (CUDA 13).
uv sync --all-extras

# (b) Serving: vLLM 0.30 (torch 2.13). Put its bin on PATH when serving (vLLM needs its ninja).
uv venv ~/.venvs/vllm --python 3.12
uv pip install --python ~/.venvs/vllm/bin/python vllm==0.30.0

# (c) SFT training. Python 3.11.
uv venv ~/.venvs/sft --python 3.11
uv pip install --python ~/.venvs/sft/bin/python torch==2.14.0 torchvision==0.29.0 transformers==5.17.0 trl==1.14.0 \
  peft==0.21.0 accelerate==1.15.0 datasets==5.0.1 flash-linear-attention==0.5.2 pillow scipy pytest
uv pip install --python ~/.venvs/sft/bin/python -e . --no-deps

# (d) GRPO with colocated vLLM rollouts: vLLM 0.30 plus the trainer stack. Python 3.12.
uv venv ~/.venvs/grpo --python 3.12
uv pip install --python ~/.venvs/grpo/bin/python vllm==0.30.0 trl==1.14.0 peft==0.21.0 accelerate==1.15.0 \
  datasets==5.0.1 flash-linear-attention==0.5.2 scipy pytest
uv pip install --python ~/.venvs/grpo/bin/python -e . --no-deps
```

Check: `~/.venvs/sft/bin/python -m pytest -q` (30 tests, CPU only). `causal_conv1d` is not needed (the torch fallback
is used).

## 4. Models

```bash
export HF_HOME=/data/hf
.venv/bin/hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
.venv/bin/hf download google/siglip2-giant-opt-patch16-384
```

`sft_train.BASE_MODEL` points at that Qwen snapshot directory; update it if you use another revision.

## 5. Datasets

### MP16-Pro (training pool and retrieval gallery)
Hugging Face [`Jia-py/MP16-Pro`](https://huggingface.co/datasets/Jia-py/MP16-Pro) (gated: accept the terms and run
`hf auth login` first), revision `e50ce1ef84f157fa80b17563e9bdb0af1d510b5c`: ~4.12M Flickr photos in tar shards
`mp-16-images00..18`, `metadata/MP16_Pro_filtered.csv` (columns include `AUTHOR`, `city`, `county`, `state`,
`country`) and `metadata/tar_index.pkl` (image id → shard offset).

```bash
.venv/bin/hf download Jia-py/MP16-Pro --repo-type dataset --revision e50ce1ef84f157fa80b17563e9bdb0af1d510b5c \
  --local-dir /data/hf/datasets/MP16-Pro
```

### OSV-5M (second retrieval gallery)
Hugging Face [`osv5m/osv5m`](https://huggingface.co/datasets/osv5m/osv5m) → `/data/hf/datasets/osv5m`. Only its
embeddings are read at run time. It has no photographer ids, so OSV rows are never excluded as same-photographer.

### Benchmarks: im2gps3k and yfcc4k
Expected layout under `/data/pinpoint` (read by `src/geo_search_env/data/benchmarks.py`):

```
im2gps3k/im2gps3k_places365.csv   columns IMG_ID, AUTHOR, LAT, LON, ...   (3,000 images; 2,997 embedded)
im2gps3k/images/<IMG_ID>          e.g. 1000269685_e60e9cdfb4_1125_78841376@N00.jpg
yfcc4k/yfcc4k.csv                 columns IMG_ID, OwnerNSID, LAT, LON, ... (YFCC100M metadata; 4,536 images)
yfcc4k/images/<IMG_ID>            e.g. 10201275523.jpg
<bench>/image_embeddings/google_siglip2-giant-opt-patch16-384/{manifest.json,image_ids.txt,embeddings.f16.bin}
```

These are the standard im2gps3k test set (Vo et al., ICCV 2017) and the YFCC4k subset of YFCC100M; the
`im2gps3k_places365.csv` metadata is the version distributed with
[GeoEstimation](https://github.com/TIBHannover/GeoEstimation) / [G3](https://github.com/Applied-Machine-Learning-Lab/G3).
The author columns (`AUTHOR`, `OwnerNSID`) are required: they drive the same-photographer exclusion. Our copies came with
the Pinpoint submission and their original download URLs were not recorded, so check row and image counts against the
numbers above.

The tune/eval split is a fixed hash of `benchmark:image_id` (`strategy_search.load_world`), so it is reproducible.

### SigLIP2 embedding caches
All galleries and benchmarks use the raw `get_image_features` output of `google/siglip2-giant-opt-patch16-384` (fp16,
dim 1536, *not* normalized), stored as flat binaries with a `manifest.json`:

- galleries: `embeddings.f16.bin`, `latlon_deg.f32.bin`, `row_index.i64.bin` (row → metadata CSV row) and
  `image_ids.txt`; manifest keys `files` and `shapes.embeddings`;
- benchmarks: `embeddings.f16.bin` and `image_ids.txt`; manifest keys `files` and `embedding_dim`.

Build them with `experiment/embed_cache.py` (GPU; writes to the paths in section 2 unless `--out` is given):

```bash
.venv/bin/python -m geo_search_env.experiment.embed_cache benchmark im2gps3k   # and yfcc4k
.venv/bin/python -m geo_search_env.experiment.embed_cache gallery mp16         # ~11 h for our cache, bound by JPEG decoding
.venv/bin/python -m geo_search_env.experiment.embed_cache gallery osv5m        # ~13 h
```

Our caches came from the Pinpoint submission's builder. `embed_cache` reproduces them: same rows in the same order
(gallery row order depends on the DataLoader layout, so keep the default `--workers`/`--batch-size`, or anything in
`artifacts/` that stores gallery row numbers is invalid), and embeddings at cosine median 0.9999, min 0.997 to ours
on the checked prefixes (it resizes with PIL where Pinpoint used torchvision). To check a copy, build a prefix and compare, e.g.
`embed_cache gallery mp16 --limit 2048 --out /tmp/mp16-check --compare /data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384`.

### Pinpoint retrieval model
The Pinpoint submission is a separate codebase that is not public. From it we need only:

```
$PINPOINT_ROOT/exp/contrastive_retrieval/checkpoints/ckpt_best.pt                  (1.2 GB)
$PINPOINT_ROOT/exp/contrastive_retrieval/checkpoints/retrieval_index/<hash>/        (2 GB)
    manifest.json, gps_embeddings.bin, gps_latlon_deg.bin
```

The index manifest stores the checkpoint's absolute path: if you move the checkout, set `checkpoint_path` in that
`manifest.json` to the new location. Pinpoint's retriever trained on MP16 photos with `md5(image_id) % 100 < 99`,
which is why training uses only bucket 99 (see `LEARNINGS.md`). Without Pinpoint the photo-matching galleries still
work, but the `mp16_gps` neighbours, and so the candidate pools, cannot be rebuilt.

### GeoNames (nearby-place evidence and geocoding baseline)
```bash
mkdir -p /data/pinpoint/geonames && cd /data/pinpoint/geonames
curl -LO https://download.geonames.org/export/dump/allCountries.zip && unzip allCountries.zip && rm allCountries.zip
curl -LO https://download.geonames.org/export/dump/featureCodes_en.txt
curl -LO https://download.geonames.org/export/dump/countryInfo.txt
```

### Optional
- `/data/pinpoint/yfcc26k`: reserve training data (no photo overlap with MP16 or the benchmarks; lat/lon only).
- `/data/pinpoint/wikimedia`: planned third test set; not yet wired into `data/benchmarks.py`.

## 6. Build the caches (in order)

Run from the repo root as `.venv/bin/python -m geo_search_env.experiment.<module> <node>`; the first lines of each
module document its nodes. Everything is written under `artifacts/`, which is git-ignored.

Benchmark side (`--root artifacts/strategy_search`, the default):
1. `strategy_search neighbors`: top-1000 MP16/OSV photo matches and top-500 Pinpoint GPS neighbours per benchmark
   query, same photographer excluded (GPU; streams the galleries from disk).
2. `strategy_search region_head`: region-classifier head over MP16 embeddings.
3. `strategy_search coarse`: selects the head and the Pinpoint kNN vote settings (`coarse.json`).
4. `strategy_search search`: 20 pooled candidates per query plus reranker features (`search_features.npz`). The
   one-step reranker is refit from this file on the tune halves whenever it is needed (`verifiers.reranker_ranking`).
5. Optional: `pivot_diagnostics region_search` (region-restricted retrieval, `pivot_region_pools.npz`).

MP16 training side (`--root artifacts/sft`, the default):
1. `sft_data pool`: Pinpoint-held-out bucket-99 photos, minus benchmark photographers and near-duplicates of benchmark
   photos (`queries.json`).
2. `sft_data candidates`: neighbours, region head and the benchmark pipeline's candidates for those photos (needs the
   benchmark `coarse.json` and `search_features.npz`).
3. `sft_data leak_check`: candidate quality on held-out vs trained-on photos (sanity check).
4. `sft_data overlay`: P(burned-in GPS coordinates) per photo from base Qwen served by vLLM (see §7).
5. `sft_data dataset --variant sft_retrieval`: `sft_retrieval.jsonl` (34,523 train / 3,836 val). Other variants:
   `sft`, `sft_evidence`.

Analyses: `llm_advantage` (Gemini labels need `OPENROUTER_API_KEY`), `query_headroom {crops,text}`,
`evidence_test landmarks` (GeoNames landmark index, ~1 min) then `evidence_test run --evidence {none,photos,places}`.

## 7. Serving, training and evaluation

```bash
# Serve a model for evaluation (base Qwen or a merged run under /data/pinpoint/sft/<run>/merged):
PATH=~/.venvs/vllm/bin:$PATH vllm serve <model dir> --served-model-name vlm --port 8765 --dtype bfloat16 \
  --gpu-memory-utilization 0.7 --max-model-len 4096 --limit-mm-per-prompt '{"image":1}' \
  --mm-processor-kwargs '{"max_pixels": 786432}'

# Evaluate on the full eval halves (3,795 queries; greedy + 8 samples at T=0.7):
.venv/bin/python -m geo_search_env.experiment.pivot_diagnostics vlm_sampling --model vlm --prompt sft-retrieval \
  --temperature 0.7 --full-eval

# SFT (LoRA r=32 on all LLM projections; ~4 h for 34.5k examples at batch 8x4), then merge for vLLM:
~/.venvs/sft/bin/python -m geo_search_env.experiment.sft_train train --examples 34523 --run sft-34k-retrieval \
  --data artifacts/sft/sft_retrieval.jsonl
~/.venvs/sft/bin/python -m geo_search_env.experiment.sft_train merge --run sft-34k-retrieval

# GRPO from a merged SFT run (TRL, colocated vLLM; ~45 s/step):
PATH=~/.venvs/grpo/bin:$PATH ~/.venvs/grpo/bin/python -m geo_search_env.experiment.grpo_train train \
  --init sft-34k-retrieval --data artifacts/sft/sft_retrieval.jsonl --run grpo-kl --hours 5
~/.venvs/grpo/bin/python -m geo_search_env.experiment.grpo_train merge --init sft-34k-retrieval --run grpo-kl
```

End-to-end scripts in `scripts/` wait for an idle GPU, serve, evaluate and stop their own server:
`eval_run.sh <run>` (300 subset + MP16 val), `full_eval.sh <run> <prompt>` (full eval halves) and `evidence_test.sh`.
The expected numbers for each model are in `LEARNINGS.md`.
