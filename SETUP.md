# Setup

How to rebuild the environments, models, datasets and caches for this repo from scratch. See `README.md` for what
the project is and `LEARNINGS.md` for results.

## 0. Scope

The commands below rebuild the retrieval pipeline and the single-turn SFT/GRPO baselines; the current plan is in
`CLAUDE.md`.

Data rules: train only on Pinpoint-held-out MP16 bucket 99 (except the knowledge SFT, see `CLAUDE.md`), exclude
same-photographer gallery images, benchmark duplicates and burned-in GPS overlays, and never train on benchmarks.

## 1. Hardware and disk

- One NVIDIA GPU with ≥ 32 GB (CUDA 13 wheels). Linux, Python 3.11 and 3.12, [uv](https://github.com/astral-sh/uv).
- ~550 GB of disk, mostly MP16-Pro images (371 GB). Gallery embeddings take ~28 GB, the Pinpoint model.zip 27 GB with
  its unzipped copy, model weights ~20 GB, and each SFT/GRPO run ~9 GB. OSV-5M images (251 GB) are optional.

## 2. Paths the code assumes

Run commands from `~/pinpoint-rl`. Storage defaults are anchored to this checkout by
`src/geo_search_env/paths.py`; `HF_HOME`, `HF_HUB_CACHE` and `PINPOINT_ROOT` can override them.
All project datasets, caches, checkpoints, and training environments live inside this folder.

| Path | Used for | Defined in |
|---|---|---|
| `data/hf` (`HF_HOME`) | Hugging Face cache; Qwen3.5-4B snapshot | `sft_train.BASE_MODEL` |
| `data/hf/datasets/MP16-Pro` | MP16 metadata, tar shards, `tar_index.pkl` | `strategy_search.MP16_CSV`, `paths.MP16_ROOT` |
| `data/pinpoint/{mp16,osv5m}-embed/siglip2-giant-opt-patch16-384` | gallery embeddings | `strategy_search.MP16_EMBED`, `OSV_EMBED` |
| `data/pinpoint/{im2gps3k,yfcc4k}` | benchmark CSVs, images, embeddings | `data/benchmarks.py` (`DATA_ROOT`) |
| `data/pinpoint/geonames` | GeoNames dump (fuzzy geocoding, feature codes) | not read by current code (planned agent tool) |
| `data/pinpoint/sft` | SFT/GRPO checkpoints | `sft_train.RUNS` |
| `$PINPOINT_ROOT` | Pinpoint checkpoints (contrastive retriever, attention reranker) + attention retrieval index | `models/pinpoint.py`, `models/pinpoint_reranker.py` |
| `data/pinpoint/overture` | Overture division polygons (knowledge SFT place labels) | `place_labels.py` |
| `data/pinpoint/commons26` | commons26 benchmark (built by `commons_bench.py`; not on this machine yet) | `data/benchmarks.py`, `commons_bench.py` |
| `.venvs/{vllm,sft,grpo,overture}` | serving, training, and the DuckDB env for place labels (`envs/*.txt`) | `scripts/*.sh`, module usage lines |

Copy `.env.example` to `.env`, fill it in, and load it with `set -a; . ./.env; set +a`.

## 3. Python environments

Four main environments, because training, serving and analysis pin different torch versions, plus `.venvs/overture`
(duckdb + shapely + pyarrow) for place labels. `envs/<name>.txt` lists the exact packages each one had on the old machine
(`uv pip install -r envs/<name>.txt`).

```bash
# (a) Repo env: analysis, caches, dataset building, eval clients (Python 3.11).
uv sync --all-extras

# (b) Serving: vLLM. Put its bin on PATH when serving (vLLM needs its ninja).
uv venv .venvs/vllm --python 3.12
uv pip install --python .venvs/vllm/bin/python vllm==0.30.0

# (c) SFT training.
uv venv .venvs/sft --python 3.11
uv pip install --python .venvs/sft/bin/python torch==2.14.0 torchvision==0.29.0 transformers==5.17.0 trl==1.14.0 \
  peft==0.21.0 accelerate==1.15.0 datasets==5.0.1 flash-linear-attention==0.5.2 pillow scipy pytest
uv pip install --python .venvs/sft/bin/python -e . --no-deps

# (d) GRPO with colocated vLLM rollouts.
uv venv .venvs/grpo --python 3.12
uv pip install --python .venvs/grpo/bin/python vllm==0.30.0 trl==1.14.0 peft==0.21.0 accelerate==1.15.0 \
  datasets==5.0.1 flash-linear-attention==0.5.2 scipy pytest
uv pip install --python .venvs/grpo/bin/python -e . --no-deps
```

Check: `.venvs/sft/bin/python -m pytest -q` (CPU only).

## 4. Models

```bash
export HF_HOME="$PWD/data/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
.venv/bin/hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
.venv/bin/hf download google/siglip2-giant-opt-patch16-384
```

`sft_train.BASE_MODEL` points at that Qwen snapshot; update it if you use another revision.

## 5. Datasets

### MP16-Pro (training pool and retrieval gallery)
Gated on Hugging Face: accept the terms and run `hf auth login` first.

```bash
.venv/bin/hf download Jia-py/MP16-Pro --repo-type dataset --revision e50ce1ef84f157fa80b17563e9bdb0af1d510b5c \
  --local-dir data/hf/datasets/MP16-Pro
```

### OSV-5M (second retrieval gallery)
[`osv5m/osv5m`](https://huggingface.co/datasets/osv5m/osv5m) → `data/hf/datasets/osv5m`. Only needed for planned
evidence acquisition or to rebuild its embeddings. It has no photographer ids, so it is never same-photographer excluded.

### Benchmarks and SigLIP2 embedding caches
The im2gps3k and yfcc4k test sets and the MP16/OSV-5M gallery embeddings are on Hugging Face at
[`kinghorton42/geo-benchmarks`](https://huggingface.co/datasets/kinghorton42/geo-benchmarks):

```bash
REPO=kinghorton42/geo-benchmarks REV=9b4031ca477e64c4eecddaed6fecbdc6e41e5ed8
.venv/bin/hf download $REPO im2gps3k.zip yfcc4k.zip --repo-type dataset --revision $REV --local-dir "$PWD/data/downloads/bench"
unzip -q "$PWD/data/downloads/bench"/im2gps3k.zip -d data/pinpoint && unzip -q "$PWD/data/downloads/bench"/yfcc4k.zip -d data/pinpoint
.venv/bin/python -c 'from huggingface_hub import snapshot_download as d; d("'$REPO'", repo_type="dataset",
  revision="'$REV'", local_dir="data/pinpoint", allow_patterns=["mp16-embed/*", "osv5m-embed/*"],
  ignore_patterns=["*text_*"])'   # ~28 GB; text_* (MP16 text embeddings) are unused
```

This gives `data/pinpoint/{im2gps3k,yfcc4k}` (CSV + images; the `AUTHOR`/`OwnerNSID` columns drive the
same-photographer exclusion) and `data/pinpoint/{mp16,osv5m}-embed/siglip2-giant-opt-patch16-384`.

The benchmark embeddings are not in the repo; build them with `experiment/embed_cache.py` (GPU, a few minutes each):

```bash
.venv/bin/python -m geo_search_env.experiment.embed_cache benchmark im2gps3k   # and yfcc4k
```

All caches are the raw `get_image_features` output of `google/siglip2-giant-opt-patch16-384` (fp16, unnormalized).
`embed_cache gallery {mp16,osv5m}` can rebuild the galleries from the raw images (~11 / ~13 h). Keep its default
`--workers`/`--batch-size`, because gallery row order depends on them and `artifacts/` stores row numbers.

### Pinpoint retrieval model
Not public. We need only the contrastive checkpoint and MP16 retrieval index from its released `model.zip`:

```bash
mkdir -p $PINPOINT_ROOT && cd $PINPOINT_ROOT && unzip -q /path/to/model.zip
```

The index's rows are mapped to our MP16 cache via `_pinpoint_torch.pinpoint_mp16_csv_rows`, so our MP16 cache must
cover every indexed photo. Pinpoint's retriever trained on MP16 photos with `md5(image_id) % 100 < 99`, hence training
on bucket 99 only. Without Pinpoint the photo galleries still work, but the candidate pools cannot be rebuilt.

### GeoNames
```bash
mkdir -p "$PWD/data/pinpoint/geonames" && cd "$PWD/data/pinpoint/geonames"
curl -LO https://download.geonames.org/export/dump/allCountries.zip && unzip allCountries.zip && rm allCountries.zip
curl -LO https://download.geonames.org/export/dump/featureCodes_en.txt
curl -LO https://download.geonames.org/export/dump/countryInfo.txt
```

## 6. Build the caches (in order)

Run from the repo root as `.venv/bin/python -m geo_search_env.experiment.<module> <node>`; each module's docstring
documents its nodes. Output goes to `artifacts/` (git-ignored).

Candidates and the reference (`artifacts/pinpoint_reference`): Pinpoint's attention reranker (`models/pinpoint_reranker.py`).
1. `pinpoint_reference bench`: its pool and top-1 for every benchmark photo (GPU, ~5 min).
2. `pinpoint_reference mp16`: the same for the MP16 query pool (after `sft_data pool`; the report includes the leak check).

Benchmark side (`artifacts/strategy_search`):
1. `strategy_search neighbors`: MP16/OSV photo matches and Pinpoint GPS neighbours per query (GPU; the evidence prompts).
2. `strategy_search region_head`, `strategy_search coarse`: region-classifier head, head and kNN vote settings.

MP16 training side (`artifacts/sft`):
1. `sft_data pool`: bucket-99 query photos (then `pinpoint_reference mp16`).
2. `sft_data overlay`: burned-in GPS filter (needs base Qwen served, see §7).
3. `sft_data dataset --variant sft_retrieval`: the SFT dataset.

Photo sets (`artifacts/photo_sets`): `photo_sets select --tag dev --n 1000` (MP16 dev), `--tag devall --n 100000` (all of it),
`--source bench --tag val --n 1000` and `--source bench --tag full --n 100000`; then `place_labels pool-labels --tag <tag>`
(`.venvs/overture`; needs `place_labels label-all` once for `artifacts/place_labels/mp16.parquet`). Knowledge diagnostics:
`knowledge_check` (lessons 52-54).

## 7. Serving, training and evaluation

```bash
# Serve base Qwen or a merged run (data/pinpoint/sft/<run>/merged).
# Raise the image limit for exemplar-photo evidence tests.
PATH=.venvs/vllm/bin:$PATH vllm serve <model dir> --served-model-name vlm --port 8765 --dtype bfloat16 \
  --gpu-memory-utilization 0.7 --max-model-len 4096 --limit-mm-per-prompt '{"image":1}' \
  --mm-processor-kwargs '{"max_pixels": 786432}'

# Evaluate on the full eval halves:
.venv/bin/python -m geo_search_env.experiment.pivot_diagnostics vlm_sampling --model vlm --prompt sft-retrieval \
  --temperature 0.7 --full-eval

# SFT, then merge for vLLM:
.venvs/sft/bin/python -m geo_search_env.experiment.sft_train train --examples 34523 --run sft-34k-retrieval \
  --data artifacts/sft/sft_retrieval.jsonl
.venvs/sft/bin/python -m geo_search_env.experiment.sft_train merge --run sft-34k-retrieval

# GRPO from a merged SFT run:
PATH=.venvs/grpo/bin:$PATH .venvs/grpo/bin/python -m geo_search_env.experiment.grpo_train train \
  --init sft-34k-retrieval --data artifacts/sft/sft_retrieval.jsonl --run grpo-kl --hours 5
.venvs/grpo/bin/python -m geo_search_env.experiment.grpo_train merge --init sft-34k-retrieval --run grpo-kl
```

`scripts/` has end-to-end wrappers that serve, run and stop their own server: `observe.sh`, `clue_judge.sh`, and the pipeline
(`full_pipeline.sh`, `retrained_pipeline.sh`, `pinpoint_train.sh`). Expected numbers are in `LEARNINGS.md`.
