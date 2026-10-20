# Track R: retrain retrieval (2026-10-20)

One of two tracks run in parallel sessions (the other: `SEARCH_PLAN.md`). Read `LEARNINGS.md` first; lesson N is in
`docs/archive/LESSONS_1-137.md#lN`. Lessons from this track go in `LESSONS.md` under "Retrieval track" as R1, R2, ...

## Why
Retrieval is where the pipeline loses most (dev3k, original protocol, pipeline 47.3% < 25 km; lessons 129-130):
- 26.5 pts: no pool candidate within 25 km. 24.6 of them are photos *without* the photographer's own photo in the pool (59% of dev3k, where the
  pipeline gets 33.7% vs 66.6% with one), concentrated where the gallery is sparse, indoors and in nature. commons26 dev: 49.1 pts.
- 11.8 pts: a right candidate is in the pool but at noise-level reranker weight (rank ~9-29 of ~46), so no area forms on it (lesson 122).
- At 1 km the index has a photo within 1 km of the truth for ~85% of queries; most reranker misses are ranking, not coverage (lesson 134).
Retraining Pinpoint's reranker was our largest single gain (+3, lessons 70, 108), and every head feature built from the *same* embeddings
lands within +-0.5 (lessons 122, 124, 126, 127). The retriever under the reranker has never been retrained. The user accepted that this track
is supervised work on the candidate generator (2026-10-20): it is the search tool the RL agent of track S calls, and raises its ceiling.

## What the retriever is
- `pinpoint_train/src/pinpoint/contrastive_retrieval/` (`lightning_module.py`, `train.py`): image-to-GPS contrastive model over frozen SigLIP2
  embeddings (siglip2-giant-opt-patch16-384, 1536-d; MP16 at `data/pinpoint/mp16-embed/`, OSV-5M at `data/pinpoint/osv5m-embed/`). Shared
  residual-MLP image trunk with per-source adapters (mp16, osv5m); location tower over unit xyz with learned Fourier features (sigmas 1 / 16 /
  256); SigLIP pairwise sigmoid loss per source, weighted. Released `ckpt_best.pt`: 256-d output, 105M params, 50k steps, lr 1e-3, OSV loss
  weight 0.5 and an image-neighbourhood term (weight 0.2, distance-weighted negatives) that **this repo's code lacks** (the release came from a
  newer codebase). Training buckets: md5(image_id) % 100 < 99 (bucket 99, our train / dev3k, was held out).
- Index (`attention_reranker/retrieval_index.py`; released copy under `data/pinpoint/pinpoint-submission/exp/contrastive_retrieval/checkpoints/
  attention_retrieval_index/`, staged to `/opt/dlami/nvme/pinpoint/attention_retrieval_index`, wiped daily): 4,080,926 MP16 rows (bucket 99 not
  in it), per photo the 256-d contrastive image embedding, the location tower's embedding of its GPS, raw SigLIP2 int8, latlon; plus 4.89M OSV
  support embeddings. ~13 GB; the raw SigLIP2 part (6.3 GB) doesn't change with a new retriever.
- Three searches feed the reranker (`models/pinpoint_reranker.py`): contrastive image-image, raw SigLIP2 cosine, image-to-GPS; released pools
  8 / 2 / 2, ours 32 / 8 / 8 (`widen`). The reranker scores the union with retrieval features (similarity / rank per modality).
- To swap in a new retriever: `CONTRASTIVE_CHECKPOINT` is hard-coded in `pinpoint_reranker.py` (make it an env var / argument), rebuild the
  index (`ensure_attention_retrieval_index`), then re-fine-tune the reranker on the new pools (`scripts/pinpoint_train.sh`, setft_own recipe,
  ~65 min, ~27 GB GPU) and rebuild the pipeline (`pinpoint_set extract` / `photo-set --ckpt-tag`, then `scripts/retrained_pipeline.sh`).

## First steps (cheap before expensive)
1. **Recall diagnostics on dev3k** with the released retriever, original protocol (`PINPOINT_KEEP_OWN=1`): per modality and for the union,
   recall@K (K = 8 ... 1,000) of a candidate within 1 / 25 km, split by own-photo-in-pool, gallery density and scene type. Where does the
   nearest right photo rank in each modality? This says which search to improve and how deep the reranker's list must go.
2. **Reproduce, then change**: retrain the retriever with this repo's code to near the released recall (or fine-tune from `ckpt_best.pt`),
   so later changes are measured against a like-for-like baseline. Re-indexing is one forward pass over 4.08M embeddings.
3. Then the ideas that target the misses, chosen by step 1. Candidates, not prescriptions:
   - objectives for photos with no own photo in the gallery (photographer-excluded positives, image-image same-area positives, hard
     negatives from the reranker's confident far mistakes; lesson 38: look-alike-only negatives were a trap);
   - more in-distribution training data: the bucket-99 train split (never dev3k ids), and the 44.2M geotagged YFCC100M photos outside MP16
     (`data/yfcc100m/geo.npz`; images must be fetched and SigLIP2-embedded, ~80% still on Flickr; scale-probe a subset first);
   - a MegaLoc full-index search as a fourth modality (~6 h build; lesson 95 tested MegaLoc only on a SigLIP2 top-200 shortlist).

## Go / no-go
- Intermediate: pool recall and area oracle on dev3k beyond the released retriever (area oracle 61.7, pool oracle 73.1), especially on
  photos without an own photo. A recall gain that the reranker can't convert is not a gain: check reranker top-1 after re-fine-tuning.
- Final: >= 1 pt at 25 km on dev3k end to end beyond the current pipeline (47.3 / 30.8 with `--one-km 0.5`), then commons26 dev.
- Leak guards: dev3k photos and photographers never in retriever or reranker training; YFCC additions drop every im2gps3k / yfcc4k / commons26
  photo and photographer (yfcc4k photos are YFCC photos). Test set only when the user asks. Confirm long runs with the user first.

## Coordination with track S (shared machine; both briefs say the same)
- Work in your own worktree / branch (`retrieval-track`); track S uses `search-track`. Both start from main with this brief merged.
- Write to `artifacts/retrieval/` and new checkpoint / index tags (e.g. `ret_*`); never overwrite shared caches (`artifacts/agent/*`,
  `artifacts/pinpoint_train/setft_own`, the released index). New indexes go on the EBS disk (993 GB free), staged to NVMe only for speed.
- GPU (L40S, 46 GB): track S keeps <= 16 GB (resident index ~10.5 GB + reranker), track R <= 28 GB (a reranker fine-tune needs ~27 GB). vLLM or
  anything larger only after checking `nvidia-smi` that the other track is idle.
- RAM (30 GB, no swap): run long jobs with `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0` by default. A job that needs up to
  20G first checks `systemctl --user list-units --type=scope` and `free -g`, and never overlaps another job above 12G.
- CPU: the reranker fine-tune is CPU-bound (~400 ms / step); expect it to slow track S's jobs while it runs.
- The machine stops at 02:00 UTC daily and wipes NVMe: checkpoint long jobs, re-stage indexes after a restart.
- Lessons only in your own `LESSONS.md` section; touch `LEARNINGS.md` / `CLAUDE.md` only for your own track's lines.
- Hand-off: when a retriever beats the released one end to end, record the checkpoint and index paths in your lesson; track S's search tool
  should take them as parameters so it can switch.
