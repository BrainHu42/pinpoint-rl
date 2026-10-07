# Learnings so far (Sep 2026)

What the base-model diagnostic, SFT, single-turn GRPO and the follow-up headroom checks established, and what they
rule out. The current direction is `AGENT_DIRECTION.md` (superseded plans in `archive/`); the numbered lessons with full numbers are in `LESSONS.md`
(lesson numbers cited anywhere, e.g. "LEARNINGS 50", refer to it). Numbers are % of
photos within 1 km / 25 km of the truth, with same-photographer gallery images excluded. "Full eval halves" = every
im2gps3k + yfcc4k eval-half query (n = 3,795; SE about 0.6 pts at 1 km, 0.8 at 25 km). "300 subset" = the stratified
study subset (SE about 2 / 3 pts; don't trust gaps under ~4 pts there).

## Summary (read this first; numbered lessons in `LESSONS.md`)

**Reference changed (2026-10-07, lesson 54).** "The reranker" is now Pinpoint's attention reranker (its released final model, ported
exactly): top-1 **17.5 / 37.2** on the full eval halves (MP16 dev 16.0 / 33.9), and its pool (<= 12 candidates, oracle 30.1 / 54.7) is the
pool we choose from. In lessons 1-53 and the numbers below, "the reranker" is our earlier one-step reranker (an MLP over 17 hand-made
features, top-1 16.7 / 38.2) over our ~17-candidate pool (oracle 33.4 / 59.0); gains quoted against it need re-measuring against Pinpoint.

**Where we are (2026-10-04).** Against Pinpoint's reranker top-1, the best confirmed gain is **+0.9 points** [+0.3, +1.6] top-1 within 25 km,
from a fine-tuned 4B comparator (query photo + exemplar photo of a candidate -> same place?) combined with the reranker's rank (lessons 24, 29, 40, 41).
On the 3,713 im2gps3k + yfcc4k eval-half photos (placeholders dropped), nothing fitted on them:

| | < 1 km | < 25 km | < 200 km |
|---|---|---|---|
| Reranker top-1 (baseline) | 17.1% | 38.9% | 56.4% |
| + comparator-b, 4 exemplars, combiner trained on dev | 17.2% | 39.9% | 56.9% |
| Oracle over the reranker's top 8 | 30.1% | 52.5% | 70.1% |
| Oracle over the whole ~17-candidate pool | 34.2% | 60.2% | 80.3% |

The headroom is in choosing (13.5 pts at 25 km to the top-8 oracle). How much a model could recover is open: a rough by-eye look at 30 photos (one reader, lesson 35)
put it near 3 pts, but Gemini closed ~2/3 of the gap on the 300 subset (lesson 9) and our open models, zero-shot, close none of it. Place knowledge is the likely
lever; we are testing it on the 4B first and scaling up only on a promising signal.
Every lever tried lands between -0.5 and +1 point. Wikimedia (lesson 42): reranker top-1 25.0% within 25 km, top-8 oracle 38.9%, comparator +0.4 to +0.8 (balanced subset): no wikimedia-specific gain.
Near-misses (21.9% of benchmark photos, top-1 1-25 km off; lessons 43-46): 1 km oracle over local candidates 30.1% vs 17.1% today; a comparator trained for the band
reaches AUC 0.8 but the combined top-1 gain at 1 km is +0.8 [+0.2, +1.3] on the benchmarks, 4x the band data (lesson 45) gives the same (+0.9), and better
combiners (incumbent-relative features, a switching margin; lesson 46) give the same (+0.7). The single-exemplar visual signal is the limit, not data or calibration.

**One line per lesson.**
- 1-4 (SFT / GRPO): the 4B reaches reranker parity, not better; GRPO sharpens but doesn't discover; the LLM wins only on photos with readable text.
- 5-10: "where to search" is not the bottleneck; GeoNames needs fuzzy matching; rewriting retrieval queries adds < 2 oracle pts; per-candidate evidence helps an
  untrained chooser a little; go/no-go must be measured beyond what we already have (the shown-candidate oracle), on the full eval halves.
- 11-19 (stage 1, new evidence): 4B / 27B search queries, geotagged Wikipedia, transcribed text, an 80M-place name index add 1-3 oracle pts over the pool and
  nothing for a learned chooser; the evidence is right on the photos retrieval is already right on (17, 18, 19).
- 20: train / dev / val audit is sound; Flickr "no longer available" placeholders (flagged) and near-duplicates make absolute numbers ~1.5 pts optimistic.
- 21: the VLM's photo attributes are accurate but redundant with what retrieval encodes.
- 22-23: a zero-shot exemplar judge separates the true place from far away, not from its neighbours; its skill is mostly exemplar similarity.
- 24-26, 29: the fine-tuned comparator is the only positive: +0.7 [+0.3, +1.2] on all benchmark photos; pairwise and 25 km variants are no better.
- 27-28: bigger zero-shot choosers (9B, 27B) and nearby Wikipedia text stay below the reranker.
- 30: the gallery isn't thin; reranking deeper than the pool (50 clusters) is worse (-0.8).
- 31-32: a region-diversified pool adds ~2 oracle pts at matched budget; the region head is the bottleneck and a bigger one barely helps.
- 33-34: the models' own answers and nearby place categories add nothing beyond rank + comparator.
- 35: by eye, of 30 choosing photos 9 are near-misses at the 25 km line, 10 have nothing place-specific, 6 show a specific place, 4 need knowledge, 1 is label noise.
- 36-37: coarse thresholds (200 / 750 / 2500 km) have as much headroom and the same null results; the 4B's candidate-free guess is far below retrieval.
- 38: a geo-aware adapter on SigLIP2 adds ~+1 pt candidate recall over the region prior; Pinpoint's own retriever has worse top-10 recall than raw + prior.
- 39-41: several exemplars per candidate help ~+0.3-0.6 without retraining; comparator-b (3x pairs) adds ~+0.3 more; full benchmark test +0.9 (the rule dev
  picks is null, +0.4; the post-hoc best-of-4 rule +1.7 is a hypothesis, not a result).
- 42: wikimedia is harder (top-1 25.0% < 25 km) with the same choosing headroom; the comparator gives +0.4 to +0.8 there, nothing wikimedia-specific.
- 43: near-misses are the largest 1 km target; 93% have a gallery photo within 1 km of the truth, 73% among cached neighbours, 59% among our local candidates.
- 44-45: a near-band comparator (c: 55k rows, d: 219k rows) lifts within-photo AUC 0.735 -> ~0.80 but top-1 < 1 km only +0.8 / +0.9; 4x data adds nothing.
- 46: combiner variants (relative-to-top-1 features, switching margin) chosen by CV on held-out train photos: +0.7; fixes and breaks move together.
- 47-49 (map search pivot): searching within 25 km of the top-1 reaches 70-82% of near-misses within 100-300 photos (fixed list 56%), but SigLIP2 (AUC
  0.54), keypoint matching (0.52; it finds what is shown, not where the camera stood) and the comparator (0.58) can't separate the right place from its
  look-alikes; 4 exemplars on the fixed list: AUC 0.84, +0.9 at 1 km. The old directions 1-3 (`archive/directions-2026-10-05.md`) are tested and closed.
- 51: concept-query similarity around each candidate is real alone but adds +0.3 / +0.0 to the reranker (lesson 15's open variant).
- 52: knowledge SFT (250k photos) names the right country / region +7.7 / +4.4 pts more often where the reranker fails, but as a chooser adds
  -0.2 / -0.1; true-region knowledge would add +10.8 at 25 km, yet the model names the right region on only 10.9% of those photos. GRPO not run.
- 53: not a coverage gap: at equal training photos per region the model names the region 55% on reranker-right photos, 6% on
  the fixable ones. Those photos don't look like their region; next is non-appearance clues via tools (`AGENT_DIRECTION.md`).
- 54: Pinpoint's attention reranker, ported exactly, is the reference: 17.5 / 37.2 (one-step reranker 16.7 / 38.2), pool oracle 30.1 / 54.7
  with <= 12 candidates; 371 region-fixable target photos (+9.5 at 25 km), the knowledge SFT names their region 13.2%.

**Next.** `AGENT_DIRECTION.md` (agreed 2026-10-07): an agent that geolocates like a professional, with constraint clues (text, rules,
feature co-occurrence map queries) through tools and retrieval as the candidate generator. Superseded plans (knowledge SFT + GRPO, the
2026-10-05 directions) are in `archive/`.

**Traps worth remembering.** Choose scorers on dev, not by looking at val or the benchmarks (41). A judge's Yes/No logit level can drift in training; always
check the failed-score count and restrict output to the answer tokens (40). Compare against the reranker top-1, not current greedy; the pool oracle is the
baseline for new evidence (10, 13).

## Headline numbers (full eval halves)

| System | Greedy | Single sample | Best of 8 | Right answer among shown candidates |
|---|---|---|---|---|
| Pinpoint attention reranker #1 (the reference since lesson 54) | 17.5 / 37.2 | | | 30.1 / 54.7 (its pool) |
| One-step reranker #1 (the reference in lessons 1-53) | 16.7 / 38.2 | | | |
| sft-34k (reranker top-10 in prompt) | 15.8 / 37.6 | 11.5 / 33.0 | 22.3 / 48.0 | 30.8 / 53.6 |
| sft-34k-retrieval (all ~17 pooled candidates, retrieval order, evidence; no reranker) | 16.3 / 37.2 | 12.6 / 32.8 | 22.3 / 48.2 | 33.4 / 59.0 |
| grpo-kl step 200 (from sft-34k-retrieval) | 16.9 / 37.9 | 15.9 / 36.8 | 21.7 / 45.1 | 33.4 / 59.0 |

300 subset only: base Qwen3.5-4B greedy 12.0 / 38.0, best-of-8 21.7 / 49.7; Gemini 3.8 Flash with top-10 candidates
27.7 / 55.7 (best-of-8 31.3 / 58.0; oracle over its samples + candidates 40.7 / 65.7).

## Data and leakage rules we established
- Pinpoint's retriever trained on MP16 md5(image_id) % 100 < 99; its photos get inflated candidates (Pinpoint top-1
  <25 km 40.6% vs 30.4% held out). Train only on the bucket-99 pool (38k; 34.5k train / 3.8k val split by
  photographer); its candidate quality matches yfcc4k eval.
- yfcc4k shares photographers with MP16: always exclude same-photographer gallery rows (`tests/test_author_exclusion.py`).
- Burned-in GPS overlays: base-Qwen P(yes) ≥ 0.64 drops 8 photos of the pool.
- Reserve data: yfcc26k (~19.3k usable after photographer exclusion; lat/lon only).

## Engineering facts
- HF processors ignore `chat_template_kwargs`; pass `enable_thinking=False` directly (TRL GRPO passes
  `chat_template_kwargs` straight through, which works). Prompt suffix must be `assistant\n<think>\n\n</think>\n\n`.
- Merged LoRA checkpoints load in vLLM 0.30 (MTP weights dropped; fine without MTP).
- TRL SFT `chunked_nll` computes logits only at labelled positions (LoRA must not wrap lm_head).
- Keep gradient checkpointing on: batch 4 without it OOMs on ~1,250-token prompts.
- GRPO step (~45 s, 8 photos × 8 samples) is dominated by trainer passes over 64 copies of a ~1,200-token prompt
  (old-policy forward for vLLM importance sampling, reference forward for KL, forward+backward); vLLM generation is
  only 4.5 s. Prompt length and duplicated prompt compute are the cost, not generation.
- TRL colocated vLLM needs `mm_processor_kwargs={"max_pixels": 786432}` patched in (`grpo_train.py`), or image-token
  counts disagree with the trainer. Generating 64 answers at once with HF generate OOMs in the prefill.
- `np.load` NpzFile re-reads an array on every key access: wrap in `dict(...)` (`strategy_search coarse` hung 45+ min
  without it; 3.5 min with it).
- vLLM 0.30's Triton build needs Python 3.12 headers; without `python3.12-dev`, `uv python install 3.12` and set
  `C_INCLUDE_PATH` to its `include/python3.12`. When sharing the GPU at ~0.3 utilization, lower `--max-num-seqs`
  (Mamba cache blocks limit it), and retry if a neighbour's memory changes during profiling.
- This machine has 30 GB RAM and no swap. `MP16Images` used to unpickle the 4.1M-entry tar index (~5.4 GB per process);
  four dataloader workers ran the machine out of memory and it had to be rebooted (2026-10-06). It now memory-maps a
  compact index (`metadata/tar_index_compact`, ~150 MB shared). Run long jobs in a capped cgroup
  (`systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0`, as `scripts/knowledge_run.sh` does).
- Knowledge SFT on the L40S: ~20 photos/s at batch 8 x 2 with LoRA dropout 0, the language model `torch.compile`d and
  sequences padded to 240 (`--compile --pad-to 240`); 15.5 photos/s without. vLLM here needs uv's Python 3.12 headers
  on `C_INCLUDE_PATH` and `VLLM_USE_FLASHINFER_SAMPLER=0` (no CUDA toolkit for its JIT).
- Temperature: T=0.7 for SFT-model sampling; resolution caps below 786k px change nothing within noise.

## Dead ends (don't redo)
- Verifying candidates by image matching (SigLIP, SIFT/RANSAC) or zero-shot VLM judges (Gemma).
- A region choice learned from retrieval evidence; VLM-chosen region + region-restricted retrieval top-1.
- Zero-shot Gemma 4; Qwen3.5 thinking mode; llama.cpp for rollouts.
- More SFT data or temperature tuning to raise best-of-8 (flat at ~21–23 / 48–55 across all settings).
- GRPO without KL/entropy control; longer runs of the same single-turn GRPO setup (flat after ~25 steps).
- Query rewriting for retrieval: SigLIP2 text queries of place names; fixed image crops (see lesson 7).
- Evidence numbers on the reranker's top-10 (superseded by the no-reranker prompt before it finished training).
- A "LLM picks region → retrieve inside it" pipeline (lesson 5).

## Kept artifacts
- None of the old machine's models or caches were transferred (2026-10-06). On this machine: `artifacts/strategy_search/`
  (benchmark caches, rebuilt with the released Pinpoint index; reproduced the one-step reranker 16.8 / 38.5, oracle 33.6 / 58.9),
  `artifacts/pinpoint_reference/` (Pinpoint's pools), `artifacts/photo_sets/` and `data/pinpoint/geonames/allCountries.txt`.
  Everything else is rebuilt per `SETUP.md`.
