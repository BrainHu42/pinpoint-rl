# Learnings so far (Sep 2026)

What the base-model diagnostic, SFT, single-turn GRPO and the follow-up headroom checks established, and what they
rule out. The current plan lives in CLAUDE.md ("Research plan"). Numbers are % of
photos within 1 km / 25 km of the truth, with same-photographer gallery images excluded. "Full eval halves" = every
im2gps3k + yfcc4k eval-half query (n = 3,795; SE about 0.6 pts at 1 km, 0.8 at 25 km). "300 subset" = the stratified
study subset (SE about 2 / 3 pts; don't trust gaps under ~4 pts there).

## Headline numbers (full eval halves)

| System | Greedy | Single sample | Best of 8 | Right answer among shown candidates |
|---|---|---|---|---|
| One-step reranker #1 | 16.7 / 38.2 | | | |
| sft-34k (reranker top-10 in prompt) | 15.8 / 37.6 | 11.5 / 33.0 | 22.3 / 48.0 | 30.8 / 53.6 |
| sft-34k-retrieval (all ~17 pooled candidates, retrieval order, evidence; no reranker) | 16.3 / 37.2 | 12.6 / 32.8 | 22.3 / 48.2 | 33.4 / 59.0 |
| grpo-kl step 200 (from sft-34k-retrieval) | 16.9 / 37.9 | 15.9 / 36.8 | 21.7 / 45.1 | 33.4 / 59.0 |

300 subset only: base Qwen3.5-4B greedy 12.0 / 38.0, best-of-8 21.7 / 49.7; Gemini 3.8 Flash with top-10 candidates
27.7 / 55.7 (best-of-8 31.3 / 58.0; oracle over its samples + candidates 40.7 / 65.7).

## Main lessons

1. **Selection among one-shot retrieval results is saturated.** SFT and GRPO both land at reranker parity, while the
   right answer is in the shown list for 33 / 59%. The model and the reranker agree on >90% of photos; disagreements
   roughly cancel (model-only right 2.6 / 3.9%, reranker-only 3.0 / 4.8%).
2. **SFT lifts the 4B model from below the reranker to parity, not past it.** On the 300 subset, greedy rose +4–6 pts
   over base, mostly by learning to follow the reranker (#1 copied 41% → 67%). More data (5k → 34.5k) lowered val loss
   (0.349 → 0.328) but MP16-val accuracy stayed flat. Dropping the reranker (retrieval order + evidence numbers) costs
   nothing and raises the candidate ceiling (+2.6 / +5.4).
3. **GRPO sharpens; it doesn't discover.** It converts best-of-8 headroom into single-sample accuracy (+3–4 pts) and
   shrinks best-of-8. With lr 2e-5, no KL and T=0.7 it collapsed (entropy 0.13 → 0.025, 44% of groups with identical
   rewards, best-of-8 −9 pts at 25 km). lr 5e-6, KL 0.04 and T=1.0 kept entropy at ~0.17 but greedy only moved +0.6.
   Geo-R (AAAI'26) reports the same "vanishing advantages" problem and fixes it with hard-photo filtering.
4. **The LLM's advantage is real but narrow.** On photos with readable text it beats the reranker (<25 km 64.9 vs
   60.9; wins 6.9% vs loses 2.9%). On generic photos (no text, no nameable place; nature, events, interiors) it
   loses (18.3 vs 21.0). The model's off-list guesses are almost always wrong (11 of 270 within 25 km).
5. **"Where to search" is not the bottleneck.** Retrieval restricted to the true region
   jumps to 64.4 / 48.9% within 25 km (im2gps3k / yfcc4k, vs 38.4 / 19.7 unrestricted). The VLM picks the right region
   more often than the region head (62 vs 58%; 46 vs 42%), yet retrieving inside the VLM's region scores *below* the
   VLM's own answer (45.6 vs 48.3; 27.0 vs 30.1), and the oracle over its sampled regions adds only ~6 pts. Whole-image
   retrieval matches overall scene appearance (e.g. "crowd in red" → St. Louis for a Kauffman Stadium photo).

6. **Offline geocoding needs fuzzy matching.** Exact GeoNames lookup of Gemini's confident place names: 55% match at
   all; best match within 1 / 25 km for 27 / 47%, most-populous match 22 / 39%. Misses are name variants.

7. **Rewriting the retrieval query doesn't add candidates the list is missing (go/no-go, 2026-09-29).**
   - Text: SigLIP2 text queries of Gemini's confident place names (645 photos) land within 1 / 25 km for 0.8 / 3.1%
     (best of top-10 clusters 2.5 / 6.2%) vs GeoNames 22 / 39% and whole-image retrieval 33 / 70%. SigLIP's text tower
     knows concepts ("a stadium"), not specific places ("Golden Gate Bridge" → Finland, Java, Russia).
   - Crops (10 fixed crops per eval photo, full eval halves): at equal budget, best of 10 crop top-1s 18.3 / 36.7 vs
     whole-image top-10 24.9 / 45.0; the centre crop alone is worse than the whole image (9.9 / 23.9 vs 12.4 / 27.0).
     Adding the 10 crop guesses to the ~17 shown candidates raises the oracle only 33.4 / 59.0 → 34.8 / 60.6.
   - So the "learning what to search for" plan failed its go/no-go. The missing accuracy is in *choosing* among
     candidates retrieval already returns (oracle 59% vs greedy 38% within 25 km), not in finding new ones.

8. **Per-candidate evidence helps an untrained chooser a little (`experiment/evidence_test.py`, base 4B zero-shot,
   reranker top-10, full eval halves).** Candidates only 10.5 / 33.9; + up to 4 GeoNames landmarks within 1 km
   (60% of candidates have one) 12.1 / 35.2; + one 256 px MP16 exemplar photo per candidate (70% the most similar
   retrieved photo within 1 km, else the nearest photo) 11.7 / 35.2. Both gains are consistent on im2gps3k and yfcc4k.
   Photos cut off-list guesses (11% → 5%) and raise model-only wins (3.1 → 3.9%); reranker-only wins fall 7.4 → 6.4-6.8%.
   Open question: does training (SFT on these prompts) turn this into a large gain? That is the proposed next run.

9. **The selection cap is the chooser's knowledge, not the candidate list.** With the same kind of prompt (photo +
   top-10 candidates, no extra evidence), on the 300 subset: reranker 18.7 / 45.0, SFT 4B 17.7 / 43.0, Gemini 3.8
   Flash 27.7 / 55.7, candidate oracle 32.0 / 59.7. A model that knows what places look like closes ~2/3 of the gap.
   For a 4B model, per-candidate evidence (lesson 8) or tools are the substitute for that knowledge.

10. **Method: define go/no-go against what you already have.** The query-rewriting go criterion ("combined oracle beats
    current greedy by ≥ 8 pts") would have passed trivially, because the existing candidate list already beats greedy by
    21 pts. The right comparison was what new queries add beyond the shown-candidate oracle (< 2 pts). Also use the
    full eval halves for decisions; 300-subset differences flipped sign several times.

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
- `np.load` NpzFile re-reads an array on every key access: wrap in `dict(...)`.
- vLLM eval servers: `--gpu-memory-utilization 0.7` so a small InnerSight job doesn't block startup.
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
- `/data/pinpoint/sft/sft-34k-retrieval` (adapter + merged): best SFT model, starting point for the agent.
- `/data/pinpoint/sft/grpo-kl` (checkpoint-200 + merged): best greedy so far.
- `artifacts/sft/`: MP16 query pool, neighbour/candidate caches, `sft_retrieval.jsonl` (+ `sft.jsonl`), logs.
- `artifacts/strategy_search/`: benchmark caches, all result JSONs, Gemini labels for 1,683 eval photos
  (`llm_advantage_labels.json`; OpenRouter credits ran out before the other ~2,100), crop-query search cache
  (`query_headroom_crops.npz`), evidence-test answers (`evidence_test_vlm_*`).
- `/data/pinpoint/geonames/allCountries.txt` (GeoNames dump; `evidence_test landmarks` rebuilds the landmark index).
- Environments: `.venv` (analysis), `~/.venvs/sft` (SFT), `~/.venvs/grpo` (GRPO with vLLM), `~/.venvs/vllm` (serving).
