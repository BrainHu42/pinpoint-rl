# Handoff (2026-10-06)

For the next person or agent picking this up. Read this, then `CLAUDE.md` (rules, plan, code map) and the summary at the top of `LEARNINGS.md`
(numbered lessons 1-50). All work is on branch `worktree-pivot-query-evidence` (draft PR #3); `main` is far behind. `artifacts/` (caches, results,
place labels; 2.6 GB) is not in git: download it from the public dataset repo into the repo root with
`hf download kinghorton42/geo-benchmarks --repo-type dataset --include "artifacts/*" --local-dir .` (same repo has the benchmark zips and the
SigLIP2 gallery embeddings). Pruned before upload: the geo-adapter training copy (`geo_embed data` rebuilds it) and the superseded v1 labels.

## The problem in one paragraph

Photo geolocation on im2gps3k / yfcc4k (wikimedia as a third test set). A retrieval pipeline (Pinpoint: SigLIP2 photo matching over MP16 + OSV-5M,
a region head, a small reranker) proposes ~17 candidate locations per photo. Baseline to beat is **the reranker's top-1**, not Gemini. On the 3,713
benchmark eval-half photos: top-1 17.1% < 1 km / 38.9% < 25 km / 56.4% < 200 km; the oracle over the whole pool is 34.2 / 60.2 / 80.3. So the
right answer is usually already in the pool, and **the headroom is in choosing among candidates** (~20 pts at 25 km), not in finding new ones.

## What we've learned (details and numbers in LEARNINGS.md)

- **Fine-tuning the 4B to pick a candidate (SFT, GRPO) reaches reranker parity, not better** (1-4). GRPO sharpens; it does not discover.
- **New evidence doesn't help** (5-21, the old "stage 1"): 4B / 27B search queries, offline Wikipedia, an 80M-place name index, transcribed text,
  photo attributes vs map attributes add 1-3 oracle pts and nothing for a learned chooser. They re-encode what image retrieval already knows.
- **The only positive signal: a fine-tuned 4B comparator** (query photo + an exemplar photo of a candidate -> same place?). Best confirmed result
  **+0.9 [+0.3, +1.6] top-1 < 25 km** (comparator-b, 4 exemplars, combiner fitted on dev; 24-29, 39-41). More pairs, pairwise / 25 km variants,
  better combiners and more exemplars all land at +0.7 to +1.3 (40-48). Nothing wikimedia-specific (42).
- **Zero-shot choosers (4B / 9B / 27B), with or without Wikipedia text, are all below the reranker** (27-28, 37). The base 4B knows little geography.
- **Near-misses (top-1 within 25 km but not 1 km; 22% of photos) are the biggest 1 km target**, but no verifier separates the exact spot from
  look-alikes nearby: comparator-d (near-band, 219k pairs) AUC 0.8 on the fixed list but +0.9 at 1 km; keypoint matching (LightGlue) and map search
  near chance among look-alikes (43-49). Matching finds *what* is shown, labels are *where the camera stood*.
- **Process lessons**: measure gains against the reranker top-1 and the pool oracle, not against current greedy; choose scorers on MP16 dev, never by
  looking at benchmark numbers; use the full eval halves (the 300 subset is too noisy); check failed-score counts when using logit judges (40).

## In flight: knowledge SFT go / no-go (LEARNINGS 50)

**Hypothesis.** The missing ingredient for choosing is place knowledge. Teach the 4B to name where a photo is ("country > region > city >
neighbourhood") from many geotagged MP16 photos, then score each candidate's place name by log P(name | photo) minus log P(name | no photo) and
combine it with the reranker rank. (User's decision: this stage may use MP16 buckets 0-98; the chooser stays on bucket 99.)

**Done.**
- Place labels for all 4.1M MP16 photos from Overture division polygons (`place_labels.py label-all`, DuckDB, ~1 h on CPU) ->
  `artifacts/place_labels/mp16.parquet`. Labels were fixed on 2026-10-05: the city is the most populous containing locality (not the smallest),
  and the finer level is kept only if smaller than the city with a different name. Old labels kept as `*_v1*`.
  Depth: 23% reach neighbourhood, 46% city, 27% region, 3% unlabeled.
- Candidate names for dev / val: `artifacts/place_labels/candidates_{dev,val,full}.json` (with the fixed labels).
- Zero-shot control, base 4B: within-photo AUC 0.50-0.53, rank + name scores **-0.7 at 25 km** on dev; right country first on 48.9% of the 538
  multi-country dev photos vs the reranker's 67.5%. At 448 px (the training size): -0.6 dev / -0.4 val. These used the *old* labels; the run
  below re-scores the base model on the new labels.
- Training photo selection (`knowledge_data select`, 47 s): 250k photos from 36,753 cities in 191 countries, blocked photographers (dev,
  benchmarks, wikimedia) and near-duplicates of any evaluated photo removed, at most 400 per city -> `artifacts/knowledge/selected.json`.

**Not done.** The overnight run on 2026-10-05 died at selection (`.venv` lacks pyarrow; fixed in `0eb2237`), so nothing has trained.
`scripts/knowledge_run.sh` runs the rest end to end:
1. burned-in-GPS filter with the base VLM (vLLM, ~30 min), `knowledge_data dataset` -> `artifacts/knowledge/knowledge.jsonl`;
2. LoRA SFT at 448x448 (`sft_train train --batch 8 --accumulation 2 --no-checkpointing`; 10.9 photos/s and ~27 GB on the 5090, so ~6.4 h for 250k);
3. merge, then `scripts/name_scores.sh` on dev and val for the trained model **and** the base model at 448 px (the control).

```bash
setsid nohup bash scripts/knowledge_run.sh > artifacts/query_evidence/logs/knowledge_run.log 2>&1 &
```
On a bigger GPU, raise the batch / `--gpu-memory-utilization` in the script and consider more photos (`PHOTOS=...`; 3.58M are eligible). It reruns
selection first (deterministic, seed 0).

**How to read the result.** Primary: change in top-1 < 25 km vs the reranker top-1 from the CV combiner (rank + name scores), dev then val;
`name_score.py report` prints it. Secondary: within-photo AUC per level and right-country-first rate on multi-country photos, trained vs base 4B.
The usual bar in this project has been **+2 pts at 25 km** over the reranker top-1 (it would be the first lever to clear it). A large AUC gain with
no top-1 gain means knowledge exists but the combination is the problem (try adding it to the comparator combiner); no AUC gain means 250k photos
of SFT doesn't teach the 4B usable place knowledge.

## What to do next

1. Run the knowledge SFT above and report the go / no-go numbers to the user before anything else.
2. If it clears the bar: combine name scores with comparator-b in one combiner, confirm on the full benchmark eval halves and wikimedia, then try
   more data / a 9B LoRA. This also gives the RL agent (the project's framing: decide which tools to call and when to stop) a tool with real signal.
3. If it fails: the 4B choosing line is largely exhausted (LEARNINGS directions 1-3 are tested and closed). Bring the options to the user rather
   than starting new small experiments; candidates are a bigger model's knowledge (9B / 27B LoRA, the 5090 can only QLoRA the 27B slowly), or
   revisiting the framing. Don't redo anything in LEARNINGS "Dead ends" / "Not promising".
4. Final numbers: whatever method wins, on all benchmark eval-half photos and wikimedia (loaded by `data/benchmarks.py`, candidates from
   `wikimedia_eval.py`; so far only the baseline, oracle and comparator were run on it, lesson 42).

## Practical notes

- Environments: `.venv` (analysis; **no pyarrow**), `~/.venvs/sft` (training, also reads parquet), `~/.venvs/vllm` (serving), `~/.venvs/grpo`,
  and small tool envs `geo` (rasterio), `overture` (duckdb, place labels), `match` (LightGlue). Exact package lists in `envs/`.
- Kept models: `/data/pinpoint/sft/<run>/adapter` (comparator-a/b/c/d, pairwise-a, comparator-25km, sft-34k-retrieval, grpo-kl). Merged weights
  are re-created with `sft_train merge --run <run>`.
- The 27B is served from a GGUF with llama.cpp (`scripts/serve27b.sh`, set `LLAMA_CPP`).
- The old machine's GPU was shared with other projects: scripts wait for free GPU memory and stop only their own PIDs. Keep that discipline if the
  new machine is shared.
- User preferences: confirm plans before long runs; report concisely with numbers; no Gemini anywhere in experiments; few images per prompt for
  the 4B; tie experiments to the research idea rather than small tweaks.
