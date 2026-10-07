# Plan: knowledge warm start, then RL

Status: agreed 2026-10-06. Step 1 done 2026-10-07 (`knowledge-a`); its knowledge did not turn into better choices (lesson 52), so the
step-2 GRPO runs were not started (gate in `OVERNIGHT.md`). Results: `artifacts/overnight/REPORT.md`.

**Question.** Can a model learn geographic knowledge that visual similarity doesn't already capture? Measured on
real geolocation, as success on the photos where visual similarity (the reranker) fails.

**Why this design.** SFT and GRPO choosers reached reranker parity (16.9 / 37.9 vs 16.7 / 38.2 within 1 / 25 km):
GRPO sharpens what the model knows, it doesn't discover (LEARNINGS 1-3), and the base 4B knows little geography
(27-28, 50). So give the model knowledge from data first, then let RL learn to use it.

## Steps

**1. Knowledge warm start (SFT, ~8-10 h).** LoRA SFT of Qwen3.5-4B on photo -> "country > region > city", 250k MP16
photos from buckets 0-98 (`artifacts/knowledge/selected.json`, Overture labels in `artifacts/place_labels/`; user's
bucket exception). Burned-in-GPS filter first. 448 px.

- **Check before RL (~30 min):** on benchmark eval-half photos where the reranker top-1 is wrong at 25 km, does the
  trained model name the right city / region more often than the base 4B? If not, stop here.

**2. RL agent (GRPO, two runs of ~5 h).** Photo + the retrieval candidates (the visual-similarity tool's output) ->
answer, distance reward, MP16 bucket-99 train photos (34.5k), lesson-3 recipe: hard-photo filtering, KL 0.04, T=1.0,
lr 5e-6, ~200 steps.
- **Run A:** initialised from the step-1 knowledge model.
- **Run B (control):** initialised from the base 4B, same recipe.

**3. Measure on the full im2gps3k + yfcc4k eval halves (3,795).**
- Fixes vs breaks: of photos where the reranker is wrong at 25 km, how many the agent gets right; of photos where it is
  right, how many the agent gets wrong.
- Run A minus Run B = what the learned knowledge adds (same RL, same tool).
- Top-1 within 1 / 25 km vs the reranker (16.7 / 38.2); fixes split by readable text yes / no (lesson 4).

| Result | Meaning |
|---|---|
| A fixes clearly more of retrieval's failures than B and beats the reranker | The model learned knowledge visual similarity lacks and RL uses it; add tools (geocoder, place search) |
| A ≈ B | The warm start didn't add usable knowledge at 4B; next is a larger base model |
| Step 1 check fails | Stop before RL: 250k photos of SFT don't teach the 4B place knowledge |

**Limitation (accepted).** A fix where retrieval fails could come from the 4B's own vision rather than knowledge; the
claim is "succeeds where visual similarity fails".

**Risk.** The knowledge model only saw short name answers and may follow the candidate prompt worse than the base;
check step-0 format errors and accuracy before each GRPO run.

## Work

1. Port to this machine (~2 h): `knowledge_data`, `sft_train`, `grpo_train`, `name_score`, the run scripts, and the
   `/data/...` paths stored in artifact JSONs (remap at load). Smoke-test each stage on a few photos.
2. Step 1 (confirm with the user first), check, step 2 runs (confirm first), report.

Artifacts come from `kinghorton42/geo-benchmarks` revision `2c4f566` (the old machine's caches, reproduce
16.7 / 38.2 reranker and 33.4 / 59.0 pool oracle); our rebuilt version-6 caches are in `artifacts/strategy_search_v6`.
