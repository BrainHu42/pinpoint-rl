# Stage-1 research plan (archived 2026-10-06)

The research-plan section of CLAUDE.md as of 2026-10-05, kept for its definitions (oracle accuracy, evidence informativeness,
the dev / val protocol). Stage 1 closed with LEARNINGS 11-21; the current state is in HANDOFF.md.

**Stage 1 (now): acquire new evidence.** Can Qwen3.5-4B write search queries that retrieve evidence beyond what we
already have (whole-image retrieval, the reranker's candidates) that contains the answer? **Metric: oracle accuracy**
= % of photos where at least one location stage 2 would see (pooled candidates + retrieved evidence coordinates) is
within 1 / 25 / 200 km (street / city / region) of the truth. It is the ceiling for any stage-2 chooser, with no
chooser in the loop. The baseline is the whole candidate pool the pipeline finds (~17 per photo), not the reranker's
top-10; the gain over it is the stage-1 result. Report it next to the reranker top-1 / top-10 and an extra-whole-image-
retrieval control at matched budget (`stage1_eval.py`), since the oracle only grows with more results. **Second metric
(user's point): evidence informativeness**, because evidence can help choose among candidates without adding one:
support of a candidate = results within 25 km, its rate on correct vs wrong candidates, within-photo AUC, and top-1
of -rank + w * support (w fitted on dev), against the reranker top-1. Report both axes. Backends: SigLIP2 photo search, offline geotagged Wikipedia (`wiki_backend.py`), later live APIs.
**Stage 2 (later): consume the evidence and decide between candidates.** Out of scope until stage 1 works. The first
attempt (LEARNINGS 11: 4B, query photo + six evidence photos in one prompt) failed and is not a fair test: too many
images for a 4B model (see the small-VLM-prompts memory).
- Protocol (user's decision, 2026-10-02): **develop on the MP16 dev set** (1,000 held-out MP16 val photos; tag `dev`,
  `query_evidence select`) and **validate on 1,000 photos from the im2gps3k / yfcc4k eval halves** (500 each, fixed
  seed; tag `val`, `query_evidence select --source bench --tag val`), not the full 3,795, to keep runs manageable.
  Validation reference (reranker top-1 / shown top-10 oracle, % <1 km / <25 km): 18.4 / 42.8 and 34.1 / 59.0
  (im2gps3k 20.4 / 50.8 and 38.8 / 64.2; yfcc4k 16.4 / 34.8 and 29.4 / 53.8). The 50/50 mix is not comparable to the
  pooled 3,795 numbers (yfcc4k is 61% of those). Final test: wikimedia once its loader exists (still to add).
- Long-term goal: an RL agent that learns what to search for, how to interpret new evidence, and when to stop. The
  final answer may be an initial candidate, a retrieved image's location, or any lat/lon. Stage 1 reward = the oracle
  accuracy gain from the retrieved coordinates, computed from ground truth.
- Where things stand (2026-10-03; LEARNINGS 11-29, always against the reranker top-1):
  - Stage 1 (acquiring evidence) is closed for now: search queries from the 4B / 27B (SigLIP2, Wikipedia, an 80M-place name
    index), transcribed text and photo attributes against offline map attributes all add 1-3 oracle points over the
    candidate pool and nothing for a learned chooser, because they re-encode what image retrieval already knows
    (LEARNINGS 11-21). A third of the headroom (pool oracle 64.5% vs reranker top-1 43.5% <25 km on val) is in choosing.
  - Stage 2 (consuming evidence): the one signal with real discrimination is comparing the query photo with an exemplar
    photo of each candidate. A fine-tuned 4B comparator (query + exemplar -> same place?, LoRA, 48k pairs from MP16 train
    photos' top-8 candidates, `comparator_*.py`) lifts top-1 <25 km by +0.7 [+0.3, +1.2] on all 3,713 benchmark eval-half
    photos (im2gps3k +1.4, yfcc4k +0.3), combiner weight fitted on MP16 dev only (LEARNINGS 22-26, 29). Pairwise and 25 km
    variants, zero-shot judges, Wikipedia text per candidate, and bigger zero-shot choosers (4B / 9B / 27B, all below the
    reranker) do not beat it (LEARNINGS 25-28).
  - Since then (2026-10-05, LEARNINGS 38-50): more exemplars, 3-4x more comparator pairs, a near-miss (1-25 km) comparator, better combiners,
    keypoint matching and map search all stay at +0.7 to +1.3 or below. The open experiment is knowledge SFT (LEARNINGS 50; see `HANDOFF.md`).
    The query design is in `archive/QUERY_EVIDENCE_PLAN.md` (superseded).
- Supersedes the earlier plan (per-candidate evidence SFT: exemplar photos + GeoNames landmarks), which was never run.
- Go/no-go rule learned the hard way: measure what a change adds *beyond what we already have* (the reranker top-1
  and the shown-candidate oracle), not against current greedy.
