# Learnings (rewritten 2026-10-15)

What is true now, in one page. Numbered lessons, one entry each: `LESSONS.md` (cited as "lesson N"; full original text at git tag
`lessons-full-2026-10-15`). Plan: `EXPERT_PLAN.md`. Numbers are % of photos within 1 / 25 km of the truth unless stated.

## Conventions
- **Reference**: Pinpoint's attention reranker (released model, exact port; lesson 54). Lessons 1-53 used our retired one-step reranker.
- **Primary benchmark** (user, 2026-10-15): im2gps3k + yfcc4k eval halves (3,795 photos), the original Pinpoint protocol (query photographer's
  photos kept in the index), a table per benchmark x 1 / 25 / 200 / 750 / 2500 km vs the released Pinpoint. Photographer-excluded numbers
  (our training rule so far) are secondary. im2gps3k is the same under both (no shared photographers); yfcc4k is not (same YFCC100M source).
- **Splits**: train = MP16 bucket 99 minus dev (35,359; the only photos Pinpoint's encoder never saw, lesson 75); dev3k (3,000; paired SE
  ~0.26 pts, so a 1-pt gain resolves); test = the eval halves, touched only when the user asks. Decide on dev3k / train CV.
- **Go/no-go**: measure what a change adds beyond the current best pipeline and the pool oracle, not beyond a weaker baseline (lesson 10);
  target >= 1 pt at 25 km. Small subsets are for plumbing.
- **No paid LLMs** (user, 2026-10-15): local open models and our own models only. No distillation from Gemini / Claude; ground-truth labels only.

## Where we are
**Best pipeline (lesson 94, `action_agent.py`)**: Pinpoint's reranker fine-tuned on its own training data with wide pools and photographer
exclusion (`setft_a`, lesson 70) -> candidate areas (agreement clusters + clue places) -> an answer head over area tokens, with costly actions
(zoom = restricted Pinpoint search in an area, read = 4B clue reading + text search, recognise = MegaLoc same-place similarity) chosen by a
REINFORCE policy. dev3k 39.7% < 25 km (Pinpoint 33.4). The policy matches "take every action"; its value is cost (lesson 98).

Test set (lesson 98), RL lambda 0:

| Protocol | Set | Released Pinpoint | Ours | Change at 25 km |
|---|---|---|---|---|
| original (kept, primary) | both | 28.9 / 46.4 / 60.7 / 75.5 / 86.7 | 25.5 / 48.2 / 61.5 / 75.2 / 86.7 | +1.8 [+0.7, +2.9] |
| | im2gps3k | 21.8 / 48.9 / 64.6 / 79.6 / 89.8 | 22.8 / 53.6 / 67.4 / 79.9 / 90.8 | +4.7 [+3.1, +6.4] |
| | yfcc4k | 33.5 / 44.8 / 58.2 / 73.0 / 84.8 | 27.3 / 44.7 / 57.7 / 72.1 / 84.1 | -0.1 |
| photographer excluded | both | 17.5 / 37.2 / 53.9 / 71.4 / 84.6 | 19.1 / 43.5 / 58.4 / 73.2 / 85.5 | +6.2 [+5.2, +7.3] |

Under the primary protocol we lose at 1 km on yfcc4k (-6.2): every model was trained not to use the photographer's own photos, and zooming
moves answers off them. Retraining the chain without the exclusion is pending (~4-5 h machine time; must drop the query photo itself, since
MP16 queries are in the index; lesson 74 found the exclusion was the fine-tune's key ingredient, so re-test that first).
Lesson 108 did: under the original protocol setft_a is below released on dev3k (-2.3 / -1.0 at 1 / 25 km; it skips the photographer's own
near-identical photos), and the same fine-tune without exclusion (`setft_own`) is +0.9 / +2.0 over released. Pipeline retrain in progress.

## What moved the number (all retrieval-side)
| Change | Gain at 25 km (train CV / dev3k) | Lesson |
|---|---|---|
| Agreement among Pinpoint's candidates (score mass within 25 km) | +1.0 / +2.0 | 68 |
| Areas as hypotheses, Pinpoint restricted to each, a learned comparator | +2.9 / +3.8 | 69 |
| Fine-tuning Pinpoint's reranker (wide pools, photographer exclusion, set head) | +3.1 / +3.3 | 70, 74 |
| Area model / answer head on the fine-tuned reranker | +4.9 / +5.4 | 71 |
| Clue reading (4B + text search + fine-tuned clue judge) | +0.5 / +1.0 | 66, 76 |
| MegaLoc same-place similarity per area | +0.8-1.0 / +0.5-1.1 | 93 |

## Why almost everything else adds nothing
1. **Retrieval's wrong candidates look like the photo.** They were retrieved for looking alike, so any visual or descriptive check agrees with
   them as much as with the right one: attributes (21), map features (34, 77), crops (89), round trips (90), fetched references at global or
   patch level (99-101), pair discriminants (103). Verification signals plateau at AUC ~0.6-0.74 against right-vs-wrong areas.
2. **Evidence and knowledge sit where retrieval already succeeds.** Text, names, landmarks, the 4B's and the knowledge SFT's place knowledge
   are right on the photos Pinpoint gets right and silent or wrong on its failures (17-19, 52-53, 65, 78-79, 92).
3. **About half the photos carry nothing place-specific** (food, pets, interiors, concerts; 47% of dev3k, lesson 67). Of the rest, most
   mistakes are near-misses (another scene of the same kind of place 25-100 km away). The right area's own photos are "same kind of scene
   nearby", not same-spot matches (90, 91, 95, 97).
4. **Open models can't propose or verify beyond retrieval.** The 4B / 9B / 27B name too few right places (59, 82, 88), are confidently wrong
   where they disagree with Pinpoint (84), and as A/B readers are at chance and badly calibrated (99, 106). RL on their outputs learns silence
   (87, 96). Retrieval proposes regions far better (84: best of 16 +19 vs +6).
5. **The ceiling is choosing, not finding**: some Pinpoint area is right for ~51% of dev3k, the head picks ~40%; restricted to the true region
   Pinpoint reaches 61% (64). Refinement to the next distance bucket has +11-17 oracle points but no signal (91).

## The one signal beyond the head, and why it's out of reach
Sonnet told the two candidate areas' town names (not shown photos) and following it only when it is confident: +0.94 pts at 25 km on dev3k
(8 fixes / 3 breaks; lessons 104-105). Its fixes are recognitions of events, venues, signs and landmarks retrieval can't use, and its
confidence is calibrated. Paid LLMs are ruled out, and the local 4B and Qwen3.8-27B are at chance with uncalibrated confidence on the same
pairs (lesson 106). Open: whether a local model can be trained (ground-truth reward on decisive pairs) to that calibration.

## Method rules learned the hard way
- Choose scorers, thresholds and combiners on dev / train, never by looking at the test set (41, 66). Pre-declare the threshold (105 used 104's).
- Report fixes and breaks, not only accuracy; a switch needs > 50% precision at the base rate of mistakes (79: AUC 0.88 wasn't enough at 1:10).
- Judges: restrict output to the answer tokens and check the failed count (40); test both A/B orders for position bias (99, 106).
- Best-of-N ceilings grow with N; they are upper bounds, not forecasts (83-84).
- Photos Pinpoint's encoder trained on are easier than evaluation: train only on bucket 99 (54, 75). Keep near-duplicates in mind (~4%, 20).

## Data and leakage
- Bucket 99 = MP16 md5(image_id) % 100 == 99, never seen by Pinpoint's retriever; dev3k and train come from it (`artifacts/agent/splits.json`).
- The `val` photo set is 1,000 test photos (never use); burned-in GPS photos are filtered out of train and dev.
- Fetched references (YFCC / Flickr) must drop the query photo and near-duplicates (yfcc4k photos are YFCC photos) and, by default, the
  photographer. Text-search clue evidence always drops the photo and its photographer.

## Engineering facts
- RAM 30 GB, no swap: long jobs in `systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0`. `np.load` NpzFile re-reads an array on
  every key access: load each array once. `MP16Images` memory-maps a compact tar index (the pickled one OOMed the machine).
- Pinpoint fine-tune: ~400 ms / step, CPU-bound in pinpoint-main's pool union; 6,000 steps ~65 min; validation flat from ~4,500 (74).
  Training against the shipped index needs the "mp16" source alias or leave-one-out silently breaks (70).
- vLLM 0.30: Python 3.12 headers on `C_INCLUDE_PATH`, `VLLM_USE_FLASHINFER_SAMPLER=0`, no MTP; serve with `scripts/clue_judge.sh`
  (`RUN="<module> <cmd>"`). Force JSON with `response_format` json_schema (the 4B ignores "JSON only"). Thinking off via
  `chat_template_kwargs` in requests; HF processors need `enable_thinking=False` directly.
- GRPO (TRL, colocated vLLM): patch `mm_processor_kwargs` max_pixels; steps are dominated by trainer passes over duplicated prompts.
- The machine stops daily at 02:00 UTC and the NVMe drive is wiped (`text_search stage`, `tools/maps` stage after a restart).
- MegaLoc: DINOv2 ViT-B/14 + SALAD, output linearly projected to 8,448-d (chunks of it are not clusters); ~190 photos / s at 322 px.

## Assumptions questioned (2026-10-15): reopened or under-tested
- **"Only bucket 99 is in-distribution" (75) is false**: 44.2M geotagged non-video YFCC100M photos are not in MP16 (all 4.12M MP16 ids are
  in YFCC), so Pinpoint's encoder never trained on them. Answer-head curve +0.5-0.8 per doubling (75). Caveats: MP16 filtered them out for a
  reason (quality, per-user caps), images must be fetched (~90% alive), test photos and photographers must be excluded.
- **"1 km refinement has no signal" (91) predates MegaLoc (93)**: tested in 107, a learned local chooser with MegaLoc gets +0.6-0.8 at
  1 km on dev3k (vs +0.1 without), 25 km unchanged; real but below the 1-pt bar.
  Fetched references also did best at 1 km (99: 78 vs 67%; 101: AUC 0.885 vs margin 0.857, n = 77).
- **"RL only buys cost" (94, 98) is about this action set** (cheap, independent, fixed reveals), not RL; outcome-dependent sequential actions
  are untested. **"RL on the 4B learns silence" (87, 96)** came from rare-reward free-text pilots; a dense binary task (named A/B, 104-106) is untried.
- **"Open models can't recognise" (59, 88, 106)**: only Qwen 4B / 9B / 27B, thinking off; other open families, geo-tuned open VLMs, the 27B with
  thinking, and ~70B at 4 bits (fits the L40S) are untested. Knowledge SFT (52) was one 1-epoch run with loss still falling.
- **"MegaLoc can't propose" (95)** was tested on a SigLIP2 top-200 shortlist only; a full-index MegaLoc search (~6 h) was never built.
- **Ceilings from one reader by eye on small samples** (67: 47% "nothing place-specific" on 120 photos; 35: ~20% recoverable on 30) should not
  cap ambition: Sonnet fixed photos such passes call generic (a white rhino -> San Diego Safari Park, a fire-lookout shirt -> Castaic).
- **Sub-threshold gains were discarded one by one** (crops +0.4, OSM names +0.33, region hypotheses +0.3 on dev3k); never stacked.
- Untested constraints: skyline vs elevation model (mountain photos); sun / shadow with capture date (metadata, a protocol question).

## Dead ends (don't redo without a new idea)
- Choosing with the 4B from a candidate list, SFT or GRPO (1-3); query rewriting and search-query evidence (7, 11-17).
- Exemplar comparators of every kind (22-26, 39-49): saturate near +1 pt.
- Region heads, geo adapters, deeper pools, longer lists (30-32, 38, 74); cross-fitted labels (75).
- Photo attributes, place categories, map features, OSM names, word statistics as evidence (21, 34, 77-78, 92).
- Landmark / object crop search and descriptors (58-61); names as candidate generators (79); MegaLoc as a proposer (95).
- RL on open-model text outputs (clues, queries) with label rewards: learns silence (87, 96).
- Visual verification of candidates beyond MegaLoc's 12 photos per area: more photos, crops, round trips, fetched references, patches,
  pair discriminants (89-90, 97, 99-101, 103). Local LLM readers for A/B choice (99, 106).
