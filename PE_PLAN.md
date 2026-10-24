# PE-core-G as the image model (track R with track V, 2026-10-22)

**Question.** Every Pinpoint score comes from one frozen SigLIP2-g embedding, and track R found it is the ceiling (R2-R4). Does Meta's
Perception Encoder (PE-core-G, arXiv 2504.13181) carry location signal SigLIP2 lacks, and where in the pipeline does it add most? Using both
SigLIP2 and PE-G is fine when both pull their weight (user, 2026-10-22); the evidence so far says they do (PE-G + SigLIP2 columns +2.04 vs
PE-G alone +1.37).

**Measure.** dev3k, original protocol, % < 25 / 1 km over the current answer head (47.2 / 30.8). Bar: >= +1.0 pt at 25 km, 1 km not down
by more than 0.3. Test set untouched. Main test sets: im2gps3k + yfcc4k (dev3k is their dev); commons26 dev is auxiliary, a
secondary check reported alongside, never the deciding set (user, 2026-10-23).

## Workstreams (user, 2026-10-23: two in parallel, each reporting to the user)
| Workstream | Owner | Scope | Reports to |
|---|---|---|---|
| R: PE-G retrieval | track R session (this plan) | B2 full-index embed (nightly), a PE-G-aware picker for the added candidates, PE-G candidate files (extra pool rows for train / dev3k / commons26 dev) | the user |
| V: value network | track V session (`VALUE_PLAN.md`, `VALUE_HANDOFF.md`, branch `worktree-value-agent`) | finish step 1 (lesson V6), then step 1 with the V5 columns; run the V5 answer head on track R's candidate files | the user |

Not started without the user's go-ahead: the regime checks (separate < 25 km / > 25 km models; VALUE_HANDOFF "Pending"), and VALUE_PLAN step 2's
vLLM readers (~35 GB GPU, conflicts with B2). Track V asks the user before runs over ~2 h. Track R and track V talk only about machine
sharing (B2 has GPU priority; track V stays under 20 GB) and the PE-G candidate files.

## Status (update as steps finish)
| # | Step | Owner | When (Pacific) | Status | Result |
|---|---|---|---|---|---|
| 1 | PE-G / PE-L / DINOv3 as per-area columns on the head (gate, CV on dev3k) | track V | done | done | PE-G +0.79 / +0.63, PE-L +0.23, DINOv3-7B +0.23, PE-G + DINOv3 +1.04 |
| 2 | Control: SigLIP2's final embedding as the same columns (no GPU, MP16 cache) | track R | done | done | SigLIP2 -0.03; SigLIP2 + PE-G +1.19 (SE 0.35): the column form adds nothing, the gain is PE-G's |
| 3 | Each model alone: is the best-matching area right? | track R | done | done | PE-G 67.5% vs SigLIP2 63.4% (head's misses: 30.9% vs 24.6%) |
| 4 | V5: answer head retrained on train with the columns, 3 seeds (head alone 47.17 / 30.63) | track V | done | **done, passes** | PE-G +1.37 / +0.97; DINOv3 +0.78; SigLIP2 +0.29; PE-G + DINOv3 +1.59 (pre-declared bar); **PE-G + SigLIP2 +2.04 / +1.28 (SE 0.38)**; all three +2.12 (lesson V5, worktree-value-agent 3f7d1f5) |
| 5 | One PE-G pass at 378 px (~SigLIP2's 384): pooled output (resolution) + mean-pooled layers 32 / 38 / 44 / 47 (intermediate layers), same gate and solo check | track R | done | done (lesson R16) | 378 px pooled +0.87 (448 px +0.79; solo 67.1 vs 67.5%): resolution isn't the source, use 378 px. Mean-pooled layers 32 / 38 / 44 / 47 by cosine +0.34 / +0.19 / +0.39 / +0.37: this readout fails, but it is not the paper's test (5b) |
| 5b | On-claim layer test: a geolocation probe trained per layer on PE-G patch tokens (the paper's Fig. 8 protocol) | track R | done | **done, no-go (R16)** | dev3k top cell within 200 km: blocks 32 / 38 / 44 / 47 9.0 / 14.0 / 25.1 / 34.9, last 45.6, pooled 46.5: the output is best, as for the paper's ImageNet probe. By-product: the pooled region probe as head columns +1.38, on top of PE-G + SigLIP2 cosine +1.18 -> +1.86 / +1.03 (gate) |
| 5c | Region-probe columns in the trained head (train-side columns cross-fitted by photographer). Bar (track V, declared): >= +0.5 at 25 km over pe_g + siglip_final on dev3k, commons26 dev not down | track R | done | **done, passes (R16)** | dev3k 49.21 -> **49.82** (+0.61, SE 0.26), commons26 dev 26.89 -> **27.70** (+0.82, SE 0.48); total over today's head **+2.66 / +1.39** dev3k, **+2.70 / +0.24** commons26 dev |
| 6 | Commit PE-G next to SigLIP2 in the answer head; confirm on commons26 dev (below, A) | track V (`scripts/second_model_commons.sh`, ac43ec4) | done | **done, passes** | commons26 dev (head 25.01 / 7.68): PE-G +1.16 / -0.03 (SE 0.45), **PE-G + SigLIP2 +1.88 / +0.10 (SE 0.54)**; dev3k reproduced (+1.37, +2.04); lesson V5 |
| 7 | Retrieval pre-check on a 10% MP16 gallery (below, B1): PE-G 378 px embedding of 408,092 index photos, then recall@K of SigLIP2 / contrastive / PE-G / unions on dev3k, plus the same small trained projection on PE-G vs SigLIP2 | track R | done 11:42 PM Oct 22 | **done, passes (R17)** | raw PE-G + contrastive at K=32: 61.3 vs 61.1 (+0.2); **trained PE-G 62.7 vs contrastive 57.2 (+5.5), SigLIP2 same recipe 57.7** (25 km, eval half) |
| 8 | Full PE-G index (below, B2): embed all 4,080,926 index photos at 378 px (B1's 408k reused; 224 px rejected, R17), priority order dev3k pools -> commons26 dev pools -> train pools -> the rest | track R | resumed 3:12 PM Oct 23 (user); dev3k + commons26 dev pools done; train pools (992k) ~10 h: ~350k by the 7 PM stop, the rest after the restart, then ~2.5M more (~26 h); resume with `scripts/b2_embed.sh` | running | 592,771 / 4,080,926 at 3:19 PM |
| 9 | PE-G-aware picking (R18): reranker score + lambda x trained PE-G cosine over the setft_own pool (+ PE-G top 16) | track R | Oct 23 | **done, passes** | dev3k 44.30 -> **47.50** (+3.20 +- 0.55; 48.23 with PE-G top 16), commons26 dev 19.65 -> **21.80** (+2.15 +- 0.91; 22.62); SigLIP2 same recipe +0.27 / -0.31 |
| 10 | Trained PE-G similarity as per-area columns in the trained head (R19), over pe_g + siglip_final + pe_probe | track R | Oct 23 | **done, passes on dev3k** | dev3k 49.82 -> **50.87** (+1.05 / +0.67), commons26 dev 27.70 -> **28.18** (+0.48 / +0.13); total over today's head **+3.70 / +2.06**, **+3.17 / +0.38** |

Step 4 settled that the gain is real, so PE-G stays. Step 5 says whether it is PE's knowledge or just
more pixels (448 vs 378 px), sets the resolution for any re-embed, and tried the cheapest middle-layer readout (cosine of mean-pooled tokens; it fails). Lower priority, run only if asked: SigLIP2 intermediate layers (~40 min GPU); step 2
and the paper make a gain unlikely.

## Step 5b: the paper's claim, tested as the paper tests it
The paper reads every layer with a readout *trained* on that layer's frozen features (an attention probe for classification; for OCR / VQA a
projector into an LLM; detection and depth decoders) or by patch-to-patch matching (tracking). Step 5 compared mean-pooled vectors by cosine,
which only the pooled output is trained for, and which averages away the local detail (signs, text, architecture) middle layers are said to
hold. Geolocation is a fine-grained, long-tail task, closer to the paper's OCR / VQA tasks (best layers 38 and 47 of 50) than to ImageNet
(best layer 50).
- Photos: the 35,359 train photos plus ~65k random MP16 photos (never dev3k or benchmark photos), and the 3,000 dev3k queries. PE-G at 378 px
  (~30 photos/s: ~1 h).
- Features: patch tokens of blocks 32 / 38 / 44 / 47 and of the last block, average-pooled 27 x 27 -> 7 x 7 (49 tokens; ~75 GB on EBS),
  plus the pooled output.
- Readout: the same attention probe per layer (one cross-attention block + linear layer), trained to predict geographic cells (~2-4k MP16
  cells); for the pooled output a linear probe and an MLP. Same recipe and budget for every layer.
- Measure on dev3k: % of queries whose predicted cell is within 25 / 200 / 750 km. Go: a middle layer beats both the last block's tokens and
  the pooled output by >= 2 pts at 200 or 750 km (SE ~0.9). Then the probe's probability of each candidate area's cell becomes head columns,
  gated as in step 2 against the pooled-output probe.
- Optional, lower: patch-level matching between the query and an area's photos at block ~32 (the paper's tracking protocol); lesson 101's
  patch matcher (MegaLoc / DINOv2 patches, fetched references) saturated, so it goes after the probe.

## Next steps (PE-G next to SigLIP2)
V5 passed the bar, so PE-G stays. Each step keeps SigLIP2 and asks what PE-G adds on top of it.

- **A. Answer head: PE-G per-area columns next to SigLIP2's** (V5's +2.04 / +1.28 configuration), pooled output; 448 px as in V5, or 378 px (step 5: equal within noise, cheaper).
  - Confirm on commons26 dev (977 photos; the PE-G columns need its queries + area photos embedded, ~30 min). Go: not below the current
    head there.
  - Optional simplification: the same head without MegaLoc; drop MegaLoc only if 25 km and 1 km hold.
  - Inference cost: PE-G on the query + ~12 photos per candidate area. This re-ranks existing candidates only.
- **B. Retrieval: PE-G as a search next to SigLIP2's**, the only lever on retrieval misses (26.5 pts on dev3k).
  - **B1 (pre-check, ~4-5 h GPU, one night):** embed a random 10% of MP16 (~410k photos) with PE-G. On that gallery, compare recall@K
    (within 25 km, K = 8 / 32 / 256) for dev3k queries: SigLIP2 alone, PE-G alone, and their union at equal total K. Go: the union beats
    SigLIP2 at equal K by >= 2 pts at K = 32, or PE-G finds a right photo for >= 3% of the photos SigLIP2 misses.
  - **B2 (if B1 passes):** full re-embed of MP16 with PE-G (4.12M photos; ~40-50 h at 448 px, less at 378, resumable over 2-3 nights
    because of the 02:00 shutdown) as a fourth search in the reranker's pool, then re-fine-tune the reranker with PE-G similarity features.
    Long run: confirm with the user first.
  - **Decisions for the night of Oct 22 (user):** B1's bar is measured against Pinpoint's trained (contrastive) search, the one the
    reranker uses, not raw SigLIP2: go if contrastive + PE-G at equal K beats contrastive alone by >= 2 pts at 25 km, K = 32, or PE-G's
    top-32 finds a photo within 25 km for >= 3% of the photos contrastive's top-32 misses. If it passes, B2 starts right away
    (`scripts/b2_embed.sh`; resume after a restart with the same command). If it fails, a ~2 h test instead: a small search layer trained
    on PE-G embeddings of half the 10% gallery (positives by distance only, photographers split), compared with contrastive on the other half.
- Both A and B keep every SigLIP2 component; no backbone swap is planned unless B shows PE-G retrieval makes SigLIP2's redundant.

## Option: an all-PE-G system (draft for the user, 2026-10-24; not approved, nothing launched)
**Question.** Is the frozen encoder the system's bottleneck? R17-R19: a location-trained layer on PE-G beats the same layer on SigLIP2 by
~+5 recall and ~+3 in the head, and SigLIP2 adds ~0 once PE-G is in, yet search, reranker and head base features all still run on SigLIP2.
Test: the same Pinpoint recipe trained from scratch on each encoder, then our pipeline on each.

| # | Step | Work | GPU (est.) |
|---|---|---|---|
| 0 | B2: PE-G of the 4.08M index rows (running; 29% at 04:25 UTC Oct 24) | done by itself | ~25-30 h left |
| 1 | Contrastive retriever (image + GPS towers, 256-d) from scratch on PE-G, released recipe (50k steps, batch 4096) on MP16 only | `retriever_train.py` hard-codes 1536 and fine-tune-only: make it a from-scratch trainer on any cache (the module itself takes `embed_dim`) | ~1 h |
| 2 | Index v6 rebuild: image / GPS from the new towers, raw = PE-G int8 (1280-d) | `retrieval_index.ensure_attention_retrieval_index` reads any cache in the mp16-embed format: write B2 in that format | ~1-2 h |
| 3 | Reranker from scratch on PE-G (`pinpoint_train`, raw dim from the checkpoint; set head; no OSV support token: worth ~0, lesson 136, and OSV-5M images aren't on disk) | the inference port insists on the OSV token: make it optional | ~2-9 h (6k-50k steps) |
| 4 | Downstream on the new pools: extract, photo sets, hypotheses / restricted / zoom, compare features (`retrained_pipeline.sh`); clue reading unchanged | re-run, PE-G paths | ~4-6 h |
| 5 | Answer head (+ R19's PE-G columns) and the RL policy; dev3k, then im2gps3k + yfcc4k when the user asks; commons26 as the final check | existing code | ~1 h |
| C | **Control:** steps 1-5 with SigLIP2 under the same from-scratch recipe, so recipe and encoder aren't confounded | same code, SigLIP2 cache | ~9-18 h |

Total ~3-4 days of GPU after B2 (with nightly restarts), ~2-3 days of code. Paper table: released Pinpoint, Pinpoint-from-scratch on
SigLIP2, Pinpoint-from-scratch on PE-G, our agent on each. Still on SigLIP2 unless changed: the agent's glance (crop_look view 0),
graph diffusion, near-duplicate labels (minor). Cheaper first look: steps 1-3 at 6k reranker steps (~1 day) before the full recipe.

## Resuming B2 after the daily restart
The embedding checkpoints every 6,400 photos to `artifacts/retrieval/b2/` (EBS; `done.npy` marks embedded rows). After the machine is back,
from this worktree:
```
cd /home/brianhu/pinpoint-rl/.claude/worktrees/retrieval-track
tail -3 artifacts/retrieval/b2/embed.log                      # last checkpoint
setsid nohup bash scripts/b2_embed.sh >> artifacts/retrieval/b2/embed.log 2>&1 &    # continues from done.npy
```
Done when the log says `b2 embed exit` after `4,080,926 / 4,080,926`. Next (to plan with the user): train the PE-G projection on MP16
photos outside the index split used for evaluation and put it in the reranker's candidate pool next to (or instead of) the contrastive search,
then measure the reranker's top-1 on dev3k.

## Where things are
- Track R: `scripts/siglip_gate.py` (branch `worktree-retrieval-track`), outputs `artifacts/retrieval/gate/` (`gate_siglip_final.log`, `solo.log`).
- Track V: `scripts/second_model_gate.py` (`embed` / `gate` / `head` / `siglip`), `scripts/second_model_where.py` (branch `worktree-value-agent`),
  embeddings and logs `artifacts/value/`; V5 lands in its `LESSONS.md`.
- Paper notes: best layers are intermediate for local tasks (OCR, detection), the last layer for global ones; PE-G beats SigLIP2-g by 17.5
  pts on zero-shot Countries211.
