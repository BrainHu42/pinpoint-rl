# PE-core-G as the image model (track R with track V, 2026-10-22)

**Question.** Every Pinpoint score comes from one frozen SigLIP2-g embedding, and track R found it is the ceiling (R2-R4). Does Meta's
Perception Encoder (PE-core-G, arXiv 2504.13181) carry location signal SigLIP2 lacks, and if so, should it *replace* a model rather than add
one (the user wants few models)?

**Measure.** dev3k, original protocol, % < 25 / 1 km over the current answer head (47.2 / 30.8). Bar: >= +1.0 pt at 25 km, 1 km not down
by more than 0.3. Test set untouched.

## Status (update as steps finish)
| # | Step | Owner | When (UTC) | Status | Result |
|---|---|---|---|---|---|
| 1 | PE-G / PE-L / DINOv3 as per-area columns on the head (gate, CV on dev3k) | track V | done | done | PE-G +0.79 / +0.63, PE-L +0.23, DINOv3-7B +0.23, PE-G + DINOv3 +1.04 |
| 2 | Control: SigLIP2's final embedding as the same columns (no GPU, MP16 cache) | track R | done | done | SigLIP2 -0.03; SigLIP2 + PE-G +1.19 (SE 0.35): the column form adds nothing, the gain is PE-G's |
| 3 | Each model alone: is the best-matching area right? | track R | done | done | PE-G 67.5% vs SigLIP2 63.4% (head's misses: 30.9% vs 24.6%) |
| 4 | V5: answer head retrained on train with the columns, 3 seeds (head alone 47.17 / 30.63) | track V | done | **done, passes** | PE-G +1.37 / +0.97; DINOv3 +0.78; SigLIP2 +0.29; PE-G + DINOv3 +1.59 (pre-declared bar); **PE-G + SigLIP2 +2.04 / +1.28 (SE 0.38)**; all three +2.12 (lesson V5, worktree-value-agent 3f7d1f5) |
| 5 | One PE-G pass at 378 px (~SigLIP2's 384): pooled output (resolution) + mean-pooled layers 32 / 38 / 44 / 47 (intermediate layers), same gate and solo check | track R | 19:40-~21:00 | running (`scripts/pe_layers.py`, log `artifacts/retrieval/gate/pe_layers.log`) | |
| 6 | Decision (below) with the user | user | after 4-5 | | |
| 7 | The chosen swap (below) | track R | next days | | |

Step 4 settled that the gain is real (the >= +1.0 row of the decision table below applies). Step 5 says whether it is PE's knowledge or just
more pixels (448 vs 378 px), sets the resolution for any re-embed, and whether an intermediate layer beats the pooled output (go: a layer
>= +0.5 over the pooled 378 px columns in the gate). Lower priority, run only if asked: SigLIP2 intermediate layers (~40 min GPU); step 2
and the paper make a gain unlikely.

## Decision after steps 4-5
| Outcome of V5 | Then |
|---|---|
| PE-G (alone or + SigLIP2 columns) < +0.5 | Close the PE line (lesson R16). |
| +0.5 to +1.0 | Note as a stackable sub-threshold gain; no new model. |
| >= +1.0 | Pick one swap (step 7), keeping the model count flat: |

Swaps, cheapest first:
- **A. PE-G replaces MegaLoc** (MegaLoc's same-place columns are worth ~+0.8, lesson 93). One trained-head run: head with PE-G columns and
  no MegaLoc vs the current head. Go if >= the current head at 25 km and 1 km. Inference cost: PE-G on the query + ~12 photos per area.
- **B. PE-G replaces SigLIP2 as the backbone**: re-embed MP16 (4.12M photos; ~40-50 h at 448 px, less at 384, resumable over 2-3 nights
  because of the 02:00 shutdown), retrain the retriever and the reranker on it, rebuild the pipeline. The only option that also attacks
  retrieval misses (26.5 pts on dev3k); A only re-ranks existing candidates. Long run: confirm with the user first.

## Where things are
- Track R: `scripts/siglip_gate.py` (branch `worktree-retrieval-track`), outputs `artifacts/retrieval/gate/` (`gate_siglip_final.log`, `solo.log`).
- Track V: `scripts/second_model_gate.py` (`embed` / `gate` / `head` / `siglip`), `scripts/second_model_where.py` (branch `worktree-value-agent`),
  embeddings and logs `artifacts/value/`; V5 lands in its `LESSONS.md`.
- Paper notes: best layers are intermediate for local tasks (OCR, detection), the last layer for global ones; PE-G beats SigLIP2-g by 17.5
  pts on zero-shot Countries211.
