# Overnight report (2026-10-07; plan `OVERNIGHT.md`, log `artifacts/overnight/STATE.md`)

**Decision: NO-GO. The GRPO A/B runs were not started.** The knowledge model's knowledge doesn't turn into better candidate choices, and
the diagnosis shows why: it lacks region knowledge on exactly the photos where retrieval fails.

## Ceiling check (no training; chosen / fitted on MP16 dev, applied once to the 3,795 eval-half photos)
| Change in top-1 < 25 km vs reranker (38.2%) | knowledge-a | base 4B |
|---|---|---|
| Rule chosen on dev: best-ranked top-3 candidate in the named country | -0.2 [-0.5, +0.1] (fixes 12, breaks 19) | -0.1 [-0.3, +0.2] (12 / 15) |
| Rank + candidate-name log-prob combiner | -0.1 [-0.2, +0.0] (6 / 10) | -0.4 [-0.7, -0.2] (16 / 31) |
| Dev AUC, best per level: country / region / city | 0.541 / 0.565 / 0.585 | 0.516 / 0.545 / 0.543 |

Gate (fixed before the results): +0.5 at 25 km with CI > 0, or dev AUC +0.03 at country or region. It failed on both: +0.025 / +0.020.
Near miss: cumulative-to-city AUC +0.042, but that level was not part of the gate.

## Why (`knowledge_oracle.py`, eval halves)
Of the 3,795 photos, 792 have the reranker wrong with a right candidate in the pool. Rule: take the best-ranked candidate in the **true** place.
| Perfect knowledge at | < 25 km change | fixes / breaks | knowledge-a names it, on those fixes | base 4B |
|---|---|---|---|---|
| country | +3.3 [+2.7, +3.8] | 126 / 2 | 20.6% | 18.3% |
| region | **+10.8 [+9.8, +11.8]** | 413 / 4 | **10.9%** | 9.0% |
| city | +8.4 [+7.5, +9.3] | 324 / 7 | 5.9% | 3.4% |

- Region knowledge would be worth more than anything tried so far, so knowledge is the right lever.
- The model has that knowledge mostly where retrieval already succeeds: right region on 52.3% of the photos the reranker gets right, 10.9% on the fixable ones.
- The SFT raised the fixable-photo rate only 9.0 -> 10.9%. GRPO sharpens existing knowledge (lesson 3), and this is too little to sharpen.

## Also done
- Fixed a bug in my rule code ("country" matched at city depth) before the decision; the dev choice and result were unchanged in kind.
- Lesson 52 added to `LESSONS.md` and `LEARNINGS.md`, `KNOWLEDGE_PLAN.md` status updated. Commits are on branch `worktree-knowledge-ceiling`
  (worktree `.claude/worktrees/knowledge-ceiling`). The push failed: no GitHub credentials on this machine.
- The GPU was used only for the ceiling check (04:00-04:27) and has been idle since.

## Recommended next step (needs your go-ahead)
Measure how much region knowledge more capacity buys before training anything. Only the 4B is downloaded. Download Qwen3.5-9B, then get its zero-shot
answers on the eval halves (`knowledge_check answer --name base9b`, about 10 min) and rerun `knowledge_oracle`. The number to watch is the right-region
rate on the 413 fixable photos (4B 9.0%, knowledge-a 10.9%). If a bigger model, or Gemini as a reference, is far above that, scale up
(bigger base model plus knowledge SFT); if not, the knowledge has to come from tools at inference (text reading, place search), which fits the RL-agent goal.
