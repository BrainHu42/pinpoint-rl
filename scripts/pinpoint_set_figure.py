# Figure: the look-zoom-rerank design proposed in PINPOINT_SET_PLAN.md, its go / no-go path and the bars it must clear.
# Usage: .venvs/sft/bin/python scripts/pinpoint_set_figure.py   (matplotlib lives in the sft env) -> docs/pinpoint_set_design.png
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

SURFACE, INK, INK2, MUTED, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # reference categorical slots 1-3 (validated all-pairs)

fig = plt.figure(figsize=(18, 11), facecolor=SURFACE)
ax = fig.add_axes([0.0, 0.0, 0.62, 1.0])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, title, body, edge=RULE, ls="-", lw=1.6, title_color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=edge, lw=lw, ls=ls))
    ax.text(x + 1.5, y + h - 2.2, title, fontsize=12.5, fontweight="bold", color=title_color, va="top")
    ax.text(x + 1.5, y + h - 6.0, body, fontsize=10, color=INK2, va="top", linespacing=1.45)


def arrow(x0, y0, x1, y1, label=None, lx=None, ly=None, ha="left"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=1.4))
    if label:
        ax.text(lx, ly, label, fontsize=9, color=MUTED, ha=ha, va="center")


def tokens(x, y, n, color, dashed=False, w=1.7, gap=0.55):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * (w + gap), y), w, 2.6, fc="white" if dashed else color, ec=color, lw=1.4,
                               ls="--" if dashed else "-"))
    return x + n * (w + gap) - gap


ax.text(3, 97.5, "Pinpoint-Set: look, zoom, rerank once", fontsize=17, fontweight="bold", color=INK, va="top")
ax.text(3, 93.6, "Retrieval is the only core tool; one set-aware model trained for area-level correctness replaces the "
        "hand-built comparator", fontsize=11, color=INK2, va="top")

# 1 Look, 2 Zoom
box(3, 74, 42, 15, "1  Look: Pinpoint global search",
    "Search the 4.08M-photo MP16 gallery\n(image / raw / GPS similarity)\n-> ~46 global candidates", edge=BLUE)
box(52, 74, 45, 15, "2  Zoom: Pinpoint restricted to each area",
    "Search again inside each of the top-K areas\n(+-25 km box) -> ~12 candidates per area.\nRight area given: 61% within 25 km on dev3k", edge=ORANGE)
arrow(45.6, 81.5, 51.4, 81.5, "top-K\nareas", 48.5, 85.5, ha="center")

# 3 Rerank once
RX, RY, RW, RH = 3, 29, 66, 38
box(RX, RY, RW, RH, "3  Rerank once: one set-aware model", "", edge=INK, lw=1.8)
arrow(24, 73.5, 24, 67.5, "global", 25, 70.5)
arrow(66, 73.5, 60, 67.5, "zoomed", 64.5, 70.5)
ax.text(5, 61.2, "All ~100 candidates attend to each other in one pass", fontsize=10, color=INK2, va="top")
ty = 52.5
end = tokens(5, ty, 8, BLUE)
ax.text(5, ty - 1.4, "global candidates (~46)", fontsize=9, color=INK2, va="top")
gx = end + 2.2
area_x = []
for a in range(3):
    s = gx
    gx = tokens(gx, ty, 4, ORANGE) + 1.6
    area_x.append((s, gx - 1.6))
    ax.text(s, ty - 1.4, f"area {a + 1}", fontsize=9, color=INK2, va="top")
ax.text(area_x[0][0], ty - 4.1, "zoomed candidates (~12 per area)", fontsize=9, color=INK2, va="top")
cx = gx + 0.6
tokens(cx, ty, 2, AQUA, dashed=True)
ax.text(cx, ty - 1.4, "clue\ntokens", fontsize=9, color=INK2, va="top", linespacing=1.2)
for s, e in area_x:  # area tokens (step 2) attend to their members
    mid = (s + e) / 2
    ax.add_patch(Rectangle((mid - 1.1, ty + 4.6), 2.2, 2.6, fc="white", ec=INK, lw=1.4))
    for i in range(4):
        ax.plot([mid, s + i * 2.25 + 0.85], [ty + 4.6, ty + 2.6], color=MUTED, lw=0.7)
ax.text(area_x[-1][1] + 0.8, ty + 5.9, "area tokens (step 2)", fontsize=9, color=INK2, va="center")

rows = [("Inputs", "Pinpoint's CLS states, scores, pairwise distances"),
        ("Output", "P(area) x P(spot | area)"),
        ("Loss", "expected hit: -log softmax mass on candidates < 25 km (x2 < 1 km)"),
        ("Replaces", "area-grouping rule, comparator + hand features, pick-rule choice")]
for i, (k, v) in enumerate(rows):
    yy = 42.5 - i * 3.2
    ax.text(5, yy, k, fontsize=10, fontweight="bold", color=INK, va="top")
    ax.text(13.5, yy, v, fontsize=10, color=INK2, va="top")

# 4 Optional clue tokens
box(73, 41, 24, 26, "4  Optional clue tokens",
    "The only LLM use: the 4B\nreads text in the photo.\nGeocode it (offline search,\nlater live geocoders), judge\nit -> token with place +\njudge score.\nWorth +0.6-0.7 today", edge=AQUA, ls="--")
arrow(72.4, 50, 69.6, 50)

# Answer
box(3, 4, 42, 19, "Answer",
    "Spot with the highest P(area) x P(spot | area).\nSame output at 25 km and 1 km;\nno separate pick rule")
arrow(24, 28.5, 24, 22.5)

# Removed
box(52, 4, 45, 19, "Removed from the current pipeline", "", edge=RULE, ls="--", title_color=MUTED)
for i, t in enumerate(["4B place proposals (+0.1-0.2)", "4B plausibility judges (no gain)",
                       "clue features and 29-36 hand comparator features", "the separate comparator",
                       "the choice between pick rules (agreement vs set layer)"]):
    yy = 16.3 - i * 2.9
    ax.text(54, yy, "x", fontsize=10, color=MUTED, va="center", fontweight="bold")
    ax.text(56.5, yy, t, fontsize=10, color=MUTED, va="center")

# Right: path with go / no-go
fig.text(0.645, 0.885, "Path, each step with a go / no-go", fontsize=15, fontweight="bold", color=INK)
fig.text(0.645, 0.857, "Decide on train 5-fold CV, check on dev3k; test set only when asked", fontsize=10.5, color=INK2)
steps = [("Set model, global + zoomed, frozen Pinpoint states", "go if >= +3.4 at 25 km, ideally near +4.1"),
         ("Area-then-spot output (area tokens)", "keeps step 1 at 25 km, gains at 1 km"),
         ("Clue tokens (judge score + place)", "recovers the clues' +0.6"),
         ("Retrain Pinpoint's reranker: set objective,\nin-area hard negatives, all 4M photos", "beats step 3; the main lever"),
         ("Later: LLM agent trained with RL chooses\nzooms and live lookups", "beats fixed top-K at equal cost")]
sx = fig.add_axes([0.645, 0.43, 0.34, 0.41])
sx.set_xlim(0, 100)
sx.set_ylim(0, 100)
sx.axis("off")
sx.plot([3, 3], [8, 92], color=RULE, lw=2, zorder=0)
for i, (what, go) in enumerate(steps):
    yy = 92 - i * 21
    sx.add_patch(plt.Circle((3, yy), 2.6, fc="white" if i == 4 else INK, ec=INK, lw=1.4, zorder=2))
    sx.text(3, yy, str(i + 1), fontsize=9, color=INK if i == 4 else "white", ha="center", va="center", fontweight="bold", zorder=3)
    sx.text(9, yy + 2.5, what, fontsize=10.5, color=INK, va="top", linespacing=1.3)
    sx.text(9, yy - (8.5 if "\n" in what else 4.5), go, fontsize=9.5, color=MUTED, va="top")

# Right: bars step 1 must clear
fig.text(0.645, 0.375, "Bars to clear (points within 25 km over Pinpoint, train CV)", fontsize=12, fontweight="bold", color=INK)
bx = fig.add_axes([0.80, 0.13, 0.15, 0.22], facecolor=SURFACE)
bars = [("Set layer alone", 2.0, 2.1), ("Pinpoint areas + set layer\n(step 1 go bar)", 3.4, 4.0), ("Full current pipeline\n(step 1 ideal)", 4.1, 4.7)]
y = list(range(len(bars)))[::-1]
bx.barh(y, [b[1] for b in bars], height=0.56, color=BLUE, edgecolor=SURFACE, linewidth=2)
for yi, (_, tr, dv) in zip(y, bars):
    bx.text(tr + 0.08, yi, f"+{tr:.1f}", va="center", fontsize=11, color=INK, fontweight="bold")
    bx.text(5.4, yi, f"dev3k +{dv:.1f}", va="center", fontsize=9.5, color=MUTED)
bx.set_yticks(y, [b[0] for b in bars], fontsize=10, color=INK)
bx.set_xlim(0, 6.9)
bx.set_xticks(range(5))
for s in ("top", "right", "left"):
    bx.spines[s].set_visible(False)
bx.spines["bottom"].set_color(RULE)
bx.tick_params(axis="x", colors=MUTED)
bx.tick_params(axis="y", length=0)
bx.xaxis.grid(True, color=RULE, lw=0.8)
bx.set_axisbelow(True)
fig.text(0.645, 0.075, "Pinpoint alone: 32.4% within 25 km on train, 33.4% on dev3k. Current pipeline\n"
         "on the benchmark eval halves (one run): +4.6 (41.8 vs 37.2).\nIf step 1 fails, keep it and go to step 4.",
         fontsize=9.5, color=MUTED, va="top", linespacing=1.5)

out = Path("docs/pinpoint_set_design.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=130, facecolor=SURFACE)
print(out)
