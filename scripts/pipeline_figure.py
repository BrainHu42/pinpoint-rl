# Figure: how the current geolocation pipeline works (HYPOTHESIS_PLAN.md, lessons 68-69, set-aware Pinpoint) and what each part adds.
# Usage: .venvs/sft/bin/python scripts/pipeline_figure.py   (matplotlib lives in the sft env) -> docs/pipeline_overview.png
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SURFACE, INK, INK2, MUTED, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # reference categorical slots 1-3 (validated all-pairs)
NEW = "#2a78d6"

fig = plt.figure(figsize=(17, 10.5), facecolor=SURFACE)
ax = fig.add_axes([0.0, 0.0, 0.60, 1.0])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.set_facecolor(SURFACE)


def box(x, y, w, h, title, body, edge=RULE, tag=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=edge, lw=1.6))
    ax.text(x + 1.5, y + h - 2.2, title, fontsize=12.5, fontweight="bold", color=INK, va="top")
    ax.text(x + 1.5, y + h - 6.3, body, fontsize=10, color=INK2, va="top", linespacing=1.45)
    if tag:
        ax.text(x + w - 1.2, y + h - 2.2, tag, fontsize=9, color=NEW, va="top", ha="right", fontweight="bold")


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=1.4))


ax.text(3, 97.5, "How the pipeline geolocates one photo", fontsize=17, fontweight="bold", color=INK, va="top")
ax.text(3, 93.6, "Propose areas, test each with Pinpoint restricted to it, pick one with a learned comparator", fontsize=11, color=INK2, va="top")

# 1 Pinpoint global search
box(3, 74, 44, 15, "1  Pinpoint: global search",
    "SigLIP2 embedding of the photo; search 4.08M geotagged\nMP16 photos (image / raw / GPS similarity) -> ~46 candidates;\nPinpoint's reranker scores each candidate on its own")
# 2 Set layer
box(52, 74, 45, 15, "2  Set-aware layer",
    "Sees all candidates together: Pinpoint's internal state,\nscore and pairwise distances -> a probability per candidate.\nTrained to maximise the chance of a pick within 25 km", edge=NEW, tag="NEW")
arrow(47.5, 81.5, 51.5, 81.5)

# 3 Hypotheses
box(3, 44, 86, 24, "3  Propose ~6 area hypotheses", "")
srcs = [(BLUE, "Pinpoint's top-5 areas", "agreement areas: candidates grouped\nby score mass within 25 km"),
        (ORANGE, "Clue places", "4B reads text / recalls names ->\noffline text search -> clue judge"),
        (AQUA, "4B place proposals", "the 4B names up to 3 places,\ngeocoded (adds only +0.1-0.2)")]
for i, (c, t, b) in enumerate(srcs):
    x = 5 + i * 28
    ax.add_patch(FancyBboxPatch((x, 46.5), 26, 14, boxstyle="round,pad=0.3,rounding_size=1", fc="white", ec=c, lw=2.2))
    ax.add_patch(plt.Circle((x + 2.2, 58.2), 1.0, color=c))
    ax.text(x + 4.2, 58.3, t, fontsize=10.5, fontweight="bold", color=INK, va="center")
    ax.text(x + 1.5, 55.2, b, fontsize=9.5, color=INK2, va="top", linespacing=1.4)
arrow(25, 73.5, 25, 68.5)

# 4 Test each hypothesis
box(3, 22, 94, 16, "4  Test each hypothesis: Pinpoint restricted to a +-25 km box",
    "Evidence per area: restricted score mass and top scores, support in Pinpoint's pool, distinct photographers,\n"
    "gallery density, area rank and source, clue statistics, distance to Pinpoint's top-1, and the set layer's\n"
    "probability inside the area / whether its top pick falls inside")
arrow(46, 43.5, 46, 38.5)
arrow(93, 73.5, 93, 38.5)
ax.text(94, 56, "set-layer\nprobabilities", fontsize=9, color=MUTED, ha="left", va="center")

# 5 Choose and answer
box(3, 2, 45, 14, "5  Comparator picks one area",
    "Set-level MLP over the photo's hypotheses\n(listwise loss, GPS labels, 5-model ensemble)")
box(52, 2, 45, 14, "6  Answer inside the chosen area",
    "Set layer's pick if it falls inside (+1.6 at 1 km),\nelse the agreement pick (best at 25 km)")
arrow(25, 21.5, 25, 16.5)
arrow(48.5, 9, 51.5, 9)

# Right: what each part adds
bx = fig.add_axes([0.765, 0.30, 0.20, 0.52], facecolor=SURFACE)
rows = [("Agreement rule\n(Pinpoint's pool)", 1.0, 2.0), ("Set layer alone", 2.0, 2.1), ("Hypothesis testing,\nPinpoint only", 2.7, 3.9),
        ("+ clue places\n+ 4B proposals", 3.5, 4.6), ("+ set layer\n(current pipeline)", 4.1, 4.7)]
y = list(range(len(rows)))[::-1]
bars = bx.barh(y, [r[1] for r in rows], height=0.56, color=BLUE, edgecolor=SURFACE, linewidth=2)
for yi, (label, tr, dv) in zip(y, rows):
    bx.text(tr + 0.08, yi, f"+{tr:.1f}", va="center", fontsize=11, color=INK, fontweight="bold")
    bx.text(5.0, yi, f"dev3k +{dv:.1f}", va="center", fontsize=9.5, color=MUTED)
bx.set_yticks(y, [r[0] for r in rows], fontsize=10, color=INK)
bx.set_xlim(0, 6.3)
bx.set_xticks(range(5))
bx.set_xlabel("points within 25 km over Pinpoint\n(train, 5-fold CV)", fontsize=10, color=INK2)
for s in ("top", "right", "left"):
    bx.spines[s].set_visible(False)
bx.spines["bottom"].set_color(RULE)
bx.tick_params(axis="x", colors=MUTED)
bx.tick_params(axis="y", length=0)
bx.xaxis.grid(True, color=RULE, lw=0.8)
bx.set_axisbelow(True)
fig.text(0.655, 0.89, "What each part adds", fontsize=15, fontweight="bold", color=INK)
fig.text(0.655, 0.86, "Pinpoint alone: 32.4% within 25 km on train, 33.4% on dev3k", fontsize=10.5, color=INK2)
fig.text(0.655, 0.20, "Tried, no gain: more hypotheses (areas 6-10), 35k extra comparator\n"
         "photos, 4B plausibility judges (zero-shot and fine-tuned),\n"
         "the set layer run inside each area.\n\nTest set (im2gps3k / yfcc4k) never used.", fontsize=9.5, color=MUTED, va="top", linespacing=1.5)

out = Path("docs/pipeline_overview.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=130, facecolor=SURFACE)
print(out)
