# Figure: how the best geolocation pipeline works (PINPOINT_SET_PLAN.md results, lessons 68-70) and what each part adds.
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


def note(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=SURFACE, ec=color, lw=1.4, ls=(0, (4, 3))))
    ax.text(x + 1.5, y + h - 2.0, title, fontsize=10.5, fontweight="bold", color=INK, va="top")
    ax.text(x + 1.5, y + h - 5.4, body, fontsize=9.2, color=INK2, va="top", linespacing=1.4)


ax.text(3, 97.5, "How the best pipeline geolocates one photo", fontsize=17, fontweight="bold", color=INK, va="top")
ax.text(3, 93.6, "Look (global search), zoom into candidate areas, choose an area from Pinpoint's own states inside each zoom",
        fontsize=11, color=INK2, va="top")

# 1 Look
box(3, 75, 44, 14, "1  Look: Pinpoint global search",
    "Search 4.08M geotagged MP16 photos (image / raw /\nGPS similarity) -> ~46 candidates; Pinpoint's reranker\nscores each, the set layer rescores them jointly")
note(52, 75, 45, 14, "Drop-in for step 1: retrained reranker (step 4)",
     "Pinpoint's reranker fine-tuned on its own 4M photos (fixed\nleave-one-out, photographer exclusion, co-trained set head):\n+3.1 / +3.3 alone. Not yet combined with steps 2-5.", BLUE)

# 2 Propose
box(3, 50, 94, 20, "2  Propose ~5-6 areas", "")
srcs = [(BLUE, "Pinpoint's top-5 areas", "candidates grouped by score mass\nwithin 25 km (agreement areas)"),
        (ORANGE, "Clue places (+0.7)", "4B reads text in the photo -> offline\ntext search -> zero-shot clue judge")]
for i, (c, t, b) in enumerate(srcs):
    x = 5 + i * 31
    ax.add_patch(FancyBboxPatch((x, 52.5), 29, 11, boxstyle="round,pad=0.3,rounding_size=1", fc="white", ec=c, lw=2.2))
    ax.add_patch(plt.Circle((x + 2.2, 61.2), 1.0, color=c))
    ax.text(x + 4.2, 61.3, t, fontsize=10.5, fontweight="bold", color=INK, va="center")
    ax.text(x + 1.5, 58.4, b, fontsize=9.5, color=INK2, va="top", linespacing=1.4)
ax.text(69, 61.5, "Dropped: the 4B's own place\nproposals (+0.2 train, -0.3 dev3k)\nand plausibility judges (no gain)", fontsize=9.3, color=MUTED, va="top", linespacing=1.4)
arrow(25, 74.5, 25, 70.5)

# 3 Zoom
box(3, 28, 60, 16, "3  Zoom: Pinpoint restricted to each area",
    "Search again inside a +-25 km box around each area\n-> ~12 candidates per area, each with Pinpoint's CLS state,\nscore and retrieval features")
note(67, 28, 30, 16, "Option: RL zoom agent (step 5)",
     "Policy picks which areas to zoom\nfrom pre-zoom evidence. 1 zoom:\n~90% of the gain; 2 zooms: all of it\n(vs zooming all ~6)", AQUA)
arrow(25, 49.5, 25, 44.5)
arrow(63.5, 36, 66.5, 36)

# 4 Area model and 5 answer
box(3, 3, 45, 19, "4  Area model picks one area",
    "Per area: attention-pool its zoom's CLS states\n+ evidence (score mass, support, photographers,\nclue statistics, set-layer mass); one transformer\nlayer across the photo's areas (listwise loss)", edge=NEW, tag="NEW")
box(52, 3, 45, 19, "5  Answer inside the chosen area",
    "Agreement pick (best at 25 km)\nor the set layer's pick if it falls inside\n(best at 1 km: +1.7 train / +1.8 dev3k)")
arrow(25, 27.5, 25, 22.5)
arrow(48.5, 12.5, 51.5, 12.5)

# Right: what each part adds
bx = fig.add_axes([0.80, 0.30, 0.17, 0.52], facecolor=SURFACE)
rows = [("Agreement rule\n(Pinpoint's pool)", 1.0, 2.0, BLUE), ("Set layer alone", 2.0, 2.1, BLUE),
        ("Retrained reranker\nalone (step 4)", 3.1, 3.3, BLUE), ("Area model,\nPinpoint areas only", 3.5, 3.9, BLUE),
        ("+ clue places\n(best pipeline)", 4.2, 5.2, BLUE), ("Previous pipeline\n(hand comparator)", 4.1, 4.7, MUTED)]
y = list(range(len(rows)))[::-1]
bx.barh(y, [r[1] for r in rows], height=0.56, color=[r[3] for r in rows], edgecolor=SURFACE, linewidth=2)
for yi, (label, tr, dv, _) in zip(y, rows):
    bx.text(tr + 0.08, yi, f"+{tr:.1f}", va="center", fontsize=11, color=INK, fontweight="bold")
    bx.text(5.3, yi, f"dev3k +{dv:.1f}", va="center", fontsize=9.5, color=MUTED)
bx.set_yticks(y, [r[0] for r in rows], fontsize=10, color=INK)
bx.set_xlim(0, 6.8)
bx.set_xticks(range(6))
bx.set_xlabel("points within 25 km over Pinpoint\n(train, 5-fold CV; agreement pick)", fontsize=10, color=INK2)
for s_ in ("top", "right", "left"):
    bx.spines[s_].set_visible(False)
bx.spines["bottom"].set_color(RULE)
bx.tick_params(axis="x", colors=MUTED)
bx.tick_params(axis="y", length=0)
bx.xaxis.grid(True, color=RULE, lw=0.8)
bx.set_axisbelow(True)
fig.text(0.655, 0.89, "What each part adds", fontsize=15, fontweight="bold", color=INK)
fig.text(0.655, 0.86, "Pinpoint alone: 32.4% within 25 km on train, 33.4% on dev3k", fontsize=10.5, color=INK2)
fig.text(0.655, 0.20, "Tried, no gain: one flat set model over global + zoomed\ncandidates (+1.4), the set head's own picks, more hypotheses,\n"
         "4B plausibility judges, extra comparator photos.\n\nBest pipeline not yet run on the test set; the previous one\n"
         "gave +4.6 on the im2gps3k / yfcc4k eval halves.", fontsize=9.5, color=MUTED, va="top", linespacing=1.5)

out = Path("docs/pipeline_overview.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=130, facecolor=SURFACE)
print(out)
