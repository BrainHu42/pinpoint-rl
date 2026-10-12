# Figure: how the merged geolocation pipeline works (PINPOINT_SET_PLAN.md, lessons 68-71) and what each part adds.
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


ax.text(3, 97.5, "How the merged pipeline geolocates one photo", fontsize=17, fontweight="bold", color=INK, va="top")
ax.text(3, 93.6, "A budgeted hypothesis-testing agent: the retrained Pinpoint is its search tool; it decides between areas, not photos",
        fontsize=11, color=INK2, va="top")

# 1 Look
box(3, 75, 94, 14, "1  Look: retrained Pinpoint reranker, global search",
    "Search 4.08M geotagged MP16 photos (image / raw / GPS similarity) -> ~46 candidates, scored by Pinpoint's reranker\n"
    "fine-tuned on its own photos (wide candidate lists, photographer exclusion, co-trained set head; +3.1 / +3.3 alone).\n"
    "Its softmax replaces the separate set layer.",
    edge=NEW, tag="RETRAINED")

# 2 Propose
box(3, 50, 94, 20, "2  Propose ~4-6 areas = area tokens", "")
srcs = [(BLUE, "Pinpoint's top-5 areas", "candidates grouped by score mass\nwithin 25 km (agreement areas)"),
        (ORANGE, "Clue places (+0.7)", "4B reads text in the photo -> offline\ntext search -> zero-shot clue judge")]
for i, (c, t, b) in enumerate(srcs):
    x = 5 + i * 31
    ax.add_patch(FancyBboxPatch((x, 52.5), 29, 11, boxstyle="round,pad=0.3,rounding_size=1", fc="white", ec=c, lw=2.2))
    ax.add_patch(plt.Circle((x + 2.2, 61.2), 1.0, color=c))
    ax.text(x + 4.2, 61.3, t, fontsize=10.5, fontweight="bold", color=INK, va="center")
    ax.text(x + 1.5, 58.4, b, fontsize=9.5, color=INK2, va="top", linespacing=1.4)
ax.text(69, 63, "Each token starts with pre-zoom\nevidence: pool support, density,\nclue statistics, distance to the\ntop-1, probability mass in the area",
        fontsize=9.3, color=INK2, va="top", linespacing=1.4)
arrow(25, 74.5, 25, 70.5)

# 3 Zoom
box(3, 28, 60, 16, "3  Zoom (budget: 2 areas)",
    "Action head picks the next area to search again inside a +-25 km box\n(RL, sequential). Each zoom adds its ~12 candidates' CLS states and\nscores to that area's token. 2 zooms = zooming all areas")
note(67, 28, 30, 16, "Fixed rule does as well",
     "Zooming Pinpoint's top-2 areas:\n+4.7 / +5.2 (agent +4.7 / +5.5).\nNo zoom at all: +3.9 / +3.8", MUTED)
arrow(25, 49.5, 25, 44.5)
arrow(63.5, 36, 66.5, 36)

# 4 Answer head and 5 answer
box(3, 3, 45, 19, "4  Answer head picks one area",
    "Transformer over the photo's area tokens,\nzoomed or not (trained on random zoom subsets,\nlistwise loss, GPS labels only)", edge=NEW, tag="NEW")
box(52, 3, 45, 19, "5  Answer inside the chosen area",
    "Agreement pick in its zoom (best at 25 km), or the\nglobal top pick if inside (keeps 1 km: +1.7 / +1.1).\nUnzoomed area: best global candidate inside it")
arrow(25, 27.5, 25, 22.5)
arrow(48.5, 12.5, 51.5, 12.5)

# Right: what each part adds
bx = fig.add_axes([0.80, 0.30, 0.17, 0.52], facecolor=SURFACE)
rows = [("Agreement rule\n(Pinpoint's pool)", 1.0, 2.0, BLUE), ("Set layer alone", 2.0, 2.1, BLUE),
        ("Retrained reranker\nalone", 3.1, 3.3, BLUE), ("Area model on\nreleased Pinpoint", 4.2, 5.2, BLUE),
        ("Merged, 2 zooms\n(agent)", 4.7, 5.5, BLUE), ("Merged, all zoomed\n(best)", 4.9, 5.4, BLUE),
        ("Previous pipeline\n(hand comparator)", 4.1, 4.7, MUTED)]
y = list(range(len(rows)))[::-1]
bx.barh(y, [r[1] for r in rows], height=0.56, color=[r[3] for r in rows], edgecolor=SURFACE, linewidth=2)
for yi, (label, tr, dv, _) in zip(y, rows):
    bx.text(tr + 0.08, yi, f"+{tr:.1f}", va="center", fontsize=11, color=INK, fontweight="bold")
    bx.text(6.3, yi, f"dev3k +{dv:.1f}", va="center", fontsize=9.5, color=MUTED)
bx.set_yticks(y, [r[0] for r in rows], fontsize=10, color=INK)
bx.set_xlim(0, 7.9)
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
         "4B place proposals and plausibility judges.\n\nMerged pipeline not yet run on the test set; the previous one\n"
         "gave +4.6 on the im2gps3k / yfcc4k eval halves.", fontsize=9.5, color=MUTED, va="top", linespacing=1.5)

out = Path("docs/pipeline_overview.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=130, facecolor=SURFACE)
print(out)
