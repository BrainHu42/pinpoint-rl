# Figure: the current best pipeline (action_agent.py on the setft_own reranker; LEARNINGS.md "Where we are", lessons 98, 108-111, 129-130)
# and where it stands. Usage: .venvs/sft/bin/python scripts/pipeline_current_figure.py (matplotlib lives in the sft env) -> docs/pipeline_current.png
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SURFACE, INK, INK2, MUTED, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # reference categorical slots 1-3 (validated all-pairs): Pinpoint, reading, MegaLoc
GREYS = ("#a9a8a2", "#c9c8c2", "#e1e0db")  # losses, darkest = closest to being right

fig = plt.figure(figsize=(19, 11.5), facecolor=SURFACE)
ax = fig.add_axes([0.0, 0.0, 0.64, 1.0])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, title, body, edge=RULE, lw=1.6, fc="white", tag=None, tag_color=MUTED, title_size=12.5, body_size=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.1", fc=fc, ec=edge, lw=lw))
    ax.text(x + 1.4, y + h - 1.8, title, fontsize=title_size, fontweight="bold", color=INK, va="top")
    ax.text(x + 1.4, y + h - 5.4, body, fontsize=body_size, color=INK2, va="top", linespacing=1.42)
    if tag:
        ax.text(x + w - 1.2, y + h - 1.8, tag, fontsize=9, color=tag_color, va="top", ha="right", fontweight="bold")


def dot(x, y, color):
    ax.add_patch(plt.Circle((x, y), 0.75, color=color, transform=ax.transData))


def arrow(x0, y0, x1, y1, color=MUTED, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14, color=color, lw=1.4, linestyle=ls))


ax.text(2.5, 98.2, "The current best pipeline: how one photo gets a location", fontsize=18, fontweight="bold", color=INK, va="top")
ax.text(2.5, 94.6, "An agent that tests a handful of candidate areas. Pinpoint is its search tool; costly tools add evidence; "
        "a learned head commits to one area and a point in it.", fontsize=11, color=INK2, va="top")

# 1. search
box(2.5, 77.5, 95, 13.5, "1   Search: Pinpoint, retrained",
    "Query photo -> SigLIP2 embedding -> Pinpoint's retrieval over its index of 4.08M geotagged MP16 photos -> ~46 candidate photos,\n"
    "scored by Pinpoint's attention reranker fine-tuned on MP16 with the photographer's photos kept (setft_own, lesson 108).\n"
    "Alone on dev3k: 27.4 / 42.3% -> 28.3 / 44.3% within 1 / 25 km (released -> retrained top-1).", edge=BLUE, tag="RETRAINED", tag_color=BLUE)
arrow(20, 77.0, 20, 72.2)

# 2. areas
box(2.5, 50.5, 95, 21, "2   Candidate areas = tokens (~4.6 per photo)", "", edge=RULE)
box(5, 53, 34, 12.5, "      Pinpoint areas (free)",
    "candidates grouped by reranker score\nmass within 25 km (agreement areas),\nwith pool support, density, rank", edge=BLUE, title_size=11, body_size=9.5)
dot(7.2, 63.6, BLUE)
box(42, 53, 34, 12.5, "      Clue places (appear after Read)",
    "places named by the photo's text,\nfound by offline text search\n(GeoNames, Overture, YFCC)", edge=ORANGE, title_size=11, body_size=9.5)
dot(44.2, 63.6, ORANGE)
ax.text(79, 64.8, "Each token carries features:\nscore mass, pool support,\nphotographers, distance to\nthe top pick; more columns\nare filled in by the tools.",
        fontsize=9.3, color=INK2, va="top", linespacing=1.4)
arrow(20, 50.0, 20, 46.2)

# 3. tools + policy
box(2.5, 20.5, 95, 25, "3   Tools, chosen step by step by an RL policy (REINFORCE, costs in units)", "", edge=RULE)
tools = [
    ("Zoom area k  (1)", "re-run Pinpoint restricted to\nthe area (+-25 km): ~12 photos\nand a sharper point inside it", BLUE),
    ("Read  (2)", "the 4B VLM reads text and clues;\ntext search + clue judge add clue\nplaces and clue columns", ORANGE),
    ("Recognise  (1)", "MegaLoc same-place similarity\nbetween the photo and each\narea's ~12 photos", AQUA),
    ("Stop", "hand the current state to\nthe answer head", MUTED),
]
for i, (t, b, c) in enumerate(tools):
    x = 5 + i * 23
    box(x, 24, 21, 14.8, "      " + t, b, edge=c, title_size=10.5, body_size=9.2)
    dot(x + 2.1, 37.0, c)
ax.text(5, 43.0 - 0.2, "", fontsize=1)
arrow(20, 20.0, 20, 16.2)

# 4. answer
box(2.5, 1.5, 60, 14.2, "4   Answer head commits to (area, point)",
    "A small transformer over the area tokens, trained on GPS labels only with random reveal\n"
    "patterns, so it can answer from any state. Point = the area's best global candidate, or\n"
    "the zoomed peak; loss rewards < 25 km, plus 0.5 x < 1 km with --one-km 0.5 (lesson 111).", edge=BLUE)
box(66, 1.5, 31.5, 14.2, "Being tested now (track V)",
    "PE-Core-G + DINOv3-7B similarity per\narea as extra Recognise columns:\n+1.04 pts on the dev3k gate; the trained\nhead test runs tonight.",
    edge=MUTED, fc=SURFACE, title_size=10.5, body_size=9.2)
ax.patches[-1].set_linestyle((0, (4, 3)))
ax.text(97.5, 42.4, "At cost 0 the policy ties 'take every tool' (lesson 98):\nits value is the cost curve, not accuracy.",
        fontsize=9.3, color=INK2, va="top", ha="right", linespacing=1.4, style="italic")

# right: results table
rx = fig.add_axes([0.665, 0.52, 0.32, 0.40])
rx.axis("off")
rx.set_xlim(0, 100)
rx.set_ylim(0, 100)
rx.text(0, 100, "Results: % of photos within 1 / 25 km", fontsize=14, fontweight="bold", color=INK, va="top")
rx.text(0, 92, "Test = im2gps3k + yfcc4k eval halves, original protocol\n(the photographer's photos kept in the index)", fontsize=10, color=INK2, va="top",
        linespacing=1.4)
cols = (44, 64, 84)
rows = [("", ("both", "im2gps3k", "yfcc4k"), True),
        ("Released Pinpoint", ("28.9 / 46.4", "21.8 / 48.9", "33.5 / 44.8"), False),
        ("Pipeline (RL, cost 0)", ("29.1 / 51.2", "22.4 / 52.8", "33.3 / 50.2"), False),
        ("  + --one-km 0.5", ("31.6 / 50.6", "23.2 / 52.4", "37.1 / 49.5"), False)]
y = 72
for label, vals, header in rows:
    rx.text(0, y, label, fontsize=10.5, color=INK, va="center", fontweight="bold" if label.startswith("Pipeline") else "normal")
    for cx, v in zip(cols, vals):
        rx.text(cx, y, v, fontsize=10.5 if not header else 9.5, color=MUTED if header else INK, va="center", ha="center",
                fontweight="bold" if label.startswith("Pipeline") else "normal")
    if header:
        rx.plot([0, 98], [y - 4, y - 4], color=RULE, lw=1)
    y -= 10
rx.text(0, y - 1, "dev3k (MP16, 3,000 photos): released 27.4 / 42.3, pipeline 28.5 / 47.9,\n--one-km 30.8 / 47.3. commons26 test: released 4.6 / 16.8, pipeline 6.1 / 24.2.",
        fontsize=9.3, color=INK2, va="top", linespacing=1.45)

# right: where photos end up (dev3k, --one-km pipeline, 47.3% right)
bx = fig.add_axes([0.665, 0.17, 0.31, 0.20])
bx.set_facecolor(SURFACE)
bx.set_xlim(0, 100)
bx.set_ylim(-0.6, 1.9)
for s_ in bx.spines.values():
    s_.set_visible(False)
bx.set_yticks([])
bx.set_xticks([0, 25, 50, 75, 100])
bx.tick_params(axis="x", colors=MUTED, labelsize=9, length=0)
bx.set_xlabel("% of dev3k photos", fontsize=9.5, color=MUTED)
fig.text(0.665, 0.435, "Where dev3k photos end up (25 km)", fontsize=14, fontweight="bold", color=INK, va="top")
fig.text(0.665, 0.405, "The --one-km pipeline (47.3% right; lessons 129-130): most of the gap is retrieval.\n"
         "Area oracle = some candidate area is right; pool oracle = some candidate photo is.", fontsize=9.5, color=INK2, va="top", linespacing=1.45)
parts = [(47.3, "Right", BLUE, "white"), (14.4, "Wrong area chosen (median 381 km off)", GREYS[0], INK),
         (11.8, "Right photo retrieved but never becomes an area", GREYS[1], INK),
         (26.5, "Retrieval miss: no candidate within 25 km", GREYS[2], INK)]
left = 0.0
for w, label, c, tc in parts:
    bx.barh(0, w - 0.25, left=left + 0.125, height=0.9, color=c, edgecolor="none")
    bx.text(left + w / 2, 0, f"{w:.1f}", ha="center", va="center", fontsize=11, color=tc, fontweight="bold")
    left += w
for xv, label, ha, dx in ((61.7, "area oracle 61.7", "right", -0.8), (73.1, "pool oracle 73.1", "left", 0.8)):
    bx.plot([xv, xv], [0.5, 1.15], color=INK2, lw=1.2)
    bx.text(xv + dx, 1.2, label, ha=ha, va="bottom", fontsize=8.8, color=INK2)
for k, (w, label, c, _) in enumerate(parts):  # legend: identity is never colour alone
    lx, ly = 0.665 + (k % 2) * 0.16, 0.085 - (k // 2) * 0.03
    fig.patches.append(FancyBboxPatch((lx, ly - 0.007), 0.009, 0.014, boxstyle="round,pad=0,rounding_size=0.002", transform=fig.transFigure,
                                      fc=c, ec="none"))
    fig.text(lx + 0.013, ly, label if k == 0 else label.replace(" (median", "\n(median").replace(" but never", "\nbut never").replace(": no", ":\nno"),
             fontsize=8.8, color=INK2, va="center", linespacing=1.2)

out = Path("docs/pipeline_current.png")
fig.savefig(out, dpi=130, facecolor=SURFACE)
print(f"-> {out}")
