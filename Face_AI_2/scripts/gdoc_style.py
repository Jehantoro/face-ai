"""Shared look for the pictures of the Week 5-7 report: the same style as the Week 3-4 pictures
(light surface, rounded boxes tagged FIXED / FROZEN / TRAINED, bold title with a grey subtitle, Segoe UI)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch                # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "reports" / "gdoc_week5-7" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, PANEL = "#fbfbfa", "#ffffff"
INK, INK2, MUTED, GRID, RULE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#c9c7c0"
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED, GREY, GREEN = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6",
                                                         "#d03b3b", "#9b9a96", "#008300")
SHADE = "#f0efec"
KIND = {
    "fixed": dict(fill="#f4f3f0", edge="#c9c7c0", pill="#8f8e8a", label="FIXED"),
    "frozen": dict(fill="#eef0f5", edge="#8a90a3", pill="#5f6577", label="FROZEN"),
    "finger": dict(fill="#e9f7f1", edge="#1baf7a", pill="#12855c", label="FINGER-TAUGHT"),
    "trained": dict(fill="#eaf2fd", edge="#2a78d6", pill="#2a78d6", label="TRAINED"),
    "linear": dict(fill="#efedfb", edge="#6250d6", pill="#6250d6", label="TRAINED"),
    "plain": dict(fill="#ffffff", edge="#c9c7c0", pill=None, label=None),
    "note": dict(fill="#fbfbfa", edge="#e4e3df", pill=None, label=None),
}
W = 12.0                                                                      # figure width in inches (2,400 px)


def setup():
    plt.rcParams.update({
        "font.family": "Segoe UI", "font.size": 15, "axes.titlesize": 17, "axes.titleweight": "bold",
        "axes.labelsize": 14.5, "xtick.labelsize": 13.5, "ytick.labelsize": 13.5, "legend.fontsize": 14,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": RULE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True, "legend.frameon": False, "axes.titlelocation": "left",
        "axes.titlepad": 10, "lines.linewidth": 2.0, "patch.linewidth": 0, "svg.fonttype": "none",
    })


def canvas(h, w=W):
    """A drawing canvas in units of 1/100 inch, origin top-left (x right, y down)."""
    setup()
    fig = plt.figure(figsize=(w, h), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w * 100)
    ax.set_ylim(h * 100, 0)
    ax.axis("off")
    return fig, ax


def title(ax, text, sub=None, x=40, y=52, size=27, subsize=16.5):
    ax.text(x, y, text, fontsize=size, fontweight="bold", color=INK, va="baseline")
    if sub:
        for i, line in enumerate(sub if isinstance(sub, (list, tuple)) else [sub]):
            ax.text(x, y + 40 + 28 * i, line, fontsize=subsize, color=INK2, va="baseline")


def pill(ax, cx, cy, text, color, size=12.5, pad=0.32):
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size, fontweight="bold", color="white",
            bbox=dict(boxstyle=f"round,pad={pad},rounding_size=0.9", fc=color, ec="white", lw=1.6), zorder=6)


def box(ax, x, y, w, h, kind="plain", head=None, lines=(), headsize=19, size=15.5, tag=None, lw=2.2,
        align="center", line_gap=27, head_gap=33, valign="center", color=None, zorder=3):
    k = KIND[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14", fc=k["fill"],
                                ec=k["edge"] if color is None else color, lw=lw, zorder=zorder))
    label = tag if tag is not None else k["label"]
    if label and k["pill"]:
        pill(ax, x + w / 2, y, label, k["pill"] if color is None else color)
    n = (1 if head else 0) + len(lines)
    total = (head_gap if head else 0) + line_gap * max(len(lines) - 1, 0)
    if valign == "center":
        cy = y + h / 2 - total / 2
    else:
        cy = y + 42
    tx = {"center": x + w / 2, "left": x + 22}[align]
    if head:
        ax.text(tx, cy, head, ha=align, va="center", fontsize=headsize, fontweight="bold", color=INK, zorder=zorder + 1)
        cy += head_gap
    for line in lines:
        ax.text(tx, cy, line, ha=align, va="center", fontsize=size, color=INK2, zorder=zorder + 1)
        cy += line_gap
    return (x, y, w, h)


def arrow(ax, p1, p2, color=GREY, lw=2.6, style="-|>", ms=20, rad=0.0, ls="-", zorder=2):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=zorder)
    ax.add_patch(a)


def legend_kinds(ax, x, y, kinds, gap=330, size=15.5):
    """One row of tag explanations, like the Week 3-4 pictures' footers."""
    text = {"fixed": "our code, no learning", "frozen": "pretrained by others, never changed",
            "finger": "learned here, the finger pulse as teacher", "trained": "learned here from the health labels"}
    cx = x
    for k in kinds:
        pill(ax, cx + 50, y, KIND[k]["label"], KIND[k]["pill"], size=12.5)
        ax.text(cx + 50 + 14 * len(KIND[k]["label"]) / 2 + 30, y, text[k], fontsize=size, color=INK2, va="center")
        cx += gap


def save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def face(ax, cx, cy, s=1.0, skin="#f2d4bd", hair="#5a4636", shirt="#a9bfd8", line="#7a6a5e", z=3, clip=None):
    """A simple drawn face with shoulders (no participant is shown in this report). clip: (x, y, w, h) to cut it to."""
    from matplotlib.patches import Ellipse, Polygon, Arc, Circle, Rectangle
    n0, l0 = len(ax.patches), len(ax.lines)
    _face(ax, cx, cy, s, skin, hair, shirt, line, z)
    if clip is not None:
        r = Rectangle(clip[:2], clip[2], clip[3], transform=ax.transData)
        for art in list(ax.patches[n0:]) + list(ax.lines[l0:]):
            art.set_clip_path(r)


def _face(ax, cx, cy, s, skin, hair, shirt, line, z):
    from matplotlib.patches import Ellipse, Polygon, Arc, Circle
    # shoulders and chest
    ax.add_patch(Polygon([(cx - 190 * s, cy + 300 * s), (cx - 150 * s, cy + 190 * s), (cx - 45 * s, cy + 150 * s),
                          (cx + 45 * s, cy + 150 * s), (cx + 150 * s, cy + 190 * s), (cx + 190 * s, cy + 300 * s)],
                         closed=True, fc=shirt, ec="none", zorder=z))
    ax.add_patch(Polygon([(cx - 38 * s, cy + 95 * s), (cx + 38 * s, cy + 95 * s), (cx + 42 * s, cy + 160 * s),
                          (cx - 42 * s, cy + 160 * s)], closed=True, fc=skin, ec="none", zorder=z))
    # head
    ax.add_patch(Ellipse((cx, cy), 190 * s, 245 * s, fc=skin, ec=line, lw=1.2, zorder=z + 1))
    ax.add_patch(Arc((cx, cy - 38 * s), 196 * s, 180 * s, theta1=180, theta2=360, lw=26 * s, color=hair, zorder=z + 2))
    for dx in (-40, 40):
        ax.add_patch(Ellipse((cx + dx * s, cy - 10 * s), 26 * s, 13 * s, fc="white", ec=line, lw=1.0, zorder=z + 2))
        ax.add_patch(Circle((cx + dx * s, cy - 10 * s), 5.5 * s, fc="#3b302a", zorder=z + 3))
        ax.plot([cx + (dx - 18) * s, cx + (dx + 18) * s], [cy - 32 * s, cy - 34 * s], color=hair, lw=3 * s, zorder=z + 2)
    ax.plot([cx, cx - 8 * s, cx + 4 * s], [cy - 2 * s, cy + 32 * s, cy + 34 * s], color=line, lw=1.6, zorder=z + 2)
    ax.add_patch(Arc((cx, cy + 52 * s), 60 * s, 26 * s, theta1=200, theta2=340, lw=2.2, color="#a8584e", zorder=z + 2))
