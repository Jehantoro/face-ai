"""Drawn pictures (no data) for the Week 5-7 report.
    .venv/Scripts/python.exe -X utf8 scripts/gdoc_diagrams.py [name ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gdoc_style import (AQUA, BLUE, GREY, INK, INK2, KIND, MUTED, ORANGE, RED, RULE, SHADE, SURFACE, VIOLET,  # noqa: E402
                        YELLOW, arrow, box, canvas, face, legend_kinds, pill, save, title)
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, Polygon, Circle   # noqa: E402

FIGS = {}
TEAL_T = "#12855c"


def fig(fn):
    FIGS[fn.__name__] = fn
    return fn


def lines_at(ax, x, y, items, size=14.5, gap=25, color=INK2, weight="normal", ha="left"):
    for i, t in enumerate(items):
        ax.text(x, y + gap * i, t, fontsize=size, color=color, fontweight=weight, va="center", ha=ha)


def wave(ax, x, y, w, h, color, cycles=4.5, phase=0.0, lw=2.4, sharp=False, z=5):
    t = np.linspace(0, 1, 300)
    s = np.sin(2 * np.pi * cycles * t + phase)
    if sharp:                                                            # a pulse-like wave: quick rise, slow fall
        s = np.sin(2 * np.pi * cycles * t + phase) + 0.35 * np.sin(4 * np.pi * cycles * t + phase + 0.6)
    s = s / np.abs(s).max()
    ax.plot(x + w * t, y - h / 2 * s, color=color, lw=lw, zorder=z, solid_capstyle="round")


# ------------------------------------------------------------------ Data processing part
@fig
def d1_pipeline():
    f, ax = canvas(7.75)
    title(ax, "From one video to the model inputs",
          "One pass over every frame of all 1,200 frontal videos. Everything here is fixed code: nothing learns.")
    box(ax, 40, 150, 170, 560, "plain", "One video", ["about 3 min", "≈5,400", "frames", "30 per s", "640 × 480"],
        headsize=18, size=15)
    box(ax, 245, 330, 205, 200, "fixed", "True time", ["the printed clock is", "read on each frame:", "one 30-per-second", "clock for all"],
        headsize=18, size=14.5, line_gap=25)
    box(ax, 485, 330, 205, 200, "fixed", "The face", ["YuNet finds the", "box and 5 points", "on every frame,", "smoothed in time"],
        headsize=18, size=14.5, line_gap=25)
    arrow(ax, (210, 430), (245, 430))
    arrow(ax, (450, 430), (485, 430))
    outs = [("24 sharp face crops", "256 × 256 px, over the 3 minutes", "→ face encoders: age, BMI, sex", BLUE),
            ("Face clips", "first minute, 72 × 72 px per frame", "→ PhysNet, RhythmFormer: pulse", TEAL_T),
            ("Skin colour", "forehead, cheeks, face; whole video", "→ trace network, POS, CHROM: pulse", TEAL_T),
            ("Upper-body motion", "shoulders and chest, up and down", "→ breathing numbers", INK2)]
    ys = [140, 258, 376, 494]
    for (head, l1, l2, col), y in zip(outs, ys):
        ax.add_patch(FancyBboxPatch((735, y), 430, 100, boxstyle="round,pad=0,rounding_size=14", fc="white", ec=RULE, lw=2.2, zorder=3))
        ax.text(755, y + 28, head, fontsize=17, fontweight="bold", color=INK, va="center", zorder=4)
        ax.text(755, y + 55, l1, fontsize=14, color=INK2, va="center", zorder=4)
        ax.text(755, y + 80, l2, fontsize=14, color=col, va="center", zorder=4, fontweight="semibold")
        arrow(ax, (690, 430), (735, y + 50))
    ax.add_patch(FancyBboxPatch((735, 622), 430, 100, boxstyle="round,pad=0,rounding_size=14", fc="#fff7f2", ec=ORANGE, lw=2.2, zorder=3))
    ax.text(755, 650, "Finger pulse (contact sensor)", fontsize=17, fontweight="bold", color=INK, va="center", zorder=4)
    ax.text(755, 677, "recorded with the video, same clock", fontsize=14, color=INK2, va="center", zorder=4)
    ax.text(755, 702, "→ only teaches the pulse networks", fontsize=14, color=ORANGE, va="center", zorder=4, fontweight="semibold")
    return save(f, "d1_pipeline")


@fig
def d3_folds():
    f, ax = canvas(7.9)
    title(ax, "Testing on people the model never saw",
          "Nested cross-validation over all 600 people. Both videos of a person always stay in the same group.")
    x0, wtot = 235, 925
    wb = wtot / 5
    # row A: the 600 people in 5 folds
    ax.text(40, 172, "600 people,", fontsize=15.5, fontweight="bold", color=INK, va="center")
    ax.text(40, 196, "dealt into 5 folds", fontsize=14.5, color=INK2, va="center")
    for i in range(5):
        ax.add_patch(FancyBboxPatch((x0 + i * wb + 4, 160), wb - 8, 48, boxstyle="round,pad=0,rounding_size=8", fc="#eceae5", ec="none"))
        ax.text(x0 + i * wb + wb / 2, 184, f"fold {i + 1}: 120 people", ha="center", va="center", fontsize=14, color=INK2)
    # row B: outer loop
    ax.text(40, 282, "Outer loop:", fontsize=15.5, fontweight="bold", color=INK, va="center")
    ax.text(40, 306, "scores", fontsize=14.5, color=INK2, va="center")
    for i in range(5):
        test = i == 2
        ax.add_patch(FancyBboxPatch((x0 + i * wb + 4, 270), wb - 8, 48, boxstyle="round,pad=0,rounding_size=8",
                                    fc="#fde6dc" if test else "#e3eefb", ec=ORANGE if test else BLUE, lw=2))
        ax.text(x0 + i * wb + wb / 2, 294, "test: 120 people" if test else "training", ha="center", va="center", fontsize=14,
                color=INK, fontweight="bold" if test else "normal")
    ax.text(x0 + 2.5 * wb, 342, "5 rounds: each fold is the test fold once; the other 480 people are the training people",
            ha="center", va="center", fontsize=14, color=INK2)
    # row C: inner loop inside the 480
    ax.text(40, 410, "Inner loop:", fontsize=15.5, fontweight="bold", color=INK, va="center")
    ax.text(40, 434, "chooses, inside", fontsize=14.5, color=INK2, va="center")
    ax.text(40, 458, "the 480 only", fontsize=14.5, color=INK2, va="center")
    wi = 925 / 4
    for i in range(4):
        chk = i == 1
        ax.add_patch(FancyBboxPatch((x0 + i * wi + 4, 398), wi - 8, 48, boxstyle="round,pad=0,rounding_size=8",
                                    fc="#fff4d6" if chk else "#e3eefb", ec=YELLOW if chk else BLUE, lw=2))
        ax.text(x0 + i * wi + wi / 2, 422, "inner check: 120" if chk else "inner training", ha="center", va="center",
                fontsize=14, color=INK, fontweight="bold" if chk else "normal")
    ax.text(x0 + 2 * wi, 470, "4 rounds: every candidate learns on 360 training people and is scored on the other 120",
            ha="center", va="center", fontsize=14, color=INK2)
    # row D: the flow inside one outer round
    steps = [("Score", ["every candidate", "on the inner", "checks"]),
             ("Choose", ["per target: the", "best inputs and", "model (or 2–3", "averaged)"]),
             ("Refit", ["the choice on", "all 480", "training people"]),
             ("Predict", ["the 120 test", "people, then", "compare with", "their labels"])]
    bx, bw = 235, 205
    for i, (h, ls) in enumerate(steps):
        x = bx + i * (bw + 35)
        box(ax, x, 510, bw, 150, "trained" if i < 3 else "plain", h, ls, headsize=17, size=14, line_gap=23, head_gap=28,
            tag="" if True else None)
        if i:
            arrow(ax, (x - 35, 585), (x, 585))
    ax.text(40, 585, "In each", fontsize=15.5, fontweight="bold", color=INK, va="center")
    ax.text(40, 609, "outer round", fontsize=14.5, color=INK2, va="center")
    # row E: repeats
    ax.add_patch(FancyBboxPatch((235, 690), 925, 80, boxstyle="round,pad=0,rounding_size=12", fc="white", ec=RULE, lw=1.8))
    ax.text(255, 716, "× 5 rounds: every person gets an answer from a model that never saw them.", fontsize=14.5, color=INK, va="center")
    ax.text(255, 746, "× 3 shuffles: the 600 people are dealt into folds 3 times, and every result is the average.",
            fontsize=14.5, color=INK, va="center")
    return save(f, "d3_folds")


@fig
def d4_cuts():
    f, ax = canvas(8.4)
    title(ax, "What is cut out of every video",
          "The face box is found on every frame; everything else is placed from it. The face is a drawing.")
    fx, fy, fw, fh = 50, 140, 560, 420
    ax.add_patch(Rectangle((fx, fy), fw, fh, fc="#e9ece8", ec=RULE, lw=2, zorder=1))
    ax.add_patch(Rectangle((fx, fy), fw, 50, fc="#2b2b2b", ec="none", zorder=2))
    ax.text(fx + 14, fy + 25, "2023-05-14 10:42:07.233", fontsize=13, color="white", va="center", family="Consolas", zorder=3)
    cx, cy, s = 330, 366, 0.74
    face(ax, cx, cy, s=s, clip=(fx, fy, fw, fh))
    ax.add_patch(Rectangle((fx, fy), fw, fh, fc="none", ec=RULE, lw=2, zorder=21))
    bw, bh = 175 * s, 215 * s
    side = max(bw, bh)
    bcy = cy + 4
    ax.add_patch(Rectangle((cx - bw / 2, bcy - bh / 2), bw, bh, fc="none", ec=INK, lw=2.2, ls=(0, (5, 4)), zorder=9))
    for k, col, ls, lw in ((1.4, BLUE, "-", 3), (2.0, BLUE, (0, (7, 5)), 2.4), (1.5, AQUA, (0, (2, 3)), 3)):
        L = side * k
        ax.add_patch(Rectangle((cx - L / 2, bcy - L / 2), L, L, fc="none", ec=col, lw=lw, ls=ls, zorder=10))
    for (dx, dy, w, h) in ((0, -58, 70, 26), (-44, 22, 32, 30), (44, 22, 32, 30)):
        ax.add_patch(Rectangle((cx + dx * s - w * s / 2, cy + dy * s - h * s / 2), w * s, h * s, fc=ORANGE, alpha=0.35,
                               ec=ORANGE, lw=2, zorder=11))
    for (dx, dy) in ((-150, 192), (150, 192), (0, 222)):
        ax.add_patch(Rectangle((cx + dx * s - 36, cy + dy * s - 20), 72, 40, fc=VIOLET, alpha=0.18, ec=VIOLET, lw=2.2, zorder=22))
    for (dx, dy) in ((-150, 192), (150, 192)):
        off = 46 if dx > 0 else -46
        arrow(ax, (cx + dx * s + off, cy + dy * s + 16), (cx + dx * s + off, cy + dy * s - 16),
              color=VIOLET, lw=2, ms=13, style="<|-|>", zorder=23)
    items = [(INK, (0, (5, 4)), 2.2, "Face box (YuNet)", "found on every frame, then smoothed"),
             (BLUE, "-", 3, "Tight crop: 1.4 × the box", "24 per video, 256 × 256 px"),
             (BLUE, (0, (7, 5)), 2.4, "Wide crop: 2 × the box", "the same 24 moments"),
             (AQUA, (0, (2, 3)), 3, "Clip: 1.5 × the box", "first minute, 72 × 72 px, clock blanked"),
             (ORANGE, None, 0, "Skin regions", "forehead, both cheeks (+ the whole face)"),
             (VIOLET, None, 0, "Motion windows", "both shoulders and the chest")]
    y = 168
    for col, ls, lw, h, sub in items:
        if ls is None:
            ax.add_patch(Rectangle((650, y - 14), 50, 28, fc=col, alpha=0.3, ec=col, lw=2))
        else:
            ax.plot([650, 700], [y, y], color=col, lw=lw, ls=ls)
        ax.text(720, y - 9, h, fontsize=15.5, fontweight="bold", color=INK, va="center")
        ax.text(720, y + 17, sub, fontsize=14, color=INK2, va="center")
        y += 66
    t0, t1 = 150, 1150
    ax.text(40, 616, "Over the 3 minutes of one video", fontsize=15.5, fontweight="bold", color=INK, va="center", zorder=30)
    for lab, yy in (("24 crops", 660), ("clips", 702), ("traces", 744)):
        ax.text(40, yy, lab, fontsize=14, color=INK2, va="center", zorder=30)
    for i in range(24):
        x = t0 + (t1 - t0) * (i + 0.5) / 24
        ax.add_patch(Rectangle((x - 8, 649), 16, 22, fc=BLUE, ec="none", zorder=30))
    ax.add_patch(Rectangle((t0, 691), (t1 - t0) / 3, 22, fc="#d6f2e6", ec=AQUA, lw=1.5, zorder=30))
    ax.text(t0 + (t1 - t0) / 6, 702, "first minute only", ha="center", va="center", fontsize=13.5, color=INK, zorder=31)
    ax.add_patch(Rectangle((t0, 733), t1 - t0, 22, fc="#eceae5", ec="none", zorder=30))
    ax.text((t0 + t1) / 2, 744, "skin colour, landmarks and motion: every frame of the whole video", ha="center",
            va="center", fontsize=13.5, color=INK2, zorder=31)
    for sec in (0, 60, 120, 180):
        x = t0 + (t1 - t0) * sec / 180
        ax.plot([x, x], [762, 770], color=MUTED, lw=1.5, zorder=30)
        ax.text(x, 788, f"{sec} s", ha="center", va="center", fontsize=13, color=MUTED, zorder=30)
    return save(f, "d4_cuts")


# ------------------------------------------------------------------ Model part
@fig
def m1_overview():
    f, ax = canvas(9.0)
    title(ax, "The final model in one picture",
          "One 3-minute video in, 10 answers out. The tag on each part says how it learned (key below).")
    box(ax, 40, 205, 130, 520, "plain", "One", ["video,", "3 min,", "every", "frame"], headsize=18, size=15)
    rows = [
        (205, "FACE  ·  age, BMI, sex", BLUE,
         [("fixed", "24 crops", ["wide + tight,", "256 × 256 px"]),
          ("frozen", "Face encoders", ["CLIP ViT-L/14 and", "4 others; average", "of the 24 crops"])]),
        (405, "PULSE  ·  stiffness, breathing", TEAL_T,
         [("fixed", "Clips + colour", ["first-minute clips,", "4-region traces"]),
          ("finger", "3 pulse networks", ["trace network,", "PhysNet,", "RhythmFormer"]),
          ("fixed", "Rhythm numbers", ["heart rate, trend,", "quality, beat-to-", "beat variability"])]),
        (605, "CLASSIC SIGNALS  ·  breathing", INK2,
         [("fixed", "Colour + motion", ["skin colour,", "shoulders, chest"]),
          ("fixed", "POS · CHROM", ["classic pulse and", "breathing numbers"])]),
    ]
    xs, w = [205, 425, 645], 195
    for y, lab, col, bxs in rows:
        ax.text(205, y - 42, lab, fontsize=14.5, fontweight="bold", color=col, va="center")
        for i, (kind, h, ls) in enumerate(bxs):
            box(ax, xs[i], y, w, 125, kind, h, ls, headsize=16, size=13.5, line_gap=22, head_gap=27)
            if i:
                arrow(ax, (xs[i - 1] + w, y + 62), (xs[i], y + 62))
        arrow(ax, (170, 465), (205, y + 62))
        arrow(ax, (xs[len(bxs) - 1] + w, y + 62), (875, 465))
    box(ax, 875, 300, 165, 330, "trained", "Specialists", ["one per target,", "chosen by the", "inner loop from", "ridge, PLS,", "kernel ridge,", "trees, Face_AI", "head or the", "median guess"],
        headsize=17, size=13.5, line_gap=24, head_gap=30)
    outs = [(215, ["Age", "BMI", "Sex"], True), (420, ["Stiffness", "Breathing"], True),
            (580, ["SpO2, temp.,", "HbA1c, chol.,", "stress ≈ guess"], False)]
    for y, ls, strong in outs:
        hh = 24 * len(ls) + 26
        ax.add_patch(FancyBboxPatch((1065, y), 125, hh, boxstyle="round,pad=0,rounding_size=12", fc="white", ec=RULE, lw=2))
        for i, t in enumerate(ls):
            ax.text(1127, y + 25 + 24 * i, t, ha="center", va="center", fontsize=13.5 if strong else 12.5,
                    color=INK if strong else INK2, fontweight="bold" if strong else "normal")
        arrow(ax, (1040, 465), (1065, y + hh / 2))
    keys = [("fixed", "our code, no learning"), ("frozen", "pretrained by others, never changed"),
            ("finger", "learned here, the finger pulse as teacher"), ("trained", "learned here from the health labels")]
    for i, (k, txt) in enumerate(keys):
        x, y = 40 + (i % 2) * 590, 790 + (i // 2) * 44
        pill(ax, x + 70, y, KIND[k]["label"], KIND[k]["pill"], size=12.5)
        ax.text(x + 165, y, txt, fontsize=14.5, color=INK2, va="center")
    return save(f, "m1_overview")


@fig
def m2_encoder():
    f, ax = canvas(7.6)
    title(ax, "How a frozen face encoder turns a crop into numbers",
          "CLIP ViT-L/14, the main face encoder: 304 million numbers, trained by OpenAI and never changed by us.")
    gx, gy, gs = 50, 165, 220
    ax.add_patch(Rectangle((gx, gy), gs, gs, fc="#e9ece8", ec=RULE, lw=2))
    face(ax, gx + gs / 2, gy + gs / 2 - 6, s=0.55, clip=(gx, gy, gs, gs))
    for i in range(17):
        ax.plot([gx + gs * i / 16] * 2, [gy, gy + gs], color="white", lw=0.9, alpha=0.9, zorder=21)
        ax.plot([gx, gx + gs], [gy + gs * i / 16] * 2, color="white", lw=0.9, alpha=0.9, zorder=21)
    ax.add_patch(Rectangle((gx, gy), gs, gs, fc="none", ec=RULE, lw=2, zorder=22))
    lines_at(ax, gx, gy + gs + 30, ["a 224 × 224 px crop,", "cut into 16 × 16", "patches of 14 px"], size=14.5, gap=25)
    steps = [("Patch → token", ["each patch's 588", "numbers become", "1,024: a token"]),
             ("+ position", ["+ 1 summary", "token = 257", "tokens"]),
             ("24 blocks", ["self-attention", "(16 heads) + a", "small network"]),
             ("Summary", ["token: 1,024", "numbers + its", "projection: 768"])]
    x = 310
    for i, (h, ls) in enumerate(steps):
        box(ax, x, 190, 195, 170, "frozen", h, ls, headsize=16.5, size=14, line_gap=24, head_gap=30, tag="FROZEN" if i == 0 else "")
        arrow(ax, (x - 28, 275), (x, 275))
        x += 218
    ax.text(310, 400, "1,792 numbers per crop. The 24 crops of a video are averaged,", fontsize=14.5, color=INK, va="center")
    ax.text(310, 427, "so every video becomes one list of 1,792 numbers.", fontsize=14.5, color=INK, va="center")
    ax.add_patch(FancyBboxPatch((40, 490), 1120, 250, boxstyle="round,pad=0,rounding_size=14", fc="white", ec=RULE, lw=1.8))
    ax.text(62, 522, "How CLIP learned to describe pictures (long before we used it)", fontsize=16, fontweight="bold", color=INK, va="center")
    lines_at(ax, 62, 562, ["OpenAI showed it 400 million pictures from the internet, each with the caption written for it.",
                           "It learned to put each picture next to its own caption, among many wrong captions.",
                           "So its numbers carry what people write about pictures: young or old, heavy or thin, man or woman.",
                           "That is a likely reason why CLIP reads age, BMI and sex better than DINOv2 (pictures only).",
                           "Nobody named the 1,792 numbers: the specialists learn which combinations go with each target."],
             size=14.5, gap=33)
    return save(f, "m2_encoder")


@fig
def m3_teacher():
    f, ax = canvas(8.1)
    title(ax, "Teaching a network to read the pulse, with the finger as teacher",
          "The finger sensor is only used while training. At prediction time the network sees the video alone.")
    # top flow
    box(ax, 40, 170, 200, 130, "fixed", "A short window", ["5–10 s of clip frames", "or colour traces"], headsize=16.5, size=14)
    box(ax, 285, 170, 210, 130, "finger", "Pulse network", ["trace net, PhysNet", "or RhythmFormer"], headsize=16.5, size=14)
    arrow(ax, (240, 235), (285, 235))
    ax.add_patch(FancyBboxPatch((540, 160), 255, 70, boxstyle="round,pad=0,rounding_size=10", fc="white", ec=AQUA, lw=2))
    ax.text(552, 178, "camera wave", fontsize=13.5, color=TEAL_T, fontweight="bold", va="center")
    wave(ax, 555, 208, 225, 30, AQUA, cycles=6, sharp=True)
    ax.add_patch(FancyBboxPatch((540, 252), 255, 70, boxstyle="round,pad=0,rounding_size=10", fc="#fff7f2", ec=ORANGE, lw=2))
    ax.text(552, 270, "finger wave", fontsize=13.5, color=ORANGE, fontweight="bold", va="center")
    wave(ax, 555, 300, 225, 30, ORANGE, cycles=6, sharp=True, phase=0.25)
    arrow(ax, (495, 235), (540, 197))
    box(ax, 840, 170, 320, 150, "plain", "How wrong?", ["1 − how well the waves rise and", "fall together (best of ±3 frames)", "+ 0.1 × mismatch of their rhythm"],
        headsize=16.5, size=14, line_gap=24)
    arrow(ax, (795, 197), (840, 230))
    arrow(ax, (795, 288), (840, 262))
    arrow(ax, (1000, 320), (390, 300), color=AQUA, lw=2.4, rad=-0.18, ls=(0, (6, 4)))
    ax.text(700, 382, "backpropagation and AdamW nudge every weight to make the waves agree", ha="center", fontsize=14,
            color=TEAL_T, va="center")
    # cross-fitting
    ax.text(40, 450, "Cross-fitting: nobody's pulse comes from a network that saw them", fontsize=16, fontweight="bold", color=INK, va="center")
    x0, wb = 300, 172
    for k in range(5):
        y = 490 + k * 46
        ax.text(40, y + 17, f"network {k + 1}", fontsize=14, color=INK2, va="center")
        for i in range(5):
            me = i == k
            ax.add_patch(FancyBboxPatch((x0 + i * wb + 3, y), wb - 6, 34, boxstyle="round,pad=0,rounding_size=6",
                                        fc="#fde6dc" if me else "#e9f7f1", ec=ORANGE if me else AQUA, lw=1.6))
            ax.text(x0 + i * wb + wb / 2, y + 17, "reads the pulse" if me else "learns", ha="center", va="center",
                    fontsize=13.5, color=INK if me else TEAL_T, fontweight="bold" if me else "normal")
    ax.text(x0 + 2.5 * wb, 490 + 5 * 46 + 12, "the 5 folds of people (shuffle 1)", ha="center", fontsize=13.5, color=MUTED, va="center")
    lines_at(ax, 40, 770, ["Each network learns from the clean finger recordings of 4 folds (10% of those people held back to pick the best round),",
                           "then reads the pulse of the fifth fold. No health label is used anywhere in this step."], size=14.5, gap=27)
    return save(f, "m3_teacher")


@fig
def m4_pulse_nets():
    f, ax = canvas(8.4)
    title(ax, "The three pulse networks",
          "Each one turns video into a pulse wave, one value per frame. All three were taught by the finger sensor.")
    cols = [("Trace network", "339,969 weights · ours, from scratch",
             [("Input", "20 signals over the whole video:", "R, G, B of 4 skin regions + POS", "and CHROM per region"),
              ("Layers", "1-D convolutions along time;", "8 blocks that look 1, 2, 4, 8", "samples apart (twice), with skips"),
              ("Training", "40 rounds of random 10-s", "windows; left and right cheek", "swapped at random")],
             "1.23 bpm · 94% within 5 bpm"),
            ("PhysNet", "769,777 weights · ours, from scratch",
             [("Input", "160 frames of 72 × 72 px:", "each frame + its difference", "from the frame before"),
              ("Layers", "3-D convolutions over space and", "time; pooling shrinks the image,", "then time is stretched back"),
              ("Training", "6 rounds of 160-frame windows;", "full precision (a pulse is only", "0.1–1% of the brightness)")],
             "1.09 bpm · 95% within 5 bpm"),
            ("RhythmFormer", "3,326,017 weights · published",
             [("Input", "160 frames, enlarged to", "128 × 128 px", ""),
              ("Layers", "a convolution stem, then attention", "blocks that compare frames at", "several time scales"),
              ("Training", "started from weights trained on", "PURE (another dataset), then", "fine-tuned here for 12 rounds")],
             "0.99 bpm · 95% within 5 bpm")]
    for i, (name, sub, parts, res) in enumerate(cols):
        x = 40 + i * 380
        ax.add_patch(FancyBboxPatch((x, 140), 350, 650, boxstyle="round,pad=0,rounding_size=16", fc=KIND["finger"]["fill"],
                                    ec=AQUA, lw=2.2))
        pill(ax, x + 175, 140, "FINGER-TAUGHT", TEAL_T)
        ax.text(x + 175, 185, name, ha="center", fontsize=19, fontweight="bold", color=INK, va="center")
        ax.text(x + 175, 215, sub, ha="center", fontsize=13.5, color=INK2, va="center")
        y = 255
        for h, a, b, c in parts:
            ax.add_patch(FancyBboxPatch((x + 18, y), 314, 128, boxstyle="round,pad=0,rounding_size=10", fc="white", ec=RULE, lw=1.6))
            ax.text(x + 34, y + 25, h, fontsize=14.5, fontweight="bold", color=TEAL_T, va="center")
            for j, t in enumerate((a, b, c)):
                ax.text(x + 34, y + 54 + 23 * j, t, fontsize=13.5, color=INK2, va="center")
            y += 145
            if h != "Training":
                arrow(ax, (x + 175, y - 17), (x + 175, y - 2), color=GREY, lw=2, ms=14)
        ax.text(x + 175, 735, "Heart-rate error vs the finger", ha="center", fontsize=13.5, color=INK2, va="center")
        ax.text(x + 175, 762, res, ha="center", fontsize=15.5, fontweight="bold", color=INK, va="center")
    return save(f, "m4_pulse_nets")


@fig
def m5_specialists():
    f, ax = canvas(8.2)
    title(ax, "How each target gets its specialist",
          "Inside every outer round, using the 480 training people only. Example: arterial stiffness, shuffle 1, round 1.")
    # routes
    box(ax, 40, 150, 245, 300, "plain", "44 routes", ["a route = one set of", "numbers, e.g.", "· CLIP-L, tight crops", "· PhysNet rhythm", "· classic signals +", "   every pulse network", "· predicted age/sex/BMI"],
        headsize=17, size=14, line_gap=26, head_gap=30, align="left")
    cands = [("Ridge", "a straight line, kept small"), ("PLS", "a line on a few summaries"),
             ("Kernel ridge", "a line that can bend"), ("Boosted trees", "150 small decision trees"),
             ("Face_AI head", "a small network (faces)"), ("Median guess", "the middle value: abstain")]
    ax.add_patch(FancyBboxPatch((320, 150), 440, 300, boxstyle="round,pad=0,rounding_size=14", fc=KIND["trained"]["fill"], ec=BLUE, lw=2.2))
    pill(ax, 540, 150, "TRAINED", BLUE)
    ax.text(340, 182, "6 candidate models per route", fontsize=16, fontweight="bold", color=INK, va="center")
    for i, (h, d) in enumerate(cands):
        ax.text(342, 222 + 37 * i, h, fontsize=14.5, fontweight="bold", color=INK, va="center")
        ax.text(505, 222 + 37 * i, d, fontsize=13.5, color=INK2, va="center")
    arrow(ax, (285, 300), (320, 300))
    box(ax, 795, 150, 365, 300, "plain", "Inner loop picks", ["each candidate learns on 3 inner", "groups and is scored on the 4th,", "4 times; the lowest error wins", "→ every route gets its best", "model and an honest score"],
        headsize=17, size=14, line_gap=27, head_gap=32)
    arrow(ax, (760, 300), (795, 300))
    # ensemble
    ax.text(40, 500, "Then the routes compete", fontsize=16.5, fontweight="bold", color=INK, va="center")
    opts = [("best route alone", ["PhysNet rhythm + shape", "trees · inner error 1.914"]),
            ("average of the best 2", ["+ PhysNet rhythm", "trees · inner error 1.926"]),
            ("average of the best 3", ["+ RhythmFormer rhythm", "trees · inner error 1.934"])]
    for i, (h, ls) in enumerate(opts):
        box(ax, 40 + i * 290, 530, 265, 120, "plain", h, ls, headsize=15.5, size=13.5, line_gap=23, head_gap=28)
    box(ax, 920, 530, 240, 120, "trained", "Kept: best 3", ["averaged, inner", "error 1.906"], headsize=15.5, size=13.5,
        line_gap=23, head_gap=28, tag="")
    arrow(ax, (885, 590), (920, 590))
    lines_at(ax, 40, 700, ["The winner is refitted on all 480 training people and predicts the 120 test people. This happens",
                           "15 times per target (5 rounds × 3 shuffles), so a target can get different routes in different rounds.",
                           "If nothing beats the median guess on the inner checks, the median guess is used: the target abstains."],
             size=14.5, gap=28)
    return save(f, "m5_specialists")


@fig
def m6_face_head():
    f, ax = canvas(6.7)
    title(ax, "Face_AI's task head, inside Face_AI_2",
          "The small network that gave Face_AI its age edge, now one of the candidates for age, BMI and sex.")
    box(ax, 40, 165, 170, 150, "fixed", "24 crops", ["of one video"], headsize=17, size=14)
    box(ax, 245, 165, 190, 150, "frozen", "CLIP-L/14", ["1,792 numbers", "per crop"], headsize=17, size=14)
    arrow(ax, (210, 240), (245, 240))
    ax.add_patch(FancyBboxPatch((470, 150), 300, 85, boxstyle="round,pad=0,rounding_size=12", fc="#fff4d6", ec=YELLOW, lw=2))
    ax.text(490, 176, "While training", fontsize=15, fontweight="bold", color=INK, va="center")
    ax.text(490, 206, "average a random 8 of the 24", fontsize=14, color=INK2, va="center")
    ax.add_patch(FancyBboxPatch((470, 250), 300, 85, boxstyle="round,pad=0,rounding_size=12", fc="white", ec=RULE, lw=2))
    ax.text(490, 276, "When predicting", fontsize=15, fontweight="bold", color=INK, va="center")
    ax.text(490, 306, "average all 24 crops", fontsize=14, color=INK2, va="center")
    arrow(ax, (435, 240), (470, 192))
    arrow(ax, (435, 240), (470, 292))
    ax.add_patch(FancyBboxPatch((805, 140), 355, 210, boxstyle="round,pad=0,rounding_size=14", fc=KIND["trained"]["fill"], ec=BLUE, lw=2.2))
    pill(ax, 982, 140, "TRAINED", BLUE)
    lines_at(ax, 825, 182, ["1,792 numbers, scaled", "→ hide 30% at random (dropout)", "→ 64 neurons, ReLU", "→ 1 answer"], size=14.5, gap=33)
    ax.text(825, 322, "114,817 weights per head", fontsize=14, color=BLUE, fontweight="bold", va="center")
    arrow(ax, (770, 192), (805, 230))
    arrow(ax, (770, 292), (805, 260))
    rows = [("One head per target", "age (in log years), BMI (log) or sex (a score turned into a probability)"),
            ("Learning", "Adam, all training videos at once, up to 400 rounds"),
            ("Stopping", "15% of the training people are held back; stop after 30 rounds without improvement"),
            ("3 seeds", "trained 3 times from different random starts; the 3 answers are averaged"),
            ("Aimed at the median", "the answers are shifted so their middle matches the training people's middle")]
    y = 410
    for h, d in rows:
        ax.text(40, y, h, fontsize=15, fontweight="bold", color=INK, va="center")
        ax.text(300, y, d, fontsize=14.5, color=INK2, va="center")
        y += 52
    return save(f, "m6_face_head")


if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        print(FIGS[n]())
