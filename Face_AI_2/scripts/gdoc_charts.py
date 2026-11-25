"""Charts (from saved results) for the Week 5-7 report. Needs reports/gdoc_week5-7/numbers.json and
recordings.csv (scripts/gdoc_report_numbers.py). Nothing is trained.
    .venv/Scripts/python.exe -X utf8 scripts/gdoc_charts.py [name ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt                                             # noqa: E402
from matplotlib.lines import Line2D                                          # noqa: E402
from gdoc_style import (AQUA, BLUE, GREY, GRID, INK, INK2, MUTED, ORANGE, RED, RULE, SHADE, SURFACE, VIOLET,  # noqa: E402
                        YELLOW, save, setup)

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports" / "gdoc_week5-7"
NUM = json.loads((REP / "numbers.json").read_text(encoding="utf-8"))
REC = pd.read_csv(REP / "recordings.csv", dtype={"patient_id": str})
W = 12.0
FIGS = {}
TG = ["saturation", "temperature", "glycated_hemoglobin", "cholesterol", "respiratory", "rigidity", "stress", "age", "bmi"]
NAME = {"saturation": "Oxygen saturation", "temperature": "Temperature", "glycated_hemoglobin": "HbA1c",
        "cholesterol": "Cholesterol", "respiratory": "Breathing rate", "rigidity": "Arterial stiffness",
        "stress": "Stress (PSM-25)", "age": "Age", "bmi": "BMI", "sex": "Recorded sex"}
UNIT = {"saturation": "%", "temperature": "°C", "glycated_hemoglobin": "%", "cholesterol": "mmol/L",
        "respiratory": "breaths/min", "rigidity": "index", "stress": "score", "age": "years", "bmi": "kg/m²"}
CAUGHT, MISSED, FALSE, CLEARED = BLUE, RED, YELLOW, "#cfcdc7"


def fig_(fn):
    FIGS[fn.__name__] = fn
    return fn


def header(fig, title, sub=None, top=None, gap=0.95):
    """Title and grey subtitle at the top-left, positioned in inches; the panels start below them."""
    H = fig.get_figheight()
    subs = [] if not sub else (list(sub) if isinstance(sub, (list, tuple)) else [sub])
    fig.text(0.033, 1 - 0.16 / H, title, fontsize=24, fontweight="bold", color=INK, va="top")
    for i, line in enumerate(subs):
        fig.text(0.033, 1 - (0.62 + 0.29 * i) / H, line, fontsize=15, color=INK2, va="top")
    fig.subplots_adjust(top=1 - (0.62 + 0.29 * len(subs) + gap) / H)


def panel_title(ax, t, sub=None):
    ax.set_title(t, loc="left", fontsize=16, fontweight="bold", color=INK, pad=24 if sub else 10)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=12.5, color=INK2, va="bottom")


def note(ax, lines, loc="upper left", size=12.5):
    x, y, ha, va = {"upper left": (0.03, 0.97, "left", "top"), "lower right": (0.97, 0.03, "right", "bottom"),
                    "upper right": (0.97, 0.97, "right", "top"), "lower left": (0.03, 0.03, "left", "bottom")}[loc]
    ax.text(x, y, "\n".join(lines), transform=ax.transAxes, ha=ha, va=va, fontsize=size, color=INK, linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.3", fc="white", ec=GRID, lw=1.2), zorder=10)


# ------------------------------------------------------------------ data part
@fig_
def d2_labels():
    setup()
    fig, axes = plt.subplots(2, 5, figsize=(W, 6.9), dpi=200)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.80, bottom=0.08, hspace=0.75, wspace=0.32)
    header(fig, "What the ten labels look like",
           ["All 1,200 recordings. The dark line is the median: the 'median guess' always answers it. The number under",
            "each title is how far that guess is off on average (its MAE), the error every model has to beat."], top=0.985)
    L = NUM["labels"]
    R = NUM["results"]["v2_video"]
    steps = {"saturation": 1, "temperature": 0.1, "respiratory": 1}
    for ax, t in zip(axes.ravel(), TG + ["sex"]):
        if t == "sex":
            n_w, n_m = NUM["data"]["women"], NUM["data"]["men"]
            ax.bar([0, 1], [n_w, n_m], width=0.6, color=BLUE)
            ax.set_xticks([0, 1], ["female", "male"])
            ax.set_ylim(0, 470)
            for i, v in enumerate((n_w, n_m)):
                ax.text(i, v + 12, f"{v}", ha="center", fontsize=12.5, color=INK)
            panel_title(ax, NAME[t], "people · guess: female")
            ax.grid(axis="x", visible=False)
            continue
        y = REC[f"y_{t}"].to_numpy()
        if t in steps:
            s = steps[t]
            lo, hi = np.floor(y.min() / s) * s, np.ceil(y.max() / s) * s
            bins = np.arange(lo - s / 2, hi + s, s)
        else:
            lo, hi = np.percentile(y, [0, 99.5])
            bins = np.linspace(lo, hi, 26)
        ax.hist(np.clip(y, bins[0], bins[-1]), bins=bins, color=BLUE, rwidth=0.82)
        ax.axvline(L[t]["median"], color=INK, lw=2)
        g = R[t]["guess"]
        u = {"respiratory": "/min", "rigidity": "", "stress": ""}.get(t, " " + UNIT[t])
        panel_title(ax, NAME[t], f"guess off by {g:.3g}{u}")
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=12)
        if t == "saturation":
            ax.set_xlim(85.5, 99.5)
        if t == "age":
            ax.text(0.97, 0.9, f"{NUM['data']['age_18_24']} of 600\naged 18–24", transform=ax.transAxes, ha="right",
                    va="top", fontsize=12, color=INK2)
        if t == "respiratory":
            ax.text(0.98, 0.95, f"mean before\nexercise {L[t]['mean_before']:.1f},\nafter {L[t]['mean_after']:.1f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=11.5, color=INK2)
        ax.set_xlabel(UNIT[t], fontsize=12.5)
    return save(fig, "d2_labels")


@fig_
def d5_signals():
    from scipy.signal import butter, filtfilt, welch
    import v2_core as C
    from v2_features import pos
    setup()
    key = "1107_FullHDwebcam_after"
    i = int(np.where(C.df["key"] == key)[0][0])
    T = np.load(C.CACHE2 / "traces" / f"{key}.npz")
    rgb, mot, valid = T["rgb"].astype(float), T["motion"].astype(float), T["valid"]
    P = np.load(C.CACHE2 / "ppg_full.npz")["ppg"][i]
    LAG = 2
    fin = np.concatenate([np.repeat(P[:1], LAG), P[:-LAG]])
    fs = 30.0

    def bp(x, lo, hi):
        b, a = butter(3, [lo / (fs / 2), hi / (fs / 2)], "band")
        x = np.where(np.isfinite(x), x, np.nanmedian(x))
        return filtfilt(b, a, x)

    t = np.arange(len(rgb)) / fs
    fig, axes = plt.subplots(4, 1, figsize=(W, 10.6), dpi=200, gridspec_kw=dict(height_ratios=[1, 1, 1, 1.25]))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.885, bottom=0.06, hspace=0.75)
    header(fig, "The signals of one recording",
           ["Participant 1107, just after the squats (finger signal clean). Only the finger trace comes from a contact sensor;",
            "everything else is read from the video. A pulse changes the skin colour by well under 1%."], top=0.988)
    s0, s1 = int(20 * fs), int(30 * fs)
    g = rgb[:, 1]
    ma = pd.Series(g).rolling(30, center=True, min_periods=5).mean().to_numpy()
    ax = axes[0]
    ax.plot(t[s0:s1], 100 * (g[s0:s1] / ma[s0:s1] - 1), color=GREY, lw=1.8)
    panel_title(ax, "Raw colour: the green of the forehead", "change from its own 1-second average, in % (seconds 20–30)")
    ax.set_ylabel("%")
    ax = axes[1]
    w = bp(pos(rgb[:, 0:3]), 0.7, 3.5)
    f = bp(fin, 0.7, 3.5)
    z = lambda x: (x - x[s0:s1].mean()) / x[s0:s1].std()
    ax.plot(t[s0:s1], z(f)[s0:s1], color=ORANGE, lw=2.2, label="finger sensor")
    ax.plot(t[s0:s1], z(w)[s0:s1], color=BLUE, lw=1.8, label="camera: POS on the forehead (classic)")
    panel_title(ax, "The pulse: classic camera method vs the finger", "both filtered to 42–210 beats per minute and scaled")
    ax.legend(loc="upper right", ncol=2, fontsize=12.5, bbox_to_anchor=(1, 1.32))
    ax.set_yticks([])
    ax = axes[2]
    m = bp(mot[:, 2], 0.13, 0.6)
    s2, s3 = 0, int(60 * fs)
    ax.plot(t[s2:s3], m[s2:s3], color=VIOLET, lw=1.8)
    panel_title(ax, "Breathing: the chest moving up and down", "the first minute, filtered to 8–36 breaths per minute")
    ax.set_yticks([])
    ax = axes[3]

    def hr_track(x, win=20, step=5):
        out_t, out_h = [], []
        for st in np.arange(0, len(x) / fs - win, step):
            a, b = int(st * fs), int((st + win) * fs)
            seg = x[a:b]
            if np.isfinite(seg).mean() < 0.9:
                continue
            seg = bp(seg, 0.7, 3.5)
            ff, pp = welch(seg, fs, nperseg=len(seg), nfft=8192)
            k = (ff > 0.7) & (ff < 3.5)
            out_t.append(st + win / 2)
            out_h.append(60 * ff[k][np.argmax(pp[k])])
        return np.array(out_t), np.array(out_h)

    tf, hf = hr_track(P)
    wave = np.load(C.CACHE2 / "rppg" / "tracenet" / "wave_full.npy")[i]
    tt, ht = hr_track(wave)
    tc, hc = hr_track(pos(rgb[:, 0:3]))
    ax.plot(tc, hc, color=GREY, lw=1.6, marker="o", ms=4, label="camera: classic POS")
    ax.plot(tt, ht, color=AQUA, lw=2.4, label="camera: trace network")
    ax.plot(tf, hf, color=ORANGE, lw=2.2, ls=(0, (5, 3)), label="finger sensor")
    panel_title(ax, "Heart rate over the whole video", "20-second windows, every 5 seconds: the heart slowing down after the squats")
    ax.set_xlabel("seconds since the start of the video")
    ax.set_ylabel("beats per minute")
    ax.legend(loc="lower left", ncol=3, fontsize=12.5, frameon=True, facecolor="white", edgecolor=GRID)
    for a in axes[:3]:
        a.set_xlabel("seconds")
    return save(fig, "d5_signals")


# ------------------------------------------------------------------ results part
@fig_
def r1_training():
    setup()
    nets = [("tracenet", "Trace network", AQUA), ("physnet_diff", "PhysNet", BLUE), ("rhythmformer_pure", "RhythmFormer (fine-tuned)", VIOLET)]
    fig, axes = plt.subplots(3, 3, figsize=(W, 9.6), dpi=200, sharex="col")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.07, hspace=0.42, wspace=0.28)
    header(fig, "How the pulse networks learned",
           ["Each line is one of the 5 cross-fitting networks. Training loss on the training windows; wave agreement and heart-rate",
            "error on the 10% of training people held back (60–80 recordings). The dot marks the round each network kept."], top=0.99)
    for c, (m, name, col) in enumerate(nets):
        L = pd.DataFrame(NUM["pulse_nets"][m]["log"])
        kept = {int(k): v for k, v in NUM["pulse_nets"][m]["kept_epoch"].items()}
        for r, (field, lab) in enumerate((("loss", "training loss"), ("val_wave_r", "wave r with the finger"),
                                          ("val_hr_mae", "heart-rate error (bpm)"))):
            ax = axes[r, c]
            for fold, d in L.groupby("fold"):
                d = d.sort_values("epoch")
                ax.plot(d.epoch, d[field], color=col, lw=1.8, alpha=0.75, marker="o", ms=3.5)
                kk = d[d.epoch == kept[int(fold)]]
                ax.plot(kk.epoch, kk[field], "o", ms=9, mfc="white", mec=INK, mew=1.6, zorder=5)
            if r == 0:
                panel_title(ax, name, f"{NUM['pulse_nets'][m]['params']:,} weights · {NUM['pulse_nets'][m]['epochs_run']} rounds")
            if c == 0:
                ax.set_ylabel(lab)
            if r == 2:
                ax.set_xlabel("training round (epoch)")
            if field == "val_wave_r":
                ax.set_ylim(0.7, 0.95)
            if field == "val_hr_mae":
                ax.set_ylim(0, 3.2)
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    return save(fig, "r1_training")


@fig_
def r2_camera_hr():
    setup()
    hr = pd.read_csv(ROOT / "results" / "v2" / "camera_hr_first_minute.csv")
    cols = [c for c in hr.columns if c not in ("key", "patient_id", "step", "finger")]
    short = ["v1: classic, first minute", "v2 classic: 4 regions", "v2 trace network",
             "v2 PhysNet", "RhythmFormer as published (no training here)", "RhythmFormer, fine-tuned"]
    colors = [GREY, GREY, AQUA, BLUE, ORANGE, VIOLET]
    fig, axes = plt.subplots(2, 3, figsize=(W, 8.6), dpi=200)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.85, bottom=0.08, hspace=0.42, wspace=0.25)
    header(fig, "Camera heart rate vs the finger sensor",
           ["All 1,200 recordings, first minute, the same heart-rate method for every signal. Points on the line are right.",
            "Each network's numbers come from the network that never saw that person."], top=0.99)
    f = hr["finger"].to_numpy()
    for ax, c, s, col in zip(axes.ravel(), cols, short, colors):
        h = hr[c].to_numpy()
        ok = np.isfinite(h) & np.isfinite(f)
        e = np.abs(h[ok] - f[ok])
        ax.scatter(f[ok], h[ok], s=9, color=col, alpha=0.5, lw=0)
        ax.plot([40, 160], [40, 160], color=INK, lw=1.2)
        ax.set_xlim(45, 160)
        ax.set_ylim(40, 185)
        panel_title(ax, s if len(s) < 34 else s.split(" (")[0], "(no training on MCD)" if "published" in s else None)
        note(ax, [f"MAE {e.mean():.2f} bpm", f"within 5 bpm: {np.mean(e <= 5):.0%}", f"r = {np.corrcoef(h[ok], f[ok])[0, 1]:.2f}"],
             loc="upper right", size=12)
        ax.set_xlabel("finger (bpm)")
        ax.set_ylabel("camera (bpm)")
    return save(fig, "r2_camera_hr")


@fig_
def r3_skill():
    setup()
    R = NUM["results"]
    order = ["age", "bmi", "rigidity", "respiratory", "saturation", "cholesterol", "glycated_hemoglobin", "stress", "temperature"]
    fig, ax = plt.subplots(figsize=(W, 7.4), dpi=200)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.8, bottom=0.1)
    header(fig, "How much smaller the error is than the median guess",
           ["Error reduction (skill) with its 95% interval, all 600 people, 3 shuffles. 0 = no better than guessing.",
            "An interval that stays right of 0 means the model clearly beats the guess."], gap=0.75)
    tiers = [("v1_video", "Face_AI_2 v1, video", GREY, 0.27), ("v2_video", "Face_AI_2 v2, video (final)", BLUE, 0.0),
             ("v2_mm", "v2 + finger PPG + ECG (needs sensors)", VIOLET, -0.27)]
    for k, t in enumerate(order):
        for tier, lab, col, off in tiers:
            r = R[tier][t]
            y = k + off
            ax.plot([100 * r["skill_lo"], 100 * r["skill_hi"]], [y, y], color=col, lw=2.6, solid_capstyle="round")
            ax.plot(100 * r["skill"], y, "o", ms=8.5, color=col, mec=SURFACE, mew=1.5, zorder=4)
        r = R["v2_video"][t]
        ax.text(72, k, f"{100 * r['skill']:+.1f}%", va="center", ha="right", fontsize=12.5, color=INK)
    ax.axvline(0, color=INK, lw=1.3)
    ax.set_yticks(range(len(order)), [NAME[t] for t in order], fontsize=13.5)
    ax.invert_yaxis()
    ax.set_xlim(-8, 74)
    ax.set_xlabel("error reduction vs the median guess (%)")
    ax.grid(axis="y", visible=False)
    ax.text(72, -0.7, "v2 video", ha="right", fontsize=12, color=INK2)
    handles = [Line2D([0], [0], color=c, lw=2.6, marker="o", ms=8, label=l) for _, l, c, _ in tiers]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.01, 1.01), ncol=3, fontsize=12.5, frameon=False)
    return save(fig, "r3_skill")


@fig_
def r4_pred_true():
    setup()
    fig, axes = plt.subplots(2, 3, figsize=(W, 8.4), dpi=200)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.85, bottom=0.08, hspace=0.45, wspace=0.27)
    header(fig, "Predicted vs true values",
           ["All 1,200 recordings (shuffle 1), each answered by a model that never saw that person. The line is a perfect",
            "answer. A flat band means the model gives almost everyone the same answer."], top=0.99)
    P = NUM["pred_vs_true"]
    for ax, t in zip(axes.ravel(), ["age", "bmi", "rigidity", "respiratory", "saturation", "sex"]):
        if t == "sex":
            y = REC["y_sex"].to_numpy()
            p = REC["p_sex"].to_numpy()
            ok = np.isfinite(y)
            rng = np.random.default_rng(0)
            ax.scatter(y[ok] + rng.uniform(-0.18, 0.18, ok.sum()), p[ok], s=9, color=BLUE, alpha=0.45, lw=0)
            bad = ok & ((p >= 0.5) != (y == 1))
            ax.scatter(y[bad] + rng.uniform(-0.18, 0.18, bad.sum()), p[bad], s=60, color=RED, marker="v", lw=0, zorder=5)
            ax.axhline(0.5, color=INK, lw=1.2)
            ax.set_xticks([0, 1], ["recorded female", "recorded male"])
            ax.set_ylabel("answer: P(male)")
            panel_title(ax, "Recorded sex", f"{NUM['sex']['acc']:.2%} right · red: the 8 wrong")
            continue
        y = REC[f"y_{t}"].to_numpy()
        p = REC[f"p_{t}"].to_numpy()
        lo, hi = np.nanpercentile(np.r_[y, p], [0.3, 99.7])
        if t == "saturation":
            lo, hi = 86, 100
        if t == "respiratory":
            lo, hi = 14.5, 24.5
        ax.scatter(y, p, s=9, color=BLUE, alpha=0.45, lw=0)
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        panel_title(ax, NAME[t], f"r = {P[t]['r']:.2f} · error {NUM['results']['v2_video'][t]['mae']:.3g} {UNIT[t] if t != 'rigidity' else ''}")
        ax.set_xlabel(f"recorded ({UNIT[t]})")
        ax.set_ylabel("answer")
    return save(fig, "r4_pred_true")


@fig_
def r5_h2h():
    setup()
    H = pd.DataFrame(NUM["h2h"]).set_index("target")
    order = ["bmi", "rigidity", "stress", "saturation", "cholesterol", "glycated_hemoglobin", "respiratory", "temperature", "age"]
    fig, ax = plt.subplots(figsize=(W, 6.8), dpi=200)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.78, bottom=0.12)
    header(fig, "Face_AI_2 v2 vs Face_AI on Face_AI's 90 test people",
           ["Both learned from the same 510 people. Difference in average error (v2 minus Face_AI) as a share of the",
            "median guess's error, with its 95% interval. Left of 0 = Face_AI_2 v2 better."], gap=0.8)
    for k, t in enumerate(order):
        r = H.loc[t]
        g = r["guess"]
        d, lo, hi = 100 * r["v2_minus_faceai"] / g, 100 * r["lo"] / g, 100 * r["hi"] / g
        col = BLUE if hi < 0 else (ORANGE if lo > 0 else GREY)
        ax.plot([lo, hi], [k, k], color=col, lw=3, solid_capstyle="round")
        ax.plot(d, k, "o", ms=10, color=col, mec=SURFACE, mew=1.5, zorder=4)
        lab = "v2 better" if hi < 0 else ("Face_AI better" if lo > 0 else "no clear difference")
        fm = (lambda v: f"{v:.3f}") if r["guess"] < 1 else (lambda v: f"{v:.2f}")
        ax.text(23, k, f"{fm(r['faceai2_v2'])} vs {fm(r['face_ai_newest'])}  ·  {lab}", va="center", fontsize=12.5,
                color=INK if col != GREY else INK2)
    ax.axvline(0, color=INK, lw=1.3)
    ax.set_yticks(range(len(order)), [NAME[t] for t in order], fontsize=13.5)
    ax.invert_yaxis()
    ax.set_xlim(-22, 62)
    ax.set_xlabel("v2 − Face_AI, % of the median guess's error")
    ax.grid(axis="y", visible=False)
    ax.text(23, -0.85, "error: v2 vs Face_AI", fontsize=12, color=INK2)
    return save(fig, "r5_h2h")


# ------------------------------------------------------------------ results in detail
def every_answer(targets, name, title_sub):
    setup()
    n = len(targets)
    cols = 3 if n > 4 else 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(W, 4.2 * rows + 1.4), dpi=200, squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=1 - 1.25 / (4.2 * rows + 1.4), bottom=0.07, hspace=0.5, wspace=0.27)
    header(fig, "Every answer, and the bad cases", title_sub, top=0.99)
    mae = {t: NUM["results"]["v2_video"][t]["mae"] for t in TG}
    for ax, t in zip(axes.ravel(), targets):
        y = REC[f"y_{t}"].to_numpy()
        p = REC[f"p_{t}"].to_numpy()
        if t == "sex":
            ok = np.isfinite(y)
            rng = np.random.default_rng(1)
            jit = rng.uniform(-0.18, 0.18, len(y))
            bad = ok & ((p >= 0.5) != (y == 1))
            ax.scatter((y + jit)[ok & ~bad], p[ok & ~bad], s=9, color=CLEARED, lw=0)
            ax.scatter((y + jit)[bad], p[bad], s=55, color=RED, marker="v", lw=0, zorder=5)
            ax.axhline(0.5, color=INK, lw=1.2)
            ax.set_xticks([0, 1], ["recorded female", "recorded male"])
            ax.set_ylabel("answer: P(male)")
            panel_title(ax, "Recorded sex", f"{int(bad.sum())} wrong recordings (4 people, both sessions)")
            continue
        lo, hi = np.nanmin(np.r_[y, p]), np.nanmax(np.r_[y, p])
        pad = 0.03 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        xs = np.linspace(lo, hi, 2)
        th = 3 * mae[t]
        ax.fill_between(xs, xs - th, xs + th, color=SHADE, lw=0, zorder=0)
        bad = np.abs(p - y) > th
        ax.scatter(y[~bad], p[~bad], s=9, color=CLEARED, lw=0)
        ax.scatter(y[bad], p[bad], s=40, color=RED, marker="v", lw=0, zorder=5)
        ax.plot(xs, xs, color=INK, lw=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        panel_title(ax, NAME[t], f"{int(bad.sum())} bad cases · beyond {th:.3g} {UNIT[t] if t != 'rigidity' else ''}")
        ax.set_xlabel(f"recorded ({UNIT[t]})")
        ax.set_ylabel("answer")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return save(fig, name)


@fig_
def e1a_answers():
    return every_answer(["saturation", "temperature", "glycated_hemoglobin", "cholesterol", "respiratory", "rigidity"], "e1a_answers",
                        ["All 1,200 recordings (shuffle 1). Grey band: within 3 × the model's average error. Red: a bad case,",
                         "an answer further off than that. The diagonal line is a perfect answer."])


@fig_
def e1b_answers():
    return every_answer(["stress", "age", "bmi", "sex"], "e1b_answers",
                        ["(continued) For sex, a bad case is an answer on the wrong side of 50%."])


SS = {r["condition"]: r for r in NUM["sens_spec_cv"]}
CONDS = [("Oxygen saturation below 95%", "saturation", 95, "low", "below normal"),
         ("Temperature 36.8 °C or more", "temperature", 36.8, "high", "warmest 5%"),
         ("HbA1c 5.7% or more", "glycated_hemoglobin", 5.7, "high", "prediabetes range"),
         ("HbA1c 6.5% or more", "glycated_hemoglobin", 6.5, "high", "diabetes range"),
         ("Cholesterol 5.2 mmol/L or more", "cholesterol", 5.2, "high", "above desirable"),
         ("Respiratory rate above 20/min", "respiratory", 20.5, "high", "above adult normal"),
         ("Stiffness 12.4 or more", "rigidity", 12.4, "high", "about the top 10%"),
         ("Stress 6.2 or more", "stress", 6.2, "high", "PSM-25 high band"),
         ("Age 40 or older", "age", 40, "high", "older adults"),
         ("BMI 30 or more", "bmi", 30, "high", "obesity (WHO)"),
         ("Recorded male", "sex", 0.5, "high", "P(male) 50% or more")]
ASIS = {"saturation": 94.5, "temperature": 36.75, "glycated_hemoglobin": None, "cholesterol": 5.15, "respiratory": 20.5,
        "rigidity": 12.395, "stress": 6.195, "age": 39.5, "bmi": 29.95, "sex": 0.5}


def sens_panels(conds, name, sub, legend):
    setup()
    n = len(conds) + (1 if legend else 0)
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(W, 4.3 * rows + 1.4), dpi=200, squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=1 - 1.3 / (4.3 * rows + 1.4), bottom=0.06, hspace=0.55, wspace=0.3)
    header(fig, "Sensitivity and specificity at clinical cut-offs", sub, top=0.99)
    for ax, (cname, t, cut, side, why) in zip(axes.ravel(), conds):
        r = SS[cname]
        y = REC[f"y_{t}"].to_numpy()
        p = REC[f"p_{t}"].to_numpy()
        ok = np.isfinite(y)
        y, p = y[ok], p[ok]
        if t == "sex":
            cond = y == 1
            yes = p >= 0.5
        else:
            cond = (y < cut) if side == "low" else (y >= cut - 1e-9 if t != "respiratory" else y > 20)
            ac = {"glycated_hemoglobin": 5.695 if cut == 5.7 else 6.495}.get(t, ASIS[t])
            yes = (p < ac) if side == "low" else (p >= ac)
        tp, fn, fp, tn = cond & yes, cond & ~yes, ~cond & yes, ~cond & ~yes
        xj = y.astype(float)
        if t == "sex":
            xj = y + np.random.default_rng(2).uniform(-0.18, 0.18, len(y))
        ax.scatter(xj[tn], p[tn], s=8, color=CLEARED, lw=0)
        ax.scatter(xj[tp], p[tp], s=22, color=CAUGHT, lw=0, zorder=4)
        ax.scatter(xj[fn], p[fn], s=34, color=MISSED, marker="v", lw=0, zorder=5)
        ax.scatter(xj[fp], p[fp], s=34, color=FALSE, marker="^", lw=0, zorder=5)
        xl = ax.get_xlim()
        if t == "sex":
            ax.axvline(0.5, color=INK, lw=1.4)
            ax.axhline(0.5, color=INK, lw=1.4)
            ax.set_xticks([0, 1], ["recorded\nfemale", "recorded\nmale"])
        else:
            xcut = cut if t != "respiratory" else 20.5
            ax.axvline(xcut, color=INK, lw=1.4)
            if t != "glycated_hemoglobin" or True:
                ac = {"glycated_hemoglobin": 5.695 if cut == 5.7 else 6.495}.get(t, ASIS[t])
                ax.axhline(ac, color=INK, lw=1.4)
            if side == "low":
                ax.axvspan(xl[0], xcut, color=SHADE, zorder=0)
            else:
                ax.axvspan(xcut, xl[1], color=SHADE, zorder=0)
            ax.set_xlim(xl)
        cv = r["cv_cut_median"]
        constant = r["answer_max"] - r["answer_min"] < 1e-6
        if not constant:
            ax.axhline(cv, color=INK, lw=1.4, ls=(0, (5, 4)))
        panel_title(ax, cname, f"{why} · {r['with_it']} of {r['recordings']:,}")
        lines = [f"Sensitivity {r['as_is_sens']:.0%} ({r['as_is']['tp'] if 'as_is' in r else int(tp.sum())} of {r['with_it']})",
                 f"Specificity {r['as_is_spec']:.0%}"]
        lines = [f"Sensitivity {100 * r['as_is_sens']:.0f}% ({int(tp.sum())} of {int(cond.sum())})",
                 f"Specificity {100 * r['as_is_spec']:.1f}% ({int(tn.sum())} of {int((~cond).sum())})"]
        if constant:
            lines.append("one answer for everyone · AUC 0.50")
        else:
            lines.append(f"Dashed: {100 * r['cv_sens']:.0f}% / {100 * r['cv_spec']:.0f}% · AUC {r['auc']:.2f}")
        loc = "upper left"
        if t in ("saturation",):
            loc = "lower left"
        if t in ("age", "sex"):
            loc = "lower right"
        note(ax, lines, loc=loc, size=11.5)
        ax.set_xlabel(f"recorded ({UNIT.get(t, '')})" if t != "sex" else "")
        ax.set_ylabel("answer" if t != "sex" else "answer: P(male)")
    k = len(conds)
    for ax in axes.ravel()[k:]:
        ax.axis("off")
    if legend:
        ax = axes.ravel()[k]
        items = [Line2D([0], [0], marker="o", ls="", color=CAUGHT, ms=9, label="caught (true positive)"),
                 Line2D([0], [0], marker="v", ls="", color=MISSED, ms=10, label="missed (false negative)"),
                 Line2D([0], [0], marker="^", ls="", color=FALSE, ms=10, label="false alarm (false positive)"),
                 Line2D([0], [0], marker="o", ls="", color=CLEARED, ms=9, label="correctly cleared (true negative)"),
                 Line2D([0], [0], color=INK, lw=1.4, label="clinical cut-off"),
                 Line2D([0], [0], color=INK, lw=1.4, ls=(0, (5, 4)), label="cut-off chosen on other people")]
        ax.legend(handles=items, loc="upper left", fontsize=12.5, frameon=False)
        ax.text(0.0, 0.18, "Sensitivity = caught ÷ all who have it\nSpecificity = cleared ÷ all who do not\nAUC: 0.5 = coin flip, 1 = perfect order",
                transform=ax.transAxes, fontsize=12.5, color=INK2, va="top", linespacing=1.5)
    return save(fig, name)


@fig_
def e2a_sens():
    return sens_panels(CONDS[:6], "e2a_sens", ["All 1,200 recordings (shuffle 1). Shaded: has the condition (recorded value). Solid lines: the",
                                               "clinical cut-off. Dashed: the best cut-off on the answer, chosen on the other folds' people."], False)


@fig_
def e2b_sens():
    return sens_panels(CONDS[6:], "e2b_sens", ["(continued)"], True)


def roc_panels(conds, name, sub, legend):
    from sklearn.metrics import roc_curve
    setup()
    n = len(conds) + (1 if legend else 0)
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(W, 4.0 * rows + 1.3), dpi=200, squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=1 - 1.25 / (4.0 * rows + 1.3), bottom=0.07, hspace=0.5, wspace=0.3)
    header(fig, "ROC curves", sub, top=0.99)
    for ax, (cname, t, cut, side, why) in zip(axes.ravel(), conds):
        r = SS[cname]
        y = REC[f"y_{t}"].to_numpy()
        p = REC[f"p_{t}"].to_numpy()
        ok = np.isfinite(y)
        y, p = y[ok], p[ok]
        cond = (y == 1) if t == "sex" else ((y < cut) if side == "low" else (y > 20 if t == "respiratory" else y >= cut - 1e-9))
        score = -p if side == "low" else p
        fpr, tpr, _ = roc_curve(cond, score)
        ax.fill_between(fpr, tpr, color="#e3eefb", lw=0)
        ax.plot(fpr, tpr, color=BLUE, lw=2.2)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, ls=(0, (4, 4)))
        ax.plot(1 - r["as_is_spec"], r["as_is_sens"], "o", ms=11, mfc="white", mec=INK, mew=2, zorder=5)
        if r["answer_max"] - r["answer_min"] > 1e-6:
            ax.plot(1 - r["cv_spec"], r["cv_sens"], "D", ms=9, color=INK, zorder=5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        short = cname.replace("Respiratory rate", "Breathing").replace("Oxygen saturation", "SpO2").replace("Cholesterol", "Chol.")
        panel_title(ax, short, f"AUC {r['auc']:.2f} ({r['auc_lo']:.2f}–{r['auc_hi']:.2f})")
        ax.set_xlabel("false alarms (1 − specificity)")
        ax.set_ylabel("sensitivity")
    k = len(conds)
    for ax in axes.ravel()[k:]:
        ax.axis("off")
    if legend:
        ax = axes.ravel()[k]
        items = [Line2D([0], [0], color=BLUE, lw=2.2, label="every cut-off on the answer"),
                 Line2D([0], [0], color=MUTED, lw=1.2, ls=(0, (4, 4)), label="coin flip"),
                 Line2D([0], [0], marker="o", ls="", mfc="white", mec=INK, mew=2, ms=11, label="clinical cut-off, answer as it is"),
                 Line2D([0], [0], marker="D", ls="", color=INK, ms=9, label="cut-off chosen on other people")]
        ax.legend(handles=items, loc="upper left", fontsize=12.5, frameon=False)
    return save(fig, name)


@fig_
def e3a_roc():
    return roc_panels(CONDS[:6], "e3a_roc", ["Sensitivity against false alarms for every possible cut-off on the answer (shuffle 1). The top-left",
                                             "corner is perfect; the dashed diagonal is a coin flip. Interval: 95%, people redrawn 2,000 times."], False)


@fig_
def e3b_roc():
    return roc_panels(CONDS[6:], "e3b_roc", ["(continued)"], True)


if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        print(FIGS[n]())
