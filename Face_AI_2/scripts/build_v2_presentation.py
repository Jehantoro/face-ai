"""Build the final-model slide deck, presentation/presentation.html, from saved results only.

It follows the look, slide order and controls of Face_AI's deck (Face_AI/report/presentation.html): 23 slides of
1920 x 1080, arrow keys or a click move between slides, N shows the speaker notes, F full screen. Nothing is trained:
every number is read from results/, features/, data/cache_v2/ or data/db.csv (via v2_core), and every chart is drawn
from those files.

    .venv/Scripts/python.exe -X utf8 scripts/build_v2_presentation.py
"""
from __future__ import annotations

import base64
import glob
import io
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
import numpy as np                                                         # noqa: E402
import pandas as pd                                                        # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                      # noqa: E402
from scipy.signal import butter, filtfilt                                  # noqa: E402
from sklearn.metrics import roc_auc_score                                  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_core as C                                                        # noqa: E402

PROJECT = C.PROJECT
OUT = PROJECT / "presentation" / "presentation.html"

# ------------------------------------------------------------------ colours and type (Face_AI deck)
BG, CREAM, SEC, MUTED, GOLD, BLUE = "#0E1522", "#F3EFE6", "#A9B4C7", "#8794AA", "#E2B45C", "#6EA1F2"
CARD, CARD_B, HI, HI_B, FRZ, FRZ_B = "#151F31", "#26324A", "#18273F", "#34507A", "#1A2233", "#2F3B52"
GREY = "#5E6A82"            # comparison series in charts (passes the colour checks against BLUE on BG)
SERIF = "font-family:'Source Serif 4', Georgia, serif"

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 16, "text.color": SEC, "axes.labelcolor": SEC,
    "axes.edgecolor": CARD_B, "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
    "xtick.color": SEC, "ytick.color": SEC, "axes.grid": True, "grid.color": "#1C2638", "grid.linewidth": 1,
    "axes.axisbelow": True, "axes.titlesize": 18, "axes.titleweight": "bold", "axes.titlecolor": CREAM,
    "axes.titlelocation": "left", "legend.frameon": False, "axes.unicode_minus": True,
})


def fig_px(w, h, **kw):
    """A figure sized in slide pixels, saved at 2.5x for sharpness (1 pt of type = 1.39 slide px)."""
    return plt.subplots(figsize=(w / 100, h / 100), dpi=100, **kw)


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=250)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def pct(x, d=1):
    return f"{x * 100:+.{d}f}%".replace("-", "−")


def num(x, d=3):
    return f"{x:.{d}f}".replace("-", "−")


# ------------------------------------------------------------------ labels and saved predictions
df, Y, T = C.df, C.Y_nat, C.ALL_TARGETS
LABEL = {"saturation": "Oxygen saturation", "temperature": "Temperature", "glycated_hemoglobin": "HbA1c",
         "cholesterol": "Cholesterol", "respiratory": "Respiratory rate", "rigidity": "Arterial stiffness",
         "stress": "Stress", "age": "Age", "bmi": "BMI", "sex": "Recorded sex"}
UNIT = {"saturation": "%", "temperature": "°C", "glycated_hemoglobin": "%", "cholesterol": "mmol/L",
        "respiratory": "breaths/min", "rigidity": "index", "stress": "score", "age": "years", "bmi": "kg/m²"}


def lname(t):
    """The target's name inside a sentence."""
    return LABEL[t] if t in ("bmi", "glycated_hemoglobin") else LABEL[t].lower()
key2i = {k: i for i, k in enumerate(df["key"])}


def load_oof(pattern):
    o = pd.concat([pd.read_csv(f, dtype={"patient_id": str}) for f in sorted(glob.glob(str(C.OOF / pattern)))],
                  ignore_index=True)
    o["i"] = o["key"].map(key2i)
    return o


OOF = {"v2": load_oof("video_specialists_v2_r*.csv"), "v1": load_oof("video_specialists_r*.csv"),
       "guess": load_oof("constant_r*.csv"), "mm": load_oof("multimodal_specialists_v2_r*.csv")}
people = np.unique(df["patient_id"])
pidx = {p: k for k, p in enumerate(people)}
W = np.random.default_rng(C.SEED).multinomial(len(people), np.full(len(people), 1 / len(people)), size=2000).astype(float)


def per_person(o, j):
    e = np.abs(Y[o["i"].to_numpy(), j] - o[f"pred_{T[j]}"].to_numpy(float))
    p = o["patient_id"].map(pidx).to_numpy()
    return (np.bincount(p, np.nan_to_num(e), len(people)), np.bincount(p, np.isfinite(e).astype(float), len(people)))


MAE, SKILL, CI = {}, {}, {}
for name, o in OOF.items():
    for j, t in enumerate(T[:-1]):
        s, n = per_person(o, j)
        MAE[name, t] = s.sum() / n.sum()
        if name != "guess":
            sg, ng = per_person(OOF["guess"], j)
            SKILL[name, t] = 1 - MAE[name, t] / (sg.sum() / ng.sum())
            boot = 1 - ((W @ s) / (W @ n)) / ((W @ sg) / (W @ ng))
            CI[name, t] = tuple(np.percentile(boot, [2.5, 97.5]))
    ys, ps = Y[o["i"].to_numpy(), -1], o["pred_sex"].to_numpy(float)
    ok = np.isfinite(ys)
    MAE[name, "sex_acc"] = float(np.mean((ps[ok] > 0.5) == (ys[ok] > 0.5)))
    if name != "guess":
        MAE[name, "sex_auc"] = float(roc_auc_score(ys[ok], ps[ok]))
_ys, _ps = Y[OOF["v2"]["i"].to_numpy(), -1], OOF["v2"]["pred_sex"].to_numpy(float)
sex_wrong = int(round(np.sum((_ps[np.isfinite(_ys)] > 0.5) != (_ys[np.isfinite(_ys)] > 0.5)) / 3))   # per shuffle
assert abs(MAE["v2", "age"] - 1.5846) < 1e-3 and abs(MAE["v2", "rigidity"] - 1.9507) < 1e-3   # = tiers_metrics.csv

sex_per_person = df.groupby("patient_id")["sex"].agg(lambda s: "/".join(sorted(set(s)))).value_counts()
n_f, n_m, n_mixed = int(sex_per_person.get("F", 0)), int(sex_per_person.get("M", 0)), int(sex_per_person.get("F/M", 0))
video_log = pd.read_csv(C.CACHE2 / "log.csv")
frames_median, seconds_median = float(video_log["kept"].median()), float(video_log["duration_s"].median())

# ------------------------------------------------------------------ what each target reads (official video tier)
import run_v2_final as F                                                   # noqa: E402
import v2_nested as N                                                      # noqa: E402

KNOWN = set(F.VIDEO_V2)


def split_routes(s):
    """Route names contain '+', so a joined selection is split by backtracking over the known names."""
    toks = s.split("+")

    def go(i):
        if i == len(toks):
            return []
        for j in range(len(toks), i, -1):
            c = "+".join(toks[i:j])
            if c in KNOWN:
                rest = go(j)
                if rest is not None:
                    return [c] + rest
        return None
    return go(0)


def blocks_of(r):
    if r in F.CHAINS_VIDEO:
        return F.CHAINS_VIDEO[r] + ["demo"]
    if r in N.MLP_ROUTES:
        return N.MLP_ROUTES[r]
    if r in N.MLP_AUG_ROUTES:
        return ["app:" + N.MLP_AUG_ROUTES[r]]
    return N.ROUTES[r][0]


NET = {"physnet_diff": "PhysNet", "rhythmformer_pure": "RhythmFormer", "tracenet": "Trace net"}
ENC = {"clip_l14": "CLIP-L", "clip_b16": "CLIP-B", "dinov2_b": "DINOv2-B", "dinov2_s": "DINOv2-S", "mefem_b": "MeFEm-B"}


def inputs_of(b):
    if b == "dino":
        return ["DINOv2-S"]
    if b == "clip":
        return ["CLIP-B"]
    if b.startswith("app:"):
        return [ENC[b[4:].rsplit("_", 1)[0]]]
    if b == "temporal":
        return ["v1 pulse"]
    if b == "pulse2":
        return ["Pulse"]
    if b == "breath2":
        return ["Breathing"]
    if b.startswith("deep:"):
        return [NET[b[5:]]]
    if b.startswith("shape:"):
        return [NET[b[6:]], "Shape"]
    if b == "demo":
        return ["Chain"]
    raise KeyError(b)


def kind(cfg):
    for p, k in (("ridge", "Ridge"), ("pls", "PLS"), ("krr", "Kernel ridge"), ("hgb", "Boosted trees"),
                 ("const", "Median guess"), ("mlp_heads", "Face_AI head")):
        if cfg.startswith(p):
            return k
    return cfg


sel = pd.read_csv(C.RESULTS / "v2" / "tiers_selection.csv")
sel = sel[sel["tier"] == "video_v2_ens"]
READS, KINDS, M_ROUTES, cache = {}, Counter(), {}, {}
for t, g in sel.groupby("target", sort=False):
    reads, m = Counter(), Counter()
    for row in g.itertuples():
        rs = split_routes(row.routes)
        m[len(rs)] += 1
        seen = set()
        for r in rs:
            if (r, row.repeat) not in cache:
                cache[r, row.repeat] = N.load_route(r, row.repeat)[1]
            lg = cache[r, row.repeat]
            k = kind(lg[(lg["fold"] == row.fold) & (lg["target"] == t)]["config"].iloc[0])
            KINDS[k] += 1
            if k != "Median guess":
                for b in blocks_of(r):
                    seen.update(inputs_of(b))
        reads.update(seen)
    READS[t], M_ROUTES[t] = reads, m
N_CHOICES = int(sel.groupby("target").size().iloc[0])
N_ROUTES = len(F.VIDEO_V2)

# ------------------------------------------------------------------ camera pulse
hr = pd.read_csv(C.RESULTS / "v2" / "camera_hr_first_minute.csv")
HR_COL = {"v1": "v1 (POS/CHROM/green, 3 patches, first minute)",
          "classic": "v2 classic (POS/CHROM, 4 regions, best quality)",
          "trace": "v2 trace network (taught by the finger, cross-fitted)",
          "physnet": "v2 PhysNet on face clips (taught by the finger, cross-fitted)",
          "rf0": "RhythmFormer, published PURE weights, no MCD training (zero-shot)",
          "rf": "RhythmFormer, PURE weights fine-tuned on MCD with the finger (cross-fitted)"}
HR = {k: dict(mae=float(np.mean(np.abs(hr[c] - hr["finger"]))), w5=float(np.mean(np.abs(hr[c] - hr["finger"]) <= 5)))
      for k, c in HR_COL.items()}


def train_logs():
    out = {}
    for m, name in (("tracenet", "Trace network"), ("physnet_diff", "PhysNet"), ("rhythmformer_pure", "RhythmFormer")):
        if m == "tracenet":
            L = [dict(fold=int(e["tag"].split()[-1]), **e) for e in
                 json.load(open(C.CACHE2 / "rppg" / m / "metrics.json"))["log"]]
        else:
            L = [e for e in json.load(open(C.CACHE2 / "rppg" / m / "log.json")) if e["epoch"] != "test"]
        folds = {}
        for e in L:
            folds.setdefault(e["fold"], []).append((int(e["epoch"]), e["loss"], e["val_wave_r_median"],
                                                    e["val_wave_r_median"] - e["val_hr_mae"] / 20))
        out[name] = folds
    return out


LOGS = train_logs()


def snippet():
    """6 s of the recording whose minute-long camera pulse (fine-tuned RhythmFormer) matches the finger at the median."""
    P = np.load(PROJECT / "data" / "cache" / "ppg_sync.npz")
    ppg = P["ppg"].astype(float)
    ppg = np.concatenate([np.repeat(ppg[:, :1], 2, 1), ppg[:, :-2]], 1)    # y[t] = ppg[t - 2], as in rppg_deep.py
    rf = np.load(C.CACHE2 / "rppg" / "rhythmformer_pure" / "wave.npy").astype(float)
    b, a = butter(2, [0.5 / 15, 8 / 15], btype="band")
    n = min(ppg.shape[1], rf.shape[1])
    rs = np.full(len(rf), np.nan)
    for i in range(len(rf)):
        x, y = rf[i, :n], ppg[i, :n]
        if np.isfinite(x).all() and np.isfinite(y).all() and y.std() > 0:
            rs[i] = np.corrcoef(x, filtfilt(b, a, y))[0, 1]
    q = C.CACHE2 / "ppg_quality.npy"
    ok = np.isfinite(rs)
    if q.exists() and np.load(q).dtype == bool:
        ok &= np.load(q)
    i = int(np.nanargmin(np.where(ok, np.abs(rs - np.nanmedian(rs[ok])), np.nan)))
    w = slice(900, 1080)                                                    # 6 s from the middle of the first minute
    z = lambda v: (v - v.mean()) / v.std()                                  # noqa: E731
    # both waves are flipped for display only, so the upstroke of each beat points up
    return dict(key=df["key"].iat[i], r=float(rs[i]), t=np.arange(900, 1080) / 30,
                finger=-z(filtfilt(b, a, ppg[i, :n])[w]), camera=-z(rf[i, w]))


SNIP = snippet()
R = pd.DataFrame(Y, columns=T).corr(method="spearman")

# Route results quoted from results/v2/report_tables.md ("Pulse routes on their own"; nested CV, 3 shuffles).
ROUTES_PULSE = {"classic signals only": (1.282, 2.031), "RhythmFormer pulse alone": (1.333, 1.955)}
# Stiffness vs heart rate (results/v2/stiffness_vs_heart_rate.csv, quoted in the v2 report, section 4.3).
STIFF_PARTIAL = {"video": 0.09, "sensors": 0.30}


# ================================================================== charts
def chart_spearman():
    cmap = LinearSegmentedColormap.from_list("bg", [BLUE, "#2B4370", "#141C2B", "#5A4A2A", GOLD])
    order = ["saturation", "temperature", "glycated_hemoglobin", "cholesterol", "respiratory", "rigidity", "stress",
             "age", "bmi", "sex"]
    M = R.loc[order, order].to_numpy().copy()
    np.fill_diagonal(M, np.nan)
    fig, ax = fig_px(736, 640)
    ax.grid(False)
    im = ax.imshow(M, cmap=cmap, vmin=-0.4, vmax=0.4)
    top = np.nanmax(np.abs(M))
    for i in range(10):
        for j in range(10):
            v = M[i, j]
            if np.isfinite(v) and abs(v) >= 0.15:
                strong = abs(v) == top
                ax.text(j, i, f"{v:+.2f}".replace("-", "−"), ha="center", va="center", fontsize=13,
                        color=BG if strong else CREAM, fontweight="bold" if strong else "normal")
    short = {"saturation": "SpO2", "respiratory": "Breathing", "rigidity": "Stiffness", "sex": "Sex"}
    names = [short.get(t, LABEL[t]) for t in order]
    ax.set_xticks(range(10), names, rotation=45, ha="right", rotation_mode="anchor", fontsize=15)
    ax.set_yticks(range(10), names, fontsize=15)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, ticks=[-0.4, -0.2, 0, 0.2, 0.4])
    cb.ax.set_yticklabels(["−0.4", "−0.2", "0", "+0.2", "+0.4"], fontsize=14)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return png(fig)


def chart_snippet(s):
    fig, ax = fig_px(900, 520)
    ax.plot(s["t"], s["finger"], color=GREY, lw=2.6, solid_capstyle="round", label="finger sensor")
    ax.plot(s["t"], s["camera"], color=BLUE, lw=2.6, solid_capstyle="round", label="camera (fine-tuned RhythmFormer)")
    ax.set_yticks([])
    ax.set_ylabel("Pulse (standardised)", fontsize=15)
    ax.set_xlabel("Seconds into the video", fontsize=15)
    ax.tick_params(labelsize=15, length=0)
    ax.margins(x=0)
    ax.set_ylim(min(s["finger"].min(), s["camera"].min()) - 0.4, max(s["finger"].max(), s["camera"].max()) + 1.6)
    ax.legend(loc="upper left", ncol=2, fontsize=15, labelcolor=SEC, handlelength=1.6, borderaxespad=0.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return png(fig)


def chart_hr():
    fig, axes = fig_px(1664, 550, ncols=3, sharey=True)
    for ax, (k, title, col) in zip(axes, [("v1", "Hand-made, first version", GREY),
                                          ("rf0", "RhythmFormer, out of the box", GREY),
                                          ("rf", "RhythmFormer, fine-tuned here", BLUE)]):
        ax.plot([40, 170], [40, 170], color="#C3CBD8", lw=1.2, zorder=1)
        ax.scatter(hr["finger"], hr[HR_COL[k]].clip(40, 170), s=9, color=col, alpha=0.55, lw=0, zorder=2)
        ax.set_title(title, fontsize=16, pad=34)
        ax.text(0, 1.03, f"{HR[k]['w5'] * 100:.0f}% within 5 bpm · error {HR[k]['mae']:.2f} bpm", transform=ax.transAxes,
                fontsize=14, color=SEC, va="bottom")
        ax.set_xlim(40, 170)
        ax.set_ylim(40, 170)
        ax.set_xticks([60, 100, 140])
        ax.set_yticks([60, 100, 140])
        ax.set_aspect("equal")
        ax.tick_params(labelsize=14, length=0)
        ax.set_xlabel("Finger (bpm)", fontsize=15)
    axes[0].set_ylabel("Camera (bpm)", fontsize=15)
    fig.tight_layout(w_pad=1.4)
    return png(fig)


def chart_training():
    fig, axes = fig_px(1664, 500, ncols=3, sharey=True)
    for ax, (name, folds) in zip(axes, LOGS.items()):
        for f, rows in sorted(folds.items()):
            ep = [r[0] for r in rows]
            ax.plot(ep, [r[2] for r in rows], color=BLUE, lw=2, alpha=0.75, solid_capstyle="round")
            b = max(rows, key=lambda r: r[3])
            ax.scatter([b[0]], [b[2]], s=60, color=GOLD, zorder=3, edgecolor=BG, linewidth=1.5)
        n_ep = max(r[0] for rows in folds.values() for r in rows)
        ax.set_title(f"{name} · {n_ep} epochs", fontsize=17)
        ax.set_xlabel("Epoch", fontsize=15)
        ax.tick_params(labelsize=14, length=0)
        ax.set_xticks(sorted({r[0] for rows in folds.values() for r in rows})[::(2 if n_ep > 12 else 1)])
    axes[0].set_ylabel("Agreement with the finger (r)", fontsize=15)
    axes[0].set_ylim(0.78, 0.92)
    axes[2].scatter([], [], s=60, color=GOLD, label="kept epoch")
    axes[2].plot([], [], color=BLUE, lw=2, label="one of 5 folds")
    axes[2].legend(loc="lower right", fontsize=14, labelcolor=SEC)
    fig.tight_layout(w_pad=1.5)
    return png(fig)


def chart_skill():
    order = sorted(T[:-1], key=lambda t: -SKILL["v2", t])
    fig, ax = fig_px(860, 617)
    for k, t in enumerate(order):
        for dy, name, col in ((-0.17, "v1", GREY), (0.17, "v2", BLUE)):
            lo, hi = CI[name, t]
            y = k + dy
            ax.plot([lo * 100, hi * 100], [y, y], color=col, lw=4, solid_capstyle="round")
            ax.scatter([SKILL[name, t] * 100], [y], s=70, color=col, zorder=3, edgecolor=BG, linewidth=1.5)
    ax.axvline(0, color="#C3CBD8", lw=1.2)
    ax.set_yticks(range(len(order)), [LABEL[t] for t in order], fontsize=15)
    ax.invert_yaxis()
    ax.set_xlabel("Skill vs always guessing the median (%) · 600 people", fontsize=14)
    ax.tick_params(labelsize=14, length=0)
    ax.scatter([], [], s=70, color=GREY, label="first version")
    ax.scatter([], [], s=70, color=BLUE, label="final model")
    ax.legend(loc="lower right", fontsize=14, labelcolor=SEC)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return png(fig)


IMG = {"spearman": chart_spearman(), "snippet": chart_snippet(SNIP), "hr": chart_hr(), "training": chart_training(),
       "skill": chart_skill()}


# ================================================================== slide parts (the Face_AI deck's patterns)
def section(sid, body, cover=False):
    bg = "radial-gradient(circle at 78% 22%, #1D2C48 0%, #0E1522 58%)" if cover else BG
    fade = ' data-transition="fade"' if cover else ""
    return (f'<section id="{sid}"{fade} style="background:{bg}; color:{CREAM}; font-family:\'IBM Plex Sans\', Arial, '
            f'sans-serif; padding:128px; display:flex; flex-direction:column; gap:56px">\n{body}\n</section>')


def header(kicker, title):
    return (f'<div style="display:flex; flex-direction:column; gap:16px">\n'
            f'<p style="font-size:24px; font-weight:500; letter-spacing:4px; text-transform:uppercase; color:{GOLD}">{kicker}</p>\n'
            f'<h2 style="{SERIF}; font-size:64px; font-weight:500; line-height:1.1; letter-spacing:-0.5px; color:{CREAM}">{title}</h2>\n</div>')


def big(value, label, size=120, color=GOLD, label_size=28, label_color=SEC):
    lh = "1" if size >= 100 else "1.1"
    return (f'<div style="display:flex; flex-direction:column; gap:8px"><p style="{SERIF}; font-size:{size}px; '
            f'font-weight:400; line-height:{lh}; color:{color}; white-space:nowrap">{value}</p>'
            f'<p style="font-size:{label_size}px; color:{label_color}">{label}</p></div>')


def box(title, sub, kind="plain", width=268, pad="20px 24px"):
    bg, bd = {"plain": (CARD, CARD_B), "hi": (HI, HI_B), "trained": (HI, BLUE), "frozen": (FRZ, FRZ_B),
              "out": (CARD, GOLD)}[kind]
    tcol = GOLD if kind == "out" else CREAM
    scol = BLUE if kind in ("hi", "trained") else MUTED
    w = f"width:{width}px; " if width else ""
    return (f'<div style="{w}display:flex; flex-direction:column; gap:4px; background:{bg}; border:1px solid {bd}; '
            f'border-radius:14px; padding:{pad}"><p style="font-size:24px; font-weight:600; color:{tcol}">{title}</p>'
            f'<p style="font-size:24px; color:{scol}">{sub}</p></div>')


ARROW_R = ('<svg aria-hidden="true" width="48" height="24" viewBox="0 0 48 24" style="flex:none; align-self:center">'
           '<path d="M0 7.2H28.8V0L48 12L28.8 24V16.8H0Z" fill="#4E5D78"/></svg>')
ARROW_D = ('<svg aria-hidden="true" width="28" height="40" viewBox="0 0 28 40" style="flex:none; align-self:center">'
           '<path d="M8.4 0H19.6V24H28L14 40L0 24H8.4Z" fill="#4E5D78"/></svg>')


def flow(*boxes):
    return '<div style="display:flex; gap:16px">' + ARROW_R.join(boxes) + "</div>"


def chip(text, size=24, pad="10px 24px", hi=False):
    bg, bd = (HI, HI_B) if hi else (CARD, CARD_B)
    return (f'<p style="font-size:{size}px; color:{CREAM}; background:{bg}; border:1px solid {bd}; padding:{pad}; '
            f'border-radius:40px">{text}</p>')


def small_caps(text, color=MUTED):
    return f'<p style="font-size:24px; letter-spacing:2px; text-transform:uppercase; color:{color}">{text}</p>'


def foot(text):
    return f'<p style="font-size:24px; color:{MUTED}">{text}</p>'


def img(key, alt, w, h):
    return (f'<img src="data:image/png;base64,{IMG[key]}" alt="{alt}" style="width:{w}px; height:{h}px; '
            f'object-fit:contain">')


def card(kicker, inner, gold=False, kicker_color=None, pad=56, gap=16, justify=""):
    bd = GOLD if gold else CARD_B
    kc = kicker_color or (GOLD if gold else BLUE)
    j = f" justify-content:{justify};" if justify else ""
    return (f'<div style="flex:1; display:flex; flex-direction:column; gap:{gap}px;{j} background:{CARD}; '
            f'border:1px solid {bd}; border-radius:24px; padding:{pad}px">\n'
            f'<p style="font-size:24px; font-weight:500; letter-spacing:4px; text-transform:uppercase; color:{kc}">{kicker}</p>\n'
            f'{inner}\n</div>')


TH = f"color:{MUTED}; font-weight:500; padding:9px 16px"
TD = "padding:9px 16px"

SLIDES, NOTES = [], []


def add(sid, body, notes, cover=False):
    SLIDES.append(section(sid, body, cover))
    NOTES.append(" ".join(notes.split()))


# ------------------------------------------------------------------ 1 cover
add("cover", f"""<div style="width:96px; height:3px; background:{GOLD}"></div>
<p style="font-size:24px; font-weight:500; letter-spacing:4px; text-transform:uppercase; color:{GOLD}">Face AI · Group 3</p>
<h1 style="{SERIF}; font-size:120px; font-weight:400; line-height:1.02; letter-spacing:-2px; color:{CREAM}; width:1400px">Health from a 3-minute face video</h1>
<p style="font-size:28px; color:#A9B4C7">Final model (Face_AI_2 v2) · 2 November 2026</p>""",
    """This is the final model of our project: predicting health from an ordinary three-minute face video. It builds
    on everything before it: it reads the whole video instead of one minute, learns a camera pulse from the finger
    sensor, and picks a separate specialist model for every target. I will go through the data, how we process the
    video, the model, how we trained and checked it, and the results. At the end I will say honestly what works, what
    does not, and what we would do next.""", cover=True)
SLIDES[-1] = SLIDES[-1].replace("display:flex; flex-direction:column; gap:56px", "display:flex; flex-direction:column; justify-content:end; gap:36px", 1)

# ------------------------------------------------------------------ 2 goal
pill = lambda t, hi=False: (f'<p style="font-size:28px; color:{CREAM}; background:{HI if hi else CARD}; border:1px solid '  # noqa: E731
                            f'{HI_B if hi else CARD_B}; padding:12px 28px; border-radius:40px">{t}</p>')
point = lambda t: (f'<div style="flex:1; display:flex; flex-direction:column; gap:20px"><div style="width:48px; height:2px; '  # noqa: E731
                   f'background:{GOLD}"></div><p style="font-size:28px; color:{CREAM}">{t}</p></div>')
add("goal", header("Goal", "Ten health targets from one face video") + f"""
<div style="display:flex; gap:64px">
<div style="flex:2; display:flex; flex-direction:column; gap:20px">
{small_caps("Physiology · 7", "#8794AA")}
<div style="display:flex; flex-wrap:wrap; gap:16px">{''.join(pill(t) for t in ["Oxygen saturation", "Temperature", "HbA1c", "Cholesterol", "Respiratory rate", "Arterial stiffness", "Stress (PSM-25)"])}</div>
</div>
<div style="flex:1; display:flex; flex-direction:column; gap:20px">
{small_caps("Appearance · 3", "#8794AA")}
<div style="display:flex; flex-wrap:wrap; gap:16px">{''.join(pill(t, True) for t in ["Age", "BMI", "Recorded sex"])}</div>
</div>
</div>
<div style="flex:1"></div>
<div style="display:flex; gap:64px">{point("Video only at prediction time")}{point("Every target vs a median guess")}{point("Judged on unseen people")}</div>""",
    """We predict the same ten targets as before: seven physiological ones, from oxygen saturation to stress, and three
    appearance ones, age, BMI and recorded sex. At prediction time the model sees only the video. A finger sensor is
    used during training only, as a teacher for the camera pulse, never as an input. Every target is compared with
    always guessing the median, and every score comes from people the model never saw.""")

# ------------------------------------------------------------------ 3 data
rows = []
for t in T[:-1]:
    y = Y[:, T.index(t)]
    med, sd = np.nanmedian(y), np.nanstd(y, ddof=1)
    fmt = {"temperature": (1, 2), "glycated_hemoglobin": (1, 2), "cholesterol": (1, 2), "saturation": (0, 1),
           "respiratory": (0, 1), "rigidity": (1, 1), "stress": (1, 1), "age": (0, 1), "bmi": (1, 1)}[t]
    gold_sd = t in ("saturation", "temperature")
    gold_med = t == "age"
    rows.append(f'<tr><td style="color:{CREAM}; {TD}">{LABEL[t]}{" (PSM-25)" if t == "stress" else ""}</td>'
                f'<td style="{TD}">{UNIT[t]}</td>'
                f'<td style="text-align:right; {"color:" + GOLD + "; " if gold_med else ""}{TD}">{med:.{fmt[0]}f}</td>'
                f'<td style="text-align:right; {"color:" + GOLD + "; " if gold_sd else ""}{TD}">{sd:.{fmt[1]}f}</td></tr>')
rows.append(f'<tr><td style="color:{CREAM}; {TD}">Recorded sex ({n_f} F · {n_m} M)</td><td style="{TD}">F / M</td>'
            f'<td style="text-align:right; {TD}">—</td><td style="text-align:right; {TD}">—</td></tr>')
add("data", header("Data", "The data") + f"""
<div style="display:flex; gap:96px">
<div style="width:480px; display:flex; flex-direction:column; gap:40px">
{big("600", "people, before and after exercise")}
{big("1,200", "frontal webcam videos", color=CREAM)}
{big("3 min", f"per video · about {round(frames_median, -2):,.0f} frames", color=CREAM)}
</div>
<div style="flex:1; display:flex; flex-direction:column; gap:20px">
<table style="font-size:24px; color:#A9B4C7; border:1px solid {CARD_B}">
<tr><th style="width:40%; text-align:left; {TH}">Target</th><th style="width:24%; text-align:left; {TH}">Unit</th><th style="width:18%; text-align:right; {TH}">Median</th><th style="width:18%; text-align:right; {TH}">SD</th></tr>
{chr(10).join(rows)}
</table>
{foot(f"All 600 people · 1,200 recordings · {n_mixed} people's recorded sex differs between sessions")}
</div>
</div>""",
    f"""The dataset has 600 people, each filmed twice, before and after exercise, so 1,200 frontal webcam videos. The
    final model uses the whole recording: about three minutes, or {round(frames_median, -2):,.0f} frames at 30 frames
    per second. The table shows the labels of all 600 people, because every person is scored once in our
    cross-validation. Oxygen saturation and temperature barely vary, with spreads of only 1.3% and 0.13 °C, and the
    median age is 20: most people are 18 to 24, so older people are rare and harder to predict.""")

# ------------------------------------------------------------------ 4 correlation
iu = np.triu_indices(10, 1)
pairs = sorted([(abs(R.iat[i, j]), R.iat[i, j], T[i], T[j]) for i, j in zip(*iu)], reverse=True)
n_small = int(sum(abs(R.iat[i, j]) < 0.2 for i, j in zip(*iu)))
top = pairs[0]
add("correlation", header("Data", "Spearman correlation") + f"""
<div style="display:flex; gap:96px; align-items:center">
{img("spearman", f"Heatmap of Spearman rank correlations between the 10 targets on 1,200 recordings; the strongest pair, BMI and recorded sex, is {top[1]:+.2f}", 736, 640)}
<div style="flex:1; display:flex; flex-direction:column; gap:56px">
{big(f"{top[1]:+.2f}", "strongest pair: BMI and recorded sex")}
{big(f"{n_small} of 45", "pairs almost unrelated (|ρ| below 0.2)", color=CREAM)}
{foot("1,200 recordings · 600 people")}
</div>
</div>""",
    f"""Spearman correlation shows whether two targets rise and fall together, from −1 to +1. Over all 1,200 recordings
    the strongest pair is BMI and recorded sex at {top[1]:+.2f}, then {lname(pairs[1][2])} and {lname(pairs[1][3])} at
    {pairs[1][1]:+.2f}, and {lname(pairs[2][2])} and {lname(pairs[2][3])} at {pairs[2][1]:+.2f}. {n_small} of the 45 pairs are below 0.2, so the targets are nearly independent of each other.
    No target can be read off another; each needs its own evidence in the video.""".replace("-", "−"))

# ------------------------------------------------------------------ 5 targets
add("targets", header("Approach", "Two kinds of targets") + f"""
<div style="display:flex; gap:40px">
{card("Appearance", f'''<h3 style="{SERIF}; font-size:36px; font-weight:500; line-height:1.2; color:{CREAM}">Age · BMI · Recorded sex</h3>
<div style="height:24px"></div>
<p style="{SERIF}; font-size:120px; font-weight:400; line-height:1; color:{GOLD}">24 crops</p>
<p style="font-size:28px; color:{SEC}">spread over the video: the face does not change</p>
<p style="font-size:24px; color:{MUTED}">Sharp 256-pixel crops cut from the full-size frame</p>''')}
{card("Physiology", f'''<h3 style="{SERIF}; font-size:36px; font-weight:500; line-height:1.2; color:{CREAM}">The other 7 targets</h3>
<div style="height:24px"></div>
<p style="{SERIF}; font-size:120px; font-weight:400; line-height:1; color:{GOLD}">{round(frames_median, -2):,.0f}</p>
<p style="font-size:28px; color:{SEC}">frames: pulse and breathing need time</p>
<p style="font-size:24px; color:{MUTED}">Temperature, HbA1c, cholesterol: steady during a recording</p>''')}
</div>""",
    """The targets fall into two groups. Appearance targets, age, BMI and sex, look the same in every frame, so the
    model reads 24 sharp face crops spread over the video and averages what it sees. Physiological signals such as
    pulse and breathing change every second, so they need the continuous signal from every frame. Temperature, HbA1c
    and cholesterol do not change during a recording at all, so a longer video cannot help them.""")

# ------------------------------------------------------------------ 6 split
cell = lambda judged: (f'<p style="font-size:24px; background:{BLUE if judged else CARD}; color:{BG if judged else MUTED}; '  # noqa: E731
                       f'text-align:center; padding:6px; border-radius:8px">{"judged" if judged else "train"}</p>')
grid = [f'<div></div>'] + [f'<p style="font-size:24px; color:{MUTED}; text-align:center">Group {g}</p>' for g in range(1, 6)]
for r in range(1, 6):
    grid.append(f'<p style="font-size:24px; color:{MUTED}">Round {r}</p>')
    grid += [cell(g == r) for g in range(1, 6)]
add("split", header("Data", "Data split") + f"""
<div style="display:flex; gap:12px">
<div style="width:1316px; display:flex; flex-direction:column; gap:6px; background:{HI}; border:1px solid {HI_B}; border-radius:16px; padding:28px 36px"><p style="font-size:28px; font-weight:600; color:{CREAM}">Training · one round</p><p style="font-size:24px; color:{SEC}">480 people · 960 videos</p></div>
<div style="flex:1; display:flex; flex-direction:column; gap:6px; background:{CARD}; border:1px solid {GOLD}; border-radius:16px; padding:28px 28px"><p style="font-size:28px; font-weight:600; color:{GOLD}">Judged</p><p style="font-size:24px; color:{SEC}">120 people</p></div>
</div>
<div style="display:flex; gap:96px; align-items:center">
<div style="display:grid; grid-template-columns:132px repeat(5, 140px); gap:8px">
{chr(10).join(grid)}
</div>
<div style="flex:1; display:flex; flex-direction:column; gap:40px">
{big("600", "people, each judged by a model that never saw them")}
{big("× 3", "shuffles of the five groups", size=64, color=CREAM)}
{foot("Both sessions of a person stay in one group")}
</div>
</div>""",
    """Instead of one fixed test group, the final model scores all 600 people. We split them into five groups of 120;
    in each round the model trains on four groups, 480 people, and is judged on the fifth, so after five rounds
    everyone has been judged once, by a model that never saw them. Both videos of a person always stay in the same
    group. We repeat the whole thing with three different shuffles of the groups, so one lucky split cannot flatter
    the result. These are exactly the people, groups and rules of our first version, so the two compare fairly.""")

# ------------------------------------------------------------------ 7 video
sm = "16px 22px"
row = lambda lab, *bx: (f'<div style="display:flex; flex-direction:column; gap:12px">{small_caps(lab)}'  # noqa: E731
                        f'{flow(*bx)}</div>')
step_s = seconds_median / 24
add("video", header("Video processing", "Video processing") + f"""
<div style="display:flex; flex-direction:column; gap:22px">
{row("Face pictures", box("3-min video", f"≈{round(frames_median, -2):,.0f} frames", pad=sm), box("24 frames", f"one every {step_s:.1f} s", pad=sm), box("Face detector", "YuNet", pad=sm), box("Crop 256 px", "tight + wide", pad=sm), box("5 face encoders", "≤ 1,792 numbers", "hi", pad=sm))}
{row("Colour signals · whole video", box("Every frame", f"≈{round(frames_median, -2):,.0f}", pad=sm), box("Face tracking", "box + 5 landmarks", pad=sm), box("4 skin regions", "colour over time", pad=sm), box("Body motion", "shoulders, chest", pad=sm), box("Hand-made signals", "144 numbers", "hi", pad=sm))}
{row("Face clips · first minute", box("First 60 s", "1,800 frames", pad=sm), box("Face clip", "72 × 72 px", pad=sm), box("Pulse network", "taught by the finger", pad=sm), box("Camera pulse", "one wave", pad=sm), box("Rhythm numbers", "19–31 per network", "hi", pad=sm))}
</div>
<div style="display:flex; gap:16px; align-items:center">
<p style="font-size:24px; letter-spacing:2px; text-transform:uppercase; color:{GOLD}; width:120px">Also</p>
{chip("True time of every frame from the printed clock")}
{chip("Clock band blanked in the clips")}
{chip(f"{int((video_log['status'] == 'ok').sum()):,} of 1,200 videos processed")}
</div>""",
    f"""Each video goes down three paths. For the face pictures, we take 24 frames spread over the whole video, one
    every {step_s:.1f} seconds, find the face with the YuNet detector, cut a sharp 256-pixel crop from the full-size
    frame, in a tight and a wide version, and pass it to five frozen face encoders. For the colour signals, we track
    the face in every frame, record the skin colour of four regions and the up-and-down motion of the shoulders and
    chest, and turn them into 144 hand-made numbers for pulse and breathing; the third pulse network also reads these
    colour traces. For the face clips, the first minute becomes a small 72-pixel face video for the two other pulse
    networks. Every frame gets its true time from the printed clock, so the finger sensor lines up with the video
    sample by sample.""")

# ------------------------------------------------------------------ 8 camera pulse
add("signals", header("Video processing", "Camera pulse") + f"""
<div style="display:flex; gap:96px; align-items:center">
{img("snippet", "Six seconds of one recording: the pulse that the fine-tuned RhythmFormer reads from the face rises and falls with the finger sensor beat by beat", 900, 520)}
<div style="flex:1; display:flex; flex-direction:column; gap:48px">
{big("144", "hand-made numbers: pulse and breathing", size=64, color=CREAM)}
{big("3", "pulse networks taught by the finger sensor", size=64, color=CREAM)}
{big(f"{HR['rf']['w5'] * 100:.0f}%", "camera heart rate within 5 bpm of the finger sensor")}
</div>
</div>""",
    f"""Each heartbeat changes the skin colour very slightly, by well under one percent, and breathing moves the
    shoulders a little. The chart shows six seconds of one recording whose camera pulse agrees with the finger exactly
    as well as the median recording: in blue the pulse that the fine-tuned RhythmFormer network reads from the face,
    in grey the finger sensor, rising and falling together beat by beat. From the colour and motion traces we compute
    144 hand-made numbers, and three pulse networks, taught by the finger sensor, give a much cleaner pulse. The camera heart rate is now within 5 beats per minute of the finger for
    {HR['rf']['w5'] * 100:.0f}% of recordings, up from {HR['v1']['w5'] * 100:.0f}% in our first version.""")

# ------------------------------------------------------------------ 9 architecture
X = [0, 351, 702, 1053, 1404]
RY, RH = [0, 165, 330], 130


def abox(x, y, w, h, title, sub, kind="plain"):
    bg, bd = {"plain": (CARD, CARD_B), "hi": (HI, HI_B), "trained": (HI, BLUE), "frozen": (FRZ, FRZ_B),
              "out": (CARD, GOLD)}[kind]
    tcol = GOLD if kind == "out" else CREAM
    scol = BLUE if kind in ("hi", "trained") else MUTED
    return (f'<div style="position:absolute; left:{x}px; top:{y}px; width:{w}px; height:{h}px; display:flex; '
            f'flex-direction:column; justify-content:center; gap:4px; background:{bg}; border:1px solid {bd}; '
            f'border-radius:14px; padding:16px 24px"><p style="font-size:24px; font-weight:600; color:{tcol}">{title}</p>'
            f'<p style="font-size:24px; color:{scol}">{sub}</p></div>')


lanes = [("24 face crops", "256 px", "5 face encoders", "frozen", "frozen", "Average of 24", "768–1,792 numbers"),
         ("Colour + motion", "every frame", "Hand-made signals", "no training", "plain", "Signals", "144 numbers"),
         ("Face clips", "first 60 s", "3 pulse networks", "taught by the finger", "trained", "Rhythm numbers",
          "19–31 per network")]
parts, lines = [], []
for (a, a2, b, b2, bk, c, c2), y in zip(lanes, RY):
    parts += [abox(X[0], y, 260, RH, a, a2), abox(X[1], y, 260, RH, b, b2, bk), abox(X[2], y, 260, RH, c, c2)]
    cy = y + RH // 2
    lines += [f"260,{cy} 351,{cy}", f"611,{cy} 702,{cy}", f"962,{cy} 1053,{cy}"]
parts.append(abox(X[3], 0, 260, 460, "Specialists", "one per target, chosen inside the training people", "trained"))
parts.append(abox(X[4], 40, 260, RH, "Combine", "best route, or mean of 2–3"))
parts.append(abox(X[4], 290, 260, RH, "10 outputs", "9 numbers + sex", "out"))
lines += ["1313,105 1404,105", "1534,170 1534,290"]
svg = ('<svg aria-hidden="true" width="1664" height="460" viewBox="0 0 1664 460" style="position:absolute; left:0; top:0; '
       'overflow:visible; pointer-events:none"><defs><marker id="head" viewBox="0 0 10 10" refX="10" refY="5" '
       'markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#8794AA"/></marker></defs>'
       + "".join(f'<polyline points="{p}" fill="none" stroke="#8794AA" stroke-width="3" stroke-linecap="round" '
                 f'stroke-linejoin="round" marker-end="url(#head)"/>' for p in lines) + "</svg>")
add("architecture", header("Model", "Architecture") + f"""
<div style="position:relative; width:1664px; height:460px">
{chr(10).join(parts)}
{svg}
</div>
<div style="display:flex; gap:16px">
{chip("Age, sex → CLIP-L + Face_AI heads", pad="12px 28px")}
{chip("BMI → CLIP-L, MeFEm-B · ridge, PLS", pad="12px 28px")}
{chip("Stiffness, breathing → pulse numbers · trees, PLS", pad="12px 28px")}
</div>""",
    """This is the whole model. Three kinds of feature makers turn the video into numbers, and none of them ever sees a
    health label: five frozen face encoders on the 24 crops, hand-made pulse and breathing signals, and three pulse
    networks that learned the pulse from the finger sensor. For each target, a specialist is chosen from these inputs
    and a menu of simple models, by a cross-validation inside the training people only. The final prediction is the
    best route alone, or the average of the best two or three. In the end, age and sex use CLIP with Face_AI's small
    task heads, BMI uses CLIP and MeFEm with ridge or PLS, and stiffness and breathing use the pulse numbers with
    boosted trees or PLS.""")

# ------------------------------------------------------------------ 10 inside CLIP ViT-L/14
grid16 = "".join(f"M{x * 13.75:g} 0V220" for x in range(1, 16)) + "".join(f"M0 {y * 13.75:g}H220" for y in range(1, 16))
bars24 = "\n".join(f'<div style="width:200px; height:6px; background:{HI_B}; border-radius:2px"></div>' for _ in range(24))
cells = "".join(f'<rect x="0" y="{1 + 14 * k}" width="56" height="11" rx="2" fill="{BLUE}"/>' for k in range(15)) + \
    f'<rect x="0" y="211" width="56" height="8" rx="2" fill="{BLUE}"/>'
col = lambda pic, t, s: (f'<div style="width:264px; display:flex; flex-direction:column; gap:16px"><div style="height:240px; '  # noqa: E731
                         f'display:flex; flex-direction:column; align-items:center; justify-content:center; gap:3px">{pic}</div>'
                         f'<p style="font-size:24px; font-weight:600; color:{CREAM}">{t}</p><p style="font-size:24px; color:{MUTED}">{s}</p></div>')
sq = lambda n, txt: (f'<div style="width:{n}px; height:{n}px; display:flex; align-items:center; justify-content:center; '  # noqa: E731
                     f'background:{CARD}; border:1px solid {FRZ_B}; border-radius:8px"><p style="font-size:24px; color:{MUTED}">{txt}</p></div>')
add("clip", header("Model", "Inside CLIP ViT-L/14") + f"""
<div style="display:flex; gap:16px">
{col(sq(200, "256"), "Face crop", "256 px wide · 180 px tight")}
{ARROW_R}
{col(sq(180, "224"), "Resized", "224 × 224 px")}
{ARROW_R}
{col(f'<svg aria-label="A 16 by 16 grid of patches, one highlighted" width="220" height="220" viewBox="0 0 220 220"><rect x="1" y="1" width="218" height="218" fill="{CARD}" stroke="{FRZ_B}" stroke-width="2"/><rect x="55" y="82.5" width="13.75" height="13.75" fill="{GOLD}"/><path d="{grid16}" stroke="{CARD_B}" stroke-width="1" fill="none"/></svg>', "256 patches", "+ 1 summary token")}
{ARROW_R}
{col(bars24, "24 transformer blocks", "self-attention")}
{ARROW_R}
{col(f'<svg aria-label="A column of number cells standing for the 1,792 numbers" width="56" height="220" viewBox="0 0 56 220">{cells}</svg>', "1,792 numbers", "describe the face")}
</div>
<div style="display:flex; align-items:baseline; gap:40px">
<p style="{SERIF}; font-size:120px; font-weight:400; line-height:1; color:{GOLD}; white-space:nowrap">304 M</p>
<p style="font-size:28px; color:{SEC}">parameters, all frozen · pretrained on 400 million images with captions</p>
</div>""",
    f"""Our strongest face encoder is CLIP ViT-L/14 from OpenAI, a vision transformer pretrained on 400 million images
    paired with their captions; we keep it frozen, so its 304 million parameters never change. Each face crop is
    resized to 224 × 224 pixels and cut into 256 patches of 14 × 14 pixels, plus one summary token. Twenty-four
    transformer blocks let every patch compare itself with every other patch. We keep the summary token, 1,024
    numbers, and its 768-number projection: 1,792 numbers per crop, averaged over the 24 crops. DINOv2, CLIP-B and
    MeFEm, a face model built for medical use, were candidates too; CLIP-L was picked in all {N_CHOICES} age
    choices.""")

# ------------------------------------------------------------------ 11 specialists
ORDER_K = ["Ridge", "Boosted trees", "PLS", "Face_AI head", "Kernel ridge", "Median guess"]
kmax = max(KINDS.values())
bars = "\n".join(
    f'<div style="display:flex; align-items:center; gap:24px"><p style="width:230px; font-size:24px; color:{SEC}">{k}</p>'
    f'<div style="width:{KINDS[k] / kmax * 420:.0f}px; height:24px; background:{BLUE}; border-radius:0 4px 4px 0"></div>'
    f'<p style="font-size:24px; color:{CREAM}">{KINDS[k]}</p></div>' for k in ORDER_K)
add("specialists", header("Model", "Specialists") + f"""
<div style="display:flex; gap:120px; align-items:center">
<div style="width:600px; display:flex; flex-direction:column; gap:8px">
{box(f"{N_ROUTES} routes", "face, signals, pulse, combinations", width=None, pad="16px 24px")}
{ARROW_D}
{box("6 model types", "ridge, PLS, kernel ridge, trees, Face_AI head, median", "trained", width=None, pad="16px 24px")}
{ARROW_D}
{box("Inner 4-fold cross-validation", "picks route + model, training people only", "trained", width=None, pad="16px 24px")}
{ARROW_D}
{box("Best route, or mean of best 2–3", "one prediction × 10 targets", "out", width=None, pad="16px 24px")}
</div>
<div style="flex:1; display:flex; flex-direction:column; gap:28px">
{small_caps("What the inner loop picked")}
<div style="display:flex; flex-direction:column; gap:18px">
{bars}
</div>
{foot(f"{sum(KINDS.values())} route picks · 10 targets × {N_CHOICES} choices")}
</div>
</div>""",
    f"""A route is one set of inputs, for example CLIP-L on the tight crops, or the three pulse networks together;
    {N_ROUTES} routes compete for every target. Six model types are tried: ridge, PLS and kernel ridge, which are simple
    regularised models, boosted trees on the signal and pulse routes, Face_AI's small task head on the face routes for
    age, BMI and sex, and the median guess, which lets a target abstain when nothing helps. Inside the 480 training people, a 4-fold cross-validation picks the best route
    and model, and whether averaging the best two or three routes is better still. Over all choices, ridge, trees and
    PLS were picked most; Face_AI's head won for age and sex, and temperature almost always fell back to the median
    guess.""")

# ------------------------------------------------------------------ 12 pulse networks
add("timeseries", header("Model", "Pulse networks, taught by the finger") + f"""
{img("hr", "Three scatter plots of camera heart rate against the finger: the first version's hand-made estimate is exact or far off, RhythmFormer out of the box locks near 150 bpm, fine-tuned RhythmFormer lies on the diagonal", 1664, 550)}
<p style="font-size:28px; color:{SEC}">Average heart-rate error: <span style="color:{GOLD}">{HR['rf']['mae']:.2f} bpm</span> RhythmFormer fine-tuned · {HR['physnet']['mae']:.2f} PhysNet · {HR['trace']['mae']:.2f} trace network</p>""",
    f"""To get a clean pulse, we trained three networks to reproduce the finger sensor's pulse from the face: PhysNet,
    a 3-D convolutional network on the face clips; RhythmFormer, a published transformer that we started from weights
    trained on another dataset, PURE; and a 1-D network on the colour traces of the whole video. Each is cross-fitted:
    a person's pulse always comes from a network that never saw that person, and no health label is used. The charts
    show heart rate against the finger. The hand-made estimate of our first version is either exact or far off;
    RhythmFormer out of the box locks onto a false rhythm near 150 beats per minute; fine-tuned on our data it lies on
    the diagonal, with an average error of {HR['rf']['mae']:.2f} beats per minute, against {HR['physnet']['mae']:.2f}
    for PhysNet and {HR['trace']['mae']:.2f} for the trace network. Each pulse reaches the specialists as 19 to 31
    rhythm numbers: heart rate, its trend over the video, signal quality and beat-to-beat variability.""")

# ------------------------------------------------------------------ 13 hyperparameters
kv = lambda k, v: (f'<div style="display:flex; justify-content:space-between; gap:32px; padding:14px 0; border-bottom:1px solid '  # noqa: E731
                   f'{CARD_B}"><p style="font-size:24px; color:{MUTED}">{k}</p><p style="font-size:24px; color:{CREAM}; text-align:right">{v}</p></div>')
add("hyperparameters", header("Training", "Hyperparameters") + f"""
<div style="display:flex; gap:96px">
<div style="flex:1; display:flex; flex-direction:column; gap:16px">
{small_caps("Pulse networks", BLUE)}
<div style="display:flex; flex-direction:column">
{kv("Teacher", "finger pulse, shifted 2 frames")}
{kv("Loss", "1 − best r within ±3 frames + 0.1 × spectrum")}
{kv("Optimiser", "AdamW, one-cycle schedule")}
{kv("Learning rate", "0.0003 · trace network 0.001")}
{kv("Epochs", "PhysNet 6 · RhythmFormer 12 · trace 40")}
{kv("Window", "160 frames (5.3 s) · trace 300")}
{kv("Best epoch", "on 10% held-back training people")}
{kv("Cross-fitting", "5 person folds")}
</div>
</div>
<div style="flex:1; display:flex; flex-direction:column; gap:16px">
{small_caps("Specialists", BLUE)}
<div style="display:flex; flex-direction:column">
{kv("Ridge penalty", f"{len(C.RIDGE_ALPHAS)} values, 0.0001 to 10")}
{kv("PLS components", ", ".join(str(k) for k in C.PLS_KS))}
{kv("Kernel ridge", f"{len(C.KRR_LAMBDAS)} penalties, 0.1 to 100")}
{kv("Boosted trees", "150 trees, depth 3, rate 0.05")}
{kv("Face_AI head", "dropout 30% · 64 units · 3 seeds")}
{kv("Head early stop", "15% of training people · patience 30")}
{kv("Inner CV", f"{C.INNER_FOLDS} folds, training people only")}
{kv("Uncertainty", "2,000 bootstrap resamples of people")}
</div>
</div>
</div>""",
    """These are the settings. The pulse networks are trained with the finger pulse as the target, shifted by two
    frames, the lag we measured between the two recordings. The loss rewards a wave that rises and falls with the
    finger's, allowing up to three frames of timing jitter, plus a small term that matches the heart-rate peak in the
    spectrum. The specialists' settings are small grids, and the inner cross-validation picks among them; Face_AI's
    head keeps its original design, dropout and 64 hidden units, with early stopping on 15% of the training people and
    3 seeds averaged. Every 95% interval comes from 2,000 bootstrap resamples of people.""")

# ------------------------------------------------------------------ 14 training curves
last_r = {n: [rows[-1][2] for rows in f.values()] for n, f in LOGS.items()}
lo_r, hi_r = min(min(v) for v in last_r.values()), max(max(v) for v in last_r.values())
loss_rf = [np.mean([rows[0][1] for rows in LOGS["RhythmFormer"].values()]),
           np.mean([rows[-1][1] for rows in LOGS["RhythmFormer"].values()])]
add("loss", header("Training", "Training vs held-back people") +
    img("training", "Agreement with the finger on held-back training people per epoch, for the 5 folds of each pulse network: it rises quickly and levels off; the kept epoch of each fold is marked", 1664, 500) +
    f'\n<p style="font-size:28px; color:{SEC}">Agreement with the finger on held-back people reaches <span style="color:{GOLD}">{lo_r:.2f}–{hi_r:.2f}</span>; each fold keeps its best epoch.</p>',
    f"""This chart shows how the three pulse networks learned, one line per fold. The vertical axis is how closely the
    network's pulse follows the finger on held-back training people, where 1 would be a perfect match. Agreement rises
    quickly and ends between {lo_r:.2f} and {hi_r:.2f}: the trace network and RhythmFormer flatten out, while PhysNet,
    stopped after 6 epochs to save time, was still creeping up. Meanwhile the training loss keeps falling, for
    RhythmFormer from {loss_rf[0]:.2f} to {loss_rf[1]:.2f}. Each fold keeps the epoch where the held-back people score
    best, marked in gold, so a network does not memorise the people it trained on.""")

# ------------------------------------------------------------------ 15 what each target reads
COLS = [("Face encoders", ["CLIP-L", "CLIP-B", "DINOv2-S", "DINOv2-B", "MeFEm-B"]),
        ("Hand-made", ["Pulse", "Breathing", "v1 pulse"]),
        ("Pulse networks", ["Trace net", "PhysNet", "RhythmFormer", "Shape"]),
        ("Chain", ["Chain"])]
COL_NAME = {"Pulse": "Colour pulse", "Breathing": "Breathing motion", "v1 pulse": "v1 pulse", "Shape": "Pulse shape",
            "Chain": "Predicted age, sex, BMI", "RhythmFormer": "Rhythm&shy;Former"}
ROWS_T = ["age", "bmi", "sex", "rigidity", "respiratory", "saturation", "glycated_hemoglobin", "cholesterol", "stress",
          "temperature"]
flat = [c for _, cs in COLS for c in cs]
head1 = (f'<div style="display:grid; grid-template-columns:250px repeat({len(flat) - 1}, 96px) 150px; gap:5px; '
         f'align-items:end">')
groups = '<div></div>' + "".join(
    f'<p style="grid-column:span {len(cs)}; font-size:20px; letter-spacing:2px; text-transform:uppercase; color:{GOLD}; '
    f'border-bottom:1px solid {CARD_B}; padding-bottom:6px">{g}</p>' for g, cs in COLS)
names = '<div></div>' + "".join(f'<p style="font-size:19px; line-height:1.2; color:{MUTED}; text-align:center">'
                                f'{COL_NAME.get(c, c)}</p>' for c in flat)
body = []
for t in ROWS_T:
    body.append(f'<p style="font-size:24px; color:{CREAM}">{LABEL[t]}</p>')
    for c in flat:
        n = READS[t].get(c, 0)
        a = n / N_CHOICES
        fill = f"rgba(110,161,242,{0.12 + 0.88 * a:.2f})" if n else "transparent"
        txt = (BG if a > 0.55 else CREAM) if n else MUTED
        body.append(f'<p style="height:38px; display:flex; align-items:center; justify-content:center; font-size:21px; '
                    f'color:{txt}; background:{fill}; border-radius:8px">{n if n else "·"}</p>')
matrix = head1 + groups + names + "".join(body) + "</div>"
add("learning", header("Training", "What each target reads") + f"""
{matrix}
{foot(f"How many of the {N_CHOICES} choices (5 folds × 3 shuffles) read each input · picks of the median guess read nothing")}""",
    f"""This table shows what the final specialists actually read, counted over the {N_CHOICES} choices per target, five
    folds times three shuffles. Age, BMI and sex read the face encoders, CLIP-L above all, and BMI also MeFEm.
    Arterial stiffness reads the pulse networks, PhysNet in every choice, and respiratory rate reads the hand-made
    breathing and pulse signals together with the networks. The weak targets read a little of everything, cholesterol
    leans on the predicted age, sex and BMI, and temperature reads nothing at all: it falls back to the median
    guess.""")

# ------------------------------------------------------------------ 16 nested cross-validation
icell = lambda chk: (f'<p style="font-size:24px; background:{GOLD if chk else CARD}; color:{BG if chk else MUTED}; '  # noqa: E731
                     f'text-align:center; padding:6px; border-radius:8px">{"check" if chk else "fit"}</p>')
igrid = ['<div></div>'] + [f'<p style="font-size:24px; color:{MUTED}; text-align:center">{g}</p>' for g in "ABCD"]
for r in range(4):
    igrid.append(f'<p style="font-size:24px; color:{MUTED}">Inner {r + 1}</p>')
    igrid += [icell(g == r) for g in range(4)]
add("crossvalidation", header("Checking", "Nested cross-validation") + f"""
<div style="display:flex; gap:80px; align-items:center">
<div style="display:flex; flex-direction:column; gap:20px">
{small_caps("Inside one round: the 480 training people")}
<div style="display:grid; grid-template-columns:120px repeat(4, 120px); gap:8px">
{chr(10).join(igrid)}
</div>
{foot("4 groups of 120 training people")}
</div>
<div style="width:420px; display:flex; flex-direction:column; gap:12px">
{box("Pick", "route · model · mean of 1–3", "trained", width=None)}
{ARROW_D}
{box("Refit on all 480", "the winner only", width=None)}
{ARROW_D}
{box("Judge 120 people", "never seen, never used", "out", width=None)}
</div>
<div style="flex:1; display:flex; flex-direction:column; gap:40px">
{big("0", "choices made on the people being judged")}
{big(str(N_CHOICES), "separate choices per target", size=64, color=CREAM)}
</div>
</div>""",
    f"""The scores are honest only if no choice is made on the people being judged. So inside each round, the 480
    training people are split again into four groups: every route, model and averaging rule is fitted on three groups
    and checked on the fourth, four times over. Only the winner is refitted on all 480 people, and then it predicts the
    120 judged people. This happens separately in each of the {N_CHOICES} rounds, five folds times three shuffles, so
    every target gets {N_CHOICES} separate choices, and none of them ever saw the people it is scored on.""")

# ------------------------------------------------------------------ 17 results
rcard = lambda k, v, unit, line, sub: card(k, f'''<div style="display:flex; align-items:baseline; gap:16px"><p style="{SERIF}; font-size:120px; font-weight:400; line-height:1; color:{GOLD}">{v}</p>{f'<p style="font-size:28px; color:{SEC}">{unit}</p>' if unit else ''}</div>
<p style="font-size:28px; color:{CREAM}">{line}</p>
<p style="font-size:24px; color:{MUTED}">{sub}</p>''')
add("results", header("Results", "Results on all 600 people") + f"""
<div style="display:flex; gap:40px">
{rcard("Age", f"{MAE['v2', 'age']:.2f}", "years", f"{pct(SKILL['v2', 'age'], 0)} vs median guess", f"Guess {MAE['guess', 'age']:.2f} · v1 {MAE['v1', 'age']:.2f}")}
{rcard("BMI", f"{MAE['v2', 'bmi']:.2f}", "kg/m²", f"{pct(SKILL['v2', 'bmi'], 0)} vs median guess", f"Guess {MAE['guess', 'bmi']:.2f} · v1 {MAE['v1', 'bmi']:.2f}")}
{rcard("Recorded sex", f"{MAE['v2', 'sex_acc'] * 100:.1f}%", "", f"accuracy · AUC {MAE['v2', 'sex_auc']:.3f}", f"Always F {MAE['guess', 'sex_acc'] * 100:.1f}% · v1 {MAE['v1', 'sex_acc'] * 100:.1f}%")}
</div>
<div style="flex:1"></div>
{foot("Average error (MAE) · nested cross-validation, 600 people × 3 shuffles")}""",
    f"""These are the results for the appearance targets, over all 600 people and three shuffles. Age is predicted
    within {MAE['v2', 'age']:.2f} years on average, {SKILL['v2', 'age'] * 100:.0f}% better than guessing the median and
    better than our first version at {MAE['v1', 'age']:.2f}; the gain is in the 18-to-24 group, now 1.09 years. BMI
    has an average error of {MAE['v2', 'bmi']:.2f} kg/m², {SKILL['v2', 'bmi'] * 100:.0f}% better than guessing, and
    recorded sex is {MAE['v2', 'sex_acc'] * 100:.1f}% accurate. The few sex errors are always the same {sex_wrong // 2}
    people, both sessions, in every model, so their recorded sex is worth checking.""")

# ------------------------------------------------------------------ 18 table
trs = []
for t in T[:-1]:
    hi_row = t in ("age", "bmi", "rigidity", "respiratory")
    rs = ' style="background:#1A2438"' if hi_row else ""
    c0 = CREAM if hi_row else SEC
    cv = f"color:{GOLD}; " if hi_row else ""
    trs.append(f'<tr{rs}><td style="color:{c0}; {TD}">{LABEL[t]} ({UNIT[t]})</td>'
               f'<td style="text-align:right; {"color:" + CREAM + "; " if hi_row else ""}{TD}">{num(MAE["guess", t])}</td>'
               f'<td style="text-align:right; {"color:" + CREAM + "; " if hi_row else ""}{TD}">{num(MAE["v1", t])}</td>'
               f'<td style="text-align:right; {cv}{TD}">{num(MAE["v2", t])}</td>'
               f'<td style="text-align:right; {cv}{TD}">{pct(SKILL["v2", t])}</td></tr>')
trs.append(f'<tr><td style="color:{SEC}; {TD}">Recorded sex (accuracy)</td>'
           f'<td style="text-align:right; {TD}">{MAE["guess", "sex_acc"] * 100:.1f}%</td>'
           f'<td style="text-align:right; {TD}">{MAE["v1", "sex_acc"] * 100:.1f}%</td>'
           f'<td style="text-align:right; color:{CREAM}; {TD}">{MAE["v2", "sex_acc"] * 100:.1f}%</td>'
           f'<td style="text-align:right; {TD}">—</td></tr>')
add("mae", header("Results", "Average error (MAE) for every target") + f"""
<div style="display:flex; flex-direction:column; gap:24px">
<table style="font-size:24px; color:#8794AA; border:1px solid {CARD_B}">
<tr><th style="width:40%; text-align:left; {TH}">Target (unit)</th><th style="width:15%; text-align:right; {TH}">Median guess</th><th style="width:15%; text-align:right; {TH}">First version</th><th style="width:15%; text-align:right; color:{CREAM}; font-weight:500; padding:9px 16px">Final model</th><th style="width:15%; text-align:right; {TH}">vs guess</th></tr>
{chr(10).join(trs)}
</table>
{foot("600 people × 3 shuffles · lower is better · vs guess: how much smaller the error is than always guessing the median")}
</div>""",
    f"""This table shows the average error for every target: always guessing the median, our first version, and the
    final model. The last column shows how much smaller the final error is than the median guess. Age and BMI are the
    big wins, at {SKILL['v2', 'age'] * 100:.0f}% and {SKILL['v2', 'bmi'] * 100:.0f}%. The camera pulse doubled the gain
    for arterial stiffness, from {SKILL['v1', 'rigidity'] * 100:.0f}% in the first version to
    {SKILL['v2', 'rigidity'] * 100:.0f}%, and took respiratory rate from the level of guessing to
    {SKILL['v2', 'respiratory'] * 100:.0f}%. For the other five targets all three columns are within a few percent of
    each other.""")

# ------------------------------------------------------------------ 19 what similar numbers mean
PHYS = ["saturation", "temperature", "glycated_hemoglobin", "cholesterol", "respiratory", "rigidity", "stress"]
clear = [t for t in PHYS if CI["v2", t][0] > 0]
add("meaning", header("Results", "What similar numbers mean") + f"""
<div style="display:flex; gap:96px; align-items:center">
{img("skill", "Skill of each target with 95% intervals for the final model and the first version: age and BMI are far above zero, arterial stiffness and respiratory rate clearly above zero, the other five intervals cross zero", 860, 617)}
<div style="flex:1; display:flex; flex-direction:column; gap:56px">
<div style="display:flex; flex-direction:column; gap:12px">
<p style="{SERIF}; font-size:48px; font-weight:400; line-height:1.15; color:{CREAM}">Error ≈ median guess</p>
<p style="font-size:28px; color:{SEC}">The model learned nothing for that target</p>
</div>
{big(f"{len(clear)} of 7", "physiological targets clearly beat the guess")}
{foot(" and ".join(LABEL[t].capitalize() if k == 0 else LABEL[t].lower() for k, t in enumerate(sorted(clear, key=lambda t: -SKILL['v2', t]))) + "; the other intervals cross 0")}
</div>
</div>""",
    f"""When a model's error is about the same as always guessing the median, it has learned nothing useful for that
    target. The chart shows each target's skill with its 95% interval, for the final model in blue and the first
    version in grey. Age and BMI are far above zero, and {len(clear)} physiological targets are clearly above zero
    too: arterial stiffness at {pct(SKILL['v2', 'rigidity'], 0)} and respiratory rate at
    {pct(SKILL['v2', 'respiratory'], 0)}. For oxygen saturation, HbA1c, cholesterol, stress and temperature, every
    interval still crosses zero, so none of them is clearly better than guessing.""")

# ------------------------------------------------------------------ 20 did the camera pulse help?
prow = lambda name, b, s, final=False: (  # noqa: E731
    f'<div style="display:flex; align-items:center; padding:18px 0; border-bottom:1px solid {CARD_B}">'
    f'<div style="flex:1; display:flex; align-items:center; gap:16px"><p style="font-size:28px; {"font-weight:500; color:" + GOLD if final else "color:" + SEC}">{name}</p>'
    + (f'<p style="font-size:24px; color:{GOLD}; border:1px solid {GOLD}; border-radius:40px; padding:2px 16px">final</p>' if final else "")
    + f'</div><p style="width:170px; text-align:right; font-size:28px; {"font-weight:500; color:" + GOLD if final else "color:" + SEC}">{b:.3f}</p>'
    f'<p style="width:170px; text-align:right; font-size:28px; {"font-weight:500; color:" + GOLD if final else "color:" + SEC}">{s:.3f}</p></div>')
gain_b = 1 - MAE["v2", "respiratory"] / MAE["v1", "respiratory"]
gain_s = 1 - MAE["v2", "rigidity"] / MAE["v1", "rigidity"]
add("physiology", header("Results", "Did the camera pulse help?") + f"""
<div style="display:flex; gap:96px; align-items:flex-start">
<div style="width:960px; display:flex; flex-direction:column; gap:16px">
<div style="display:flex; padding:0 0 12px; border-bottom:1px solid #3A4A66"><p style="flex:1; font-size:24px; color:{MUTED}">Video only · MAE</p><p style="width:170px; text-align:right; font-size:24px; color:{MUTED}">Breathing</p><p style="width:170px; text-align:right; font-size:24px; color:{MUTED}">Stiffness</p></div>
<div style="display:flex; flex-direction:column">
{prow("Median guess", MAE["guess", "respiratory"], MAE["guess", "rigidity"])}
{prow("First version (hand-made pulse)", MAE["v1", "respiratory"], MAE["v1", "rigidity"])}
{prow("Classic signals, redone", *ROUTES_PULSE["classic signals only"])}
{prow("A pulse network alone", *ROUTES_PULSE["RhythmFormer pulse alone"])}
{prow("Everything, chosen per target", MAE["v2", "respiratory"], MAE["v2", "rigidity"], final=True)}
</div>
{foot("Nested cross-validation, 600 people × 3 shuffles · lower is better")}
</div>
<div style="flex:1; display:flex; flex-direction:column; gap:48px">
{small_caps("Compared with the first version")}
{big(f"{gain_b * 100:.0f}%", "smaller breathing error", size=64, color=CREAM)}
{big(f"{gain_s * 100:.0f}%", "smaller stiffness error", size=64, color=CREAM)}
{foot("Adding RhythmFormer, the best pulse, moved no score by more than 0.004")}
</div>
</div>""",
    f"""Did the better pulse help the health targets? For breathing and stiffness, yes. The left side compares the
    median guess, our first version, the redone classic signals, one pulse network on its own, and the final model,
    which chooses among everything. Breathing comes mostly from the classic signals and stiffness mostly from the
    pulse networks; together, both errors are about {gain_b * 100:.0f}% smaller than in the first version. RhythmFormer,
    the most accurate pulse, changed no score by more than 0.004, because heart rate was already read within about one
    beat per minute.""")

# ------------------------------------------------------------------ 21 limits
lrow = lambda name, why, r=None, last=False: (  # noqa: E731
    f'<div style="display:flex; gap:32px; align-items:baseline; padding:13px 0; border-top:1px solid {CARD_B}'
    f'{"; border-bottom:1px solid " + CARD_B if last else ""}"><p style="width:420px; font-size:28px; font-weight:500; color:{CREAM}">{name}</p>'
    f'<p style="flex:1; font-size:28px; color:{SEC}">{why}</p>'
    + (f'<p style="width:200px; font-size:28px; font-weight:500; color:{GOLD}; text-align:right">{r}</p>' if r else "") + "</div>")
add("limits", header("Why", "Why video alone is not enough") + f"""
<div style="display:flex; flex-direction:column; gap:32px">
<div style="display:flex; flex-direction:column; gap:12px">
{small_caps("Needs another sensor or test · 5 targets")}
<div style="display:flex; flex-direction:column">
{lrow("Oxygen saturation", "Oximeters use infrared light; a webcam does not see it")}
{lrow("Temperature", "Body heat is infrared too")}
{lrow("HbA1c · Cholesterol", "Blood chemistry: needs a blood test")}
{lrow("Stress (PSM-25)", "A questionnaire about the past days, not a body signal", last=True)}
</div>
</div>
<div style="display:flex; flex-direction:column; gap:12px">
{small_caps("In the video, but limited · 2 targets")}
<div style="display:flex; flex-direction:column">
{lrow("Arterial stiffness", "The camera gets each beat's timing, not its shape: mostly heart rate", f"ρ = {STIFF_PARTIAL['video']:.2f}")}
{lrow("Respiratory rate", "The video's breathing rate barely matches the label; the gain is rest vs exercise", "|ρ| ≤ 0.11", last=True)}
</div>
<p style="font-size:24px; color:{MUTED}; padding:12px 0 0 0">ρ = rank correlation (0 = no link, 1 = perfect): stiffness prediction with heart rate held fixed; video breathing rate vs label</p>
</div>
</div>""",
    f"""Why does video alone still not work for most physiological targets? Five need another sensor or test: oximeters
    use infrared light that a webcam cannot see, body heat is infrared too, HbA1c and cholesterol are blood chemistry,
    and stress is a questionnaire about the past days. The two that improved have limits too. The camera pulse
    reproduces the timing of each beat but not its shape, so once heart rate is held fixed, the stiffness prediction
    keeps a correlation of only {STIFF_PARTIAL['video']:.2f}. And the breathing rate read from the video barely
    matches the recorded one, because the count was taken at another moment; the gain comes from telling rest from
    after exercise.""")

# ------------------------------------------------------------------ 22 contact sensors
others = [t for t in T[:-1] if t not in ("rigidity", "respiratory")]
max_other = max(abs(MAE["mm", t] - MAE["v2", t]) for t in others)
scard = lambda k, a, b, line, gold=False: card(k, f'''<p style="{SERIF}; font-size:64px; font-weight:400; line-height:1.1; color:{CREAM}; white-space:nowrap">{a} <span style="color:{MUTED}">→</span> <span style="color:{GOLD}">{b}</span></p>
<p style="height:80px; font-size:28px; color:{SEC}">{line}</p>''', gold=gold, gap=24, justify="space-between")  # noqa: E731
add("improve", header("What we tested", "Adding contact sensors") + f"""
<div style="display:flex; gap:40px; flex:1">
{scard("Arterial stiffness", f"{MAE['v2', 'rigidity']:.2f}", f"{MAE['mm', 'rigidity']:.2f}", f"{pct(SKILL['v2', 'rigidity'], 0)} → {pct(SKILL['mm', 'rigidity'], 0)} vs guess · beyond heart rate ρ {STIFF_PARTIAL['video']:.2f} → {STIFF_PARTIAL['sensors']:.2f}", gold=True)}
{scard("Respiratory rate", f"{MAE['v2', 'respiratory']:.2f}", f"{MAE['mm', 'respiratory']:.2f}", f"{pct(SKILL['v2', 'respiratory'], 0)} → {pct(SKILL['mm', 'respiratory'], 0)} vs guess")}
{card("Other 8 targets", f'''<p style="{SERIF}; font-size:64px; font-weight:400; line-height:1.1; color:{CREAM}">No change</p>
<p style="height:80px; font-size:28px; color:{SEC}">every error within {max_other:.3f}</p>''', kicker_color=MUTED, gap=24, justify="space-between")}
</div>
<p style="font-size:28px; color:{SEC}">Video + finger PPG + ECG as extra inputs, same people and rules. An upper bound: stiffness itself is measured with a pulse sensor.</p>""",
    f"""As planned in the last update, we tested the sensor data as extra inputs: the finger PPG and the ECG recorded
    with each video. They help exactly the two targets that the camera pulse helped: stiffness drops to
    {MAE['mm', 'rigidity']:.2f} and breathing to {MAE['mm', 'respiratory']:.2f}, {SKILL['mm', 'rigidity'] * 100:.0f}% and
    {SKILL['mm', 'respiratory'] * 100:.0f}% better than guessing. Unlike the camera, the contact sensors keep
    information beyond heart rate, a correlation of {STIFF_PARTIAL['sensors']:.2f} for stiffness. The other eight
    targets do not change, so for them neither the video nor these sensors are enough. BIA was not tested.""")

# ------------------------------------------------------------------ 23 summary
srow = lambda a, b: (f'<div style="display:flex; justify-content:space-between"><p style="font-size:28px; color:{SEC}">{a}</p>'  # noqa: E731
                     f'<p style="font-size:28px; color:{GOLD}">{b}</p></div>')
scard2 = lambda k, h, inner, gold=False: card(k, f'''<div style="display:flex; flex-direction:column; gap:24px">
<p style="{SERIF}; font-size:48px; font-weight:400; line-height:1.15; color:{CREAM}">{h}</p>
{inner}
</div>''', gold=gold, kicker_color=None if gold else MUTED, justify="space-between")
add("summary", header("Summary", "Where we are") + f"""
<div style="display:flex; gap:40px; flex:1">
{scard2("Learned", "Age, BMI, sex", f'<div style="height:136px; display:flex; flex-direction:column; gap:8px">{srow("Age", f"{MAE["v2", "age"]:.2f} years")}{srow("BMI", f"{MAE["v2", "bmi"]:.2f} kg/m²")}{srow("Sex", f"{MAE["v2", "sex_acc"] * 100:.1f}%")}</div>', gold=True)}
{scard2("Partly", "Pulse targets", f'<p style="height:136px; font-size:28px; color:{SEC}">Stiffness {pct(SKILL["v2", "rigidity"], 0)}, breathing {pct(SKILL["v2", "respiratory"], 0)} vs guess: mostly heart rate and rest vs exercise</p>')}
{scard2("Not yet", "Five targets", f'<p style="height:136px; font-size:28px; color:{SEC}">SpO2, temperature, HbA1c, cholesterol, stress: at the median guess</p>')}
</div>
<p style="font-size:28px; color:{SEC}"><span style="color:{GOLD}">Next:</span> a pulse network that keeps the beat's shape · test on de-identified video · check {sex_wrong // 2} sex labels</p>""",
    f"""To summarise: from a face video, the final model reads age, BMI and recorded sex well, with errors of
    {MAE['v2', 'age']:.2f} years and {MAE['v2', 'bmi']:.2f} kg/m² and {MAE['v2', 'sex_acc'] * 100:.1f}% accuracy. A
    camera pulse taught by the finger sensor now gives real but modest gains for stiffness and breathing, mostly
    through heart rate and the difference between rest and exercise. Five targets remain at the median guess, because
    a webcam cannot see what they depend on. Next we would train a pulse network that keeps the shape of each beat,
    test the pipeline on de-identified video for privacy, and check the {sex_wrong // 2} sex labels that every model
    disagrees with. Thank you, I am happy to take questions.""")

# ================================================================== page (the Face_AI deck's shell and controls)
HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face AI Final Model</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&amp;family=Source+Serif+4:wght@300..700&amp;display=swap">
<style>
  html, body { margin: 0; height: 100%; background: #080C14; overflow: hidden; }
  #stage { position: fixed; inset: 0; display: flex; align-items: center; justify-content: center; }
  body.notes #stage { bottom: 28vh; }
  #deck { position: relative; width: 1920px; height: 1080px; flex: none; transform-origin: center center; }
  #deck > section { position: absolute; left: 0; top: 0; width: 1920px; height: 1080px; overflow: hidden; }
  #deck > section:not(.active) { display: none !important; }
  #deck section *, #deck section *::before, #deck section *::after { box-sizing: border-box; margin: 0; }
  #deck > section { box-sizing: border-box; }
  #deck h1 { font-size: 96px; font-weight: 600; line-height: 1.1; }
  #deck h2 { font-size: 64px; font-weight: 600; line-height: 1.15; }
  #deck h3 { font-size: 44px; font-weight: 600; line-height: 1.2; }
  #deck p, #deck li { line-height: 1.4; }
  #deck b { font-weight: 600; }
  #deck ul { padding-left: 1.2em; }
  #deck table { width: 100%; border-collapse: collapse; border: none !important; }
  #deck th, #deck td { border: none; border-bottom: 1px solid #26324A; padding: 0.35em 0.6em; text-align: left;
                        vertical-align: middle; font-weight: 400; line-height: 1.4; }
  #deck th { font-weight: 600; border-bottom-color: #3A4A66; }
  #deck img, #deck svg { display: block; flex-shrink: 0; }
  #help { position: fixed; right: 16px; bottom: 10px; color: #8D98AB; font: 14px 'IBM Plex Sans', Arial, sans-serif; user-select: none; }
  #notes { display: none; position: fixed; left: 0; right: 0; bottom: 0; height: 28vh; overflow: auto; box-sizing: border-box;
            padding: 16px 40px 24px; background: #0A0F18; border-top: 1px solid #26324A; color: #D9DEE7;
            font: 20px/1.55 'IBM Plex Sans', Arial, sans-serif; }
  #notes-head { margin: 0 0 8px; color: #E2B45C; font-size: 13px; font-weight: 500; letter-spacing: 3px; text-transform: uppercase; }
  #notes-text { margin: 0; max-width: 1400px; }
  body.notes #notes { display: block; }
  body.notes #help { display: none; }
  @media print {
    @page { size: 1920px 1080px; margin: 0; }
    html, body { height: auto; overflow: visible; background: none; }
    #stage { position: static; display: block; }
    #deck { width: auto; height: auto; transform: none !important; }
    #deck > section { position: relative; display: flex !important; break-after: page; }
    #help, #notes { display: none !important; }
  }
</style>
</head>
<body>
<div id="stage"><div id="deck">
"""
SCRIPT = """  const deck = document.getElementById('deck'), stage = document.getElementById('stage'), count = document.getElementById('count');
  const slides = Array.from(deck.children);
  let current = 0;
  function fit() { const r = stage.getBoundingClientRect(); deck.style.transform = 'scale(' + Math.min(r.width / 1920, r.height / 1080) + ')'; }
  function show(n) {
    current = Math.max(0, Math.min(slides.length - 1, n));
    slides.forEach((s, k) => s.classList.toggle('active', k === current));
    count.textContent = (current + 1) + ' / ' + slides.length;
    document.getElementById('notes-head').textContent = 'Notes · slide ' + (current + 1) + ' / ' + slides.length;
    document.getElementById('notes-text').textContent = NOTES[current] || '(no notes for this slide)';
    history.replaceState(null, '', location.pathname + location.search + '#' + (current + 1));
  }
  function toggleNotes() { document.body.classList.toggle('notes'); fit(); }
  addEventListener('keydown', e => {
    if (['ArrowRight', 'ArrowDown', 'PageDown', ' ', 'Enter'].includes(e.key)) { show(current + 1); e.preventDefault(); }
    else if (['ArrowLeft', 'ArrowUp', 'PageUp', 'Backspace'].includes(e.key)) { show(current - 1); e.preventDefault(); }
    else if (e.key === 'Home') show(0);
    else if (e.key === 'End') show(slides.length - 1);
    else if (e.key === 'n' || e.key === 'N') toggleNotes();
    else if (e.key === 'f' || e.key === 'F') { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(); }
  });
  addEventListener('click', e => { if (!e.target.closest('#notes')) show(current + (e.clientX > innerWidth / 2 ? 1 : -1)); });
  addEventListener('resize', fit);
  if (new URLSearchParams(location.search).has('notes')) document.body.classList.add('notes');
  fit();
  show((parseInt(location.hash.slice(1), 10) || 1) - 1);
"""


def main():
    assert len(SLIDES) == len(NOTES) == 23, (len(SLIDES), len(NOTES))
    page = (HEAD + "\n".join(SLIDES) + "\n</div></div>\n"
            '<div id="notes"><p id="notes-head"></p><p id="notes-text"></p></div>\n'
            '<div id="help"><span id="count"></span> · ← → or click to move · N notes · F full screen</div>\n'
            "<script>\n  const NOTES = " + json.dumps(NOTES, ensure_ascii=False).replace("</", "<\\/") + ";\n"
            + SCRIPT + "</script>\n</body>\n</html>\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({len(page) / 1024:.0f} KB, {len(SLIDES)} slides)")
    print("snippet:", SNIP["key"], round(SNIP["r"], 3), "| sex errors per shuffle:", sex_wrong,
          "| clear physiological targets:", clear)
    print("camera HR (MAE, within 5):", {k: (round(v["mae"], 2), round(v["w5"], 3)) for k, v in HR.items()})
    print("MAE v2/v1/guess/mm:", {t: tuple(round(MAE[n, t], 3) for n in ("v2", "v1", "guess", "mm")) for t in T[:-1]})
    print("skill v2 [CI]:", {t: (round(SKILL["v2", t] * 100, 1), *(round(c * 100, 1) for c in CI["v2", t])) for t in T[:-1]})
    print("sex acc/auc:", round(MAE["v2", "sex_acc"], 4), round(MAE["v2", "sex_auc"], 4), "| route picks:", dict(KINDS))


if __name__ == "__main__":
    main()
