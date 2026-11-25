"""Code cells of FaceAI_v2.ipynb that read saved results (kept apart from the step texts)."""

STEP3_CODE = r'''
# ▶ Step 3.1 — Camera heart rate vs the finger sensor (first minute, all 1,200 recordings)
# Every heart rate here uses the SAME estimator as the finger reference (Welch, 17-s segments); the per-recording
# values are saved by scripts/make_v2_report_tables.py in results/v2/camera_hr_first_minute.csv.
import rppg_deep as R
P = np.load(PROJECT / "data" / "cache" / "ppg_sync.npz")["ppg"]
hr = pd.read_csv(V2 / "camera_hr_first_minute.csv")
finger = hr["finger"].to_numpy()
import textwrap
cams = {c: hr[c].to_numpy() for c in hr.columns if c not in ("key", "patient_id", "step", "finger")}
rows = {}
for label, h in cams.items():
    ok = np.isfinite(h) & np.isfinite(finger)
    e = np.abs(h[ok] - finger[ok])
    rows[label] = {"MAE (bpm)": e.mean(), "median error": np.median(e), "within 5 bpm": np.mean(e <= 5),
                   "within 10 bpm": np.mean(e <= 10), "r with finger": np.corrcoef(h[ok], finger[ok])[0, 1]}
display(pd.DataFrame(rows).T.round(3))
ncol = 3 if len(cams) > 4 else len(cams)
nrow = int(np.ceil(len(cams) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.1 * nrow), squeeze=False)
axes = axes.ravel()
for ax, (label, h) in zip(axes, cams.items()):
    ok = np.isfinite(h) & np.isfinite(finger)
    ax.scatter(finger[ok], h[ok], s=8, alpha=0.45, color=BLUE, edgecolor="none")
    ax.plot([40, 160], [40, 160], color=INK, lw=1); ax.set_xlim(40, 160); ax.set_ylim(40, 160)
    ax.set_title(textwrap.fill(label.split(" (")[0], 40), fontsize=9); ax.set_xlabel("finger heart rate (bpm)")
    ax.text(0.03, 0.93, f"within 5 bpm: {rows[label]['within 5 bpm']:.0%}" + "\n" + f"r = {rows[label]['r with finger']:.2f}",
            transform=ax.transAxes, fontsize=8, va="top")
for ax in axes[len(cams):]:
    ax.axis("off")
for ax in axes[::ncol]:
    ax.set_ylabel("camera heart rate (bpm)")
fig.suptitle("Camera pulse vs finger sensor: points on the line are right", y=1.02); fig.tight_layout(); savefig(fig, "v2_01_camera_vs_finger"); plt.show()
'''

STEP4_CODE = r'''
# ▶ Step 4.1 — The pulse networks: accuracy on people they never saw, and one example
import make_v2_report_tables as T
NETS = {"tracenet": "trace network", "physnet_diff": "PhysNet",
        "rhythmformer_pure_zeroshot": "RhythmFormer, published PURE weights, no MCD training",
        "rhythmformer_pure": "RhythmFormer, PURE weights fine-tuned on MCD"}
acc = {}
for m, name in NETS.items():
    f = PROJECT / "data" / "cache_v2" / "rppg" / m / "metrics.json"
    if f.exists():
        mm = json.loads(f.read_text())
        acc[(name, "all 1,200")], acc[(name, "clean finger pulse")] = mm["all"], mm["clean_ppg"]
display(Markdown("**Camera pulse vs finger, recording by recording** (each recording scored by a network that never saw that person)"),
        pd.DataFrame(acc).T.round(3))
if (PROJECT / "data" / "cache_v2" / "rppg" / "rhythmformer_pure" / "wave.npy").exists():
    display(Markdown("**Fine-tuned RhythmFormer minus the other two networks** (per person; 95% interval from 2,000 bootstrap resamples of people)"),
            Markdown(T.pulse_paired_table()))
i = int(np.where(C.df["key"] == "1020_FullHDwebcam_before")[0][0])
t = np.arange(300) / 30
fin = R.bandpass(P[i]); fin = fin[[max(j - R.LAG, 0) for j in range(len(fin))]]
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(t, fin[600:900] / fin[600:900].std(), color=ORANGE, lw=1.6, label="finger pulse (aligned)")
for m, c in (("tracenet", GREEN), ("physnet_diff", BLUE), ("rhythmformer_pure", VIOLET)):
    f = PROJECT / "data" / "cache_v2" / "rppg" / m / "wave.npy"
    if f.exists():
        w = R.bandpass(np.load(f)[i]); ax.plot(t, w[600:900] / w[600:900].std(), color=c, lw=1.1, label=f"camera ({NETS[m].split(',')[0]})")
ax.set_xlabel("seconds 20-30 of the video"); ax.legend(loc="upper right", ncol=4); ax.set_title("One recording the networks never saw: camera pulse vs finger pulse")
fig.tight_layout(); savefig(fig, "v2_02_pulse_example"); plt.show()
'''

STEP5_CODE = r'''
# ▶ Step 5.1 — Breathing and exercise state from the video: what the new numbers know
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
def feats(file):
    with np.load(PROJECT / "features" / file) as z:
        return z["features"].astype(float), list(z["names"])
B, bn = feats("v2_breathing.npz")
y, after = C.df["respiratory"].to_numpy(), C.df["after"].to_numpy()
corr = pd.Series({n: spearmanr(B[:, j], y, nan_policy="omit")[0] for j, n in enumerate(bn) if "breaths/min" in n or "rate" in n})
display(Markdown("**Spearman correlation with the doctor's breathing count** (video breathing rates)"), corr.sort_values().round(3).to_frame("ρ").T)
def state_auc(X):
    X = np.where(np.isfinite(X), X, np.nanmedian(X, 0)); fold, p = C.session_folds(0), np.zeros(len(X))
    best = 0
    for Cc in (0.01, 0.1, 1.0):
        for k in range(5):
            tr, te = fold != k, fold == k
            sc = StandardScaler().fit(X[tr])
            p[te] = LogisticRegression(C=Cc, max_iter=5000).fit(sc.transform(X[tr]), after[tr]).predict_proba(sc.transform(X[te]))[:, 1]
        best = max(best, roc_auc_score(after, p))
    return best
sets = {"v1 camera pulse features": [feats("rppg_v2.npz")[0]], "v2 classic pulse (3 min)": [feats("v2_pulse_classic.npz")[0]],
        "v2 breathing motion": [B]}
for m in ("tracenet", "physnet_diff", "rhythmformer_pure"):
    if (PROJECT / "features" / f"v2_pulse_deep_{m}.npz").exists():
        sets[f"v2 deep pulse ({m})"] = [feats(f"v2_pulse_deep_{m}.npz")[0]]
sets["all v2 signal features"] = sum(sets.values(), [])
auc = pd.Series({k: state_auc(np.column_stack(v)) for k, v in sets.items()})
display(Markdown("**Can the video tell 'just exercised' from 'rested'?** (ROC AUC; finger PPG + ECG reach 0.74)"), auc.round(3).to_frame("AUC").T)
'''

STEP6_CODE = r'''
# ▶ Step 6.1 — Face routes for age, BMI and sex: v1 embeddings vs sharper crops, stronger encoders, per-crop networks
import v2_nested as N
rows = {}
for f in sorted((V2 / "routes").glob("*_r0.npz")):
    route = f.stem[:-3]
    if not (route in ("dino", "clip", "appearance") or route.startswith(("app_", "mlp"))):
        continue
    if not all((V2 / "routes" / f"{route}_r{r}.npz").exists() for r in range(3)):
        continue
    Pm = {r: N.load_route(route, r)[0] for r in range(3)}
    s = C.quick_scores(Pm, ["age", "bmi", "sex"])
    rows[route] = {"age MAE": s["age"][0], "age skill": s["age"][1], "BMI MAE": s["bmi"][0], "BMI skill": s["bmi"][1], "sex accuracy": s["sex"]}
face = pd.DataFrame(rows).T.sort_values("age MAE")
display(face.round(3))
'''

STEP7_CODE = r'''
# ▶ Step 7.1 — v2 specialists vs v1, per target (3 shuffles; interval = paired person bootstrap of v2 − v1 MAE)
tab = pd.read_csv(V2 / "tiers_metrics.csv")
show = {}
for tier, base in (("video_v2_ens", "video_specialists"), ("multimodal_v2_ens", "multimodal_specialists")):
    t = tab[tab.tier == tier].set_index("target")
    v1m = pd.read_csv(PROJECT / "results" / "metrics" / "regression_metrics.csv").query("model == @base").set_index("target")
    for tg in C.REG_TARGETS:
        r = t.loc[tg]
        show[(tier, NAME[tg])] = {"v1 MAE": v1m.loc[tg, "mae"], "v2 MAE": r["mae"], "v1 skill": v1m.loc[tg, "skill"], "v2 skill": r["skill"],
                                  "v2 − v1 [95% CI]": f"{r['vs_old']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]",
                                  "verdict": "v2 better" if r["hi"] < 0 else ("v1 better" if r["lo"] > 0 else "no clear difference")}
display(pd.DataFrame(show).T)
sx = tab[tab.target == "sex"].set_index("tier")["acc"]
display(Markdown(f"**Sex accuracy:** v2 video {sx.get('video_v2_ens', np.nan):.4f} | v2 multimodal {sx.get('multimodal_v2_ens', np.nan):.4f} (v1: 0.9933)"))
'''

STEP7B_CODE = r'''
# ▶ Step 7.2 — Graph: skill per target, v1 vs v2 (video only and with the contact sensors)
v1all = pd.read_csv(PROJECT / "results" / "metrics" / "regression_metrics.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, (tier, base, title) in zip(axes, (("video_v2_ens", "video_specialists", "Video only"), ("multimodal_v2_ens", "multimodal_specialists", "Video + PPG + ECG"))):
    a = v1all.query("model == @base").set_index("target").reindex(C.REG_TARGETS)
    b = tab[tab.tier == tier].set_index("target").reindex(C.REG_TARGETS)
    yy = np.arange(len(C.REG_TARGETS))
    ax.barh(yy + 0.2, a["skill"], height=0.38, color=GREY, label="v1")
    ax.barh(yy - 0.2, b["skill"], height=0.38, color=BLUE if tier.startswith("video") else VIOLET, label="v2")
    ax.axvline(0, color=INK, lw=1); ax.set_yticks(yy, [NAME[t] for t in C.REG_TARGETS]); ax.invert_yaxis()
    ax.set_xlabel("skill vs the median guess"); ax.set_title(title); ax.legend(loc="lower right")
fig.tight_layout(); savefig(fig, "v2_03_skill_v1_vs_v2"); plt.show()
sel = pd.read_csv(V2 / "tiers_selection.csv")
s = sel[sel.tier == "video_v2_ens"]
display(Markdown("**What the inner loop chose for the v2 video tier** (share of the 15 outer folds; `+` = an average of routes)"))
display(s.groupby("target")["routes"].agg(lambda x: x.value_counts(normalize=True).round(2).head(3).to_dict()).to_frame("most frequent choices"))
'''

STEP7C_CODE = r'''
# ▶ Step 7.3 — Does a better pulse help the predictions? Each pulse as a route, pulse-shape numbers, and the tiers at each stage
import make_v2_report_tables as T
display(Markdown("**Pulse routes on their own** (video only, nested CV, 3 shuffles; MAE (skill))"), Markdown(T.pulse_routes_table()))
if (V2 / "pulse_shape_routes.csv").exists():
    display(Markdown("**Pulse-shape numbers from the camera pulse:** do they follow the finger's (no labels), and do they help as routes?"),
            Markdown(T.shape_tables()))
if (V2 / "before_rhythmformer" / "tier_video_v2_ens.npz").exists():
    display(Markdown("**The v2 tiers at each stage:** without RhythmFormer → + RhythmFormer → + shape numbers (paired person bootstrap of each step)"),
            Markdown(T.late_additions_table()))
'''

STEP8_CODE = r'''
# ▶ Step 8.1 — Head-to-head on Face_AI's 90 test people (both models learned from the same 510 people)
h2h = pd.read_csv(V2 / "h2h.csv").set_index("target")
h2h["verdict"] = np.where(h2h["hi"] < 0, "Face_AI_2 v2 better", np.where(h2h["lo"] > 0, "Face_AI better", "no clear difference"))
display(h2h.rename(index=NAME).round(3))
extra = json.loads((V2 / "h2h_extra.json").read_text())
display(Markdown(f"**Sex:** wrong recordings — Face_AI_2 v2 {extra['sex']['faceai2_v2 wrong']}, Face_AI {extra['sex']['face_ai wrong']} (of {extra['sex']['of']})"))
display(Markdown("**Age error by age group** (MAE per person)"), pd.DataFrame(extra["ages"]).round(2))
'''

STEP9_CODE = r'''
# ▶ Step 9.1 — Table 1: the final v2 results on their own (every person scored by a model that never saw them)
MEASURE = {"saturation": "MAE (%)", "temperature": "MAE (°C)", "glycated_hemoglobin": "MAE (%)",
           "cholesterol": "MAE (mmol/L)", "respiratory": "MAE (breaths/min)", "rigidity": "MAE (index)",
           "stress": "MAE (PSM-25 points)", "age": "MAE (years)", "bmi": "MAE (kg/m²)", "sex": "accuracy"}
SHORT = {t: NAME[t].split(" (")[0] for t in C.ALL_TARGETS}


def load_tier(name):
    """The saved final predictions: one table (1,200 recordings × 10 targets) per shuffle."""
    with np.load(V2 / f"tier_{name}.npz") as z:
        return {int(k[1:]): z[k] for k in z.files}


def person_error(P, t):
    """One error per person: averaged over their 2 recordings and the 3 shuffles (sex: the share of wrong guesses)."""
    j = C.ALL_TARGETS.index(t)
    y = C.Y_nat[:, j]
    ok = np.isfinite(y)                                       # 2 people have no clear sex label: 598 people for sex
    err = (lambda p: ((p >= 0.5) != y).astype(float)) if t == "sex" else (lambda p: np.abs(p - y))
    per_shuffle = [pd.Series(err(P[r][:, j])[ok]).groupby(C.pids[ok]).mean() for r in sorted(P)]
    return pd.concat(per_shuffle, axis=1).mean(axis=1)


def resample(*errors, B=2000):
    """Redraw the people 2,000 times (the same redraws for every model) and average each model's errors."""
    n = len(errors[0])
    idx = np.random.default_rng(C.SEED).integers(0, n, size=(B, n))
    return [e.to_numpy()[idx].mean(axis=1) for e in errors]


def result_cell(P, t):
    e = person_error(P, t)
    lo, hi = np.percentile(resample(e)[0], [2.5, 97.5])
    return f"{1 - e.mean():.2%} [{1 - hi:.2%}, {1 - lo:.2%}]" if t == "sex" else f"{e.mean():.3f} [{lo:.3f}, {hi:.3f}]"


FINAL = {tier: load_tier(tier) for tier in ("video_v2_ens", "multimodal_v2_ens")}
saved = pd.read_csv(V2 / "tiers_metrics.csv").set_index(["tier", "target"])
for tier, P in FINAL.items():                                 # check: the same numbers as Step 7.1 and the saved CSV
    for t in C.ALL_TARGETS:
        e = person_error(P, t).mean()
        assert np.isclose(1 - e if t == "sex" else e, saved.loc[(tier, t), "acc" if t == "sex" else "mae"], rtol=0, atol=1e-9)
table1 = pd.DataFrame({SHORT[t]: {"measure": MEASURE[t],
                                  "video only (the face model)": result_cell(FINAL["video_v2_ens"], t),
                                  "video + finger PPG + ECG": result_cell(FINAL["multimodal_v2_ens"], t)}
                       for t in C.ALL_TARGETS}).T
display(Markdown("**Table 1 · Face_AI_2 v2, final results** (600 people; MAE: lower is better, accuracy: higher is "
                 "better; [ ] = 95% interval)"), table1)
'''

STEP9B_CODE = r'''
# ▶ Step 9.2 — Table 2: the final model next to the median guess and v1 (same 600 people, same folds, same 3 shuffles)
TIER = "video_v2_ens"                  # ← "multimodal_v2_ens" shows the same table for video + finger PPG + ECG
V1_NAME = "video_specialists" if TIER.startswith("video") else "multimodal_specialists"
GUESS = {r: C.oof_matrix("constant", r) for r in range(C.N_REPEATS)}   # the median guess; for sex: always "female"
V1 = {r: C.oof_matrix(V1_NAME, r) for r in range(C.N_REPEATS)}
rows = {}
for t in C.ALL_TARGETS:
    e2, eg, e1 = person_error(FINAL[TIER], t), person_error(GUESS, t), person_error(V1, t)
    b2, bg, b1 = resample(e2, eg, e1)
    cut = 1 - e2.mean() / eg.mean()                                    # "skill": how much smaller v2's error is
    cut_lo, cut_hi = np.percentile(1 - b2 / bg, [2.5, 97.5])
    d_lo, d_hi = np.percentile(b2 - b1, [2.5, 97.5])                   # v2 − v1
    if t != "sex":                                                     # check: the same interval as Step 7.1
        assert np.allclose([d_lo, d_hi], saved.loc[(TIER, t), ["lo", "hi"]].to_numpy(float), rtol=0, atol=1e-9)
    fmt = (lambda e: f"{1 - e.mean():.2%}") if t == "sex" else (lambda e: f"{e.mean():.3f}")
    rows[SHORT[t]] = {"measure": MEASURE[t], "median guess": fmt(eg), "v1": fmt(e1), "v2 (final)": fmt(e2),
                      "error reduction vs median guess [95% CI]": f"{cut:.1%} [{cut_lo:.1%}, {cut_hi:.1%}]",
                      "clearly beats the median guess?": "yes" if cut_lo > 0 else ("no, worse" if cut_hi < 0 else "no clear difference"),
                      "clearly better than v1?": "yes" if d_hi < 0 else ("no, v1 better" if d_lo > 0 else "no clear difference")}
label = "video only" if TIER.startswith("video") else "video + finger PPG + ECG"
display(Markdown(f"**Table 2 · Face_AI_2 v2 ({label}) vs the median guess and v1** (MAE: lower is better; accuracy: "
                 "higher is better; 'clearly' = the whole 95% interval is on the better side)"), pd.DataFrame(rows).T)
'''

APPENDIX_CODE = r"""
# ▶ Appendix — The official v2 results, and the line of results/v2/tiers_metrics.csv that holds each number
tm = pd.read_csv(V2 / "tiers_metrics.csv")
tm["line"] = tm.index + 2                                    # line 1 of the CSV is the header
cols = {}
for label, tier in (("Video only", "video_v2_ens"), ("Video + PPG + ECG", "multimodal_v2_ens")):
    t = tm[tm.tier == tier].set_index("target").reindex(C.ALL_TARGETS)
    cols[(label, "result")] = [f"{t.at[k, 'acc']:.2%}" if k == "sex" else f"{t.at[k, 'mae']:.3f}" for k in t.index]
    cols[(label, "line")] = t["line"].astype(int).tolist()
final = pd.DataFrame(cols, index=[NAME[k] + (" (accuracy)" if k == "sex" else "") for k in C.ALL_TARGETS])
display(Markdown("**Official v2 results:** average error (MAE) per target, accuracy for sex. `line` = the line of "
                 "`results/v2/tiers_metrics.csv` that holds the number (tiers `video_v2_ens` and `multimodal_v2_ens`)."))
display(final)
"""
