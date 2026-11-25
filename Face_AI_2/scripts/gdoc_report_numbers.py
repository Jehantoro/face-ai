"""Every number for the "Week 5-7" tab of the group's weekly-report Google Doc: Face_AI_2 v2, the final model.

Reads saved results only (nothing is trained or re-run). Writes reports/gdoc_week5-7/numbers.json, plus
recordings.csv (one row per recording: labels, shuffle-0 answers, flags) and sens_spec.csv, for the figure script.

    .venv/Scripts/python.exe -X utf8 scripts/gdoc_report_numbers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_core as C                                                        # noqa: E402

V2 = C.RESULTS / "v2"
OUT = C.PROJECT / "reports" / "gdoc_week5-7"
OUT.mkdir(parents=True, exist_ok=True)
FACE_AI = C.PROJECT.parent / "Face_AI"
B, RNG_SEED = 2000, C.SEED
TG = C.REG_TARGETS
NUM = {}


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    return x


def load_tier(name, folder=V2):
    with np.load(folder / f"tier_{name}.npz") as z:
        return {int(k[1:]): z[k] for k in z.files}


df = C.df
Y = C.Y_nat
pids = C.pids
J = {t: C.ALL_TARGETS.index(t) for t in C.ALL_TARGETS}

# ------------------------------------------------------------------ 1. the data
raw = pd.read_csv(C.DB_CSV, dtype={"patient_id": str})
people = df.groupby("patient_id").first()
age_p = people["age"]
NUM["data"] = {
    "db_rows": len(raw), "cameras": sorted(raw["camera"].unique()), "recordings": len(df), "people": int(df.patient_id.nunique()),
    "women": int((people["sex"] == "F").sum()), "men": int((people["sex"] == "M").sum()),
    "sex_hidden_recordings": int(np.isnan(df["sex_label"]).sum()),
    "sex_hidden_people": int(df.loc[df["sex_label"].isna(), "patient_id"].nunique()),
    "age_18_24": int(((age_p >= 18) & (age_p <= 24)).sum()), "age_over_24": int((age_p > 24).sum()),
    "age_over_30": int((age_p > 30).sum()), "age_40_plus": int((age_p >= 40).sum()), "age_max": float(age_p.max()),
}
MEASURED = {"saturation": "finger pulse oximeter", "temperature": "non-contact thermometer",
            "glycated_hemoglobin": "finger-prick test", "cholesterol": "finger-prick test",
            "respiratory": "counted by a doctor", "rigidity": "photoplethysmograph index",
            "stress": "PSM-25 questionnaire (item mean)", "age": "records", "bmi": "scale + height", "sex": "records"}
before = df[df.step == "before"].set_index("patient_id")
after = df[df.step == "after"].set_index("patient_id")
common = before.index.intersection(after.index)
lab = {}
for t in TG:
    v = df[t]
    b, a = before.loc[common, t], after.loc[common, t]
    same = float((np.isclose(b, a)).mean())
    rho = float(spearmanr(b, a)[0])
    lab[t] = {"measured": MEASURED[t], "median": float(v.median()), "mean": float(v.mean()), "sd": float(v.std()),
              "min": float(v.min()), "max": float(v.max()), "same_both_sessions": same, "retest_rho": rho,
              "mean_before": float(b.mean()), "mean_after": float(a.mean())}
lab["saturation"]["share_98_99"] = float(df["saturation"].isin([98, 99]).mean())
lab["temperature"]["share_36_5_to_36_7"] = float(df["temperature"].between(36.45, 36.75).mean())
lab["temperature"]["share_36_6"] = float(np.isclose(df["temperature"], 36.6).mean())
s_ok = df["sex_label"].notna()
lab["sex"] = {"measured": "records", "same_both_sessions": float(1 - NUM["data"]["sex_hidden_people"] / NUM["data"]["people"])}
NUM["labels"] = lab
NUM["data"]["extreme"] = {"hba1c_max": float(df.glycated_hemoglobin.max()), "chol_min": float(df.cholesterol.min()),
                          "chol_min_person": df.loc[df.cholesterol.idxmin(), "patient_id"],
                          "stiff_max": float(df.rigidity.max()),
                          "pulse_before_db": float(before["pulse"].mean()), "pulse_after_db": float(after["pulse"].mean())}

# folds
folds = pd.read_csv(C.RESULTS / "folds.csv", dtype={"patient_id": str})
fb = []
for r in range(C.N_REPEATS):
    f = folds[folds.repeat == r].set_index("patient_id")["fold"]
    pp = people.assign(fold=f.reindex(people.index).to_numpy())
    g = pp.groupby("fold")
    fb.append(pd.DataFrame({"people": g.size(), "men": g["sex"].apply(lambda s: int((s == "M").sum())),
                            "over_30": g["age"].apply(lambda s: int((s > 30).sum())), "mean_age": g["age"].mean()}))
NUM["folds_shuffle0"] = fb[0].reset_index().to_dict("records")
NUM["folds_range"] = {"men": [int(min(x.men.min() for x in fb)), int(max(x.men.max() for x in fb))],
                      "over_30": [int(min(x.over_30.min() for x in fb)), int(max(x.over_30.max() for x in fb))]}

# ------------------------------------------------------------------ 2. the video pass
log = pd.read_csv(C.CACHE2 / "log.csv").set_index("key").loc[df["key"]]
q = np.load(C.CACHE2 / "ppg_quality.npy")
NUM["video_pass"] = {
    "ok": int((log.status == "ok").sum()), "frames_total": int(log.kept.sum()),
    "frames_min": int(log.kept.min()), "frames_median": float(log.kept.median()), "frames_max": int(log.kept.max()),
    "duration_min": float(log.duration_s.min()), "duration_median": float(log.duration_s.median()),
    "duration_max": float(log.duration_s.max()),
    "clip_valid_min": float(log.clip_valid.min()), "grid_valid_min": float(log.grid_valid.min()),
    "grid_valid_mean": float(log.grid_valid.mean()),
    "time_check": log.time_check.value_counts().to_dict(),
    "skin_rule_fallback_videos": int((~log.skin_rule.str.split(",").apply(lambda s: all(x == "skin" for x in s))).sum()),
    "cpu_seconds_total": float(log.seconds.sum()),
    "finger_clean": int((q[:, 0] > 0).sum()), "finger_noisy": int((q[:, 0] <= 0).sum()), "finger_flat": int(np.isnan(q[:, 0]).sum()),
}

# ------------------------------------------------------------------ 3. pulse networks: training logs and sizes
import torch                                                                # noqa: E402


def n_params(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return int(sum(v.numel() for k, v in sd.items() if not any(s in k for s in ("running_", "num_batches"))))


nets = {}
for m, name in (("tracenet", "trace network"), ("physnet_diff", "PhysNet"), ("rhythmformer_pure", "RhythmFormer (fine-tuned)")):
    d = C.CACHE2 / "rppg" / m
    met = json.loads((d / "metrics.json").read_text())
    lg = met["log"] if m == "tracenet" else json.loads((d / "log.json").read_text())
    rows = []
    for e in lg:
        if e.get("epoch") == "test":
            continue
        fold = e.get("fold", int(str(e.get("tag", "fold 0")).split()[-1]) if "tag" in e else 0)
        rows.append({"fold": fold, "epoch": int(e["epoch"]), "loss": e["loss"], "val_hr_mae": e["val_hr_mae"],
                     "val_within5": e["val_within5"], "val_wave_r": e["val_wave_r_median"], "seconds": e.get("seconds")})
    L = pd.DataFrame(rows)
    L["score"] = L.val_wave_r - L.val_hr_mae / 20
    kept = L.loc[L.groupby("fold")["score"].idxmax(), ["fold", "epoch"]].set_index("fold")["epoch"].to_dict()
    tests = [e for e in lg if e.get("epoch") == "test"]
    nets[m] = {"name": name, "params": n_params(d / "fold0.pt"), "log": L.to_dict("records"), "kept_epoch": kept,
               "epochs_run": int(L.epoch.max()), "all": met["all"], "clean": met["clean_ppg"],
               "fold_test": [{k: v for k, v in t.items()} for t in tests],
               "train_minutes": None if L.seconds.isna().all() else float(L.seconds.sum() / 60)}
z = json.loads((C.CACHE2 / "rppg" / "rhythmformer_pure_zeroshot" / "metrics.json").read_text())
nets["rhythmformer_zeroshot"] = {"all": z["all"], "clean": z["clean_ppg"]}
NUM["pulse_nets"] = nets

# camera heart rate vs finger (first minute; the saved per-recording file)
hr = pd.read_csv(V2 / "camera_hr_first_minute.csv")
finger = hr["finger"].to_numpy()
stiff = df.set_index("key").loc[hr["key"], "rigidity"].to_numpy()
cam = {}
for col in [c for c in hr.columns if c not in ("key", "patient_id", "step", "finger")]:
    h = hr[col].to_numpy()
    ok = np.isfinite(h) & np.isfinite(finger)
    e = np.abs(h[ok] - finger[ok])
    cam[col] = {"mae": float(e.mean()), "median": float(np.median(e)), "within5": float(np.mean(e <= 5)),
                "within10": float(np.mean(e <= 10)), "r": float(np.corrcoef(h[ok], finger[ok])[0, 1]),
                "rho_stiffness": float(spearmanr(h[ok], stiff[ok])[0]), "n": int(ok.sum()),
                "near_150": int(np.sum((h[ok] > 140) & (h[ok] < 160) & (finger[ok] < 120)))}
cam["finger"] = {"rho_stiffness": float(spearmanr(finger, stiff, nan_policy="omit")[0])}
NUM["camera_hr"] = cam
NUM["finger_hr_before_after"] = {"before": float(np.nanmean(finger[hr.step.to_numpy() == "before"])),
                                 "after": float(np.nanmean(finger[hr.step.to_numpy() == "after"]))}

# ------------------------------------------------------------------ 4. final results (Step 9 of FaceAI_v2.ipynb)
FINAL = {t: load_tier(t) for t in ("video_v2_ens", "multimodal_v2_ens")}
GUESS = {r: C.oof_matrix("constant", r) for r in range(C.N_REPEATS)}
V1 = {"video": {r: C.oof_matrix("video_specialists", r) for r in range(3)},
      "multimodal": {r: C.oof_matrix("multimodal_specialists", r) for r in range(3)}}


def person_error(P, t, repeats=None):
    j = J[t]
    y = Y[:, j]
    ok = np.isfinite(y)
    err = (lambda p: ((p >= 0.5) != y).astype(float)) if t == "sex" else (lambda p: np.abs(p - y))
    reps = sorted(P) if repeats is None else repeats
    return pd.concat([pd.Series(err(P[r][:, j])[ok]).groupby(pids[ok]).mean() for r in reps], axis=1).mean(axis=1)


def resample(*errors):
    n = len(errors[0])
    idx = np.random.default_rng(RNG_SEED).integers(0, n, size=(B, n))
    return [e.to_numpy()[idx].mean(axis=1) for e in errors]


def summary(P, t):
    e, eg = person_error(P, t), person_error(GUESS, t)
    b, bg = resample(e, eg)
    lo, hi = np.percentile(b, [2.5, 97.5])
    slo, shi = np.percentile(1 - b / bg, [2.5, 97.5])
    return {"mae": float(e.mean()), "lo": float(lo), "hi": float(hi), "guess": float(eg.mean()),
            "skill": float(1 - e.mean() / eg.mean()), "skill_lo": float(slo), "skill_hi": float(shi)}


def paired(Pa, Pb, t):
    ea, eb = person_error(Pa, t), person_error(Pb, t)
    ba, bb = resample(ea, eb)
    lo, hi = np.percentile(ba - bb, [2.5, 97.5])
    return {"diff": float(ea.mean() - eb.mean()), "lo": float(lo), "hi": float(hi)}


res = {}
for tier, P in (("v2_video", FINAL["video_v2_ens"]), ("v2_mm", FINAL["multimodal_v2_ens"]),
                ("v1_video", V1["video"]), ("v1_mm", V1["multimodal"])):
    res[tier] = {t: summary(P, t) for t in C.ALL_TARGETS}
res["v2_vs_v1_video"] = {t: paired(FINAL["video_v2_ens"], V1["video"], t) for t in C.ALL_TARGETS}
res["v2_vs_v1_mm"] = {t: paired(FINAL["multimodal_v2_ens"], V1["multimodal"], t) for t in C.ALL_TARGETS}
res["mm_vs_video_v2"] = {t: paired(FINAL["multimodal_v2_ens"], FINAL["video_v2_ens"], t) for t in C.ALL_TARGETS}
saved = pd.read_csv(V2 / "tiers_metrics.csv").set_index(["tier", "target"])
for t in TG:                                                                 # same numbers as the notebook's Step 9
    assert abs(res["v2_video"][t]["mae"] - saved.loc[("video_v2_ens", t), "mae"]) < 1e-9
    assert abs(res["v2_vs_v1_video"][t]["lo"] - saved.loc[("video_v2_ens", t), "lo"]) < 1e-9
NUM["results"] = res

# r, spread and sex details (shuffle 0, and the mean over shuffles)
P0 = FINAL["video_v2_ens"][0]
rr = {}
for t in TG:
    j = J[t]
    rs = [pearsonr(FINAL["video_v2_ens"][r][:, j], Y[:, j])[0] for r in range(3)]
    sp = [np.std(FINAL["video_v2_ens"][r][:, j]) / np.std(Y[:, j]) for r in range(3)]
    rr[t] = {"r": float(np.mean(rs)), "r_shuffles": [float(x) for x in rs], "spread_ratio": float(np.mean(sp)),
             "answer_min": float(P0[:, j].min()), "answer_max": float(P0[:, j].max())}
NUM["pred_vs_true"] = rr
from sklearn.metrics import roc_auc_score                                   # noqa: E402
ok = np.isfinite(Y[:, J["sex"]])
ys = Y[ok, J["sex"]].astype(int)
sx = []
for r in range(3):
    p = FINAL["video_v2_ens"][r][ok, J["sex"]]
    yhat = (p >= 0.5).astype(int)
    tp, fn = int(((yhat == 1) & (ys == 1)).sum()), int(((yhat == 0) & (ys == 1)).sum())
    tn, fp = int(((yhat == 0) & (ys == 0)).sum()), int(((yhat == 1) & (ys == 0)).sum())
    sx.append({"acc": (tp + tn) / len(ys), "bal": 0.5 * (tp / (tp + fn) + tn / (tn + fp)), "auc": roc_auc_score(ys, p),
               "tp": tp, "fn": fn, "tn": tn, "fp": fp})
NUM["sex"] = {"shuffles": sx, "acc": float(np.mean([s["acc"] for s in sx])), "bal": float(np.mean([s["bal"] for s in sx])),
              "auc": float(np.mean([s["auc"] for s in sx])), "n_rec": int(ok.sum()), "n_people": int(len(np.unique(pids[ok]))),
              "women_rec": int((ys == 0).sum()), "men_rec": int((ys == 1).sum()),
              "v1_auc": float(np.mean([roc_auc_score(ys, V1["video"][r][ok, J["sex"]]) for r in range(3)]))}

# age by group (per person, 3 shuffles)
ea, eg, e1 = person_error(FINAL["video_v2_ens"], "age"), person_error(GUESS, "age"), person_error(V1["video"], "age")
young = (age_p.reindex(ea.index) <= 24).to_numpy()
NUM["age_groups"] = {g: {"people": int(m.sum()), "v2": float(ea[m].mean()), "guess": float(eg[m].mean()), "v1": float(e1[m].mean())}
                     for g, m in (("18-24", young), ("over 24", ~young))}
eb = person_error(FINAL["video_v2_ens"], "bmi")
bmi_p = people["bmi"].reindex(eb.index).to_numpy()
NUM["bmi_groups"] = {g: {"people": int(m.sum()), "v2": float(eb[m].mean())}
                     for g, m in (("under 25", bmi_p < 25), ("25-30", (bmi_p >= 25) & (bmi_p < 30)), ("30 or more", bmi_p >= 30))}

# ------------------------------------------------------------------ 5. one answer per recording (shuffle 0) and the bad cases
mae = {t: res["v2_video"][t]["mae"] for t in TG}
rec = df[["key", "patient_id", "step"]].copy()
rec["fold0"] = C.session_folds(0)
for t in C.ALL_TARGETS:
    rec[f"y_{t}"] = Y[:, J[t]]
    rec[f"p_{t}"] = P0[:, J[t]]
    rec[f"pm_{t}"] = FINAL["multimodal_v2_ens"][0][:, J[t]]
    rec[f"v1_{t}"] = V1["video"][0][:, J[t]]
    rec[f"pavg_{t}"] = np.mean([FINAL["video_v2_ens"][r][:, J[t]] for r in range(3)], 0)
bad = pd.DataFrame(index=rec.index)
for t in TG:
    bad[t] = np.abs(rec[f"p_{t}"] - rec[f"y_{t}"]) > 3 * mae[t]
bad["sex"] = np.isfinite(rec["y_sex"]) & ((rec["p_sex"] >= 0.5) != (rec["y_sex"] == 1))
rec["bad_any"] = bad.any(axis=1)
for t in C.ALL_TARGETS:
    rec[f"bad_{t}"] = bad[t]
# possible video / signal flaws
traces_bright, trace_ok = [], []
for k in df["key"]:
    T = np.load(C.CACHE2 / "traces" / f"{k}.npz")
    rgb = T["rgb"]
    face = rgb[:, 9:12] if rgb.shape[1] >= 12 else rgb[:, :3]
    traces_bright.append(float(np.nanmean(face)))
rec["brightness"] = traces_bright
rec["finger_noisy"] = ~(q[:, 0] > 0)
rec["unreadable_share"] = (log["unreadable_stamps"] / log["kept"]).to_numpy()
rec["grid_valid"] = log["grid_valid"].to_numpy()
rec["skin_fallback"] = (~log.skin_rule.str.split(",").apply(lambda s: all(x == "skin" for x in s))).to_numpy()
cls = hr.set_index("key").loc[df["key"]]
rec["classic_hr"] = cls["v2 classic (POS/CHROM, 4 regions, best quality)"].to_numpy()
rec["finger_hr"] = cls["finger"].to_numpy()
rec["camera_hr_rf"] = cls["RhythmFormer, PURE weights fine-tuned on MCD with the finger (cross-fitted)"].to_numpy()
rec["camera_hr_miss5"] = np.abs(rec["camera_hr_rf"] - rec["finger_hr"]) > 5
rec["dark"] = rec["brightness"] <= np.nanpercentile(rec["brightness"], 10)
counts = {t: int(bad[t].sum()) for t in C.ALL_TARGETS}
flag_rates = {}
base_rate = float(rec["bad_any"].mean())
for f in ("finger_noisy", "skin_fallback", "dark", "camera_hr_miss5"):
    m = rec[f].astype(bool)
    flag_rates[f] = {"videos": int(m.sum()), "bad_rate": float(rec.loc[m, "bad_any"].mean()) if m.any() else None,
                     "bad_rate_others": float(rec.loc[~m, "bad_any"].mean())}
after_m = (rec.step == "after")
flag_rates["after_exercise"] = {"videos": int(after_m.sum()), "bad_rate": float(rec.loc[after_m, "bad_any"].mean()),
                                "bad_rate_others": float(rec.loc[~after_m, "bad_any"].mean())}
# direction of the misses for age and BMI (answered too low?)
dirs = {}
for t in TG:
    m = bad[t]
    dirs[t] = {"too_low": int(((rec[f"p_{t}"] < rec[f"y_{t}"]) & m).sum()), "too_high": int(((rec[f"p_{t}"] > rec[f"y_{t}"]) & m).sum()),
               "threshold": 3 * mae[t]}
worst_age = rec.loc[bad["age"]].assign(err=lambda d: d.p_age - d.y_age).sort_values("err")
NUM["bad_cases"] = {"rule": "abs(answer - recorded) > 3 x the model's cross-validated MAE; sex: wrong side of 50%",
                    "counts": counts, "recordings_with_any": int(rec.bad_any.sum()), "people_with_any": int(rec.loc[rec.bad_any, "patient_id"].nunique()),
                    "base_rate": base_rate, "flags": flag_rates, "directions": dirs,
                    "age_too_high": worst_age[worst_age.err > 0][["patient_id", "step", "y_age", "p_age"]].to_dict("records"),
                    "age_most_too_low": worst_age.head(6)[["patient_id", "step", "y_age", "p_age"]].to_dict("records"),
                    "sex_wrong": rec.loc[bad["sex"], ["patient_id", "step", "y_sex", "p_sex"]].to_dict("records")}
for pid in ("3778", "3153", "7539", "6329"):
    NUM["bad_cases"][f"person_{pid}"] = rec.loc[rec.patient_id == pid, ["step", "y_age", "pavg_age", "y_cholesterol", "pavg_cholesterol",
                                                                         "y_sex", "pavg_sex"]].to_dict("records")

# ------------------------------------------------------------------ 6. sensitivity and specificity (clinical cut-offs, as in Week 3-4)
CUTS = [  # name, target, condition on the recorded value, yes-rule for the answer "as it is", why
    ("Oxygen saturation below 95%", "saturation", lambda y: y < 95, lambda p: p < 94.5, "below the normal range of 95-100%", "low"),
    ("Temperature 36.8 °C or more", "temperature", lambda y: y >= 36.8 - 1e-9, lambda p: p >= 36.75, "the warmest 5%; nobody has a fever", "high"),
    ("HbA1c 5.7% or more", "glycated_hemoglobin", lambda y: y >= 5.7 - 1e-9, lambda p: p >= 5.695, "prediabetes or diabetes range", "high"),
    ("HbA1c 6.5% or more", "glycated_hemoglobin", lambda y: y >= 6.5 - 1e-9, lambda p: p >= 6.495, "diabetes range", "high"),
    ("Cholesterol 5.2 mmol/L or more", "cholesterol", lambda y: y >= 5.2 - 1e-9, lambda p: p >= 5.15, "above the desirable level", "high"),
    ("Respiratory rate above 20/min", "respiratory", lambda y: y > 20, lambda p: p >= 20.5, "faster than the adult normal of 12-20", "high"),
    ("Stiffness 12.4 or more", "rigidity", lambda y: y >= 12.4 - 1e-9, lambda p: p >= 12.395, "about the top 10%; no clinical cut-off is known", "high"),
    ("Stress 6.2 or more", "stress", lambda y: y >= 6.2 - 1e-9, lambda p: p >= 6.195, "the PSM-25 'high' band, if the column is the mean item score", "high"),
    ("Age 40 or older", "age", lambda y: y >= 40, lambda p: p >= 39.5, "older adults", "high"),
    ("BMI 30 or more", "bmi", lambda y: y >= 30, lambda p: p >= 29.95, "obesity (WHO)", "high"),
    ("Recorded male", "sex", lambda y: y == 1, lambda p: p >= 0.5, "positive = male: P(male) of 50% or more", "high"),
]


def boot_rates(yes, cond, people_ids, n=B):
    """Person bootstrap of sensitivity and specificity (recordings of a person stay together)."""
    up = np.unique(people_ids)
    pos = {p: i for i, p in enumerate(up)}
    gi = np.array([pos[p] for p in people_ids])
    tp = np.bincount(gi, weights=(yes & cond), minlength=len(up))
    fn = np.bincount(gi, weights=(~yes & cond), minlength=len(up))
    tn = np.bincount(gi, weights=(~yes & ~cond), minlength=len(up))
    fp = np.bincount(gi, weights=(yes & ~cond), minlength=len(up))
    idx = np.random.default_rng(RNG_SEED).integers(0, len(up), size=(n, len(up)))
    TP, FN, TN, FP = tp[idx].sum(1), fn[idx].sum(1), tn[idx].sum(1), fp[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        se, sp = TP / (TP + FN), TN / (TN + FP)
    return np.nanpercentile(se, [2.5, 97.5]), np.nanpercentile(sp, [2.5, 97.5])


def auc_ci(score, cond, people_ids, n=B):
    up = np.unique(people_ids)
    groups = [np.flatnonzero(people_ids == p) for p in up]
    rng = np.random.default_rng(RNG_SEED)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(up), len(up))
        ii = np.concatenate([groups[i] for i in pick])
        c = cond[ii]
        if c.all() or not c.any():
            continue
        vals.append(roc_auc_score(c, score[ii]))
    return float(roc_auc_score(cond, score)), np.percentile(vals, [2.5, 97.5])


def best_cut(score, cond):
    """The cut-off on the answer with the highest sensitivity + specificity (answers at or above it say yes)."""
    u = np.unique(score)
    cand = np.concatenate([[u[0] - 1e-9], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-9]])
    n_pos, n_neg = cond.sum(), (~cond).sum()
    order = np.argsort(score)
    s_sorted, c_sorted = score[order], cond[order]
    cum_pos = np.concatenate([[0], np.cumsum(c_sorted)])                    # positives below each cut
    cum_neg = np.concatenate([[0], np.cumsum(~c_sorted)])                   # negatives below each cut
    k = np.searchsorted(s_sorted, cand, side="left")
    youden = (n_pos - cum_pos[k]) / max(n_pos, 1) + cum_neg[k] / max(n_neg, 1)
    return float(cand[int(np.argmax(youden))])


def rates(yes, cond):
    tp, fn = int((yes & cond).sum()), int((~yes & cond).sum())
    tn, fp = int((~yes & ~cond).sum()), int((yes & ~cond).sum())
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "sens": tp / (tp + fn) if tp + fn else np.nan,
            "spec": tn / (tn + fp) if tn + fp else np.nan}


ss_rows = []
for name, t, cond_fn, yes_fn, why, side in CUTS:
    j = J[t]
    for r in range(3):
        P = FINAL["video_v2_ens"][r]
        ok = np.isfinite(Y[:, j])
        y, p, pid = Y[ok, j], P[ok, j], pids[ok]
        fold = C.session_folds(r)[ok]
        cond = cond_fn(y).astype(bool)
        sign = -1.0 if side == "low" else 1.0                               # "low" conditions: lower answer = yes
        score = sign * p
        as_is = rates(yes_fn(p).astype(bool), cond)
        # cross-validated best cut-off: chosen on the other folds' (honest) answers, applied to the held-out fold
        yes_cv = np.zeros(len(y), bool)
        cuts = []
        for k in range(C.N_FOLDS):
            te = fold == k
            c = best_cut(score[~te], cond[~te])
            cuts.append(sign * c)
            yes_cv[te] = score[te] >= c
        cv = rates(yes_cv, cond)
        c_all = best_cut(score, cond)                                        # chosen on everyone (for the pictures)
        row = {"condition": name, "target": t, "why": why, "shuffle": r, "recordings": int(len(y)),
               "people": int(len(np.unique(pid))), "with_it": int(cond.sum()), "people_with_it": int(len(np.unique(pid[cond]))),
               "as_is_sens": as_is["sens"], "as_is_spec": as_is["spec"], "as_is": as_is,
               "cv_sens": cv["sens"], "cv_spec": cv["spec"], "cv": cv,
               "cv_cut_median": float(np.median(cuts)), "cut_all": float(sign * c_all), "side": side}
        if r == 0:
            a, (alo, ahi) = auc_ci(score, cond, pid)
            (selo, sehi), (splo, sphi) = boot_rates(yes_fn(p).astype(bool), cond, pid)
            (cselo, csehi), (csplo, csphi) = boot_rates(yes_cv, cond, pid)
            row.update({"auc": a, "auc_lo": float(alo), "auc_hi": float(ahi),
                        "as_is_sens_ci": [float(selo), float(sehi)], "as_is_spec_ci": [float(splo), float(sphi)],
                        "cv_sens_ci": [float(cselo), float(csehi)], "cv_spec_ci": [float(csplo), float(csphi)],
                        "answer_min": float(p.min()), "answer_max": float(p.max())})
        else:
            row["auc"] = float(roc_auc_score(cond, score))
        ss_rows.append(row)
SS = pd.DataFrame(ss_rows)
NUM["sens_spec_cv"] = SS[SS.shuffle == 0].drop(columns=["as_is", "cv"]).to_dict("records")
NUM["sens_spec_auc_range"] = SS.groupby("condition")["auc"].agg(["min", "max"]).reset_index().to_dict("records")
NUM["sens_spec_cv_mean"] = SS.groupby("condition")[["as_is_sens", "as_is_spec", "cv_sens", "cv_spec", "auc"]].mean().reset_index().to_dict("records")
SS[SS.shuffle == 0].drop(columns=["as_is", "cv"]).to_csv(OUT / "sens_spec.csv", index=False)

# shortcuts behind some physiological AUCs (as in Week 3-4)
age_score = rec["p_age"].to_numpy()
sc = {}
for name, t, cond_fn, yes_fn, why, side in CUTS:
    if t in ("age", "sex", "bmi"):
        continue
    j = J[t]
    ok = np.isfinite(Y[:, j])
    cond = cond_fn(Y[ok, j]).astype(bool)
    sc[name] = {"auc_answer": float(roc_auc_score(cond, (-1 if side == "low" else 1) * P0[ok, j])),
                "auc_true_age": float(roc_auc_score(cond, df["age"].to_numpy()[ok])),
                "auc_age_answer": float(roc_auc_score(cond, age_score[ok])),
                "auc_after_exercise": float(roc_auc_score(cond, df["after"].to_numpy()[ok])),
                "r_with_age_answer": float(np.corrcoef(P0[ok, j], age_score[ok])[0, 1]),
                "share_after": float(df["after"].to_numpy()[ok][cond].mean()),
                "people_over_40": int(len(np.unique(pids[ok][cond & (df['age'].to_numpy()[ok] >= 40)]))),
                "camera_hr_auc": float(roc_auc_score(cond[np.isfinite(rec['camera_hr_rf'].to_numpy()[ok])],
                                                     rec['camera_hr_rf'].to_numpy()[ok][np.isfinite(rec['camera_hr_rf'].to_numpy()[ok])]))}
NUM["sens_spec_shortcuts"] = sc
resp_b = rec.loc[rec.step == "before", "p_respiratory"].mean(), rec.loc[rec.step == "after", "p_respiratory"].mean()
NUM["resp_answers_before_after"] = {"answer_before": float(resp_b[0]), "answer_after": float(resp_b[1]),
                                     "label_before": float(rec.loc[rec.step == "before", "y_respiratory"].mean()),
                                     "label_after": float(rec.loc[rec.step == "after", "y_respiratory"].mean())}

# Face_AI's 90 test people: v2 (refitted on the same 510 people) next to Face_AI's own numbers
fx = pd.read_csv(C.OOF / "fixed_split_video_specialists_v2.csv", dtype={"patient_id": str}).set_index("key")
yt = df.set_index("key").loc[fx.index]
fa = pd.read_csv(FACE_AI / "report" / "eval_sensitivity_specificity.csv", encoding="utf-8-sig")
tst = []
nontest = ~df["patient_id"].isin(fx["patient_id"].unique()).to_numpy()
for name, t, cond_fn, yes_fn, why, side in CUTS:
    j = J[t]
    col = "sex_label" if t == "sex" else t
    y = yt[col].to_numpy(float)
    p = fx[f"pred_{t}"].to_numpy(float)
    ok = np.isfinite(y)
    y, p, pid = y[ok], p[ok], fx["patient_id"].to_numpy()[ok]
    cond = cond_fn(y).astype(bool)
    sign = -1.0 if side == "low" else 1.0
    # cut-off chosen on the 510 non-test people's cross-validated answers (shuffle 0), like Face_AI's
    okn = nontest & np.isfinite(Y[:, j])
    c = best_cut(sign * P0[okn, j], cond_fn(Y[okn, j]).astype(bool))
    a1, a2 = rates(yes_fn(p).astype(bool), cond), rates(sign * p >= c, cond)
    auc, (alo, ahi) = auc_ci(sign * p, cond, pid) if 0 < cond.sum() < len(cond) else (np.nan, (np.nan, np.nan))
    f = fa[(fa.condition.str.replace("–", "-") == name.replace("–", "-")) & (fa.group == "test")]
    frow = f.iloc[0].to_dict() if len(f) else {}
    tst.append({"condition": name, "recordings": int(len(y)), "with_it": int(cond.sum()), "people_with_it": int(len(np.unique(pid[cond]))),
                "v2_as_is": a1, "v2_cut": float(sign * c), "v2_best": a2, "v2_auc": auc, "v2_auc_lo": float(alo), "v2_auc_hi": float(ahi),
                "faceai_as_is_sens": frow.get("sensitivity %"), "faceai_as_is_spec": frow.get("specificity %"),
                "faceai_best_sens": frow.get("best cut-off: sensitivity %"), "faceai_best_spec": frow.get("best cut-off: specificity %"),
                "faceai_auc": frow.get("AUC"), "faceai_auc_ci": frow.get("AUC 95% interval")})
NUM["sens_spec_test90"] = tst

# ------------------------------------------------------------------ 7. head-to-head with Face_AI (saved)
h2h = pd.read_csv(V2 / "h2h.csv").set_index("target")
NUM["h2h"] = h2h.reset_index().to_dict("records")
NUM["h2h_extra"] = json.loads((V2 / "h2h_extra.json").read_text())

# ------------------------------------------------------------------ 8. face routes (FaceAI_v2.ipynb Step 6.1) and the augmentation
import v2_nested as N                                                       # noqa: E402
fr = {}
for f in sorted((V2 / "routes").glob("*_r0.npz")):
    route = f.stem[:-3]
    if not (route in ("dino", "clip", "appearance") or route.startswith(("app_", "mlp"))):
        continue
    if not all((V2 / "routes" / f"{route}_r{r}.npz").exists() for r in range(3)):
        continue
    Pm = {r: N.load_route(route, r)[0] for r in range(3)}
    s = C.quick_scores(Pm, ["age", "bmi", "sex"])
    fr[route] = {"age": s["age"][0], "age_skill": s["age"][1], "bmi": s["bmi"][0], "bmi_skill": s["bmi"][1], "sex": s["sex"]}
NUM["face_routes"] = fr

with np.load(C.FEATURES / "v2_clip_l14_tight.npz") as z:
    pos = {str(k): i for i, k in enumerate(z["keys"])}
    crops = z["crops"][[pos[k] for k in df["key"]]].astype(np.float32)      # (1200, 24, D)
full = crops.mean(1)
mu = full.mean(0)
rng = np.random.default_rng(RNG_SEED)


def cos(a, b):
    a, b = a - mu, b - mu
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b)))


sub8 = [cos(full[i], crops[i, rng.choice(24, 8, replace=False)].mean(0)) for i in range(len(df))]
one = [cos(full[i], crops[i, rng.integers(24)]) for i in range(len(df))]
key_idx = {k: i for i, k in enumerate(df["key"])}
other = []
for i, (pid, st) in enumerate(zip(df["patient_id"], df["step"])):
    o = f"{pid}_FullHDwebcam_{'after' if st == 'before' else 'before'}"
    if o in key_idx:
        other.append(cos(full[i], full[key_idx[o]]))
perm = rng.permutation(len(df))
another = [cos(full[i], full[j]) for i, j in zip(range(len(df)), perm) if pids[i] != pids[j]]
first8 = [cos(full[i], crops[i, :8].mean(0)) for i in range(len(df))]
NUM["augmentation"] = {"random_8_of_24": float(np.median(sub8)), "one_crop": float(np.median(one)),
                       "first_8_crops": float(np.median(first8)),
                       "other_session": float(np.median(other)), "another_person": float(np.median(another))}

# ------------------------------------------------------------------ 9. which inputs and models each target uses (video tier)
sel = pd.read_csv(V2 / "tiers_selection.csv")
s = sel[sel.tier == "video_v2_ens"]
NUM["ensemble_choice"] = s.groupby("target")["m"].value_counts().unstack(fill_value=0).to_dict("index")
NUM["selection_examples"] = s.groupby("target")["routes"].agg(lambda x: x.value_counts().head(3).to_dict()).to_dict()

# ------------------------------------------------------------------ 10. saved checks
NUM["stiffness_vs_hr"] = pd.read_csv(V2 / "stiffness_vs_heart_rate.csv").to_dict("records")
NUM["state_auc"] = pd.read_csv(V2 / "state_detection_auc.csv").to_dict("records")
NUM["oracle"] = pd.read_csv(V2 / "oracle_finger_hr.csv").to_dict("records")
NUM["strict"] = pd.read_csv(V2 / "strict_nesting_check.csv").to_dict("records")

rec.to_csv(OUT / "recordings.csv", index=False)
(OUT / "numbers.json").write_text(json.dumps(jsonable(NUM), indent=1, ensure_ascii=False), encoding="utf-8")
print("saved", OUT / "numbers.json")
