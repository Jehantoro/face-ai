"""Markdown tables for the v2 report, read straight from the saved results (no numbers typed by hand).

    .venv/Scripts/python.exe scripts/make_v2_report_tables.py > results/v2/report_tables.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_core as C                                                        # noqa: E402

V2 = C.RESULTS / "v2"
DEEP_LABELS = {"tracenet": "v2 trace network (taught by the finger, cross-fitted)",
               "physnet_diff": "v2 PhysNet on face clips (taught by the finger, cross-fitted)",
               "rhythmformer_pure_zeroshot": "RhythmFormer, published PURE weights, no MCD training (zero-shot)",
               "rhythmformer_pure": "RhythmFormer, PURE weights fine-tuned on MCD with the finger (cross-fitted)"}
UNIT ={"saturation": "%", "temperature": "°C", "glycated_hemoglobin": "%", "cholesterol": "mmol/L", "respiratory": "/min",
        "rigidity": "index", "stress": "score", "age": "years", "bmi": "kg/m²"}


def fmt(v, d=3):
    return "" if pd.isna(v) else f"{v:.{d}f}"


def signed(v, d=3):
    return "" if pd.isna(v) else (f"+{v:.{d}f}" if v > 0 else f"−{abs(v):.{d}f}" if v < 0 else f"{0:.{d}f}")


def tier_table():
    tab = pd.read_csv(V2 / "tiers_metrics.csv")
    v1 = pd.read_csv(C.MET / "regression_metrics.csv").set_index(["model", "target"])
    out = ["| Target | Median guess MAE | v1 video | **v2 video** | v2 − v1 [95% CI] | v1 + PPG/ECG | **v2 + PPG/ECG** | v2 − v1 [95% CI] |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t in C.REG_TARGETS:
        a = tab[(tab.tier == "video_v2_ens") & (tab.target == t)].iloc[0]
        b = tab[(tab.tier == "multimodal_v2_ens") & (tab.target == t)].iloc[0]
        va, vb = v1.loc[("video_specialists", t)], v1.loc[("multimodal_specialists", t)]
        bold = lambda r: "**" if r["hi"] < 0 else ""
        out.append(f"| {C.NAME[t]} | {fmt(va['mae_constant'])} | {fmt(va['mae'])} ({signed(va['skill'], 2)}) | "
                   f"{bold(a)}{fmt(a['mae'])} ({signed(a['skill'], 2)}){bold(a)} | {signed(a['vs_old'])} [{signed(a['lo'])}, {signed(a['hi'])}] | "
                   f"{fmt(vb['mae'])} ({signed(vb['skill'], 2)}) | {bold(b)}{fmt(b['mae'])} ({signed(b['skill'], 2)}){bold(b)} | "
                   f"{signed(b['vs_old'])} [{signed(b['lo'])}, {signed(b['hi'])}] |")
    sx = tab[tab.target == "sex"].set_index("tier")["acc"]
    cls = pd.read_csv(C.MET / "classification_metrics.csv").set_index("model")["accuracy"]
    out.append(f"| Sex (accuracy) | {fmt(cls['constant'], 4)} | {fmt(cls['video_specialists'], 4)} | {fmt(sx['video_v2_ens'], 4)} | | "
               f"{fmt(cls['multimodal_specialists'], 4)} | {fmt(sx['multimodal_v2_ens'], 4)} | |")
    return "\n".join(out)


def h2h_table():
    h = pd.read_csv(V2 / "h2h.csv").set_index("target")
    extra = json.loads((V2 / "h2h_extra.json").read_text())
    out = ["| Target | Median guess | Face_AI newest | Face_AI_2 v1 | **Face_AI_2 v2** | v2 − Face_AI [95% CI] | Verdict |",
           "|---|---:|---:|---:|---:|---:|---|"]
    for t in C.REG_TARGETS:
        r = h.loc[t]
        verdict = "**v2 better**" if r["hi"] < 0 else ("Face_AI better" if r["lo"] > 0 else "no clear difference")
        out.append(f"| {C.NAME[t]} | {fmt(r['guess'])} | {fmt(r['face_ai_newest'])} | {fmt(r['faceai2_v1'])} | {fmt(r['faceai2_v2'])} | "
                   f"{signed(r['v2_minus_faceai'])} [{signed(r['lo'])}, {signed(r['hi'])}] | {verdict} |")
    s = extra["sex"]
    out.append(f"| Sex (wrong recordings of {s['of']}) | | {s['face_ai wrong']} | | {s['faceai2_v2 wrong']} | | |")
    ages = pd.DataFrame(extra["ages"])
    lines = ["", "| Age group | Face_AI newest | Face_AI_2 v2 |", "|---|---:|---:|"]
    for band in ages.index:
        lines.append(f"| {band} | {fmt(ages.loc[band, 'face_ai_newest'], 2)} | {fmt(ages.loc[band, 'faceai2_v2'], 2)} |")
    return "\n".join(out + lines)


def pulse_routes_table(targets=("respiratory", "rigidity", "stress")):
    """Each finger-taught network's pulse numbers as a route on their own and with the classic signals (video only,
    3 shuffles, nested CV): does a cleaner pulse wave give better predictions? MAE (skill)."""
    import v2_nested as N
    rows = [("signals2", "classic signals only (classic pulse + breathing + v1 pulse)")]
    for m, label in (("tracenet", "trace network"), ("physnet_diff", "PhysNet"), ("rhythmformer_pure", "RhythmFormer, fine-tuned")):
        rows += [(f"deep_{m}", f"{label} pulse alone"), (f"signals2+deep_{m}", f"classic signals + {label} pulse")]
    rows.append(("signals2+deep_all", "classic signals + all network pulses"))
    out = ["| Route | " + " | ".join(C.NAME[t] for t in targets) + " |", "|---|" + "---:|" * len(targets)]
    for route, label in rows:
        if not all(N.route_path(route, r).exists() for r in range(3)):
            continue
        s = C.quick_scores({r: N.load_route(route, r)[0] for r in range(3)}, list(targets))
        out.append(f"| {label} | " + " | ".join(f"{fmt(s[t][0])} ({signed(s[t][1], 2)})" for t in targets) + " |")
    return "\n".join(out)


STAGES = (("Without RhythmFormer", "before_rhythmformer"), ("+ RhythmFormer", "before_shape"), ("+ shape numbers (final)", None))


def _ci(d):
    return f"{signed(d[0])} [{signed(d[1])}, {signed(d[2])}]"


def age_by_group(P_by_repeat):
    """Age MAE per person within 18-24 / over 24, averaged over the shuffles given."""
    j = C.ALL_TARGETS.index("age")
    band = np.where(C.Y_nat[:, j] <= 24, "18-24", ">24")
    per = [pd.DataFrame({"e": np.abs(P[:, j] - C.Y_nat[:, j]), "pid": C.pids, "band": band})
           .groupby(["band", "pid"])["e"].mean().groupby("band").mean() for P in P_by_repeat.values()]
    return pd.concat(per, axis=1).mean(axis=1)


def late_additions_table():
    """The two late additions (2 November), each tested under the rule every route follows (it becomes a candidate; only
    the inner loop chooses): the fine-tuned RhythmFormer pulse, then pulse-shape numbers. MAE per stage, the paired
    person bootstrap of each step (3 shuffles), and in how many of the 15 outer folds the inner loop picked such a route."""
    dirs = [V2 if sub is None else V2 / sub for _, sub in STAGES]
    if not all((d / "tier_video_v2_ens.npz").exists() for d in dirs):
        return "(the saved stages are missing)"
    sel_rf, sel_sh = pd.read_csv(dirs[1] / "tiers_selection.csv"), pd.read_csv(dirs[2] / "tiers_selection.csv")
    out = ["| Tier | Target | " + " | ".join(n for n, _ in STAGES) + " | RhythmFormer step [95% CI] | Shape step [95% CI] | "
           "Folds picking RhythmFormer / shape routes |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    ages = []
    for tier, label in (("video_v2_ens", "video"), ("multimodal_v2_ens", "+ PPG/ECG")):
        P = []
        for d in dirs:
            with np.load(d / f"tier_{tier}.npz") as z:
                P.append({r: z[f"r{r}"] for r in range(3)})
        S = [C.quick_scores(p) for p in P]
        for t in C.REG_TARGETS:
            n_rf = int(sel_rf[(sel_rf.tier == tier) & (sel_rf.target == t)]["routes"].str.contains("rhythmformer").sum())
            n_sh = int(sel_sh[(sel_sh.tier == tier) & (sel_sh.target == t)]["routes"].str.contains("shape_").sum())
            out.append(f"| {label} | {C.NAME[t]} | " + " | ".join(fmt(s[t][0]) for s in S)
                       + f" | {_ci(C.paired_ci(P[1], P[0], t))} | {_ci(C.paired_ci(P[2], P[1], t))} | {n_rf} / {n_sh} of 15 |")
        out.append(f"| {label} | Sex (accuracy) | " + " | ".join(fmt(s["sex"], 4) for s in S) + " | | | |")
        g = [age_by_group(p) for p in P]
        ages.append(f"| {label} | " + " → ".join(fmt(x["18-24"]) for x in g) + " | " + " → ".join(fmt(x[">24"]) for x in g) + " |")
    lines = out + ["", "| Tier | Age MAE 18–24 (three stages) | Age MAE over 24 (three stages) |", "|---|---:|---:|"] + ages
    H = [pd.read_csv(d / "h2h.csv").set_index("target") for d in dirs]
    lines += ["", "| Head-to-head target | Face_AI newest | " + " | ".join(n for n, _ in STAGES) + " | final: v2 − Face_AI [95% CI] |",
              "|---|---:|---:|---:|---:|---:|"]
    for t in C.REG_TARGETS:
        h = H[2].loc[t]
        lines.append(f"| {C.NAME[t]} | {fmt(h['face_ai_newest'])} | " + " | ".join(fmt(x.loc[t, "faceai2_v2"]) for x in H)
                     + f" | {signed(h['v2_minus_faceai'])} [{signed(h['lo'])}, {signed(h['hi'])}] |")
    return "\n".join(lines)


def shape_tables():
    """Pulse-shape numbers (pulse_shape.py, check_pulse_shape_routes.py): do the camera's follow the finger's beyond heart
    rate (no labels), and do they help as routes (nested CV, 3 shuffles)?"""
    a = pd.read_csv(V2 / "pulse_shape_agreement.csv")
    nets = (("tracenet", "trace network"), ("physnet_diff", "PhysNet"), ("rhythmformer_pure", "RhythmFormer"))
    rows = (("rise_time_s", "rise time"), ("width50_frac", "width at half height"), ("systolic_area_frac", "share of area before the peak"),
            ("apg_b_a", "second-derivative b/a"), ("template_corr", "how alike the beats are"))
    out = ["| Shape number (partial ρ given heart rate) | Finger at 30 fps vs finger sensor at 100 Hz | "
           + " | ".join(f"Camera ({n}) vs finger" for _, n in nets) + " |", "|---|---:|" + "---:|" * len(nets)]
    get = lambda comp, num: a[(a["compare"] == comp) & (a["number"] == num)]["partial rho | HR"]
    for num, label in rows:
        f = get("finger 30 fps vs finger PPG tier (100 Hz)", num)
        out.append(f"| {label} | {fmt(f.iloc[0], 2) if len(f) else ''} | "
                   + " | ".join(fmt(get(f"camera ({m}) vs finger, clean finger", num).iloc[0], 2) for m, _ in nets) + " |")
    t = a[a["number"].str.startswith("tmpl_")].groupby("compare")["partial rho | HR"].median()
    out.append("| the beat itself (median over its 25 points) | | " + " | ".join(fmt(t[f"camera ({m}) vs finger, clean finger"], 2) for m, _ in nets) + " |")
    notch = [int(round(C.load_features(C.FEATURES / f"v2_pulse_shape_{m}.npz", C.df["key"])[
        :, C.feature_names(C.FEATURES / f"v2_pulse_shape_{m}.npz").index("shape notch_found")].sum())) for m, _ in nets]
    fn = a[(a["compare"] == "finger 30 fps vs finger PPG tier (100 Hz)") & (a["number"] == "notch_height")]["n"]
    out.append(f"| recordings with a visible notch (reflected wave) | {int(fn.iloc[0]) if len(fn) else ''} of 1,197 (both) | "
               + " | ".join(f"{k} of 1,200" for k in notch) + " |")
    r = pd.read_csv(V2 / "pulse_shape_routes.csv")
    out += ["", "| Network | Target | Rhythm numbers (MAE) | Shape numbers (MAE) | Rhythm + shape (MAE) | Rhythm + shape minus rhythm [95% CI] |",
            "|---|---|---:|---:|---:|---:|"]
    for _, x in r.iterrows():
        out.append(f"| {x['network']} | {x['target']} | {fmt(x['rhythm numbers (MAE)'])} | {fmt(x['shape numbers (MAE)'])} | "
                   f"{fmt(x['rhythm + shape (MAE)'])} | {signed(x['rhythm + shape minus rhythm'])} [{signed(x['lo'])}, {signed(x['hi'])}] |")
    return "\n".join(out)


def classic_first_minute_hr(key):
    """v2 classic pulse: the best region/method (highest SNR) on the first minute, heart rate with the SAME
    estimator as the finger reference (rppg_deep.hr_from_wave: Welch, 17-s segments)."""
    import rppg_deep as R
    from v2_features import REGIONS, bandpass, chrom, fill, pos
    T = np.load(C.CACHE2 / "traces" / f"{key}.npz")
    rgb, valid = T["rgb"][:1800].astype(float), T["valid"][:1800]
    best = (-np.inf, np.nan)
    for r in range(len(REGIONS)):
        cols = [fill(rgb[:, 3 * r + c], valid) for c in range(3)]
        if any(c is None for c in cols) or min(np.min(c) for c in cols) <= 0:
            continue
        x = np.column_stack(cols)
        for fn in (pos, chrom):
            hr, snr = R.hr_from_wave(bandpass(fn(x), 0.7, 3.5))
            if np.isfinite(snr) and snr > best[0]:
                best = (snr, hr)
    return best[1]


def pulse_table():
    """Every camera pulse's first-minute heart rate with the same estimator as the finger reference."""
    import rppg_deep as R
    from joblib import Parallel, delayed
    P = np.load(C.PROJECT / "data" / "cache" / "ppg_sync.npz")["ppg"]
    finger = np.array([R.hr_from_wave(R.bandpass(p))[0] for p in P])

    def col(f, n):
        with np.load(C.FEATURES / f) as z:
            return z["features"][:, list(z["names"]).index(n)].astype(float)
    cams = {"v1 (POS/CHROM/green, 3 patches, first minute)": col("rppg_v2.npz", "pulse bpm (best quality)"),
            "v2 classic (POS/CHROM, 4 regions, best quality)":
                np.array(Parallel(n_jobs=8)(delayed(classic_first_minute_hr)(k) for k in C.df["key"]))}
    for m, label in DEEP_LABELS.items():
        f = C.CACHE2 / "rppg" / m / "wave.npy"
        if f.exists():
            W = np.load(f)
            cams[label] = np.array([R.hr_from_wave(R.bandpass(w[:1800]))[0] for w in W])
    y = C.df["rigidity"].to_numpy()
    from scipy.stats import spearmanr
    out_csv = C.df[["key", "patient_id", "step"]].copy()                  # per-recording heart rates, for the notebook
    out_csv["finger"] = finger
    for label, hr in cams.items():
        out_csv[label] = hr
    out_csv.to_csv(V2 / "camera_hr_first_minute.csv", index=False)
    out = ["| Camera pulse | MAE (bpm) | Median error | Within 5 bpm | r with finger | ρ with stiffness |", "|---|---:|---:|---:|---:|---:|"]
    for label, hr in cams.items():
        ok = np.isfinite(hr) & np.isfinite(finger)
        e = np.abs(hr[ok] - finger[ok])
        out.append(f"| {label} | {e.mean():.2f} | {np.median(e):.2f} | {np.mean(e <= 5):.0%} | {np.corrcoef(hr[ok], finger[ok])[0, 1]:.2f} | "
                   f"{spearmanr(hr, y, nan_policy='omit')[0]:.2f} |")
    out.append(f"| *finger sensor itself (reference)* | 0 | 0 | 100% | 1 | {spearmanr(finger, y, nan_policy='omit')[0]:.2f} |")
    return "\n".join(out)


def _best_r(w, y, s=3):
    """Wave agreement as in rppg_deep.evaluate: the best Pearson r over -3..+3 frame shifts (both band-passed)."""
    import rppg_deep as R
    a, b = R.bandpass(np.asarray(w, float)), R.bandpass(np.asarray(y, float))
    return max(np.corrcoef(a[s:len(a) - s], b[s + k:len(b) - s + k])[0, 1] for k in range(-s, s + 1))


def pulse_paired_table(model="rhythmformer_pure", others=(("physnet_diff", "PhysNet"), ("tracenet", "trace network"))):
    """`model` minus each other finger-taught network, recording by recording on the first minute: heart-rate error
    against the finger and the wave r with the finger, averaged per person; 95% intervals from 2,000 bootstrap
    resamples of people. Clean finger = recordings whose finger pulse has a positive SNR (the networks' training set)."""
    import rppg_deep as R
    ppg, q = R.load_labels()                                             # finger pulse aligned with the face (lag 2)
    sign = np.load(C.CACHE2 / "ppg_sign.npy") if (C.CACHE2 / "ppg_sign.npy").exists() else np.ones(len(C.df))
    ref = ppg[:, :1800] * sign[:, None]
    finger = np.array([R.hr_from_wave(R.bandpass(p))[0] for p in ref])
    clean = (np.isfinite(q[:, 0]) & (q[:, 0] > 0)) if q is not None else np.ones(len(C.df), bool)
    err, shape = {}, {}
    for m in [model] + [o for o, _ in others]:
        W = np.load(C.CACHE2 / "rppg" / m / "wave.npy")[:, :1800]
        err[m] = np.abs(np.array([R.hr_from_wave(R.bandpass(w))[0] for w in W]) - finger)
        shape[m] = np.array([_best_r(W[v], ref[v]) for v in range(len(W))])
    people = np.unique(C.pids)
    boot = np.random.default_rng(C.SEED).integers(0, len(people), size=(2000, len(people)))

    def diff(a, b, mask):
        d = pd.Series(np.where(mask, a - b, np.nan)).groupby(C.pids).mean().reindex(people).to_numpy()
        bs = np.nanmean(d[boot], 1)
        return f"{signed(np.nanmean(d), 3)} [{signed(np.percentile(bs, 2.5), 3)}, {signed(np.percentile(bs, 97.5), 3)}]"
    out = ["| Recordings | Fine-tuned RhythmFormer minus | Heart-rate error (bpm) [95% CI] | Wave r with the finger [95% CI] |",
           "|---|---|---:|---:|"]
    for label, mask in ((f"all {len(C.df):,}", np.ones(len(C.df), bool)), (f"clean finger ({int(clean.sum()):,})", clean)):
        for o, name in others:
            out.append(f"| {label} | {name} | {diff(err[model], err[o], mask)} | {diff(shape[model], shape[o], mask)} |")
    big = ["", "| Network | Recordings > 5 bpm off (all) | > 5 bpm off (clean finger) | Mean wave r with the finger (all) |", "|---|---:|---:|---:|"]
    for m in [model] + [o for o, _ in others]:
        big.append(f"| {DEEP_LABELS[m]} | {int((err[m] > 5).sum())} | {int((err[m][clean] > 5).sum())} | {np.nanmean(shape[m]):.3f} |")
    return "\n".join(out + big)


if __name__ == "__main__":
    print("## Camera pulse vs finger (first minute, all 1,200 recordings)\n")
    print(pulse_table())
    if (C.CACHE2 / "rppg" / "rhythmformer_pure" / "wave.npy").exists():
        print("\n## Fine-tuned RhythmFormer vs the other finger-taught networks (paired, per person)\n")
        print(pulse_paired_table())
        print("\n## Pulse routes on their own (video only, nested CV, 3 shuffles; MAE (skill))\n")
        print(pulse_routes_table())
    print("\n## v2 vs v1 (nested CV, 3 shuffles; MAE (skill); bold = interval of v2 − v1 entirely below 0)\n")
    print(tier_table())
    if (V2 / "h2h.csv").exists():
        print("\n## Head-to-head on Face_AI's 90 test people\n")
        print(h2h_table())
    if (V2 / "pulse_shape_routes.csv").exists():
        print("\n## Pulse-shape numbers from the camera pulse (agreement with the finger; as routes)\n")
        print(shape_tables())
    if (V2 / "before_rhythmformer" / "tier_video_v2_ens.npz").exists():
        print("\n## The two late additions: RhythmFormer, then pulse-shape numbers (3 shuffles, paired person bootstrap)\n")
        print(late_additions_table())
