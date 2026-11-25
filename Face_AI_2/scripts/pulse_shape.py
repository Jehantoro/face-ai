"""Pulse-SHAPE numbers from a camera pulse, the same kind the finger PPG features use (v1 notebook, Step 5.2), at
30 frames per second: the median beat (foot to foot, scaled 0-1, 100 points), its rise time, widths at 25/50/75 %,
the notch and the reflected (diastolic) wave, the second-derivative b/a ratio, how alike the beats are, and the beat,
its first and second derivative as 25 numbers each.

The v2 deep-pulse features (v2_features.py deep) only describe heart rate, rhythm and signal quality, so a network
whose wave follows the finger's *shape* better (RhythmFormer) had no way to show it. These numbers give it one.

    .venv/Scripts/python.exe scripts/pulse_shape.py check              camera vs finger shape numbers (no health labels)
    .venv/Scripts/python.exe scripts/pulse_shape.py build <model> ...  features/v2_pulse_shape_<model>.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import find_peaks, resample_poly
from scipy.stats import rankdata, skew

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_core as C                                                        # noqa: E402
from v2_features import bandpass, fill, peak                               # noqa: E402

FS, UP = 30.0, 4                     # the pulse is upsampled 4x (band-limited) before the beats are cut
MODELS = ("tracenet", "physnet_diff", "rhythmformer_pure")
SCALARS = ["beat_duration_s", "rise_time_frac", "rise_time_s", "width25_frac", "width50_frac", "width75_frac",
           "template_corr", "notch_found", "notch_height", "reflection_index", "delta_t_frac", "delta_t_s", "apg_b_a",
           "systolic_area_frac", "beats_used"]


def shape_row(w, seconds=60.0):
    """Shape numbers of one pulse wave (first `seconds`). NaN where too few clean beats are found."""
    out = {k: np.nan for k in SCALARS} | {f"{n}_{j:02d}": np.nan for n in ("tmpl", "d1", "d2") for j in range(25)}
    w = np.asarray(w, float)[:int(seconds * FS)]
    ok = np.flatnonzero(np.isfinite(w))
    if len(ok) < 20 * FS:
        return out
    w = fill(w[:ok[-1] + 1])
    if w is None or np.std(w) == 0:
        return out
    hz, _ = peak(bandpass(w, 0.7, 3.5), (0.7, 3.5), nperseg=1024)
    fs = FS * UP
    xf = resample_poly(bandpass(w, 0.5, 8.0), UP, 1)
    if skew(np.diff(xf)) < 0:                                        # the fast upstroke points up
        xf = -xf
    dist = int(fs * 0.5 / hz) if np.isfinite(hz) and hz > 0 else int(0.3 * fs)
    peaks, _ = find_peaks(xf, distance=max(dist, int(0.25 * fs)), prominence=0.3 * np.std(xf))
    if len(peaks) < 15:
        return out
    feet = np.array([a + np.argmin(xf[a:b]) for a, b in zip(peaks[:-1], peaks[1:])])
    beats, durs = [], []
    for a, b in zip(feet[:-1], feet[1:]):
        if 0.33 * fs <= b - a <= 2.0 * fs:
            seg = xf[a:b + 1]
            seg = (seg - seg.min()) / (np.ptp(seg) + 1e-9)
            beats.append(np.interp(np.linspace(0, len(seg) - 1, 100), np.arange(len(seg)), seg))
            durs.append((b - a) / fs)
    if len(beats) < 10:
        return out
    beats = np.array(beats)
    tmpl = np.median(beats, 0)
    T = float(np.median(durs))
    ip = int(np.argmax(tmpl))
    out.update({"beat_duration_s": T, "rise_time_frac": ip / 100, "rise_time_s": ip / 100 * T,
                "template_corr": float(np.mean([np.corrcoef(bt, tmpl)[0, 1] for bt in beats])),
                "systolic_area_frac": float(tmpl[:ip + 1].sum() / (tmpl.sum() + 1e-9)), "beats_used": float(len(beats))})
    for lvl in (0.25, 0.5, 0.75):
        out[f"width{int(lvl * 100)}_frac"] = float(np.mean(tmpl > lvl))
    d1 = np.gradient(tmpl)
    d2 = np.gradient(d1)
    mins, _ = find_peaks(-tmpl[ip + 5:90]) if ip + 5 < 88 else (np.array([], int), None)
    if len(mins):
        n = ip + 5 + mins[0]
        dia = n + int(np.argmax(tmpl[n:90]))
        out.update({"notch_found": 1.0, "notch_height": float(tmpl[n]), "reflection_index": float(tmpl[dia]),
                    "delta_t_frac": (dia - ip) / 100, "delta_t_s": (dia - ip) / 100 * T})
    else:
        out["notch_found"] = 0.0
    a_i = int(np.argmax(d2[:max(ip, 3)]))
    b_i = a_i + int(np.argmin(d2[a_i:ip + 10])) if ip + 10 > a_i else a_i
    if d2[a_i] > 0:
        out["apg_b_a"] = float(d2[b_i] / d2[a_i])
    for name, v in (("tmpl", tmpl), ("d1", d1 / (np.abs(d1).max() + 1e-9)), ("d2", d2 / (np.abs(d2).max() + 1e-9))):
        for j, val in enumerate(v[::4]):
            out[f"{name}_{j:02d}"] = float(val)
    return out


def shape_matrix(waves):
    rows = Parallel(n_jobs=8)(delayed(shape_row)(w) for w in waves)
    names = list(rows[0])
    return np.array([[r[n] for n in names] for r in rows], float), names


def finger_waves():
    import rppg_deep as R
    ppg, q = R.load_labels()
    sign = np.load(C.CACHE2 / "ppg_sign.npy") if (C.CACHE2 / "ppg_sign.npy").exists() else np.ones(len(C.df))
    clean = np.isfinite(q[:, 0]) & (q[:, 0] > 0)
    return ppg * sign[:, None], clean


def partial_spearman(x, y, z):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 30:
        return np.nan, np.nan, int(ok.sum())
    rx, ry, rz = (rankdata(v[ok]) for v in (x, y, z))
    Z = np.column_stack([np.ones(ok.sum()), rz])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1]), float(np.corrcoef(res(rx), res(ry))[0, 1]), int(ok.sum())


def check():
    """No health labels: (1) do the 30-fps finger shape numbers match the finger PPG tier's 100-Hz ones? (2) do each
    network's camera shape numbers follow the finger's, beyond what heart rate explains?"""
    import rppg_deep as R
    fw, clean = finger_waves()
    Ff, names = shape_matrix(fw)
    finger_hr = np.array([R.hr_from_wave(R.bandpass(p))[0] for p in fw])
    rows = []
    ppg = C.load_features(C.FEATURES / "ppg.npz", C.SESS)
    ppg_names = C.feature_names(C.FEATURES / "ppg.npz")
    for n in SCALARS:
        m = "ppg_" + n                                                # the same number in the finger PPG tier
        if m in ppg_names:
            r, pr, k = partial_spearman(Ff[:, names.index(n)], ppg[:, ppg_names.index(m)], finger_hr)
            rows.append({"compare": "finger 30 fps vs finger PPG tier (100 Hz)", "number": n, "rho": r, "partial rho | HR": pr, "n": k})
    for m in MODELS:
        W = np.load(C.CACHE2 / "rppg" / m / "wave.npy")
        Fc, names_c = shape_matrix(W)
        assert names_c == names
        np.save(C.CACHE2 / "rppg" / m / "shape_check.npy", Fc)
        for j, n in enumerate(names):
            if n in SCALARS or n.startswith("tmpl_"):
                x, y = Fc[:, j].copy(), Ff[:, j].copy()
                x[~clean], y[~clean] = np.nan, np.nan
                r, pr, k = partial_spearman(x, y, finger_hr)
                rows.append({"compare": f"camera ({m}) vs finger, clean finger", "number": n, "rho": r, "partial rho | HR": pr, "n": k})
    tab = pd.DataFrame(rows)
    tab.to_csv(C.RESULTS / "v2" / "pulse_shape_agreement.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 400)
    print(tab[tab.number.isin(SCALARS)].round(3).to_string(index=False))
    t = tab[tab.number.str.startswith("tmpl_")]
    print("\nbeat-shape points (tmpl_00..24): median rho / median partial rho by network")
    print(t.groupby("compare")[["rho", "partial rho | HR"]].median().round(3).to_string())


def build(model):
    W = np.load(C.CACHE2 / "rppg" / model / "wave.npy")
    X, names = shape_matrix(W)
    out = C.FEATURES / f"v2_pulse_shape_{model}.npz"
    C.save_features(out, C.df["key"].tolist(), X, [f"shape {n}" for n in names])
    print(f"saved {out.name} {X.shape}; rows without enough clean beats: {int(np.isnan(X[:, 1]).sum())}")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    else:
        for m in sys.argv[2:]:
            build(m)
