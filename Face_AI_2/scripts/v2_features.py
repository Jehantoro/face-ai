"""Label-free v2 video features, one row per session (df order), saved in features/:

  v2_pulse_classic.npz   pulse from skin colour over the WHOLE ~3-minute video (POS and CHROM on 4 skin regions):
                         heart rate, quality, a heart-rate trajectory in 20-s windows (start, end, slope: the recovery
                         after the squats), beat-to-beat variability.
  v2_breathing.npz       breathing from the vertical motion of both shoulders, the chest and the head, and from slow
                         skin-colour changes: rate, strength and amplitude over the whole video and the first minute.
  v2_pulse_deep_<m>.npz  the same pulse summary from a deep network's camera pulse (first minute; rppg_deep.py output).

    .venv/Scripts/python.exe scripts/v2_features.py classic|breathing|deep <model>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, filtfilt, find_peaks, welch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_core as C                                                        # noqa: E402

FS = 30.0
HR_BAND, RESP_BAND = (0.7, 3.5), (0.13, 0.6)
REGIONS = ["forehead", "right cheek", "left cheek", "face"]


def bandpass(x, lo, hi, fs=FS, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], "band")
    return filtfilt(b, a, x, axis=0)


def fill(x, valid=None):
    x = np.array(x, float)
    bad = ~np.isfinite(x) if valid is None else (~np.asarray(valid, bool) | ~np.isfinite(x))
    if bad.all():
        return None
    i = np.arange(len(x))
    x[bad] = np.interp(i[bad], i[~bad], x[~bad])
    return x


def pos(rgb, win=1.6):
    w = int(win * FS)
    W = sliding_window_view(rgb, (w, 3))[:, 0]
    Cn = W / W.mean(axis=1, keepdims=True)
    S1, S2 = Cn[..., 1] - Cn[..., 2], -2 * Cn[..., 0] + Cn[..., 1] + Cn[..., 2]
    h = S1 + (S1.std(axis=1, keepdims=True) / (S2.std(axis=1, keepdims=True) + 1e-9)) * S2
    h = h - h.mean(axis=1, keepdims=True)
    out, cnt = np.zeros(len(rgb)), np.zeros(len(rgb))
    for i in range(w):
        out[i:i + len(h)] += h[:, i]
        cnt[i:i + len(h)] += 1
    return out / np.maximum(cnt, 1)


def chrom(rgb):
    n = rgb / rgb.mean(0)
    X = bandpass(3 * n[:, 0] - 2 * n[:, 1], *HR_BAND)
    Y = bandpass(1.5 * n[:, 0] + n[:, 1] - 1.5 * n[:, 2], *HR_BAND)
    return X - (X.std() / (Y.std() + 1e-9)) * Y


def peak(x, band, fs=FS, nperseg=None, around=None, width=0.25):
    """Strongest frequency in `band` (optionally only within +-width Hz of `around`), and an SNR in dB:
    power within +-0.1 Hz of the peak and of its 2nd harmonic vs the rest of the band."""
    x = np.asarray(x, float)
    if x is None or not np.isfinite(x).all() or x.std() == 0 or len(x) < 64:
        return np.nan, np.nan
    f, p = welch(x - x.mean(), fs, nperseg=min(len(x), nperseg or 512), nfft=8192)
    m = (f >= band[0]) & (f <= band[1])
    sel = m & (np.abs(f - around) <= width) if around is not None and np.isfinite(around) else m
    if sel.sum() < 3 or p[m].sum() == 0:
        return np.nan, np.nan
    fp = f[sel][np.argmax(p[sel])]
    sig = p[m][(np.abs(f[m] - fp) <= 0.1) | (np.abs(f[m] - 2 * fp) <= 0.1)].sum()
    return float(fp), float(10 * np.log10(sig / max(p[m].sum() - sig, 1e-12)))


def hrv(wave, hr_bpm):
    if not np.isfinite(hr_bpm):
        return dict(ibi_sd_ms=np.nan, ibi_rmssd_ms=np.nan, ibi_pnn50=np.nan)
    pk, _ = find_peaks(bandpass(wave, *HR_BAND), distance=max(int(FS * 0.6 * 60 / hr_bpm), 6))
    ibi = np.diff(pk) / FS
    med = np.median(ibi) if len(ibi) else np.nan
    ibi = ibi[(ibi > 0.3) & (ibi < 2.0) & (np.abs(ibi - med) < 0.3 * med)]
    if len(ibi) < 10:
        return dict(ibi_sd_ms=np.nan, ibi_rmssd_ms=np.nan, ibi_pnn50=np.nan)
    d = np.diff(ibi)
    return dict(ibi_sd_ms=1000 * ibi.std(), ibi_rmssd_ms=1000 * np.sqrt(np.mean(d ** 2)), ibi_pnn50=float(np.mean(np.abs(d) > 0.05)))


def pulse_summary(wave, prefix, win_s=20, step_s=10):
    """Heart rate, quality, trajectory and variability of one camera pulse signal."""
    f = {}
    hz, snr = peak(bandpass(wave, *HR_BAND), HR_BAND, nperseg=1024)
    f[f"{prefix} hr"], f[f"{prefix} snr"] = 60 * hz, snr
    w, s = int(win_s * FS), int(step_s * FS)
    hrs, snrs, tt = [], [], []
    for a in range(0, len(wave) - w + 1, s):
        seg = bandpass(wave[a:a + w], *HR_BAND)
        h1, q1 = peak(seg, HR_BAND, around=hz if snr > -3 else None, width=0.3)   # track near the whole-video rate
        hrs.append(60 * h1); snrs.append(q1); tt.append((a + w / 2) / FS)
    hrs, tt = np.array(hrs), np.array(tt)
    ok = np.isfinite(hrs)
    f[f"{prefix} hr first window"] = hrs[0] if len(hrs) else np.nan
    f[f"{prefix} hr last window"] = hrs[-1] if len(hrs) else np.nan
    f[f"{prefix} hr window median"] = np.nanmedian(hrs) if ok.any() else np.nan
    f[f"{prefix} hr window sd"] = np.nanstd(hrs) if ok.sum() > 2 else np.nan
    f[f"{prefix} hr slope bpm/min"] = float(np.polyfit(tt[ok] / 60, hrs[ok], 1)[0]) if ok.sum() > 2 else np.nan
    n3 = max(len(hrs) // 3, 1)
    f[f"{prefix} hr early minus late"] = (np.nanmedian(hrs[:n3]) - np.nanmedian(hrs[-n3:])) if ok.sum() > 2 else np.nan
    f[f"{prefix} window snr median"] = np.nanmedian(snrs) if len(snrs) else np.nan
    for k, v in hrv(wave, 60 * hz).items():
        f[f"{prefix} {k}"] = v
    return f


def load_traces(key):
    return np.load(C.CACHE2 / "traces" / f"{key}.npz")


# ------------------------------------------------------------------ classic pulse over the whole video
def classic_row(key):
    T = load_traces(key)
    rgb, valid = T["rgb"].astype(float), T["valid"]
    f, best = {}, (-np.inf, None)
    for r, reg in enumerate(REGIONS):
        cols = [fill(rgb[:, 3 * r + c], valid) for c in range(3)]
        if any(c is None for c in cols) or min(np.min(c) for c in cols) <= 0:
            for m in ("pos", "chrom"):
                f[f"{reg} {m} hr"], f[f"{reg} {m} snr"] = np.nan, np.nan
            continue
        x = np.column_stack(cols)
        for m, fn in (("pos", pos), ("chrom", chrom)):
            wave = fn(x)
            hz, snr = peak(bandpass(wave, *HR_BAND), HR_BAND, nperseg=1024)
            f[f"{reg} {m} hr"], f[f"{reg} {m} snr"] = 60 * hz, snr
            if np.isfinite(snr) and snr > best[0]:
                best = (snr, wave)
    hrs = np.array([f[k] for k in f if k.endswith(" hr")], float)
    snrs = np.array([f[k] for k in f if k.endswith(" snr")], float)
    ok = np.isfinite(hrs) & np.isfinite(snrs)
    f["hr median of 8"] = np.median(hrs[ok]) if ok.any() else np.nan
    f["hr spread of 8"] = np.std(hrs[ok]) if ok.any() else np.nan
    f["hr agreement (share within 5 bpm of median)"] = float(np.mean(np.abs(hrs[ok] - np.median(hrs[ok])) <= 5)) if ok.any() else np.nan
    if best[1] is not None:
        f.update(pulse_summary(best[1], "best"))
        f.update(pulse_summary(best[1][:1800], "best first minute"))
    f["valid fraction"] = float(np.mean(valid))
    f["duration s"] = float(T["grid_seconds"][-1])
    return f


# ------------------------------------------------------------------ breathing
def breathing_row(key):
    T = load_traces(key)
    valid = T["valid"]
    lm = T["landmarks"].astype(float)
    sig = {"right shoulder": T["motion"][:, 0], "left shoulder": T["motion"][:, 1], "chest": T["motion"][:, 2],
           "head vertical": np.nanmean(lm[:, 1::2], 1), "head sideways": np.nanmean(lm[:, 0::2], 1)}
    face_g = T["rgb"][:, 10].astype(float)                                   # whole-face green
    sig["face colour"] = face_g / np.nanmean(face_g) - 1
    f = {}
    rates, strengths = [], []
    for name, x in sig.items():
        x = fill(x, valid)
        for span, sl in (("whole", slice(None)), ("first minute", slice(0, 1800))):
            if x is None or len(x[sl]) < 600:
                f[f"{name} {span} breaths/min"], f[f"{name} {span} strength"], f[f"{name} {span} amplitude"] = np.nan, np.nan, np.nan
                continue
            y = bandpass(x[sl] - np.mean(x[sl]), 0.08, 0.7, order=2)
            hz, q = peak(y, RESP_BAND, nperseg=min(len(y), 1024))
            f[f"{name} {span} breaths/min"], f[f"{name} {span} strength"] = 60 * hz, q
            f[f"{name} {span} amplitude"] = float(np.std(y))
            if span == "whole" and name in ("right shoulder", "left shoulder", "chest", "head vertical") and np.isfinite(q):
                rates.append(60 * hz); strengths.append(q)
    rates, strengths = np.array(rates), np.array(strengths)
    f["breathing rate (strongest motion)"] = rates[np.argmax(strengths)] if len(rates) else np.nan
    f["breathing rate (median motion)"] = np.median(rates) if len(rates) else np.nan
    f["breathing rate spread"] = np.std(rates) if len(rates) else np.nan
    # shoulders together: the sum cancels sideways sway, the difference keeps it
    a, b = fill(T["motion"][:, 0], valid), fill(T["motion"][:, 1], valid)
    if a is not None and b is not None:
        y = bandpass((a + b) / 2 - np.mean((a + b) / 2), 0.08, 0.7, order=2)
        hz, q = peak(y, RESP_BAND, nperseg=1024)
        f["shoulders together breaths/min"], f["shoulders together strength"], f["shoulders together amplitude"] = 60 * hz, q, float(np.std(y))
    else:
        f["shoulders together breaths/min"] = f["shoulders together strength"] = f["shoulders together amplitude"] = np.nan
    f["head movement sd (px)"] = float(np.nanstd(np.nanmean(lm[:, 1::2], 1)))
    f["face size change (%)"] = float(100 * np.nanstd(T["box"][:, 2]) / np.nanmean(T["box"][:, 2]))
    return f


# ------------------------------------------------------------------ deep camera pulse
def deep_row(wave):
    """Pulse summary of a deep network's camera pulse: the first minute (10-s windows) and, when the network read
    the whole video, the whole video too (20-s windows: the heart-rate trajectory over ~3 minutes)."""
    wave = np.asarray(wave, float)
    ok = np.flatnonzero(np.isfinite(wave))
    if len(ok) < 900:
        return {}
    wave = fill(wave[:ok[-1] + 1])                                          # drop the end padding, fill small gaps
    f = pulse_summary(wave[:1800], "deep", win_s=10, step_s=5)
    g = pulse_summary(wave[:1800], "deep20", win_s=20, step_s=10)
    f.update({k: v for k, v in g.items() if "window" in k or "slope" in k or "early" in k})
    if len(wave) > 3600:
        f.update(pulse_summary(wave, "deep whole video", win_s=20, step_s=10))
    return f


def build(kind, model=None):
    keys = C.df["key"].tolist()
    if kind == "classic":
        rows = Parallel(n_jobs=8)(delayed(classic_row)(k) for k in keys)
        out = C.FEATURES / "v2_pulse_classic.npz"
    elif kind == "breathing":
        rows = Parallel(n_jobs=8)(delayed(breathing_row)(k) for k in keys)
        out = C.FEATURES / "v2_breathing.npz"
    else:
        d = C.CACHE2 / "rppg" / model
        waves = np.load(d / "wave_full.npy") if (d / "wave_full.npy").exists() else np.load(d / "wave.npy")
        rows = Parallel(n_jobs=8)(delayed(deep_row)(waves[i]) for i in range(len(keys)))
        out = C.FEATURES / f"v2_pulse_deep_{model}.npz"
    names = sorted({k for r in rows for k in r})
    X = np.array([[r.get(n, np.nan) for n in names] for r in rows], float)
    C.save_features(out, keys, X, names)
    print(f"saved {out.name} {X.shape}; columns with gaps: {int((~np.isfinite(X)).any(0).sum())}")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
