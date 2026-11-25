"""Deep camera-pulse (rPPG) networks for Face_AI_2 v2: train on the face clips with the synced finger pulse as the
target, CROSS-FITTED over the shuffle-0 person folds, so every person's camera pulse comes from a network that never
saw that person. The finger pulse is only a training signal for the pulse extractor; at prediction time the network
needs the video alone, and none of the 10 health labels is ever used here.

    .venv/Scripts/python.exe scripts/rppg_deep.py <model> [--epochs N] [--folds 0,1,2,3,4] [--zero-shot]

Writes data/cache_v2/rppg/<model>/wave.npy (1200 x 1800 camera pulse, df order), metrics.json and a log.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_core as C                                                        # noqa: E402
from rppg_models import build_model                                        # noqa: E402

FS = 30.0
CLIPS = C.CACHE2 / "clips72"
LAG = 2          # measured on 150 clean videos: the camera pulse at frame t matches the synced finger value at t - 2
                 # (median best lag; IQR -4..-2 frames, drift within the minute ~0), so label[t] = ppg[t - 2]
SHIFTS = 3       # the loss takes the best Pearson over +-3 frames, so the remaining per-video jitter does not hurt
torch.backends.cudnn.benchmark = True


# ------------------------------------------------------------------ data
def load_labels():
    P = np.load(C.PROJECT / "data" / "cache" / "ppg_sync.npz")
    assert (P["keys"] == C.df["key"].to_numpy()).all()
    ppg = P["ppg"].astype(np.float32)
    ppg = np.concatenate([np.repeat(ppg[:, :1], LAG, 1), ppg[:, :-LAG]], 1)          # align with the face: y[t] = ppg[t-2]
    q = np.load(C.CACHE2 / "ppg_quality.npy") if (C.CACHE2 / "ppg_quality.npy").exists() else None
    return ppg, q


class ClipWindows(torch.utils.data.Dataset):
    """Random windows of `T` frames from the cached 72x72 face clips, with the finger pulse as target."""
    def __init__(self, idx, ppg, T, per_video, size, sign, train=True):
        self.idx, self.ppg, self.T, self.per_video, self.size, self.sign, self.train = idx, ppg, T, per_video, size, sign, train

    def __len__(self):
        return len(self.idx) * self.per_video

    def __getitem__(self, n):
        v = int(self.idx[n % len(self.idx)])
        clip = np.load(CLIPS / f"{C.df['key'].iat[v]}.npy", mmap_mode="r")
        rng = np.random.default_rng() if self.train else np.random.default_rng(n)
        s = int(rng.integers(0, clip.shape[0] - self.T + 1))
        x = np.asarray(clip[s:s + self.T], np.float32)                     # (T, 72, 72, 3)
        y = self.ppg[v, s:s + self.T].astype(np.float32) * self.sign[v]
        if self.train:
            if rng.random() < 0.5:
                x = x[:, :, ::-1]                                          # mirror (the pulse does not care)
            if rng.random() < 0.5:                                         # small zoom / shift jitter
                c = int(rng.integers(0, 9))
                x = x[:, c // 3:72 - (2 - c // 3), c % 3:72 - (2 - c % 3)]
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(3, 0, 1, 2)  # (3, T, H, W)
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        y = (y - y.mean()) / (y.std() + 1e-6)
        return x, torch.from_numpy(y)


# ------------------------------------------------------------------ losses and heart rate
def _pearson(a, b):
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)


def neg_pearson(pred, y, shifts=SHIFTS):
    """1 - the best Pearson correlation over label shifts of -shifts..+shifts frames (per sample)."""
    T = pred.shape[1]
    p = pred[:, shifts:T - shifts]
    r = torch.stack([_pearson(p, y[:, shifts + s:T - shifts + s]) for s in range(-shifts, shifts + 1)], 1)
    return (1 - r.max(1).values).mean()


def spectral_ce(pred, y, fs=FS, lo=0.7, hi=3.5, nfft=1024):
    """Cross-entropy between the predicted signal's power spectrum (as logits) and the label's peak frequency."""
    win = torch.hann_window(pred.shape[1], device=pred.device)
    P = torch.fft.rfft((pred - pred.mean(1, keepdim=True)) * win, n=nfft).abs() ** 2
    Y = torch.fft.rfft((y - y.mean(1, keepdim=True)) * win, n=nfft).abs() ** 2
    f = torch.fft.rfftfreq(nfft, 1 / fs).to(pred.device)
    m = (f >= lo) & (f <= hi)
    P, Y = P[:, m], Y[:, m]
    target = Y.argmax(1)
    return F.cross_entropy(torch.log(P / P.sum(1, keepdim=True) + 1e-8) * 1.0, target)


def hr_from_wave(w, fs=FS, lo=0.7, hi=3.5):
    from scipy.signal import welch
    w = np.asarray(w, float)
    if not np.isfinite(w).all() or w.std() == 0:
        return np.nan, np.nan
    f, p = welch(w - w.mean(), fs, nperseg=min(len(w), 512), nfft=4096)
    m = (f >= lo) & (f <= hi)
    fp = f[m][np.argmax(p[m])]
    sig = p[m][(np.abs(f[m] - fp) <= 0.1) | (np.abs(f[m] - 2 * fp) <= 0.1)].sum()
    return 60 * fp, 10 * np.log10(sig / max(p[m].sum() - sig, 1e-12))


def bandpass(x, lo=0.7, hi=3.5, fs=FS):
    from scipy.signal import butter, filtfilt
    b, a = butter(3, [lo / (fs / 2), hi / (fs / 2)], "band")
    return filtfilt(b, a, x)


# ------------------------------------------------------------------ inference over a whole minute
@torch.no_grad()
def predict_video(model, v, T, size, device, stride=None, fp32=True):
    stride = stride or T // 2
    clip = np.load(CLIPS / f"{C.df['key'].iat[v]}.npy", mmap_mode="r")
    n = clip.shape[0]
    starts = list(range(0, n - T + 1, stride))
    if starts[-1] != n - T:
        starts.append(n - T)
    out, cnt = np.zeros(n), np.zeros(n)
    win = np.hanning(T) + 1e-3
    xs = []
    for s in starts:
        x = torch.from_numpy(np.asarray(clip[s:s + T], np.float32)).permute(3, 0, 1, 2)
        if x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        xs.append(x)
    for b in range(0, len(xs), 8):
        batch = torch.stack(xs[b:b + 8]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not fp32):
            pred = model(batch).float().cpu().numpy()
        for k, s in enumerate(starts[b:b + 8]):
            p = pred[k]
            p = (p - p.mean()) / (p.std() + 1e-6)
            out[s:s + T] += p * win
            cnt[s:s + T] += win
    return out / np.maximum(cnt, 1e-6)


def evaluate(waves, ppg, idx):
    hr_c = np.array([hr_from_wave(bandpass(waves[v]))[0] for v in idx])
    hr_f = np.array([hr_from_wave(bandpass(ppg[v]))[0] for v in idx])
    err = np.abs(hr_c - hr_f)
    def best_r(a, b, s=SHIFTS):
        return max(np.corrcoef(a[s:len(a) - s], b[s + k:len(b) - s + k])[0, 1] for k in range(-s, s + 1))
    r = np.array([best_r(bandpass(waves[v]), bandpass(ppg[v])) for v in idx])
    return dict(n=len(idx), hr_mae=float(np.nanmean(err)), hr_median_ae=float(np.nanmedian(err)),
                within5=float(np.nanmean(err <= 5)), within10=float(np.nanmean(err <= 10)),
                hr_r=float(np.corrcoef(hr_c[np.isfinite(hr_c)], hr_f[np.isfinite(hr_c)])[0, 1]),
                wave_r_median=float(np.nanmedian(r)))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--per-video", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--zero-shot", action="store_true", help="no training: predict with the pretrained weights only")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    device = "cuda"
    out_dir = C.CACHE2 / "rppg" / (a.model + ("_zeroshot" if a.zero_shot else "") + a.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    ppg, q = load_labels()
    snr = q[:, 0] if q is not None else np.zeros(len(C.df))
    sign = np.load(C.CACHE2 / "ppg_sign.npy") if (C.CACHE2 / "ppg_sign.npy").exists() else np.ones(len(C.df))
    fold = C.session_folds(0)
    wave_path = out_dir / "wave.npy"
    waves = np.load(wave_path) if wave_path.exists() else np.full((len(C.df), 1800), np.nan, np.float32)
    log = []
    for k in [int(s) for s in a.folds.split(",")]:
        model, spec = build_model(a.model)
        T, size = spec["T"], spec["size"]
        model = model.to(device)
        te = np.flatnonzero(fold == k)
        if not a.zero_shot:
            tr_all = np.flatnonzero((fold != k) & np.isfinite(snr) & (snr > 0))      # clean finger pulse only
            rng = np.random.default_rng(C.SEED + k)
            people = np.unique(C.pids[tr_all])
            va_people = set(rng.choice(people, size=len(people) // 10, replace=False))
            is_va = np.array([C.pids[v] in va_people for v in tr_all])
            tr, va = tr_all[~is_va], tr_all[is_va]
            dl = torch.utils.data.DataLoader(ClipWindows(tr, ppg, T, a.per_video, size, sign), batch_size=a.batch,
                                             shuffle=True, num_workers=a.workers, persistent_workers=True,
                                             pin_memory=False, drop_last=True)
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
            steps = a.epochs * len(dl)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
            best, best_state = -np.inf, None
            for ep in range(a.epochs):
                model.train()
                t0, losses = time.time(), []
                for x, y in dl:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not spec.get("fp32", False)):
                        p = model(x).float()
                    loss = neg_pearson(p, y) + 0.1 * spectral_ce(p, y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step(); sched.step()
                    losses.append(loss.item())
                if ep % 2 == 0 and ep != a.epochs - 1:                       # validate every 2nd epoch and at the end
                    print(f"fold {k} ep {ep + 1:2d} loss {np.mean(losses):.4f} ({time.time() - t0:.0f}s)", flush=True)
                    continue
                model.eval()
                vw = {v: predict_video(model, v, T, size, device) for v in va[:60]}
                vm = evaluate(vw, ppg * sign[:, None], va[:60])
                score = vm["wave_r_median"] - vm["hr_mae"] / 20
                log.append(dict(fold=k, epoch=ep + 1, loss=float(np.mean(losses)), seconds=time.time() - t0, **{f"val_{n}": v for n, v in vm.items()}))
                print(f"fold {k} ep {ep + 1:2d} loss {np.mean(losses):.4f} | val HR MAE {vm['hr_mae']:.2f} within5 {vm['within5']:.2f} "
                      f"wave r {vm['wave_r_median']:.3f} ({time.time() - t0:.0f}s)", flush=True)
                if score > best:
                    best, best_state = score, {n: v.detach().clone() for n, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            torch.save(best_state, out_dir / f"fold{k}.pt")
        model.eval()
        for v in te:
            waves[v] = predict_video(model, v, T, size, device)
        np.save(wave_path, waves)
        tm = evaluate(waves, ppg * sign[:, None], te)
        print(f"fold {k} TEST: {json.dumps({n: round(v, 3) for n, v in tm.items()})}", flush=True)
        log.append(dict(fold=k, epoch="test", **{f"test_{n}": v for n, v in tm.items()}))
        (out_dir / "log.json").write_text(json.dumps(log, indent=1))
    done = np.isfinite(waves).all(1)
    if done.all():
        m = evaluate(waves, ppg * sign[:, None], np.arange(len(C.df)))
        m_clean = evaluate(waves, ppg * sign[:, None], np.flatnonzero(np.isfinite(snr) & (snr > 0)))
        (out_dir / "metrics.json").write_text(json.dumps(dict(all=m, clean_ppg=m_clean), indent=1))
        print("ALL:", json.dumps({n: round(v, 3) for n, v in m.items()}))
        print("CLEAN PPG:", json.dumps({n: round(v, 3) for n, v in m_clean.items()}))


if __name__ == "__main__":
    main()
