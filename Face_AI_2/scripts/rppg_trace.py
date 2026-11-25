"""A camera-pulse network on the skin-colour TRACES (4 skin regions x RGB, plus POS and CHROM per region), trained
on the synced finger pulse over the whole ~3-minute video, CROSS-FITTED over the shuffle-0 person folds (every
person's camera pulse comes from a network that never saw that person). Fully convolutional, so it reads a whole
video at once. The finger pulse is only a training signal; at prediction time the network needs the video alone.

    .venv/Scripts/python.exe scripts/rppg_trace.py [--epochs 40]        cross-fitted camera pulse for all 1,200 videos
    .venv/Scripts/python.exe scripts/rppg_trace.py --strict             the strictly nested version (a check, see below)

Writes data/cache_v2/rppg/tracenet/wave_full.npy (1200 x 5460, NaN-padded), wave.npy (first minute) and metrics.json.

--strict: for every outer fold k, the training people's camera pulse comes from networks trained WITHOUT fold k
(inner cross-fitting over the other 4 folds), so not even the finger pulse of the test people touched the features
of the training people. Writes wave_full_strict_k{k}.npy; scripts/check_strict_nesting.py compares the two.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_core as C                                                        # noqa: E402
import rppg_deep as R                                                      # noqa: E402
from v2_features import bandpass, chrom, fill, pos                         # noqa: E402

N = 5460
OUT = C.CACHE2 / "rppg" / "tracenet"
SWAP = [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 12, 13, 16, 17, 14, 15, 18, 19]   # right <-> left cheek channels


def video_inputs(key):
    """(20, N) inputs: 12 normalised colour channels + POS and CHROM of the 4 regions; and a validity mask."""
    T = np.load(C.CACHE2 / "traces" / f"{key}.npz")
    rgb, valid = T["rgb"].astype(float), T["valid"]
    L = min(len(valid), N)
    X = np.zeros((20, N), np.float32)
    ok = np.zeros(N, bool)
    ok[:L] = valid[:L]
    k = 30                                                                  # 1-s moving average
    ker = np.ones(k) / k
    for r in range(4):
        cols = [fill(rgb[:L, 3 * r + c], valid[:L]) for c in range(3)]
        if any(c is None for c in cols) or min(np.min(c) for c in cols) <= 0:
            continue
        x = np.column_stack(cols)
        for c in range(3):
            ma = np.convolve(x[:, c], ker, mode="same")
            ma[:k] = ma[k]; ma[-k:] = ma[-k - 1]
            X[3 * r + c, :L] = x[:, c] / ma - 1
        for j, fn in enumerate((pos, chrom)):
            w = bandpass(fn(x), 0.6, 4.0)
            X[12 + 2 * r + j, :L] = w / (w.std() + 1e-9)
    for c in range(12):
        s = X[c, :L].std()
        X[c, :L] = np.clip(X[c, :L] / (s + 1e-9), -5, 5) if s > 0 else 0
    X[12:] = np.clip(X[12:], -5, 5)
    return X, ok


class Block(nn.Module):
    def __init__(self, ch, d):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(ch, ch, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(ch), nn.GELU(),
                                 nn.Conv1d(ch, ch, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(ch))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class TraceNet2(nn.Module):
    def __init__(self, cin=20, ch=64, dil=(1, 2, 4, 8, 1, 2, 4, 8)):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv1d(cin, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.GELU())
        self.blocks = nn.Sequential(*[Block(ch, d) for d in dil])
        self.out = nn.Conv1d(ch, 1, 1)

    def forward(self, x):                                                   # (B, 20, T) -> (B, T)
        return self.out(self.blocks(self.inp(x)))[:, 0]


class Data:
    """Everything in GPU memory once: inputs, labels (finger pulse aligned with the face: y[t] = ppg[t-2])."""
    def __init__(self, dev="cuda"):
        from joblib import Parallel, delayed
        t0 = time.time()
        res = Parallel(n_jobs=8)(delayed(video_inputs)(k) for k in C.df["key"])
        self.X = np.stack([r[0] for r in res]); self.OK = np.stack([r[1] for r in res])
        P = np.load(C.CACHE2 / "ppg_full.npz")["ppg"]
        Y = np.concatenate([np.repeat(P[:, :1], R.LAG, 1), P[:, :-R.LAG]], 1)
        self.Yok = np.isfinite(Y)
        self.Yfirst = np.nan_to_num(Y[:, :1800])
        self.lens = (self.OK & self.Yok).sum(1)
        self.Xt = torch.tensor(self.X, device=dev)
        self.Yt = torch.tensor(np.where(self.Yok, Y, 0).astype(np.float32), device=dev)
        self.snr = np.load(C.CACHE2 / "ppg_quality.npy")[:, 0]
        self.dev = dev
        print(f"inputs ready {self.X.shape} in {time.time() - t0:.0f}s", flush=True)


def train_model(D, train_idx, seed, epochs=40, win=300, per_video=8, tag=""):
    """Train on the given videos (clean finger pulse only; 10% of their people held back to pick the best epoch)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr_all = np.asarray([v for v in train_idx if D.snr[v] > 0])
    people = np.unique(C.pids[tr_all])
    va_people = set(rng.choice(people, size=len(people) // 10, replace=False))
    is_va = np.array([C.pids[v] in va_people for v in tr_all])
    tr, va = tr_all[~is_va], tr_all[is_va]
    net = TraceNet2().to(D.dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    steps = epochs * (len(tr) * per_video // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    best, best_state, log = -np.inf, None, []
    for ep in range(epochs):
        net.train()
        vids = np.repeat(tr, per_video)
        rng.shuffle(vids)
        losses = []
        for b in range(len(vids) // 64):
            v = vids[b * 64:(b + 1) * 64]
            s = np.array([rng.integers(0, max(int(D.lens[i]) - win, 1)) for i in v])
            idx = torch.tensor(s[:, None] + np.arange(win)[None], device=D.dev)
            vv = torch.tensor(v, device=D.dev)
            xb = D.Xt[vv].gather(2, idx[:, None, :].expand(-1, D.X.shape[1], -1))
            yb = D.Yt[vv].gather(1, idx)
            yb = (yb - yb.mean(1, keepdim=True)) / (yb.std(1, keepdim=True) + 1e-6)
            if rng.random() < 0.5:
                xb = xb[:, SWAP]
            p = net(xb)
            loss = R.neg_pearson(p, yb) + 0.1 * R.spectral_ce(p, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            losses.append(loss.item())
        if ep % 4 == 3 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                pv = {i: net(D.Xt[i:i + 1])[0, :1800].cpu().numpy() for i in va}
            m = R.evaluate(pv, D.Yfirst, va)
            score = m["wave_r_median"] - m["hr_mae"] / 20
            log.append(dict(tag=tag, epoch=ep + 1, loss=float(np.mean(losses)), **{f"val_{n}": float(x) for n, x in m.items()}))
            print(f"{tag} ep {ep + 1:2d} loss {np.mean(losses):.4f} | val HR MAE {m['hr_mae']:.2f} within5 {m['within5']:.2f} "
                  f"wave r {m['wave_r_median']:.3f}", flush=True)
            if score > best:
                best, best_state = score, {n: t.detach().clone() for n, t in net.state_dict().items()}
    return best_state, log


@torch.no_grad()
def predict(D, state, idx):
    net = TraceNet2().to(D.dev)
    net.load_state_dict(state)
    net.eval()
    out = {}
    for i in idx:
        w = net(D.Xt[i:i + 1])[0].cpu().numpy()
        w[~D.OK[i]] = np.nan
        out[i] = w
    return out


def crossfit(D, epochs):
    fold = C.session_folds(0)
    wave = np.full((len(C.df), N), np.nan, np.float32)
    logs = []
    for k in range(C.N_FOLDS):
        state, log = train_model(D, np.flatnonzero(fold != k), C.SEED + k, epochs, tag=f"fold {k}")
        torch.save(state, OUT / f"fold{k}.pt")
        for i, w in predict(D, state, np.flatnonzero(fold == k)).items():
            wave[i] = w
        logs += log
    np.save(OUT / "wave_full.npy", wave)
    first = np.nan_to_num(wave[:, :1800])
    np.save(OUT / "wave.npy", first)
    m_all = R.evaluate({i: first[i] for i in range(len(C.df))}, D.Yfirst, np.arange(len(C.df)))
    clean = np.flatnonzero(D.snr > 0)
    m_clean = R.evaluate({i: first[i] for i in clean}, D.Yfirst, clean)
    (OUT / "metrics.json").write_text(json.dumps(dict(all=m_all, clean_ppg=m_clean, log=logs), indent=1))
    print("ALL:", json.dumps({n: round(v, 3) for n, v in m_all.items()}))
    print("CLEAN PPG:", json.dumps({n: round(v, 3) for n, v in m_clean.items()}))


def strict(D, epochs):
    fold = C.session_folds(0)
    for k in range(C.N_FOLDS):
        path = OUT / f"wave_full_strict_k{k}.npy"
        if path.exists():
            continue
        wave = np.full((len(C.df), N), np.nan, np.float32)
        state = torch.load(OUT / f"fold{k}.pt", weights_only=True)                       # trained on all folds != k
        for i, w in predict(D, state, np.flatnonzero(fold == k)).items():
            wave[i] = w
        for j in [f for f in range(C.N_FOLDS) if f != k]:                                # inner cross-fitting without k
            st, _ = train_model(D, np.flatnonzero((fold != k) & (fold != j)), C.SEED + 100 * k + j, epochs, tag=f"strict k{k} j{j}")
            for i, w in predict(D, st, np.flatnonzero(fold == j)).items():
                wave[i] = w
        np.save(path, wave)
        print(f"saved {path.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    D = Data()
    strict(D, a.epochs) if a.strict else crossfit(D, a.epochs)


if __name__ == "__main__":
    main()
