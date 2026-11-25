"""The synced finger pulse for the WHOLE video on the v2 30 Hz clock (time since the first timing-file row, as in
data/cache/ppg_sync.npz), so camera-pulse networks can learn from all ~3 minutes, not only the first one.
Row i of a ppg_sync file belongs to row i of the video's timing file.

    .venv/Scripts/python.exe scripts/ppg_full.py   ->  data/cache_v2/ppg_full.npz  (keys, ppg (1200 x 5460, NaN-padded), n)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_core as C
from extract_video_v2 import RAW, load_timing

N = 5460                                                    # 182 s at 30 Hz (longest video ~180.5 s)
out = np.full((len(C.df), N), np.nan, np.float32)
n_ok = np.zeros(len(C.df), int)
for i, key in enumerate(C.df["key"]):
    vals = np.loadtxt(C.PROJECT / "data" / "ppg_sync" / f"{key}.txt", ndmin=2)[:, 0]
    elapsed, _ = load_timing(RAW / "meta" / f"{key}.txt")
    n = min(len(vals), len(elapsed))
    grid = np.arange(N) / 30.0
    inside = grid <= elapsed[n - 1]
    out[i, inside] = np.interp(grid[inside], elapsed[:n], vals[:n])
    n_ok[i] = int(inside.sum())
P = np.load(C.PROJECT / "data" / "cache" / "ppg_sync.npz")
assert (P["keys"] == C.df["key"].to_numpy()).all()
print("max |difference| to ppg_sync.npz on the first minute:", float(np.nanmax(np.abs(out[:, :1800] - P["ppg"]))))
np.savez_compressed(C.CACHE2 / "ppg_full.npz", keys=C.df["key"].to_numpy().astype(str), ppg=out, n=n_ok)
print("saved; samples per video min/median/max:", n_ok.min(), int(np.median(n_ok)), n_ok.max())
