"""Is the stiffness prediction more than a heart-rate index? Partial (rank) correlation between predicted and
measured stiffness with the finger heart rate during the video held constant, for v1 and v2 (3 shuffles averaged).

    .venv/Scripts/python.exe scripts/stiffness_vs_heart_rate.py   ->  results/v2/stiffness_vs_heart_rate.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_core as C                                                        # noqa: E402
import rppg_deep as R                                                      # noqa: E402


def partial_spearman(x, y, z):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    rx, ry, rz = (rankdata(v[ok]) for v in (x, y, z))
    Z = np.column_stack([np.ones(ok.sum()), rz])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(res(rx), res(ry))[0, 1]), float(np.corrcoef(rx, ry)[0, 1])


def main():
    P = np.load(C.PROJECT / "data" / "cache" / "ppg_sync.npz")["ppg"]
    finger_hr = np.array([R.hr_from_wave(R.bandpass(p))[0] for p in P])
    y = C.Y_nat[:, C.ALL_TARGETS.index("rigidity")]
    j = C.ALL_TARGETS.index("rigidity")
    rows = []
    for name in ("video_specialists", "video_specialists_v2", "multimodal_specialists", "multimodal_specialists_v2"):
        if C.load_oof(name) is None:
            continue
        pr, plain = zip(*[partial_spearman(C.oof_matrix(name, r)[:, j], y, finger_hr) for r in range(3)])
        rows.append({"model": name, "spearman(pred, stiffness)": np.mean(plain),
                     "partial spearman | finger heart rate": np.mean(pr)})
    hr_only = partial_spearman(finger_hr, y, np.zeros_like(y) + np.random.default_rng(0).normal(size=len(y)))[1]
    rows.append({"model": "finger heart rate itself", "spearman(pred, stiffness)": hr_only, "partial spearman | finger heart rate": 0.0})
    tab = pd.DataFrame(rows)
    print(tab.round(3).to_string(index=False))
    tab.to_csv(C.RESULTS / "v2" / "stiffness_vs_heart_rate.csv", index=False)


if __name__ == "__main__":
    main()
