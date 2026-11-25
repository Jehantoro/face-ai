"""Check: does cross-fitting the camera-pulse network over the shuffle-0 folds (each person's pulse comes from a
network that never saw that person) give the same downstream results as STRICT nesting (for outer fold k, even the
training people's pulse comes from networks that never saw fold k's finger pulse)?

Runs the nested specialists on shuffle 0 with the trace network's pulse features both ways and prints the per-target
MAE of each and the paired difference. Needs rppg_trace.py and rppg_trace.py --strict to have run.

    .venv/Scripts/python.exe scripts/check_strict_nesting.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_core as C                                                        # noqa: E402
import v2_nested as N                                                      # noqa: E402
from v2_features import deep_row                                           # noqa: E402

OUT = C.CACHE2 / "rppg" / "tracenet"
TARGETS = ["respiratory", "rigidity", "saturation", "stress"]


def features(waves):
    rows = Parallel(n_jobs=8)(delayed(deep_row)(w) for w in waves)
    names = sorted({k for r in rows for k in r})
    return np.array([[r.get(n, np.nan) for n in names] for r in rows], float), names


def main():
    base = [N.block(b) for b in ("pulse2", "breath2", "temporal")]
    fold = C.session_folds(0)
    Xc, names_c = features(np.load(OUT / "wave_full.npy"))
    res = {}
    for label in ("cross-fitted", "strict"):
        oof = np.full((len(C.df), len(C.ALL_TARGETS)), np.nan)
        for k in range(C.N_FOLDS):
            if label == "strict":
                Xk, names_k = features(np.load(OUT / f"wave_full_strict_k{k}.npy"))
                assert names_k == names_c
            else:
                Xk = Xc
            X = np.column_stack(base + [Xk])
            sizes = [b.shape[1] for b in base] + [Xk.shape[1]]
            o, _, _, _ = C.run_route("check", X, sizes, fold, 0, use_hgb=True, outer=[k], hgb_targets=TARGETS)
            oof[fold == k] = o[fold == k]
        res[label] = {0: oof}
        print(label, {t: round(v[0], 4) for t, v in C.quick_scores(res[label], TARGETS).items()}, flush=True)
    rows = []
    for t in TARGETS:
        d, lo, hi = C.paired_ci(res["strict"], res["cross-fitted"], t)
        rows.append({"target": t, "MAE cross-fitted": C.quick_scores(res["cross-fitted"], [t])[t][0],
                     "MAE strict": C.quick_scores(res["strict"], [t])[t][0], "strict - cross-fitted": d, "lo": lo, "hi": hi})
    tab = pd.DataFrame(rows)
    print(tab.round(4).to_string(index=False))
    tab.to_csv(C.RESULTS / "v2" / "strict_nesting_check.csv", index=False)


if __name__ == "__main__":
    main()
