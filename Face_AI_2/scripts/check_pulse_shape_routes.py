"""Diagnostic: do pulse-SHAPE numbers from the camera pulse (pulse_shape.py) help the specialists?

Nested CV (same folds, candidates and rule as every v2 route, 3 shuffles) of each network's shape numbers alone and
added to its rhythm numbers, against the rhythm numbers alone. Decided before running: these routes join the v2
tiers only if adding shape numbers gives a clear gain (paired interval entirely below 0) for stiffness or breathing
with at least one network; otherwise they stay a diagnostic and the tiers are left as they are.

Outcome (2 November 2026): only PhysNet for breathing met the rule (-0.005 [-0.010, -0.001]; stiffness got worse with
shape numbers for every network), so the shape routes joined the tiers (v2_nested.add_dynamic_routes, run_v2_final.py).
The tiers did not change (video stiffness +0.003 [-0.003, +0.009]); see the report, section 4.5.

    .venv/Scripts/python.exe scripts/check_pulse_shape_routes.py run <model> ...   compute the routes (cached)
    .venv/Scripts/python.exe scripts/check_pulse_shape_routes.py report           -> results/v2/pulse_shape_routes.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_core as C                                                        # noqa: E402
import v2_nested as N                                                      # noqa: E402

MODELS = ("tracenet", "physnet_diff", "rhythmformer_pure")
TARGETS = ("respiratory", "rigidity", "stress")
N.add_dynamic_routes()                       # registers shape_<m> and deep+shape_<m> for every v2_pulse_shape_<m>.npz


def report():
    rows = []
    for m in MODELS:
        P = {name: {r: N.load_route(name, r)[0] for r in range(3)} for name in (f"deep_{m}", f"shape_{m}", f"deep+shape_{m}")}
        for t in TARGETS:
            s = {name: C.quick_scores(p, [t])[t] for name, p in P.items()}
            d, lo, hi = C.paired_ci(P[f"deep+shape_{m}"], P[f"deep_{m}"], t)
            rows.append({"network": m, "target": C.NAME[t], "rhythm numbers (MAE)": s[f"deep_{m}"][0],
                         "shape numbers (MAE)": s[f"shape_{m}"][0], "rhythm + shape (MAE)": s[f"deep+shape_{m}"][0],
                         "rhythm + shape minus rhythm": d, "lo": lo, "hi": hi})
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(tab.round(4).to_string(index=False))
    tab.to_csv(C.RESULTS / "v2" / "pulse_shape_routes.csv", index=False)
    gain = tab[tab.target.isin([C.NAME["respiratory"], C.NAME["rigidity"]]) & (tab.hi < 0)]
    print("\nclear gain (joins the tiers):" if len(gain) else "\nno clear gain: the shape routes stay a diagnostic")
    if len(gain):
        print(gain.round(4).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        for m in sys.argv[2:]:
            for r in range(C.N_REPEATS):
                N.compute_route(f"shape_{m}", r)
                N.compute_route(f"deep+shape_{m}", r)
    else:
        report()
