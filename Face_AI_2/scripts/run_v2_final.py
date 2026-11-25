"""The final v2 evaluation, in one command (every route must already be cached by `v2_nested.py run ...`):

1. stacking routes with the honest predicted age / BMI / sex (stage 1 = every face route);
2. the two v2 tiers (video only; video + PPG + ECG): EVERY route below is a candidate and only the inner loop chooses,
   including whether to average the best 2-3 routes. The route lists were fixed before looking at v2 tier results;
3. the head-to-head with Face_AI's newest model on its 90 test people (all routes refitted on its 510 people).

    .venv/Scripts/python.exe scripts/run_v2_final.py [--skip-h2h]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_core as C                                                        # noqa: E402
import v2_nested as N                                                      # noqa: E402

N.add_dynamic_routes()
DEEP = [r[len("deep_"):] for r in N.ROUTES if r.startswith("deep_")]
APP = (["dino", "clip", "appearance"] + [r for r in N.ROUTES if r.startswith("app_")] + list(N.MLP_ROUTES) + list(N.MLP_AUG_ROUTES))
SIGNAL = (["temporal", "pulse2", "breath2", "signals2"] + [f"deep_{m}" for m in DEEP] + [f"signals2+deep_{m}" for m in DEEP]
          + (["signals2+deep_all"] if "signals2+deep_all" in N.ROUTES else [])
          + [r for r in N.ROUTES if r.startswith(("shape_", "deep+shape_"))])   # pulse-shape routes (2 Nov; report 4.5)
CHAINS_VIDEO = {"v2_video_all+demo": ["dino", "clip", "temporal"],
                "v2_signals+demo": ["pulse2", "breath2", "temporal"] + [f"deep:{m}" for m in DEEP],
                "v2_demo": []}
CHAINS_MM = {"v2_video+physio+demo": ["dino", "clip", "temporal", "ppg", "ecg"]}
PHYSIO = ["ppg", "ecg", "physio", "video+physio", "ppg_hgb", "ecg_hgb", "physio+signals2"]
VIDEO_V2 = ["video_all"] + APP + SIGNAL + list(CHAINS_VIDEO)
MM_V2 = VIDEO_V2 + PHYSIO + list(CHAINS_MM)


def main():
    t0 = time.time()
    missing = [r for r in APP + SIGNAL + PHYSIO + ["video_all"] for k in range(3) if not N.route_path(r, k).exists()]
    if missing:
        print("routes still missing (run v2_nested.py run ...):", sorted(set(missing)))
        for r in sorted(set(missing)):
            for k in range(3):
                N.compute_route(r, k)
    for name, base in {**CHAINS_VIDEO, **CHAINS_MM}.items():
        for k in range(3):
            N.compute_chain(name, base, APP, k)
    print(f"chains done ({time.time() - t0:.0f}s)", flush=True)
    sys.argv = ["v2_nested.py", "tiers", "video_v2=" + ",".join(VIDEO_V2), "multimodal_v2=" + ",".join(MM_V2)]
    N.main()
    print(f"tiers done ({time.time() - t0:.0f}s)", flush=True)
    if "--skip-h2h" in sys.argv[1:] or "--skip-h2h" in sys.orig_argv:
        return
    for r in ["video_all"] + APP + SIGNAL:
        N.compute_route_fixed(r)
    for name, base in CHAINS_VIDEO.items():
        N.compute_chain_fixed(name, base, APP)
    sys.argv = ["v2_nested.py", "h2h", ",".join(VIDEO_V2)]
    N.main()
    print(f"head-to-head done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
