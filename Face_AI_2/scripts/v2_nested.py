"""v2 specialists: the notebook's nested cross-validation (same folds, candidates, selection rule and metrics) with new
feature routes, a per-crop network candidate for the appearance targets, and an ensemble rule chosen in the inner loop.

Every route's result is cached in results/v2/routes/<route>_r<repeat>.npz, so adding a route only computes that route.

    .venv/Scripts/python.exe scripts/v2_nested.py run <route> [<route> ...]     compute routes (3 shuffles)
    .venv/Scripts/python.exe scripts/v2_nested.py tiers                        assemble tiers + metrics
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_core as C                                                        # noqa: E402

V2 = C.RESULTS / "v2"
ROUTES_DIR = V2 / "routes"
APPEARANCE = ["age", "bmi", "sex"]


# ------------------------------------------------------------------ feature blocks
def _f(name, keys=None):
    return C.load_features(C.FEATURES / name, C.df["key"] if keys is None else keys)


def block(name):
    """Feature block by short name (mean features, df order)."""
    base = {"dino": ("dino_vits14.npz", None), "clip": ("clip_vitb16.npz", None), "temporal": ("rppg_v2.npz", None),
            "ppg": ("ppg.npz", C.SESS), "ecg": ("ecg.npz", C.SESS),
            "pulse2": ("v2_pulse_classic.npz", None), "breath2": ("v2_breathing.npz", None)}
    if name in base:
        f, keys = base[name]
        return _f(f, keys)
    if name.startswith("deep:"):                                    # deep:<model>
        return _f(f"v2_pulse_deep_{name[5:]}.npz")
    if name.startswith("shape:"):                                   # shape:<model> (pulse_shape.py; diagnostic only)
        return _f(f"v2_pulse_shape_{name[6:]}.npz")
    if name.startswith("app:"):                                     # app:<encoder>_<crop>
        return _f(f"v2_{name[4:]}.npz")
    raise KeyError(name)


def crops(name):
    with np.load(C.FEATURES / f"v2_{name}.npz", allow_pickle=False) as z:
        pos = {str(k): i for i, k in enumerate(z["keys"])}
        return z["crops"][[pos[k] for k in C.df["key"]]]


# route name -> (list of blocks, use trees?)
ROUTES = {
    # the notebook's routes (recomputed here so the ensemble rule has their inner predictions)
    "dino": (["dino"], False), "clip": (["clip"], False), "appearance": (["dino", "clip"], False),
    "temporal": (["temporal"], True), "video_all": (["dino", "clip", "temporal"], False),
    "ppg": (["ppg"], False), "ecg": (["ecg"], False), "physio": (["ppg", "ecg"], True),   # as v1: trees only on temporal/physio
    "ppg_hgb": (["ppg"], True), "ecg_hgb": (["ecg"], True),                                 # v2: trees on each sensor set too
    "video+physio": (["dino", "clip", "temporal", "ppg", "ecg"], False),
    # v2 signal routes
    "pulse2": (["pulse2"], True), "breath2": (["breath2"], True), "signals2": (["pulse2", "breath2", "temporal"], True),
}


def add_dynamic_routes():
    """Routes for whatever v2 feature files exist (deep pulse models, new face encoders)."""
    deep = []
    for p in sorted(C.FEATURES.glob("v2_pulse_deep_*.npz")):
        m = p.stem.replace("v2_pulse_deep_", "")
        deep.append(f"deep:{m}")
        ROUTES[f"deep_{m}"] = ([f"deep:{m}"], True)
        ROUTES[f"signals2+deep_{m}"] = (["pulse2", "breath2", "temporal", f"deep:{m}"], True)
    if len(deep) > 1:                                                 # every camera pulse together
        ROUTES["signals2+deep_all"] = (["pulse2", "breath2", "temporal"] + deep, True)
    ROUTES["physio+signals2"] = (["ppg", "ecg", "pulse2", "breath2"] + deep, True)
    for p in sorted(C.FEATURES.glob("v2_pulse_shape_*.npz")):         # pulse-shape numbers (pulse_shape.py), added 2 Nov
        m = p.stem.replace("v2_pulse_shape_", "")
        ROUTES[f"shape_{m}"] = ([f"shape:{m}"], True)
        ROUTES[f"deep+shape_{m}"] = ([f"deep:{m}", f"shape:{m}"], True)
    for p in sorted(C.FEATURES.glob("v2_*_wide.npz")) + sorted(C.FEATURES.glob("v2_*_tight.npz")):
        e = p.stem[3:]
        ROUTES[f"app_{e}"] = ([f"app:{e}"], False)


# ------------------------------------------------------------------ per-crop network candidate (appearance targets)
def crop_mlp_fit_predict(Ftr, Ytr, Fte, sex_mask_tr, seeds=3, epochs=40, hidden=256, lr=1e-3, wd=5e-2, drop=0.3):
    """A small network trained on single crops (each of the 24 crops of a video is one example, so the network sees
    24 views of every person), predicting age, BMI (log) and sex together; at test time its outputs are averaged
    over the crops. Returns predictions for all 10 targets: the 7 others are the training median (never chosen)."""
    import torch
    import torch.nn as nn
    dev = "cuda"
    n_tr, n_c, D = Ftr.shape
    mu, sd = Ftr.reshape(-1, D).mean(0), Ftr.reshape(-1, D).std(0) + 1e-6
    Xtr = torch.tensor(((Ftr - mu) / sd).reshape(-1, D), dtype=torch.float32, device=dev)
    Xte = torch.tensor(((Fte - mu) / sd).reshape(-1, D), dtype=torch.float32, device=dev)
    cols = [C.ALL_TARGETS.index(t) for t in APPEARANCE]
    reg = cols[:2]
    ym, ys = Ytr[:, reg].mean(0), Ytr[:, reg].std(0) + 1e-9
    yr = torch.tensor(np.repeat((Ytr[:, reg] - ym) / ys, n_c, 0), dtype=torch.float32, device=dev)
    ysex = torch.tensor(np.repeat(Ytr[:, cols[2]], n_c), dtype=torch.float32, device=dev)
    msex = torch.tensor(np.repeat(sex_mask_tr, n_c), dtype=torch.bool, device=dev)
    preds = []
    for s in range(seeds):
        torch.manual_seed(C.SEED + s)
        net = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, 3)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        steps_per_epoch = int(np.ceil(len(Xtr) / 256))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.1)
        for ep in range(epochs):
            perm = torch.randperm(len(Xtr), device=dev)
            net.train()
            for b in range(steps_per_epoch):
                i = perm[b * 256:(b + 1) * 256]
                out = net(Xtr[i])
                loss = nn.functional.smooth_l1_loss(out[:, :2], yr[i], beta=0.5)
                m = msex[i]
                if m.any():
                    loss = loss + 0.5 * nn.functional.binary_cross_entropy_with_logits(out[m, 2], ysex[i][m])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            preds.append((net(Xte).view(len(Fte), n_c, 3).mean(1).cpu().numpy(),
                          net(Xtr).view(n_tr, n_c, 3).mean(1).cpu().numpy()))
    p, p_tr = np.mean([a for a, _ in preds], 0), np.mean([b for _, b in preds], 0)
    out = np.tile(np.median(Ytr, 0), (len(Fte), 1))
    shift = np.median(Ytr[:, reg] - (p_tr[:, :2] * ys + ym), 0)      # aim at the median, like every other candidate
    out[:, reg] = p[:, :2] * ys + ym + shift
    out[:, cols[2]] = 1 / (1 + np.exp(-p[:, 2]))
    return out


def run_crop_route(route, F, repeat, outer=range(C.N_FOLDS), **kw):
    """Nested CV for the per-crop network: inner folds give the honest inner predictions used for selection."""
    fold = C.session_folds(repeat)
    oof = np.full((len(C.df), len(C.ALL_TARGETS)), np.nan)
    inner_best, log = {}, []
    for k in outer:
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        Ytr = C.Y_fit[tr].copy()
        sex_ok = np.isfinite(Ytr[:, -1])
        Ytr[~sex_ok, -1] = np.nanmean(Ytr[:, -1])
        inner = np.zeros((len(tr), len(C.ALL_TARGETS)))
        const_inner = np.zeros_like(inner)
        for va_mask in C.inner_splits(C.pids[tr], C.SEED + 1000 * repeat + 10 * k):
            itr = tr[~va_mask]
            inner[va_mask] = C.back_transform(crop_mlp_fit_predict(F[itr], Ytr[~va_mask], F[tr[va_mask]], np.isfinite(C.Y_fit[itr, -1]), **kw))
            cst = np.median(Ytr[~va_mask], 0)
            cst[-1] = Ytr[~va_mask][np.isfinite(C.Y_fit[itr, -1]), -1].mean()
            const_inner[va_mask] = C.back_transform(np.tile(cst, (va_mask.sum(), 1)))
        final = C.back_transform(crop_mlp_fit_predict(F[tr], Ytr, F[te], sex_ok, **kw))
        cst = np.median(Ytr, 0)
        cst[-1] = Ytr[sex_ok, -1].mean()
        const_final = C.back_transform(np.tile(cst, (len(te), 1)))
        for j, t in enumerate(C.ALL_TARGETS):
            s_net = C.score(inner[:, j], C.Y_nat[tr, j], C.pids[tr], t)
            s_const = C.score(const_inner[:, j], C.Y_nat[tr, j], C.pids[tr], t)
            use = t in APPEARANCE and s_net < s_const
            oof[te, j] = (final if use else const_final)[:, j]
            inner_best[(k, t)] = (inner if use else const_inner)[:, j]
            log.append({"repeat": repeat, "fold": k, "route": route, "target": t, "config": "cropnet" if use else "const",
                        "inner_score": s_net if use else s_const})
    return oof, pd.DataFrame(log), inner_best


# ------------------------------------------------------------------ small MLP heads (Face_AI's recipe) as a candidate
def mlp_heads_fit_predict(Xtr, Ytr, Xte, sizes, sex_mask_tr, pids_tr, seeds=3, hidden=64, drop=0.3, lr=3e-3, wd=1e-3,
                          max_epochs=400, patience=30):
    """Face_AI's task head on the per-video face numbers: dropout 0.3 -> Linear(p, 64) -> ReLU -> Linear(64, 1), one
    head per target (age, BMI on the log scale, sex), early stopping on 15% of the TRAINING people, 3 seeds averaged.
    The 7 other targets get the training median (so this candidate can only matter for age, BMI and sex)."""
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sc = BlockScalerProxy(sizes).fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    s = Ztr.std(0) + 1e-9                                                    # back to unit spread per column
    Ztr, Zte = Ztr / s, Zte / s
    out = np.tile(np.median(Ytr, 0), (len(Xte), 1))
    rng = np.random.default_rng(C.SEED)
    people = np.unique(pids_tr)
    va_p = set(rng.choice(people, size=max(len(people) * 15 // 100, 1), replace=False))
    is_va = np.array([p in va_p for p in pids_tr])
    Xt = torch.tensor(Ztr, dtype=torch.float32, device=dev)
    Xe = torch.tensor(Zte, dtype=torch.float32, device=dev)
    for t in APPEARANCE:
        j = C.ALL_TARGETS.index(t)
        m_ok = sex_mask_tr if t == "sex" else np.ones(len(Ytr), bool)
        tr_m, va_m = m_ok & ~is_va, m_ok & is_va
        y = Ytr[:, j].astype(np.float32)
        mu, sd = (0.0, 1.0) if t == "sex" else (y[tr_m].mean(), y[tr_m].std() + 1e-9)
        yt = torch.tensor((y - mu) / sd, dtype=torch.float32, device=dev)
        itr, iva = torch.tensor(np.flatnonzero(tr_m), device=dev), torch.tensor(np.flatnonzero(va_m), device=dev)
        loss_fn = nn.functional.binary_cross_entropy_with_logits if t == "sex" else nn.functional.mse_loss
        preds, preds_tr = [], []
        for seed in range(seeds):
            torch.manual_seed(C.SEED + seed)
            net = nn.Sequential(nn.Dropout(drop), nn.Linear(Xt.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
            best, best_state, bad = np.inf, None, 0
            for ep in range(max_epochs):
                net.train()
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(net(Xt[itr])[:, 0], yt[itr])
                loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    v = loss_fn(net(Xt[iva])[:, 0], yt[iva]).item()
                if v < best - 1e-5:
                    best, best_state, bad = v, {k: w.clone() for k, w in net.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
            net.load_state_dict(best_state); net.eval()
            with torch.no_grad():
                preds.append(net(Xe)[:, 0].cpu().numpy()); preds_tr.append(net(Xt)[:, 0].cpu().numpy())
        p, ptr = np.mean(preds, 0), np.mean(preds_tr, 0)
        if t == "sex":
            out[:, j] = 1 / (1 + np.exp(-p))
        else:
            shift = np.median(y - (ptr * sd + mu))                         # aim at the median, like every candidate
            out[:, j] = p * sd + mu + shift
    return out


class BlockScalerProxy(C.BlockScaler):
    pass


def mlp_aug_fit_predict(Ftr, Ytr, Fte, sex_mask_tr, pids_tr, sub=8, seeds=3, hidden=64, drop=0.3, lr=3e-3, wd=1e-3,
                        max_epochs=400, patience=30):
    """Face_AI's task heads with AUGMENTATION: in every epoch each training video is represented by the mean of a
    random subset of `sub` of its 24 crops (a different subset each time); validation and prediction use all 24.
    Early stopping on 15% of the training people, 3 seeds averaged; age, BMI (log) and sex only."""
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = Ftr.mean((0, 1)), Ftr.std((0, 1)) + 1e-6
    Ft = torch.tensor((Ftr - mu) / sd, dtype=torch.float32, device=dev)
    Fe = torch.tensor(((Fte - mu) / sd).mean(1), dtype=torch.float32, device=dev)
    Fall = Ft.mean(1)
    out = np.tile(np.median(Ytr, 0), (len(Fte), 1))
    rng = np.random.default_rng(C.SEED)
    people = np.unique(pids_tr)
    va_p = set(rng.choice(people, size=max(len(people) * 15 // 100, 1), replace=False))
    is_va = np.array([p in va_p for p in pids_tr])
    for t in APPEARANCE:
        j = C.ALL_TARGETS.index(t)
        m_ok = sex_mask_tr if t == "sex" else np.ones(len(Ytr), bool)
        y = Ytr[:, j].astype(np.float32)
        tr_m, va_m = m_ok & ~is_va, m_ok & is_va
        mu_y, sd_y = (0.0, 1.0) if t == "sex" else (y[tr_m].mean(), y[tr_m].std() + 1e-9)
        yt = torch.tensor((y - mu_y) / sd_y, dtype=torch.float32, device=dev)
        itr, iva = torch.tensor(np.flatnonzero(tr_m), device=dev), torch.tensor(np.flatnonzero(va_m), device=dev)
        loss_fn = nn.functional.binary_cross_entropy_with_logits if t == "sex" else nn.functional.mse_loss
        preds, preds_tr = [], []
        for seed in range(seeds):
            torch.manual_seed(C.SEED + seed)
            net = nn.Sequential(nn.Dropout(drop), nn.Linear(Ft.shape[2], hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
            best, best_state, bad = np.inf, None, 0
            for ep in range(max_epochs):
                pick = torch.argsort(torch.rand(len(itr), Ft.shape[1], device=dev), 1)[:, :sub]
                x = torch.gather(Ft[itr], 1, pick[:, :, None].expand(-1, -1, Ft.shape[2])).mean(1)
                net.train()
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(net(x)[:, 0], yt[itr])
                loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    v = loss_fn(net(Fall[iva])[:, 0], yt[iva]).item()
                if v < best - 1e-5:
                    best, best_state, bad = v, {k: w.clone() for k, w in net.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
            net.load_state_dict(best_state); net.eval()
            with torch.no_grad():
                preds.append(net(Fe)[:, 0].cpu().numpy()); preds_tr.append(net(Fall)[:, 0].cpu().numpy())
        p, ptr = np.mean(preds, 0), np.mean(preds_tr, 0)
        if t == "sex":
            out[:, j] = 1 / (1 + np.exp(-p))
        else:
            out[:, j] = p * sd_y + mu_y + np.median(y - (ptr * sd_y + mu_y))
    return out


def run_mlp_route(route, X, sizes, repeat, outer=range(C.N_FOLDS), fold=None):
    """Nested CV with two candidates per target: the MLP heads and the median guess (inner folds decide).
    X 2-D: per-video numbers (plain heads). X 3-D (videos x 24 crops x D): augmented heads."""
    fold = C.session_folds(repeat) if fold is None else fold
    if X.ndim == 3:
        fit = lambda Xa, Ya, Xb, sm, pa: mlp_aug_fit_predict(Xa, Ya, Xb, sm, pa)
    else:
        fit = lambda Xa, Ya, Xb, sm, pa: mlp_heads_fit_predict(Xa, Ya, Xb, sizes, sm, pa)
    oof = np.full((len(C.df), len(C.ALL_TARGETS)), np.nan)
    inner_best, log = {}, []
    for k in outer:
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        Ytr = C.Y_fit[tr].copy()
        sex_ok = np.isfinite(Ytr[:, -1])
        Ytr[~sex_ok, -1] = np.nanmean(Ytr[:, -1])
        inner, const_inner = np.zeros((len(tr), len(C.ALL_TARGETS))), np.zeros((len(tr), len(C.ALL_TARGETS)))
        for va_mask in C.inner_splits(C.pids[tr], C.SEED + 1000 * repeat + 10 * k):
            itr = tr[~va_mask]
            inner[va_mask] = C.back_transform(fit(X[itr], Ytr[~va_mask], X[tr[va_mask]], np.isfinite(C.Y_fit[itr, -1]), C.pids[itr]))
            cst = np.median(Ytr[~va_mask], 0)
            cst[-1] = Ytr[~va_mask][np.isfinite(C.Y_fit[itr, -1]), -1].mean()
            const_inner[va_mask] = C.back_transform(np.tile(cst, (va_mask.sum(), 1)))
        final = C.back_transform(fit(X[tr], Ytr, X[te], sex_ok, C.pids[tr]))
        cst = np.median(Ytr, 0)
        cst[-1] = Ytr[sex_ok, -1].mean()
        const_final = C.back_transform(np.tile(cst, (len(te), 1)))
        for j, t in enumerate(C.ALL_TARGETS):
            s_net = C.score(inner[:, j], C.Y_nat[tr, j], C.pids[tr], t)
            s_const = C.score(const_inner[:, j], C.Y_nat[tr, j], C.pids[tr], t)
            use = t in APPEARANCE and s_net < s_const
            oof[te, j] = (final if use else const_final)[:, j]
            inner_best[(k, t)] = (inner if use else const_inner)[:, j]
            log.append({"repeat": repeat, "fold": k, "route": route, "target": t, "config": "mlp_heads" if use else "const",
                        "inner_score": s_net if use else s_const})
    return oof, pd.DataFrame(log), inner_best


MLP_ROUTES = {"mlp_appearance": ["dino", "clip"], "mlp_clip_l14_tight": ["app:clip_l14_tight"],
              "mlp_clip_l14_wide": ["app:clip_l14_wide"],
              "mlp_app_new": ["app:clip_l14_tight", "app:dinov2_b_tight", "app:mefem_b_wide"]}
# Face_AI's heads with crop-subset augmentation (fixed before their results were seen: every encoder's tight crop + CLIP-L wide)
MLP_AUG_ROUTES = {f"mlpaug_{e}": e for e in ("dinov2_s_tight", "clip_b16_tight", "clip_l14_tight", "dinov2_b_tight",
                                              "mefem_b_tight", "clip_l14_wide")}


# ------------------------------------------------------------------ route cache
def route_path(route, repeat):
    return ROUTES_DIR / f"{route.replace(':', '-').replace('+', '_')}_r{repeat}.npz"


def save_route(route, repeat, oof, log, inner_best):
    keys = sorted(inner_best)
    np.savez_compressed(route_path(route, repeat), oof=oof, log=log.to_json(), inner_keys=np.array([f"{k}|{t}" for k, t in keys]),
                        **{f"inner_{i}": inner_best[key] for i, key in enumerate(keys)})


def load_route(route, repeat):
    z = np.load(route_path(route, repeat), allow_pickle=False)
    keys = [tuple(s.split("|")) for s in z["inner_keys"]]
    inner = {(int(k), t): z[f"inner_{i}"] for i, (k, t) in enumerate(keys)}
    return z["oof"], pd.read_json(io.StringIO(str(z["log"]))), inner


def compute_route(route, repeat):
    if route_path(route, repeat).exists():
        return
    t0 = time.time()
    fold = C.session_folds(repeat)
    if route.startswith("cropnet_"):                                  # cropnet_<encoder>_<crop>
        oof, log, inner = run_crop_route(route, crops(route[8:]).astype(np.float32), repeat)
    elif route in MLP_ROUTES:
        mats = [block(b) for b in MLP_ROUTES[route]]
        oof, log, inner = run_mlp_route(route, np.column_stack(mats), [m.shape[1] for m in mats], repeat)
    elif route in MLP_AUG_ROUTES:
        F = crops(MLP_AUG_ROUTES[route]).astype(np.float32)
        oof, log, inner = run_mlp_route(route, F, [F.shape[2]], repeat)
    else:
        names, hgb = ROUTES[route]
        mats = [block(b) for b in names]
        X, sizes = np.column_stack(mats), [m.shape[1] for m in mats]
        oof, log, inner, _ = C.run_route(route, X, sizes, fold, repeat, use_hgb=hgb)
    save_route(route, repeat, oof, log, inner)
    print(f"  route {route:28s} shuffle {repeat}: {time.time() - t0:5.0f}s", flush=True)


# ------------------------------------------------------------------ stacking with predicted age/sex/BMI (as Step 7.4)
def demo_blocks(stage1_routes, repeat):
    fold = C.session_folds(repeat)
    L = pd.concat([load_route(r, repeat)[1] for r in stage1_routes], ignore_index=True)
    best = C.select_tier(L, stage1_routes, repeat)
    out = {}
    for k in range(C.N_FOLDS):
        D = np.zeros((len(C.df), 3))
        for c, t in enumerate(["age", "bmi", "sex"]):
            rt = best[(best["fold"] == k) & (best["target"] == t)]["route"].iloc[0]
            oof, _, inner = load_route(rt, repeat)
            D[fold != k, c] = inner[(k, t)]
            D[fold == k, c] = oof[fold == k, C.ALL_TARGETS.index(t)]
        out[k] = D
    return out


def compute_chain(name, base_blocks, stage1_routes, repeat):
    """A route whose inputs include the honest predicted age/BMI/sex (stage 1 = best of stage1_routes)."""
    if route_path(name, repeat).exists():
        return
    t0 = time.time()
    fold = C.session_folds(repeat)
    demo = demo_blocks(stage1_routes, repeat)
    oof = np.full((len(C.df), len(C.ALL_TARGETS)), np.nan)
    logs, inner_all = [], {}
    mats = [block(b) for b in base_blocks]
    for k in range(C.N_FOLDS):
        X = np.column_stack(mats + [demo[k]])
        sizes = [m.shape[1] for m in mats] + [3]
        o, lg, ib, _ = C.run_route(name, X, sizes, fold, repeat, outer=[k])
        oof[fold == k] = o[fold == k]
        logs.append(lg); inner_all.update(ib)
    save_route(name, repeat, oof, pd.concat(logs, ignore_index=True), inner_all)
    print(f"  chain {name:28s} shuffle {repeat}: {time.time() - t0:5.0f}s", flush=True)


# ------------------------------------------------------------------ tiers: single best route, or an inner-chosen ensemble
def _inner_score(log, target, fold=None):
    """The inner (selection) score a route's chosen candidate got for one target (and outer fold)."""
    m = (log["target"] == target) if fold is None else ((log["target"] == target) & (log["fold"] == fold))
    return float(log.loc[m, "inner_score"].iloc[0])


def assemble_tier(routes, repeat, ensemble=True, max_m=3):
    """Per (outer fold, target): rank routes by inner score; candidates are the best route alone and the mean of the
    best 2..max_m routes; the candidate with the lowest inner score (same people, same rule) is used."""
    fold = C.session_folds(repeat)
    data = {r: load_route(r, repeat) for r in routes}
    out = np.full((len(C.df), len(C.ALL_TARGETS)), np.nan)
    rows = []
    for k in range(C.N_FOLDS):
        tr = np.where(fold != k)[0]
        for j, t in enumerate(C.ALL_TARGETS):
            sc = sorted([(_inner_score(data[r][1], t, k), r) for r in routes])
            options = {1: [sc[0][1]]}
            if ensemble:
                for m in range(2, min(max_m, len(sc)) + 1):
                    options[m] = [r for _, r in sc[:m]]
            best_m, best_s = 1, sc[0][0]
            for m, rs in options.items():
                if m == 1:
                    continue
                pred = np.mean([data[r][2][(k, t)] for r in rs], 0)
                s = C.score(pred, C.Y_nat[tr, j], C.pids[tr], t)
                if s < best_s:
                    best_m, best_s = m, s
            rs = options[best_m]
            out[fold == k, j] = np.mean([data[r][0][fold == k, j] for r in rs], 0)
            rows.append({"repeat": repeat, "fold": k, "target": t, "routes": "+".join(rs), "m": best_m, "inner_score": best_s})
    return out, pd.DataFrame(rows)


def metrics_table(tiers):
    """Per-person MAE, skill and paired 95% CI vs the notebook's tiers, averaged over the 3 shuffles."""
    ref = {n: {r: C.oof_matrix(n, r) for r in range(3)} for n in ("video_specialists", "multimodal_specialists")}
    rows = []
    for name, P in tiers.items():
        s = C.quick_scores(P)
        for t in C.ALL_TARGETS:
            row = {"tier": name, "target": t}
            if t == "sex":
                row["acc"] = s[t]
            else:
                row["mae"], row["skill"] = s[t]
                base = "multimodal_specialists" if "multimodal" in name else "video_specialists"
                d, lo, hi = C.paired_ci(P, ref[base], t)
                row.update({"vs_old": d, "lo": lo, "hi": hi})
            rows.append(row)
    return pd.DataFrame(rows)


VIDEO_V1 = ["dino", "clip", "appearance", "temporal", "video_all"]
PHYSIO_V1 = ["ppg", "ecg", "physio", "video+physio"]

# ------------------------------------------------------------------ fixed split: Face_AI's 510 learning / 90 test people
FIXED_DIR = V2 / "routes_fixed"
OLD = C.PROJECT.parent / "Face_AI"


def fixed_fold():
    split = pd.read_csv(OLD / "participant_split.csv", dtype={"patient_id": str})
    role = C.df["patient_id"].map(split.set_index("patient_id")["split"]).to_numpy()
    return np.where(role == "test", 0, 1)


def compute_route_fixed(route):
    """Step 10.12a for one route: the inner loop chooses on the 510 people only; outer fold 0 = the 90 test people."""
    path = FIXED_DIR / f"{route.replace(':', '-').replace('+', '_')}.npz"
    if path.exists():
        return
    FIXED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fold = fixed_fold()
    if route.startswith("cropnet_"):
        F = crops(route[8:]).astype(np.float32)
        # run_crop_route works on session_folds; emulate the fixed split by temporarily replacing it
        saved = C.session_folds
        C.session_folds = lambda r: fold
        try:
            oof, log, inner = run_crop_route(route, F, 0, outer=[0])
        finally:
            C.session_folds = saved
    elif route in MLP_ROUTES:
        mats = [block(b) for b in MLP_ROUTES[route]]
        oof, log, inner = run_mlp_route(route, np.column_stack(mats), [m.shape[1] for m in mats], 0, outer=[0], fold=fold)
    elif route in MLP_AUG_ROUTES:
        F = crops(MLP_AUG_ROUTES[route]).astype(np.float32)
        oof, log, inner = run_mlp_route(route, F, [F.shape[2]], 0, outer=[0], fold=fold)
    else:
        names, hgb = ROUTES[route]
        mats = [block(b) for b in names]
        X, sizes = np.column_stack(mats), [m.shape[1] for m in mats]
        oof, log, inner, _ = C.run_route(route, X, sizes, fold, 0, use_hgb=hgb, outer=[0])
    keys = sorted(inner)
    np.savez_compressed(path, oof=oof, log=log.to_json(), inner_keys=np.array([f"{k}|{t}" for k, t in keys]),
                        **{f"inner_{i}": inner[key] for i, key in enumerate(keys)})
    print(f"  fixed route {route:28s}: {time.time() - t0:5.0f}s", flush=True)


def compute_chain_fixed(name, base_blocks, stage1_routes):
    """A stacking route on the fixed split: predicted age/BMI/sex from the best stage-1 route (chosen on the 510)."""
    path = FIXED_DIR / f"{name.replace(':', '-').replace('+', '_')}.npz"
    if path.exists():
        return
    t0 = time.time()
    fold = fixed_fold()
    L = pd.concat([load_route_fixed(r)[1] for r in stage1_routes], ignore_index=True)
    best = C.select_tier(L.assign(repeat=0), stage1_routes, 0)
    D = np.zeros((len(C.df), 3))
    for c, t in enumerate(["age", "bmi", "sex"]):
        rt = best[best["target"] == t]["route"].iloc[0]
        oof, _, inner = load_route_fixed(rt)
        D[fold == 1, c] = inner[(0, t)]
        D[fold == 0, c] = oof[fold == 0, C.ALL_TARGETS.index(t)]
    mats = [block(b) for b in base_blocks]
    X, sizes = np.column_stack(mats + [D]), [m.shape[1] for m in mats] + [3]
    oof, log, inner, _ = C.run_route(name, X, sizes, fold, 0, outer=[0])
    keys = sorted(inner)
    np.savez_compressed(path, oof=oof, log=log.to_json(), inner_keys=np.array([f"{k}|{t}" for k, t in keys]),
                        **{f"inner_{i}": inner[key] for i, key in enumerate(keys)})
    print(f"  fixed chain {name:28s}: {time.time() - t0:5.0f}s", flush=True)


def load_route_fixed(route):
    z = np.load(FIXED_DIR / f"{route.replace(':', '-').replace('+', '_')}.npz", allow_pickle=False)
    keys = [tuple(s.split("|")) for s in z["inner_keys"]]
    return z["oof"], pd.read_json(io.StringIO(str(z["log"]))), {(int(k), t): z[f"inner_{i}"] for i, (k, t) in enumerate(keys)}


def head_to_head(routes, ensemble=True, max_m=3):
    """v2 specialists (chosen on the 510) vs Face_AI's newest model on its 90 test people, paired person bootstrap."""
    fold = fixed_fold()
    te, tr = np.where(fold == 0)[0], np.where(fold == 1)[0]
    data = {r: load_route_fixed(r) for r in routes}
    P = np.full((len(C.df), len(C.ALL_TARGETS)), np.nan)
    choice = []
    for j, t in enumerate(C.ALL_TARGETS):
        sc = sorted([(_inner_score(data[r][1], t), r) for r in routes])
        best_m, best_s = 1, sc[0][0]
        if ensemble:
            for m in range(2, min(max_m, len(sc)) + 1):
                rs = [r for _, r in sc[:m]]
                s = C.score(np.mean([data[r][2][(0, t)] for r in rs], 0), C.Y_nat[tr, j], C.pids[tr], t)
                if s < best_s:
                    best_m, best_s = m, s
        rs = [r for _, r in sc[:best_m]]
        P[te, j] = np.mean([data[r][0][te, j] for r in rs], 0)
        choice.append({"target": t, "routes": "+".join(rs)})
    theirs = pd.read_csv(OLD / "experiments" / "improved_cnn" / "results" / "cv" / "step13_test_predictions.csv",
                         dtype={"patient_id": str}).set_index("key")
    keys = C.df["key"].to_numpy()[te]
    pid = C.pids[te]
    boot = np.random.default_rng(C.SEED).integers(0, len(np.unique(pid)), size=(2000, len(np.unique(pid))))
    rows = []
    for j, t in enumerate(C.REG_TARGETS):
        y = C.Y_nat[te, j]
        per = lambda p: pd.Series(np.abs(np.asarray(p, float) - y)).groupby(pid).mean().to_numpy()
        e_mine, e_their = per(P[te, j]), per(theirs.loc[keys, f"pred_{t}"])
        e_old = None
        old_file = C.OOF / "fixed_split_video_specialists.csv"
        if old_file.exists():
            old = pd.read_csv(old_file, dtype={"patient_id": str}).set_index("key")
            e_old = per(old.loc[keys, f"pred_{t}"])
        e_guess = per(np.full(len(te), np.median(C.Y_nat[tr, j])))
        d = e_mine[boot].mean(1) - e_their[boot].mean(1)
        rows.append({"target": t, "guess": e_guess.mean(), "face_ai_newest": e_their.mean(),
                     "faceai2_v1": e_old.mean() if e_old is not None else np.nan, "faceai2_v2": e_mine.mean(),
                     "v2_minus_faceai": e_mine.mean() - e_their.mean(), "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5)})
    ys = C.Y_nat[te, -1]
    ok = np.isfinite(ys)
    sex = {"faceai2_v2 wrong": int(((P[te, -1][ok] >= 0.5) != ys[ok]).sum()),
           "face_ai wrong": int(((theirs.loc[keys, "pred_prob_male"].to_numpy()[ok] >= 0.5) != ys[ok]).sum()), "of": int(ok.sum())}
    band = np.where(C.Y_nat[te, C.ALL_TARGETS.index("age")] <= 24, "18-24", ">24")
    ages = {}
    for name, p in (("faceai2_v2", P[te, 7]), ("face_ai_newest", theirs.loc[keys, "pred_age"].to_numpy())):
        e = pd.DataFrame({"e": np.abs(p - C.Y_nat[te, 7]), "pid": pid, "band": band})
        ages[name] = e.groupby(["band", "pid"])["e"].mean().groupby("band").mean().to_dict()
    return pd.DataFrame(rows), sex, ages, pd.DataFrame(choice), P


def main():
    add_dynamic_routes()
    V2.mkdir(parents=True, exist_ok=True)
    ROUTES_DIR.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "run":
        for route in sys.argv[2:]:
            for r in range(C.N_REPEATS):
                compute_route(route, r)
    elif sys.argv[1] == "chain":                                      # chain <name> <stage1 routes,> <base blocks,>
        name, s1, bb = sys.argv[2], sys.argv[3].split(","), sys.argv[4].split(",")
        for r in range(C.N_REPEATS):
            compute_chain(name, bb, s1, r)
    elif sys.argv[1] == "fixed":                                      # fixed <route> ...
        for route in sys.argv[2:]:
            compute_route_fixed(route)
    elif sys.argv[1] == "chainfixed":                                 # chainfixed <name> <stage1 routes,> <base blocks,>
        compute_chain_fixed(sys.argv[2], sys.argv[4].split(","), sys.argv[3].split(","))
    elif sys.argv[1] == "h2h":                                        # h2h <route,route,...>
        routes = sys.argv[2].split(",")
        for r in routes:
            compute_route_fixed(r)
        tab, sex, ages, choice, P = head_to_head(routes)
        pd.set_option("display.width", 220, "display.max_columns", 20)
        print(tab.round(4).to_string(index=False)); print(sex); print(ages); print(choice.to_string(index=False))
        tab.to_csv(V2 / "h2h.csv", index=False)
        (V2 / "h2h_extra.json").write_text(json.dumps({"sex": sex, "ages": ages, "routes": routes}, indent=1))
        choice.to_csv(V2 / "h2h_choice.csv", index=False)
        te = fixed_fold() == 0
        out = C.df.loc[te, ["key", "patient_id", "step"]].copy()
        for j, t in enumerate(C.ALL_TARGETS):
            out[f"pred_{t}"] = P[te, j]
        out.to_csv(C.OOF / "fixed_split_video_specialists_v2.csv", index=False)
    elif sys.argv[1] == "tiers":                                      # tiers <tier>=<route,route,...> ...
        tiers, sel = {}, []
        for spec in sys.argv[2:]:
            name, routes = spec.split("=")
            routes = routes.split(",")
            for ens in (False, True):
                tname = name + ("_ens" if ens else "")
                P = {}
                for r in range(C.N_REPEATS):
                    P[r], s = assemble_tier(routes, r, ensemble=ens)
                    sel.append(s.assign(tier=tname))
                tiers[tname] = P
        tab = metrics_table(tiers)
        pd.set_option("display.width", 220, "display.max_columns", 20)
        print(tab.round(4).to_string(index=False))
        tab.to_csv(V2 / "tiers_metrics.csv", index=False)
        pd.concat(sel).to_csv(V2 / "tiers_selection.csv", index=False)
        for name, P in tiers.items():
            np.savez_compressed(V2 / f"tier_{name}.npz", **{f"r{r}": P[r] for r in P})
            if name.endswith("_ens"):                                   # also as OOF files, like the v1 tiers
                stem = {"video_v2_ens": "video_specialists_v2", "multimodal_v2_ens": "multimodal_specialists_v2"}.get(name)
                if stem:
                    for r in P:
                        C.save_oof(stem, P[r], r)


if __name__ == "__main__":
    main()
