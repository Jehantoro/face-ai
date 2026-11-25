"""Shared machinery for the v2 experiments: the SAME labels, folds, candidate models, nested selection and metrics as
FaceAI_pipeline.ipynb (Steps 1, 2, 7 and 9), as importable functions, so every new idea is scored on exactly the same
people, folds and rules as the saved results.

Nothing here changes the notebook's saved files unless a caller saves under a new name.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DB_CSV = PROJECT / "data" / "db.csv"
FEATURES, RESULTS = PROJECT / "features", PROJECT / "results"
OOF, MET = RESULTS / "oof", RESULTS / "metrics"
CACHE2 = PROJECT / "data" / "cache_v2"

REG_TARGETS = ["saturation", "temperature", "glycated_hemoglobin", "cholesterol",
               "respiratory", "rigidity", "stress", "age", "bmi"]
ALL_TARGETS = REG_TARGETS + ["sex"]
NAME = {"saturation": "SpO2 (%)", "temperature": "Temperature (°C)", "glycated_hemoglobin": "HbA1c (%)",
        "cholesterol": "Cholesterol (mmol/L)", "respiratory": "Breathing rate (/min)", "rigidity": "Arterial stiffness",
        "stress": "Stress (PSM-25)", "age": "Age (years)", "bmi": "BMI (kg/m²)", "sex": "Sex"}
LOG_TARGETS = ["age", "bmi"]
N_FOLDS, N_REPEATS, SEED = 5, 3, 20261013
RIDGE_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
PLS_KS = [1, 2, 4, 8, 16]
KRR_LAMBDAS = [0.1, 1.0, 10.0, 100.0]
INNER_FOLDS, CLIP_Z = 4, 6.0

# ------------------------------------------------------------------ labels and folds (Steps 1-2)
df = pd.read_csv(DB_CSV, dtype={"patient_id": str})
df = df[df["camera"] == "FullHDwebcam"].copy()
df["key"] = df["video"].map(lambda v: Path(v).stem)
df = df.sort_values("key").reset_index(drop=True)
df["sex_label"] = df["sex"].map({"F": 0.0, "M": 1.0})
_mixed = df.groupby("patient_id")["sex"].transform("nunique") > 1
df.loc[_mixed, "sex_label"] = np.nan
df["after"] = (df["step"] == "after").astype(float)
pids = df["patient_id"].to_numpy()
SESS = (df["patient_id"] + "_" + df["step"]).tolist()
folds = pd.read_csv(RESULTS / "folds.csv", dtype={"patient_id": str})


def session_folds(repeat):
    f = folds[folds["repeat"] == repeat].set_index("patient_id")["fold"]
    return df["patient_id"].map(f).to_numpy()


Y_nat = np.column_stack([df[REG_TARGETS].to_numpy(float), df["sex_label"].to_numpy(float)])


def target_matrix():
    Y = Y_nat.copy()
    for j, t in enumerate(ALL_TARGETS):
        if t in LOG_TARGETS:
            Y[:, j] = np.log(Y[:, j])
    return Y


Y_fit = target_matrix()


def back_transform(P):
    P = P.copy()
    for j, t in enumerate(ALL_TARGETS):
        if t in LOG_TARGETS:
            P[:, j] = np.exp(P[:, j])
        if t == "sex":
            P[:, j] = np.clip(P[:, j], 0.01, 0.99)
    return P


# ------------------------------------------------------------------ feature files
def load_features(path, keys):
    with np.load(path, allow_pickle=False) as z:
        pos = {str(k): i for i, k in enumerate(z["keys"])}
        return z["features"][[pos[str(k)] for k in keys]].astype(np.float64)


def feature_names(path):
    with np.load(path, allow_pickle=False) as z:
        return list(z["names"]) if "names" in z.files else None


def save_features(path, keys, X, names=None):
    extra = {"names": np.asarray(names, dtype=str)} if names is not None else {}
    np.savez_compressed(path, keys=np.asarray(keys, dtype=str), features=np.asarray(X, np.float32), **extra)


def base_blocks():
    """The notebook's five feature blocks, in df order."""
    return {"dino": load_features(FEATURES / "dino_vits14.npz", df["key"]),
            "clip": load_features(FEATURES / "clip_vitb16.npz", df["key"]),
            "temporal": load_features(FEATURES / "rppg_v2.npz", df["key"]),
            "ppg": load_features(FEATURES / "ppg.npz", SESS),
            "ecg": load_features(FEATURES / "ecg.npz", SESS)}


# ------------------------------------------------------------------ candidate models (Step 7.1, unchanged)
class BlockScaler:
    def __init__(self, sizes):
        self.sizes = sizes

    def fit(self, X):
        med = np.nanmedian(X, 0)
        self.med = np.where(np.isfinite(med), med, 0.0)
        Xi = np.where(np.isnan(X), self.med, X)
        self.mu, sd = Xi.mean(0), Xi.std(0)
        self.keep = sd > 1e-9
        self.sd = np.where(self.keep, sd, 1.0)
        self.w, start = np.ones(X.shape[1]), 0
        for size in self.sizes:
            sl = slice(start, start + size)
            self.w[sl] = 1.0 / np.sqrt(max(int(self.keep[sl].sum()), 1))
            start += size
        return self

    def transform(self, X):
        Xi = np.where(np.isnan(X), self.med, X)
        return (np.clip((Xi - self.mu) / self.sd, -CLIP_Z, CLIP_Z) * self.w)[:, self.keep]


def ridge_all(Ztr, Ytr, Zte):
    zm, ym = Ztr.mean(0), Ytr.mean(0)
    U, s, Vt = np.linalg.svd(Ztr - zm, full_matrices=False)
    UtY, Zc = U.T @ (Ytr - ym), Zte - zm
    return {f"ridge_a{a:g}": Zc @ (Vt.T @ ((s / (s ** 2 + a * len(Ztr)))[:, None] * UtY)) + ym for a in RIDGE_ALPHAS}


def pls_all(Ztr, Ytr, Zte):
    zm = Ztr.mean(0)
    X0, Xt = Ztr - zm, Zte - zm
    kmax = min(max(PLS_KS), X0.shape[1])
    out = {f"pls_k{k}": np.zeros((len(Zte), Ytr.shape[1])) for k in PLS_KS}
    for j in range(Ytr.shape[1]):
        ym = Ytr[:, j].mean()
        yc, X, W, P_, Q = Ytr[:, j] - ym, X0.copy(), [], [], []
        for _ in range(kmax):
            w = X.T @ yc
            nw = np.linalg.norm(w)
            if nw < 1e-12:
                break
            w /= nw
            t = X @ w
            tt = t @ t
            p, q = X.T @ t / tt, (yc @ t) / tt
            X -= np.outer(t, p)
            yc = yc - q * t
            W.append(w); P_.append(p); Q.append(q)
        W, P_, Q = np.array(W).T, np.array(P_).T, np.array(Q)
        for k in PLS_KS:
            kk = min(k, W.shape[1])
            out[f"pls_k{k}"][:, j] = Xt @ (W[:, :kk] @ np.linalg.solve(P_[:, :kk].T @ W[:, :kk], Q[:kk])) + ym
    return out


def krr_all(Ztr, Ytr, Zte):
    gamma = 1.0 / (2.0 * Ztr.var(0).sum() + 1e-12)
    sq = lambda A, B: np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0)
    K, Kt, ym = np.exp(-gamma * sq(Ztr, Ztr)), np.exp(-gamma * sq(Zte, Ztr)), Ytr.mean(0)
    lam, Qm = np.linalg.eigh(K)
    QtY = Qm.T @ (Ytr - ym)
    return {f"krr_l{l:g}": Kt @ (Qm @ (QtY / (lam + l)[:, None])) + ym for l in KRR_LAMBDAS}


def hgb_all(Xtr, Ytr, Xte, sex_mask_tr, targets=None):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    pred = np.zeros((len(Xte), Ytr.shape[1]))
    kw = dict(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=20, l2_regularization=1.0,
              early_stopping=False, random_state=0)
    for j, t in enumerate(ALL_TARGETS):
        if targets is not None and t not in targets:
            pred[:, j] = np.median(Ytr[:, j])
            continue
        if t == "sex":
            m = sex_mask_tr
            pred[:, j] = HistGradientBoostingClassifier(**kw).fit(Xtr[m], Ytr[m, j].round().astype(int)).predict_proba(Xte)[:, 1]
        else:
            pred[:, j] = HistGradientBoostingRegressor(loss="absolute_error", **kw).fit(Xtr, Ytr[:, j]).predict(Xte)
    return {"hgb": pred}


def fit_predict_all(Xtr, Ytr, Xte, sizes, use_hgb, sex_mask_tr, extra_fns=(), hgb_targets=None):
    """Every candidate's predictions (the notebook's set, plus optional extra candidate functions)."""
    sc = BlockScaler(sizes).fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    both, out = np.vstack([Zte, Ztr]), {}
    for fn in (ridge_all, pls_all, krr_all):
        for cfg, P in fn(Ztr, Ytr, both).items():
            Pte, Ptr = P[:len(Zte)], P[len(Zte):]
            shift = np.median(Ytr - Ptr, axis=0)
            shift[-1] = 0.0
            out[cfg] = Pte + shift
    if use_hgb:
        out.update(hgb_all(Xtr, Ytr, Xte, sex_mask_tr, hgb_targets))
    for fn in extra_fns:                          # e.g. GPU networks: fn(Ztr, Ytr, Zte, sex_mask_tr) -> {cfg: Pte}
        out.update(fn(Ztr, Ytr, Zte, sex_mask_tr))
    const = np.median(Ytr, axis=0)
    const[-1] = Ytr[sex_mask_tr, -1].mean() if sex_mask_tr.any() else 0.5
    out["const"] = np.tile(const, (len(Xte), 1))
    return out


def inner_splits(pids_tr, seed):
    groups = np.array_split(np.random.default_rng(seed).permutation(np.unique(pids_tr)), INNER_FOLDS)
    return [np.isin(pids_tr, g) for g in groups]


def score(pred, y, p, target):
    ok = np.isfinite(y)
    err = (pred[ok] - y[ok]) ** 2 if target == "sex" else np.abs(pred[ok] - y[ok])
    return float(pd.Series(err).groupby(p[ok]).mean().mean())


def run_route(route, X, sizes, fold, repeat, use_hgb=False, outer=range(N_FOLDS), extra_fns=(), hgb_targets=None,
              keep_inner_all=False):
    """Step 7.4 for one feature matrix: choose a candidate per (outer fold, target) on the inner folds, then predict
    the outer fold. Returns OOF predictions (real units), the selection log, the honest inner predictions of the
    chosen candidate, and (optionally) every candidate's inner and outer predictions for ensembling."""
    oof = np.full((len(df), len(ALL_TARGETS)), np.nan)
    inner_best, log, store = {}, [], {}
    for k in outer:
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        Ytr = Y_fit[tr].copy()
        sex_ok = np.isfinite(Ytr[:, -1])
        Ytr[~sex_ok, -1] = np.nanmean(Ytr[:, -1])
        inner = {}
        for va_mask in inner_splits(pids[tr], SEED + 1000 * repeat + 10 * k):
            itr, iva = tr[~va_mask], tr[va_mask]
            preds = fit_predict_all(X[itr], Ytr[~va_mask], X[iva], sizes, use_hgb, np.isfinite(Y_fit[itr, -1]),
                                    extra_fns, hgb_targets)
            for cfg, P in preds.items():
                inner.setdefault(cfg, np.zeros((len(tr), len(ALL_TARGETS))))[va_mask] = back_transform(P)
        final = fit_predict_all(X[tr], Ytr, X[te], sizes, use_hgb, sex_ok, extra_fns, hgb_targets)
        final = {c: back_transform(P) for c, P in final.items()}
        for j, t in enumerate(ALL_TARGETS):
            scores = {cfg: score(P[:, j], Y_nat[tr, j], pids[tr], t) for cfg, P in inner.items()}
            best = min(scores, key=scores.get)
            oof[te, j] = final[best][:, j]
            inner_best[(k, t)] = inner[best][:, j]
            log.append({"repeat": repeat, "fold": k, "route": route, "target": t, "config": best, "inner_score": scores[best]})
        if keep_inner_all:
            store[k] = {"tr": tr, "te": te, "inner": inner, "final": final}
    return oof, pd.DataFrame(log), inner_best, store


def select_tier(logs, routes, repeat):
    lg = logs[(logs["repeat"] == repeat) & logs["route"].isin(routes)]
    return lg.loc[lg.groupby(["fold", "target"])["inner_score"].idxmin()]


def assemble(best, route_oof, fold):
    out = np.full((len(df), len(ALL_TARGETS)), np.nan)
    for row in best.itertuples():
        out[fold == row.fold, ALL_TARGETS.index(row.target)] = route_oof[row.route][fold == row.fold, ALL_TARGETS.index(row.target)]
    return out


# ------------------------------------------------------------------ OOF files and metrics (Step 9)
def save_oof(name, P, repeat):
    out = df[["key", "patient_id", "step"]].copy()
    out["repeat"], out["fold"] = repeat, session_folds(repeat)
    for j, t in enumerate(ALL_TARGETS):
        out[f"pred_{t}"] = P[:, j]
    out.to_csv(OOF / f"{name}_r{repeat}.csv", index=False)


def load_oof(name, repeat=None):
    files = [f for f in sorted(OOF.glob(f"{name}_r*.csv")) if f.stem.rsplit("_r", 1)[0] == name
             and (repeat is None or f.stem.endswith(f"_r{repeat}"))]
    return pd.concat([pd.read_csv(f, dtype={"patient_id": str}) for f in files], ignore_index=True) if files else None


def person_mae(abs_err, p):
    return float(pd.Series(np.asarray(abs_err, float)).groupby(np.asarray(p)).mean().mean())


def oof_matrix(name, repeat):
    """An OOF file as a (sessions x targets) matrix in df order."""
    p = load_oof(name, repeat).set_index("key").loc[df["key"]]
    return p[[f"pred_{t}" for t in ALL_TARGETS]].to_numpy(float)


def quick_scores(P_by_repeat, targets=ALL_TARGETS):
    """Skill per target averaged like the notebook: per-person MAE over all repeats given, vs the median guess."""
    out = {}
    for t in targets:
        j = ALL_TARGETS.index(t)
        if t == "sex":
            from sklearn.metrics import roc_auc_score
            accs = []
            for r, P in P_by_repeat.items():
                y = Y_nat[:, j]
                ok = np.isfinite(y) & np.isfinite(P[:, j])
                accs.append(np.mean((P[ok, j] >= 0.5) == y[ok]))
            out[t] = float(np.mean(accs))
            continue
        e_m, e_c = [], []
        for r, P in P_by_repeat.items():
            C = oof_matrix("constant", r)
            e_m.append(pd.Series(np.abs(P[:, j] - Y_nat[:, j])).groupby(pids).mean())
            e_c.append(pd.Series(np.abs(C[:, j] - Y_nat[:, j])).groupby(pids).mean())
        em, ec = pd.concat(e_m, axis=1).mean(1), pd.concat(e_c, axis=1).mean(1)
        out[t] = (float(em.mean()), float(1 - em.mean() / ec.mean()))
    return out


def paired_ci(P_a, P_b, t, B=2000, seed=SEED):
    """Paired person bootstrap of MAE(a) - MAE(b) over the repeats both have (dicts repeat -> matrix)."""
    j = ALL_TARGETS.index(t)
    reps = sorted(set(P_a) & set(P_b))
    ea = pd.concat([pd.Series(np.abs(P_a[r][:, j] - Y_nat[:, j])).groupby(pids).mean() for r in reps], axis=1).mean(1)
    eb = pd.concat([pd.Series(np.abs(P_b[r][:, j] - Y_nat[:, j])).groupby(pids).mean() for r in reps], axis=1).mean(1)
    d = (ea - eb).to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(d), size=(B, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def timer():
    t0 = time.time()
    return lambda: f"{time.time() - t0:.0f}s"
