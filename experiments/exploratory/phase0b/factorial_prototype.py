"""MISDIVE Phase 0B: two-factor (algorithm x feature view) attribution of coincident misses.

Read-only use of existing Phase 0 .NET artefacts (SENTRA project) and the supplementary rescoring
(SENTRA_AEiC2027). Detectors: MLP structural/content branches (E04) and LightGBM structural/content
detectors (E10), seeds 42/123/2026, one shared training cohort. Branch thresholds: floor(0.005*N0)-th
largest validation benign score (the frozen SENTRA rule), computed on weeks 41-46 only. No training,
no tuning on evaluation data.

Windows: V  = weeks 41-46 (validation; in-sample for thresholds and checkpoint selection)
         W1 = weeks 47-52 (Phase 0 dev test; previously examined)
         W2 = weeks 53-64 (official test; previously examined)
Every result is EXPLORATORY.

Balanced contrasts (seed pairs i != j; each branch appears equally in both arms, so marginal miss rates
cancel exactly):
  view contrast for algorithms (a, b):  D_view(a,b) = 1/2 Cov(m_aS_i - m_aC_i, m_bS_j - m_bC_j)
  algorithm contrast for views (v, w):  D_alg(v,w)  = 1/2 Cov(m_Mv_i - m_Lv_i, m_Mw_j - m_Lw_j)
"""
from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path
import os

import numpy as np
import pandas as pd

SENTRA = Path(os.environ.get("MISDIVE_PHASE0_PROJECT", "MISDIVE_PHASE0_PROJECT_not_set"))
REPO = Path(os.environ.get("MISDIVE_CLEAN_RUN", Path(__file__).resolve().parents[2] / "single_factor"))
OUT = Path(__file__).resolve().parent
SEEDS = (42, 123, 2026)
ALPHA = 0.005
B = 1000
RNG_SEED = 20261126


def at_fpr(benign, alpha):  # identical to viewdep.thresholds.at_fpr
    s = np.sort(np.asarray(benign, dtype=np.float64))[::-1]
    return float(s[int(np.floor(alpha * len(s)))])


def load():
    runs, pred = SENTRA / "experiments" / "runs", SENTRA / "results" / "predictions"
    yv = np.load(SENTRA / "experiments" / "y_val.npy").astype(int)
    val, w1 = {}, {}
    for s in SEEDS:
        v = np.load(runs / f"E04_indep_semantic_s{s}" / "val_logits.npy").astype(np.float64)
        val[f"MS_{s}"], val[f"MC_{s}"] = v[:, 0], v[:, 1]
        val[f"LS_{s}"] = np.load(runs / f"E10_lgbm_structural_s{s}" / "val_logits.npy")[:, 0].astype(np.float64)
        val[f"LC_{s}"] = np.load(runs / f"E10_lgbm_content_s{s}" / "val_logits.npy")[:, 0].astype(np.float64)
        d = np.load(pred / f"devtest_E04_indep_semantic_s{s}.npy").astype(np.float64)
        w1[f"MS_{s}"], w1[f"MC_{s}"] = d[:, 0], d[:, 1]
        w1[f"LS_{s}"] = np.load(pred / f"devtest_E10_lgbm_structural_s{s}.npy")[:, 0].astype(np.float64)
        w1[f"LC_{s}"] = np.load(pred / f"devtest_E10_lgbm_content_s{s}.npy")[:, 0].astype(np.float64)
    ds = pd.read_parquet(pred / "devtest_samples.parquet")
    off = pd.read_parquet(REPO / "artifacts" / "supplementary" / "phase0_dotnet_scores_w53_64.parquet")
    w2 = {}
    for s in SEEDS:
        w2[f"MS_{s}"], w2[f"MC_{s}"] = off[f"sem_S_{s}"].values, off[f"sem_C_{s}"].values
        w2[f"LS_{s}"], w2[f"LC_{s}"] = off[f"lgbS_{s}"].values, off[f"lgbC_{s}"].values
    thr = {k: at_fpr(v[yv == 0], ALPHA) for k, v in val.items()}
    def dec(sc):
        return {k: v > thr[k] for k, v in sc.items()}
    return {
        "V weeks 41-46 (validation)": (dec(val), yv, None),
        "W1 weeks 47-52 (dev test, examined)": (dec(w1), ds.label.values, ds.family.values),
        "W2 weeks 53-64 (official test, examined)": (dec(w2), off.label.values, off.family.values),
    }


def contrasts(miss: dict, w: np.ndarray) -> dict:
    W = w.sum()
    mean = lambda x: (w @ x) / W  # noqa: E731
    cov = lambda x, y: mean(x * y) - mean(x) * mean(y)  # noqa: E731
    sp = [(i, j) for i, j in permutations(SEEDS, 2) if i < j]
    spo = list(permutations(SEEDS, 2))
    out = {}
    for a, b in (("M", "M"), ("L", "L"), ("M", "L")):
        pairs = sp if a == b else spo
        out[f"view|{a}{b}"] = np.mean([0.5 * cov(miss[f"{a}S_{i}"] - miss[f"{a}C_{i}"],
                                                 miss[f"{b}S_{j}"] - miss[f"{b}C_{j}"]) for i, j in pairs])
    for v, u in (("S", "S"), ("C", "C"), ("S", "C")):
        pairs = sp if v == u else spo
        out[f"alg|{v}{u}"] = np.mean([0.5 * cov(miss[f"M{v}_{i}"] - miss[f"L{v}_{i}"],
                                                miss[f"M{u}_{j}"] - miss[f"L{u}_{j}"]) for i, j in pairs])
    # Descriptive sharing ladder: mean covariance and J/(q1 q2) by what the two detectors share.
    lad = {}
    for x, y in permutations([f"{t}_{s}" for t in ("MS", "MC", "LS", "LC") for s in SEEDS], 2):
        if x.split("_")[1] == y.split("_")[1] or x > y:
            continue
        cat = ("same-view" if x[1] == y[1] else "cross-view") + "|" + ("same-alg" if x[0] == y[0] else "cross-alg")
        q1, q2 = mean(miss[x]), mean(miss[y])
        lad.setdefault(cat, []).append((cov(miss[x], miss[y]), mean(miss[x] * miss[y]) / (q1 * q2), mean(miss[x] * miss[y])))
    for cat, v in lad.items():
        a = np.array(v)
        out[f"ladder_cov|{cat}"], out[f"ladder_ratio|{cat}"], out[f"ladder_J|{cat}"] = a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()
    return out


def main():
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    q_rows = []
    for win, (dec, y, fam) in load().items():
        mal = y == 1
        miss = {k: (~v[mal]).astype(np.float64) for k, v in dec.items()}
        fpr = {k: v[~mal].mean() for k, v in dec.items()}
        for k in miss:
            q_rows.append(dict(window=win, detector=k, fnr_pct=100 * miss[k].mean(), fpr_pct=100 * fpr[k]))
        point = contrasts(miss, np.ones(mal.sum()))
        if fam is not None:
            f = fam[mal]
            f = np.where(pd.isna(f), np.array([f"_s{i}" for i in range(len(f))], dtype=object), f)
            _, inv = np.unique(f.astype(str), return_inverse=True)
            n = inv.max() + 1
            reps = [contrasts(miss, np.bincount(rng.integers(0, n, n), minlength=n)[inv].astype(float)) for _ in range(B)]
        for k, v in point.items():
            r = dict(window=win, quantity=k, value_pp=100 * v if not k.startswith("ladder_ratio") else v)
            if fam is not None:
                arr = np.array([x[k] for x in reps]) * (1 if k.startswith("ladder_ratio") else 100)
                r["lo"], r["hi"] = np.percentile(arr, [2.5, 97.5])
            rows.append(r)
    R = pd.DataFrame(rows)
    Q = pd.DataFrame(q_rows)
    R.to_csv(OUT / "factorial_contrasts.csv", index=False)
    Q.to_csv(OUT / "factorial_detectors.csv", index=False)
    Q["type"] = Q.detector.str[:2]
    print(Q.groupby(["window", "type"])[["fnr_pct", "fpr_pct"]].mean().round(2).to_string())
    pd.set_option("display.width", 200)
    print(R.round(3).to_string())


if __name__ == "__main__":
    main()
