"""M5: temporal and family-level diagnostics (descriptive; no inference).

* Weekly (per evaluation window) factorial effects: point estimates only.
* Burst screen (rule frozen in the study protocol): a window is flagged when the mean joint miss rate of pairs
  sharing factor 1 exceeds its median over windows by more than ``margin_pp`` while the mean joint miss rate of
  pairs not sharing factor 1 does not. The rule pools all same-level pairs and is therefore insensitive to a
  burst confined to one cell (documented negative finding of the study).
* Family decomposition of the raw factor-1 contrast 1/2 E[d_i d_j], which is additive over families.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .data import FactorialDesign
from .dependence import cross_replicate_pairs, factorial_effects, miss_matrix, pair_moments, sharing_cell


def window_effects(design: FactorialDesign, decisions: pd.DataFrame, labels, windows, margin_pp: float = 3.0) -> pd.DataFrame:
    names = design.detectors.detector.tolist()
    idx = {d: i for i, d in enumerate(names)}
    pairs = cross_replicate_pairs(design)
    ia = np.array([idx[a] for a, _ in pairs])
    ib = np.array([idx[b] for _, b in pairs])
    cells = [sharing_cell(design, a, b) for a, b in pairs]
    shared1 = np.array([c[0] == "S" for c in cells])
    y, win = np.asarray(labels), np.asarray(windows)
    rows = []
    for w in sorted(np.unique(win)):
        m = win == w
        M = miss_matrix(decisions[m], y[m], names)
        _, J, cov, _ = pair_moments(M, ia, ib, np.ones(len(M)))
        e = factorial_effects(cov * 100, cells)
        rows.append({"window": w, **{k: v for k, v in e.items() if not k.startswith("mu_")},
                     "J_shared_factor1_pct": 100 * J[shared1].mean(), "J_crossed_factor1_pct": 100 * J[~shared1].mean()})
    df = pd.DataFrame(rows)
    ms, mc = df.J_shared_factor1_pct.median(), df.J_crossed_factor1_pct.median()
    df["burst"] = (df.J_shared_factor1_pct - ms > margin_pp) & ~(df.J_crossed_factor1_pct - mc > margin_pp)
    return df


def family_decomposition(design: FactorialDesign, decisions: pd.DataFrame, labels, families, top: int = 10) -> dict:
    """Share of the raw factor-1 contrast contributed by each family, for each level of factor 2."""
    f1 = design.factors[0]
    lv1 = design.levels[f1]
    y = np.asarray(labels)
    mal = y == 1
    fam = pd.Series(np.asarray(families, dtype=object)[mal]).fillna("(none)").to_numpy()
    det = design.detectors
    reps = list(dict.fromkeys(det.replicate))
    groups = design.levels[design.factors[1]] if len(design.factors) == 2 else [None]
    out = {}
    for g in groups:
        sub = det if g is None else det[det[design.factors[1]] == g]

        def d(rep):
            x = sub[(sub.replicate == rep) & (sub[f1] == lv1[0])].detector.iloc[0]
            z = sub[(sub.replicate == rep) & (sub[f1] == lv1[1])].detector.iloc[0]
            return (~decisions[x].to_numpy()[mal]).astype(float) - (~decisions[z].to_numpy()[mal]).astype(float)

        prods = np.mean([0.5 * d(i) * d(j) for i, j in combinations(reps, 2)], axis=0)
        tot = prods.mean()
        s = (pd.Series(prods).groupby(fam).sum() / len(prods)).sort_values(ascending=False)
        out[str(g)] = {"raw_contrast_pp": 100 * tot, "top": {str(a): float(b / tot) for a, b in s.iloc[:top].items()},
                       "no_family_share": float(s.get("(none)", 0) / tot), "n_positive_families": int((s > 0).sum())}
    return out
