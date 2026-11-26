"""M1-M3: competence, pairwise miss dependence and balanced factorial attribution.

For detectors a, b with miss indicators m_a, m_b on malware:
  q_a = E[m_a],  J_ab = E[m_a m_b] (OR-system miss rate),  C_ab = J_ab - q_a q_b = Cov(m_a, m_b),
  phi_ab = C_ab / sqrt(q_a (1 - q_a) q_b (1 - q_b)).
Only pairs of detectors from different replicates enter the attribution. With two factors each pair falls
into one of four sharing cells; with mu_c the mean dependence in cell c,
  V = 1/2 (mu_SS + mu_SX - mu_XS - mu_XX)          (main effect of sharing factor 1)
  G = 1/2 (mu_SS + mu_XS - mu_SX - mu_XX)          (main effect of sharing factor 2)
  I = (mu_SS - mu_SX) - (mu_XS - mu_XX)            (interaction)
  R = mu_XX                                        (residual: pairs sharing neither factor)
where the first letter refers to factor 1 and the second to factor 2 (S = shared, X = not shared).
With one factor, the single effect is mu_S - mu_X.
"""
from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
import pandas as pd

from .data import FactorialDesign


def competence(decisions: pd.DataFrame, labels) -> pd.DataFrame:
    """M1: miss rate (FNR) and false-positive rate (FPR) of every detector."""
    y = np.asarray(labels)
    mal, ben = y == 1, y == 0
    return pd.DataFrame({"detector": decisions.columns,
                         "fnr": [float((~decisions[d].to_numpy()[mal]).mean()) for d in decisions.columns],
                         "fpr": [float(decisions[d].to_numpy()[ben].mean()) for d in decisions.columns],
                         "n_malware": int(mal.sum()), "n_benign": int(ben.sum())})


def cross_replicate_pairs(design: FactorialDesign) -> list:
    det = design.detectors
    rep = dict(zip(det.detector, det.replicate))
    return [(a, b) for a, b in combinations(det.detector.tolist(), 2) if rep[a] != rep[b]]


def sharing_cell(design: FactorialDesign, a: str, b: str, types: dict | None = None) -> str:
    ta = types[a] if types else design.type_of(a)
    tb = types[b] if types else design.type_of(b)
    return "".join("S" if x == y else "X" for x, y in zip(ta, tb))


def miss_matrix(decisions: pd.DataFrame, labels, detectors) -> np.ndarray:
    mal = np.asarray(labels) == 1
    return np.column_stack([(~decisions[d].to_numpy()[mal]).astype(np.float64) for d in detectors])


def pair_moments(M: np.ndarray, ia: np.ndarray, ib: np.ndarray, w: np.ndarray, P: np.ndarray | None = None):
    """Weighted q, J, covariance and phi for the pairs (ia[k], ib[k]); w are file weights."""
    W = w.sum()
    q = (w @ M) / W
    if P is None:
        P = M[:, ia] * M[:, ib]
    J = (w @ P) / W
    qa, qb = q[ia], q[ib]
    cov = J - qa * qb
    phi = cov / np.sqrt(qa * (1 - qa) * qb * (1 - qb))
    return q, J, cov, phi


def pair_table(design: FactorialDesign, decisions: pd.DataFrame, labels) -> pd.DataFrame:
    """M2: dependence of every cross-replicate pair (unweighted)."""
    names = design.detectors.detector.tolist()
    idx = {d: i for i, d in enumerate(names)}
    pairs = cross_replicate_pairs(design)
    M = miss_matrix(decisions, labels, names)
    ia = np.array([idx[a] for a, _ in pairs])
    ib = np.array([idx[b] for _, b in pairs])
    q, J, cov, phi = pair_moments(M, ia, ib, np.ones(len(M)))
    y = np.asarray(labels)
    ben = y == 0
    K = [float((decisions[a].to_numpy()[ben] & decisions[b].to_numpy()[ben]).mean()) if ben.any() else np.nan
         for a, b in pairs]
    return pd.DataFrame({"a": [p[0] for p in pairs], "b": [p[1] for p in pairs],
                         "cell": [sharing_cell(design, a, b) for a, b in pairs],
                         "q_a": q[ia], "q_b": q[ib], "J": J, "cov": cov, "phi": phi, "benign_joint_alarm": K})


def factorial_effects(values, cells) -> dict:
    """M3: main effects, interaction and residual from per-pair values and their sharing cells."""
    s = pd.Series(np.asarray(values, dtype=np.float64)).groupby(np.asarray(cells)).mean()
    mu = {k: float(v) for k, v in s.items()}
    if set(mu) == {"S", "X"}:
        return {"effect_1": mu["S"] - mu["X"], "residual": mu["X"], **{f"mu_{k}": v for k, v in mu.items()}}
    need = {"SS", "SX", "XS", "XX"}
    if set(mu) != need:
        raise ValueError(f"sharing cells {sorted(mu)} do not form a balanced 2x2 design")
    eff_1 = (mu["SS"] + mu["SX"] - mu["XS"] - mu["XX"]) / 2
    eff_2 = (mu["SS"] + mu["XS"] - mu["SX"] - mu["XX"]) / 2
    inter = (mu["SS"] - mu["SX"]) - (mu["XS"] - mu["XX"])
    return {"effect_1": eff_1, "effect_2": eff_2, "interaction": inter, "residual": mu["XX"],
            "effect_1_minus_effect_2": eff_1 - eff_2, **{f"mu_{k}": v for k, v in mu.items()}}


def balanced_contrasts(design: FactorialDesign, cov_by_pair: dict) -> dict:
    """Balanced single-factor contrasts 1/2 Cov(d_i, d_j) for each factor and each pairing of the other factor.

    cov_by_pair maps (a, b) and (b, a) to the covariance. Keys: '<factor>|<other levels>' (two factors) or
    '<factor>' (one factor).
    """
    det = design.detectors
    reps = sorted(det.replicate.unique(), key=str)
    lookup = {(r, tuple(row[f] for f in design.factors)): row.detector for r in reps
              for _, row in det[det.replicate == r].iterrows()}
    out = {}
    for fi, f in enumerate(design.factors):
        lv = design.levels[f]
        others = [g for g in design.factors if g != f]
        other_pairs = [()] if not others else [(u, v) for u in design.levels[others[0]] for v in design.levels[others[0]]
                                               if u <= v]
        for op in other_pairs:
            # Same level of the other factor (or one factor): unordered replicate pairs.
            # Different levels: all ordered replicate pairs (replicate i at the first level, j at the second).
            same_other = not op or op[0] == op[1]
            combos = list(combinations(reps, 2)) if same_other else list(permutations(reps, 2))
            vals = []
            for i, j in combos:
                def d(rep, level, other):
                    t = [None] * len(design.factors)
                    t[fi] = level
                    if others:
                        t[1 - fi] = other
                    return lookup[(rep, tuple(t))]
                oi = op[0] if op else None
                oj = op[1] if op else None
                x_i, y_i = d(i, lv[0], oi), d(i, lv[1], oi)
                x_j, y_j = d(j, lv[0], oj), d(j, lv[1], oj)
                vals.append(0.5 * (cov_by_pair[(x_i, x_j)] + cov_by_pair[(y_i, y_j)]
                                   - cov_by_pair[(x_i, y_j)] - cov_by_pair[(y_i, x_j)]))
            key = f if not op else f"{f}|{op[0]}-{op[1]}"
            out[key] = float(np.mean(vals))
    return out
