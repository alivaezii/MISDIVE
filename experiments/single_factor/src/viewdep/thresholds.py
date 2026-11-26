"""Validation-only threshold calibration."""
from __future__ import annotations

import numpy as np


def at_fpr(benign_scores: np.ndarray, alpha: float) -> float:
    """Threshold t with empirical FPR = mean(score > t) <= alpha: the floor(alpha*N)-th largest benign score."""
    s = np.sort(np.asarray(benign_scores, dtype=np.float64))[::-1]
    k = int(np.floor(alpha * len(s)))
    return float(s[k]) if k < len(s) else -np.inf


def joint_or_tpr(s1, s2, y, alpha: float, n_grid: int = 51) -> float:
    """Best validation TPR of h1 OR h2 subject to union FPR <= alpha (used only for early stopping).

    For each grid value of branch-1 false positives k1, branch 2 spends the remaining budget on benign
    files not already flagged by branch 1; this is exact for every k1 on the grid.
    """
    y = np.asarray(y).astype(bool)
    b1, b2, m1, m2 = s1[~y], s2[~y], s1[y], s2[y]
    kmax = int(np.floor(alpha * len(b1)))
    b1_sorted = np.sort(b1)[::-1]
    best = 0.0
    for k1 in np.unique(np.linspace(0, kmax, min(n_grid, kmax + 1)).round().astype(int)):
        t1 = b1_sorted[k1] if k1 < len(b1) else -np.inf
        flagged = b1 > t1
        rest = np.sort(b2[~flagged])[::-1]
        r = kmax - int(flagged.sum())
        t2 = rest[r] if r < len(rest) else -np.inf
        best = max(best, float(((m1 > t1) | (m2 > t2)).mean()))
    return best
