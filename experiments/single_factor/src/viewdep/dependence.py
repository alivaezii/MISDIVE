"""Excess joint misses of detector pairings and the paired family-cluster bootstrap.

For branches i and j with miss indicators m_i, m_j on malware, J = E[m_i m_j], q = E[m], and the
excess joint miss rate is J - q_i q_j = Cov(m_i, m_j). The primary contrast averages the excess of
the two same-view pairings (SS, CC) and subtracts the average excess of the two crossed pairings
(SC, CS). With d_k = m_k(S) - m_k(C) for seed k, the contrast equals Cov(d_i, d_j) / 2 exactly, so it
measures how strongly the view-specific miss pattern of one seed predicts that of another.
"""
from __future__ import annotations

import numpy as np


def pair_stats(h1: np.ndarray, h2: np.ndarray, y: np.ndarray) -> dict:
    """h1, h2: boolean detections; y: labels (1 = malware)."""
    pos, neg = y == 1, y == 0
    m1, m2 = ~h1[pos], ~h2[pos]
    q1, q2, j = m1.mean(), m2.mean(), (m1 & m2).mean()
    return {"J": float(j), "q1": float(q1), "q2": float(q2), "excess": float(j - q1 * q2),
            "joint_misses": int((m1 & m2).sum()), "system_fpr": float((h1 | h2)[neg].mean()),
            "system_fp": int((h1 | h2)[neg].sum())}


def contrast(groups: list, dec: dict, y: np.ndarray) -> tuple[float, list]:
    """Primary endpoint (fraction, not pp) and its value per seed pair."""
    per = []
    for g in groups:
        same = np.mean([pair_stats(dec[a], dec[b], y)["excess"] for a, b in g["same"]])
        crossed = np.mean([pair_stats(dec[a], dec[b], y)["excess"] for a, b in g["crossed"]])
        per.append(float(same - crossed))
    return float(np.mean(per)), per


def covariance_identity(groups: list, dec: dict, y: np.ndarray) -> float:
    """Cov(d_i, d_j)/2 averaged over seed pairs; equals `contrast` up to rounding."""
    pos = y == 1
    vals = []
    for g in groups:
        (si, sj), (ci, cj) = g["same"]
        di = (~dec[si][pos]).astype(float) - (~dec[ci][pos]).astype(float)
        dj = (~dec[sj][pos]).astype(float) - (~dec[cj][pos]).astype(float)
        vals.append((np.mean(di * dj) - di.mean() * dj.mean()) / 2)
    return float(np.mean(vals))


def clusters_from_family(family: np.ndarray) -> np.ndarray:
    """Family label as cluster; malware without a family label forms its own singleton cluster."""
    return np.array([f if isinstance(f, str) and f else f"_singleton_{i}" for i, f in enumerate(family)], dtype=object)


def bootstrap(groups: list, dec: dict, y: np.ndarray, family: np.ndarray, replicates: int, seed: int,
              chunk: int = 200) -> dict:
    """Paired family-cluster bootstrap of the primary endpoint over test malware.

    Each replicate draws families with replacement; every pairing is evaluated on the same weighted
    files, and q1, q2 and J are recomputed from the replicate weights.
    """
    pos = y == 1
    clusters = clusters_from_family(np.asarray(family, dtype=object)[pos])
    uniq, inv = np.unique(clusters, return_inverse=True)
    miss = {k: (~v[pos]).astype(np.float32) for k, v in dec.items()}
    rng = np.random.default_rng(seed)
    reps = []
    for a in range(0, replicates, chunk):
        b = min(chunk, replicates - a)
        counts = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(b)])
        w = counts[:, inv].astype(np.float32)
        tot = w.sum(1)
        q = {k: (w @ v) / tot for k, v in miss.items()}

        def excess(i, j):
            return (w @ (miss[i] * miss[j])) / tot - q[i] * q[j]

        r = np.zeros(b)
        for g in groups:
            r += np.mean([excess(i, j) for i, j in g["same"]], 0) - np.mean([excess(i, j) for i, j in g["crossed"]], 0)
        reps.append(r / len(groups))
    reps = np.concatenate(reps)
    return {"ci_low": float(np.percentile(reps, 2.5)), "ci_high": float(np.percentile(reps, 97.5)),
            "share_le_zero": float((reps <= 0).mean()), "replicates": int(len(reps)),
            "clusters": int(len(uniq)), "malware_without_family": int(sum(c.startswith("_singleton_") for c in uniq))}


def decide(point_pp: float, lo_pp: float, hi_pp: float, eoi: float) -> str:
    if point_pp >= eoi and lo_pp > 0:
        return "supported"
    if hi_pp < eoi:
        return "ruled out"
    return "unresolved"
