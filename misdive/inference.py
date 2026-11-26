"""M3/M4 uncertainty: paired cluster bootstrap over test files and exact within-replicate permutation test.

File-level uncertainty. Clusters (e.g. malware families) are resampled with replacement; every pair is
evaluated on the same resampled files and q, J, covariance and phi are recomputed in each replicate. The
interval is conditional on the trained detectors. When no cluster labels are supplied, each file is its own
cluster (i.i.d. file bootstrap); this ignores within-family dependence and is reported as such.

Training-run uncertainty. The detector-type labels are permuted within each replicate and the effects are
recomputed. With R replicates and 2x2 cells there are 24**R permutations; they are enumerated when this is at
most ``max_exact`` and sampled otherwise. Label symmetries bound the smallest attainable p-value.
"""
from __future__ import annotations

from itertools import permutations, product

import numpy as np
import pandas as pd

from .data import FactorialDesign
from .dependence import cross_replicate_pairs, factorial_effects, miss_matrix, pair_moments, sharing_cell


def cluster_index(clusters, file_ids=None) -> tuple[np.ndarray, str]:
    """Integer cluster index per file. Missing cluster labels become singleton clusters.

    Clusters are coded by the sorted order of their string labels (numpy.unique), the coding used in the study,
    so that seeded bootstraps are reproducible. Returns (index, mode) with mode 'cluster' or 'file'.
    """
    if clusters is None:
        n = len(file_ids)
        return np.arange(n), "file"
    c = pd.Series(clusters, dtype=object)
    if file_ids is not None:
        fill = "sha:" + pd.Series(file_ids, dtype=object).astype(str)
    else:
        fill = pd.Series([f"_singleton_{i}" for i in range(len(c))], dtype=object)
    c = c.where(c.notna(), fill)
    _, inv = np.unique(c.astype(str).to_numpy(), return_inverse=True)
    return inv, "cluster"


def _effects_from(values, pairs, design, types=None):
    cells = [sharing_cell(design, a, b, types) for a, b in pairs]
    return factorial_effects(values, cells)


def bootstrap_effects(design: FactorialDesign, decisions: pd.DataFrame, labels, cluster_idx: np.ndarray,
                      n_boot: int, rng: np.random.Generator) -> dict:
    """Point estimates and percentile 95% intervals of all effects on the covariance (pp) and phi scales.

    ``cluster_idx`` indexes all files (benign and malware); only malware files enter the miss statistics.
    The draw sequence (one multinomial-equivalent count vector per replicate) matches the study's analysis code.
    """
    names = design.detectors.detector.tolist()
    idx = {d: i for i, d in enumerate(names)}
    pairs = cross_replicate_pairs(design)
    ia = np.array([idx[a] for a, _ in pairs])
    ib = np.array([idx[b] for _, b in pairs])
    M = miss_matrix(decisions, labels, names)
    P = M[:, ia] * M[:, ib]
    mal = np.asarray(labels) == 1
    inv_m = np.unique(np.asarray(cluster_idx)[mal], return_inverse=True)[1]
    ncl = int(inv_m.max()) + 1
    _, _, cov, phi = pair_moments(M, ia, ib, np.ones(len(M)), P)
    out = {"n_malware": int(mal.sum()), "n_clusters": ncl}
    reps = {"cov_pp": [], "phi": []}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, ncl, ncl), minlength=ncl)[inv_m].astype(np.float64)
        _, _, c, p = pair_moments(M, ia, ib, w, P)
        reps["cov_pp"].append(_effects_from(c * 100, pairs, design))
        reps["phi"].append(_effects_from(p, pairs, design))
    for scale, vals in (("cov_pp", cov * 100), ("phi", phi)):
        point = _effects_from(vals, pairs, design)
        ci = {k: [float(x) for x in np.percentile([r[k] for r in reps[scale]], [2.5, 97.5])]
              for k in point if not k.startswith("mu_")} if n_boot else {}
        out[scale] = {"point": point, "ci95": ci}
    out["_pairs"], out["_cov"] = pairs, cov
    return out


def permutation_test(design: FactorialDesign, decisions: pd.DataFrame, labels, max_exact: int = 50_000,
                     n_random: int = 20_000, rng: np.random.Generator | None = None) -> dict:
    """One-sided p-values P(effect_perm >= effect_obs) on both scales, permuting types within replicates."""
    names = design.detectors.detector.tolist()
    idx = {d: i for i, d in enumerate(names)}
    pairs = cross_replicate_pairs(design)
    ia = np.array([idx[a] for a, _ in pairs])
    ib = np.array([idx[b] for _, b in pairs])
    M = miss_matrix(decisions, labels, names)
    _, _, cov, phi = pair_moments(M, ia, ib, np.ones(len(M)))
    det = design.detectors
    reps = list(dict.fromkeys(det.replicate))
    members = {r: det.loc[det.replicate == r, "detector"].tolist() for r in reps}
    types0 = {d: design.type_of(d) for d in names}
    obs = {"cov_pp": _effects_from(cov * 100, pairs, design), "phi": _effects_from(phi, pairs, design)}
    keys = [k for k in obs["cov_pp"] if not k.startswith("mu_") and k != "residual"]
    per_rep = {r: list(permutations([types0[d] for d in members[r]])) for r in reps}
    total = int(np.prod([len(v) for v in per_rep.values()]))
    if total <= max_exact:
        it, exact = product(*[per_rep[r] for r in reps]), True
    else:
        rng = rng or np.random.default_rng(0)
        it = ([per_rep[r][rng.integers(len(per_rep[r]))] for r in reps] for _ in range(n_random))
        exact = False
    counts = {s: {k: 0 for k in keys} for s in obs}
    n = 0
    for combo in it:
        types = {}
        for r, perm in zip(reps, combo):
            types.update(dict(zip(members[r], perm)))
        n += 1
        for s, vals in (("cov_pp", cov * 100), ("phi", phi)):
            e = _effects_from(vals, pairs, design, types)
            for k in keys:
                counts[s][k] += e[k] >= obs[s][k] - 1e-12
    return {"exact": exact, "n_permutations": n,
            "p": {s: {k: counts[s][k] / n for k in keys} for s in obs}}
