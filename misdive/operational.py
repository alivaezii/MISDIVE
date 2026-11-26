"""M6: operational assessment of candidate configurations at a matched system false-positive budget.

A configuration is a tuple of detector types combined by an OR rule (a single type is a single detector).
Every detector score is mapped to its calibration-benign tail fraction t_d(s) = #{benign > s} / N0; the
configuration flags a file when min_d t_d < tau, where tau is chosen on the calibration benign files so that at
most floor(alpha * N0) of them are flagged. The calibrated rule is applied unchanged to the evaluation window
(realised FPR is reported). A retrospective variant places tau on the evaluation benign files.

Configurations are averaged over seed combinations: same-type pairs over unordered replicate pairs, pairs of
different types over ordered replicate pairs.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd


def _tail(benign_sorted: np.ndarray, s: np.ndarray) -> np.ndarray:
    return (len(benign_sorted) - np.searchsorted(benign_sorted, s, side="right")) / len(benign_sorted)


def calibrate_configurations(configs: dict, type_members: dict, cal_scores: pd.DataFrame, cal_labels,
                             eval_scores: pd.DataFrame, eval_labels, alpha: float, cal_mask=None) -> dict:
    """Flags per configuration: {name: (flags[n_combos, n_eval], flags_retrospective[n_combos, n_eval])}.

    type_members maps a detector type to {replicate: detector column}.
    """
    yc = np.asarray(cal_labels)
    ben = (yc == 0) if cal_mask is None else ((yc == 0) & np.asarray(cal_mask))
    reps = sorted({r for m in type_members.values() for r in m}, key=str)
    ye = np.asarray(eval_labels)
    out = {}
    for name, types in configs.items():
        if len(types) == 1:
            combos = [(r,) for r in reps]
        else:
            combos = [(i, j) for i, j in permutations(reps, 2) if types[0] != types[1] or str(i) < str(j)]
        flags, retro = [], []
        for c in combos:
            mem = [type_members[t][r] for t, r in zip(types, c)]
            bs = {m: np.sort(cal_scores[m].to_numpy()[ben]) for m in mem}
            zc = -np.min([_tail(bs[m], cal_scores[m].to_numpy()) for m in mem], axis=0)
            ze = -np.min([_tail(bs[m], eval_scores[m].to_numpy()) for m in mem], axis=0)
            zb = np.sort(zc[ben])[::-1]
            tau = zb[int(np.floor(alpha * len(zb)))]
            flags.append(ze > tau)
            eb = np.sort(ze[ye == 0])[::-1]
            retro.append(ze > eb[int(np.floor(alpha * len(eb)))])
        out[name] = (np.array(flags), np.array(retro))
    return out


def configuration_table(flags: dict, eval_labels) -> pd.DataFrame:
    y = np.asarray(eval_labels)
    mal = y == 1
    rows = []
    for name, (f, fr) in flags.items():
        rows.append({"configuration": name, "fnr_pct": 100 * (1 - f[:, mal].mean(1)).mean(),
                     "realised_fpr_pct": 100 * f[:, ~mal].mean(1).mean(),
                     "fnr_target_matched_pct": 100 * (1 - fr[:, mal].mean(1)).mean(), "n_combinations": len(f)})
    return pd.DataFrame(rows)


def compare_configurations(flags: dict, eval_labels, cluster_idx: np.ndarray, comparisons: dict, n_boot: int,
                           rng: np.random.Generator) -> dict:
    """Paired cluster-bootstrap intervals for differences in mean system miss rate (pp).

    comparisons maps a name to (configuration_a, configuration_b) or (configuration_a, [configurations...]);
    a list means 'the per-replicate minimum over these configurations'.
    """
    y = np.asarray(eval_labels)
    mal = y == 1
    inv = np.asarray(cluster_idx)
    ncl = int(inv.max()) + 1
    csum = {n: np.vstack([np.bincount(inv[mal], weights=f[k, mal].astype(float), minlength=ncl) for k in range(len(f))])
            for n, (f, _) in flags.items()}
    cmal = np.bincount(inv[mal], minlength=ncl).astype(float)
    point = {n: 100 * (1 - f[:, mal].mean(1)).mean() for n, (f, _) in flags.items()}
    boot = {n: [] for n in flags}
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, ncl, ncl), minlength=ncl).astype(float)
        denom = cnt @ cmal
        for n in flags:
            boot[n].append(100 * (1 - (csum[n] @ cnt) / denom).mean())
    boot = {n: np.array(v) for n, v in boot.items()}
    out = {}
    for name, (a, b) in comparisons.items():
        if isinstance(b, (list, tuple)):
            bp, bb = min(point[x] for x in b), np.min(np.vstack([boot[x] for x in b]), axis=0)
        else:
            bp, bb = point[b], boot[b]
        d = boot[a] - bb
        out[name] = {"diff_pp": float(point[a] - bp), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return out
