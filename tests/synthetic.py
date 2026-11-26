"""Small synthetic detector pool with known structure, used by the tests and the quick-start example."""
from __future__ import annotations

import numpy as np
import pandas as pd

from misdive import FactorialDesign


def make_pool(n_files: int = 6000, replicates=(1, 2, 3), view_strength: float = 0.25, alg_strength: float = 0.10,
              seed: int = 0):
    """Binary decisions of a 2x2 (view x algorithm) pool with shared, factor-specific difficulty.

    Each malware file has latent 'hard for view v' and 'hard for algorithm g' indicators shared by every replicate;
    a detector misses a file with elevated probability when the file is hard for its view or algorithm.
    Returns (design, decisions, labels, families, windows, scores).
    """
    rng = np.random.default_rng(seed)
    labels = np.r_[np.ones(n_files // 2, int), np.zeros(n_files - n_files // 2, int)]
    mal = labels == 1
    hard = {("view", v): rng.random(n_files) < view_strength for v in ("S", "C")}
    hard.update({("alg", g): rng.random(n_files) < alg_strength for g in ("M", "L")})
    records, scores = [], {}
    for r in replicates:
        for v in ("S", "C"):
            for g in ("M", "L"):
                name = f"{g}{v}_{r}"
                records.append({"detector": name, "replicate": r, "view": v, "algorithm": g})
                p_miss = 0.03 + 0.55 * hard[("view", v)] + 0.30 * hard[("alg", g)]
                miss = mal & (rng.random(n_files) < np.clip(p_miss, 0, 0.95))
                s = rng.normal(0, 1, n_files) + np.where(mal, 4.0, 0.0) - np.where(miss, 6.0, 0.0)
                scores[name] = s
    design = FactorialDesign.from_records(records, factors=("view", "algorithm"))
    scores = pd.DataFrame(scores)
    decisions = scores > 2.0
    families = np.array([f"fam{i % 40}" if i % 9 else None for i in range(n_files)], dtype=object)
    windows = np.repeat(np.arange(6), int(np.ceil(n_files / 6)))[:n_files]
    return design, decisions, labels, families, windows, scores
