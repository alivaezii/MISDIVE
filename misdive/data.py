"""Input structures: the factorial detector design, frozen thresholds and binary decisions.

Conventions (identical to the study):
  * label 1 = malware (positive), 0 = benign;
  * a detector *flags* a file when its score is strictly greater than its frozen threshold;
  * a *miss* is a malware file that is not flagged;
  * the frozen threshold at nominal false-positive rate alpha is the floor(alpha * N0)-th largest score
    among the N0 calibration benign files (0-based index into the descending order).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd


def threshold_at_fpr(benign_scores, alpha: float) -> float:
    """Frozen-threshold rule: at most floor(alpha * N0) calibration benign files score above it."""
    s = np.sort(np.asarray(benign_scores, dtype=np.float64))[::-1]
    k = int(np.floor(alpha * len(s)))
    return float(s[k]) if k < len(s) else -np.inf


def decisions_from_scores(scores: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Binary decisions (True = flagged) from scores and frozen thresholds (strict '>')."""
    missing = [d for d in scores.columns if d not in thresholds]
    if missing:
        raise KeyError(f"no frozen threshold for detectors {missing}")
    return pd.DataFrame({d: scores[d].to_numpy() > thresholds[d] for d in scores.columns}, index=scores.index)


@dataclass(frozen=True)
class FactorialDesign:
    """Detector metadata for a balanced design with one or two binary architectural factors.

    ``detectors`` has one row per detector with columns ``detector``, ``replicate`` and one column per factor
    (e.g. ``view`` and ``algorithm``). A design is balanced when every replicate contains exactly one detector
    for every combination of factor levels, so that each detector enters both arms of every contrast.
    """

    detectors: pd.DataFrame
    factors: tuple

    @classmethod
    def from_records(cls, records, factors=("view", "algorithm")) -> "FactorialDesign":
        df = pd.DataFrame(records)
        design = cls(df.reset_index(drop=True), tuple(factors))
        design.validate()
        return design

    def validate(self) -> None:
        df, fac = self.detectors, list(self.factors)
        need = {"detector", "replicate", *fac}
        if not need.issubset(df.columns):
            raise ValueError(f"design needs columns {sorted(need)}")
        if not 1 <= len(fac) <= 2:
            raise ValueError("MISDIVE attribution is defined for one or two binary factors")
        if df.detector.duplicated().any():
            raise ValueError("detector identifiers must be unique")
        for f in fac:
            if df[f].nunique() != 2:
                raise ValueError(f"factor {f!r} must have exactly two levels")
        reps = df.replicate.unique()
        if len(reps) < 2:
            raise ValueError("at least two independently trained replicates are required (three or more recommended)")
        cells = set(product(*[sorted(df[f].unique()) for f in fac]))
        for r in reps:
            got = df.loc[df.replicate == r, fac].apply(tuple, axis=1).tolist()
            if sorted(got) != sorted(cells):
                raise ValueError(f"replicate {r!r} is unbalanced: needs exactly one detector per cell {sorted(cells)}")

    @property
    def levels(self) -> dict:
        return {f: sorted(self.detectors[f].unique()) for f in self.factors}

    def type_of(self, detector: str) -> tuple:
        row = self.detectors.set_index("detector").loc[detector]
        return tuple(row[f] for f in self.factors)

    def replicate_of(self, detector: str):
        return self.detectors.set_index("detector").loc[detector, "replicate"]

    def relabel(self, mapping: dict) -> "FactorialDesign":
        """Design with detector types reassigned (used by the permutation test)."""
        df = self.detectors.copy()
        for f_i, f in enumerate(self.factors):
            df[f] = [mapping[d][f_i] for d in df.detector]
        return FactorialDesign(df, self.factors)
