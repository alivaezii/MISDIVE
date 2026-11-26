"""Algorithm 1 of the MISDIVE methodology as a single call, plus the pre-specified decision rule."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import FactorialDesign
from .dependence import balanced_contrasts, competence, pair_table
from .diagnostics import family_decomposition, window_effects
from .inference import bootstrap_effects, cluster_index, permutation_test


@dataclass
class AuditResult:
    """Standardised outputs of an audit (all covariance effects in percentage points)."""

    competence: pd.DataFrame
    pairs: pd.DataFrame
    effects: dict
    balanced_contrasts: dict
    permutation: dict | None
    windows: pd.DataFrame | None
    families: dict | None
    cluster_mode: str
    notes: list = field(default_factory=list)

    def effects_table(self) -> pd.DataFrame:
        rows = []
        for scale in ("cov_pp", "phi"):
            for k, v in self.effects[scale]["point"].items():
                if k.startswith("mu_"):
                    continue
                lo, hi = self.effects[scale]["ci95"].get(k, [np.nan, np.nan])
                p = (self.permutation or {}).get("p", {}).get(scale, {}).get(k, np.nan)
                rows.append({"scale": scale, "estimand": k, "estimate": v, "ci_low": lo, "ci_high": hi, "p_training_runs": p})
        return pd.DataFrame(rows)

    def to_json(self, path) -> None:
        payload = {"effects": {s: self.effects[s] for s in ("cov_pp", "phi")}, "balanced_contrasts_pp": self.balanced_contrasts,
                   "permutation": self.permutation, "cluster_mode": self.cluster_mode, "notes": self.notes,
                   "competence": self.competence.to_dict(orient="records")}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, default=float)


def run_audit(design: FactorialDesign, decisions: pd.DataFrame, labels, clusters=None, file_ids=None,
              windows=None, n_boot: int = 2000, seed: int | None = 0, rng: np.random.Generator | None = None,
              permutation: bool = True) -> AuditResult:
    """M1-M5 on one evaluation window. Pass ``rng`` to continue a seeded random stream across several audits."""
    design.validate()
    y = np.asarray(labels)
    if file_ids is None:
        file_ids = np.arange(len(y)).astype(str)
    cidx, mode = cluster_index(clusters, file_ids)
    rng = rng if rng is not None else np.random.default_rng(seed)
    notes = []
    if mode == "file":
        notes.append("No cluster labels supplied: files were resampled independently; intervals ignore within-family dependence.")
    eff = bootstrap_effects(design, decisions, y, cidx, n_boot, rng)
    cov_by_pair = {}
    for (a, b), c in zip(eff.pop("_pairs"), eff.pop("_cov")):
        cov_by_pair[(a, b)] = cov_by_pair[(b, a)] = 100 * c
    bc = balanced_contrasts(design, cov_by_pair)
    perm = permutation_test(design, decisions, y) if permutation else None
    win = window_effects(design, decisions, y, windows) if windows is not None else None
    fam = family_decomposition(design, decisions, y, clusters) if clusters is not None else None
    return AuditResult(competence(decisions, y), pair_table(design, decisions, y), eff, bc, perm, win, fam, mode, notes)


def decide(result: AuditResult, mei: dict) -> dict:
    """Pre-specified rule: supported = estimate >= MEI and lower 95% bound > 0 (covariance scale);
    on the phi scale only a positive lower bound is required. ``mei`` maps estimand names to pp thresholds."""
    c, p = result.effects["cov_pp"], result.effects["phi"]
    out = {}
    for k, m in mei.items():
        out[k] = {"estimate_pp": c["point"][k], "ci95": c["ci95"][k], "mei_pp": m,
                  "supported_cov": bool(c["point"][k] >= m and c["ci95"][k][0] > 0),
                  "phi_lower_bound_positive": bool(p["ci95"][k][0] > 0),
                  "ruled_out": bool(c["ci95"][k][1] < m)}
    return out
