"""MISDIVE: controlled evaluation of common-mode misses in redundant detectors.

The package implements the estimands and procedures of the MISDIVE methodology (Manuscript V4, Algorithm 1):
detector competence (M1), pairwise miss dependence (M2), balanced factorial attribution (M3), robustness
summaries (M4), temporal and family diagnostics (M5) and operational assessment at a matched system
false-positive budget (M6). Independent validation (M7) is a procedure, documented in ``docs/methodology.md``.

MISDIVE is an evaluation methodology. It is not a detector, a classifier or an ensemble optimiser.
"""
from .audit import AuditResult, decide, run_audit
from .data import FactorialDesign, decisions_from_scores, threshold_at_fpr
from .dependence import balanced_contrasts, competence, factorial_effects, pair_table
from .inference import bootstrap_effects, cluster_index, permutation_test
from .operational import calibrate_configurations, compare_configurations

__all__ = [
    "AuditResult", "FactorialDesign", "balanced_contrasts", "bootstrap_effects", "calibrate_configurations",
    "cluster_index", "compare_configurations", "competence", "decide", "decisions_from_scores",
    "factorial_effects", "pair_table", "permutation_test", "run_audit", "threshold_at_fpr",
]
__version__ = "1.0.0"
