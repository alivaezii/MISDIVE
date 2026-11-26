"""Toolkit reproduction of the released confirmatory results (Win64 weeks 47-52) from released scores.

Reproduces, with the toolkit rather than the study script, the primary and AB2 factorial effects, their
family-cluster bootstrap intervals (seed 20261201, B = 2000, primary pool first, then AB2 on the same stream)
and the exact permutation p-values, against results/confirmatory/analysis_v2_results.json.
Runtime: a few minutes (2 x 2000 bootstrap replicates and 2 x 13,824 permutations).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from misdive import FactorialDesign, decisions_from_scores, run_audit, threshold_at_fpr

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SEEDS = (42, 123, 2026)
KEYS = {"V": "effect_1", "G": "effect_2", "I": "interaction", "R": "residual", "VmG": "effect_1_minus_effect_2"}
TOL = 1e-9

pytestmark = pytest.mark.slow


def design_for(prefix):
    """prefix maps factorial type (view letter, algorithm letter) to the released detector prefix."""
    recs = []
    for s in SEEDS:
        for t in ("MS", "MC", "LS", "LC"):
            recs.append({"detector": f"{prefix[t]}_{s}", "replicate": s, "view": t[1], "algorithm": t[0]})
    return FactorialDesign.from_records(recs, factors=("view", "algorithm"))


@pytest.fixture(scope="module")
def audits():
    tgt = pd.read_parquet(RES / "scores" / "win64_w47_52_target_scores.parquet")
    thr = json.loads((RES / "thresholds" / "win64_frozen_thresholds_0.5pct.json").read_text())["thresholds"]
    src = pd.read_parquet(RES / "scores" / "win64_w41_46_validation_scores.parquet")
    ben = src.label.to_numpy() == 0
    for d, t in thr.items():  # released thresholds equal the frozen rule applied to the released validation scores
        assert threshold_at_fpr(src[d].to_numpy()[ben], 0.005) == pytest.approx(t, abs=1e-12)
    y = tgt.label.to_numpy()
    rng = np.random.default_rng(20261201)
    out = {}
    for name, prefix in (("primary", {"MS": "MS", "MC": "MC", "LS": "LS", "LC": "LC"}),
                         ("AB2_selection_matched", {"MS": "AMS", "MC": "AMC", "LS": "LS", "LC": "LC"})):
        design = design_for(prefix)
        cols = design.detectors.detector.tolist()
        dec = decisions_from_scores(tgt[cols], {c: thr[c] for c in cols})
        out[name] = run_audit(design, dec, y, clusters=tgt.family.to_numpy(dtype=object),
                              file_ids=tgt.sha256.to_numpy(), n_boot=2000, rng=rng)
    ref = json.loads((RES / "confirmatory" / "analysis_v2_results.json").read_text())
    return out, ref


@pytest.mark.parametrize("analysis", ["primary", "AB2_selection_matched"])
def test_points_intervals_and_permutation_p(audits, analysis):
    out, ref = audits
    r, e = ref[analysis], out[analysis]
    assert e.effects["n_malware"] == r["n_malware"] and e.effects["n_clusters"] == r["n_clusters"]
    assert e.cluster_mode == "cluster"
    for scale in ("cov_pp", "phi"):
        for k, tk in KEYS.items():
            assert abs(e.effects[scale]["point"][tk] - r[scale]["point"][k]) < TOL, (scale, k)
            assert np.allclose(e.effects[scale]["ci95"][tk], r[scale]["ci95"][k], atol=TOL, rtol=0), (scale, k)
        for k in ("V", "G", "I", "VmG"):
            assert e.permutation["p"][scale][KEYS[k]] == pytest.approx(r["permutation"][scale][k], abs=1e-12)
    assert e.permutation["exact"] and e.permutation["n_permutations"] == r["permutation"]["n_permutations"] == 13824
    bc = e.balanced_contrasts
    for k_ref, k_tool in (("dview_MM", "view|M-M"), ("dview_LL", "view|L-L"), ("dview_ML", "view|L-M"),
                          ("dalg_SS", "algorithm|S-S"), ("dalg_CC", "algorithm|C-C"), ("dalg_SC", "algorithm|C-S")):
        assert abs(bc[k_tool] - r["balanced_contrasts_pp"][k_ref]) < TOL, k_ref
