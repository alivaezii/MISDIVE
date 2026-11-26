"""Software-correctness tests on synthetic data (they test the implementation, not scientific generality)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from misdive import (FactorialDesign, balanced_contrasts, cluster_index, competence, decide, decisions_from_scores,
                     factorial_effects, pair_table, permutation_test, run_audit, threshold_at_fpr)
from misdive.operational import calibrate_configurations, compare_configurations, configuration_table
from synthetic import make_pool


def test_threshold_rule_and_strict_inequality():
    benign = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    assert threshold_at_fpr(benign, 0.2) == 3.0          # floor(0.2*10)=2 -> third largest
    dec = decisions_from_scores(pd.DataFrame({"d": [3.0, 3.0001]}), {"d": 3.0})
    assert dec.d.tolist() == [False, True]              # ties are never flagged
    assert (benign > threshold_at_fpr(benign, 0.2)).mean() <= 0.2


def test_design_validation_rejects_unbalanced_pool():
    recs = [{"detector": "a1", "replicate": 1, "view": "S", "algorithm": "M"},
            {"detector": "b1", "replicate": 1, "view": "C", "algorithm": "M"},
            {"detector": "a2", "replicate": 2, "view": "S", "algorithm": "M"}]
    with pytest.raises(ValueError):
        FactorialDesign.from_records(recs)


def test_independent_misses_give_zero_effects_in_expectation():
    rng = np.random.default_rng(1)
    n = 200_000
    y = np.ones(n, int)
    recs, dec = [], {}
    for r in (1, 2, 3):
        for v in "SC":
            for g in "ML":
                d = f"{g}{v}_{r}"
                recs.append({"detector": d, "replicate": r, "view": v, "algorithm": g})
                dec[d] = rng.random(n) > 0.2
    design = FactorialDesign.from_records(recs)
    pt = pair_table(design, pd.DataFrame(dec), y)
    e = factorial_effects(pt["cov"] * 100, pt.cell)
    for k in ("effect_1", "effect_2", "interaction", "residual"):
        assert abs(e[k]) < 0.15


def test_effect_identities_with_balanced_contrasts():
    design, dec, y, *_ = make_pool()
    pt = pair_table(design, dec, y)
    e = factorial_effects(pt["cov"] * 100, pt.cell)
    cov = {}
    for a, b, c in zip(pt.a, pt.b, pt["cov"]):
        cov[(a, b)] = cov[(b, a)] = 100 * c
    bc = balanced_contrasts(design, cov)
    same_alg = np.mean([bc["view|L-L"], bc["view|M-M"]])
    assert abs(e["effect_1"] - np.mean([same_alg, bc["view|L-M"]])) < 1e-9
    assert abs(e["interaction"] - (same_alg - bc["view|L-M"])) < 1e-9
    assert e["effect_1"] > e["effect_2"] > 0


def test_one_factor_effect_equals_half_covariance_of_differences():
    design, dec, y, *_ = make_pool()
    d = design.detectors
    sub = d[d.algorithm == "M"].drop(columns="algorithm")
    one = FactorialDesign.from_records(sub.to_dict("records"), factors=("view",))
    pt = pair_table(one, dec, y)
    eff = factorial_effects(pt["cov"], pt.cell)["effect_1"]
    mal = y == 1
    miss = {k: (~dec[k].to_numpy()[mal]).astype(float) for k in dec}
    vals = []
    for i, j in ((1, 2), (1, 3), (2, 3)):
        di, dj = miss[f"MS_{i}"] - miss[f"MC_{i}"], miss[f"MS_{j}"] - miss[f"MC_{j}"]
        vals.append(0.5 * (np.mean(di * dj) - di.mean() * dj.mean()))
    assert abs(eff - np.mean(vals)) < 1e-12


def test_bootstrap_is_seed_reproducible_and_flags_missing_clusters():
    design, dec, y, fam, *_ = make_pool(n_files=3000)
    r1 = run_audit(design, dec, y, clusters=fam, n_boot=50, seed=7, permutation=False)
    r2 = run_audit(design, dec, y, clusters=fam, n_boot=50, seed=7, permutation=False)
    assert r1.effects["cov_pp"]["ci95"] == r2.effects["cov_pp"]["ci95"]
    assert r1.cluster_mode == "cluster" and not r1.notes
    r3 = run_audit(design, dec, y, clusters=None, n_boot=20, seed=7, permutation=False)
    assert r3.cluster_mode == "file" and r3.notes
    lo, hi = r1.effects["cov_pp"]["ci95"]["effect_1"]
    assert lo <= r1.effects["cov_pp"]["point"]["effect_1"] <= hi


def test_cluster_index_codes_missing_labels_as_singletons():
    idx, mode = cluster_index(np.array(["a", None, "a", None], dtype=object), ["x1", "x2", "x3", "x4"])
    assert mode == "cluster" and idx[0] == idx[2] and idx[1] != idx[3]


def test_exact_permutation_count_and_p_range():
    design, dec, y, *_ = make_pool(n_files=2000, replicates=(1, 2))
    p = permutation_test(design, dec, y)
    assert p["exact"] and p["n_permutations"] == 24 ** 2
    for s in p["p"].values():
        for v in s.values():
            assert 1 / 576 <= v <= 1


def test_decision_rule():
    design, dec, y, fam, *_ = make_pool(n_files=3000)
    res = run_audit(design, dec, y, clusters=fam, n_boot=100, seed=1, permutation=False)
    d = decide(res, {"effect_1": 0.0, "effect_2": 1000.0})
    assert d["effect_1"]["supported_cov"] and not d["effect_2"]["supported_cov"] and d["effect_2"]["ruled_out"]


def test_operational_calibration_respects_budget_on_calibration_data():
    design, dec, y, fam, win, scores = make_pool(n_files=4000)
    members = {t: {r: f"{t}_{r}" for r in (1, 2, 3)} for t in ("MS", "MC", "LS", "LC")}
    configs = {"single LS": ("LS",), "seed LS+LS": ("LS", "LS"), "view swap LS+LC": ("LS", "LC")}
    flags = calibrate_configurations(configs, members, scores, y, scores, y, alpha=0.01)
    tab = configuration_table(flags, y)
    assert (tab.realised_fpr_pct <= 1.0 + 1e-9).all()   # calibration data = evaluation data here
    idx, _ = cluster_index(fam, np.arange(len(y)).astype(str))
    cmp = compare_configurations(flags, y, idx, {"swap - seed": ("view swap LS+LC", "seed LS+LS")}, 50, np.random.default_rng(0))
    assert cmp["swap - seed"]["ci95"][0] <= cmp["swap - seed"]["ci95"][1]


def test_competence_table():
    design, dec, y, *_ = make_pool(n_files=2000)
    c = competence(dec, y)
    assert set(c.columns) >= {"detector", "fnr", "fpr"} and len(c) == 12
