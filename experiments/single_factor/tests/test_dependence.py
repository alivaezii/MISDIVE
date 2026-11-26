"""Unit tests on synthetic data for the dependence statistics and calibration rules."""
import numpy as np

from viewdep.dependence import bootstrap, contrast, covariance_identity, pair_stats
from viewdep.features import view_columns
from viewdep.thresholds import at_fpr

RNG = np.random.default_rng(0)


def _bank(n=4000):
    y = np.r_[np.ones(n // 2, int), np.zeros(n // 2, int)]
    hard_s = RNG.random(n) < 0.3   # files the structural view misses, shared across seeds
    hard_c = RNG.random(n) < 0.3
    dec = {}
    for s in (1, 2, 3):
        dec[f"S_{s}"] = ~(hard_s & (RNG.random(n) < 0.9)) | (y == 0) & (RNG.random(n) < 0.01)
        dec[f"C_{s}"] = ~(hard_c & (RNG.random(n) < 0.9)) | (y == 0) & (RNG.random(n) < 0.01)
    groups = [{"seed_pair": f"{i}-{j}", "same": [(f"S_{i}", f"S_{j}"), (f"C_{i}", f"C_{j}")],
               "crossed": [(f"S_{i}", f"C_{j}"), (f"C_{i}", f"S_{j}")]} for i, j in ((1, 2), (1, 3), (2, 3))]
    return y, dec, groups


def test_pair_stats_identical_branches():
    y, dec, _ = _bank()
    st = pair_stats(dec["S_1"], dec["S_1"], y)
    assert abs(st["J"] - st["q1"]) < 1e-12
    assert abs(st["excess"] - (st["q1"] - st["q1"] ** 2)) < 1e-12


def test_contrast_equals_half_covariance_of_view_differences():
    y, dec, groups = _bank()
    point, _ = contrast(groups, dec, y)
    assert abs(point - covariance_identity(groups, dec, y)) < 1e-12
    assert point > 0.05  # view-specific shared difficulty produces a clearly positive contrast


def test_bootstrap_interval_contains_point():
    y, dec, groups = _bank()
    fam = np.array([f"f{i % 50}" if i % 7 else None for i in range(len(y))], dtype=object)
    point, _ = contrast(groups, dec, y)
    bs = bootstrap(groups, dec, y, fam, replicates=300, seed=1)
    assert bs["ci_low"] < point < bs["ci_high"]
    assert bs["malware_without_family"] > 0


def test_threshold_respects_fpr_and_breaks_ties_conservatively():
    benign = np.r_[np.zeros(990), np.ones(10)]  # ties at the threshold are never flagged
    t = at_fpr(benign, 0.005)
    assert (benign > t).mean() <= 0.005


def test_views_partition_the_feature_vector():
    s, c = view_columns("structural"), view_columns("content")
    assert len(np.intersect1d(s, c)) == 0 and len(s) + len(c) == 2568
