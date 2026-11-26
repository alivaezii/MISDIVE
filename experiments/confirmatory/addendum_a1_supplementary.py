"""Addendum A1 implementation: S-C3 seed-level sensitivity, C5 robustness and final outcome classes.

Reads (i) the analysis_v2 results JSON for the given run and (ii) the target scores file; uses the
unchanged functions of analysis_v2.py (imported, not modified). No fitting, calibration or selection on target
data: the configuration thresholds are calibrated on validation benign files exactly as in analysis_v2.

Usage:
  python -I addendum_a1_supplementary.py --dry-run-on-validation
  python -I addendum_a1_supplementary.py --target <target_scores.parquet> --authorised-E
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("a2", HERE / "analysis_v2.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
B, RNG_SEED = 2000, 20261202


def s_c3(src, tgt):
    inv = A.clusters(tgt)
    y = tgt.label.values
    mal = y == 1
    ncl = inv.max() + 1
    rng = np.random.default_rng(RNG_SEED)
    out = {}
    for alpha in (0.005, 0.01, 0.02):
        fl = A.config_flags(src, tgt, alpha)
        fl_rows = fl["single FL"][0]
        vr = np.vstack([fl[n][0] for n in A.VIEW_RESTRICTED_PAIRS])
        def fnr(rows, w=None):
            if w is None:
                return 100 * (1 - rows[:, mal].mean(1))
            cs = np.vstack([np.bincount(inv[mal], weights=r[mal].astype(float), minlength=ncl) for r in rows])
            return cs
        cs_fl, cs_vr = fnr(fl_rows, 1), fnr(vr, 1)
        cm = np.bincount(inv[mal], minlength=ncl).astype(float)
        pt_fl, pt_vr = fnr(fl_rows), fnr(vr)
        stat = pt_fl.max() - pt_vr.min()
        reps = []
        for _ in range(B):
            cnt = np.bincount(rng.integers(0, ncl, ncl), minlength=ncl).astype(float)
            d = cnt @ cm
            reps.append((100 * (1 - (cs_fl @ cnt) / d)).max() - (100 * (1 - (cs_vr @ cnt) / d)).min())
        lo, hi = np.percentile(reps, [2.5, 97.5])
        out[str(alpha)] = {"max_FL_seed_fnr_pct": float(pt_fl.max()), "FL_seed_fnr_pct": pt_fl.tolist(),
                           "min_individual_view_restricted_pairing_fnr_pct": float(pt_vr.min()),
                           "n_individual_pairings": int(len(vr)), "statistic_pp": float(stat),
                           "ci95": [float(lo), float(hi)], "pass_upper_le_0": bool(hi <= 0)}
    return out


def classify(res):
    d = res["decisions"]
    h1 = d["H1_view_main"]["cov_supported"] and d["H1_view_main"]["phi_sign"]
    h2 = d["H2_ordering"]["cov_sign"] and d["H2_ordering"]["phi_sign"]
    h3, h4 = d["H3_algorithm_main"]["cov_supported"], d["H4_interaction"]["cov_supported"]
    ab = res["AB2_selection_matched"]["cov_pp"]
    rob = {"G": ab["ci95"]["G"][0] > 0, "I": ab["ci95"]["I"][0] > 0}
    ab2_mei = {"V": ab["point"]["V"] >= 2.0 and ab["ci95"]["V"][0] > 0,
               "G": ab["point"]["G"] >= 1.0 and ab["ci95"]["G"][0] > 0,
               "I": ab["point"]["I"] >= 1.0 and ab["ci95"]["I"][0] > 0}
    if h1 and h2 and (h3 or h4):
        robust_supported = (h3 and rob["G"]) or (h4 and rob["I"])
        cls = "FULL SUCCESS" if robust_supported else "PARTIAL SUCCESS A (restricted view attribution)"
    elif h1 and h2:
        cls = "PARTIAL SUCCESS B (no separate framework; view effect replicated)"
    else:
        cls = "FAILURE (stop)"
    return {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "AB2_robustness_established": rob,
            "AB2_meets_MEI_separately": ab2_mei, "H5": d["H5_manuscript_dview_MM"], "outcome_class": cls,
            "frozen_go_rule_label": res["go_rule"]}


def main():
    if "--dry-run-on-validation" in sys.argv:
        tgt, rdir = A.load_source(), A.POOL / "development_reproduction"
    elif "--target" in sys.argv:
        tgt, rdir = pd.read_parquet(sys.argv[sys.argv.index("--target") + 1]), A.POOL / "confirmatory_reproduction"
    else:
        sys.exit("specify --dry-run-on-validation or --target <parquet> --authorised-E")
    res = json.loads((rdir / "analysis_v2_results.json").read_text())
    src = A.load_source()
    iv = res["interventions"]["0.01"]["tests"]
    out = {"label": res["label"], "classification": classify(res), "S_C3_seed_level_sensitivity": s_c3(src, tgt),
           "IH2_relabelled": {"IH2 LS+LS minus mean single-detector baseline (LS)": iv["IH2 LS+LS minus LS"],
                              "IH2 MS+MS minus mean single-detector baseline (MS)": iv["IH2 MS+MS minus MS"]}}
    (rdir / "addendum_a1_results.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
