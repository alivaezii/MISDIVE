"""Re-derive the confirmatory two-factor attribution (Win64 weeks 47-52) with the toolkit from released scores.

Inputs (all in results/): target scores of the 21 trained detectors, frozen 0.5% thresholds, and the released
analysis JSON for comparison. Uses the study's random stream (seed 20261201; primary pool, then AB2).
Runtime: a few minutes.  Run from the repository root:
    python examples/reproduce_confirmatory_attribution.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from misdive import FactorialDesign, decisions_from_scores, run_audit

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SEEDS = (42, 123, 2026)
NAMES = {"effect_1": "V", "effect_2": "G", "interaction": "I", "residual": "R", "effect_1_minus_effect_2": "VmG"}

tgt = pd.read_parquet(RES / "scores" / "win64_w47_52_target_scores.parquet")
thr = json.loads((RES / "thresholds" / "win64_frozen_thresholds_0.5pct.json").read_text())["thresholds"]
ref = json.loads((RES / "confirmatory" / "analysis_v2_results.json").read_text())
rng = np.random.default_rng(20261201)

for key, prefix in (("primary", {"MS": "MS", "MC": "MC", "LS": "LS", "LC": "LC"}),
                    ("AB2_selection_matched", {"MS": "AMS", "MC": "AMC", "LS": "LS", "LC": "LC"})):
    recs = [{"detector": f"{prefix[t]}_{s}", "replicate": s, "view": t[1], "algorithm": t[0]}
            for s in SEEDS for t in ("MS", "MC", "LS", "LC")]
    design = FactorialDesign.from_records(recs, factors=("view", "algorithm"))
    cols = design.detectors.detector.tolist()
    dec = decisions_from_scores(tgt[cols], {c: thr[c] for c in cols})
    res = run_audit(design, dec, tgt.label.to_numpy(), clusters=tgt.family.to_numpy(dtype=object),
                    file_ids=tgt.sha256.to_numpy(), n_boot=2000, rng=rng)
    print(f"\n{key}: {res.effects['n_malware']} malware files, {res.effects['n_clusters']} clusters")
    tab = res.effects_table()
    tab = tab[tab.scale == "cov_pp"].copy()
    tab["released"] = [ref[key]["cov_pp"]["point"][NAMES[k]] for k in tab.estimand]
    tab["max_abs_diff_ci"] = [float(np.max(np.abs(np.subtract([lo, hi], ref[key]["cov_pp"]["ci95"][NAMES[k]]))))
                              for k, lo, hi in zip(tab.estimand, tab.ci_low, tab.ci_high)]
    print(tab.round(6).to_string(index=False))
