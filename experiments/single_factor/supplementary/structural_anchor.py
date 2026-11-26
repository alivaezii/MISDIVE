"""Supplementary analysis D: crossed-view pairs versus two structural detectors (ANALYSIS_PLAN.md, addendum D).

Reuses the clean-run scores, the tail-rank scoring rule and threshold placement of analysis C, and the
family-cluster bootstrap. Run with the clean-run environment:

    .venv/Scripts/python.exe supplementary/structural_anchor.py

Outputs (results/supplementary/):
    anchor_summary.csv     one row per file type x operating point (aggregate over seed pairs)
    anchor_seedpairs.csv   one row per file type x operating point x seed pair
    anchor_configs.csv     FNR, achieved FP and FPR of every configuration
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from viewdep.dependence import clusters_from_family  # noqa: E402

OUT = ROOT / "results" / "supplementary"
PROTO = json.loads((ROOT / "configs" / "protocol.json").read_text())
SEEDS = (42, 123, 2026)
SEED_PAIRS = [tuple(p) for p in PROTO["pairings"]["seed_pairs"]]
B, RNG = PROTO["inference"]["replicates"], PROTO["inference"]["rng_seed"]


def tail_rank(s: np.ndarray, v_sorted: np.ndarray) -> np.ndarray:
    """Identical to supplementary/analyses.py: interpolated rank among validation benign scores."""
    n = len(v_sorted)
    r = np.interp(s, v_sorted, np.arange(n, dtype=np.float64))
    hi, lo = s > v_sorted[-1], s < v_sorted[0]
    r[hi] = n - 1 + (s[hi] - v_sorted[-1])
    r[lo] = s[lo] - v_sorted[0]
    return r


def boot_weights(family: np.ndarray, y: np.ndarray):
    """Replicate weights over test malware; same generator and draws as analyses.boot_reps."""
    pos = y == 1
    cl = clusters_from_family(np.asarray(family, dtype=object)[pos])
    uniq, inv = np.unique(cl, return_inverse=True)
    rng = np.random.default_rng(RNG)
    for a in range(0, B, 200):
        b = min(200, B - a)
        counts = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(b)])
        yield counts[:, inv].astype(np.float32)


summary, per_pair, configs = [], [], []
for fmt, label in (("dotnet", ".NET"), ("win64", "Win64")):
    art = ROOT / "artifacts" / fmt
    te = pd.read_parquet(art / "test_scores.parquet")
    yv = pd.read_parquet(ROOT / "data" / "prep" / fmt / "validation_meta.parquet").label.values.astype(int)
    val = {}
    for s in SEEDS:
        v = np.load(art / f"val_logits_s{s}.npy").astype(np.float64)
        val[f"S_{s}"], val[f"C_{s}"] = v[:, 0], v[:, 1]
    y, fam = te.label.values.astype(int), te.family.values
    pos, neg = y == 1, y == 0
    vs = {k: np.sort(v[yv == 0]) for k, v in val.items()}
    rt = {k: tail_rank(te[k].values.astype(np.float64), vs[k]) for k in val}
    rv = {k: tail_rank(val[k], vs[k]) for k in val}
    n0t, n0v = int(neg.sum()), int((yv == 0).sum())
    pairings = {}
    for i, j in SEED_PAIRS:
        pairings[(i, j)] = {"SS": (f"S_{i}", f"S_{j}"), "CC": (f"C_{i}", f"C_{j}"),
                            "SC": (f"S_{i}", f"C_{j}"), "CS": (f"S_{j}", f"C_{i}")}
    ops = [(f"retrospective (test-matched FP), {b:g}%", "retro", b) for b in (1, 0.5, 2)] + \
          [(f"validation-selected, {b:g}%", "val", b) for b in (0.5, 1, 2)]
    for op, mode, budget in ops:
        miss, fp = {}, {}
        for (i, j), cfg in pairings.items():
            for name, (a, b) in cfg.items():
                key = (i, j, name)
                if mode == "retro":
                    zb = np.sort(np.maximum(rt[a], rt[b])[neg])[::-1]
                    tau = zb[int(round(budget / 100 * n0t))]
                else:
                    zb = np.sort(np.maximum(rv[a], rv[b])[yv == 0])[::-1]
                    tau = zb[int(round(budget / 100 * n0v))]
                flag = np.maximum(rt[a], rt[b]) > tau
                miss[key] = ~flag[pos]
                fp[key] = int(flag[neg].sum())
                configs.append({"file_type": label, "operating_point": op, "seed_pair": f"{i}-{j}", "config": name,
                                "branches": f"{a}+{b}", "fnr_pct": 100 * miss[key].mean(), "fp": fp[key],
                                "fpr_pct": 100 * fp[key] / n0t,
                                "fp_target": int(round(budget / 100 * n0t)) if mode == "retro" else None})
        # Point estimates and file-level changes per anchored comparison.
        rows_sp, gain_any, loss_any, gains, losses = [], np.zeros(pos.sum(), bool), np.zeros(pos.sum(), bool), [], []
        for (i, j) in SEED_PAIRS:
            ss, cc = miss[(i, j, "SS")], miss[(i, j, "CC")]
            cr = [miss[(i, j, "SC")], miss[(i, j, "CS")]]
            g = [int((ss & ~c).sum()) for c in cr]   # detected by crossed, missed by SS
            l = [int((~ss & c).sum()) for c in cr]   # detected by SS, missed by crossed
            for c in cr:
                gain_any |= ss & ~c
                loss_any |= ~ss & c
            gains += g
            losses += l
            fnr_cr = 100 * np.mean([c.mean() for c in cr])
            rows_sp.append({"file_type": label, "operating_point": op, "seed_pair": f"{i}-{j}",
                            "fnr_SS_pct": 100 * ss.mean(), "fnr_crossed_pct": fnr_cr, "fnr_CC_pct": 100 * cc.mean(),
                            "delta_fnr_pp": fnr_cr - 100 * ss.mean(),
                            "fp_SS": fp[(i, j, "SS")], "fp_crossed_mean": np.mean([fp[(i, j, "SC")], fp[(i, j, "CS")]]),
                            "fp_CC": fp[(i, j, "CC")],
                            "gain_files_mean": np.mean(g), "loss_files_mean": np.mean(l), "net_files_mean": np.mean(g) - np.mean(l)})
        per_pair += rows_sp
        sp = pd.DataFrame(rows_sp)
        # Bootstrap of delta_FNR (crossed - SS) and of crossed - CC, seed pairs averaged within replicates.
        mm = {k: v.astype(np.float32) for k, v in miss.items()}
        reps_ss, reps_cc = [], []
        for w in boot_weights(fam, y):
            tot = w.sum(1)
            f = {k: (w @ v) / tot for k, v in mm.items()}
            d_ss = np.mean([(f[(i, j, "SC")] + f[(i, j, "CS")]) / 2 - f[(i, j, "SS")] for i, j in SEED_PAIRS], 0)
            d_cc = np.mean([(f[(i, j, "SC")] + f[(i, j, "CS")]) / 2 - f[(i, j, "CC")] for i, j in SEED_PAIRS], 0)
            reps_ss.append(d_ss)
            reps_cc.append(d_cc)
        reps_ss, reps_cc = 100 * np.concatenate(reps_ss), 100 * np.concatenate(reps_cc)
        fps = [fp[k] for k in fp]
        summary.append({"file_type": label, "operating_point": op, "n_malware": int(pos.sum()), "n_benign": n0t,
                        "fnr_SS_pct": sp.fnr_SS_pct.mean(), "fnr_crossed_pct": sp.fnr_crossed_pct.mean(),
                        "fnr_CC_pct": sp.fnr_CC_pct.mean(),
                        "delta_fnr_pp": sp.delta_fnr_pp.mean(),
                        "delta_fnr_lo": np.percentile(reps_ss, 2.5), "delta_fnr_hi": np.percentile(reps_ss, 97.5),
                        "delta_vs_CC_pp": (sp.fnr_crossed_pct - sp.fnr_CC_pct).mean(),
                        "delta_vs_CC_lo": np.percentile(reps_cc, 2.5), "delta_vs_CC_hi": np.percentile(reps_cc, 97.5),
                        "fp_SS_mean": sp.fp_SS.mean(), "fp_crossed_mean": sp.fp_crossed_mean.mean(),
                        "fp_CC_mean": sp.fp_CC.mean(), "fp_min": min(fps), "fp_max": max(fps),
                        "fpr_SS_pct": 100 * sp.fp_SS.mean() / n0t, "fpr_crossed_pct": 100 * sp.fp_crossed_mean.mean() / n0t,
                        "gain_files_mean": np.mean(gains), "loss_files_mean": np.mean(losses),
                        "net_files_mean": np.mean(gains) - np.mean(losses),
                        "gain_file_evaluations": int(sum(gains)), "loss_file_evaluations": int(sum(losses)),
                        "gain_unique_files": int(gain_any.sum()), "loss_unique_files": int(loss_any.sum()),
                        "seed_pairs_crossed_better": int((sp.delta_fnr_pp < 0).sum())})

pd.DataFrame(summary).to_csv(OUT / "anchor_summary.csv", index=False, float_format="%.4f")
pd.DataFrame(per_pair).to_csv(OUT / "anchor_seedpairs.csv", index=False, float_format="%.4f")
pd.DataFrame(configs).to_csv(OUT / "anchor_configs.csv", index=False, float_format="%.4f")
print(pd.DataFrame(summary)[["file_type", "operating_point", "fnr_SS_pct", "fnr_crossed_pct", "fnr_CC_pct", "delta_fnr_pp",
                             "delta_fnr_lo", "delta_fnr_hi", "fp_min", "fp_max", "gain_files_mean", "loss_files_mean",
                             "seed_pairs_crossed_better"]].round(2).to_string())
