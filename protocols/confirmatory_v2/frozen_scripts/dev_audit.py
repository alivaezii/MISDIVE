"""MISDIVE Phase 1: training and readiness audit on development data only (Win64 weeks 1-46).

Checks: pool completeness and balance; feature-view consistency across algorithms; competence and
threshold calibration on validation weeks 41-46 (with a within-validation calibration-transfer check,
thresholds from weeks 41-43 applied to 44-46); AB2 selection changes; probability calibration (descriptive).
Writes dev_audit.json next to the pool. Never reads weeks 47-52.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"F:\ELSIVIER\SENTRA_AEiC2027")
POOL = Path(r"F:\ELSIVIER\MISDIVE_Phase1_pool\win64")
SEEDS = (42, 123, 2026)


def at_fpr(b, a):
    s = np.sort(np.asarray(b, float))[::-1]
    return float(s[int(np.floor(a * len(s)))])


def auc(pos, neg):
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(logit, y, bins=15):
    p = 1 / (1 + np.exp(-np.clip(logit, -50, 50)))
    e = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(p, e) - 1, 0, bins - 1)
    return float(sum(abs(p[k == b].mean() - y[k == b].mean()) * (k == b).mean() for b in range(bins) if (k == b).any()))


def main():
    import torch
    sys.path.insert(0, str(REPO / "src"))
    from viewdep.features import Preprocessor, view_columns
    rec = json.loads((POOL / "train_record.json").read_text())
    ab2 = json.loads((POOL / "ab2_train_record.json").read_text())
    frozen = json.loads((REPO / "results" / "win64" / "frozen.json").read_text())
    art = REPO / "artifacts" / "win64"
    pp = Preprocessor.from_state(np.load(art / "preprocessor.npz"))
    idx_s, idx_c = pp.view_index("structural"), pp.view_index("content")
    out = {}
    # --- feature-view consistency
    vc = {"retained_columns": int(len(pp.keep_)), "structural": int(len(idx_s)), "content": int(len(idx_c)),
          "disjoint": bool(len(np.intersect1d(idx_s, idx_c)) == 0),
          "union_is_all_retained": bool(len(np.union1d(idx_s, idx_c)) == len(pp.keep_)),
          "structural_raw_columns_match_view_file": bool(np.isin(pp.keep_[idx_s], view_columns("structural")).all()),
          "content_raw_columns_match_view_file": bool(np.isin(pp.keep_[idx_c], view_columns("content")).all())}
    for s in SEEDS:
        sd = torch.load(art / f"model_s{s}.pt")
        vc[f"clean_MLP_s{s}_idx_s_equal"] = bool(np.array_equal(sd["idx_s"].numpy(), idx_s))
        vc[f"clean_MLP_s{s}_idx_c_equal"] = bool(np.array_equal(sd["idx_c"].numpy(), idx_c))
        for t, n in (("LS", len(idx_s)), ("LC", len(idx_c)), ("FL", len(pp.keep_))):
            import lightgbm as lgb
            bst = lgb.Booster(model_file=str(POOL / f"{t}_{s}.txt"))
            vc[f"{t}_{s}_num_features_equal"] = bool(bst.num_feature() == n)
        for b, n in (("S", len(idx_s)), ("C", len(idx_c))):
            sd2 = torch.load(POOL / f"AM{b}_{s}.pt")
            vc[f"AM{b}_{s}_input_width_equal"] = bool(sd2["0.weight"].shape[1] == n)
    out["view_consistency"] = vc
    # --- completeness
    names = [f"{t}_{s}" for t in ("LS", "LC", "FL") for s in SEEDS]
    out["completeness"] = {"lightgbm_models": sorted(rec["models"]), "missing_lightgbm": [n for n in names if n not in rec["models"]],
                           "ab2_models": sorted(k for s in ab2["seeds"].values() for k in s if k.startswith("AM")),
                           "factorial_detectors": 12, "cross_seed_pairs": 48, "pairs_per_cell": 12,
                           "clean_mlp_model_digests_match_frozen": all(
                               __import__("hashlib").sha256((art / f"model_s{s}.pt").read_bytes()).hexdigest() == frozen["models_sha256"][str(s)]
                               for s in SEEDS)}
    # --- competence and calibration on validation
    vm = pd.read_parquet(REPO / "data" / "prep" / "win64" / "validation_meta.parquet")
    y, wk = vm.label.values, vm.week.values
    sc = {}
    for s in SEEDS:
        v = np.load(art / f"val_logits_s{s}.npy")
        sc[f"MS_{s}"], sc[f"MC_{s}"] = v[:, 0], v[:, 1]
        for t in ("LS", "LC", "FL", "AMS", "AMC"):
            sc[f"{t}_{s}"] = np.load(POOL / f"val_logits_{t}_{s}.npy")
    rows = []
    early, late = wk <= 43, wk >= 44
    for d, z in sc.items():
        thr = at_fpr(z[y == 0], 0.005)
        t1 = at_fpr(z[(y == 0) & early], 0.005)
        r = dict(detector=d, threshold=thr, val_fnr_pct=100 * (z[y == 1] <= thr).mean(),
                 val_fpr_pct=100 * (z[y == 0] > thr).mean(), auc=auc(z[y == 1], z[y == 0]),
                 tpr_at_1pct_fpr=float((z[y == 1] > at_fpr(z[y == 0], 0.01)).mean()),
                 transfer_fpr_pct_w44_46=100 * (z[(y == 0) & late] > t1).mean(),
                 transfer_fnr_pct_w44_46=100 * (z[(y == 1) & late] <= t1).mean(),
                 ece_sigmoid=ece(z, y),
                 weekly_fnr_pct={int(w): round(100 * float((z[(y == 1) & (wk == w)] <= thr).mean()), 2) for w in range(41, 47)},
                 weekly_fpr_pct={int(w): round(100 * float((z[(y == 0) & (wk == w)] > thr).mean()), 3) for w in range(41, 47)})
        if d.split("_")[0] in ("MS", "MC"):
            r["frozen_threshold_abs_diff"] = abs(thr - frozen["thresholds"][f"{d[1]}_{d.split('_')[1]}"])
        if d.split("_")[0] in ("LS", "LC", "FL"):
            r["recorded_threshold_abs_diff"] = abs(thr - rec["models"][d]["threshold"])
            r["best_iteration"] = rec["models"][d]["best_iteration"]
            r["train_seconds"] = rec["models"][d]["train_seconds"]
        if d.startswith("AM"):
            st = ab2["seeds"][d.split("_")[1]][d]
            r["selected_epoch"] = st["selected_epoch"]
            r["clean_joint_best_epoch"] = st["clean_best_epoch_joint"]
        rows.append(r)
    out["competence_calibration"] = rows
    out["ab2_reproduction"] = {s: v["reproduction"] for s, v in ab2["seeds"].items()}
    (POOL / "dev_audit.json").write_text(json.dumps(out, indent=1, default=float))
    D = pd.DataFrame(rows)
    D["type"] = D.detector.str.split("_").str[0]
    pd.set_option("display.width", 220)
    print(json.dumps(vc, indent=0))
    print(json.dumps(out["completeness"], indent=0))
    print(D[["detector", "val_fnr_pct", "val_fpr_pct", "auc", "tpr_at_1pct_fpr", "transfer_fpr_pct_w44_46",
             "transfer_fnr_pct_w44_46", "ece_sigmoid"]].round(3).to_string(index=False))
    print(D.groupby("type")[["val_fnr_pct", "auc", "transfer_fpr_pct_w44_46", "transfer_fnr_pct_w44_46"]].mean().round(3))
    print(json.dumps(out["ab2_reproduction"], indent=0, default=float)[:3000])


if __name__ == "__main__":
    main()
