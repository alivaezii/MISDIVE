"""MISDIVE Phase 0C: (1) two-factor attribution with main effects, interaction and a correlation-scale
sensitivity; (2) verification of the Win64 validation week-42 burst; (3) drift-invariant (centred)
prediction error. Read-only; exploratory; no unopened data.
"""
from __future__ import annotations

import importlib.util
import json
from itertools import combinations
from pathlib import Path
import os

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P0B = HERE.parent / "phase0b"
REPO = Path(os.environ.get("MISDIVE_CLEAN_RUN", Path(__file__).resolve().parents[2] / "single_factor"))
SEEDS = (42, 123, 2026)


def _mod(name, d=P0B):
    spec = importlib.util.spec_from_file_location(name, d / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def factorial():
    fp = _mod("factorial_prototype")
    rows = []
    for win, (dec, y, _) in fp.load().items():
        mal = y == 1
        m = {k: (~v[mal]).astype(float) for k, v in dec.items()}
        cells = {}
        for a, b in combinations(sorted(m), 2):
            if a.split("_")[1] == b.split("_")[1]:
                continue
            q1, q2 = m[a].mean(), m[b].mean()
            c = (m[a] * m[b]).mean() - q1 * q2
            phi = c / np.sqrt(q1 * (1 - q1) * q2 * (1 - q2))
            sv, sa = a[1] == b[1], a[0] == b[0]
            cells.setdefault((sv, sa), []).append((c, phi))
        for scale, ix in (("covariance_pp", 0), ("correlation_phi", 1)):
            mu = {k: np.mean([x[ix] for x in v]) * (100 if ix == 0 else 1) for k, v in cells.items()}
            view_main = (mu[(True, True)] + mu[(True, False)] - mu[(False, True)] - mu[(False, False)]) / 2
            alg_main = (mu[(True, True)] + mu[(False, True)] - mu[(True, False)] - mu[(False, False)]) / 2
            inter = (mu[(True, True)] - mu[(True, False)]) - (mu[(False, True)] - mu[(False, False)])
            rows.append(dict(window=win, scale=scale, SVSA=mu[(True, True)], SVXA=mu[(True, False)],
                             XVSA=mu[(False, True)], XVXA=mu[(False, False)], view_main=view_main,
                             alg_main=alg_main, interaction=inter, n_pairs={str(k): len(v) for k, v in cells.items()}))
    return pd.DataFrame(rows)


def burst():
    fr = json.loads((REPO / "results" / "win64" / "frozen.json").read_text())["thresholds"]
    vm = pd.read_parquet(REPO / "data" / "prep" / "win64" / "validation_meta.parquet")
    d = vm[["week", "label", "family"]].copy()
    for s in SEEDS:
        v = np.load(REPO / "artifacts" / "win64" / f"val_logits_s{s}.npy")
        d[f"mS{s}"] = v[:, 0] <= fr[f"S_{s}"]
        d[f"mC{s}"] = v[:, 1] <= fr[f"C_{s}"]
    m = d[d.label == 1].copy()
    out = {}
    per_week = m.groupby("week")[[f"mS{s}" for s in SEEDS] + [f"mC{s}" for s in SEEDS]].mean().mul(100).round(2)
    out["miss_rate_pct_by_week"] = per_week.to_dict()
    m["allS"] = m[[f"mS{s}" for s in SEEDS]].all(axis=1)
    m["allC"] = m[[f"mC{s}" for s in SEEDS]].all(axis=1)
    w42 = m[m.week == 42]
    core = w42[w42.allS]
    out["week42_malware"] = int(len(w42))
    out["week42_missed_by_all_three_S"] = int(len(core))
    out["other_weeks_rate_missed_by_all_three_S_pct"] = round(100 * m[m.week != 42].allS.mean(), 2)
    out["week42_core_family_counts"] = core.family.fillna("(none)").value_counts().head(8).to_dict()
    out["core_detected_by_each_C_pct"] = {f"C{s}": round(100 * (~core[f"mC{s}"]).mean(), 1) for s in SEEDS}
    out["core_detected_by_at_least_one_C"] = int((~core.allC).sum())
    out["core_detected_by_all_three_C"] = int((~core[[f"mC{s}" for s in SEEDS]]).all(axis=1).sum())
    fam = "strelastealer"
    sf = w42[w42.family == fam]
    out["family_in_week42"] = dict(n=int(len(sf)), missed_all_S=int(sf.allS.sum()),
                                   detected_any_C=int((~sf.allC).sum()),
                                   S_miss_pct={f"S{s}": round(100 * sf[f"mS{s}"].mean(), 1) for s in SEEDS},
                                   C_miss_pct={f"C{s}": round(100 * sf[f"mC{s}"].mean(), 1) for s in SEEDS})
    out["family_by_validation_week"] = m[m.family == fam].groupby("week").size().to_dict()
    # Family presence in training weeks 1-40 (metadata only).
    tm = pd.read_parquet(REPO / "data" / "prep" / "win64" / "train_meta.parquet", columns=["week", "label", "family"])
    tf = tm[tm.family == fam]
    out["family_in_training"] = dict(n=int(len(tf)), first_week=int(tf.week.min()) if len(tf) else None,
                                     by_week_last10=tf.groupby("week").size().tail(10).to_dict())
    # Crossed-pair recovery of each structural branch's week-42 misses.
    rec = {}
    for i in SEEDS:
        for j in SEEDS:
            if i == j:
                continue
            si = w42[w42[f"mS{i}"]]
            rec[f"S{i}+C{j}"] = round(100 * (~si[f"mC{j}"]).mean(), 1)
    out["week42_crossed_recovery_of_S_misses_pct"] = rec
    # Contribution of week 42 to validation SS / crossed covariance.
    def cov_by(df):
        res = {}
        for t, (x, y) in {"SS": ("mS", "mS"), "CC": ("mC", "mC"), "X": ("mS", "mC")}.items():
            vals = []
            for i, j in combinations(SEEDS, 2):
                for a, b in ((f"{x}{i}", f"{y}{j}"), (f"{x}{j}", f"{y}{i}")) if t == "X" else ((f"{x}{i}", f"{y}{j}"),):
                    A, Bv = df[a].astype(float), df[b].astype(float)
                    vals.append(100 * ((A * Bv).mean() - A.mean() * Bv.mean()))
            res[t] = round(float(np.mean(vals)), 2)
        return res
    out["validation_cov_pp_all_weeks"] = cov_by(m)
    out["validation_cov_pp_excluding_week42"] = cov_by(m[m.week != 42])
    return out


def centred_errors():
    P = pd.read_csv(HERE / "prediction_pairs.csv")
    P["e"] = P.pred_pct - P.target_J_pct
    g = P.groupby(["pool", "design", "predictor"])
    res = g.apply(lambda x: pd.Series({"MAE": x.e.abs().mean(), "common_shift": x.e.mean(),
                                       "centred_MAE": (x.e - x.e.mean()).abs().mean(),
                                       "kendall_tau": pd.Series(x.pred_pct.values).corr(pd.Series(x.target_J_pct.values), method="kendall")}))
    return res.reset_index()


def main():
    F = factorial()
    F.to_csv(HERE / "factorial_effects.csv", index=False)
    Bst = burst()
    (HERE / "week42_burst.json").write_text(json.dumps(Bst, indent=1, default=str))
    C = centred_errors()
    C.to_csv(HERE / "prediction_centred_errors.csv", index=False)
    pd.set_option("display.width", 250)
    print(F.drop(columns="n_pairs").round(3).to_string(index=False))
    print(F.n_pairs.iloc[0])
    print(json.dumps({k: v for k, v in Bst.items() if k != "miss_rate_pct_by_week"}, indent=1, default=str))
    print(pd.DataFrame(Bst["miss_rate_pct_by_week"]).to_string())
    keep = ["H-IND", "H-PAIR", "H-CATJ[L1]", "H-GRP[L1]", "H-GRP[L2]", "C-IND", "C-PAIR", "C-GRP[L0]", "C-GRP[L1]", "C-GRP[L2]"]
    print(C[C.predictor.isin(keep)].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
