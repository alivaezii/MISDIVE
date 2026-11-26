"""MISDIVE: exploratory architectural audit on existing saved scores (read-only).

Inputs (never modified): Single-factor clean-run frozen thresholds, validation logits (weeks 41-46)
and test scores (weeks 53-64, previously examined). No training, no threshold tuning: every decision
uses the frozen validation thresholds in results/<fmt>/frozen.json.

Outputs (this folder only): audit_pairings.csv, audit_weekly.csv, audit_decomposition.csv,
audit_summary.json.

All results on weeks 53-64 are EXPLORATORY (data previously examined).
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
import os

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("MISDIVE_CLEAN_RUN", Path(__file__).resolve().parents[2] / "single_factor"))
OUT = Path(__file__).resolve().parent
SEEDS = (42, 123, 2026)
B = 1000
RNG_SEED = 20261125  


def decisions(fmt: str):
    fr = json.loads((REPO / "results" / fmt / "frozen.json").read_text())["thresholds"]
    vm = pd.read_parquet(REPO / "data" / "prep" / fmt / "validation_meta.parquet")
    val = pd.DataFrame({"week": vm.week.values, "label": vm.label.values, "family": vm.family.values,
                        "sha256": vm.sha256.values})
    for s in SEEDS:
        v = np.load(REPO / "artifacts" / fmt / f"val_logits_s{s}.npy")
        val[f"S_{s}"], val[f"C_{s}"] = v[:, 0] > fr[f"S_{s}"], v[:, 1] > fr[f"C_{s}"]
    te = pd.read_parquet(REPO / "artifacts" / fmt / "test_scores.parquet")
    for s in SEEDS:
        for v in "SC":
            te[f"{v}_{s}"] = te[f"{v}_{s}"].values > fr[f"{v}_{s}"]
    return {"validation_w41_46": val, "test_w53_64": te}


def pairings():
    out = []
    for a, b in combinations(SEEDS, 2):
        sp = f"{a}-{b}"
        out += [(sp, "SS", f"S_{a}", f"S_{b}"), (sp, "CC", f"C_{a}", f"C_{b}"),
                (sp, "SC", f"S_{a}", f"C_{b}"), (sp, "CS", f"C_{a}", f"S_{b}")]
    return out


def pair_stats(d: pd.DataFrame, i: str, j: str) -> dict:
    mal, ben = d[d.label == 1], d[d.label == 0]
    mi, mj = ~mal[i].values, ~mal[j].values          # miss indicators on malware
    ai, aj = ben[i].values, ben[j].values            # false-alarm indicators on benign files
    q1, q2, J = mi.mean(), mj.mean(), (mi & mj).mean()
    r1, r2, K = ai.mean(), aj.mean(), (ai & aj).mean()
    qmin = min(q1, q2)
    return dict(q1=q1, q2=q2, J=J, indep=q1 * q2, cov=J - q1 * q2,
                ratio=J / (q1 * q2), recovery_of_better=1 - J / qmin,
                frechet_share=(J - q1 * q2) / (qmin - q1 * q2),
                r1=r1, r2=r2, K=K, sys_fpr=r1 + r2 - K, fp_cost_over_best=r1 + r2 - K - min(r1, r2),
                n_mal=len(mal), n_ben=len(ben))


def cluster_bootstrap_decomposition(d: pd.DataFrame, rng) -> dict:
    """Paired family-cluster bootstrap of crossed-minus-SS (and crossed-minus-CC) differences in J,
    in the independence term q1*q2 and in the covariance, at frozen thresholds (malware only)."""
    mal = d[d.label == 1].reset_index(drop=True)
    fam = mal.family.where(mal.family.notna(), "sha:" + mal.sha256).values
    codes, inv = np.unique(fam, return_inverse=True)
    M = {c: (~mal[c].values).astype(np.float64) for c in mal.columns if c[:2] in ("S_", "C_")}
    prs = pairings()

    def stats(w):
        W = w.sum()
        q = {c: w @ m / W for c, m in M.items()}
        res = {}
        for t in ("SS", "CC", "X"):
            sel = [p for p in prs if (p[1] in ("SC", "CS")) == (t == "X") and (t == "X" or p[1] == t)]
            Js = [w @ (M[i] * M[j]) / W for _, _, i, j in sel]
            ind = [q[i] * q[j] for _, _, i, j in sel]
            res[t] = (np.mean(Js), np.mean(ind))
        out = {}
        for ref in ("SS", "CC"):
            dJ = res["X"][0] - res[ref][0]
            dI = res["X"][1] - res[ref][1]
            out[ref] = (dJ, dI, dJ - dI)
        return out

    point = stats(np.ones(len(mal)))
    reps = {"SS": [], "CC": []}
    n_cl = len(codes)
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, n_cl, n_cl), minlength=n_cl)
        s = stats(cnt[inv].astype(np.float64))
        for ref in reps:
            reps[ref].append(s[ref])
    res = {}
    for ref in ("SS", "CC"):
        arr = np.array(reps[ref])
        lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
        res[ref] = {k: dict(pp=100 * point[ref][n], lo=100 * lo[n], hi=100 * hi[n])
                    for n, k in enumerate(("delta_J", "delta_independence_term", "delta_covariance"))}
    return res


def main():
    rows, wrows, dec_rows, summary = [], [], [], {}
    rng = np.random.default_rng(RNG_SEED)
    for fmt in ("dotnet", "win64"):
        for period, d in decisions(fmt).items():
            for sp, typ, i, j in pairings():
                rows.append(dict(format=fmt, period=period, seed_pair=sp, type=typ, pairing=f"{i}+{j}",
                                 **pair_stats(d, i, j)))
                for wk, dw in d.groupby("week"):
                    wrows.append(dict(format=fmt, period=period, week=wk, seed_pair=sp, type=typ,
                                      **{k: v for k, v in pair_stats(dw, i, j).items()
                                         if k in ("q1", "q2", "J", "cov", "ratio", "K", "sys_fpr")}))
            dec = cluster_bootstrap_decomposition(d, rng)
            for ref, v in dec.items():
                for k, x in v.items():
                    dec_rows.append(dict(format=fmt, period=period, comparison=f"crossed-minus-{ref}",
                                         quantity=k, **x))
    P = pd.DataFrame(rows)
    W = pd.DataFrame(wrows)
    D = pd.DataFrame(dec_rows)
    P.to_csv(OUT / "audit_pairings.csv", index=False)
    W.to_csv(OUT / "audit_weekly.csv", index=False)
    D.to_csv(OUT / "audit_decomposition.csv", index=False)

    # Type-level summaries and temporal stability (weekly coefficient of variation over 6 or 12 weeks).
    P["X"] = P.type.isin(["SC", "CS"])
    P["group"] = np.where(P.X, "crossed", P.type)
    W["group"] = np.where(W.type.isin(["SC", "CS"]), "crossed", W.type)
    cols = ["q1", "q2", "J", "indep", "cov", "ratio", "recovery_of_better", "frechet_share",
            "r1", "r2", "K", "sys_fpr", "fp_cost_over_best"]
    g = P.groupby(["format", "period", "group"])[cols].mean()
    summary["type_means"] = {f"{a}|{b}|{c}": {k: round(float(v), 6) for k, v in r.items()}
                             for (a, b, c), r in g.iterrows()}
    wk = W.groupby(["format", "period", "group", "week"])[["J", "cov", "q1", "ratio", "sys_fpr"]].mean()
    stab = wk.groupby(["format", "period", "group"]).agg(["mean", "std", "min", "max"])
    summary["weekly_stability"] = {f"{a}|{b}|{c}": {f"{k}_{s}": round(float(v), 6) for (k, s), v in r.items()}
                                   for (a, b, c), r in stab.iterrows()}
    # Validation-to-test transfer of the representation contrast (same-view minus crossed excess).
    tr = {}
    for fmt in ("dotnet", "win64"):
        for period in ("validation_w41_46", "test_w53_64"):
            s = P[(P.format == fmt) & (P.period == period)]
            same = s[s.type.isin(["SS", "CC"])].groupby("seed_pair")["cov"].mean()
            cross = s[s.X].groupby("seed_pair")["cov"].mean()
            tr[f"{fmt}|{period}"] = round(float(100 * (same - cross).mean()), 3)
    summary["representation_contrast_pp"] = tr
    summary["note"] = ("Frozen validation thresholds throughout. Validation thresholds were calibrated on the "
                       "same validation benign files, so validation FPRs are 0.5% by construction; checkpoints were "
                       "selected on validation OR TPR. Weeks 53-64 were previously examined: exploratory.")
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=1))
    pd.set_option("display.width", 220)
    print(g[["q1", "q2", "J", "indep", "cov", "ratio", "recovery_of_better", "frechet_share", "K",
             "sys_fpr", "fp_cost_over_best"]].mul(100).round(2).to_string())
    print(D.round(2).to_string())
    print(json.dumps(tr))
    print(stab[[("J", "mean"), ("J", "std"), ("cov", "mean"), ("cov", "std"), ("q1", "std"),
                ("ratio", "min"), ("ratio", "max")]].round(4).to_string())


if __name__ == "__main__":
    main()
