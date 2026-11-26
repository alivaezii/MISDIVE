"""MISDIVE Phase 0C: leakage-controlled audit of joint-miss prediction (read-only; no training; no tuning).

Pools (decisions at frozen validation thresholds, branch nominal FPR 0.5%):
  P0NET  Phase 0 .NET: MLP and LightGBM x structural/content x 3 seeds (12 detectors, 48 cross-seed pairs)
  CNET   clean-run .NET: MLP S/C x 3 seeds (6 detectors, 12 cross-seed pairs)
  CWIN   clean-run Win64: MLP S/C x 3 seeds (6 detectors, 12 cross-seed pairs)

Transport designs (source -> target):
  validation-internal: weeks 41-43 -> 44-46 (clean pools only; week metadata available)
  validation -> weeks 47-52 (P0NET only; examined)
  validation -> weeks 53-64 (all pools; examined)
All results are exploratory. No unopened data are read.

Predictors for target pair p = (a, b) in category c. "S" = source (historical), "T" = target.
  information class H (fully prospective: source labels only)
    H-IND      qa^S qb^S
    H-CATJ[L]  mean J^S over the category estimation set L
    H-PAIR     J^S_p (pair-specific historical joint miss rate)
    H-GRP[L]   qa^S qb^S + mean Cov^S over L          (historical marginals + group dependence)
  information class C (conditional: target-period marginal miss rates required, joint not used)
    C-IND      qa^T qb^T
    C-GRP[L]   qa^T qb^T + mean Cov^S over L          (= the Phase 0B "MISDIVE" predictor P3 when L = L0)
    C-PAIR     qa^T qb^T + Cov^S_p
  estimation sets L for category-level quantities (source period only):
    L0  all source pairs in category c, including p itself (Phase 0B choice)
    L1  leave-pair-out: excludes p          -> unseen combination of known detectors (components shared)
    L2  leave-detector-out: excludes every pair containing a or b -> no shared component in the estimate
Feasibility: every prediction is clipped to the Frechet bounds [max(0, qa+qb-1), min(qa, qb)] of the
marginals that predictor uses (source marginals for H, target marginals for C); violations are counted.
Errors: MAE, RMSE, mean signed error (pp). Uncertainty: family-cluster bootstrap over target malware
(B = 1000; target J and target marginals re-estimated; source estimates held fixed), paired differences.
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
B = 1000
RNG_SEED = 20261127


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, P0B / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cat_p0(a, b):
    return ("SV" if a[1] == b[1] else "XV") + ("SA" if a[0] == b[0] else "XA")


def cat_clean(a, b):
    return "SS" if a[0] == b[0] == "S" else "CC" if a[0] == b[0] == "C" else "X"


def seed_of(d):
    return d.split("_")[1]


class Window:
    """Malware miss matrix with family clusters for one pool and period."""

    def __init__(self, dec: dict, y, fam, sha=None):
        mal = np.asarray(y) == 1
        self.names = sorted(dec)
        self.M = np.column_stack([(~np.asarray(dec[k])[mal]).astype(np.float64) for k in self.names])
        f = np.asarray(fam, dtype=object)[mal] if fam is not None else None
        if f is None:
            f = np.arange(mal.sum()).astype(str).astype(object)
        f = np.where(pd.isna(f), np.array([f"_s{i}" for i in range(len(f))], dtype=object), f)
        _, self.inv = np.unique(f.astype(str), return_inverse=True)
        self.n_cl = self.inv.max() + 1

    def stats(self, pairs, w=None):
        w = np.ones(len(self.M)) if w is None else w
        W = w.sum()
        q = (w @ self.M) / W
        idx = {k: i for i, k in enumerate(self.names)}
        J = np.array([(w @ (self.M[:, idx[a]] * self.M[:, idx[b]])) / W for a, b in pairs])
        return dict(zip(self.names, q)), J


def pairs_of(names):
    return [(a, b) for a, b in combinations(sorted(names), 2) if seed_of(a) != seed_of(b)]


def predict(pairs, cats, qS, JS, qT):
    """Return {predictor: (pred array, n_infeasible_before_clip, n_undefined)}."""
    covS = np.array([JS[k] - qS[a] * qS[b] for k, (a, b) in enumerate(pairs)])
    out = {}

    def est_set(k, level):
        a, b = pairs[k]
        sel = [m for m in range(len(pairs)) if cats[m] == cats[k]]
        if level == "L1":
            sel = [m for m in sel if m != k]
        if level == "L2":
            sel = [m for m in sel if not ({a, b} & set(pairs[m]))]
        return sel

    def clip(v, q1, q2):
        lo, hi = max(0.0, q1 + q2 - 1), min(q1, q2)
        return min(max(v, lo), hi), not (lo - 1e-12 <= v <= hi + 1e-12)

    def run(name, f, qq):
        vals, bad, und = [], 0, 0
        for k, (a, b) in enumerate(pairs):
            v = f(k)
            if v is None or not np.isfinite(v):
                vals.append(np.nan)
                und += 1
                continue
            if qq is not None:
                v, viol = clip(v, qq[a], qq[b])
                bad += viol
            vals.append(v)
        out[name] = (np.array(vals), bad, und)

    run("H-IND", lambda k: qS[pairs[k][0]] * qS[pairs[k][1]], qS)
    run("H-PAIR", lambda k: JS[k], qS)
    run("C-IND", lambda k: qT[pairs[k][0]] * qT[pairs[k][1]], qT)
    run("C-PAIR", lambda k: qT[pairs[k][0]] * qT[pairs[k][1]] + covS[k], qT)
    for L in ("L0", "L1", "L2"):
        def m_or_none(arr, k, L=L):
            s = est_set(k, L)
            return float(np.mean(arr[s])) if s else None
        run(f"H-CATJ[{L}]", lambda k, L=L: m_or_none(JS, k, L), None)
        run(f"H-GRP[{L}]", lambda k, L=L: (None if m_or_none(covS, k, L) is None else
                                           qS[pairs[k][0]] * qS[pairs[k][1]] + m_or_none(covS, k, L)), qS)
        run(f"C-GRP[{L}]", lambda k, L=L: (None if m_or_none(covS, k, L) is None else
                                           qT[pairs[k][0]] * qT[pairs[k][1]] + m_or_none(covS, k, L)), qT)
    return out


def evaluate(pool, design, src: Window, tgt: Window, catf, rng):
    pairs = pairs_of(src.names)
    cats = [catf(a, b) for a, b in pairs]
    qS, JS = src.stats(pairs)
    qT, JT = tgt.stats(pairs)
    preds = predict(pairs, cats, qS, JS, qT)
    rows, prow = [], []
    for name, (p, bad, und) in preds.items():
        ok = np.isfinite(p)
        e = 100 * (p[ok] - JT[ok])
        rows.append(dict(pool=pool, design=design, predictor=name, n_pairs=int(ok.sum()), undefined=und,
                         infeasible_clipped=bad, MAE=np.abs(e).mean() if ok.any() else np.nan,
                         RMSE=np.sqrt((e ** 2).mean()) if ok.any() else np.nan,
                         bias=e.mean() if ok.any() else np.nan, max_abs=np.abs(e).max() if ok.any() else np.nan))
        for k in np.where(ok)[0]:
            prow.append(dict(pool=pool, design=design, predictor=name, pair=f"{pairs[k][0]}+{pairs[k][1]}",
                             cat=cats[k], pred_pct=100 * p[k], target_J_pct=100 * JT[k]))
    # Bootstrap: target re-estimated; conditional predictors recomputed with bootstrap target marginals.
    boots = {n: [] for n in preds}
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, tgt.n_cl, tgt.n_cl), minlength=tgt.n_cl)
        w = cnt[tgt.inv].astype(np.float64)
        qTb, JTb = tgt.stats(pairs, w)
        pb = predict(pairs, cats, qS, JS, qTb)
        for n, (p, _, _) in pb.items():
            ok = np.isfinite(p)
            boots[n].append(np.abs(100 * (p[ok] - JTb[ok])).mean() if ok.any() else np.nan)
    boots = {n: np.array(v) for n, v in boots.items()}
    for r in rows:
        b = boots[r["predictor"]]
        r["MAE_lo"], r["MAE_hi"] = np.nanpercentile(b, [2.5, 97.5])
    return rows, prow, boots


def pools():
    ap, fp = _mod("audit_prototype"), _mod("factorial_prototype")
    out = {}
    w = fp.load()
    keys = list(w)
    out["P0NET"] = (cat_p0, {"V41-46": Window(*w[keys[0]][:2], None),
                             "W47-52": Window(*w[keys[1]]), "W53-64": Window(*w[keys[2]])})
    for fmt, tag in (("dotnet", "CNET"), ("win64", "CWIN")):
        d = ap.decisions(fmt)
        v, t = d["validation_w41_46"], d["test_w53_64"]
        dec = lambda df: {c: df[c].values for c in df.columns if c[:2] in ("S_", "C_")}  # noqa: E731
        v1, v2 = v[v.week <= 43], v[v.week >= 44]
        out[tag] = (cat_clean, {"V41-46": Window(dec(v), v.label, v.family),
                                "V41-43": Window(dec(v1), v1.label, v1.family),
                                "V44-46": Window(dec(v2), v2.label, v2.family),
                                "V41-46-ex42": Window(dec(v[v.week != 42]), v[v.week != 42].label, v[v.week != 42].family),
                                "W53-64": Window(dec(t), t.label, t.family)})
    return out


def main():
    rng = np.random.default_rng(RNG_SEED)
    designs = {"P0NET": [("V41-46", "W47-52"), ("V41-46", "W53-64")],
               "CNET": [("V41-43", "V44-46"), ("V41-46", "W53-64")],
               "CWIN": [("V41-43", "V44-46"), ("V41-46", "W53-64"), ("V41-46-ex42", "W53-64")]}
    rows, prow, diffs = [], [], []
    for pool, (catf, wins) in pools().items():
        for s, t in designs[pool]:
            r, p, boots = evaluate(pool, f"{s}->{t}", wins[s], wins[t], catf, rng)
            rows += r
            prow += p
            for ref in ("H-PAIR", "H-IND", "H-CATJ[L1]", "H-GRP[L1]", "C-PAIR", "C-IND"):
                for cand in ("C-GRP[L0]", "C-GRP[L1]", "C-GRP[L2]", "H-GRP[L1]", "H-GRP[L2]"):
                    if cand == ref or np.isnan(boots[cand]).all():
                        continue
                    dlt = boots[cand] - boots[ref]
                    pt = next(x["MAE"] for x in r if x["predictor"] == cand) - next(x["MAE"] for x in r if x["predictor"] == ref)
                    diffs.append(dict(pool=pool, design=f"{s}->{t}", candidate=cand, reference=ref, dMAE=pt,
                                      lo=np.nanpercentile(dlt, 2.5), hi=np.nanpercentile(dlt, 97.5)))
    R, P, D = pd.DataFrame(rows), pd.DataFrame(prow), pd.DataFrame(diffs)
    R.to_csv(HERE / "prediction_errors.csv", index=False)
    P.to_csv(HERE / "prediction_pairs.csv", index=False)
    D.to_csv(HERE / "prediction_differences.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(R[["pool", "design", "predictor", "n_pairs", "undefined", "infeasible_clipped", "MAE", "MAE_lo", "MAE_hi",
             "RMSE", "bias", "max_abs"]].round(2).to_string(index=False))
    print(D.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
