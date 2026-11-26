"""MISDIVE Phase 1: exploratory inputs for hypotheses and intervention design (read-only; no training).

Pool: Phase 0 .NET detectors MS/MC (E04 MLP branches), LS/LC (E10 LightGBM views), full-feature LightGBM
FL (E01_B0) as a conventional comparator; seeds 42/123/2026; one shared training cohort.
Windows: V = weeks 41-46 (validation; in-sample), W1 = weeks 47-52 (examined), W2 = weeks 53-64 (examined;
FL not scored there). All results EXPLORATORY.

(1) Interventions at matched system false-positive budget. A configuration (one detector or an OR pair) is
    calibrated on validation benign files only: each detector score is mapped to its validation-benign tail
    fraction t_d(s) = #{validation benign > s} / N0; the configuration flags a file when min_d t_d < tau, with
    tau chosen so that at most floor(alpha*N0) validation benign files are flagged (symmetric equal-tail
    allocation; for one detector this is the frozen threshold rule). Realised target FPR is reported. A
    retrospective variant places tau on target benign files (labelled retrospective).
(2) Family decomposition of the raw view contrast 1/2 E[d_i d_j] (additive over families).
(3) Permutation negative control: detector-type labels permuted within seed.
"""
from __future__ import annotations

from itertools import combinations, permutations
from pathlib import Path
import os

import numpy as np
import pandas as pd

SENTRA = Path(os.environ.get("MISDIVE_PHASE0_PROJECT", "MISDIVE_PHASE0_PROJECT_not_set"))
REPO = Path(os.environ.get("MISDIVE_CLEAN_RUN", Path(__file__).resolve().parents[2] / "single_factor"))
OUT = Path(__file__).resolve().parent
SEEDS = (42, 123, 2026)
ALPHAS = (0.005, 0.01, 0.02)
RNG = np.random.default_rng(20261130)


def load_scores():
    runs, pred = SENTRA / "experiments" / "runs", SENTRA / "results" / "predictions"
    yv = np.load(SENTRA / "experiments" / "y_val.npy").astype(int)
    V, W1, W2 = {}, {}, {}
    off = pd.read_parquet(REPO / "artifacts" / "supplementary" / "phase0_dotnet_scores_w53_64.parquet")
    for s in SEEDS:
        v = np.load(runs / f"E04_indep_semantic_s{s}" / "val_logits.npy").astype(float)
        V[f"MS_{s}"], V[f"MC_{s}"] = v[:, 0], v[:, 1]
        d = np.load(pred / f"devtest_E04_indep_semantic_s{s}.npy").astype(float)
        W1[f"MS_{s}"], W1[f"MC_{s}"] = d[:, 0], d[:, 1]
        for t, n in (("LS", "structural"), ("LC", "content")):
            V[f"{t}_{s}"] = np.load(runs / f"E10_lgbm_{n}_s{s}" / "val_logits.npy")[:, 0].astype(float)
            W1[f"{t}_{s}"] = np.load(pred / f"devtest_E10_lgbm_{n}_s{s}.npy")[:, 0].astype(float)
        V[f"FL_{s}"] = np.load(runs / f"E01_B0_lgbm_full_s{s}" / "val_logits.npy")[:, 0].astype(float)
        W1[f"FL_{s}"] = np.load(pred / f"devtest_E01_B0_lgbm_full_s{s}.npy")[:, 0].astype(float)
        W2[f"MS_{s}"], W2[f"MC_{s}"] = off[f"sem_S_{s}"].values, off[f"sem_C_{s}"].values
        W2[f"LS_{s}"], W2[f"LC_{s}"] = off[f"lgbS_{s}"].values, off[f"lgbC_{s}"].values
    ds = pd.read_parquet(pred / "devtest_samples.parquet")
    return {"V": (V, yv, None, None), "W1": (W1, ds.label.values, ds.family.values, ds.week.values),
            "W2": (W2, off.label.values, off.family.values, off.week.values)}


def tail(benign_sorted, s):
    # fraction of validation benign scores strictly greater than s
    return (len(benign_sorted) - np.searchsorted(benign_sorted, s, side="right")) / len(benign_sorted)


def config_eval(members, Vs, yv, Ts, yt, alpha):
    ben_v = {m: np.sort(Vs[m][yv == 0]) for m in members}
    zv = -np.min([tail(ben_v[m], Vs[m]) for m in members], axis=0)
    zt = -np.min([tail(ben_v[m], Ts[m]) for m in members], axis=0)
    zb = np.sort(zv[yv == 0])[::-1]
    tau = zb[int(np.floor(alpha * len(zb)))]
    flag = zt > tau
    fnr, fpr = 1 - flag[yt == 1].mean(), flag[yt == 0].mean()
    zb_t = np.sort(zt[yt == 0])[::-1]
    tau_r = zb_t[int(np.floor(alpha * len(zb_t)))]
    fnr_r = 1 - (zt[yt == 1] > tau_r).mean()
    return fnr, fpr, fnr_r


CONFIGS = {
    "single MS": [("MS",)], "single MC": [("MC",)], "single LS": [("LS",)], "single LC": [("LC",)],
    "single FL (full-feature LightGBM)": [("FL",)],
    "seed redundancy MS+MS": [("MS", "MS")], "seed redundancy LS+LS": [("LS", "LS")],
    "seed redundancy MC+MC": [("MC", "MC")], "seed redundancy LC+LC": [("LC", "LC")],
    "seed redundancy FL+FL": [("FL", "FL")],
    "algorithm swap, structural MS+LS": [("MS", "LS")], "algorithm swap, content MC+LC": [("MC", "LC")],
    "view swap, MLP MS+MC": [("MS", "MC")], "view swap, LightGBM LS+LC": [("LS", "LC")],
    "both swapped MS+LC": [("MS", "LC")], "both swapped LS+MC": [("LS", "MC")],
}


def interventions(data):
    rows = []
    for win in ("V", "W1", "W2"):
        Ts, yt, _, _ = data[win]
        Vs, yv, _, _ = data["V"]
        for name, spec in CONFIGS.items():
            types = spec[0]
            if any(t not in {k.split("_")[0] for k in Ts} for t in types):
                continue
            for a in ALPHAS:
                vals = []
                if len(types) == 1:
                    for s in SEEDS:
                        vals.append(config_eval([f"{types[0]}_{s}"], Vs, yv, Ts, yt, a))
                else:
                    t1, t2 = types
                    sp = [(i, j) for i, j in permutations(SEEDS, 2) if (t1 != t2 or i < j)]
                    for i, j in sp:
                        vals.append(config_eval([f"{t1}_{i}", f"{t2}_{j}"], Vs, yv, Ts, yt, a))
                v = np.array(vals)
                rows.append(dict(window=win, config=name, alpha_sys=a, n=len(v), fnr_pct=100 * v[:, 0].mean(),
                                 fnr_min=100 * v[:, 0].min(), fnr_max=100 * v[:, 0].max(),
                                 realised_fpr_pct=100 * v[:, 1].mean(), fnr_retro_pct=100 * v[:, 2].mean()))
    return pd.DataFrame(rows)


def frozen_decisions(data, win):
    Vs, yv, _, _ = data["V"]
    Ts, yt, fam, week = data[win]
    thr = {k: np.sort(Vs[k][yv == 0])[::-1][int(np.floor(0.005 * (yv == 0).sum()))] for k in Vs}
    return {k: Ts[k] > thr[k] for k in Ts if not k.startswith("FL")}, yt, fam, week


def family_decomposition(data, win="W2"):
    dec, y, fam, week = frozen_decisions(data, win)
    mal = y == 1
    f = pd.Series(fam[mal]).fillna("(none)").values
    out = []
    for alg in ("M", "L"):
        prods = np.zeros(mal.sum())
        n = 0
        for i, j in combinations(SEEDS, 2):
            di = (~dec[f"{alg}S_{i}"][mal]).astype(float) - (~dec[f"{alg}C_{i}"][mal]).astype(float)
            dj = (~dec[f"{alg}S_{j}"][mal]).astype(float) - (~dec[f"{alg}C_{j}"][mal]).astype(float)
            prods += 0.5 * di * dj
            n += 1
        prods /= n
        tot = prods.mean()
        s = pd.Series(prods).groupby(f).sum() / len(prods)
        s = s.sort_values(ascending=False)
        out.append(dict(window=win, algorithm=alg, raw_contrast_pp=100 * tot,
                        top1=s.index[0], top1_share=s.iloc[0] / tot, top5_share=s.iloc[:5].sum() / tot,
                        top10_share=s.iloc[:10].sum() / tot, n_families_positive=int((s > 0).sum()),
                        no_family_share=float(s.get("(none)", 0) / tot), top5=list(s.index[:5])))
    return pd.DataFrame(out)


def effects_from_dec(dec, y):
    mal = y == 1
    m = {k: (~v[mal]).astype(float) for k, v in dec.items()}
    cells = {}
    for a, b in combinations(sorted(m), 2):
        if a.split("_")[1] == b.split("_")[1]:
            continue
        c = (m[a] * m[b]).mean() - m[a].mean() * m[b].mean()
        cells.setdefault((a[1] == b[1], a[0] == b[0]), []).append(c)
    mu = {k: 100 * np.mean(v) for k, v in cells.items()}
    V = (mu[(True, True)] + mu[(True, False)] - mu[(False, True)] - mu[(False, False)]) / 2
    G = (mu[(True, True)] + mu[(False, True)] - mu[(True, False)] - mu[(False, False)]) / 2
    I = (mu[(True, True)] - mu[(True, False)]) - (mu[(False, True)] - mu[(False, False)])
    return V, G, I, mu[(False, False)]


def permutation_control(data, win="W2", n_perm=200):
    dec, y, _, _ = frozen_decisions(data, win)
    obs = effects_from_dec(dec, y)
    types = ["MS", "MC", "LS", "LC"]
    null = []
    for _ in range(n_perm):
        d2 = {}
        for s in SEEDS:
            perm = RNG.permutation(types)
            for t, p in zip(types, perm):
                d2[f"{p}_{s}"] = dec[f"{t}_{s}"]
        null.append(effects_from_dec(d2, y))
    null = np.array(null)
    return dict(window=win, observed=dict(zip(("V", "G", "I", "residual"), np.round(obs, 3))),
                null_mean=dict(zip(("V", "G", "I", "residual"), np.round(null.mean(0), 3))),
                null_p95=dict(zip(("V", "G", "I", "residual"), np.round(np.percentile(null, 95, 0), 3))),
                p_value=dict(zip(("V", "G", "I", "residual"), [float((null[:, k] >= obs[k]).mean()) for k in range(4)])))


def main():
    data = load_scores()
    I = interventions(data)
    I.to_csv(OUT / "interventions_dotnet.csv", index=False)
    F = pd.concat([family_decomposition(data, w) for w in ("W1", "W2")])
    F.to_csv(OUT / "family_decomposition_dotnet.csv", index=False)
    P = [permutation_control(data, w) for w in ("W1", "W2")]
    pd.Series(P).to_json(OUT / "permutation_control_dotnet.json", orient="values", indent=1)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    piv = I[I.alpha_sys == 0.01].pivot_table(index="config", columns="window",
                                              values=["fnr_pct", "realised_fpr_pct", "fnr_retro_pct"]).round(2)
    print(piv.to_string())
    print(I[(I.alpha_sys != 0.01)].pivot_table(index="config", columns=["alpha_sys", "window"], values="fnr_pct").round(2).to_string())
    print(F.round(3).to_string(index=False))
    for p in P:
        print(p)


if __name__ == "__main__":
    main()
