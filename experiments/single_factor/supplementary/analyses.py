"""Supplementary analyses A (LightGBM pairings), B (random-partition control) and C (equal-FP).

The definitions are fixed in supplementary/ANALYSIS_PLAN.md (committed before computation). Run with
the clean-run environment after supplementary/score_phase0_models.py:

    .venv/Scripts/python.exe supplementary/analyses.py <SENTRA>

<SENTRA> is the original project, read-only. Outputs (results/supplementary/):
    pairings.csv  one row per window x model family x operating point x pairing
    summary.csv   balanced same-minus-crossed contrasts with 95% family-cluster bootstrap intervals
    checks.json   threshold and decision reproduction checks
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from viewdep.dependence import clusters_from_family, contrast, pair_stats  # noqa: E402
from viewdep.thresholds import at_fpr  # noqa: E402

SENTRA = Path(sys.argv[1]).resolve()
OUT = ROOT / "results" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)
PROTO = json.loads((ROOT / "configs" / "protocol.json").read_text())
SEEDS = (42, 123, 2026)
SEED_PAIRS = [tuple(p) for p in PROTO["pairings"]["seed_pairs"]]
ALPHA = PROTO["calibration"]["branch_nominal_fpr"]
B, RNG = PROTO["inference"]["replicates"], PROTO["inference"]["rng_seed"]
checks: dict = {}


def groups_for(a: str, b: str) -> list:
    return [{"seed_pair": f"{i}-{j}", "same": [(f"{a}_{i}", f"{a}_{j}"), (f"{b}_{i}", f"{b}_{j}")],
             "crossed": [(f"{a}_{i}", f"{b}_{j}"), (f"{b}_{i}", f"{a}_{j}")]} for i, j in SEED_PAIRS]


def boot_reps(groups: list, miss: dict, y: np.ndarray, family: np.ndarray, metric: str) -> np.ndarray:
    """Replicates of the balanced contrast (metric 'excess' or 'J'); identical resampling to
    viewdep.dependence.bootstrap, so equal seeds give identical replicate weights across calls."""
    pos = y == 1
    cl = clusters_from_family(np.asarray(family, dtype=object)[pos])
    uniq, inv = np.unique(cl, return_inverse=True)
    mm = {k: v.astype(np.float32) for k, v in miss.items()}
    rng = np.random.default_rng(RNG)
    reps = []
    for a in range(0, B, 200):
        b = min(200, B - a)
        counts = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(b)])
        w = counts[:, inv].astype(np.float32)
        tot = w.sum(1)
        q = {k: (w @ v) / tot for k, v in mm.items()}

        def val(i, j):
            jj = (w @ (mm[i] * mm[j])) / tot
            return jj - q[i] * q[j] if metric == "excess" else jj

        r = np.zeros(b)
        for g in groups:
            r += np.mean([val(i, j) for i, j in g["same"]], 0) - np.mean([val(i, j) for i, j in g["crossed"]], 0)
        reps.append(r / len(groups))
    return np.concatenate(reps)


def balanced(groups: list, stats: dict, key: str) -> float:
    return float(np.mean([np.mean([stats[p][key] for p in g["same"]]) - np.mean([stats[p][key] for p in g["crossed"]])
                          for g in groups]))


rows, summ, reps_store = [], [], {}


def evaluate(window: str, family_name: str, a: str, b: str, op: str, dec: dict, y: np.ndarray, fam: np.ndarray,
             week: np.ndarray, data_note: str, extra: dict | None = None) -> None:
    g = groups_for(a, b)
    pairs = {p: pair_stats(dec[p[0]], dec[p[1]], y) for gg in g for t in ("same", "crossed") for p in gg[t]}
    for gg in g:
        for t in ("same", "crossed"):
            for p in gg[t]:
                rows.append({"window": window, "models": family_name, "operating_point": op, "seed_pair": gg["seed_pair"],
                             "type": t, "branch_1": p[0], "branch_2": p[1], **pairs[p],
                             **({"tau": extra["tau"][p]} if extra and "tau" in extra else {})})
    pos = y == 1
    miss = {k: ~v[pos] for k, v in dec.items()}
    rex = boot_reps(g, miss, y, fam, "excess")
    rj = boot_reps(g, miss, y, fam, "J")
    reps_store[(window, family_name, op)] = rex
    d, per = contrast(g, dec, y)
    wk_pos = sum(contrast(g, {k: v[week == w] for k, v in dec.items()}, y[week == w])[0] > 0 for w in np.unique(week))
    mean = lambda t, k: float(np.mean([pairs[p][k] for gg in g for p in gg[t]]))
    summ.append({"window": window, "models": family_name, "operating_point": op, "data": data_note,
                 "n_malware": int(pos.sum()), "n_benign": int((~pos).sum()),
                 "q_branch_mean_pct": 100 * float(np.mean([m.mean() for m in miss.values()])),
                 "J_same_pct": 100 * mean("same", "J"), "J_crossed_pct": 100 * mean("crossed", "J"),
                 "excess_same_pp": 100 * mean("same", "excess"), "excess_crossed_pp": 100 * mean("crossed", "excess"),
                 "sysFP_same": mean("same", "system_fp"), "sysFP_crossed": mean("crossed", "system_fp"),
                 "sysFPR_same_pct": 100 * mean("same", "system_fpr"), "sysFPR_crossed_pct": 100 * mean("crossed", "system_fpr"),
                 "delta_excess_pp": 100 * d, "delta_excess_lo": 100 * np.percentile(rex, 2.5),
                 "delta_excess_hi": 100 * np.percentile(rex, 97.5),
                 "delta_J_pp": 100 * balanced(g, pairs, "J"), "delta_J_lo": 100 * np.percentile(rj, 2.5),
                 "delta_J_hi": 100 * np.percentile(rj, 97.5),
                 "delta_per_seed_pair_pp": ";".join(f"{100 * v:.2f}" for v in per),
                 "weeks_delta_positive": f"{wk_pos}/{len(np.unique(week))}"})


def frozen_dec(val: dict, yv: np.ndarray, test: dict) -> tuple[dict, dict]:
    thr = {k: at_fpr(v[yv == 0], ALPHA) for k, v in val.items()}
    return {k: test[k] > thr[k] for k in test}, thr


# ---------------------------------------------------------------- A and B: Phase 0 .NET models
runs = SENTRA / "experiments" / "runs"
yv = np.load(SENTRA / "experiments" / "y_val.npy").astype(int)
spec = {"MLP semantic (E04)": ("E04_indep_semantic_s{}", ("S", "C")),
        "MLP random partition (E05)": ("E05_indep_random_s{}", ("R1", "R2")),
        "LightGBM semantic views (E10)": (("E10_lgbm_structural_s{}", "E10_lgbm_content_s{}"), ("S", "C"))}


def val_logits(name: str, s: int, a: str, b: str) -> dict:
    if isinstance(name, tuple):
        return {f"{a}_{s}": np.load(runs / name[0].format(s) / "val_logits.npy")[:, 0].astype(np.float64),
                f"{b}_{s}": np.load(runs / name[1].format(s) / "val_logits.npy")[:, 0].astype(np.float64)}
    v = np.load(runs / name.format(s) / "val_logits.npy").astype(np.float64)
    return {f"{a}_{s}": v[:, 0], f"{b}_{s}": v[:, 1]}


ds = pd.read_parquet(SENTRA / "results" / "predictions" / "devtest_samples.parquet")
off = pd.read_parquet(ROOT / "artifacts" / "supplementary" / "phase0_dotnet_scores_w53_64.parquet")
pred = SENTRA / "results" / "predictions"
prefix = {"MLP semantic (E04)": "sem", "MLP random partition (E05)": "rnd", "LightGBM semantic views (E10)": "lgb"}

for label, (name, (a, b)) in spec.items():
    val = {k: v for s in SEEDS for k, v in val_logits(name, s, a, b).items()}
    # W1: weeks 47-52 from saved predictions.
    t1 = {}
    for s in SEEDS:
        if isinstance(name, tuple):
            t1[f"{a}_{s}"] = np.load(pred / f"devtest_{name[0].format(s)}.npy")[:, 0].astype(np.float64)
            t1[f"{b}_{s}"] = np.load(pred / f"devtest_{name[1].format(s)}.npy")[:, 0].astype(np.float64)
        else:
            v = np.load(pred / f"devtest_{name.format(s)}.npy").astype(np.float64)
            t1[f"{a}_{s}"], t1[f"{b}_{s}"] = v[:, 0], v[:, 1]
    d1, thr = frozen_dec(val, yv, t1)
    note = "Phase 0 .NET models (50% SHA256 training subsample of weeks 1-40); thresholds 0.5% on weeks 41-46"
    evaluate("W1 .NET weeks 47-52 (Phase 0 dev-test)", label, a, b, "frozen branch thresholds (0.5%)",
             d1, ds.label.values, ds.family.values, ds.week.values, note + "; saved predictions")
    # W2: official weeks 53-64, rescored checkpoints.
    p = prefix[label]
    if p == "lgb":
        t2 = {f"S_{s}": off[f"lgbS_{s}"].values for s in SEEDS} | {f"C_{s}": off[f"lgbC_{s}"].values for s in SEEDS}
    else:
        t2 = {f"{v}_{s}": off[f"{p}_{v}_{s}"].values for s in SEEDS for v in (a, b)}
    d2, _ = frozen_dec(val, yv, t2)
    evaluate("W2 .NET weeks 53-64 (official test)", label, a, b, "frozen branch thresholds (0.5%)",
             d2, off.label.values, off.family.values, off.week.values, note + "; checkpoints rescored")


# Paired differences relative to the semantic MLP (same resampled malware in every replicate).
for w in ("W1 .NET weeks 47-52 (Phase 0 dev-test)", "W2 .NET weeks 53-64 (official test)"):
    ref = next(s for s in summ if s["window"] == w and s["models"] == "MLP semantic (E04)")
    for other in ("MLP random partition (E05)", "LightGBM semantic views (E10)"):
        o = next(s for s in summ if s["window"] == w and s["models"] == other)
        r = reps_store[(w, "MLP semantic (E04)", ref["operating_point"])] - reps_store[(w, other, o["operating_point"])]
        summ.append({"window": w, "models": f"MLP semantic (E04) minus {other}", "operating_point": ref["operating_point"],
                     "data": "paired bootstrap difference of delta_excess",
                     "delta_excess_pp": ref["delta_excess_pp"] - o["delta_excess_pp"],
                     "delta_excess_lo": 100 * np.percentile(r, 2.5), "delta_excess_hi": 100 * np.percentile(r, 97.5)})
checks["W2_E04_delta_pp_reproduces_original_11.90"] = next(
    s["delta_excess_pp"] for s in summ if s["window"].startswith("W2") and s["models"] == "MLP semantic (E04)")

# ---------------------------------------------------------------- C: equal-FP, clean-run models


def tail_rank(s: np.ndarray, v_sorted: np.ndarray) -> np.ndarray:
    n = len(v_sorted)
    r = np.interp(s, v_sorted, np.arange(n, dtype=np.float64))
    hi, lo = s > v_sorted[-1], s < v_sorted[0]
    r[hi] = n - 1 + (s[hi] - v_sorted[-1])
    r[lo] = s[lo] - v_sorted[0]
    return r


for fmt, label in (("dotnet", ".NET"), ("win64", "Win64")):
    art = ROOT / "artifacts" / fmt
    te = pd.read_parquet(art / "test_scores.parquet")
    vm = pd.read_parquet(ROOT / "data" / "prep" / fmt / "validation_meta.parquet")
    frozen = json.loads((ROOT / "results" / fmt / "frozen.json").read_text())
    yvc = vm.label.values.astype(int)
    val = {}
    for s in SEEDS:
        v = np.load(art / f"val_logits_s{s}.npy").astype(np.float64)
        val[f"S_{s}"], val[f"C_{s}"] = v[:, 0], v[:, 1]
    test = {k: te[k].values.astype(np.float64) for k in val}
    y, fam, wk = te.label.values.astype(int), te.family.values, te.week.values
    dfz, thr = frozen_dec(val, yvc, test)
    checks[f"{fmt}_thresholds_match_frozen"] = bool(all(abs(thr[k] - frozen["thresholds"][k]) < 1e-9 for k in thr))
    win = f"Clean {label} weeks 53-64 (official test)"
    note = f"Clean-run {label} models (all unique files of weeks 1-40)"
    evaluate(win, "MLP semantic (clean run)", "S", "C", "frozen branch thresholds (0.5%)", dfz, y, fam, wk, note)
    vs = {k: np.sort(v[yvc == 0]) for k, v in val.items()}
    n0v = len(vs["S_42"])
    rt = {k: tail_rank(test[k], vs[k]) for k in test}
    rv = {k: tail_rank(val[k], vs[k]) for k in val}
    # Check: the rule at tau0 reproduces the frozen OR decisions.
    tau0 = n0v - 1 - int(np.floor(ALPHA * n0v))
    checks[f"{fmt}_tau0_decision_mismatches"] = int(sum(((rt[k] > tau0) != dfz[k]).sum() for k in rt))
    g = groups_for("S", "C")
    plist = [p for gg in g for t in ("same", "crossed") for p in gg[t]]
    n0t, nb = int((y == 0).sum()), y == 0
    for r in (0.01, 0.005, 0.02):
        for mode in ("retrospective (test benign)", "prospective (validation benign)"):
            dec, taus = {}, {}
            for p in plist:
                if mode.startswith("retro"):
                    zb = np.sort(np.maximum(rt[p[0]], rt[p[1]])[nb])[::-1]
                    tau = zb[int(round(r * n0t))]
                else:
                    zb = np.sort(np.maximum(rv[p[0]], rv[p[1]])[yvc == 0])[::-1]
                    tau = zb[int(round(r * n0v))]
                taus[p] = float(tau)
                # Pair-specific branch decisions; keys are made unique per pairing.
                dec[(p, 0)], dec[(p, 1)] = rt[p[0]] > tau, rt[p[1]] > tau
            # Pairing-specific branch decisions become distinct keys, so the balanced contrast and the
            # bootstrap treat each pairing's thresholds separately.
            dd, gmap = {}, []
            for gg in g:
                ng = {"seed_pair": gg["seed_pair"], "same": [], "crossed": []}
                for t in ("same", "crossed"):
                    for p in gg[t]:
                        k1, k2 = f"{p[0]}|{p[1]}|1", f"{p[0]}|{p[1]}|2"
                        dd[k1], dd[k2] = dec[(p, 0)], dec[(p, 1)]
                        ng[t].append((k1, k2))
                gmap.append(ng)
            op = f"equal system FP, {mode}, r = {100 * r:g}%"
            pairs = {pp: pair_stats(dd[pp[0]], dd[pp[1]], y) for gg in gmap for t in ("same", "crossed") for pp in gg[t]}
            for gg, og in zip(gmap, g):
                for t in ("same", "crossed"):
                    for pp, po in zip(gg[t], og[t]):
                        rows.append({"window": win, "models": "MLP semantic (clean run)", "operating_point": op,
                                     "seed_pair": gg["seed_pair"], "type": t, "branch_1": po[0], "branch_2": po[1],
                                     **pairs[pp], "tau": taus[po]})
            pos = y == 1
            miss = {k: ~v[pos] for k, v in dd.items()}
            rex = boot_reps(gmap, miss, y, fam, "excess")
            rj = boot_reps(gmap, miss, y, fam, "J")
            mean = lambda t, k: float(np.mean([pairs[pp][k] for gg in gmap for pp in gg[t]]))
            summ.append({"window": win, "models": "MLP semantic (clean run)", "operating_point": op, "data": note,
                         "n_malware": int(pos.sum()), "n_benign": n0t,
                         "J_same_pct": 100 * mean("same", "J"), "J_crossed_pct": 100 * mean("crossed", "J"),
                         "excess_same_pp": 100 * mean("same", "excess"), "excess_crossed_pp": 100 * mean("crossed", "excess"),
                         "sysFP_same": mean("same", "system_fp"), "sysFP_crossed": mean("crossed", "system_fp"),
                         "sysFPR_same_pct": 100 * mean("same", "system_fpr"),
                         "sysFPR_crossed_pct": 100 * mean("crossed", "system_fpr"),
                         "sysFP_min": min(v["system_fp"] for v in pairs.values()),
                         "sysFP_max": max(v["system_fp"] for v in pairs.values()),
                         "sysFP_target": int(round(r * n0t)) if mode.startswith("retro") else None,
                         "q_same_mean_pct": 100 * float(np.mean([[pairs[pp]["q1"], pairs[pp]["q2"]] for gg in gmap for pp in gg["same"]])),
                         "q_crossed_mean_pct": 100 * float(np.mean([[pairs[pp]["q1"], pairs[pp]["q2"]] for gg in gmap for pp in gg["crossed"]])),
                         "delta_excess_pp": 100 * balanced(gmap, pairs, "excess"),
                         "delta_excess_lo": 100 * np.percentile(rex, 2.5), "delta_excess_hi": 100 * np.percentile(rex, 97.5),
                         "delta_J_pp": 100 * balanced(gmap, pairs, "J"),
                         "delta_J_lo": 100 * np.percentile(rj, 2.5), "delta_J_hi": 100 * np.percentile(rj, 97.5)})

# Reproduction check of the manuscript endpoint and interval.
for fmt, label in (("dotnet", ".NET"), ("win64", "Win64")):
    m = json.loads((ROOT / "results" / fmt / "summary.json").read_text())
    s = next(x for x in summ if x["window"] == f"Clean {label} weeks 53-64 (official test)" and x["operating_point"].startswith("frozen"))
    checks[f"{fmt}_manuscript_delta_reproduced"] = [s["delta_excess_pp"], m["contrast_pp"]]
    checks[f"{fmt}_manuscript_ci_reproduced"] = [[s["delta_excess_lo"], s["delta_excess_hi"]], m["ci95_pp"]]

pd.DataFrame(rows).to_csv(OUT / "pairings.csv", index=False, float_format="%.6g")
pd.DataFrame(summ).to_csv(OUT / "summary.csv", index=False, float_format="%.4f")
(OUT / "checks.json").write_text(json.dumps(checks, indent=1, default=float))
print(json.dumps(checks, indent=1, default=float))
