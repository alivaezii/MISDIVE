"""MISDIVE Phase 1: analysis script for CONFIRMATORY_PROTOCOL_V2 (sections 4-8).

Modes
  --dry-run-on-validation   uses Win64 weeks 41-46 as a stand-in "target" (development data only) to
                            verify the code path; every output is labelled DRY RUN / IN-SAMPLE.
  --target <parquet>        the one-time confirmatory run on weeks 47-52.

Source (calibration) data are always weeks 41-46. Detector names: MS/MC (clean MLP, frozen thresholds),
LS/LC/FL (LightGBM), AMS/AMC (AB2 selection-matched MLP); suffix _<seed>.

Clarifications of the frozen protocol text, fixed before any confirmatory scoring (no hypothesis changed):
  C1  The phi-scale requirement for H1/H2 is a sign requirement (lower 95% bound > 0); the MEIs in pp
      apply to the covariance scale only (the protocol gives no phi-scale MEI).
  C2  IH2's "best single LS" is the mean FNR of the three single LS detectors (likewise MS), compared with
      the mean FNR of the three LS+LS (MS+MS) seed-redundant pairs.
  C3  IH3's "best configuration" is, within each bootstrap replicate, the minimum mean FNR over the
      view-restricted two-detector configuration types (seed redundancy, algorithm swap, view swap,
      both swapped; FL excluded); FL is the mean of the three single FL detectors.
  C4  Pairings for interventions with two types use all ordered cross-seed combinations; same-type pairs
      use the three unordered seed pairs.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations, permutations, product
from pathlib import Path
import os

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[2] / "results"
POOL = Path(os.environ.get("MISDIVE_OUT", Path(__file__).resolve().parents[2] / "work"))  # outputs
SEEDS = (42, 123, 2026)
TYPES = ("MS", "MC", "LS", "LC")
B, RNG_SEED = 2000, 20261201
MEI = {"V": 2.0, "G": 1.0, "I": 1.0, "H5": 2.0}


def at_fpr(benign, alpha):
    s = np.sort(np.asarray(benign, dtype=np.float64))[::-1]
    return float(s[int(np.floor(alpha * len(s)))])


# ----------------------------------------------------------------------------- data
def load_source():
    """Validation (weeks 41-46) scores of all 21 detectors, as released in results/scores."""
    return pd.read_parquet(RES / "scores" / "win64_w41_46_validation_scores.parquet")


def detectors(df):
    return [c for c in df.columns if c.split("_")[0] in ("MS", "MC", "LS", "LC", "FL", "AMS", "AMC")]


def thresholds(src, alpha):
    ben = src.label.values == 0
    return {d: at_fpr(src[d].values[ben], alpha) for d in detectors(src)}


def clusters(df):
    f = df.family.astype(object).where(df.family.notna(), "sha:" + df.sha256.astype(str)).values
    _, inv = np.unique(f.astype(str), return_inverse=True)
    return inv


# ----------------------------------------------------------------------------- factorial
def pool_names(prefix_map):
    """prefix_map maps factorial type (MS/MC/LS/LC) to the detector prefix used."""
    return {f"{t}_{s}": f"{prefix_map[t]}_{s}" for t in TYPES for s in SEEDS}


def cross_pairs():
    names = [f"{t}_{s}" for t in TYPES for s in SEEDS]
    return [(a, b) for a, b in combinations(names, 2) if a.split("_")[1] != b.split("_")[1]]


def cell(a, b, lab=None):
    ta, tb = (lab[a], lab[b]) if lab else (a[:2], b[:2])
    return ("SV" if ta[1] == tb[1] else "XV") + ("SA" if ta[0] == tb[0] else "XA")


def pair_moments(M, idx, pairs, w, P=None):
    W = w.sum()
    q = (w @ M) / W
    ia = np.array([idx[a] for a, _ in pairs])
    ib = np.array([idx[b] for _, b in pairs])
    if P is None:
        P = M[:, ia] * M[:, ib]
    J = (w @ P) / W
    qa, qb = q[ia], q[ib]
    cov = J - qa * qb
    phi = cov / np.sqrt(qa * (1 - qa) * qb * (1 - qb))
    return cov, phi


def effects_from(values, pairs, lab=None):
    cells = {}
    for v, (a, b) in zip(values, pairs):
        cells.setdefault(cell(a, b, lab), []).append(v)
    mu = {k: float(np.mean(v)) for k, v in cells.items()}
    V = (mu["SVSA"] + mu["SVXA"] - mu["XVSA"] - mu["XVXA"]) / 2
    G = (mu["SVSA"] + mu["XVSA"] - mu["SVXA"] - mu["XVXA"]) / 2
    I = (mu["SVSA"] - mu["SVXA"]) - (mu["XVSA"] - mu["XVXA"])
    return dict(V=V, G=G, I=I, R=mu["XVXA"], VmG=V - G, **{f"mu_{k}": v for k, v in mu.items()},
                n_cells={k: len(v) for k, v in cells.items()})


def balanced_contrasts(cov, pairs):
    C = {}
    for v, (a, b) in zip(cov, pairs):
        C[(a, b)] = C[(b, a)] = v
    def dv(g1, g2):
        combos = [(i, j) for i, j in permutations(SEEDS, 2) if (g1 != g2 or i < j)]
        return float(np.mean([0.5 * (C[(f"{g1}S_{i}", f"{g2}S_{j}")] + C[(f"{g1}C_{i}", f"{g2}C_{j}")]
                                     - C[(f"{g1}S_{i}", f"{g2}C_{j}")] - C[(f"{g1}C_{i}", f"{g2}S_{j}")]) for i, j in combos]))
    def da(v1, v2):
        combos = [(i, j) for i, j in permutations(SEEDS, 2) if (v1 != v2 or i < j)]
        return float(np.mean([0.5 * (C[(f"M{v1}_{i}", f"M{v2}_{j}")] + C[(f"L{v1}_{i}", f"L{v2}_{j}")]
                                     - C[(f"M{v1}_{i}", f"L{v2}_{j}")] - C[(f"L{v1}_{i}", f"M{v2}_{j}")]) for i, j in combos]))
    return {"dview_MM": dv("M", "M"), "dview_LL": dv("L", "L"), "dview_ML": dv("M", "L"),
            "dalg_SS": da("S", "S"), "dalg_CC": da("C", "C"), "dalg_SC": da("S", "C")}


def factorial(dec, y, inv, prefix_map, n_boot, rng, label):
    """dec: {detector name: bool decisions}. Returns point, CI, permutation p, identity checks."""
    nm = pool_names(prefix_map)
    mal = y == 1
    names = list(nm)
    M = np.column_stack([(~dec[nm[k]][mal]).astype(np.float64) for k in names])
    idx = {k: i for i, k in enumerate(names)}
    pairs = cross_pairs()
    inv_m = np.unique(inv[mal], return_inverse=True)[1]
    ncl = inv_m.max() + 1
    P = M[:, [idx[a] for a, _ in pairs]] * M[:, [idx[b] for _, b in pairs]]
    cov, phi = pair_moments(M, idx, pairs, np.ones(mal.sum()), P)
    out = {"label": label, "n_malware": int(mal.sum()), "n_clusters": int(ncl)}
    for scale, vals, mult in (("cov_pp", cov, 100.0), ("phi", phi, 1.0)):
        eff = effects_from(vals * mult, pairs)
        out[scale] = {"point": {k: v for k, v in eff.items() if k != "n_cells"}, "n_cells": eff["n_cells"]}
    bc = balanced_contrasts(cov * 100, pairs)
    out["balanced_contrasts_pp"] = bc
    e = out["cov_pp"]["point"]
    out["identity_check"] = {
        "V": abs(e["V"] - np.mean([np.mean([bc["dview_MM"], bc["dview_LL"]]), bc["dview_ML"]])),
        "I": abs(e["I"] - (np.mean([bc["dview_MM"], bc["dview_LL"]]) - bc["dview_ML"]))}
    assert max(out["identity_check"].values()) < 1e-9, out["identity_check"]
    # File-level paired family-cluster bootstrap.
    reps = {"cov_pp": [], "phi": [], "bc": []}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, ncl, ncl), minlength=ncl)[inv_m].astype(np.float64)
        c, p = pair_moments(M, idx, pairs, w, P)
        reps["cov_pp"].append(effects_from(c * 100, pairs))
        reps["phi"].append(effects_from(p, pairs))
        reps["bc"].append(balanced_contrasts(c * 100, pairs))
    for scale in ("cov_pp", "phi"):
        out[scale]["ci95"] = {k: list(np.percentile([r[k] for r in reps[scale]], [2.5, 97.5]))
                              for k in ("V", "G", "I", "R", "VmG")}
    out["balanced_contrasts_ci95"] = {k: list(np.percentile([r[k] for r in reps["bc"]], [2.5, 97.5])) for k in bc}
    # Replicate-level exact randomisation test: permute type labels within each seed (24^3).
    obs = {s: out[s]["point"] for s in ("cov_pp", "phi")}
    null = {s: [] for s in obs}
    for perm in product(list(permutations(TYPES)), repeat=len(SEEDS)):
        lab = {}
        for s, pm in zip(SEEDS, perm):
            for t, newt in zip(TYPES, pm):
                lab[f"{t}_{s}"] = newt
        for s, vals, mult in (("cov_pp", cov, 100.0), ("phi", phi, 1.0)):
            null[s].append(effects_from(vals * mult, pairs, lab))
    out["permutation"] = {s: {k: float(np.mean([n[k] >= obs[s][k] - 1e-12 for n in null[s]])) for k in ("V", "G", "I", "VmG")}
                          for s in null}
    out["permutation"]["n_permutations"] = len(null["cov_pp"])
    return out, cov, pairs, M, idx, inv_m


def decide(f):
    c, p = f["cov_pp"], f["phi"]
    def sup(scale, k, mei):
        lo = scale["ci95"][k][0]
        return bool(scale["point"][k] >= mei and lo > 0)
    def ruled_out(k, mei):
        return bool(c["ci95"][k][1] < mei)
    return {
        "H1_view_main": {"cov_supported": sup(c, "V", MEI["V"]), "phi_sign": bool(p["ci95"]["V"][0] > 0),
                         "ruled_out": ruled_out("V", MEI["V"])},
        "H2_ordering": {"cov_sign": bool(c["ci95"]["VmG"][0] > 0), "phi_sign": bool(p["ci95"]["VmG"][0] > 0)},
        "H3_algorithm_main": {"cov_supported": sup(c, "G", MEI["G"]), "ruled_out": ruled_out("G", MEI["G"])},
        "H4_interaction": {"cov_supported": sup(c, "I", MEI["I"]), "ruled_out": ruled_out("I", MEI["I"])},
        "H5_manuscript_dview_MM": {"estimate": f["balanced_contrasts_pp"]["dview_MM"],
                                   "ci95": f["balanced_contrasts_ci95"]["dview_MM"],
                                   "supported": bool(f["balanced_contrasts_pp"]["dview_MM"] >= MEI["H5"]
                                                     and f["balanced_contrasts_ci95"]["dview_MM"][0] > 0),
                                   "ruled_out": bool(f["balanced_contrasts_ci95"]["dview_MM"][1] < MEI["H5"])}}


def go_rule(d):
    h1 = d["H1_view_main"]["cov_supported"] and d["H1_view_main"]["phi_sign"]
    h2 = d["H2_ordering"]["cov_sign"] and d["H2_ordering"]["phi_sign"]
    h34 = d["H3_algorithm_main"]["cov_supported"] or d["H4_interaction"]["cov_supported"]
    if h1 and h2 and h34:
        return "GO: two-factor attribution replicated"
    if h1 and h2:
        return "NO SEPARATE FRAMEWORK: view effect replicates; algorithm factor not demonstrated"
    return "STOP: representation attribution not robustly replicated"


# ----------------------------------------------------------------------------- interventions
CONFIGS = {
    "single MS": ("MS",), "single MC": ("MC",), "single LS": ("LS",), "single LC": ("LC",), "single FL": ("FL",),
    "seed MS+MS": ("MS", "MS"), "seed MC+MC": ("MC", "MC"), "seed LS+LS": ("LS", "LS"), "seed LC+LC": ("LC", "LC"),
    "seed FL+FL": ("FL", "FL"),
    "alg swap MS+LS": ("MS", "LS"), "alg swap MC+LC": ("MC", "LC"),
    "view swap MS+MC": ("MS", "MC"), "view swap LS+LC": ("LS", "LC"),
    "both MS+LC": ("MS", "LC"), "both LS+MC": ("LS", "MC"),
}
VIEW_RESTRICTED_PAIRS = [k for k, v in CONFIGS.items() if len(v) == 2 and "FL" not in v]


def tail(ben_sorted, s):
    return (len(ben_sorted) - np.searchsorted(ben_sorted, s, side="right")) / len(ben_sorted)


def config_flags(src, tgt, alpha, src_mask=None):
    ben = (src.label.values == 0) if src_mask is None else ((src.label.values == 0) & src_mask)
    bsorted = {d: np.sort(src[d].values[ben]) for d in detectors(src)}
    out = {}
    for name, types in CONFIGS.items():
        combos = ([(s,) for s in SEEDS] if len(types) == 1 else
                  [(i, j) for i, j in permutations(SEEDS, 2) if (types[0] != types[1] or i < j)])
        flags, flags_retro = [], []
        for c in combos:
            mem = [f"{t}_{s}" for t, s in zip(types, c)]
            zv = -np.min([tail(bsorted[m], src[m].values) for m in mem], axis=0)
            zt = -np.min([tail(bsorted[m], tgt[m].values) for m in mem], axis=0)
            zb = np.sort(zv[ben])[::-1]
            tau = zb[int(np.floor(alpha * len(zb)))]
            flags.append(zt > tau)
            ytb = np.sort(zt[tgt.label.values == 0])[::-1]
            flags_retro.append(zt > ytb[int(np.floor(alpha * len(ytb)))])
        out[name] = (np.array(flags), np.array(flags_retro))
    return out


def interventions(src, tgt, inv, n_boot, rng, src_mask=None):
    y = tgt.label.values
    ncl = inv.max() + 1
    res = {}
    for alpha in (0.005, 0.01, 0.02):
        fl = config_flags(src, tgt, alpha, src_mask)
        mal = y == 1
        tab = {}
        for name, (f, fr) in fl.items():
            tab[name] = dict(fnr_pct=100 * (1 - f[:, mal].mean(1)).mean(), fpr_pct=100 * f[:, ~mal].mean(1).mean(),
                             fnr_retro_pct=100 * (1 - fr[:, mal].mean(1)).mean(), n_combos=int(len(f)))
        # Bootstrap of FNR (validation-calibrated) per configuration type.
        csum = {n: np.vstack([np.bincount(inv[mal], weights=f[k, mal].astype(float), minlength=ncl)
                              for k in range(len(f))]) for n, (f, _) in fl.items()}
        cmal = np.bincount(inv[mal], minlength=ncl).astype(float)
        bo = {n: [] for n in fl}
        for _ in range(n_boot):
            cnt = np.bincount(rng.integers(0, ncl, ncl), minlength=ncl).astype(float)
            denom = cnt @ cmal
            for n in fl:
                bo[n].append(100 * (1 - (csum[n] @ cnt) / denom).mean())
        bo = {n: np.array(v) for n, v in bo.items()}
        def ci(x):
            return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        best_vr = np.min(np.vstack([bo[n] for n in VIEW_RESTRICTED_PAIRS]), axis=0)
        tests = {
            "IH1 LS+LC minus LS+LS": (tab["view swap LS+LC"]["fnr_pct"] - tab["seed LS+LS"]["fnr_pct"], ci(bo["view swap LS+LC"] - bo["seed LS+LS"])),
            "IH2 LS+LS minus LS": (tab["seed LS+LS"]["fnr_pct"] - tab["single LS"]["fnr_pct"], ci(bo["seed LS+LS"] - bo["single LS"])),
            "IH2 MS+MS minus MS": (tab["seed MS+MS"]["fnr_pct"] - tab["single MS"]["fnr_pct"], ci(bo["seed MS+MS"] - bo["single MS"])),
            "IH3 FL minus best view-restricted pair": (tab["single FL"]["fnr_pct"] - min(tab[n]["fnr_pct"] for n in VIEW_RESTRICTED_PAIRS),
                                                       ci(bo["single FL"] - best_vr)),
            "IH4 MS+LS minus LS": (tab["alg swap MS+LS"]["fnr_pct"] - tab["single LS"]["fnr_pct"], ci(bo["alg swap MS+LS"] - bo["single LS"])),
        }
        verdict = {
            "IH1": tests["IH1 LS+LC minus LS+LS"][1][1] < 0,
            "IH2_LS": -1 < tests["IH2 LS+LS minus LS"][1][0] and tests["IH2 LS+LS minus LS"][1][1] < 1,
            "IH2_MS": -1 < tests["IH2 MS+MS minus MS"][1][0] and tests["IH2 MS+MS minus MS"][1][1] < 1,
            "IH3": tests["IH3 FL minus best view-restricted pair"][1][1] <= 0,
            "IH4": tests["IH4 MS+LS minus LS"][1][0] > -1,
        }
        res[str(alpha)] = {"table": tab, "tests": {k: {"diff_pp": float(v[0]), "ci95": v[1]} for k, v in tests.items()},
                           "verdict_primary_alpha_only_if_0.01": {k: bool(v) for k, v in verdict.items()}}
    return res


# ----------------------------------------------------------------------------- diagnosis
def family_decomposition(dec, y, fam):
    mal = y == 1
    f = pd.Series(fam[mal]).astype(object).fillna("(none)").values
    out = {}
    for g in ("M", "L"):
        prods = np.zeros(mal.sum())
        k = 0
        for i, j in combinations(SEEDS, 2):
            di = (~dec[f"{g}S_{i}"][mal]).astype(float) - (~dec[f"{g}C_{i}"][mal]).astype(float)
            dj = (~dec[f"{g}S_{j}"][mal]).astype(float) - (~dec[f"{g}C_{j}"][mal]).astype(float)
            prods += 0.5 * di * dj
            k += 1
        prods /= k
        tot = prods.mean()
        s = (pd.Series(prods).groupby(f).sum() / len(prods)).sort_values(ascending=False)
        out[g] = {"raw_contrast_pp": 100 * tot, "top10": {str(a): float(b / tot) for a, b in s.iloc[:10].items()},
                  "no_family_share": float(s.get("(none)", 0) / tot), "n_positive_families": int((s > 0).sum())}
    return out


def weekly(dec, y, week, inv, prefix_map):
    rows, burst = [], []
    for wk in sorted(np.unique(week)):
        m = week == wk
        nm = pool_names(prefix_map)
        mal = (y == 1) & m
        Mw = {k: (~dec[nm[k]][mal]).astype(float) for k in nm}
        pairs = cross_pairs()
        cov = np.array([(Mw[a] * Mw[b]).mean() - Mw[a].mean() * Mw[b].mean() for a, b in pairs]) * 100
        J = np.array([(Mw[a] * Mw[b]).mean() for a, b in pairs]) * 100
        e = effects_from(cov, pairs)
        same = np.mean([j for j, (a, b) in zip(J, pairs) if a[1] == b[1]])
        crossed = np.mean([j for j, (a, b) in zip(J, pairs) if a[1] != b[1]])
        rows.append(dict(week=int(wk), V=e["V"], G=e["G"], I=e["I"], R=e["R"], J_same=same, J_crossed=crossed))
    W = pd.DataFrame(rows)
    ms, mc = W.J_same.median(), W.J_crossed.median()
    W["burst"] = (W.J_same - ms > 3) & ~(W.J_crossed - mc > 3)
    return W


# ----------------------------------------------------------------------------- main
def run(tgt, out_dir: Path, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    src = load_source()
    rng = np.random.default_rng(RNG_SEED)
    y, inv = tgt.label.values, clusters(tgt)
    res = {"label": label, "n_files": int(len(tgt)), "n_malware": int((y == 1).sum()), "n_benign": int((y == 0).sum())}
    thr = {a: thresholds(src, a) for a in (0.0025, 0.005, 0.01)}
    frozen = json.loads((RES / "single_factor" / "win64" / "frozen.json").read_text())["thresholds"]
    res["mlp_threshold_check_max_abs"] = max(abs(thr[0.005][f"M{v}_{s}"] - frozen[f"{v}_{s}"]) for v in "SC" for s in SEEDS)
    dec = {a: {d: tgt[d].values > t for d, t in thr[a].items()} for a in thr}
    primary = {"MS": "MS", "MC": "MC", "LS": "LS", "LC": "LC"}
    ab2 = {"MS": "AMS", "MC": "AMC", "LS": "LS", "LC": "LC"}
    f, *_ = factorial(dec[0.005], y, inv, primary, B, rng, "primary (clean MLP + LightGBM), branch 0.5%")
    res["primary"] = f
    res["decisions"] = decide(f)
    res["go_rule"] = go_rule(res["decisions"])
    res["AB2_selection_matched"] = factorial(dec[0.005], y, inv, ab2, B, rng, "AB2 selection-matched MLP + LightGBM")[0]
    res["AB4_branch_0.25pct"] = factorial(dec[0.0025], y, inv, primary, 500, rng, "AB4 0.25%")[0]
    res["AB4_branch_1pct"] = factorial(dec[0.01], y, inv, primary, 500, rng, "AB4 1%")[0]
    # NC2 benign analogue (false-alarm dependence), point and CI.
    flip = {k: ~v for k, v in dec[0.005].items()}  # misses on benign := ~flag inverted below
    yb = np.where(y == 0, 1, 0)  # treat benign as "positives" so that miss indicator = ~(~flag) = flag
    res["NC2_benign_false_alarm_dependence"] = factorial(flip, yb, inv, primary, 500, rng, "NC2 benign false alarms")[0]
    res["interventions"] = interventions(src, tgt, inv, B, rng)
    res["AB5_interventions_source_excluding_week42"] = interventions(src, tgt, inv, 500, rng,
                                                                     src_mask=(src.week.values != 42))
    res["D1_family_decomposition"] = family_decomposition(dec[0.005], y, tgt.family.values)
    Wk = weekly(dec[0.005], y, tgt.week.values, inv, primary)
    Wk.to_csv(out_dir / "D2_D3_weekly.csv", index=False)
    res["D3_bursts"] = Wk[Wk.burst].week.tolist()
    (out_dir / "analysis_v2_results.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def main():
    if "--dry-run-on-validation" in sys.argv:
        tgt = load_source()
        res = run(tgt, POOL / "development_reproduction", "DRY RUN on weeks 41-46 (IN-SAMPLE; development data; not confirmatory)")
    elif "--target" in sys.argv:
        tgt = pd.read_parquet(sys.argv[sys.argv.index("--target") + 1])
        res = run(tgt, POOL / "confirmatory_reproduction", "CONFIRMATORY weeks 47-52")
    else:
        sys.exit("specify --dry-run-on-validation or --target <parquet> --authorised-E")
    print(json.dumps({k: res[k] for k in ("label", "go_rule", "decisions")}, indent=1, default=float))


if __name__ == "__main__":
    main()
