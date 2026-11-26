"""Write the manuscript's supplementary table and macros from results/supplementary/summary.csv only.

    .venv/Scripts/python.exe supplementary/make_tables.py
Outputs: manuscript/tables/supplementary.tex, manuscript/tables/supp_macros.tex
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S = pd.read_csv(ROOT / "results" / "supplementary" / "summary.csv")
TAB = ROOT / "manuscript" / "tables"
W1, W2 = "W1 .NET weeks 47-52 (Phase 0 dev-test)", "W2 .NET weeks 53-64 (official test)"
CN, CW = "Clean .NET weeks 53-64 (official test)", "Clean Win64 weeks 53-64 (official test)"
FROZEN = "frozen branch thresholds (0.5%)"
RETRO = "equal system FP, retrospective (test benign), r = {}%"
PROSP = "equal system FP, prospective (validation benign), r = {}%"


def row(window: str, models: str, op: str) -> pd.Series:
    m = S[(S.window == window) & (S.models == models) & (S.operating_point == op)]
    assert len(m) == 1, (window, models, op)
    return m.iloc[0]


def ci(r: pd.Series, k: str) -> str:
    return f"{r[f'{k}_pp']:.2f} [{r[f'{k}_lo']:.2f}, {r[f'{k}_hi']:.2f}]"


def sc(r: pd.Series, a: str, b: str, fmt: str = ".2f") -> str:
    return f"{r[a]:{fmt}} / {r[b]:{fmt}}"


# Table: controls (preceding-project .NET models, frozen branch thresholds at 0.5% validation FPR).
ctrl = []
models = [("MLP, structural/content views", "MLP semantic (E04)"),
          ("LightGBM, structural/content views", "LightGBM semantic views (E10)"),
          ("MLP, random partition", "MLP random partition (E05)")]
for title, w in (("Weeks 47--52 (development test of the preceding project)", W1),
                 ("Weeks 53--64 (official test; saved models rescored)", W2)):
    ctrl.append(r"\multicolumn{6}{l}{\emph{" + title + r"}} \\")
    for name, m in models:
        r = row(w, m, FROZEN)
        ctrl.append(f"{name} & {sc(r, 'J_same_pct', 'J_crossed_pct')} & {sc(r, 'excess_same_pp', 'excess_crossed_pp')} & "
                    f"{sc(r, 'sysFPR_same_pct', 'sysFPR_crossed_pct')} & {ci(r, 'delta_excess')} & {r.weeks_delta_positive} \\\\")
    for name, m in (("Paired difference: MLP views $-$ random partition", "MLP semantic (E04) minus MLP random partition (E05)"),
                    ("Paired difference: MLP views $-$ LightGBM views", "MLP semantic (E04) minus LightGBM semantic views (E10)")):
        r = row(w, m, FROZEN)
        ctrl.append(f"{name} & & & & {ci(r, 'delta_excess')} & \\\\")
(TAB / "supp_controls.tex").write_text("\n".join(ctrl) + "\n")

# Table: operating points of the manuscript pairings (clean run, weeks 53-64).
opt = []
for label, w in ((".NET", CN), ("Win64", CW)):
    opt.append(r"\multicolumn{6}{l}{\emph{" + label + r"}} \\")
    r = row(w, "MLP semantic (clean run)", FROZEN)
    opt.append(f"Frozen branch thresholds, 0.5\\% each (primary) & {sc(r, 'sysFPR_same_pct', 'sysFPR_crossed_pct')} & "
               f"{r.q_branch_mean_pct:.1f} / {r.q_branch_mean_pct:.1f} & {sc(r, 'J_same_pct', 'J_crossed_pct')} & "
               f"{ci(r, 'delta_J')} & {ci(r, 'delta_excess')} \\\\")
    for kind, pat, tag in (("Validation-selected system threshold", PROSP, "validation"),
                           ("Test-matched system FP (retrospective)", RETRO, "test")):
        for b in (0.5, 1, 2):
            r = row(w, "MLP semantic (clean run)", pat.format(f"{b:g}"))
            opt.append(f"{kind}, {b:g}\\% & {sc(r, 'sysFPR_same_pct', 'sysFPR_crossed_pct')} & "
                       f"{sc(r, 'q_same_mean_pct', 'q_crossed_mean_pct', '.1f')} & {sc(r, 'J_same_pct', 'J_crossed_pct')} & "
                       f"{ci(r, 'delta_J')} & {ci(r, 'delta_excess')} \\\\")
(TAB / "supp_operating.tex").write_text("\n".join(opt) + "\n")
old = TAB / "supplementary.tex"
if old.exists():
    old.unlink()
lines = ctrl + opt

mac = {}


def put(name: str, r: pd.Series, k: str) -> None:
    mac[name] = f"{r[f'{k}_pp']:.2f}"
    mac[name + "Lo"] = f"{r[f'{k}_lo']:.2f}"
    mac[name + "Hi"] = f"{r[f'{k}_hi']:.2f}"


put("suppLgbA", row(W1, "LightGBM semantic views (E10)", FROZEN), "delta_excess")
put("suppLgbB", row(W2, "LightGBM semantic views (E10)", FROZEN), "delta_excess")
put("suppRndA", row(W1, "MLP random partition (E05)", FROZEN), "delta_excess")
put("suppRndB", row(W2, "MLP random partition (E05)", FROZEN), "delta_excess")
put("suppSemB", row(W2, "MLP semantic (E04)", FROZEN), "delta_excess")
put("suppSemMinusRndA", row(W1, "MLP semantic (E04) minus MLP random partition (E05)", FROZEN), "delta_excess")
put("suppSemMinusRndB", row(W2, "MLP semantic (E04) minus MLP random partition (E05)", FROZEN), "delta_excess")
put("suppSemMinusLgbB", row(W2, "MLP semantic (E04) minus LightGBM semantic views (E10)", FROZEN), "delta_excess")
for fmt, w in (("Net", CN), ("Win", CW)):
    for tag, r in (("Half", 0.5), ("One", 1), ("Two", 2)):
        put(f"suppEqJ{tag}{fmt}", row(w, "MLP semantic (clean run)", RETRO.format(f"{r:g}")), "delta_J")
        put(f"suppProJ{tag}{fmt}", row(w, "MLP semantic (clean run)", PROSP.format(f"{r:g}")), "delta_J")
    one = row(w, "MLP semantic (clean run)", RETRO.format(1))
    mac[f"suppEqTarget{fmt}"] = f"{int(one.sysFP_target):,}".replace(",", "{,}")
    mac[f"suppEqQsame{fmt}"] = f"{one.q_same_mean_pct:.1f}"
    mac[f"suppEqQcross{fmt}"] = f"{one.q_crossed_mean_pct:.1f}"
    mac[f"suppEqExc{fmt}"] = f"{one.delta_excess_pp:.2f}"
for k, (w, m) in {"suppJcrossSemB": (W2, "MLP semantic (E04)"), "suppJcrossRndB": (W2, "MLP random partition (E05)")}.items():
    mac[k] = f"{row(w, m, FROZEN).J_crossed_pct:.1f}"
# Structural-anchor comparison (analysis D): crossed-view pairs versus two structural detectors.
A = pd.read_csv(ROOT / "results" / "supplementary" / "anchor_summary.csv")
AP = pd.read_csv(ROOT / "results" / "supplementary" / "anchor_seedpairs.csv")
anc = []
for label, fmt in ((".NET", "Net"), ("Win64", "Win")):
    anc.append(r"\multicolumn{7}{l}{\emph{" + label + r"}} \\")
    for op, text in (("retrospective (test-matched FP), 1%", "Test-matched FP, 1\\% (primary)"),
                     ("retrospective (test-matched FP), 0.5%", "Test-matched FP, 0.5\\%"),
                     ("retrospective (test-matched FP), 2%", "Test-matched FP, 2\\%"),
                     ("validation-selected, 0.5%", "Validation-selected, 0.5\\%"),
                     ("validation-selected, 1%", "Validation-selected, 1\\%"),
                     ("validation-selected, 2%", "Validation-selected, 2\\%")):
        r = A[(A.file_type == label) & (A.operating_point == op)].iloc[0]
        anc.append(f"{text} & {r.fpr_SS_pct:.2f} / {r.fpr_crossed_pct:.2f} & {r.fnr_SS_pct:.2f} & {r.fnr_crossed_pct:.2f} & "
                   f"{r.fnr_CC_pct:.2f} & {r.delta_fnr_pp:.2f} [{r.delta_fnr_lo:.2f}, {r.delta_fnr_hi:.2f}] & "
                   f"{r.gain_files_mean:.0f} / {r.loss_files_mean:.0f} \\\\")
    p = A[(A.file_type == label) & (A.operating_point == "retrospective (test-matched FP), 1%")].iloc[0]
    sp = AP[(AP.file_type == label) & (AP.operating_point == "retrospective (test-matched FP), 1%")]
    for k, col in (("SS", "fnr_SS_pct"), ("Cross", "fnr_crossed_pct"), ("CC", "fnr_CC_pct")):
        mac[f"anc{k}{fmt}"] = f"{p[col]:.2f}"
    mac[f"ancD{fmt}"] = f"{p.delta_fnr_pp:.2f}"
    mac[f"ancD{fmt}Abs"] = f"{abs(p.delta_fnr_pp):.2f}"
    mac[f"ancD{fmt}Lo"] = f"{p.delta_fnr_lo:.2f}"
    mac[f"ancD{fmt}Hi"] = f"{p.delta_fnr_hi:.2f}"
    mac[f"ancDmin{fmt}"] = f"{sp.delta_fnr_pp.min():.2f}"
    mac[f"ancDmax{fmt}"] = f"{sp.delta_fnr_pp.max():.2f}"
    mac[f"ancGain{fmt}"] = f"{p.gain_files_mean:.0f}"
    mac[f"ancLoss{fmt}"] = f"{p.loss_files_mean:.0f}"
    mac[f"ancNet{fmt}"] = f"{p.net_files_mean:.0f}"
    mac[f"ancGainU{fmt}"] = f"{int(p.gain_unique_files):,}".replace(",", "{,}")
    mac[f"ancLossU{fmt}"] = f"{int(p.loss_unique_files):,}".replace(",", "{,}")
    mac[f"ancFP{fmt}"] = f"{int(p.fp_min):,}".replace(",", "{,}")
    mac[f"ancCCd{fmt}"] = f"{p.delta_vs_CC_pp:.2f}"
    v = A[(A.file_type == label) & (A.operating_point == "validation-selected, 1%")].iloc[0]
    mac[f"ancVal{fmt}"] = f"{v.delta_fnr_pp:.2f}"
    mac[f"ancVal{fmt}Lo"] = f"{v.delta_fnr_lo:.2f}"
    mac[f"ancVal{fmt}Hi"] = f"{v.delta_fnr_hi:.2f}"
    mac[f"ancValFprSS{fmt}"] = f"{v.fpr_SS_pct:.2f}"
    mac[f"ancValFprCr{fmt}"] = f"{v.fpr_crossed_pct:.2f}"
import re  # noqa: E402

MINUS = re.compile(r"(?<![\w.\-])-(?=\d)")  # a leading minus sign, not a range dash
anc = [MINUS.sub("$-$", s) for s in anc]
mac = {k: (MINUS.sub("$-$", v) if k.startswith("anc") else v) for k, v in mac.items()}
(TAB / "supp_anchor.tex").write_text("\n".join(anc) + "\n")
(TAB / "supp_macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
print(f"{len(lines)} table lines, {len(mac)} macros")
