"""Generate manuscript figures, tables and numerical macros
from the released experimental results.

Output formats: PDF, SVG and 600-dpi PNG.

"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "results"
FIG, TAB = ROOT / "figures", ROOT / "work" / "tables"
SEEDS = (42, 123, 2026)
MM = 1 / 25.4
W = 140 * MM

# Okabe-Ito; tiers are distinguished by colour AND marker fill so that greyscale remains readable.
C_CONF, C_DEV, C_EXPL, C_THR, C_NEU = "#0072B2", "#E69F00", "0.45", "#D55E00", "0.25"
C_VIEW, C_ALG, C_INT, C_RES = "#0072B2", "#009E73", "#CC79A7", "0.35"
RC = {"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
      "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.7,
      "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 3, "ytick.major.size": 3,
      "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2, "pdf.fonttype": 42,
      "ps.fonttype": 42, "svg.fonttype": "path", "svg.hashsalt": "misdive-v4", "savefig.dpi": 600,
      "axes.titlelocation": "left", "axes.titleweight": "bold", "mathtext.fontset": "dejavusans"}


def save(fig, name):
    meta = {"pdf": {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None},
            "svg": {"Date": None, "Creator": None}, "png": {"Software": None}}
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", metadata=meta[ext])
    plt.close(fig)


def load():
    conf = json.loads((RES / "confirmatory" / "analysis_v2_results.json").read_text())
    add = json.loads((RES / "confirmatory" / "addendum_a1_results.json").read_text())
    dev = json.loads((RES / "development" / "analysis_v2_results.json").read_text())
    wk_c = pd.read_csv(RES / "confirmatory" / "D2_D3_weekly.csv")
    wk_d = pd.read_csv(RES / "development" / "D2_D3_weekly.csv")
    net = pd.read_csv(RES / "exploratory" / "phase0c" / "factorial_effects.csv")
    w42 = json.loads((RES / "exploratory" / "phase0c" / "week42_burst.json").read_text())
    pe = pd.read_csv(RES / "exploratory" / "phase0c" / "prediction_errors.csv")
    pdf_ = pd.read_csv(RES / "exploratory" / "phase0c" / "prediction_differences.csv")
    return conf, add, dev, wk_c, wk_d, net, w42, pe, pdf_


# ----------------------------------------------------------------------------- Figure 1: workflow
def fig_workflow():
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(W, 96 * MM))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 140)
        ax.set_ylim(0, 96)
        ax.axis("off")

        def box(x, y, w, h, title, body, fc, ec="0.3", lw=0.8, ls="-"):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec=ec, lw=lw, ls=ls))
            ax.text(x + 1.6, y + h - 2.2, title, fontsize=7.2, fontweight="bold", va="top", ha="left")
            ax.text(x + 1.6, y + h - 6.6, body, fontsize=6.7, va="top", ha="left", linespacing=1.25)

        def arrow(x0, y0, x1, y1):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.3", shrinkA=0, shrinkB=0))

        est, own = "#F2F2F2", "#DCEAF5"
        box(2, 74, 40, 19, "Inputs", "Detectors: factor levels x\nreplicates; frozen thresholds;\nlabelled temporal windows;\nfamily labels", "white")
        box(50, 74, 40, 19, "M1  Competence", "Miss rate $q$ and false-positive\nrate $r$ of every detector\nper window", est)
        box(98, 74, 40, 19, "M2  Dependence", "Joint miss $J$, covariance\n$J-q_aq_b$, correlation $\\varphi$\nfor all cross-replicate pairs", est)
        box(98, 46, 40, 22, "M3  Factorial attribution", "Balanced re-pairing over the\n2x2 sharing cells: view effect $V$,\nalgorithm effect $G$, interaction\n$I$, residual $R$", own, ec=C_CONF, lw=1.1)
        box(50, 46, 40, 22, "M4  Robustness", "Covariance and $\\varphi$ scales\n(scale-artefact check)\nOperating points\nCheckpoint-selection ablation\nReplicate-level permutation", own, ec=C_CONF, lw=1.1)
        box(2, 46, 40, 22, "M5  Diagnostics", "Weekly effects\nFamily decomposition of the\nview contrast\nBurst screen", est)
        box(2, 17, 40, 22, "M6  Operations", "Configurations at matched\nsystem FP budget\n(validation-calibrated and\ntarget-matched); FP drift", est)
        box(50, 17, 40, 22, "M7  Confirmation", "Hypotheses, thresholds and\ncode frozen and hashed\nOne evaluation on unexamined\ntemporal data", own, ec=C_CONF, lw=1.1)
        box(98, 17, 40, 22, "Outputs", "Effect estimates with file- and\nreplicate-level uncertainty\nPre-specified decisions\nArchitectural risk statements\n(no composite score)", "white")
        arrow(42.8, 83.5, 49.2, 83.5)
        arrow(90.8, 83.5, 97.2, 83.5)
        arrow(118, 73.2, 118, 68.8)
        arrow(97.2, 57, 90.8, 57)
        arrow(49.2, 57, 42.8, 57)
        arrow(22, 45.2, 22, 39.8)
        arrow(42.8, 28, 49.2, 28)
        arrow(90.8, 28, 97.2, 28)
        ax.add_patch(Rectangle((3, 4.5), 4, 3, fc=est, ec="0.3", lw=0.6))
        ax.text(8.5, 6, "Established technique applied as specified", fontsize=6.8, va="center")
        ax.add_patch(Rectangle((62, 4.5), 4, 3, fc=own, ec=C_CONF, lw=0.9))
        ax.text(67.5, 6, "Methodology-specific design element", fontsize=6.8, va="center")
        save(fig, "fig1_workflow")


# ----------------------------------------------------------------------------- Figure 2: design + heatmap
def target_phi_matrix():
    t = pd.read_parquet(RES / "scores" / "win64_w47_52_target_scores.parquet")
    fz = json.loads((RES / "single_factor" / "win64" / "frozen.json").read_text())["thresholds"]
    tr = json.loads((RES / "training" / "train_record.json").read_text())["models"]
    names = [f"{t_}_{s}" for t_ in ("MS", "MC", "LS", "LC") for s in SEEDS]
    thr = {n: (fz[f"{n[1]}_{n.split('_')[1]}"] if n[0] == "M" else tr[n]["threshold"]) for n in names}
    mal = t.label.values == 1
    M = {n: (t[n].values[mal] <= thr[n]).astype(float) for n in names}
    phi = np.full((12, 12), np.nan)
    cov = np.full((12, 12), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if a.split("_")[1] == b.split("_")[1]:
                continue
            qa, qb = M[a].mean(), M[b].mean()
            c = (M[a] * M[b]).mean() - qa * qb
            cov[i, j] = c
            phi[i, j] = c / np.sqrt(qa * (1 - qa) * qb * (1 - qb))
    return names, phi, cov


def fig_design_heatmap(conf):
    names, phi, cov = target_phi_matrix()
    # consistency with the frozen analysis: cell means of phi reproduce the confirmatory phi cells
    cells = {}
    for i, j in combinations(range(12), 2):
        if np.isnan(phi[i, j]):
            continue
        a, b = names[i], names[j]
        k = ("SV" if a[1] == b[1] else "XV") + ("SA" if a[0] == b[0] else "XA")
        cells.setdefault(k, []).append(phi[i, j])
    for k, v in cells.items():
        assert abs(np.mean(v) - conf["primary"]["phi"]["point"][f"mu_{k}"]) < 1e-9, k
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(W, 66 * MM))
        ax0 = fig.add_axes([0.02, 0.10, 0.36, 0.80])
        ax1 = fig.add_axes([0.47, 0.13, 0.40, 0.78])
        cax = fig.add_axes([0.885, 0.20, 0.015, 0.62])
        ax0.set_xlim(0, 10)
        ax0.set_ylim(0, 12.4)
        ax0.axis("off")
        ax0.text(-0.2, 12.4, "(a) Sharing cells of the 2x2 design", fontsize=8.5, fontweight="bold", va="top")
        lab = {("SV", "SA"): "SVSA\nsame view\nsame algorithm", ("SV", "XA"): "SVXA\nsame view\ndiff. algorithm",
               ("XV", "SA"): "XVSA\ndiff. view\nsame algorithm", ("XV", "XA"): "XVXA\ndiff. view\ndiff. algorithm"}
        shade = {"SVSA": "#9ECAE1", "SVXA": "#C6DBEF", "XVSA": "#DEEBF7", "XVXA": "#F7FBFF"}
        for r, v in enumerate(("SV", "XV")):
            for c, a in enumerate(("SA", "XA")):
                x, y = 1.2 + c * 4.3, 5.2 - r * 4.3
                ax0.add_patch(Rectangle((x, y), 4.1, 4.1, fc=shade[v + a], ec="0.3", lw=0.7))
                ax0.text(x + 2.05, y + 2.55, lab[(v, a)], ha="center", va="center", fontsize=6.6, linespacing=1.2)
                ax0.text(x + 2.05, y + 0.55, "12 pairs", ha="center", va="center", fontsize=6.4, color="0.25")
        ax0.text(3.25, 9.55, "same", ha="center", fontsize=6.8)
        ax0.text(7.55, 9.55, "different", ha="center", fontsize=6.8)
        ax0.text(5.4, 10.25, "algorithm", ha="center", fontsize=7, fontweight="bold")
        ax0.text(0.55, 7.25, "same", rotation=90, va="center", fontsize=6.8)
        ax0.text(0.55, 2.95, "different", rotation=90, va="center", fontsize=6.8)
        ax0.text(-0.2, 5.1, "view", rotation=90, va="center", fontsize=7, fontweight="bold")
        m = np.ma.masked_invalid(phi)
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("0.85")
        im = ax1.imshow(m, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ticks = [f"{n.split('_')[0]}{SEEDS.index(int(n.split('_')[1])) + 1}" for n in names]
        ax1.set_xticks(range(12))
        ax1.set_yticks(range(12))
        ax1.set_xticklabels(ticks, rotation=90, fontsize=6.6)
        ax1.set_yticklabels(ticks, fontsize=6.6)
        for b in (2.5, 5.5, 8.5):
            ax1.axhline(b, color="white", lw=1.2)
            ax1.axvline(b, color="white", lw=1.2)
        ax1.set_title("(b) Miss correlation $\\varphi$, Win64 weeks 47-52", fontsize=8.5)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=6.8)
        cb.set_label("$\\varphi$ (cross-seed pairs)", fontsize=7.2)
        ax1.tick_params(length=0)
        for s in ax1.spines.values():
            s.set_visible(False)
        save(fig, "fig2_design_heatmap")
    return names, phi


# ----------------------------------------------------------------------------- Figure 3: forest plot
def fig_forest(conf, dev):
    rows = [("V", "View effect $V$ (H1)", 2.0), ("VmG", "$V-G$ (H2)", 0.0), ("G", "Algorithm effect $G$ (H3)", 1.0),
            ("I", "Interaction $I$ (H4)", 1.0), ("R", "Residual $R$", None)]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(W, 70 * MM), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.08))
        fig.subplots_adjust(left=0.27, right=0.985, top=0.90, bottom=0.30)
        ext = list(rows) + [("H5", "Manuscript contrast (H5)", 2.0)]
        for ax, scale, title, xl in ((axes[0], "cov_pp", "(a) Covariance scale", "Effect (pp)"),
                                     (axes[1], "phi", "(b) Correlation scale", "Effect ($\\varphi$)")):
            for k, (key, label, mei) in enumerate(ext):
                y = len(ext) - 1 - k
                if key == "H5" and scale == "phi":
                    ax.text(0.01, y, "not defined (MLP-only contrast)", fontsize=6.6, color="0.45", va="center")
                    continue
                for res, off, col, mk, fill in ((dev, 0.17, C_DEV, "o", "white"), (conf, -0.17, C_CONF, "o", C_CONF)):
                    if key == "H5":
                        pt = res["balanced_contrasts_pp"]["dview_MM"] if "balanced_contrasts_pp" in res else None
                        pt = res["primary"]["balanced_contrasts_pp"]["dview_MM"]
                        lo, hi = res["primary"]["balanced_contrasts_ci95"]["dview_MM"]
                    else:
                        pt = res["primary"][scale]["point"][key]
                        lo, hi = res["primary"][scale]["ci95"][key]
                    ax.plot([lo, hi], [y + off] * 2, color=col, lw=1.3, solid_capstyle="butt")
                    ax.plot(pt, y + off, marker=mk, ms=4.6, mfc=fill, mec=col, mew=1.1, ls="none")
                if mei is not None and scale == "cov_pp":
                    ax.plot([mei, mei], [y - 0.42, y + 0.42], color=C_THR, ls=(0, (3, 1.6)), lw=1.0)
            ax.axvline(0, color=C_NEU, lw=0.6)
            ax.set_yticks(range(len(ext)))
            ax.set_yticklabels([r[1] for r in ext][::-1] if ax is axes[0] else [r[1] for r in ext][::-1])
            if ax is axes[1]:
                ax.set_yticklabels([])
            ax.set_ylim(-0.6, len(ext) - 0.4)
            ax.set_title(title, fontsize=8.5)
            ax.set_xlabel(xl)
            ax.xaxis.grid(True, color="0.92", lw=0.6)
            ax.set_axisbelow(True)
        axes[0].set_xlim(-0.3, 8.3)
        axes[1].set_xlim(-0.02, 0.5)
        h = [Line2D([], [], color=C_CONF, marker="o", mfc=C_CONF, mec=C_CONF, lw=1.3, ms=4.6),
             Line2D([], [], color=C_DEV, marker="o", mfc="white", mec=C_DEV, lw=1.3, ms=4.6),
             Line2D([], [], color=C_THR, ls=(0, (3, 1.6)), lw=1.0)]
        fig.legend(h, ["Confirmatory: Win64 weeks 47-52 (unexamined)", "Development: weeks 41-46 (in-sample)",
                       "Pre-specified minimum effect of interest"], loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 0.0), columnspacing=1.4, handlelength=2.2)
        save(fig, "fig3_forest")


# ----------------------------------------------------------------------------- Figure 4: interaction plot
def fig_interaction(conf, net):
    n2 = net[net.window.str.startswith("W2") & (net.scale == "covariance_pp")].iloc[0]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(W, 56 * MM), gridspec_kw=dict(wspace=0.45))
        fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.30)
        specs = [(axes[0], conf["primary"]["cov_pp"]["point"], "(a) Win64, covariance", "Mean covariance (pp)", C_CONF, "-"),
                 (axes[1], conf["primary"]["phi"]["point"], "(b) Win64, correlation", "Mean $\\varphi$", C_CONF, "-"),
                 (axes[2], {f"mu_{k}": n2[k] for k in ("SVSA", "SVXA", "XVSA", "XVXA")}, "(c) .NET, covariance", "Mean covariance (pp)", C_EXPL, "-")]
        for ax, p, title, yl, col, _ in specs:
            same = [p["mu_XVSA"], p["mu_SVSA"]]
            diff = [p["mu_XVXA"], p["mu_SVXA"]]
            ax.plot([0, 1], same, color=col, marker="o", ms=4.5, lw=1.4, label="same algorithm")
            ax.plot([0, 1], diff, color=col, marker="s", ms=4.2, mfc="white", lw=1.2, ls=(0, (4, 2)), label="different algorithm")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["different\nview", "same\nview"])
            ax.set_xlim(-0.3, 1.3)
            ax.set_ylim(0, max(same + diff) * 1.18)
            ax.set_title(title, fontsize=8.2)
            ax.set_ylabel(yl)
            ax.yaxis.grid(True, color="0.92", lw=0.6)
            ax.set_axisbelow(True)
        h = [Line2D([], [], color="0.2", marker="o", ms=4.5, lw=1.4), Line2D([], [], color="0.2", marker="s", mfc="white", ms=4.2, lw=1.2, ls=(0, (4, 2)))]
        fig.legend(h, ["Pairs sharing the learning algorithm", "Pairs with different learning algorithms"], loc="lower center",
                   ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
        save(fig, "fig4_interaction")


# ----------------------------------------------------------------------------- Figure 5: robustness / ablation
def fig_ablation(conf):
    variants = [("primary", "Primary (0.5%)"), ("AB2_selection_matched", "AB2 selection-matched"),
                ("AB4_branch_0.25pct", "AB4 branch 0.25%"), ("AB4_branch_1pct", "AB4 branch 1%")]
    eff = [("V", "$V$", 2.0, C_VIEW), ("G", "$G$", 1.0, C_ALG), ("I", "$I$", 1.0, C_INT)]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(W, 64 * MM), gridspec_kw=dict(wspace=0.1, width_ratios=[1.3, 1]))
        fig.subplots_adjust(left=0.24, right=0.985, top=0.9, bottom=0.25)
        for ax, scale, title, xl in ((axes[0], "cov_pp", "(a) Covariance scale", "Effect (pp)"),
                                     (axes[1], "phi", "(b) Correlation scale", "Effect ($\\varphi$)")):
            for vi, (vk, vl) in enumerate(variants):
                for ei, (ek, el, mei, col) in enumerate(eff):
                    y = (len(variants) - 1 - vi) + (1 - ei) * 0.24
                    p = conf[vk][scale]["point"][ek]
                    lo, hi = conf[vk][scale]["ci95"][ek]
                    ax.plot([lo, hi], [y, y], color=col, lw=1.3)
                    ax.plot(p, y, marker="o", ms=4.2, color=col, mfc=col if vk == "primary" else "white", mew=1.1)
            if scale == "cov_pp":
                for ek, el, mei, col in eff:
                    ax.axvline(mei, color=col, ls=(0, (3, 1.6)), lw=0.8, alpha=0.9)
            ax.axvline(0, color=C_NEU, lw=0.6)
            ax.set_yticks(range(len(variants)))
            ax.set_yticklabels([v[1] for v in variants][::-1] if ax is axes[0] else [])
            ax.set_ylim(-0.6, len(variants) - 0.4)
            ax.set_xlabel(xl)
            ax.set_title(title, fontsize=8.5)
            ax.xaxis.grid(True, color="0.92", lw=0.6)
            ax.set_axisbelow(True)
        axes[0].set_xlim(-0.2, 5.2)
        axes[1].set_xlim(-0.02, 0.42)
        h = [Line2D([], [], color=c, marker="o", lw=1.3, ms=4.2) for _, _, _, c in eff] + [Line2D([], [], color="0.3", ls=(0, (3, 1.6)), lw=0.8)]
        fig.legend(h, ["View effect $V$", "Algorithm effect $G$", "Interaction $I$", "MEI (colour-matched)"],
                   loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=0.9, handlelength=1.8, fontsize=7)
        save(fig, "fig5_ablation")


# ----------------------------------------------------------------------------- Figure 6: operational
OP_ORDER = [("single FL", "FL", "single"), ("seed FL+FL", "FL+FL", "seed"),
            ("single LS", "LS", "single"), ("seed LS+LS", "LS+LS", "seed"), ("view swap LS+LC", "LS+LC", "view"),
            ("both LS+MC", "LS+MC", "both"), ("alg swap MS+LS", "MS+LS", "alg"), ("both MS+LC", "MS+LC", "both"),
            ("single MS", "MS", "single"), ("seed MS+MS", "MS+MS", "seed"), ("view swap MS+MC", "MS+MC", "view"),
            ("single LC", "LC", "single"), ("seed LC+LC", "LC+LC", "seed"), ("alg swap MC+LC", "MC+LC", "alg"),
            ("single MC", "MC", "single"), ("seed MC+MC", "MC+MC", "seed")]
CAT = {"single": ("Single detector", "0.35", "o"), "seed": ("Seed redundancy", "#E69F00", "s"),
       "alg": ("Algorithm swap", "#009E73", "^"), "view": ("View swap", "#0072B2", "D"), "both": ("Both swapped", "#CC79A7", "v")}


def fig_operational(conf):
    tab = conf["interventions"]["0.01"]["table"]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(W, 100 * MM), gridspec_kw=dict(width_ratios=[1.05, 1], wspace=0.35))
        fig.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=0.27)
        ax = axes[0]
        for k, (key, lab, cat) in enumerate(OP_ORDER):
            y = len(OP_ORDER) - 1 - k
            n, col, mk = CAT[cat]
            ax.plot(tab[key]["fnr_pct"], y, marker=mk, color=col, ms=4.6, ls="none")
            ax.plot(tab[key]["fnr_retro_pct"], y, marker=mk, mfc="white", mec=col, ms=4.6, ls="none", mew=0.9)
            ax.plot([tab[key]["fnr_pct"], tab[key]["fnr_retro_pct"]], [y, y], color=col, lw=0.6, alpha=0.6)
        ax.set_yticks(range(len(OP_ORDER)))
        ax.set_yticklabels([o[1] for o in OP_ORDER][::-1], fontsize=7)
        ax.set_xlabel("System miss rate (%)")
        ax.set_title("(a) Configurations", fontsize=8.5)
        ax.xaxis.grid(True, color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 12.5)
        ax = axes[1]
        tests = [("IH1 LS+LC minus LS+LS", "IH1"), ("IH2 LS+LS minus LS", "IH2 (LS)"),
                 ("IH2 MS+MS minus MS", "IH2 (MS)"), ("IH3 FL minus best view-restricted pair", "IH3"),
                 ("IH4 MS+LS minus LS", "IH4")]
        for k, (key, lab) in enumerate(tests):
            for a, off, col, fill in (("0.005", 0.24, "0.6", "white"), ("0.01", 0.0, C_CONF, C_CONF), ("0.02", -0.24, "0.6", "0.6")):
                t = conf["interventions"][a]["tests"][key]
                y = len(tests) - 1 - k + off
                ax.plot(t["ci95"], [y, y], color=col, lw=1.2)
                ax.plot(t["diff_pp"], y, marker="o", ms=4, mfc=fill, mec=col, color=col, ls="none")
        ax.axvline(0, color=C_NEU, lw=0.6)
        for v in (-1, 1):
            ax.axvline(v, color=C_THR, ls=(0, (3, 1.6)), lw=0.8)
        ax.set_yticks(range(len(tests)))
        ax.set_yticklabels([t[1] for t in tests][::-1], fontsize=7)
        ax.set_xlabel("Difference in system miss rate (pp)")
        ax.set_title("(b) Pre-specified tests", fontsize=8.5)
        ax.set_xlim(-2.6, 2.4)
        ax.xaxis.grid(True, color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        h = [Line2D([], [], marker=CAT[c][2], color=CAT[c][1], ls="none", ms=4.6) for c in CAT]
        h += [Line2D([], [], marker="o", color="0.3", ls="none", ms=4.6), Line2D([], [], marker="o", mfc="white", mec="0.3", ls="none", ms=4.6)]
        lab = [CAT[c][0] for c in CAT] + ["Validation-calibrated threshold", "Target-matched (retrospective)"]
        h += [Line2D([], [], color=C_CONF, marker="o", lw=1.2, ms=4), Line2D([], [], color="0.6", marker="o", mfc="white", lw=1.2, ms=4),
              Line2D([], [], color="0.6", marker="o", lw=1.2, ms=4), Line2D([], [], color=C_THR, ls=(0, (3, 1.6)), lw=0.8)]
        lab += ["Budget 1% (primary)", "Budget 0.5%", "Budget 2%", "Equivalence margin (IH2, IH4)"]
        fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0,
                   handletextpad=0.3, fontsize=6.7)
        save(fig, "fig6_operational")


# ----------------------------------------------------------------------------- Figure 7: temporal + family
def fig_temporal_family(conf, wk_c, wk_d, w42):
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(W, 108 * MM))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.25, 1], hspace=0.62, wspace=0.42,
                              left=0.09, right=0.985, top=0.94, bottom=0.12)
        ax = fig.add_subplot(gs[0, :])
        wk = pd.concat([wk_d.assign(tier="dev"), wk_c.assign(tier="conf")])
        ax.axvspan(40.5, 46.5, color="#FFF3D6", lw=0)
        ax.text(43.5, 0.2, "Development (in-sample)", ha="center", fontsize=6.8, color="0.3")
        ax.text(49.5, 0.2, "Confirmatory (unexamined)", ha="center", fontsize=6.8, color="0.3")
        for key, lab, col, mk in (("V", "View effect $V$", C_VIEW, "o"), ("I", "Interaction $I$", C_INT, "^"),
                                  ("G", "Algorithm effect $G$", C_ALG, "s"), ("R", "Residual $R$", C_RES, "D")):
            for tier in ("dev", "conf"):
                s = wk[wk.tier == tier]
                ax.plot(s.week, s[key], color=col, marker=mk, ms=3.6, lw=1.1, mfc=col if tier == "conf" else "white")
            ax.plot([], [], color=col, marker=mk, ms=3.6, lw=1.1, label=lab)
        ax.axvline(46.5, color="0.5", lw=0.6)
        ax.set_xticks(range(41, 53))
        ax.set_xlim(40.5, 52.5)
        ax.set_ylim(0, 4.0)
        ax.set_xlabel("Week")
        ax.set_ylabel("Effect (pp)")
        ax.set_title("(a) Weekly attribution, Win64", fontsize=8.5)
        ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=7)
        ax.yaxis.grid(True, color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        ax = fig.add_subplot(gs[1, 0])
        mr = pd.DataFrame(w42["miss_rate_pct_by_week"])
        mr.index = mr.index.astype(int)
        for s, mk in zip(SEEDS, ("o", "s", "D")):
            ax.plot(mr.index, mr[f"mS{s}"], color=C_VIEW, marker=mk, ms=3.4, lw=1.0)
            ax.plot(mr.index, mr[f"mC{s}"], color=C_DEV, marker=mk, ms=3.4, lw=1.0, mfc="white")
        ax.set_xticks(range(41, 47))
        ax.set_xlabel("Validation week")
        ax.set_ylabel("Branch miss rate (%)")
        ax.set_title("(b) Week-42 burst (MLP branches)", fontsize=8.5)
        ax.annotate("one family: all three structural\nseeds miss 90.1% of its files", xy=(42.05, 17.4), xytext=(42.35, 18.3), fontsize=6.3, va="bottom",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))
        ax.set_ylim(0, 25)
        ax.legend([Line2D([], [], color=C_VIEW, lw=1), Line2D([], [], color=C_DEV, lw=1, marker="o", mfc="white", ms=3.4)],
                  ["Structural (3 seeds)", "Content (3 seeds)"], frameon=False, fontsize=6.6, loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.5)
        ax.yaxis.grid(True, color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        ax = fig.add_subplot(gs[1, 1])
        d1 = conf["D1_family_decomposition"]
        fams = ["(none)", "wacatac", "zeus", "expiro", "bedep"]
        y = np.arange(len(fams))
        for off, g, col, lab in ((-0.19, "M", C_VIEW, "MLP"), (0.19, "L", C_ALG, "LightGBM")):
            vals = [100 * d1[g]["top10"].get(f, 0) for f in fams]
            ax.barh(y + off, vals, height=0.36, color=col, label=lab)
        ax.set_yticks(y)
        ax.set_yticklabels(["no family"] + fams[1:], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Share of raw view contrast (%)")
        ax.set_title("(c) Family decomposition", fontsize=8.5)
        ax.legend(frameon=False, fontsize=6.6, loc="lower right")
        ax.xaxis.grid(True, color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        save(fig, "fig7_temporal_family")


# ----------------------------------------------------------------------------- tables and macros
def f(x, d=2):
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def ci(c, d=2):
    return f"[{f(c[0], d)}, {f(c[1], d)}]"


def tables_and_macros(conf, add, dev, net, pe, pdf_):
    P, Pphi = conf["primary"]["cov_pp"], conf["primary"]["phi"]
    perm = conf["primary"]["permutation"]
    bc, bcci = conf["primary"]["balanced_contrasts_pp"], conf["primary"]["balanced_contrasts_ci95"]
    dec, cls = conf["decisions"], add["classification"]
    # exact preservation of the confirmed values requested for V4
    assert (round(P["point"]["V"], 2), round(P["point"]["VmG"], 2), round(P["point"]["G"], 2), round(P["point"]["I"], 2)) == (2.55, 1.63, 0.92, 1.41)
    assert round(bc["dview_MM"], 2) == 3.41
    A2 = conf["AB2_selection_matched"]["cov_pp"]["point"]
    assert (round(A2["V"], 2), round(A2["G"], 2), round(A2["I"], 2)) == (2.58, 0.83, 1.18)
    assert cls["outcome_class"] == "FULL SUCCESS" and not dec["H3_algorithm_main"]["cov_supported"]
    mac = {}
    def m(name, val):
        mac[name] = val
    for k, nm in (("V", "V"), ("VmG", "VmG"), ("G", "G"), ("I", "I"), ("R", "R")):
        m(f"conf{nm}", f(P["point"][k])); m(f"conf{nm}Lo", f(P["ci95"][k][0])); m(f"conf{nm}Hi", f(P["ci95"][k][1]))
        m(f"conf{nm}Phi", f(Pphi["point"][k], 3)); m(f"conf{nm}PhiLo", f(Pphi["ci95"][k][0], 3)); m(f"conf{nm}PhiHi", f(Pphi["ci95"][k][1], 3))
        if k != "R":
            m(f"conf{nm}P", f(perm["cov_pp"][k], 3))
        for tag, key in (("AbTwo", "AB2_selection_matched"), ("AbFourLo", "AB4_branch_0.25pct"), ("AbFourHi", "AB4_branch_1pct"), ("NcTwo", "NC2_benign_false_alarm_dependence")):
            m(f"{tag}{nm}", f(conf[key]["cov_pp"]["point"][k])); m(f"{tag}{nm}CI", ci(conf[key]["cov_pp"]["ci95"][k]))
            m(f"{tag}{nm}Phi", f(conf[key]["phi"]["point"][k], 3))
        m(f"dev{nm}", f(dev["primary"]["cov_pp"]["point"][k])); m(f"devAbTwo{nm}", f(dev["AB2_selection_matched"]["cov_pp"]["point"][k]))
    for k in ("SVSA", "SVXA", "XVSA", "XVXA"):
        m(f"cell{k}", f(P["point"][f"mu_{k}"])); m(f"cell{k}Phi", f(Pphi["point"][f"mu_{k}"], 3))
    for k, v in bc.items():
        nm = k.replace("_", "").replace("dview", "dView").replace("dalg", "dAlg")
        m(nm, f(v)); m(nm + "CI", ci(bcci[k]))
    m("confPermIPhi", f(perm["phi"]["I"], 3)); m("abTwoPermI", f(conf["AB2_selection_matched"]["permutation"]["cov_pp"]["I"], 3))
    m("confNmal", f"{conf['n_malware']:,}".replace(",", "{,}")); m("confNben", f"{conf['n_benign']:,}".replace(",", "{,}"))
    m("confClusters", f"{conf['primary']['n_clusters']:,}".replace(",", "{,}"))
    # .NET exploratory (examined weeks 53-64)
    n2 = net[net.window.str.startswith("W2") & (net.scale == "covariance_pp")].iloc[0]
    for k in ("view_main", "alg_main", "interaction", "XVXA"):
        m("net" + {"view_main": "V", "alg_main": "G", "interaction": "I", "XVXA": "R"}[k], f(n2[k]))
    # operational
    iv = conf["interventions"]["0.01"]
    for key, nm in (("single FL", "FL"), ("view swap LS+LC", "LSLC"), ("seed LS+LS", "LSLS"), ("single LS", "LS"), ("seed FL+FL", "FLFL"),
                    ("single MS", "MS"), ("seed MS+MS", "MSMS"), ("alg swap MS+LS", "MSLS")):
        m(f"op{nm}", f(iv["table"][key]["fnr_pct"])); m(f"op{nm}Retro", f(iv["table"][key]["fnr_retro_pct"])); m(f"op{nm}Fpr", f(iv["table"][key]["fpr_pct"]))
    fprs = [v["fpr_pct"] for v in iv["table"].values()]
    m("opFprMin", f(min(fprs))); m("opFprMax", f(max(fprs)))
    for key, nm in (("IH1 LS+LC minus LS+LS", "IHone"), ("IH2 LS+LS minus LS", "IHtwoLS"), ("IH2 MS+MS minus MS", "IHtwoMS"),
                    ("IH3 FL minus best view-restricted pair", "IHthree"), ("IH4 MS+LS minus LS", "IHfour")):
        for a, tag in (("0.005", "Half"), ("0.01", ""), ("0.02", "Two")):
            t = conf["interventions"][a]["tests"][key]
            m(f"{nm}{tag}", f(t["diff_pp"])); m(f"{nm}{tag}CI", ci(t["ci95"]))
    ab5 = conf["AB5_interventions_source_excluding_week42"]["0.01"]["tests"]["IH1 LS+LC minus LS+LS"]
    m("IHoneAbFive", f(ab5["diff_pp"])); m("IHoneAbFiveCI", ci(ab5["ci95"]))
    sc3 = add["S_C3_seed_level_sensitivity"]
    m("scThree", f(sc3["0.01"]["statistic_pp"])); m("scThreeCI", ci(sc3["0.01"]["ci95"]))
    m("scThreeTwo", f(sc3["0.02"]["statistic_pp"])); m("scThreeTwoCI", ci(sc3["0.02"]["ci95"]))
    d1 = conf["D1_family_decomposition"]
    m("famNoneM", f(100 * d1["M"]["no_family_share"], 0)); m("famNoneL", f(100 * d1["L"]["no_family_share"], 0))
    m("famPosM", str(d1["M"]["n_positive_families"])); m("famPosL", str(d1["L"]["n_positive_families"]))
    # Phase 0C prediction (examined data, exploratory)
    def mae(pool, design, pred):
        r = pe[(pe.pool == pool) & (pe.design == design) & (pe.predictor == pred)].iloc[0]
        return r.MAE
    for des, tag in (("V41-46->W47-52", "A"), ("V41-46->W53-64", "B")):
        m(f"predGrp{tag}", f(mae("P0NET", des, "H-GRP[L2]"))); m(f"predPair{tag}", f(mae("P0NET", des, "H-PAIR")))
        m(f"predInd{tag}", f(mae("P0NET", des, "H-IND"))); m(f"predCat{tag}", f(mae("P0NET", des, "H-CATJ[L2]")))
    lines = ["% Generated by scripts/make_v4_assets.py from the verified result files; do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(mac.items())]
    (TAB / "macros_v4.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Table: confirmatory hypotheses
    def outc(d, key):
        return d
    rows = [
        ("H1", "View main effect $V$", "2 pp; $\\varphi>0$", P, Pphi, "V", "Supported (both scales)"),
        ("H2", "$V-G$", "$>0$; $\\varphi>0$", P, Pphi, "VmG", "Supported (both scales)"),
        ("H3", "Algorithm main effect $G$", "1 pp", P, Pphi, "G", "Not supported (below MEI)"),
        ("H4", "Interaction $I$", "1 pp", P, Pphi, "I", "Supported (file level)"),
    ]
    assert dec["H1_view_main"]["cov_supported"] and dec["H1_view_main"]["phi_sign"] and dec["H2_ordering"]["cov_sign"] and dec["H2_ordering"]["phi_sign"]
    assert dec["H4_interaction"]["cov_supported"] and dec["H5_manuscript_dview_MM"]["supported"]
    tl = []
    for hid, lab, mei, c, p, k, o in rows:
        pp = perm["cov_pp"][k]
        tl.append(f"{hid} & {lab} & {mei} & {f(c['point'][k])} & {ci(c['ci95'][k])} & {f(p['point'][k], 3)} & {f(pp, 3)} & {o} \\\\")
    tl.append(f"-- & Residual $R$ & -- & {f(P['point']['R'])} & {ci(P['ci95']['R'])} & {f(Pphi['point']['R'], 3)} & -- & Reported \\\\")
    h5 = dec["H5_manuscript_dview_MM"]
    tl.append(f"H5 & Manuscript contrast $\\Delta_{{\\mathrm{{view}}}}(\\mathrm{{MM}})$ & 2 pp & {f(h5['estimate'])} & {ci(h5['ci95'])} & -- & -- & Supported \\\\")
    (TAB / "v4_confirmatory.tex").write_text("\n".join(tl) + "\n", encoding="utf-8")

    # Table: ablations and controls
    al = []
    for key, lab in (("primary", "Primary, branch 0.5\\%"), ("AB2_selection_matched", "AB2 selection-matched MLPs"),
                     ("AB4_branch_0.25pct", "AB4 branch 0.25\\%"), ("AB4_branch_1pct", "AB4 branch 1\\%"),
                     ("NC2_benign_false_alarm_dependence", "NC2 benign false alarms")):
        c, p = conf[key]["cov_pp"], conf[key]["phi"]
        al.append(f"{lab} & {f(c['point']['V'])} {ci(c['ci95']['V'])} & {f(c['point']['G'])} {ci(c['ci95']['G'])} & "
                  f"{f(c['point']['I'])} {ci(c['ci95']['I'])} & {f(p['point']['V'], 2)} / {f(p['point']['G'], 2)} / {f(p['point']['I'], 2)} & "
                  f"{f(conf[key]['permutation']['cov_pp']['V'], 3)} / {f(conf[key]['permutation']['cov_pp']['G'], 3)} / {f(conf[key]['permutation']['cov_pp']['I'], 3)} \\\\")
    (TAB / "v4_ablation.tex").write_text("\n".join(al) + "\n", encoding="utf-8")

    # Table: operational at 1%
    ol = []
    for key, lab, cat in OP_ORDER:
        r = iv["table"][key]
        ol.append(f"{lab} & {CAT[cat][0]} & {f(r['fnr_pct'])} & {f(r['fpr_pct'])} & {f(r['fnr_retro_pct'])} \\\\")
    (TAB / "v4_operational.tex").write_text("\n".join(ol) + "\n", encoding="utf-8")
    il = []
    for key, lab, rule, vk in (("IH1 LS+LC minus LS+LS", "IH1: LS+LC $-$ LS+LS", "upper $<0$", "IH1"),
                               ("IH2 LS+LS minus LS", "IH2: LS+LS $-$ mean single LS", "within $\\pm1$", "IH2_LS"),
                               ("IH2 MS+MS minus MS", "IH2: MS+MS $-$ mean single MS", "within $\\pm1$", "IH2_MS"),
                               ("IH3 FL minus best view-restricted pair", "IH3: FL $-$ best view-restricted pair", "upper $\\le 0$", "IH3"),
                               ("IH4 MS+LS minus LS", "IH4: MS+LS $-$ LS", "lower $>-1$", "IH4")):
        cells = []
        for a in ("0.005", "0.01", "0.02"):
            t = conf["interventions"][a]["tests"][key]
            cells.append(f"{f(t['diff_pp'])} {ci(t['ci95'])}")
        verdict = "Pass" if iv["verdict_primary_alpha_only_if_0.01"][vk] else "Fail"
        il.append(f"{lab} & {rule} & {cells[0]} & \\textbf{{{cells[1]}}} & {cells[2]} & {verdict} \\\\")
    (TAB / "v4_ih.tex").write_text("\n".join(il) + "\n", encoding="utf-8")
    return mac


def pool_table():
    t = pd.read_parquet(RES / "scores" / "win64_w47_52_target_scores.parquet")
    y = t.label.values
    fz = json.loads((RES / "single_factor" / "win64" / "frozen.json").read_text())["thresholds"]
    tr = json.loads((RES / "training" / "train_record.json").read_text())["models"]
    ab = json.loads((RES / "training" / "ab2_train_record.json").read_text())["seeds"]
    da = {r["detector"]: r for r in json.loads((RES / "training" / "dev_audit.json").read_text())["competence_calibration"]}
    spec = [("MS", "MLP", "structural", "1{,}843", "joint OR TPR"), ("MC", "MLP", "content", "695", "joint OR TPR"),
            ("LS", "LightGBM", "structural", "1{,}843", "log-loss, round cap"), ("LC", "LightGBM", "content", "695", "log-loss, round cap"),
            ("FL", "LightGBM", "full (comp.)", "2{,}538", "log-loss, round cap"),
            ("AMS", "MLP (AB2)", "structural", "1{,}843", "own TPR at 0.5\\% FPR"), ("AMC", "MLP (AB2)", "content", "695", "own TPR at 0.5\\% FPR")]
    rows = []
    for k, alg, view, cols, sel in spec:
        fnr_t, fpr_t, fnr_v = [], [], []
        for s in SEEDS:
            d = f"{k}_{s}"
            th = fz[f"{k[1]}_{s}"] if k in ("MS", "MC") else (ab[str(s)][d]["threshold"] if k.startswith("AM") else tr[d]["threshold"])
            fl = t[d].values > th
            fnr_t.append(100 * (~fl[y == 1]).mean())
            fpr_t.append(100 * fl[y == 0].mean())
            fnr_v.append(da[d]["val_fnr_pct"])
        rows.append(f"{k} & {alg} & {view} & {cols} & {sel} & {f(np.mean(fnr_v))} & {f(np.mean(fnr_t))} & {f(np.mean(fpr_t))} \\\\")
    (TAB / "v4_pool.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    conf, add, dev, wk_c, wk_d, net, w42, pe, pdf_ = load()
    fig_workflow()
    fig_design_heatmap(conf)
    fig_forest(conf, dev)
    fig_interaction(conf, net)
    fig_ablation(conf)
    fig_operational(conf)
    fig_temporal_family(conf, wk_c, wk_d, w42)
    mac = tables_and_macros(conf, add, dev, net, pe, pdf_)
    pool_table()
    src = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
           (RES / "confirmatory" / "analysis_v2_results.json", RES / "confirmatory" / "addendum_a1_results.json")}
    (TAB / "v4_sources.json").write_text(json.dumps(src, indent=1))
    print(len(mac), "macros; sources", src)


if __name__ == "__main__":
    main()
