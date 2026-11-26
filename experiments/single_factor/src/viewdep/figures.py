"""Manuscript figures, generated only from files in results/, with one shared style.

Figures are drawn at their final printed width (140 mm, the Elsevier 1.5-column width, which matches the
LaTeX text width of 13.7 cm), so that the font sizes set here are the sizes that appear in print:
9 pt for axis titles and legends, 8 pt for tick labels and panel annotations.

Exports per figure: vector PDF with embedded TrueType fonts (LaTeX), SVG with text converted to paths
(Word), and a 600-dpi PNG fallback (Word versions without SVG support). Output is deterministic:
fixed style, fixed jitter offsets and no timestamps in the files.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .paths import RESULTS, ROOT  # noqa: E402

FIG_DIR = ROOT / "figures_workspace"
WIDTH_MM = 140.0
MM = 1 / 25.4
LABEL = {"dotnet": ".NET", "win64": "Win64"}

# Okabe-Ito colours (colour-blind safe); each configuration also has its own marker so that the
# figures remain readable in greyscale. The same mapping is used in every figure.
STYLE = {
    "SS": dict(color="#0072B2", marker="o", label="Structural + structural (SS)"),
    "CC": dict(color="#E69F00", marker="s", label="Content + content (CC)"),
    "X": dict(color="#009E73", marker="^", label="Crossed views (SC/CS)"),
}
REF = dict(independence=dict(color="black", ls=(0, (1, 1.5)), lw=1.0),
           effect=dict(color="#D55E00", ls=(0, (5, 2.5)), lw=1.1),
           zero=dict(color="0.25", ls="-", lw=0.7),
           guide=dict(color="0.55", ls=(0, (3, 3)), lw=0.5))
SEED_MARKERS = {"42-123": "o", "42-2026": "s", "123-2026": "D"}

RC = {
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path", "svg.hashsalt": "viewdep",
    "savefig.dpi": 600, "figure.dpi": 100, "axes.titlelocation": "left", "axes.titleweight": "bold",
}


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"pdf": {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None},
            "svg": {"Date": None, "Creator": None}, "png": {"Software": None}}
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", metadata=meta[ext])
    plt.close(fig)


def _pairing_type(pairing: str) -> str:
    a, b = pairing.split("+")
    if a[0] == b[0]:
        return "SS" if a[0] == "S" else "CC"
    return "X"


def joint_vs_independence(data: dict) -> dict:
    """Joint miss rate J against the independence reference q1*q2, one point per pairing (log axes)."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH_MM * MM, 78 * MM), sharex=True, sharey=True,
                                 gridspec_kw=dict(wspace=0.12))
        fig.subplots_adjust(left=0.10, right=0.985, top=0.93, bottom=0.30)
        lo, hi = 0.2, 50.0
        plotted = {}
        for ax, (fmt, pairs), tag in zip(axes, data.items(), ("a", "b")):
            ref, joint = 100 * pairs.q1 * pairs.q2, 100 * pairs.J
            kinds = pairs.pairing.map(_pairing_type)
            grid = np.array([lo, hi])
            for k, txt in ((2, "2×"), (5, "5×"), (10, "10×")):
                ax.plot(grid, k * grid, **REF["guide"], zorder=1)
                ax.text(hi * 0.8 / k / 1.12, hi * 0.8, txt, color="0.4", fontsize=7.5, ha="right", va="center")
            ax.plot(grid, grid, **REF["independence"], zorder=2)
            for kind in ("SS", "CC", "X"):
                sel = kinds == kind
                st = STYLE[kind]
                ax.scatter(ref[sel], joint[sel], marker=st["marker"], s=26, facecolor=st["color"],
                           edgecolor="white", linewidth=0.5, zorder=3)
                plotted[(fmt, kind)] = np.c_[ref[sel], joint[sel]]
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks])
            ax.set_yticklabels([f"{t:g}" for t in ticks])
            ax.minorticks_off()
            ax.set_title(f"({tag}) {LABEL[fmt]}")
            ax.set_xlabel(r"Independence reference $q_1 q_2$ (%)")
        axes[0].set_ylabel(r"Joint miss rate $J$ (%)")
        handles = [Line2D([], [], ls="none", marker=STYLE[k]["marker"], markersize=6, markerfacecolor=STYLE[k]["color"],
                          markeredgecolor="white", label=STYLE[k]["label"]) for k in ("SS", "CC", "X")]
        handles.append(Line2D([], [], **REF["independence"], label=r"Independence reference ($J = q_1 q_2$)"))
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.54, 0.0),
                   columnspacing=1.5, handletextpad=0.4)
        _save(fig, "joint_vs_independence")
    return plotted


def weekly_contrast(summaries: dict, weekly: dict) -> dict:
    """Primary contrast by test week: seed pairs (dependent comparisons), their mean, overall estimate."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH_MM * MM, 72 * MM), sharey=True,
                                 gridspec_kw=dict(wspace=0.08))
        fig.subplots_adjust(left=0.125, right=0.985, top=0.92, bottom=0.36)
        offsets = {"42-123": -0.2, "42-2026": 0.0, "123-2026": 0.2}
        plotted = {}
        for ax, fmt, tag in zip(axes, weekly, ("a", "b")):
            w, s = weekly[fmt], summaries[fmt]
            lo, hi = s["ci95_pp"]
            ax.axhspan(lo, hi, color="0.85", lw=0, zorder=0)
            ax.axhline(s["contrast_pp"], color="0.45", lw=0.9, zorder=1)
            ax.axhline(0, **REF["zero"], zorder=1)
            ax.axhline(2, **REF["effect"], zorder=1)
            for sp, mk in SEED_MARKERS.items():
                ax.plot(w.week + offsets[sp], w[f"pair_{sp}_pp"], ls="none", marker=mk, markersize=3.6,
                        markerfacecolor="white", markeredgecolor="0.4", markeredgewidth=0.7, zorder=2)
                plotted[(fmt, sp)] = w[f"pair_{sp}_pp"].to_numpy()
            ax.plot(w.week, w.contrast_pp, color="black", lw=1.4, marker="o", markersize=3.8, zorder=3)
            plotted[(fmt, "mean")] = w.contrast_pp.to_numpy()
            ax.set_xticks(w.week)
            ax.set_xlim(w.week.min() - 0.6, w.week.max() + 0.6)
            ax.set_xlabel("Test week")
            ax.set_title(f"({tag}) {LABEL[fmt]}")
            ax.yaxis.grid(True, color="0.92", lw=0.6)
            ax.set_axisbelow(True)
        axes[0].set_ylim(-0.5, 14.5)
        axes[0].set_yticks(range(0, 15, 2))
        axes[0].set_ylabel("Same-view minus crossed-view\nexcess joint miss (pp)")
        mean_h = Line2D([], [], color="black", lw=1.4, marker="o", markersize=3.8)
        seed_h = tuple(Line2D([], [], ls="none", marker=mk, markersize=4, markerfacecolor="white", markeredgecolor="0.4")
                       for mk in SEED_MARKERS.values())
        overall_h = (Patch(facecolor="0.85", edgecolor="none"), Line2D([], [], color="0.45", lw=0.9))
        effect_h = Line2D([], [], **REF["effect"])
        fig.legend([mean_h, seed_h, overall_h, effect_h],
                   ["Mean across seed pairs", "Individual seed pairs",
                    "Overall estimate and 95% CI", "Prespecified effect of interest (2 pp)"],
                   handler_map={tuple: HandlerTuple(ndivide=3, pad=0.2)}, loc="lower center", ncol=2,
                   frameon=False, bbox_to_anchor=(0.55, 0.0), columnspacing=2.0, handletextpad=0.5, handlelength=3.0)
        _save(fig, "weekly_contrast")
    return plotted


def main() -> None:
    fmts = ("dotnet", "win64")
    pairs = {f: pd.read_csv(RESULTS / f / "pairings.csv") for f in fmts}
    weekly = {f: pd.read_csv(RESULTS / f / "weekly.csv") for f in fmts}
    summaries = {f: json.loads((RESULTS / f / "summary.json").read_text()) for f in fmts}
    p1 = joint_vs_independence(pairs)
    p2 = weekly_contrast(summaries, weekly)
    # Plotted values must equal the source tables.
    for f in fmts:
        n = sum(len(p1[(f, k)]) for k in ("SS", "CC", "X"))
        assert n == len(pairs[f]) == 12
        assert np.allclose(np.sort(np.concatenate([p1[(f, k)][:, 1] for k in ("SS", "CC", "X")])),
                           np.sort(100 * pairs[f].J))
        assert np.allclose(p2[(f, "mean")], weekly[f].contrast_pp)
        assert np.allclose(np.mean([p2[(f, sp)] for sp in SEED_MARKERS], 0), weekly[f].contrast_pp)
        ss = p1[(f, "SS")]
        print(f"{f}: SS J {ss[:, 1].min():.2f}-{ss[:, 1].max():.2f}%, CC J {p1[(f, 'CC')][:, 1].min():.2f}-"
              f"{p1[(f, 'CC')][:, 1].max():.2f}%, crossed J {p1[(f, 'X')][:, 1].min():.2f}-{p1[(f, 'X')][:, 1].max():.2f}%; "
              f"ratio J/(q1q2) {np.min(pairs[f].J / (pairs[f].q1 * pairs[f].q2)):.2f}-{np.max(pairs[f].J / (pairs[f].q1 * pairs[f].q2)):.2f}; "
              f"weekly contrast {weekly[f].contrast_pp.min():.2f}-{weekly[f].contrast_pp.max():.2f} pp; "
              f"overall {summaries[f]['contrast_pp']:.2f} [{summaries[f]['ci95_pp'][0]:.2f}, {summaries[f]['ci95_pp'][1]:.2f}]")
    print("figure values match source tables")


if __name__ == "__main__":
    main()
