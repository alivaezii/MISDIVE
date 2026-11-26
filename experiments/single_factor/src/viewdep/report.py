"""Tables and figures for the manuscript, generated only from files in results/."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .paths import FORMATS, RESULTS, ROOT  # noqa: E402

MS = ROOT / "manuscript"
LABEL = {"dotnet": ".NET", "win64": "Win64"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def load(fmt):
    d = RESULTS / fmt
    return (json.loads((d / "summary.json").read_text()), json.loads((d / "cohort_audit.json").read_text()),
            json.loads((d / "frozen.json").read_text()), pd.read_csv(d / "pairings.csv"),
            pd.read_csv(d / "weekly.csv"), pd.read_csv(d / "branches.csv"))


def pct(x, d=2):
    return f"{100 * x:.{d}f}"


def main():
    (MS / "tables").mkdir(parents=True, exist_ok=True)
    (MS / "figures").mkdir(parents=True, exist_ok=True)
    prov = json.loads((RESULTS / "provenance_original.json").read_text())
    data = {f: load(f) for f in FORMATS}

    # Table: cohorts
    rows = []
    for f, (s, a, fz, *_rest) in data.items():
        for split in ("train", "validation", "test"):
            p = a["per_split"][split]
            rows.append(f"{LABEL[f]} & {split.capitalize()} & {p['files']:,} & {p['malware']:,} & {p['benign']:,} & "
                        f"{p['malware_without_family']:,} \\\\")
    cohort = "\n".join(rows)
    (MS / "tables" / "cohorts.tex").write_text(cohort + "\n")

    # Table: main results per format and pairing type
    lines, summary_rows = [], []
    for f, (s, a, fz, pairs, weekly, br) in data.items():
        m = s["mean_by_type"]
        o = prov[f]
        lines.append(f"{LABEL[f]} & {pct(m['same']['J'])} & {pct(m['crossed']['J'])} & {pct(m['same']['excess'])} & "
                     f"{pct(m['crossed']['excess'])} & {pct(m['same']['system_fpr'])} & {pct(m['crossed']['system_fpr'])} & "
                     f"{s['contrast_pp']:.2f} [{s['ci95_pp'][0]:.2f}, {s['ci95_pp'][1]:.2f}] & "
                     f"{s['weeks_positive']}/{s['weeks']} & {o['contrast_pp']:.2f} [{o['ci95_pp'][0]:.2f}, {o['ci95_pp'][1]:.2f}] \\\\")
        summary_rows.append({"format": f, "contrast_pp": s["contrast_pp"], "ci_low_pp": s["ci95_pp"][0],
                             "ci_high_pp": s["ci95_pp"][1], "decision": s["decision"],
                             "weeks_positive": s["weeks_positive"], "weeks": s["weeks"],
                             "J_same": m["same"]["J"], "J_crossed": m["crossed"]["J"],
                             "excess_same": m["same"]["excess"], "excess_crossed": m["crossed"]["excess"],
                             "fpr_same": m["same"]["system_fpr"], "fpr_crossed": m["crossed"]["system_fpr"],
                             "original_contrast_pp": o["contrast_pp"], "original_ci_low_pp": o["ci95_pp"][0],
                             "original_ci_high_pp": o["ci95_pp"][1]})
    (MS / "tables" / "main.tex").write_text("\n".join(lines) + "\n")
    pd.DataFrame(summary_rows).to_csv(RESULTS / "summary_both_formats.csv", index=False)

    # Table: per seed pair and branch operating points
    sp = []
    for f, (s, a, fz, pairs, weekly, br) in data.items():
        for k, v in s["contrast_per_seed_pair_pp"].items():
            sp.append(f"{LABEL[f]} & {k.replace('-', '--')} & {v:.2f} & {prov[f]['per_seed_pair_pp'][k]:.2f} \\\\")
    (MS / "tables" / "seed_pairs.tex").write_text("\n".join(sp) + "\n")
    br_rows = []
    for f, (s, a, fz, pairs, weekly, br) in data.items():
        b = br.set_index("branch")
        for seed in (42, 123, 2026):
            br_rows.append(f"{LABEL[f]} & {seed} & {pct(b.loc[f'S_{seed}', 'fnr'])} & {pct(b.loc[f'S_{seed}', 'fpr'])} & "
                           f"{pct(b.loc[f'C_{seed}', 'fnr'])} & {pct(b.loc[f'C_{seed}', 'fpr'])} & "
                           f"{fz['training'][str(seed)]['best_epoch'] + 1} \\\\")
    (MS / "tables" / "branches.tex").write_text("\n".join(br_rows) + "\n")

    # Figures: shared style and exports in viewdep.figures (PDF for LaTeX, SVG and 600-dpi PNG for Word)
    from . import figures
    figures.main()

    # Macros so the manuscript text quotes numbers from result files rather than retyped values
    mac = []
    for f, (s, a, fz, pairs, weekly, br) in data.items():
        k = "Net" if f == "dotnet" else "Win"
        m = s["mean_by_type"]
        mac += [rf"\newcommand{{\contrast{k}}}{{{s['contrast_pp']:.2f}}}",
                rf"\newcommand{{\ciLow{k}}}{{{s['ci95_pp'][0]:.2f}}}",
                rf"\newcommand{{\ciHigh{k}}}{{{s['ci95_pp'][1]:.2f}}}",
                rf"\newcommand{{\weeksPos{k}}}{{{s['weeks_positive']}}}",
                rf"\newcommand{{\nTest{k}}}{{{s['test_files']:,}}}",
                rf"\newcommand{{\nTestMal{k}}}{{{s['test_malware']:,}}}",
                rf"\newcommand{{\nTestBen{k}}}{{{s['test_benign']:,}}}",
                rf"\newcommand{{\Jsame{k}}}{{{pct(m['same']['J'])}}}",
                rf"\newcommand{{\Jcross{k}}}{{{pct(m['crossed']['J'])}}}",
                rf"\newcommand{{\Esame{k}}}{{{pct(m['same']['excess'])}}}",
                rf"\newcommand{{\Ecross{k}}}{{{pct(m['crossed']['excess'])}}}",
                rf"\newcommand{{\Fsame{k}}}{{{pct(m['same']['system_fpr'])}}}",
                rf"\newcommand{{\Fcross{k}}}{{{pct(m['crossed']['system_fpr'])}}}",
                # Descriptive ratio J / (q1 q2), averaged per pairing type (not a protocol endpoint).
                rf"\newcommand{{\Rsame{k}}}{{{(pairs.J / (pairs.q1 * pairs.q2))[pairs.type == 'same'].mean():.2f}}}",
                rf"\newcommand{{\Rcross{k}}}{{{(pairs.J / (pairs.q1 * pairs.q2))[pairs.type == 'crossed'].mean():.2f}}}",
                rf"\newcommand{{\weekMin{k}}}{{{weekly.contrast_pp.min():.2f}}}",
                rf"\newcommand{{\weekMax{k}}}{{{weekly.contrast_pp.max():.2f}}}",
                rf"\newcommand{{\clusters{k}}}{{{s['bootstrap']['clusters']:,}}}",
                rf"\newcommand{{\noFamily{k}}}{{{s['bootstrap']['malware_without_family']:,}}}",
                rf"\newcommand{{\origContrast{k}}}{{{prov[f]['contrast_pp']:.2f}}}",
                rf"\newcommand{{\origLow{k}}}{{{prov[f]['ci95_pp'][0]:.2f}}}",
                rf"\newcommand{{\origHigh{k}}}{{{prov[f]['ci95_pp'][1]:.2f}}}",
                rf"\newcommand{{\nTrain{k}}}{{{a['per_split']['train']['files']:,}}}",
                rf"\newcommand{{\nVal{k}}}{{{a['per_split']['validation']['files']:,}}}",
                rf"\newcommand{{\repeats{k}}}{{{a['exact_repeats_removed']:,}}}",
                rf"\newcommand{{\overlap{k}}}{{{a['test_sha_overlap_with_training_archive_excluded']:,}}}"]
    (MS / "tables" / "macros.tex").write_text("\n".join(mac) + "\n")
    print("report written")


if __name__ == "__main__":
    main()
