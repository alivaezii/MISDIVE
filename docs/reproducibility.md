# Reproducibility

## From this repository alone (released scores)
| Target | How | Expected |
|---|---|---|
| Toolkit correctness | `pytest -m "not slow"` | synthetic tests pass |
| Confirmatory primary and AB2 attribution with the toolkit (points, 95% CIs, permutation p, balanced contrasts) | `pytest -m slow` or `python examples/reproduce_confirmatory_attribution.py` | equal to `results/confirmatory/analysis_v2_results.json` |
| All confirmatory outputs with the study script | `python experiments/confirmatory/analysis_v2.py --target results/scores/win64_w47_52_target_scores.parquet`, then `addendum_a1_supplementary.py --target <same file>` | written to `work/confirmatory_reproduction/`; compare with `results/confirmatory/` |
| Development dry run | `analysis_v2.py --dry-run-on-validation` | compare with `results/development/` |
| Single-factor contrasts (.NET 10.74 pp, Win64 4.90 pp) | `viewdep.dependence` on the clean-run test scores and frozen thresholds | `results/single_factor/*/summary.json` |
| Figures and tables | `python experiments/figures_tables/make_figures_tables.py` | compare with `figures/` |

Bootstrap intervals depend on the random stream: seed 20261201 for the confirmatory analysis (primary pool, then
AB2, AB4, NC2, interventions, in that order), 20261202 for addendum S-C3, and 20261124 for the single-factor
bootstrap.

## Requires external data or compute (not reproducible from this repository alone)
- Reproducing model training and score generation requires the original EMBER2024 dataset and additional computational resources. Trained model checkpoints are not included in this repository.
