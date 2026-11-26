# MISDIVE — Miss Diversity Evaluation for detector pools

Code, frozen protocols, integrity records, released scores and results that accompany the manuscript
*MISDIVE: Miss Diversity Evaluation for Reliable AI-Driven Malware Detection Architectures* 

MISDIVE audits whether the **misses** of several malware detectors are dependent, and attributes that dependence
to shared design factors (here: the feature *view* and the learning *algorithm*). It works on binary decisions
(or scores plus thresholds) of a pool of trained detectors. It does not train detectors itself.

> **Scope.** The toolkit runs on any balanced pool of 1–2 binary design factors with at least two independent
> training replicates per cell. That is *software portability*. The scientific findings below were demonstrated on
> EMBER2024 Win64 and .NET only (one confirmatory window, Win64 weeks 47–52). Running the toolkit on other data
> does not, by itself, establish that those findings generalise.

## Repository layout

| Path | Content |
|---|---|
| `misdive/` | Lightweight toolkit implementing Algorithm 1 (modules M1–M7, below) |
| `tests/` | Synthetic correctness tests; exact reproduction test of the confirmatory attribution |
| `examples/` | `synthetic_quickstart.py`, `reproduce_confirmatory_attribution.py` |
| `protocols/` | Frozen protocols, addendum and erratum, SHA-256 records, frozen copies of the scripts as hashed |
| `experiments/` | Study code: single-factor clean run (`viewdep`), confirmatory training/scoring/analysis, exploratory analyses, figure and table generation |
| `results/` | Released result files, per-file detector scores, frozen thresholds |
| `figures/` | Figures as in Manuscript |
| `docs/` | Methodology, evidence and provenance, data, reproducibility |

## Toolkit

```python
from misdive import FactorialDesign, decisions_from_scores, run_audit, decide

design = FactorialDesign.from_records(records, factors=("view", "algorithm"))  # detector, replicate, factor levels
decisions = decisions_from_scores(scores, thresholds)                            # flagged iff score > threshold
result = run_audit(design, decisions, labels, clusters=families, file_ids=sha256, windows=weeks,
                   n_boot=2000, seed=20261201)
print(result.effects_table())
print(decide(result, mei={"effect_1": 2.0, "effect_2": 1.0, "interaction": 1.0}))
```

| Module | Function(s) | Output |
|---|---|---|
| M1 Competence | `competence` | FNR, FPR per detector |
| M2 Dependence | `pair_table` | q, J (OR-system miss rate), covariance, φ per cross-replicate pair |
| M3 Factorial attribution | `factorial_effects`, `balanced_contrasts` | main effects, interaction, residual; ½Cov(dᵢ, dⱼ) contrasts |
| M4 Robustness / uncertainty | `bootstrap_effects`, `permutation_test` | family-cluster bootstrap CIs (file level); within-replicate permutation p (training-run level) |
| M5 Diagnostics | `window_effects`, `family_decomposition` | per-window effects with the burst flag; family contributions |
| M6 Operations | `calibrate_configurations`, `compare_configurations` | OR-configurations calibrated on validation benign files; FNR differences with CIs |
| M7 Confirmation | `decide` | the pre-specified rule: supported if estimate ≥ MEI and lower 95% bound > 0 |

If no cluster labels are given, files are resampled independently and the result carries a note saying so.
Effects are reported in percentage points (covariance) and on the φ scale.

## Principal released results

From `results/confirmatory/analysis_v2_results.json` (Win64 weeks 47–52; 60,000 files, 30,000 malware in
8,971 clusters; 0.5% FPR thresholds frozen on weeks 41–46; B = 2000; 13,824 exact permutations):

| Effect (covariance, pp) | Estimate | 95% CI (files) | Permutation p (training runs) |
|---|---|---|---|
| View main effect V | 2.55 | [1.57, 3.99] | 0.0093 (smallest attainable) |
| Algorithm main effect G | 0.92 | [0.62, 1.31] | 0.056 |
| Interaction I | 1.41 | [0.91, 2.06] | 0.111 |

Frozen go-rule label: *GO: two-factor attribution replicated*. Addendum A1 outcome class: *FULL SUCCESS*.
The algorithm effect is below its minimum effect of interest (1.0 pp). The interaction is not established at the
training-run level. Single-factor clean run (`results/single_factor/*/summary.json`): view contrast 10.74 pp
[8.73, 12.53] (.NET) and 4.90 pp [4.08, 5.39] (Win64). See `docs/evidence_and_provenance.md` for the
confirmatory, exploratory and development status of every result.

## Installation and tests

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[plots,test]"
pytest -m "not slow"                                # synthetic tests (< 1 min)
pytest -m slow                                      # exact reproduction of the confirmatory attribution (minutes)
python examples/synthetic_quickstart.py
```

What can and cannot be reproduced from this repository alone is listed in `docs/reproducibility.md`.

## Data

The detectors were trained and evaluated on EMBER2024 (Joyce et al., KDD 2025, doi:10.1145/3711896.3737431;
Hugging Face `joyce8/EMBER2024`, revision `3d23efe`). No malware binaries, raw feature vectors or model checkpoints
are included. See `docs/data.md`.



