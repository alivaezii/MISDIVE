# Confirmatory protocol V2: two-factor attribution of common-mode misses, Win64 weeks 47–52

**Version:** 2.0.

**Supersedes** `phase0c/CONFIRMATORY_PROTOCOL_WIN64_W47_52.md` (v1.0). Version 1.0 had joint-miss prediction as its primary question. It was never executed and is retained unchanged. **Prediction is not evaluated in V2.**

**Status:** FROZEN TEXT, NOT AUTHORISED FOR EXECUTION. Two separate explicit authorisations are needed:
- **Authorisation T:** training of the detectors in §3.2 (and, optionally, §3.3).
- **Authorisation E:** one-time vectorisation and scoring of Win64 weeks 47–52.

**Digest:** the SHA-256 of this file is in `PROTOCOL_V2_SHA256.txt`. A freeze record holding the digests of the trained models, thresholds and the analysis script must be completed after T and before E.

**Disclosure** (to be reproduced in any report):
- **Already examined by the investigators:**
  - Win64 clean-run results on weeks 41–46 and 53–64.
  - .NET Phase 0 results on weeks 41–64.
  - The Phase 0B/0C/1 exploratory analyses, including the .NET factorial effects that motivate H2–H4.
- **Status of weeks 47–52:** only SHA-256 digests have been read.
- **Consequence:** the data are unexamined, but the hypotheses are informed by examined data.

---

## 1. Hypotheses

The estimands are defined in §4. "Supported" means estimate ≥ the minimum effect of interest (MEI) **and** lower 95% bound > 0.

| Id | Hypothesis | MEI | Basis |
|---|---|---|---|
| **H1** (primary) | **View main effect** *V* > 0 | 2 pp | the manuscript's prespecified effect of interest for view-associated dependence; .NET *V* = 7.9–9.1 pp |
| **H2** (primary) | **Ordering:** *V* − *G* > 0 | 0 pp (sign) | .NET *V* − *G* ≈ 5 pp |
| H3 | **Algorithm main effect** *G* > 0 | 1 pp | .NET *G* = 2.9–3.2 pp. The MEI is about a third of that, set before seeing Win64 |
| H4 | **Interaction** *I* > 0: algorithm sharing adds dependence mainly within a shared view | 1 pp | .NET *I* = 4.7–5.4 pp. At replicate level .NET permutation p was 0.10–0.14, so H4 is uncertain |
| H5 | **Manuscript estimand replicates:** Δ_view(MLP) on existing frozen models | 2 pp | blind replication of the manuscript contrast in a new window, under its original decision rule |

**Primary decision.** H1 and H2 both supported, on the covariance scale **and** on the correlation (φ) scale. Robustness on the φ scale is required because of the scale artefact in §4.3.

**Operational intervention hypotheses** (secondary). All are tested at matched **system** false-positive budget α_sys = 1%, with the budget set on validation benign files; realised FPR is reported.

| Id | Hypothesis | Pass condition |
|---|---|---|
| IH1 | View swap beats seed redundancy for LightGBM: FNR(LS + LC) < FNR(LS + LS) | upper 95% bound of the difference < 0 |
| IH2 | Seed redundancy is ineffective: \|FNR(LS + LS) − FNR(best single LS)\| < 1 pp, and the same for MS + MS | equivalence: 95% CI of the difference within ±1 pp |
| IH3 | **Conventional dominance:** a single full-feature LightGBM has FNR ≤ every two-detector configuration of view-restricted detectors | upper 95% bound of FNR(FL) − FNR(best configuration) ≤ 0. Reported whatever the outcome |
| IH4 | Algorithm swap within the structural view does **not** beat the stronger structural detector alone: FNR(MS + LS) ≥ FNR(LS) − 1 pp | lower 95% bound of FNR(MS + LS) − FNR(LS) > −1 pp |

## 2. Data

| Role | Data |
|---|---|
| Training | Win64 weeks 1–40: the clean-run cohort, 400,000 unique files (`SENTRA_AEiC2027/data/prep/win64/train.npy`) |
| Source | weeks 41–46: 60,000 files. Used for early stopping, thresholds and system-FP calibration. All weeks are kept; excluding week 42 is a sensitivity analysis only |
| Target | **weeks 47–52**: the 60,000 deduplicated SHA-256 digests in `data/weeks/win64/unique_47..52.parquet`. Vectorised with the existing extractor and preprocessor, then scored once |
| Not used | weeks 53–64, the challenge set, Win32 and all other file types |

**Labels:** label ∈ {0, 1}; any other value is excluded and counted.

**Family:** ClarAVy family, used for bootstrap clusters and diagnosis. Files with a missing family form singleton clusters.

## 3. Detector pool

**3.1 Existing detectors (frozen):** clean-run Win64 two-branch MLPs with seeds 42, 123 and 2026, giving branches MS_s (structural) and MC_s (content). Their thresholds are in `results/win64/frozen.json`.

**3.2 New detectors (Authorisation T)**, built by `phase1/train_win64_lgbm_pool.py`:
- **Detectors:** LS_s (structural), LC_s (content) and FL_s (full vector; comparator only), for s ∈ {42, 123, 2026}.
- **Settings:** the SENTRA Phase 0 E10/E01 settings:
  - objective binary, learning_rate 0.05, num_leaves 64, min_data_in_leaf 20;
  - feature_fraction 0.5, bagging_fraction 0.8, bagging_freq 1, lambda_l2 1.0, max_bin 63, num_threads 8;
  - up to 1,000 rounds, with early stopping after 50 rounds on validation log-loss.
- **Seeds:** seed = bagging_seed = feature_fraction_seed = s.
- **Determinism flags:** deterministic = True and force_col_wise = True. These are added for reproducibility and do not change the method.
- **Environment:** LightGBM 4.7.0; inputs cast from float16 to float32.
- **Columns:** `Preprocessor.view_index("structural" | "content")` of the clean Win64 preprocessor, so the columns are identical to those of the MLP branches.
- **Thresholds:** the floor(0.005·N₀)-th largest validation benign score, flagging only on a strict ">".

**3.3 Optional ablation detectors (Authorisation T, separately listed):**
- **Optional ablation AB2, selection-matched MLPs:**
  - *Detectors:* six single-branch MLPs, one per view and seed (64-32-1).
  - *Settings:* the same optimiser settings as the clean run.
  - *Early stopping:* on each detector's **own** validation TPR at 0.5% FPR.
  - *Purpose:* removes the checkpoint-selection coupling between MLP branches (§4.4).
- **Optional ablation AB3, random-partition MLPs:** two-branch MLPs on one random partition with matched dimensions (seed 20261124), 3 seeds. This separates the effect of the *semantic* split from the effect of *any* split.

**3.4 Pairs:**
- The factorial set has 12 factorial detectors (MS, MC, LS, LC × 3 seeds), giving 48 pairs whose seeds differ.
- Same-seed pairs are excluded.
- The pairs fall into four cells of 12 pairs each:

| Cell | View | Algorithm |
|---|---|---|
| SVSA | same | same |
| SVXA | same | different |
| XVSA | different | same |
| XVXA | different | different |

## 4. Estimands

### 4.1 Definitions

On target malware, *m*_d ∈ {0, 1} is the miss indicator at the frozen threshold. For a pair (a, b), Cov_ab = E[*m*_a*m*_b] − *q*_a*q*_b. The cell means *μ*_cell are averages over the 12 pairs in a cell, in pp:

| Estimand | Formula |
|---|---|
| View main effect | *V* = ½(*μ*_SVSA + *μ*_SVXA − *μ*_XVSA − *μ*_XVXA) |
| Algorithm main effect | *G* = ½(*μ*_SVSA + *μ*_XVSA − *μ*_SVXA − *μ*_XVXA) |
| Interaction | *I* = (*μ*_SVSA − *μ*_SVXA) − (*μ*_XVSA − *μ*_XVXA) |
| Residual dependence | *R* = *μ*_XVXA |

**Interpreting *R*.** *R* is the dependence that remains between detectors sharing neither factor. It is attributable to everything they still share: the training files and labels, deduplication, the EMBER feature extractor and the inherent difficulty of files. It is **not** a factor effect.

### 4.2 Balanced contrasts (reported)

| Contrast | Pairs |
|---|---|
| Δ_view(MM), Δ_view(LL), Δ_view(ML) | view contrasts for MLP–MLP, LightGBM–LightGBM and cross-algorithm pairs |
| Δ_alg(SS), Δ_alg(CC), Δ_alg(SC) | algorithm contrasts for structural, content and cross-view pairs |

Each contrast is ½ Cov(*d*_i, *d*_j), with *d* the within-seed difference between the two levels.

**Identities (checked by assertion):**
- *V* = mean of [mean(Δ_view(MM), Δ_view(LL)), Δ_view(ML)].
- *I* = mean(Δ_view(MM), Δ_view(LL)) − Δ_view(ML).

### 4.3 Scales

- **Primary scale:** covariance (pp).
- **Required companion:** the same estimands with every pair's covariance replaced by its φ coefficient, Cov / √(*q*_a(1−*q*_a)*q*_b(1−*q*_b)).
- **Why the companion is required:** under a constant-correlation null, the covariance-scale view contrast is not zero. It equals ½ρ(σ_X − σ_Y)², where σ = √(*q*(1−*q*)). For the Win64 MLP clean-run test weeks this is about 0.47 pp of 4.90 pp; for .NET it is about 0.02 pp.

### 4.4 Known confounders (reported, not removed)

1. **Competence differs across levels.** This affects magnitudes (§4.3).
2. **Selection criteria differ by algorithm.** MLP checkpoints were selected on within-seed OR-system TPR; LightGBM on log-loss. AB2 addresses this.
3. **Fixed levels.** There is one semantic partition and two specific learners. AB3 addresses the "any split" alternative.
4. **The source period is in-sample** for model selection and thresholds.

## 5. Uncertainty

- **File-level uncertainty (primary).**
  - *Procedure:* paired family-cluster bootstrap over target malware, B = 2000, `numpy.random.default_rng(20261201)`.
  - *Estimands:* all of them, recomputed in each replicate.
  - *Interval:* percentile 95%.
  - *Conditioning:* this inference is conditional on the trained detectors.
- **Replicate-level uncertainty (secondary, reported).**
  - *Procedure:* an exact randomisation test that permutes the four detector-type labels within each seed. There are 24³ = 13,824 permutations; all are enumerated.
  - *Statistic:* the one-sided p-value for *V*, *G* and *I*.
  - *Purpose:* this is the negative control NC1. It is not a pass condition, because with 3 seeds its power is limited.
- **Intervention uncertainty.** Paired family-cluster bootstrap over target files, with malware and benign files resampled within cluster.

## 6. Interventions (secondary): evaluation at matched system false-positive budget

**Calibration (validation benign files only):**
1. Each detector score is mapped to its validation-benign tail fraction *t*_d.
2. A configuration flags a file when min_d *t*_d < τ.
3. τ is set so that at most floor(α_sys·*N*₀) validation benign files are flagged, with α_sys ∈ {0.5%, 1%, 2%}. 1% is primary.

**Reporting:**
- realised target FPR;
- FNR;
- a retrospective variant with τ placed on target benign files (labelled retrospective).

**Configurations:**
- **Single detectors:** MS, MC, LS, LC and FL.
- **Seed redundancy:** XX for each type, including FL + FL.
- **Swaps:**
  - algorithm swap within a view: MS + LS, MC + LC;
  - view swap within an algorithm: MS + MC, LS + LC;
  - both factors swapped: MS + LC, LS + MC.
- **Averaging:** all ordered cross-seed combinations are averaged.

## 7. Negative controls and ablations

| Id | Control or ablation | Expected under the attribution claim |
|---|---|---|
| NC1 | Within-seed permutation of the type labels (§5) | observed *V* in the upper tail of the null |
| NC2 | Benign-side analogue (joint false alarms) | reported. No expectation is used for decisions |
| AB1 | Single-factor view contrast with MLP only (H5) | replicates the manuscript estimand |
| AB2 | Selection-matched MLPs replace the clean MLPs in the factorial | *G* and *I* keep their sign. If they vanish, the .NET *G* and *I* were selection artefacts |
| AB3 | Random-partition MLPs as a pseudo-view | Δ_random > 0 but < Δ_semantic: the semantic split adds dependence beyond a generic split (.NET exploratory: 4.76 against 11.90 pp) |
| AB4 | Branch thresholds at nominal 0.25%, 0.5% and 1% | sign stability |
| AB5 | Source excluding week 42 | affects only the system-FP calibration of interventions |
| AB6 | φ scale | required for the primary decision (§1) |

## 8. Diagnosis (descriptive; no pass/fail)

| Id | Procedure |
|---|---|
| D1 | Family decomposition of the raw view contrast ½E[*d*_i*d*_j], which is exactly additive over families. Report the top-10 family shares, the share from files without a family, and the number of families with a positive contribution |
| D2 | Weekly *V*, *G* and *I* |
| D3 | Burst screen: a week in which the same-view joint miss rate exceeds its six-week median by > 3 pp while the crossed rate does not. For each flagged week, report its family composition and its cross-level recovery |

## 9. Decision rules

| Outcome | Conclusion |
|---|---|
| H1 and H2 supported on both scales, and at least one of H3 or H4 supported | **Go:** two-factor attribution replicated in a second file type on unexamined data. Writing a methods-and-evidence paper is justified |
| H1 and H2 supported; H3 and H4 not supported | The view effect replicates, but the algorithm factor adds nothing demonstrable. MISDIVE reduces to the manuscript's single-factor result plus replication: **no separate framework** |
| H1 not supported, or supported only on the covariance scale | Representation attribution does not replicate robustly. **Stop**; report the negative result |
| IH3 holds | Deployment guidance: no view-restricted redundant configuration beats a single full-feature detector. Attribution then explains dependence but does not justify view-restricted architectures on detection grounds |
| AB2 removes *G* or *I* | The algorithm and interaction effects are attributed to checkpoint-selection coupling, not to the learners |

**Negative results** are reported in full, with equal prominence.

## 10. Execution steps (only after T and then E)

1. Run `train_win64_lgbm_pool.py --authorised-T`, plus the optional ablations if they are authorised. Write `FREEZE_RECORD_V2.json` with the model, threshold and analysis-script digests and this protocol's digest.
2. Vectorise weeks 47–52 with a new script. It reads `Win64_train.zip` members 47–52 only, uses the existing extractor and preprocessor, and modifies no existing file. Record the counts by week and class.
3. Score all detectors once. Run the analysis script recorded in step 1.
4. Report §1, §6, §7 and §8 completely.
