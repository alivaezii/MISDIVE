# Confirmatory protocol: Win64 weeks 47–52 (MISDIVE design-factor attribution and prospective joint-miss prediction)

**Protocol version:** 1.0
**Status:** FROZEN TEXT, NOT AUTHORISED FOR EXECUTION.

**Authorisation.** Execution requires two explicit authorisations from the user, each given in chat:
- **Authorisation T:** permission to train the additional detectors in §3.2.
- **Authorisation E:** permission to vectorise and score Win64 weeks 47–52 once.

**Freezing.** The SHA-256 digest of this file is recorded in `PROTOCOL_SHA256.txt`. Before Authorisation E, a second record must add the digests of the trained models and thresholds. Any change to this file after Authorisation E invalidates the confirmatory status of the results.

**Disclosure (to be reproduced in any report):**
- The investigators have examined the clean-run Win64 results on weeks 41–46 (validation) and 53–64 (official test).
- They have examined the Phase 0 .NET results for weeks 41–64, including the exploratory prediction and attribution analyses of Phases 0B and 0C.
- Weeks 47–52 lie between these periods. Their labels, features and scores have never been read; only SHA-256 digests were read, for deduplication.
- The hypotheses below are therefore informed by examined data. The *data* are unexamined; the *hypotheses* were not formed blind.

---

## 1. Questions

**Primary (P).** For pairings of known detector types that have not been jointly measured (and whose dependence is estimated without any shared detector), does the prospective MISDIVE predictor predict the joint miss rate on weeks 47–52 within a stated margin of what a historical measurement of the same pairing would achieve? The predictor uses historical individual miss rates plus historical design-factor-level dependence. A further requirement is that it predicts substantially better than every predictor that uses no pairing-level history.

**Secondary attribution questions (A):**
- Do the two-factor structure and its ordering, both found exploratorily on .NET, hold in a new file type on unexamined data?
- The structure consists of a feature-view main effect, an algorithm main effect and their interaction.
- The ordering is: view sharing > algorithm sharing, with algorithm sharing mattering mainly within a shared view.

**Secondary diagnostic questions (D):** descriptive only; see §7.

## 2. Data

| Role | Data | Use |
|---|---|---|
| Training | Win64 weeks 1–40, all unique files (clean-run cohort) | only for the new detectors of §3.2 |
| Source ("historical") | Win64 weeks 41–46 (60,000 files; validation) | thresholds, historical marginals, historical dependence |
| Target | **Win64 weeks 47–52**: all unique files after the existing deduplication (60,000 SHA-256 digests in `data/weeks/win64/unique_47..52.parquet`) | evaluation only, scored once |
| Excluded | weeks 53–64, challenge set, other file types | not used |

**Labels and families.** Label ∈ {0, 1}; any other value is excluded and counted. The ClarAVy family is used only to define bootstrap clusters; a missing family is a singleton cluster.

**Data availability assumptions:**

| Assumption | Holds for |
|---|---|
| Per-file decisions of every detector on the source period exist | prospective predictors (class H) |
| Each detector's individual miss rate on the target period is known, but not the joint miss rate | conditional predictors (class C). Secondary only, because this assumption is rarely realistic: per-file target labels would usually also give the joint rate |
| Target labels | used **only** to compute the evaluation targets |

## 3. Detector pool

### 3.1 Existing detectors (frozen; no change)

Clean-run Win64 two-branch MLPs, seeds 42, 123 and 2026. Their branches are:
- MS_s: structural view;
- MC_s: content view.

Their checkpoints, preprocessor and thresholds are recorded in `SENTRA_AEiC2027/results/win64/frozen.json` (protocol SHA-256 `d2ba3068…`).

### 3.2 New detectors (require Authorisation T)

LightGBM detectors on the same Win64 preprocessed arrays and the same structural and content column indices as the MLP branches:
- LS_s: structural view;
- LC_s: content view.

Seeds are 42, 123 and 2026.

**Settings,** identical to SENTRA Phase 0 E10 (`src/sentra/training/classical.py`):

| Group | Settings |
|---|---|
| Model | objective binary; learning_rate 0.05; num_leaves 64; min_data_in_leaf 20 |
| Sampling and regularisation | feature_fraction 0.5; bagging_fraction 0.8; bagging_freq 1; lambda_l2 1.0; max_bin 63 |
| Run | num_threads 8; up to 1000 rounds; early stopping 50 rounds on validation (weeks 41–46) log-loss |
| Seeds | seed = bagging_seed = feature_fraction_seed = s |
| Environment | LightGBM 4.7.0; inputs cast float16 → float32 |

**Thresholds:** for every detector, the floor(0.005·N₀)-th largest source benign score, where N₀ is the number of source benign files, with decision score > threshold (ties never flagged). This is `viewdep.thresholds.at_fpr`.

**What is recorded:** model and threshold digests are recorded before Authorisation E.

**Known asymmetry, which must be reported:**
- MLP checkpoints were selected on source-period OR-system TPR.
- LightGBM detectors were selected on source-period log-loss.
- Both use the source period, which is therefore in-sample for model selection and thresholds.

### 3.3 Pairs and categories

**Pairs:** all 48 pairs of the 12 detectors whose seeds differ. Same-seed pairs are excluded, as in all earlier analyses.

**Categories** (fixed):

| Category | View | Algorithm |
|---|---|---|
| SVSA | same | same |
| SVXA | same | different |
| XVSA | different | same |
| XVXA | different | different |

Each category has 12 pairs.

## 4. Estimands and prediction equations

**Indicators.** On malware, *m*_d ∈ {0, 1} is detector d's miss indicator. *q*_d^S and *q*_d^T are its source and target miss rates. For pair p = (a, b): *J*_p^S, *J*_p^T are the joint miss rates and Cov_p^S = *J*_p^S − *q*_a^S *q*_b^S.

**Estimation set L2(p).** The set of source pairs in the same category as p that contain neither a nor b. It is non-empty for all 48 pairs under this design; if it is empty, the prediction is undefined and counted.

**Category dependence.**

C̄_L2(p) = mean of Cov_m^S over m ∈ L2(p)

**Feasibility.** Each prediction is clipped to the Fréchet interval of the marginals that the predictor uses:

[max(0, *q*_a + *q*_b − 1), min(*q*_a, *q*_b)]

The number of clipped predictions is reported.

### 4.1 Predictors

| Id | Class | Equation | Role |
|---|---|---|---|
| **M-H** | prospective | Ĵ_p = *q*_a^S *q*_b^S + C̄_L2(p) | **primary MISDIVE predictor** |
| B-IND | prospective | *q*_a^S *q*_b^S | independence baseline |
| B-CAT | prospective | mean of *J*_m^S over m ∈ L2(p) | historical pairing-type mean |
| B-PAIR | prospective | *J*_p^S | historical pair-specific rate. Its information is *not* available for a genuinely unmeasured pairing, so it is the reference for non-inferiority, not a competitor |
| M-C | conditional | *q*_a^T *q*_b^T + C̄_L2(p) | secondary (Phase 0B predictor, with L2) |
| B-C-IND | conditional | *q*_a^T *q*_b^T | secondary baseline |
| B-C-PAIR | conditional | *q*_a^T *q*_b^T + Cov_p^S | secondary reference |

**No target joint information** is used by any predictor. Target marginal miss rates are used only by the conditional class.

### 4.2 Attribution estimands

Category mean covariances *μ*_SVSA, *μ*_SVXA, *μ*_XVSA, *μ*_XVXA are computed on target malware over the 12 pairs in each category, in pp.

| Effect | Formula |
|---|---|
| View main effect | V = ½(*μ*_SVSA + *μ*_SVXA − *μ*_XVSA − *μ*_XVXA) |
| Algorithm main effect | G = ½(*μ*_SVSA + *μ*_XVSA − *μ*_SVXA − *μ*_XVXA) |
| Interaction | I = (*μ*_SVSA − *μ*_SVXA) − (*μ*_XVSA − *μ*_XVXA) |

**Balanced contrasts.** The balanced view contrasts Δ_view(MM), Δ_view(LL) and Δ_view(ML), and the algorithm contrasts Δ_alg(SS), Δ_alg(CC) and Δ_alg(SC), follow the Phase 0B definitions. Each detector enters both arms equally.

**Sensitivity.** The same effects are reported on the correlation (φ) scale.

## 5. Error metrics and uncertainty

**Primary metric.** MAE = mean over the 48 pairs of |Ĵ_p − *J*_p^T|, in pp.

**Secondary metrics:**
- the common shift (mean signed error);
- the centred MAE (MAE after subtracting the common shift);
- RMSE and the maximum absolute error;
- Kendall's τ between predictions and targets;
- every metric per category.

**Uncertainty.**
- *Procedure:* paired family-cluster bootstrap over **target** malware files, with B = 2000 and generator `numpy.random.default_rng(20261128)`.
- *What is resampled:* every replicate resamples clusters with replacement, then recomputes the target joint rates and the target marginals (the latter for class C).
- *What is fixed:* source quantities are held fixed, so inference is conditional on the observed history.
- *Intervals:* differences in MAE between predictors are computed within replicates, with percentile 95% intervals.

**Source-uncertainty sensitivity (reported; not decisive).** A second bootstrap that also resamples source clusters, B = 500, seed 20261129.

**Overlapping pairs.**
- The 48 pairs share detectors, so their errors are dependent.
- The file-level cluster bootstrap captures sampling dependence through shared files. It does not support inference to a population of detectors.
- Per-category results and a leave-one-seed-out sensitivity are reported: MAE over the 16 pairs not involving each seed in turn.
- No claim is made about genuinely new detectors.

## 6. Decision criteria (fixed)

The margins are justified in the Phase 0C report §9. They were derived from validation-only backtests (weeks 41–43 → 44–46) and from the planned use of the predictor. They were not tuned on any examined test period.

| Criterion | Test | Pass condition |
|---|---|---|
| **H1 non-inferiority** | MAE(M-H) − MAE(B-PAIR) | upper 95% bound < **δ = 0.5 pp** |
| **H2 superiority over predictors without pairing history** | MAE(M-H) relative to min{MAE(B-IND), MAE(B-CAT)} | the ratio ≤ **0.5**, **and** the upper 95% bound of [MAE(M-H) − the minimum] < 0 |
| A1 | view main effect V | V > 0 with lower 95% bound > 0 |
| A2 | ordering | V − G > 0 with lower 95% bound > 0 |
| A3 | interaction | I > 0 with lower 95% bound > 0 |

**Primary decision:** the predictive claim is **confirmed** only if **H1 and H2 both pass**.

**Attribution:** A1–A3 are evaluated independently. Each pass or failure is reported. No multiplicity adjustment is applied to A1–A3; they are secondary and are reported as such.

**Conditional class:** M-C against B-C-PAIR and B-C-IND is reported with the same margins as a secondary analysis. It cannot confirm the primary claim.

## 7. Diagnostics (descriptive; no pass/fail)

- **Decomposition:** crossed-minus-same-view competence/dependence decomposition, at frozen thresholds and at matched realised FP. Matched FP uses the target benign files and is labelled retrospective.
- **Benign-side dependence:** joint false-alarm rate *K* by category.
- **Weekly burst screen:** per category and week, with any family responsible for > 50% of a burst reported. A burst is a week in which the same-view joint miss rate exceeds its six-week median by > 3 pp while the crossed rate does not.

## 8. Interpretation

| Outcome | Interpretation |
|---|---|
| H1 and H2 pass | For unmeasured pairings of *known detector types*, historical marginals plus design-factor-level dependence can stand in for a historical pairing measurement at a 1–6 week horizon in Win64. Not shown: unseen detector types, other representations, longer horizons, *k* > 2 |
| H2 passes, H1 fails | Group-level dependence beats naive predictors but is not a substitute for pairing-level history. The predictive module is of limited practical value |
| H2 fails | The predictive module is **rejected**. Attribution results are reported on their own |
| A1 passes, A2 or A3 fails | Representation dependence replicates; the factor ordering or interaction does not transfer from .NET |
| A1 fails | Representation-associated common-mode dependence does not replicate in this window, which contradicts the manuscript's direction for this window |

**Negative results** are reported in full, with the same prominence as positive ones.

## 9. Execution steps (after Authorisations T and E only)

1. Train LS_s and LC_s (§3.2). Compute thresholds. Record digests in `FREEZE_RECORD.json` beside this file, together with this file's digest.
2. Vectorise Win64 weeks 47–52 from the existing `Win64_train.zip` with the existing extractor, preprocessor and deduplication. This is a separate script; no existing file is modified. Record the counts by week and class.
3. Score all 12 detectors once. Write decisions. Compute §4–§7 with a script whose digest is recorded before step 2.
4. Report every criterion.
