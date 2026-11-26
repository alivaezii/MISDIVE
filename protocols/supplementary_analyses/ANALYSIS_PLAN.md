# Supplementary analyses: plan fixed before computation

Status: written and committed **before** any of the analyses below were computed. All evaluation
windows used here have been examined before:

- **Weeks 47–52 (.NET):** the Phase 0 development test, unlocked for the Phase 0 GO/NO-GO decision.
- **Weeks 53–64:** the official .NET and Win64 test weeks, examined in the original confirmatory
  evaluations and in the clean run.

Every result is therefore a **supplementary analysis of previously examined test data**, not a
confirmation. The primary endpoint and results of Manuscript V1 are not changed.

## Common definitions (unchanged from `configs/protocol.json`)

- **Branch threshold:** the floor(0.005 N0)-th largest validation benign score (weeks 41–46); a file
  is flagged when its score is strictly greater than the threshold.
- **Pairing statistics:** q1, q2, J, excess = J − q1·q2, and the realised OR-system FP count and
  FPR, all computed with `viewdep.dependence.pair_stats`.
- **Balanced contrast Δ:** for each seed pair, the mean excess over the two same-view pairings minus
  the mean excess over the two crossed pairings, then averaged over the three seed pairs
  (`viewdep.dependence.contrast`).
- **Interval:** paired family-cluster bootstrap over test malware (B = 2000, rng seed 20261124),
  with seed pairs averaged inside each replicate. Branches shared across seed pairs are therefore
  resampled jointly, which respects the dependence between overlapping seed pairs.

## A. LightGBM pairing analysis

- **Models:** Phase 0 runs `E10_lgbm_structural_s{42,123,2026}` and `E10_lgbm_content_s{…}`.
  - Training cohort: Phase 0 .NET training cohort (weeks 1–40, deterministic 50% SHA256 subsample,
    100,320 files).
  - Preprocessing: the Phase 0 preprocessor (`experiments/preprocessor.npz`).
  - Columns: the semantic view indices of `sentra.data.dataset.view_indices`.
  - Early stopping: on validation weeks 41–46.
- **Pairings:** S_i+S_j, C_i+C_j versus S_i+C_j, C_i+S_j, with the endpoint defined exactly as above.
- **Windows:**
  - (W1) weeks 47–52, from saved predictions.
  - (W2) official weeks 53–64, from saved checkpoints rescored on the clean-run raw test features.
    W2 is used only if rescoring the semantic MLP checkpoints reproduces the saved original official
    scores.
- **Reference:** the independent semantic MLPs `E04_indep_semantic_s*` evaluated in the same window,
  with the same thresholds rule.

## B. Random-partition control

- **Models:** `E05_indep_random_s{42,123,2026}`.
  - Two-branch MLP, identical to E04 except for the input partition.
  - The partition is one fixed random permutation (rng 20261124) of the 2,120 retained columns,
    split into blocks R1 and R2 with the dimensionalities of the structural and content views.
- **Pairings:** R1_i+R1_j, R2_i+R2_j versus R1_i+R2_j, R2_i+R1_j. Same windows, thresholds and
  bootstrap as A.
- **Paired difference:** Δ_semantic − Δ_random, computed within each bootstrap replicate on the same
  resampled malware.
- **Scope:** conclusions are conditional on this single random partition.

## C. Equal-FP analysis of the manuscript's pairings

- **Data:** the clean-run models and test scores (`artifacts/<format>/test_scores.parquet`) and the
  saved validation logits, for .NET and Win64.

### System score and threshold-placement rule

1. **Branch tail score.** For each branch k, let v_k be its sorted validation benign logits
   (N0v files). The tail score is the interpolated rank `r_k(s) = interp(s, v_k, 0..N0v−1)`.
   Outside the validation range the rank is extended linearly with slope 1 per logit unit, so the
   map stays strictly monotone.
2. **Pairing score and threshold.** The pairing's system score is z = max(r_a, r_b). A file is
   flagged when z > τ. This is equivalent to giving both branches thresholds at the same
   validation-benign tail level. At τ0 = N0v − 1 − floor(0.005 N0v) it reproduces the manuscript's
   frozen OR decision, apart from ties.
3. **Retrospective equal-FP operating point.**
   - Target: K = round(r · N0test) system false positives on test benign files, for r = 1%
     (primary) and r = 0.5% and 2% (secondary).
   - Placement: τ is set to the (K+1)-th largest test-benign z.
   - Ties: with strict ">", tied scores are never flagged, so the achieved FP count is at most K.
     The achieved counts are reported.
   - The same K applies to all twelve pairings of a file type.
4. **Prospective validation-calibrated operating point.** The same rule applied with validation
   benign files instead of test benign files: K_v = round(r · N0v), with the realised test FP count
   reported.
5. **Reported per pairing and operating point:** τ, achieved FP count and FPR, branch q1 and q2 at
   the implied thresholds, J (= system FNR), and excess.
6. **Aggregates:**
   - ΔJ: same-view minus crossed-view J, balanced by seed pair as for Δ.
   - Δexcess: the same balanced contrast applied to the excess.
   - Bootstrap over test malware, conditional on the placed thresholds; benign files are not
     resampled.
7. **Interpretation constraint.** Changing τ changes both branches' operating points. The marginal
   miss rates are then no longer identical on the two sides, so ΔJ at equal FP is a detection
   comparison, not the manuscript's dependence endpoint.

The frozen-validation operating points of Manuscript V1 are reported unchanged, alongside these
results.

## Integrity checks (must pass before results are used)

- The saved validation logits reproduce the branch thresholds in `results/<format>/frozen.json`.
- The rescored E04 official .NET scores match `confirm_view/official/official_scores.parquet`,
  aligned by SHA256 (maximum absolute difference < 1e-4). The resulting Δ reproduces the original
  +11.90 pp.
- The rescored LightGBM and MLP dev-test scores match the saved dev-test predictions.
- Files are aligned by SHA256 across all score sources, and the label and family columns agree.

## Addendum D (fixed before computation): structural-anchor comparison

Question: does replacing the second structural detector of a structural pair with a content detector
lower the system miss rate at equal system false positives? Averaging SS and CC (analysis C) can hide
the stronger same-view configuration, so SS is compared directly.

- Data: clean-run models, official weeks 53-64, .NET and Win64 (`artifacts/<format>/test_scores.parquet`,
  saved validation logits). Supplementary analysis of previously examined test data.
- Scoring and thresholds: identical to analysis C (validation-benign tail ranks, system score =
  max rank, flag if > tau). Primary operating point: retrospective, tau placed per pairing so that
  exactly K = round(0.01 * N0test) test benign files are flagged (ties never flagged; achieved counts
  reported). Sensitivity: K at 0.5% and 2%; validation-selected tau at 0.5%, 1%, 2% with realised
  test FP counts reported (not matched).
- For each seed pair (i, j) in {(42,123), (42,2026), (123,2026)}:
  SS_ij = S_i+S_j; crossed_ij = mean of FNR(S_i+C_j) and FNR(S_j+C_i); CC_ij = C_i+C_j (context).
  delta_FNR_ij = crossed_ij - FNR(SS_ij). Endpoint: delta_FNR = mean over the three seed pairs, in pp.
  Negative values favour crossed-view pairing.
- Uncertainty: paired family-cluster bootstrap over test malware (B = 2000, rng 20261124, the same
  replicate weights for every configuration), seed pairs averaged inside each replicate. Intervals
  are conditional on the fitted models and on the placed thresholds (benign files not resampled).
- File-level changes, per anchored comparison (seed pair x crossed pairing vs SS) before averaging:
  malware detected by crossed but missed by SS, detected by SS but missed by crossed, and net. Reported
  as the mean over the six anchored comparisons, and as unique files over the union of comparisons;
  repeated evaluations of the same file are counted separately from unique files.
- Reading: delta_FNR <= -2 pp with an interval excluding 0 supports an advantage of crossed pairing
  over SS; an interval including 0 or a difference within 2 pp is reported as small or uncertain; a
  positive interval is adverse. The 2-pp reference is the manuscript's effect of interest.
