# Methodology implemented by the toolkit

The toolkit implements the estimators defined in the manuscript (Algorithm 1). It adds no estimator.

## Inputs
- A pool of trained detectors, each with a replicate identifier (independent training run, e.g. seed) and its
  level on one or two binary design factors. Every replicate must contain exactly one detector per cell.
- Binary decisions on a common evaluation window, or scores together with thresholds (`decisions_from_scores`;
  a file is flagged iff score > threshold). `threshold_at_fpr(benign, α)` returns the ⌊α·N₀⌋-th largest
  (0-based) calibration benign score, so the calibration FPR is at most α.
- File labels (1 malware, 0 benign); optionally cluster labels (e.g. malware families) and time windows.

## Estimators
For detectors a, b with miss indicators mₐ, m_b on the malware files:
q_a = E[mₐ], J_ab = E[mₐ m_b] (miss rate of the OR system), C_ab = J_ab − q_a q_b,
φ_ab = C_ab / √(q_a(1−q_a) q_b(1−q_b)). Only pairs from different replicates are used, so no pair shares a
training run. Each pair falls into a sharing cell (first letter: factor 1, second: factor 2; S shared, X not
shared). With μ the mean over the pairs of a cell:

- effect_1 (V) = ½(μ_SS + μ_SX − μ_XS − μ_XX)
- effect_2 (G) = ½(μ_SS + μ_XS − μ_SX − μ_XX)
- interaction (I) = (μ_SS − μ_SX) − (μ_XS − μ_XX)
- residual (R) = μ_XX

The manuscript's cell names SVSA, SVXA, XVSA and XVXA correspond to SS, SX, XS and XX with factor 1 = view.
With a single factor, effect_1 = μ_S − μ_X. Balanced contrasts ½Cov(dᵢ, dⱼ), with dᵢ the difference of the
miss indicators of the two levels in replicate i, are reported per level pairing of the other factor. V equals
the mean of the same-level and mixed-level view contrasts; I equals their difference (checked in the tests).

## Uncertainty
- **Files (conditional on the trained detectors):** paired cluster bootstrap. Clusters are resampled with
  replacement; q, J, C and φ of every pair are recomputed on the same resample. Percentile 95% intervals.
  Files with no cluster label become singleton clusters keyed by their identifier. Without any cluster labels
  the bootstrap resamples files independently, and the result says so.
- **Training runs:** type labels are permuted within each replicate (24^R permutations for a 2×2 design;
  exact enumeration when feasible). The smallest attainable one-sided p-value is limited by label symmetries
  (128/13,824 ≈ 0.0093 for R = 3).

## Decision rule (M7)
`decide(result, mei)`: an effect is supported if its covariance estimate is ≥ its minimum effect of interest
(MEI) and its lower 95% bound is > 0. On the φ scale only a positive lower bound is checked. An effect is
ruled out if its upper bound is below the MEI. MEIs must be fixed before the evaluation data are examined.

## Diagnostics and operations
- `window_effects`: effects per time window; a window is flagged as a burst when the same-view joint miss rate
  exceeds its median by more than 3 pp while the crossed-view rate does not.
- `family_decomposition`: contribution of each cluster to the view contrast.
- `calibrate_configurations` / `compare_configurations`: OR-combinations of detectors on validation-calibrated
  tail fractions, with the combined threshold set on validation benign files at budget α; FNR differences
  with cluster-bootstrap intervals.
