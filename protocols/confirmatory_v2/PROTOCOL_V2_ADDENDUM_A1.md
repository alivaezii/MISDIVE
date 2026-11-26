# Protocol V2: Addendum A1 (interpretation decisions)

**Recorded:** 2026-11-25. This was before any access to Win64 weeks 47–52. The digest of this file is in `PROTOCOL_V2_ADDENDUM_A1_SHA256.txt`.

**Scope.** This addendum records the user's decisions on the interpretation points C1–C7 raised in `INTERPRETATION_REVIEW_V2.md` (SHA-256 `7174bc83…c301`).

**Preserved unchanged:**
- `CONFIRMATORY_PROTOCOL_V2_WIN64_W47_52.md` (`b39c180d…b942`);
- `FREEZE_RECORD_V2.json` (`1c21509c…889e`);
- every frozen script.

**What this addendum does not change:** no hypothesis, threshold, minimum effect of interest (MEI), statistical test or model.

## Decisions

| Id | Decision |
|---|---|
| C1 | **Accepted.** On the covariance scale, the numerical MEIs apply: H1 *V* ≥ 2 pp and lower bound > 0; H2 lower bound of *V* − *G* > 0. On the correlation (φ) scale, the requirement is lower 95% bound > 0 for *V* (H1) and for *V* − *G* (H2) |
| C2 | **Accepted, with relabelling.** The IH2 comparator is the mean FNR across the three single detectors of a type. It is reported as the **"mean single-detector baseline"**, not as the "best single detector" |
| C3 | **Current interpretation retained; its limitation is disclosed.** The primary IH3 comparison stays as implemented: mean FL against the per-replicate minimum over view-restricted pair types. **Pre-specified supplementary sensitivity analysis S-C3:** *every individual* FL seed against *every individual* view-restricted pairing (each seed combination of each of the 12 pair types). Statistic: max over FL seeds of FNR minus min over individual pairings of FNR, computed within each bootstrap replicate. Pass: upper 95% bound ≤ 0. Computed at α_sys = 1% (primary) and at 0.5% and 2%. S-C3 is supplementary and does not replace IH3 |
| C4 | **Accepted** |
| C5 | **Accepted, with clarification.** In the AB2 pool (covariance scale), a 95% lower bound ≤ 0 for *G* (or *I*) means **robustness to the checkpoint-selection confounder is not established** for that effect. It does **not** show that the effect has disappeared. The original MEIs are preserved, and whether the AB2 estimates meet them is reported separately. AB2 robustness is "established" for an effect when its AB2 lower bound is > 0 |
| C6 | **Accepted.** H1 supported on both scales without H2 is classified as **STOP** |
| C7 | **Accepted.** H5 (the manuscript's Δ_view(MLP); ≥ 2 pp and lower bound > 0; "ruled out" if upper bound < 2 pp) is reported independently of the MISDIVE go rule |

## Outcome classes (final)

H1–H4 are as in protocol V2 §1 with C1 applied.

| Class | Condition | Conclusion |
|---|---|---|
| **Full success** | H1 and H2 and (H3 or H4), **and** AB2 robustness is established (C5) for at least one supported effect among *G* (H3) and *I* (H4) | two-factor attribution replicated, with at least one supported algorithm or interaction effect robust to selection coupling |
| **Partial success A** | H1 and H2 and (H3 or H4), but AB2 robustness is **not** established for any supported effect | **restricted view-attribution conclusion.** No claim of robust algorithm or interaction attribution |
| **Partial success B** | H1 and H2; neither H3 nor H4 | no separate framework; view effect replicated |
| **Failure** | H1 not supported; or H1 supported on the covariance scale only; or H1 without H2 (C6) | stop; negative result reported |

**Reported in every case:**
- AB2 MEI attainment (*V*, *G*, *I*);
- H5;
- IH1–IH4 (IH2 labelled with the mean single-detector baseline);
- S-C3;
- NC1, NC2, AB4, AB5, AB6;
- D1–D3.

## Implementation for these decisions

- **`analysis_v2.py`:** unchanged. It already implements C1, C2 (computation), C3 (primary), C4, C6 and C7.
- **`addendum_a1_supplementary.py`:** a new file, hashed before target access. It:
  - implements S-C3;
  - applies C5 and the final outcome classes to the output of `analysis_v2.py`;
  - relabels IH2 as specified.

  It reads only the target scores file produced once by `score_target_w47_52.py`, together with the source data. It performs no fitting, calibration or selection on target data.
