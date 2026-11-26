# Pre-evaluation review: protocol V2 wording, interpretations and decision criteria

**Date:** 2026-11-25.

**Status:** prepared before Authorisation E.
- Weeks 47–52 have not been opened, vectorised, inspected or scored.
- No threshold, hypothesis, protocol text or script was changed by this review.

**Digests referenced:**
- Protocol V2: `b39c180d…b942`.
- Freeze record: `1c21509c…889e`.
- `analysis_v2.py`: `27f30bdc…6b49`.
- `score_target_w47_52.py`: `11f142a9…84b8`.

---

## 1. Protocol wording and interpretations, side by side

The quotations are verbatim from `CONFIRMATORY_PROTOCOL_V2_WIN64_W47_52.md`, with line numbers.

| Id | Exact protocol wording | Interpretation implemented in `analysis_v2.py` |
|---|---|---|
| **C1** | L25: *"Supported" means estimate ≥ the minimum effect of interest (MEI) **and** lower 95% bound > 0.* <br> L35: ***Primary decision.** H1 and H2 both supported, on the covariance scale **and** on the correlation (φ) scale.* <br> L29/L30 MEIs: H1 = 2 pp; H2 = 0 pp (sign) | On the covariance scale: H1 needs *V* ≥ 2 pp and lower bound > 0; H2 needs lower bound of *V* − *G* > 0. On the φ scale: lower 95% bound > 0 for *V* (H1) and for *V* − *G* (H2). No φ-scale MEI is applied, because the protocol defines none and a value in pp has no meaning on the φ scale |
| **C2** | L42, IH2: *Seed redundancy is ineffective: \|FNR(LS + LS) − FNR(best single LS)\| < 1 pp, and the same for MS + MS* — pass: *equivalence: 95% CI of the difference within ±1 pp* | "Best single LS" = the **mean** FNR of the three single LS detectors. It is compared with the mean FNR of the three LS + LS seed pairs; likewise for MS |
| **C3** | L43, IH3: *a single full-feature LightGBM has FNR ≤ every two-detector configuration of view-restricted detectors* — pass: *upper 95% bound of FNR(FL) − FNR(best configuration) ≤ 0* | FL = mean of the three single FL detectors. "Best configuration" = within each bootstrap replicate, the minimum over the 12 view-restricted pair *types* (seed redundancy, algorithm swap, view swap, both swapped). Each type is averaged over its seed combinations |
| **C4** | L168: *Averaging: all ordered cross-seed combinations are averaged.* | Pairs of two different types: all 6 ordered seed combinations. Pairs of the same type: the 3 unordered seed pairs |

**Further ambiguities found in this review.** These were not listed earlier. They concern how results are read, not the code, except C6, which the code already implements.

| Id | Exact protocol wording | Reading proposed (for your approval) |
|---|---|---|
| **C5** | L199: *AB2 removes G or I → The algorithm and interaction effects are attributed to checkpoint-selection coupling, not to the learners* <br> L177: *AB2 … G and I keep their sign. If they vanish, the .NET G and I were selection artefacts* | "Removes" (= "vanishes") means that in the AB2 pool, on the covariance scale, the 95% lower bound of *G* (respectively *I*) is ≤ 0. This is the natural reading of "keep their sign". Whether the AB2 effects reach their MEIs is reported alongside. `analysis_v2.py` writes the AB2 estimates and intervals, so this reading needs no code change |
| **C6** | L195–197 decision table: rows cover (H1 and H2 and [H3 or H4]); (H1 and H2, without H3 and H4); and (H1 not supported, or covariance-only). **No row covers "H1 supported on both scales but H2 not supported".** | This case is classified as **STOP**, which is how `go_rule()` already treats it |
| **C7** | L29–L33: H5 (manuscript estimand) is listed as a primary hypothesis, but it does not appear in the §9 decision table | H5 is decided by its own rule (≥ 2 pp, lower bound > 0, as in the manuscript) and reported separately. It does not enter the MISDIVE go rule. This is how `decide()` and `go_rule()` already behave |

## 2. Does any interpretation change the tests, success criteria or go/no-go?

| Id | Changes a statistical test? | Changes a success criterion? | Can change the go/no-go? |
|---|---|---|---|
| C1 | No. The bootstrap and its intervals are as specified | It makes the φ clause operational. It is the weakest **defined** reading of that clause: no numeric φ MEI exists that it could replace | Yes, in principle. A φ MEI would make H1 harder to pass. In the dry run the φ lower bounds are far from 0 (*V*: 0.27; *V* − *G*: 0.12), so in practice the φ clause is unlikely to bind, and H1 is decided by the covariance MEI |
| C2 | No | It fixes the comparator of a **secondary** test | No: IH2 is not part of the go rule |
| C3 | No | It fixes the comparator of a **secondary** test | No: IH3 informs deployment guidance only (L198) |
| C4 | No. For an OR rule, (i, j) and (j, i) of the same type are the same configuration, so averaging over ordered combinations gives exactly the same mean | No | No |
| C5 | No | It defines when the AB2 interpretation row (L199) applies | It does not change the go **label** produced by L195. It does change whether the algorithm and interaction conclusion may be attributed to the learners (§4) |
| C6 | No | It fills a gap in the table | Yes: it decides that a case the protocol omits is a failure. That is the most stringent option |
| C7 | No | No | No |

## 3. Do any of these interpretations relax the original requirements?

- **C1: not relaxed against any defined requirement, but it is the least demanding operational reading.** The protocol gives no φ-scale MEI. Inventing one now would *add* a requirement that was never pre-specified and that would be set with knowledge of the development-data φ values. Sign-only is the only reading that adds nothing. Consequence, to be stated in the report: the φ requirement protects against the scale artefact (a positive covariance contrast with zero correlation contrast). It does not impose a φ effect size.
- **C2: not a relaxation; a substitution of a defined quantity for an undefined one.** "Best single LS" needs a selection rule, and the protocol gives none.
  - Choosing the best seed on the **target** weeks would select on evaluation data, which is not acceptable.
  - Choosing it on **validation** would be a defensible literal alternative.
  - The direction of the difference between the mean and the validation-best seed is not fixed in advance, so neither reading is systematically more lenient.
  - IH2 is secondary.
- **C3: partly less strict than the strictest literal reading. Disclosed.**
  - "≤ *every* two-detector configuration" could be read at the level of **individual** pairings (each seed combination) and each individual FL seed. The minimum over individual pairings is lower than the minimum over type averages, so that reading is harder to pass.
  - C3 compares averages, so it is **less strict** than that reading.
  - It is **stricter** than comparing with a single, fixed best type, because the minimum is taken inside every bootstrap replicate.
  - IH3 is secondary.
  - *Options before E:* **(a)** keep C3, disclosed as above; **(b)** tighten C3 to the individual-pairing reading. Option (b) requires editing `analysis_v2.py`, recording the new digest and updating the freeze record. I recommend **(a)**, because a post-freeze code change carries more risk than a disclosed secondary-test reading. Option (b) only ever tightens.
- **C4: identical** to the literal text.
- **C5: not a relaxation.** "Keep their sign" is the protocol's own expectation. Reporting MEI attainment alongside adds information without changing the rule.
- **C6: the most stringent** possible completion of the gap.
- **C7: identical** to how the protocol separates H5 from the MISDIVE decision table.

**Conclusion.** No interpretation relaxes a requirement that the protocol defines numerically. C1 is the least demanding reading of an undefined clause. C3 is less strict than the strictest literal reading of a secondary test, and is disclosed. Neither affects the go rule except through C1's φ clause, which is unlikely to bind.

## 4. Complete final decision criteria

All criteria are evaluated once, on weeks 47–52, at the frozen thresholds (branch 0.5%). Notation:
- **H1** = (*V* ≥ 2 pp and *V* lower bound > 0, covariance) **and** (*V* lower bound > 0, φ).
- **H2** = (*V* − *G* lower bound > 0, covariance) **and** (the same on φ).
- **H3** = *G* ≥ 1 pp and lower bound > 0 (covariance).
- **H4** = *I* ≥ 1 pp and lower bound > 0 (covariance).

Each bound is from a family-cluster bootstrap with B = 2000 and seed 20261201, using percentile 95% intervals.

| Outcome class | Condition | Conclusion |
|---|---|---|
| **Full success** | H1 and H2 and (H3 or H4), **and** AB2 does not remove the supported effect(s) under C5 | Two-factor attribution is replicated on unexamined data, with the algorithm and/or interaction effect not explained by selection coupling. A methods-and-evidence paper is justified |
| **Partial success A** | H1 and H2 and (H3 or H4), **but** AB2 removes every supported algorithm or interaction effect (C5) | The go label is produced by L195, but by L199 the algorithm and interaction effects are attributed to checkpoint-selection coupling. Substantively this is **view-only attribution** |
| **Partial success B** | H1 and H2; neither H3 nor H4 | **No separate framework** (L196): view effect replicated; MISDIVE reduces to the manuscript's single-factor result plus replication |
| **Failure** | H1 not supported; **or** H1 supported only on covariance; **or** H1 supported but H2 not (C6) | **Stop** (L197); report the negative result |
| Reported in every case | H5 (manuscript Δ_view(MLP) ≥ 2 pp, lower bound > 0; "ruled out" if upper bound < 2 pp); IH1–IH4 at α_sys = 1% with 0.5% and 2% as sensitivity; NC1 permutation p-values (not a pass condition); NC2; AB4; AB5; AB6; D1–D3 | Secondary; descriptive or interpretive |

**Classification of Partial success A.** "Partial success A" is my classification, combining L195 with L199. The protocol does not name it. It is the more conservative reading of those two rows taken together. **I ask you to confirm it before E.**

## 5. Role of the independently selected MLP ablation (AB2)

**AB2 is a robustness and interpretation check. It is not a primary success criterion.**
- **The protocol's description of it:**
  - it is listed under "Optional ablation detectors" (L75–L80);
  - under ablations, with an expected sign pattern (L177);
  - as an interpretation row of the decision table (L199).
- **The primary criteria never mention it.**
- **What it can do:** under C5 it can restrict *what is concluded* (Partial success A). It cannot turn a failure into a success, nor raise the outcome class.

**Development-data indication** (weeks 41–46, in-sample; already reported):
- AB2 lowered *G* from 1.03 to 0.76 pp, below the 1-pp MEI.
- AB2 lowered *I* from 1.73 to 1.02 pp.
- Both lower bounds stayed > 0. Under C5 neither effect would be "removed". If *G* met its MEI only in the primary pool, the report would have to state that its size depends on the selection procedure.

## 6. Hash verification (recomputed 2026-11-25)

| Check | Result |
|---|---|
| 50 files in `FREEZE_RECORD_V2.json` | 0 mismatches |
| Protocol V2 and `train_win64_lgbm_pool.py` against `PROTOCOL_V2_SHA256.txt` (the original record) | match |
| Protocol v1 (Phase 0C) | unchanged |
| 9 LightGBM models against `train_record.json`, and against the digests printed in the training logs at training time | match |
| 6 AB2 models against `ab2_train_record.json` | match |
| Clean-run models, preprocessor, `protocol.json` and `feature_views.json` against `frozen.json` | match |
| Thresholds recomputed from validation logits (21 detectors) against the records | max abs difference 0.0 |
| `train_win64_mlp_ablation.py` and `train_win64_lgbm_FL_resume.py` against the digests at run time (`TRAINING_SCRIPTS_SHA256_at_run.txt`) | match (final versions) |
| `analysis_v2.py`, `score_target_w47_52.py` | as recorded in the freeze record |
| `F:\ELSIVIER\MISDIVE_Phase1_pool\win64\target_w47_52` | does not exist |

**Note on FL training.** FL_42 and FL_123 were trained by the intermediate version of the resume script (`081605d9…`). FL_2026 was trained by the final version (`80baec5e…`). The two differ only in the `--seeds` option. Both digests are in the at-run record.

## 7. Scientific recommendation

**Proceed with the independent evaluation, provided you first accept or amend C1–C7.**

**Reasons to proceed:**
1. **The evaluation is fully specified and verified.** Pool, thresholds, scripts, digests and decision rules are fixed. Nothing remains to be tuned.
2. **An expected negative is not a reason to skip.** The development data suggest a borderline outcome; Partial success B or Failure are plausible. Running only when success looks likely would itself be a selection bias.
3. **The same run gives a blind replication of the manuscript's estimand (H5)** on unexamined data. That is the only blind confirmation the project can still obtain, and it has value independent of MISDIVE.

**Costs and caveats:**
- It consumes the only unexamined labelled window.
- The hypotheses were informed by examined data, which the report must disclose.
- The weeks are only 1–6 weeks after validation.
- The detector pool is small (3 seeds), so replicate-level inference is weak.
- LightGBM is far more competent than the MLPs, so the covariance-scale effects are confounded by competence.

**What a positive result would not establish:**
- generality beyond these two views, two learners and one benchmark;
- systems of more than two detectors;
- any causal effect of representation or learning algorithm as such.

**Decisions needed from you before Authorisation E:**
- accept C1, C2, C4 and C7 as implemented;
- accept C5 and C6 as proposed;
- choose C3 option (a) or (b);
- confirm the Partial success A classification.

Once you accept them as written, nothing needs to be changed before execution. If you choose C3(b), or amend any other reading, I will make only that change, record the new digests, and report them before running.
