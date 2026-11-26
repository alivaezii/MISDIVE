# Evidence status and provenance of the released results

| Status | Where | What it is |
|---|---|---|
| **Confirmatory** | `results/confirmatory/` | Single pre-registered evaluation on Win64 weeks 47–52 under `protocols/confirmatory_v2/` (protocol V2, addendum A1, erratum). Thresholds, configurations and MEIs were frozen on weeks 41–46 before the target scores were produced. |
| **Clean run (single factor)** | `results/single_factor/` | View-dependence contrast on .NET and Win64 test weeks 53–64, under the frozen protocol `protocols/clean_run/protocol.json`. Frozen thresholds and model digests in `*/frozen.json`. |
| **Development (in-sample)** | `results/development/`, `results/training/` | Dry run of the confirmatory analysis on validation weeks 41–46, used to check the code path; the label inside the JSON states "IN-SAMPLE; not confirmatory". |
| **Exploratory** | `results/exploratory/` | Phase 0b/0c prototypes and Phase 1 design inputs that motivated the confirmatory protocol. Not confirmatory evidence. |
| Superseded | `protocols/superseded_v1_never_executed/` | Protocol V1, frozen and then replaced by V2 before any execution; kept for transparency. |

## Integrity records
- `protocols/confirmatory_v2/*_SHA256.txt`, `FREEZE_RECORD_V2.json`, `TRAINING_SCRIPTS_SHA256_at_run.txt`:
  SHA-256 digests recorded when the protocol and scripts were frozen.
- `protocols/confirmatory_v2/frozen_scripts/`: byte-identical copies of the scripts as hashed. They and the
  freeze record contain the absolute local paths of the original machine. These were left unchanged on purpose,
  because editing them would break the digests. They are records, not runnable code. The runnable versions are in
  `experiments/`, with paths taken from environment variables.
- `CONFIRMATORY_RESULTS_SHA256.txt`, `TARGET_SCORES_SHA256.txt`: digests of the confirmatory outputs and the
  target score file as produced.

## Differences between `experiments/` and the frozen scripts
Only input/output locations were changed. The released versions read from `results/` and write to `work/`. The
authorisation check of the one-time target run was removed from `analysis_v2.py`, because the released scores
make re-running safe. No estimator, threshold, seed or decision rule was changed. The release QA compared the
re-run outputs with the released JSON files.
