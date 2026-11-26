# Data

## Source dataset
EMBER2024 (Joyce et al., Proc. KDD 2025 V.2, pp. 5516–5526, doi:10.1145/3711896.3737431), obtained from the
Hugging Face repository `joyce8/EMBER2024` at revision `3d23efe`. Features were extracted with the official
EMBER2024 extractor, pinned in `experiments/single_factor/requirements.txt`. Feature-view definitions:
`protocols/clean_run/feature_views.json`.

## Included in this repository
| File | Content |
|---|---|
| `results/scores/win64_w41_46_validation_scores.parquet` | Validation weeks 41–46: SHA-256, week, label, family, scores of the 21 Phase 1 detectors |
| `results/scores/win64_w47_52_target_scores.parquet` | Same for the 60,000 files of the confirmatory target weeks 47–52 |
| `results/scores/{dotnet,win64}_w53_64_clean_run_test_scores.parquet` | Scores of the single-factor clean-run detectors on test weeks 53–64 |
| `results/thresholds/win64_frozen_thresholds_0.5pct.json` | Frozen 0.5% FPR thresholds |

Family labels and SHA-256 identifiers come from the EMBER2024 metadata.

## Not included
- Malware or benign binaries, raw EMBER2024 feature vectors (obtain them from the dataset authors).
- Trained model checkpoints. Their SHA-256 digests are in the frozen/training records.
- The private Phase 0 working project that produced some exploratory .NET inputs.


