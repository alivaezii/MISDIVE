"""Rescore saved Phase 0 .NET checkpoints (no retraining) for supplementary analyses A and B.

Runs with the original SENTRA environment, which provides the `sentra` package and LightGBM, and
reads the original project read-only:

    <SENTRA>/.venv/Scripts/python.exe supplementary/score_phase0_models.py <SENTRA>

Models: E04_indep_semantic, E05_indep_random (two-branch MLPs) and E10_lgbm_{structural,content},
seeds 42, 123 and 2026, all trained on the Phase 0 training cohort with the Phase 0 preprocessor.

Integrity checks:
  1. Rescoring the Phase 0 development test (weeks 47-52) reproduces the saved predictions.
  2. Rescoring the official .NET test (weeks 53-64, clean-run raw features) reproduces the saved
     original E04 official scores, aligned by SHA256.

Outputs:
  artifacts/supplementary/phase0_dotnet_scores_w53_64.parquet
  results/supplementary/rescoring_checks.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SENTRA = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(SENTRA / "src"))
from sentra.data import dataset as D  # noqa: E402
from sentra.models.mlp import TwoBranch  # noqa: E402
from sentra.training import classical as CL  # noqa: E402

RUNS = SENTRA / "experiments" / "runs"
SEEDS = (42, 123, 2026)
OUT_ART = ROOT / "artifacts" / "supplementary"
OUT_RES = ROOT / "results" / "supplementary"
OUT_ART.mkdir(parents=True, exist_ok=True)
OUT_RES.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


z = np.load(SENTRA / "experiments" / "preprocessor.npz")
pp = D.Preprocessor()
pp.mean_, pp.std_, pp.keep_ = z["mean"], z["std"], z["keep"]
keep = pp.keep_


def load(run: str):
    cfg = json.loads((RUNS / run / "status.json").read_text())["config"]
    if cfg["kind"] == "lgbm":
        import lightgbm as lgb
        i1, i2 = D.view_indices(keep, "semantic")
        return ("lgbm", lgb.Booster(model_file=str(RUNS / run / "model.txt")),
                i1 if cfg["view"] == "structural" else i2)
    i1, i2 = D.view_indices(keep, cfg["view"])
    m = TwoBranch(i1, i2)
    m.load_state_dict(torch.load(RUNS / run / "model.pt"))
    m.eval()
    return ("torch", m, None)


def score(model, X: np.ndarray) -> np.ndarray:
    typ, m, cols = model
    if typ == "lgbm":
        return CL.lgbm_logit(m, X, cols)[:, None]
    with torch.no_grad():
        out = [torch.stack(m(torch.from_numpy(X[s:s + 8192])), 1) for s in range(0, len(X), 8192)]
    return torch.cat(out).numpy().astype(np.float64)


RUN_IDS = {"sem": "E04_indep_semantic_s{}", "rnd": "E05_indep_random_s{}",
           "lgbS": "E10_lgbm_structural_s{}", "lgbC": "E10_lgbm_content_s{}"}
models = {(k, s): load(v.format(s)) for k, v in RUN_IDS.items() for s in SEEDS}
checks = {"preprocessor_sha256": sha(SENTRA / "experiments" / "preprocessor.npz"),
          "model_sha256": {RUN_IDS[k].format(s): sha(RUNS / RUN_IDS[k].format(s) /
                                                     ("model.txt" if k.startswith("lgb") else "model.pt"))
                           for k, s in models},
          "view_dims": {v: [len(a) for a in D.view_indices(keep, v)] for v in ("semantic", "random")},
          "n_retained_columns": int(len(keep))}

# 1. Development-test reproduction (weeks 47-52).
Xd = pp.transform(np.load(SENTRA / "data" / "processed" / "X_devtest.npy", mmap_mode="r"), np.arange(30000))
dev = {}
for (k, s), m in models.items():
    saved = np.load(SENTRA / "results" / "predictions" / f"devtest_{RUN_IDS[k].format(s)}.npy").astype(np.float64)
    dev[RUN_IDS[k].format(s)] = float(np.abs(score(m, Xd) - saved).max())
checks["devtest_max_abs_diff"] = dev
del Xd

# 2. Official test weeks 53-64 from clean-run raw features.
wk = ROOT / "data" / "weeks" / "dotnet"
X = np.concatenate([np.load(wk / f"X_{w}.npy") for w in range(53, 65)])
meta = pd.concat([pd.read_parquet(wk / f"meta_{w}.parquet") for w in range(53, 65)], ignore_index=True)
Xt = pp.transform(X, np.arange(len(X)))
del X
cols = {}
for (k, s), m in models.items():
    sc = score(m, Xt)
    if k in ("sem", "rnd"):
        a, b = ("S", "C") if k == "sem" else ("R1", "R2")
        cols[f"{k}_{a}_{s}"], cols[f"{k}_{b}_{s}"] = sc[:, 0], sc[:, 1]
    else:
        cols[f"{k}_{s}"] = sc[:, 0]
df = pd.concat([meta[["sha256", "week", "label", "family"]].reset_index(drop=True), pd.DataFrame(cols)], axis=1)
orig = pd.read_parquet(SENTRA / "confirm_view" / "official" / "official_scores.parquet").set_index("sha256")
assert set(orig.index) == set(df.sha256), "official cohort differs from the clean-run raw test cohort"
o = orig.loc[df.sha256]
checks["official_rows"] = int(len(df))
checks["official_label_agreement"] = bool((o.label.values == df.label.values).all())
checks["official_E04_max_abs_diff"] = {f"{v}_{s}": float(np.abs(o[f"{v}_{s}"].values - df[f"sem_{v}_{s}"].values).max())
                                       for s in SEEDS for v in "SC"}
p = OUT_ART / "phase0_dotnet_scores_w53_64.parquet"
df.to_parquet(p, index=False)
checks["output_sha256"] = sha(p)
(OUT_RES / "rescoring_checks.json").write_text(json.dumps(checks, indent=1))
print(json.dumps(checks, indent=1))
