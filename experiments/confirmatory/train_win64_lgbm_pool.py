"""MISDIVE Phase 1: training script for the Win64 LightGBM detectors.

Trains, on the clean-run Win64 cohort (weeks 1-40; early stopping on weeks 41-46), nine LightGBM detectors:
  LS_s  structural view, LC_s  content view, FL_s  full feature vector (conventional comparator),
  for s in (42, 123, 2026), Reads the clean repository read-only
and writes only to OUT. Weeks 47-52 are never read by this script.

Usage:
  python -I train_win64_lgbm_pool.py --authorised-T
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
import os

import numpy as np

REPO = Path(os.environ.get("MISDIVE_CLEAN_RUN", Path(__file__).resolve().parents[1] / "single_factor"))
OUT = Path(os.environ.get("MISDIVE_POOL", Path(__file__).resolve().parents[2] / "work" / "pool" / "win64"))
SEEDS = (42, 123, 2026)
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=64, min_data_in_leaf=20, feature_fraction=0.5,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=63, num_threads=8, verbose=-1,
              deterministic=True, force_col_wise=True)  # last two: reproducibility flags only
ROUNDS, EARLY, ALPHA = 1000, 50, 0.005


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def at_fpr(benign, alpha):  # identical to viewdep.thresholds.at_fpr
    s = np.sort(np.asarray(benign, dtype=np.float64))[::-1]
    return float(s[int(np.floor(alpha * len(s)))])


def main():
    if "--authorised-T" not in sys.argv:
        sys.exit("Refusing to train: authorisation T has not been given (see CONFIRMATORY_PROTOCOL_V2).")
    import lightgbm as lgb
    import pandas as pd
    sys.path.insert(0, str(REPO / "src"))
    from viewdep.features import Preprocessor
    assert lgb.__version__ == "4.7.0", lgb.__version__
    OUT.mkdir(parents=True, exist_ok=True)
    prep = REPO / "data" / "prep" / "win64"
    pp = Preprocessor.from_state(np.load(REPO / "artifacts" / "win64" / "preprocessor.npz"))
    cols = {"LS": pp.view_index("structural"), "LC": pp.view_index("content"), "FL": None}
    Xtr = np.load(prep / "train.npy", mmap_mode="r")
    Xva = np.load(prep / "validation.npy", mmap_mode="r")
    ytr = pd.read_parquet(prep / "train_meta.parquet").label.values.astype(int)
    yva = pd.read_parquet(prep / "validation_meta.parquet").label.values.astype(int)
    record = {"inputs": {p.name: sha(p) for p in (prep / "train.npy", prep / "validation.npy",
                                                  REPO / "artifacts" / "win64" / "preprocessor.npz")},
              "params": PARAMS, "rounds": ROUNDS, "early_stopping": EARLY, "lightgbm": lgb.__version__,
              "models": {}}
    for kind, c in cols.items():
        xt = np.asarray(Xtr if c is None else Xtr[:, c], dtype=np.float32)
        xv = np.asarray(Xva if c is None else Xva[:, c], dtype=np.float32)
        for s in SEEDS:
            name = f"{kind}_{s}"
            t0 = time.time()
            p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s)
            dtr = lgb.Dataset(xt, ytr, free_raw_data=False)
            dva = lgb.Dataset(xv, yva, reference=dtr, free_raw_data=False)
            bst = lgb.train(p, dtr, ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(EARLY, verbose=False)])
            path = OUT / f"{name}.txt"
            bst.save_model(str(path), num_iteration=bst.best_iteration)
            v = bst.predict(xv, num_iteration=bst.best_iteration, raw_score=True).astype(np.float64)
            np.save(OUT / f"val_logits_{name}.npy", v)
            thr = at_fpr(v[yva == 0], ALPHA)
            record["models"][name] = {"sha256": sha(path), "best_iteration": bst.best_iteration,
                                      "threshold": thr, "train_seconds": time.time() - t0,
                                      "val_fnr_at_threshold": float((v[yva == 1] <= thr).mean())}
            print(name, record["models"][name], flush=True)
        del xt, xv
    (OUT / "train_record.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
