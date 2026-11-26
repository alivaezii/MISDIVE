"""Resume of the authorised Win64 LightGBM training after a memory failure (process incident, not a design change).

The frozen script `train_win64_lgbm_pool.py` (unchanged; SHA-256 recorded) trained LS_* and LC_* successfully,
then failed with `LightGBMError: bad allocation` while constructing the first full-feature (FL_42) dataset,
before any FL model existed. Free physical memory at the time was ~3.7 GB of 15.6 GB.

This script imports PARAMS, ROUNDS, EARLY, ALPHA and SEEDS from the frozen script, so the
hyperparameters are identical. It trains only FL_s, in a fresh process. To reduce peak memory it frees the raw
float32 matrices once the LightGBM datasets are constructed (free_raw_data=True), which does not change the
model. It then writes train_record.json for all nine models: LS/LC entries are reconstructed from the saved
model files (digest, best iteration), their validation logits (threshold) and the training log (seconds).
"""
from __future__ import annotations

import gc
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
import os

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("frozen", HERE / "train_win64_lgbm_pool.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


def main():
    if "--authorised-T" not in sys.argv:
        sys.exit("Explicit training confirmation is required.")
    import lightgbm as lgb
    import pandas as pd
    assert lgb.__version__ == "4.7.0"
    prep = F.REPO / "data" / "prep" / "win64"
    ytr = pd.read_parquet(prep / "train_meta.parquet").label.values.astype(int)
    yva = pd.read_parquet(prep / "validation_meta.parquet").label.values.astype(int)
    record = {"inputs": {p.name: F.sha(p) for p in (prep / "train.npy", prep / "validation.npy",
                                                    F.REPO / "artifacts" / "win64" / "preprocessor.npz")},
              "params": F.PARAMS, "rounds": F.ROUNDS, "early_stopping": F.EARLY, "lightgbm": lgb.__version__,
              "models": {}, "incident": "FL training resumed in a fresh process after 'bad allocation' "
                                       "(train_win64_lgbm_FL_resume.py); LS/LC entries reconstructed from saved files."}
    log = (F.OUT / "train_lgbm.log").read_text()
    secs = {m.group(1): float(m.group(2)) for m in re.finditer(r"^(L[SC]_\d+) \{.*?'train_seconds': ([0-9.]+)", log, re.M)}
    for kind in ("LS", "LC"):
        for s in F.SEEDS:
            name = f"{kind}_{s}"
            path = F.OUT / f"{name}.txt"
            bst = lgb.Booster(model_file=str(path))
            v = np.load(F.OUT / f"val_logits_{name}.npy")
            thr = F.at_fpr(v[yva == 0], F.ALPHA)
            record["models"][name] = {"sha256": F.sha(path), "best_iteration": bst.current_iteration(),
                                      "threshold": thr, "train_seconds": secs.get(name),
                                      "val_fnr_at_threshold": float((v[yva == 1] <= thr).mean()),
                                      "reconstructed_from_saved_files": True}
    seeds = F.SEEDS
    if "--seeds" in sys.argv:  # train only these FL seeds in a fresh process; keep earlier FL entries
        seeds = tuple(int(x) for x in sys.argv[sys.argv.index("--seeds") + 1].split(","))
        prev = json.loads((F.OUT / "train_record.json").read_text())
        record["models"].update({k: v for k, v in prev["models"].items() if k.startswith("FL_")})
        record["incident"] += f" FL seeds {seeds} trained in a further fresh process after a second 'bad allocation'."
    for s in seeds:
        name = f"FL_{s}"
        t0 = time.time()
        xt = np.asarray(np.load(prep / "train.npy", mmap_mode="r"), dtype=np.float32)
        xv = np.asarray(np.load(prep / "validation.npy", mmap_mode="r"), dtype=np.float32)
        p = dict(F.PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s)
        # Dataset parameters (max_bin etc.) must be supplied at construction; in the frozen script the
        # datasets are constructed inside lgb.train with the same parameter dict p, so binning is identical.
        dtr = lgb.Dataset(xt, ytr, params=p, free_raw_data=True)
        dva = lgb.Dataset(xv, yva, reference=dtr, params=p, free_raw_data=True)
        dtr.construct()
        dva.construct()
        del xt
        gc.collect()
        bst = lgb.train(p, dtr, F.ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(F.EARLY, verbose=False)])
        path = F.OUT / f"{name}.txt"
        bst.save_model(str(path), num_iteration=bst.best_iteration)
        del dtr, dva, bst
        gc.collect()
        b2 = lgb.Booster(model_file=str(path))
        v = b2.predict(xv, raw_score=True).astype(np.float64)
        np.save(F.OUT / f"val_logits_{name}.npy", v)
        thr = F.at_fpr(v[yva == 0], F.ALPHA)
        record["models"][name] = {"sha256": F.sha(path), "best_iteration": b2.current_iteration(), "threshold": thr,
                                  "train_seconds": time.time() - t0, "val_fnr_at_threshold": float((v[yva == 1] <= thr).mean())}
        print(name, record["models"][name], flush=True)
        del xv, b2
        gc.collect()
        (F.OUT / "train_record.json").write_text(json.dumps(record, indent=1))
    (F.OUT / "train_record.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
